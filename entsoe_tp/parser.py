"""
XML parsing for Transparency Platform market documents.

Two things here are easy to get wrong and silently corrupt the resulting series,
so they are handled explicitly:

1. ``curveType=A03`` (variable-sized blocks) is the platform default for every
   data item. A ``Point`` is published only when the value *changes*; a gap in
   the ``position`` sequence means "the previous value continues". Reading the
   points as a dense list therefore drops intervals and shifts every subsequent
   value earlier. :func:`_expand_period` walks the full slot range implied by the
   period's ``timeInterval`` and ``resolution``, carrying values forward.

2. A rejected request still returns **HTTP 200**. The body is an
   ``Acknowledgement_MarketDocument`` carrying a ``Reason`` code rather than the
   expected publication document, so the root element is checked before parsing.
"""

import xml.etree.ElementTree as ET

import pandas as pd

# Reason code the platform uses for "your query was fine, there is just no data".
# Treated as an empty result rather than an error so that gaps in a long
# historical range do not abort a multi-year download.
NO_DATA_REASON_CODE = "999"

_RESOLUTIONS = {
    "PT15M": pd.Timedelta(minutes=15),
    "PT30M": pd.Timedelta(minutes=30),
    "PT60M": pd.Timedelta(hours=1),
    "PT1H": pd.Timedelta(hours=1),
    "P1D": pd.Timedelta(days=1),
    "P7D": pd.Timedelta(days=7),
}

# Direct children of <TimeSeries> worth keeping as columns. Anything else is
# ignored; these are the ones used to disambiguate overlapping series.
_TS_ATTRS = (
    "businessType",
    "contract_MarketAgreement.type",
    "classificationSequence_AttributeInstanceComponent.position",
    "curveType",
    "inBiddingZone_Domain.mRID",
    "outBiddingZone_Domain.mRID",
    # Units, so a column can be labelled with what it actually holds rather than
    # an assumed one. Currency varies by zone (not every market publishes EUR).
    "currency_Unit.name",
    "price_Measure_Unit.name",
    "quantity_Measure_Unit.name",
)

# ENTSO-E unit codes, mapped to how they are normally written.
UNIT_NAMES = {
    "MAW": "MW",
    "MWH": "MWh",
    "KWH": "kWh",
}


def unit_label(code):
    """Human-readable form of an ENTSO-E unit code, or the code itself."""
    if not code:
        return None
    return UNIT_NAMES.get(code.upper(), code)


class TransparencyError(RuntimeError):
    """Raised when the platform rejects a request or returns something unusable."""


class UnexpectedResolution(TransparencyError):
    """Raised when a period's resolution is not the one the caller requires."""


def _localname(tag):
    """Strip the XML namespace: ``{urn:...}Period`` -> ``Period``."""
    return tag.rpartition("}")[2]


def _iterfind(elem, name):
    """Direct children matching a local name, namespace-agnostic.

    Namespace URIs differ between document types (and change between schema
    versions), so matching on the local name keeps the parser from breaking
    every time ENTSO-E bumps a schema.
    """
    return [c for c in elem if _localname(c.tag) == name]


def _findtext(elem, name, default=None):
    for c in _iterfind(elem, name):
        return c.text
    return default


def _utc(text):
    """Parse a platform timestamp to UTC.

    The documented format carries a ``Z`` suffix and all platform times are UTC,
    but a bare timestamp is treated as UTC rather than allowed to raise -- an
    offset-less value would otherwise fail deep in the parse rather than here.
    """
    if text is None:
        raise TransparencyError("Missing timestamp in timeInterval")
    stamp = pd.Timestamp(text)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _check_acknowledgement(root):
    """Raise (or signal 'no data') if the document is a rejection."""
    if _localname(root.tag) != "Acknowledgement_MarketDocument":
        return False

    reasons = []
    for reason in _iterfind(root, "Reason"):
        code = _findtext(reason, "code", "")
        text = _findtext(reason, "text", "")
        reasons.append((code, text))

    if all(code == NO_DATA_REASON_CODE for code, _ in reasons) and reasons:
        return True  # no data, not an error

    detail = "; ".join(f"{code}: {text}" for code, text in reasons) or "no reason given"
    raise TransparencyError(f"Request rejected by the Transparency Platform ({detail})")


def _expand_period(period, value_tag, expect_resolution):
    """Expand one ``<Period>`` into ``(timestamp, value)`` pairs in UTC.

    Honours ``curveType=A03``: positions absent from the document inherit the
    most recent published value, and the final value is carried to the end of
    the period.
    """
    interval = _iterfind(period, "timeInterval")
    if not interval:
        raise TransparencyError("Period is missing its timeInterval")

    start = _utc(_findtext(interval[0], "start"))
    end = _utc(_findtext(interval[0], "end"))

    raw_resolution = _findtext(period, "resolution")
    if expect_resolution is not None and raw_resolution != expect_resolution:
        raise UnexpectedResolution(
            f"Period {start}/{end} has resolution {raw_resolution!r}, expected "
            f"{expect_resolution!r}. Day-ahead data moved to 15-minute market time "
            f"units from 2025-10-01; restrict the range or aggregate before use."
        )

    try:
        step = _RESOLUTIONS[raw_resolution]
    except KeyError:
        raise TransparencyError(f"Unsupported resolution {raw_resolution!r}") from None

    points = {}
    for point in _iterfind(period, "Point"):
        position = int(_findtext(point, "position"))
        value = _findtext(point, value_tag)
        if value is None:
            raise TransparencyError(
                f"Point {position} in period {start}/{end} has no <{value_tag}>"
            )
        points[position] = float(value)

    if not points:
        return []

    n_slots = int((end - start) / step)
    highest = max(points)
    if highest > n_slots:
        raise TransparencyError(
            f"Period {start}/{end} at {raw_resolution} holds {n_slots} slots but "
            f"declares position {highest}"
        )

    expanded = []
    current = None
    for slot in range(1, n_slots + 1):
        if slot in points:
            current = points[slot]
        if current is None:
            # Nothing published yet for the start of the period. Under A03 there
            # is no earlier value to carry forward, so the slot is genuinely
            # unknown rather than zero.
            continue
        expanded.append((start + (slot - 1) * step, current))

    return expanded


def parse_document(xml_text, value_tag, expect_resolution="PT60M"):
    """Parse a market document into a tidy frame.

    Returns a DataFrame with a ``timestamp`` column (UTC, tz-aware), a ``value``
    column, a ``psr_type`` column (``None`` when the document does not break the
    series down by production type) and one column per entry in :data:`_TS_ATTRS`.

    ``value_tag`` is ``"price.amount"`` for price documents (A44) and
    ``"quantity"`` for load and generation documents (A65, A69, A71, A75).

    Set ``expect_resolution=None`` to accept whatever the platform returns.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise TransparencyError(f"Response is not valid XML: {exc}") from exc

    columns = ["timestamp", "value", "psr_type"] + list(_TS_ATTRS)

    if _check_acknowledgement(root):
        return pd.DataFrame(columns=columns)

    rows = []
    for ts in _iterfind(root, "TimeSeries"):
        attrs = {name: _findtext(ts, name) for name in _TS_ATTRS}

        psr_type = None
        for mkt in _iterfind(ts, "MktPSRType"):
            psr_type = _findtext(mkt, "psrType")

        for period in _iterfind(ts, "Period"):
            for timestamp, value in _expand_period(period, value_tag, expect_resolution):
                rows.append({"timestamp": timestamp, "value": value,
                             "psr_type": psr_type, **attrs})

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, columns=columns)


def to_series(frame, aggregate="first"):
    """Collapse a tidy frame from :func:`parse_document` into a UTC-indexed Series.

    ``aggregate="first"`` keeps the earliest row for each timestamp, which is the
    right choice for prices: a range can contain several overlapping publications
    of the same hour and they should not be averaged together.

    ``aggregate="sum"`` adds values sharing a timestamp, which is how the wind and
    solar forecast (A69) is combined -- it arrives as one TimeSeries per
    production type (solar, wind onshore, wind offshore).
    """
    if frame.empty:
        return pd.Series(dtype="float64",
                         index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
                         name="value")

    if aggregate == "sum":
        # Sum across production types, but guard against the same production type
        # being published twice for one timestamp: de-duplicate within psr_type
        # first, then add the types together.
        deduped = frame.drop_duplicates(subset=["timestamp", "psr_type"], keep="first")
        grouped = deduped.groupby("timestamp")["value"].sum()
    elif aggregate == "first":
        grouped = frame.drop_duplicates(subset=["timestamp"], keep="first") \
                       .set_index("timestamp")["value"]
    else:
        raise ValueError(f"Unknown aggregate {aggregate!r}")

    return grouped.sort_index().rename("value")
