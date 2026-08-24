"""
Build an epftoolbox-compatible dataset from the Transparency Platform.

    python -m entsoe_tp.build_dataset --zone DK1 --start 2016-01-01 --end 2024-12-31

Writes ``datasets/<ZONE>.csv``. Two exogenous layouts are available, chosen with
``--exog``.

Columns are named for what they hold, with the unit taken from the document
itself (currency varies by market, so it is never assumed).

``--exog load-windsolar`` (default):

==========================================  =========================  ==========================
Column                                      Data item                  Query
==========================================  =========================  ==========================
Day-ahead price (EUR/MWh)                   Day-ahead prices [12.1.D]  ``A44``
Day-ahead load forecast (MW)                Total load forecast        ``A65`` + ``processType=A01``
Day-ahead wind and solar forecast (MW)      Wind & solar [14.1.D]      ``A69`` + ``processType=A01``
==========================================  =========================  ==========================

``--exog load-wind-solar`` splits the renewables by production type:

==========================================  =========================  ==========================
Column                                      Data item                  Query
==========================================  =========================  ==========================
Day-ahead price (EUR/MWh)                   Day-ahead prices [12.1.D]  ``A44``
Day-ahead load forecast (MW)                Total load forecast        ``A65`` + ``processType=A01``
Day-ahead wind forecast (on- and offshore)  Wind [14.1.D]              ``A69``, ``psrType`` B18+B19
Day-ahead solar forecast (MW)               Solar [14.1.D]             ``A69``, ``psrType`` B16
==========================================  =========================  ==========================

The A69 document carries one TimeSeries per production type, so both renewable
columns come from a single query that is fetched once and split afterwards.

``--include-reservoir`` appends Water Reservoirs and Hydro Storage Plants
[16.1.D] (``A72`` + ``processType=A16``) as a further column. It is published
weekly and is a *stock* rather than a flow, so it is held constant between
publications rather than interpolated, and each value only takes effect once the
week it describes has ended -- interpolating or applying it sooner would use
information that was not yet available.

Note that the exogenous count drives the LEAR feature count
(``96 + 7 + 72 * n_exogenous``), and ``LassoLarsIC`` requires more training
samples than features -- so a 3-exogenous dataset (319 features) cannot be
calibrated on windows shorter than roughly a year, and adding the reservoir
column (391 features) pushes that minimum to about 399 days.

Re-running is incremental. Raw XML is cached per request, and finished columns
are cached under ``.cache/entsoe/columns/``, so adding a column to an existing
dataset downloads and parses only that column.

Both exogenous series are day-ahead *forecasts*, so they are genuinely available
at the time a day-ahead price forecast has to be made. Realised load and
generation are not, and are deliberately not used here.

``read_data`` assigns column names positionally -- the first column after the
index becomes ``Price`` and the rest become ``Exogenous 1..N`` -- so the header
names written here are for human readers; the *column order* is what binds the
data to the model.
"""

import argparse
import hashlib
import json
import os
import sys

import pandas as pd

from .areas import lookup
from .client import TransparencyClient
from .hourly import (
    GridError, assert_epftoolbox_grid, to_local_hourly_grid, to_local_hourly_step,
)
from .parser import TransparencyError, parse_document, to_series, unit_label

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
DEFAULT_CACHE = os.path.join(PROJECT_ROOT, ".cache", "entsoe")

# The EU-wide deadline for day-ahead market time units to move from 60 to 15
# minutes. It is *not* when any given zone actually switched: DK1 moved in April
# 2025 and NO2 in February, both well before it. Nothing decides behaviour from
# this date -- the real boundary is measured per zone by _first_non_hourly --
# and it is kept only as a reference point for reporting.
MTU_SWITCHOVER = pd.Timestamp("2025-10-01")

# psrType codes used by the wind & solar forecast document (A69).
PSR_SOLAR = "B16"
PSR_WIND_OFFSHORE = "B18"
PSR_WIND_ONSHORE = "B19"

EXOG_LAYOUTS = ("load-windsolar", "load-wind-solar")

# Water Reservoirs and Hydro Storage Plants [16.1.D] is published *weekly* and
# is a stock (stored energy) rather than a flow, so it is expanded onto the
# hourly grid as a step function rather than aligned like the hourly series.
RESERVOIR_PERIOD = pd.Timedelta(days=7)

HOURLY_RESOLUTION = "PT60M"


def _first_non_hourly(frame):
    """First timestamp in a parsed frame that is not hourly, or None.

    Zones moved to 15-minute market time units on different dates -- DK1 in
    April 2025, NO2 in February -- so where the hourly era ends is a property of
    the data, not a constant. It has to be measured per zone and per data item.
    """
    if frame.empty or "resolution" not in frame.columns:
        return None
    other = frame[frame["resolution"] != HOURLY_RESOLUTION]
    return None if other.empty else other["timestamp"].min()


def _filter_psr(*codes):
    """Build a filter keeping only the given production types from an A69 frame."""
    def filter_fn(frame):
        if frame.empty:
            return frame
        kept = frame[frame["psr_type"].isin(codes)]
        if kept.empty:
            raise ValueError(
                f"The platform returned no data for production type(s) "
                f"{', '.join(codes)}. This zone may not publish them separately."
            )
        return kept

    # Recorded so the column cache key distinguishes wind from solar; both are
    # derived from the same query with the same aggregate.
    filter_fn.psr_codes = list(codes)
    return filter_fn


def _filter_day_ahead(frame):
    """Keep only the day-ahead auction series from a price document.

    A price query can return several TimeSeries covering the same hours -- for
    example intraday alongside day-ahead. Dropping the others before collapsing
    to a series stops an intraday price from displacing the day-ahead one.
    """
    if frame.empty:
        return frame

    contract = frame["contract_MarketAgreement.type"]
    if not contract.notna().any():
        return frame

    day_ahead = frame[contract == "A01"]
    # Some documents omit the attribute entirely; only narrow if doing so leaves
    # something behind.
    return day_ahead if not day_ahead.empty else frame


def _fetch_frame(client, params, value_tag, start_utc, end_utc, label,
                 expect_resolution="PT60M", quiet=False):
    """Fetch one query's whole date range and return the combined tidy frame.

    Kept separate from column derivation so that a single query feeding several
    columns -- as A69 does for wind and solar -- is only downloaded once.
    """
    def progress(number, total, chunk_start, _chunk_end):
        if not quiet:
            print(f"  [{label}] chunk {number}/{total}  {chunk_start:%Y-%m}", flush=True)

    documents = client.fetch(params, start_utc, end_utc, progress=progress)

    frames = [parse_document(doc, value_tag, expect_resolution=expect_resolution)
              for doc in documents]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _price_unit(frame):
    """Currency and measure unit of a price document, e.g. ``EUR/MWh``.

    Read from the document rather than assumed: not every market publishes in
    euros.
    """
    if frame.empty:
        return None
    currency = frame["currency_Unit.name"].dropna()
    measure = frame["price_Measure_Unit.name"].dropna()
    if currency.empty or measure.empty:
        return None
    return f"{currency.iloc[0]}/{unit_label(measure.iloc[0])}"


def _quantity_unit(frame):
    """Measure unit of a load or generation document, e.g. ``MW``."""
    if frame.empty:
        return None
    measure = frame["quantity_Measure_Unit.name"].dropna()
    return unit_label(measure.iloc[0]) if not measure.empty else None


def _with_unit(name, unit):
    return f"{name} ({unit})" if unit else name


def _column_cache_path(cache_dir, zone, label, key):
    """Where a finished column is cached, keyed by everything that shapes it.

    Downloads are already incremental -- the raw-XML cache is keyed on request
    parameters, so adding a data item re-fetches only that item. This caches the
    *parsed and aligned* column as well, so adding one column to an existing
    dataset costs one download and one parse rather than re-deriving every
    column from a decade of cached XML.
    """
    material = json.dumps(key, sort_keys=True, default=str)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    safe = "".join(c if c.isalnum() else "_" for c in label)[:40]
    return os.path.join(cache_dir, "columns", f"{zone}_{safe}_{digest}.csv")


def _load_cached_column(path):
    """Return ``(series, label)`` from a cached column, or None.

    The label is stored with the data because it carries the unit read from the
    document, which is only known after a fetch. Without it a cached column
    would come back named differently from a freshly built one.
    """
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return frame.iloc[:, 0], frame.columns[0]


def _store_cached_column(path, series, label):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    series.to_frame(label).to_csv(path, date_format="%Y-%m-%dT%H:%M:%S")


def _column_specs(area, exog, include_reservoir=False):
    """Return (queries, columns) for the requested exogenous layout.

    ``queries`` maps a query key to (params, value_tag). ``columns`` is an
    ordered list of (column_name, query_key, aggregate, filter_fn), so several
    columns may share one query.

    Column names describe the series. ``read_data`` renames columns positionally
    to ``Price``/``Exogenous N`` when loading, so the *order* is what binds the
    data to the model -- these names are for whoever opens the CSV.
    """
    queries = {
        "price": ({"documentType": "A44",
                   "in_Domain": area.eic,
                   "out_Domain": area.eic}, "price.amount", "PT60M"),
        "load": ({"documentType": "A65",
                  "processType": "A01",
                  "outBiddingZone_Domain": area.eic}, "quantity", "PT60M"),
        "renewables": ({"documentType": "A69",
                        "processType": "A01",
                        "in_Domain": area.eic}, "quantity", "PT60M"),
        # Water Reservoirs and Hydro Storage Plants [16.1.D]: weekly stored
        # energy. processType A16 is "realised".
        # None: the weekly series is expanded as a step function, which does
        # not depend on the resolution being any particular value.
        "reservoir": ({"documentType": "A72",
                       "processType": "A16",
                       "in_Domain": area.eic}, "quantity", None),
    }

    columns = [
        ("Day-ahead price", "price", "first", _filter_day_ahead),
        ("Day-ahead load forecast", "load", "first", None),
    ]

    if exog == "load-windsolar":
        columns.append(("Day-ahead wind and solar forecast",
                        "renewables", "sum", None))
    elif exog == "load-wind-solar":
        columns.append(("Day-ahead wind forecast (on- and offshore)",
                        "renewables", "sum",
                        _filter_psr(PSR_WIND_OFFSHORE, PSR_WIND_ONSHORE)))
        columns.append(("Day-ahead solar forecast", "renewables", "sum",
                        _filter_psr(PSR_SOLAR)))
    else:
        raise ValueError(
            f"Unknown --exog layout {exog!r}. Choose one of: {', '.join(EXOG_LAYOUTS)}"
        )

    if include_reservoir:
        # Appended last so the existing columns keep their positions, which is
        # what read_data binds on.
        columns.append(("Water reservoir and hydro storage",
                        "reservoir", "sum", None))

    # Drop any query no column actually needs.
    used = {key for _, key, _, _ in columns}
    queries = {k: v for k, v in queries.items() if k in used}

    return queries, columns


def build(zone, start, end, cache_dir=DEFAULT_CACHE, token=None, max_gap=3,
          exog="load-windsolar", allow_gaps=False, include_reservoir=False,
          reservoir_lag_days=0, refresh_columns=False,
          stop_at_resolution_change=False, quiet=False):
    """Fetch and assemble the dataset. Returns the finished DataFrame."""
    area = lookup(zone)
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()

    if start_date > end_date:
        raise ValueError(f"--start {start_date.date()} is after --end {end_date.date()}")
    # The platform works in UTC, but the calendar days we want are local ones.
    start_utc = start_date.tz_localize(area.tz, nonexistent="shift_forward").tz_convert("UTC")
    end_utc = (end_date + pd.Timedelta(days=1)).tz_localize(
        area.tz, nonexistent="shift_forward").tz_convert("UTC")

    client = TransparencyClient(token=token, cache_dir=cache_dir)

    if not quiet:
        print(f"Zone {area.code} ({area.name}, {area.eic}) in {area.tz}")
        print(f"Range {start_date.date()} to {end_date.date()} inclusive")

    queries, column_specs = _column_specs(area, exog, include_reservoir)

    # Plan first: a cached column needs no query at all, so adding one column to
    # an existing dataset fetches only that column's data item.
    columns = {}
    plan = []
    for label, key, aggregate, filter_fn in column_specs:
        cache_key = {
            "eic": area.eic, "tz": area.tz, "label": label, "key": key,
            "params": queries[key][0], "aggregate": aggregate,
            "start": str(start_date.date()), "end": str(end_date.date()),
            "max_gap": max_gap, "allow_gaps": allow_gaps,
            "reservoir_lag_days": reservoir_lag_days,
            "filter": getattr(filter_fn, "__name__", None) if filter_fn else None,
            "psr": getattr(filter_fn, "psr_codes", None),
        }
        cache_path = (_column_cache_path(cache_dir, area.code, label, cache_key)
                      if cache_dir and not refresh_columns else None)

        cached = _load_cached_column(cache_path) if cache_path else None
        if cached is not None:
            cached_series, cached_label = cached
            if not quiet:
                print(f"  [{key}] reusing cached column: {cached_label}")
            columns[cached_label] = cached_series
            continue

        plan.append((label, key, aggregate, filter_fn, cache_path))

    # Fetch each still-needed query once. Parsing is permissive so that a change
    # of market time unit is measured rather than raised on: where the hourly
    # era ends differs by zone, so it cannot be assumed.
    raw = {}
    units = {}
    for key in dict.fromkeys(spec[1] for spec in plan):
        params, value_tag, _ = queries[key]
        raw[key] = _fetch_frame(client, params, value_tag, start_utc, end_utc,
                                key, expect_resolution=None, quiet=quiet)
        if raw[key].empty:
            raise ValueError(
                f"The platform returned no data for {key!r} in zone "
                f"{area.code}. Check that this zone publishes that data item "
                f"for the requested range."
            )
        units[key] = (_price_unit(raw[key]) if key == "price"
                      else _quantity_unit(raw[key]))

    # The reservoir series is weekly by design, so only the hourly items are
    # checked for a resolution change.
    cutoffs = [_first_non_hourly(frame) for key, frame in raw.items()
               if key != "reservoir"]
    cutoffs = [c for c in cutoffs if c is not None]

    if cutoffs:
        first_change = min(cutoffs)
        local_change = first_change.tz_convert(area.tz).tz_localize(None)
        last_hourly_day = local_change.normalize() - pd.Timedelta(days=1)

        if last_hourly_day < end_date:
            if not stop_at_resolution_change:
                raise ValueError(
                    f"Zone {area.code} leaves hourly resolution at "
                    f"{local_change} local ({first_change} UTC), inside the "
                    f"requested range. The epftoolbox models need 24 hourly "
                    f"prices per day, so re-run with "
                    f"--end {last_hourly_day.date()}, or pass "
                    f"--stop-at-resolution-change to truncate automatically."
                )
            if not quiet:
                print(f"\n  Market time unit changes at {local_change} local; "
                      f"truncating to {last_hourly_day.date()}\n")
            return build(
                zone=zone, start=start, end=last_hourly_day, cache_dir=cache_dir,
                token=token, max_gap=max_gap, exog=exog, allow_gaps=allow_gaps,
                include_reservoir=include_reservoir,
                reservoir_lag_days=reservoir_lag_days,
                refresh_columns=refresh_columns,
                stop_at_resolution_change=False, quiet=quiet,
            )

    for label, key, aggregate, filter_fn, cache_path in plan:
        frame_for_column = raw[key]
        if key != "reservoir":
            # Any sub-hourly rows left inside the range would put timestamps
            # off the hourly grid, so they are dropped rather than aggregated:
            # how to aggregate them is a modelling choice, not a parsing one.
            frame_for_column = frame_for_column[
                frame_for_column["resolution"] == HOURLY_RESOLUTION
            ]
        if filter_fn is not None:
            frame_for_column = filter_fn(frame_for_column)
        series = to_series(frame_for_column, aggregate=aggregate)
        if series.empty:
            raise ValueError(
                f"The platform returned no data for {label} in zone {area.code}. "
                f"Check that this zone publishes that data item for the requested range."
            )

        if key == "reservoir":
            # Weekly stock: hold each level until the next publication, and only
            # after the week it describes has ended.
            column = to_local_hourly_step(
                series, area.tz, start_date, end_date,
                available_after=RESERVOIR_PERIOD
                + pd.Timedelta(days=reservoir_lag_days),
                allow_gaps=allow_gaps,
            )
        else:
            column = to_local_hourly_grid(
                series, area.tz, start_date, end_date, max_gap=max_gap,
                allow_gaps=allow_gaps,
            )

        labelled = _with_unit(label, units.get(key))
        columns[labelled] = column
        if cache_path:
            _store_cached_column(cache_path, column, labelled)

    frame = pd.DataFrame(columns)
    # Naive local market time, not UTC -- see hourly.py. Named explicitly so the
    # CSV is unambiguous on its own, without requiring this module's docstring.
    frame.index.name = f"Date ({area.tz} local time, naive, ISO 8601)"
    assert_epftoolbox_grid(frame, allow_nan=allow_gaps)

    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build an epftoolbox-compatible CSV from the ENTSO-E "
                    "Transparency Platform.")
    parser.add_argument("--zone", required=True,
                        help="Bidding zone code, e.g. DK1, SE3, NO2, DE_LU")
    parser.add_argument("--start", required=True, help="First day, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Last day (inclusive), YYYY-MM-DD")
    parser.add_argument("--out", default=None,
                        help="Output CSV (default: datasets/<ZONE>.csv)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the local raw-XML cache and re-download")
    parser.add_argument("--max-gap", type=int, default=3,
                        help="Longest run of missing hours to interpolate (default 3)")
    parser.add_argument("--exog", choices=EXOG_LAYOUTS, default="load-windsolar",
                        help="Exogenous layout: 'load-windsolar' (2 columns, default) "
                             "or 'load-wind-solar' (3 columns, renewables split by "
                             "production type)")
    parser.add_argument("--include-reservoir", action="store_true",
                        help="Append Water Reservoirs and Hydro Storage Plants "
                             "[16.1.D] as a further exogenous column. Published "
                             "weekly, so it is held constant between publications "
                             "rather than interpolated.")
    parser.add_argument("--reservoir-lag-days", type=int, default=0,
                        help="Extra delay, in days, before a weekly reservoir "
                             "value is treated as known (default 0, on top of the "
                             "7-day period it describes). Raise it to model the "
                             "platform's own reporting delay.")
    parser.add_argument("--stop-at-resolution-change", action="store_true",
                        help="Truncate the range at the last full hourly day "
                             "instead of failing, when the zone's market time "
                             "unit changes inside it. Zones switch on different "
                             "dates, so this finds the boundary per zone.")
    parser.add_argument("--refresh-columns", action="store_true",
                        help="Recompute every column instead of reusing cached "
                             "ones. Raw XML is still served from cache.")
    parser.add_argument("--allow-gaps", action="store_true",
                        help="Acquisition only: write exactly what the platform "
                             "published, leaving every missing hour as NaN. "
                             "Interpolates nothing and ignores --max-gap. The "
                             "result cannot be fed to a model as-is.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args(argv)

    out = args.out or os.path.join(PROJECT_ROOT, "datasets", f"{args.zone.upper()}.csv")

    # These are all "you asked for something impossible" errors with actionable
    # messages; a traceback adds noise without adding information.
    try:
        frame = build(
            zone=args.zone,
            start=args.start,
            end=args.end,
            cache_dir=None if args.no_cache else DEFAULT_CACHE,
            max_gap=args.max_gap,
            exog=args.exog,
            allow_gaps=args.allow_gaps,
            include_reservoir=args.include_reservoir,
            reservoir_lag_days=args.reservoir_lag_days,
            refresh_columns=args.refresh_columns,
            stop_at_resolution_change=args.stop_at_resolution_change,
            quiet=args.quiet,
        )
    except (KeyError, ValueError, GridError, TransparencyError) as exc:
        message = exc.args[0] if exc.args else str(exc)
        print(f"error: {message}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    frame.to_csv(out, date_format="%Y-%m-%dT%H:%M:%S")

    days = len(frame) // 24
    print(f"\nWrote {out}")
    print(f"  {len(frame)} rows ({days} days), {frame.index[0]} to {frame.index[-1]}")
    print(f"  columns: {list(frame.columns)}")

    missing = frame.isna().sum()
    if missing.any():
        print(f"  missing values: {missing.to_dict()}")

    # read_data's years_test split counts 364-day "years" and does positional
    # arithmetic on the index, so ranges that are not whole 364-day blocks put
    # the train/test boundary somewhere other than midnight.
    if days % 364:
        print(f"\nNote: {days} days is not a multiple of 364. read_data(years_test=N) "
              f"measures a year as 52 weeks, so prefer begin_test_date/end_test_date "
              f"for an exact split.")

    dataset = os.path.splitext(os.path.basename(out))[0]
    print(f"\nLoad it with:\n  read_data(path='datasets', dataset='{dataset}', years_test=1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
