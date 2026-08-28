"""
Raw capture of day-ahead price and forecast data across the Nordic zones and DE-LU.

    python -m entsoe_tp.raw_dump

This is deliberately *not* ``build_dataset``. That module produces a modelling
dataset and enforces the invariants the epftoolbox models need: a strict hourly
grid, exactly 24 rows per local calendar day, DST folding, and a refusal to mix
market time unit resolutions. Every one of those would destroy what this dump is
for, which is to study how resolution, availability and completeness change
across zones and over time.

So nothing here is normalised, resampled, aligned, imputed or rejected:

* **Native resolution is preserved.** A response that switches from ``PT60M`` to
  ``PT15M`` mid-range is recorded as it arrived, with the resolution attached to
  every observation rather than assumed for the document.
* **No grid is imposed.** Rows exist only where the platform published a value;
  absence is absence, not an interpolated hour.
* **Timestamps are UTC**, as published. No local-time conversion, so no folded
  autumn hour and no nonexistent spring hour, and the same instant means the
  same thing in every zone.
* **Nothing aborts.** A zone that publishes no solar, a data item that returns
  an empty document, a request that fails outright -- each is recorded in the
  coverage manifest and the run continues.

The output is one long/tidy Parquet file: one row per observation, with the
document and TimeSeries metadata carried alongside so that changes in
``curveType``, ``businessType`` or unit are visible rather than lost.

Four data items are captured per zone:

===================  ==========================================  ===========
``variable``         Data item                                   Query
===================  ==========================================  ===========
price                Day-ahead prices [12.1.D]                   ``A44``
load_forecast        Day-ahead total load forecast [6.1.B]       ``A65``/``A01``
generation_forecast  Day-ahead wind & solar forecast [14.1.D]    ``A69``/``A01``
reservoir            Water reservoirs & hydro storage [16.1.D]   ``A72``/``A16``
===================  ==========================================  ===========

The A69 document is specifically the *wind and solar* generation forecast, so
its production types are whichever of solar (B16), wind onshore (B19) and wind
offshore (B18) a zone publishes. Nothing is filtered: unexpected codes are
recorded rather than dropped.

The reservoir series is weekly (``P7D``) rather than hourly, which needs no
special handling: its resolution is recorded per observation like any other, and
nothing is resampled or aligned to it. Zones with no hydro storage return
nothing and the manifest says so.

Requires ``pyarrow`` (see requirements.txt).
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

from .areas import lookup
from .client import TransparencyClient
from .parser import TransparencyError, parse_document

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
DEFAULT_CACHE = os.path.join(PROJECT_ROOT, ".cache", "entsoe")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "datasets", "nordic_baltic_raw.parquet")

# The Nordic bidding zones, plus DE-LU. Germany-Luxembourg is the large thermal
# market the Nordic zones are coupled to, so its price and forecasts are captured
# on the same footing as theirs rather than treated as an afterthought.
#
# The Baltic zones were dropped: their wind forecasts ramp up late and
# incompletely -- Latvia's never stabilises -- which is documented in section 4
# of data_cleaning.ipynb.
DEFAULT_ZONES = (
    "DK1", "DK2",
    "NO1", "NO2", "NO3", "NO4", "NO5",
    "SE1", "SE2", "SE3", "SE4",
    "FI",
    "DE_LU",
)

DEFAULT_START = "2016-01-01"
# Inclusive. The day-ahead market time unit changes on this date, so including
# it captures the transition itself rather than stopping just short of it.
DEFAULT_END = "2025-10-01"


def _queries(eic):
    """The three data items, keyed by the variable name used in the output."""
    return {
        "price": {
            "params": {"documentType": "A44", "in_Domain": eic, "out_Domain": eic},
            "value_tag": "price.amount",
        },
        "load_forecast": {
            "params": {"documentType": "A65", "processType": "A01",
                       "outBiddingZone_Domain": eic},
            "value_tag": "quantity",
        },
        "generation_forecast": {
            "params": {"documentType": "A69", "processType": "A01",
                       "in_Domain": eic},
            "value_tag": "quantity",
        },
        # Water Reservoirs and Hydro Storage Plants [16.1.D]. Published weekly
        # (P7D) rather than hourly, which needs no special handling here: the
        # resolution is recorded per observation like any other, and nothing is
        # resampled or aligned. Zones without hydro storage simply return
        # nothing, which the manifest records.
        "reservoir": {
            "params": {"documentType": "A72", "processType": "A16",
                       "in_Domain": eic},
            "value_tag": "quantity",
        },
    }


# Output schema. Fixed and explicit so every per-zone part shares it, which is
# what lets the parts be streamed into one Parquet file without loading them all.
COLUMNS = [
    "zone", "eic", "variable", "document_type", "process_type",
    "timestamp_utc", "value", "resolution", "psr_type",
    "business_type", "curve_type", "contract_type", "unit", "currency",
]

CATEGORICAL = [
    "zone", "eic", "variable", "document_type", "process_type", "resolution",
    "psr_type", "business_type", "curve_type", "contract_type", "unit", "currency",
]


def _shape(frame, zone, eic, variable, params):
    """Rename and order a parsed frame into the output schema."""
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)

    unit = frame.get("quantity_Measure_Unit.name")
    if variable == "price":
        unit = frame.get("price_Measure_Unit.name")

    out = pd.DataFrame({
        "zone": zone,
        "eic": eic,
        "variable": variable,
        "document_type": params["documentType"],
        "process_type": params.get("processType"),
        "timestamp_utc": frame["timestamp"],
        "value": pd.to_numeric(frame["value"], errors="coerce"),
        "resolution": frame["resolution"],
        "psr_type": frame["psr_type"],
        "business_type": frame.get("businessType"),
        "curve_type": frame.get("curveType"),
        "contract_type": frame.get("contract_MarketAgreement.type"),
        "unit": unit,
        "currency": frame.get("currency_Unit.name"),
    })
    return out[COLUMNS]


def variables_in_part(path):
    """Which data items an existing per-zone part already holds.

    Resume has to be per *series*, not per zone. A part written before a data
    item existed is not stale -- it is incomplete -- so the zone must be revisited
    for the missing item alone rather than skipped or rebuilt from scratch.
    """
    if not os.path.exists(path):
        return set()
    try:
        existing = pd.read_parquet(path, columns=["variable"])
    except (OSError, ValueError):
        return set()
    return set(existing["variable"].astype("string").dropna().unique())


def fetch_zone(client, zone, start_utc, end_utc, variables=None, quiet=False):
    """Fetch the requested data items for one zone.

    ``variables`` limits the fetch to particular items, so a zone whose part is
    missing only one series costs one query rather than four.

    Returns ``(frame, manifest_rows)``. A failure for one data item is recorded
    and the others still run: the point of the dump is to find out what is
    available, so an absence must be data rather than a crash.
    """
    area = lookup(zone)
    frames = []
    manifest = []

    wanted = _queries(area.eic)
    if variables is not None:
        wanted = {k: v for k, v in wanted.items() if k in variables}

    for variable, spec in wanted.items():
        started = time.time()
        record = {
            "zone": zone, "eic": area.eic, "variable": variable,
            "document_type": spec["params"]["documentType"],
        }
        try:
            def progress(number, total, chunk_start, _end):
                if not quiet:
                    print(f"    {variable:20} chunk {number}/{total}  "
                          f"{chunk_start:%Y-%m}", end="\r", flush=True)

            documents = client.fetch(spec["params"], start_utc, end_utc,
                                     progress=progress)

            # expect_resolution=None: accept whatever the platform returns,
            # including a mid-range change of market time unit.
            parsed = [parse_document(doc, spec["value_tag"], expect_resolution=None)
                      for doc in documents]
            parsed = [f for f in parsed if not f.empty]

            if parsed:
                combined = pd.concat(parsed, ignore_index=True)
                shaped = _shape(combined, zone, area.eic, variable, spec["params"])
                frames.append(shaped)
                record.update({
                    "rows": len(shaped),
                    "first_utc": str(shaped["timestamp_utc"].min()),
                    "last_utc": str(shaped["timestamp_utc"].max()),
                    "resolutions": ",".join(sorted(shaped["resolution"].dropna().unique())),
                    "psr_types": ",".join(sorted(shaped["psr_type"].dropna().unique())),
                    "error": None,
                })
            else:
                record.update({"rows": 0, "first_utc": None, "last_utc": None,
                               "resolutions": "", "psr_types": "",
                               "error": "no data returned"})
        except (TransparencyError, ValueError, KeyError) as exc:
            # Recorded, not raised: one unavailable item must not cost the run.
            record.update({"rows": 0, "first_utc": None, "last_utc": None,
                           "resolutions": "", "psr_types": "",
                           "error": f"{type(exc).__name__}: {exc}"[:300]})

        record["seconds"] = round(time.time() - started, 1)
        manifest.append(record)

        if not quiet:
            status = record["error"] or (
                f"{record['rows']:,} rows  {record['resolutions']}"
                + (f"  [{record['psr_types']}]" if record["psr_types"] else "")
            )
            print(f"    {variable:20} {status}{' ' * 20}")

    if frames:
        frame = pd.concat(frames, ignore_index=True)
    else:
        frame = pd.DataFrame(columns=COLUMNS)

    return frame, manifest


def _write_part(frame, path):
    """Write one zone's rows, with categorical dtypes to keep the file small."""
    out = frame.copy()
    for column in CATEGORICAL:
        out[column] = out[column].astype("string").astype("category")
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out.to_parquet(path, index=False)


def combine_parts(part_paths, out_path):
    """Stream per-zone parts into one Parquet file.

    Streaming rather than concatenating in pandas: the full dump runs to several
    million rows, and only one zone needs to be in memory at a time.
    """
    import pyarrow.parquet as pq

    writer = None
    total = 0
    try:
        for path in part_paths:
            table = pq.read_table(path)
            if table.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema,
                                          compression="snappy")
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            total += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    return total


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Dump raw ENTSO-E day-ahead price and forecast data for the "
                    "Nordic and Baltic zones, preserving native resolution.")
    parser.add_argument("--zones", default=",".join(DEFAULT_ZONES),
                        help="Comma-separated zone codes "
                             f"(default: all {len(DEFAULT_ZONES)} Nordic/Baltic)")
    parser.add_argument("--start", default=DEFAULT_START,
                        help=f"First day, YYYY-MM-DD (default {DEFAULT_START})")
    parser.add_argument("--end", default=DEFAULT_END,
                        help=f"Last day inclusive, YYYY-MM-DD (default {DEFAULT_END})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Output Parquet file")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the raw-XML cache and re-download")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch zones that already have a part file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("error: this dump writes Parquet and needs pyarrow.\n"
              "  pip install pyarrow", file=sys.stderr)
        return 1

    zones = [z.strip().upper() for z in args.zones.split(",") if z.strip()]
    try:
        for zone in zones:
            lookup(zone)
    except KeyError as exc:
        print(f"error: {exc.args[0]}", file=sys.stderr)
        return 1

    # A single UTC window for every zone, so the same instant is covered
    # everywhere and cross-zone comparison needs no timezone reasoning.
    start_utc = pd.Timestamp(args.start).tz_localize("UTC")
    end_utc = (pd.Timestamp(args.end) + pd.Timedelta(days=1)).tz_localize("UTC")
    if start_utc >= end_utc:
        print(f"error: --start {args.start} is not before --end {args.end}",
              file=sys.stderr)
        return 1

    out_path = os.path.abspath(args.out)
    parts_dir = os.path.splitext(out_path)[0] + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    client = TransparencyClient(cache_dir=None if args.no_cache else DEFAULT_CACHE)

    print(f"Zones {len(zones)}: {', '.join(zones)}")
    print(f"Window {start_utc} to {end_utc} (UTC, --end inclusive)")
    print(f"Parts in {parts_dir}\n")

    manifest = []
    started = time.time()

    for number, zone in enumerate(zones, 1):
        part_path = os.path.join(parts_dir, f"{zone}.parquet")
        manifest_path = os.path.join(parts_dir, f"{zone}.manifest.json")

        all_variables = list(_queries("x"))
        have = set() if args.refresh else variables_in_part(part_path)
        missing = [v for v in all_variables if v not in have]

        previous_manifest = []
        if os.path.exists(manifest_path) and not args.refresh:
            with open(manifest_path, encoding="utf-8") as handle:
                previous_manifest = json.load(handle)
            # A data item recorded as returning nothing was still asked for, so
            # it is answered, not missing.
            # Only a row without an error counts as answered. A variable that
            # failed still has a manifest row, so counting it here would make a
            # re-run skip it and leave --refresh -- which re-downloads every
            # variable for every zone -- as the only way to retry.
            answered = {row["variable"] for row in previous_manifest
                        if not row.get("error")}
            missing = [v for v in all_variables if v not in have and v not in answered]

        if not missing:
            print(f"[{number}/{len(zones)}] {zone}: all {len(all_variables)} "
                  f"data items present, skipping")
            manifest.extend(previous_manifest)
            continue

        if have or previous_manifest:
            print(f"[{number}/{len(zones)}] {zone}: adding {', '.join(missing)} "
                  f"to an existing part")
        else:
            print(f"[{number}/{len(zones)}] {zone}")

        frame, zone_manifest = fetch_zone(client, zone, start_utc, end_utc,
                                          variables=missing, quiet=args.quiet)

        # Merge rather than replace, so the series already downloaded are neither
        # re-fetched nor lost.
        if os.path.exists(part_path) and not args.refresh:
            existing = pd.read_parquet(part_path)
            frame = pd.concat([existing, frame], ignore_index=True)
        _write_part(frame, part_path)

        zone_manifest = [row for row in previous_manifest
                         if row["variable"] not in missing] + zone_manifest
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(zone_manifest, handle, indent=2)
        manifest.extend(zone_manifest)
        print(f"    -> {len(frame):,} rows in the part\n")

    part_paths = [os.path.join(parts_dir, f"{z}.parquet") for z in zones]
    part_paths = [p for p in part_paths if os.path.exists(p)]
    total = combine_parts(part_paths, out_path)

    manifest_frame = pd.DataFrame(manifest)
    manifest_csv = os.path.splitext(out_path)[0] + "_manifest.csv"
    manifest_frame.to_csv(manifest_csv, index=False)

    print(f"Wrote {out_path}")
    print(f"  {total:,} observations from {len(part_paths)} zone(s) in "
          f"{time.time() - started:.0f}s")
    print(f"  size {os.path.getsize(out_path) / 1e6:.1f} MB")
    print(f"Coverage manifest: {manifest_csv}")

    if len(manifest_frame):
        empty = manifest_frame[manifest_frame["rows"] == 0]
        if len(empty):
            print(f"\n  {len(empty)} zone/item combination(s) returned nothing:")
            for _, row in empty.iterrows():
                print(f"    {row['zone']:4} {row['variable']:20} {row['error']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
