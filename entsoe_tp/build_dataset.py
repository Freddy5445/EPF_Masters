"""
Build an epftoolbox-compatible dataset from the Transparency Platform.

    python -m entsoe_tp.build_dataset --zone DK1 --start 2016-01-01 --end 2024-12-31

Writes ``datasets/<ZONE>.csv`` with the columns ``Date, Price, Exogenous 1,
Exogenous 2``:

===============  =========================================  ==========================
Column           Data item                                  Query
===============  =========================================  ==========================
Price            Day-ahead prices [12.1.D]                  ``A44``
Exogenous 1      Day-ahead total load forecast [6.1.B]      ``A65`` + ``processType=A01``
Exogenous 2      Day-ahead wind & solar forecast [14.1.D]   ``A69`` + ``processType=A01``
===============  =========================================  ==========================

Both exogenous series are day-ahead *forecasts*, so they are genuinely available
at the time a day-ahead price forecast has to be made. Realised load and
generation are not, and are deliberately not used here.

``read_data`` assigns column names positionally -- the first column after the
index becomes ``Price`` and the rest become ``Exogenous 1..N`` -- so the header
names written here are for human readers; the *order* is what matters.
"""

import argparse
import os
import sys

import pandas as pd

from .areas import lookup
from .client import TransparencyClient
from .hourly import GridError, assert_epftoolbox_grid, to_local_hourly_grid
from .parser import TransparencyError, parse_document, to_series

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
DEFAULT_CACHE = os.path.join(PROJECT_ROOT, ".cache", "entsoe")

# Day-ahead market time units moved from 60 to 15 minutes across European
# bidding zones on this date. Everything before it is hourly, which is what the
# epftoolbox models assume; refuse to cross the boundary rather than silently
# mixing resolutions in one series.
MTU_SWITCHOVER = pd.Timestamp("2025-10-01")


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


def _fetch_series(client, params, value_tag, aggregate, start_utc, end_utc,
                  label, filter_fn=None, quiet=False):
    def progress(number, total, chunk_start, _chunk_end):
        if not quiet:
            print(f"  [{label}] chunk {number}/{total}  {chunk_start:%Y-%m}", flush=True)

    documents = client.fetch(params, start_utc, end_utc, progress=progress)

    frames = [parse_document(doc, value_tag) for doc in documents]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.Series(dtype="float64",
                         index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))

    combined = pd.concat(frames, ignore_index=True)
    if filter_fn is not None:
        combined = filter_fn(combined)

    return to_series(combined, aggregate=aggregate)


def build(zone, start, end, cache_dir=DEFAULT_CACHE, token=None, max_gap=3, quiet=False):
    """Fetch and assemble the dataset. Returns the finished DataFrame."""
    area = lookup(zone)
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()

    if start_date > end_date:
        raise ValueError(f"--start {start_date.date()} is after --end {end_date.date()}")
    if end_date >= MTU_SWITCHOVER:
        raise ValueError(
            f"--end {end_date.date()} reaches into the 15-minute market time unit era "
            f"(from {MTU_SWITCHOVER.date()}). The epftoolbox models require 24 hourly "
            f"prices per day; choose an earlier end date."
        )

    # The platform works in UTC, but the calendar days we want are local ones.
    start_utc = start_date.tz_localize(area.tz, nonexistent="shift_forward").tz_convert("UTC")
    end_utc = (end_date + pd.Timedelta(days=1)).tz_localize(
        area.tz, nonexistent="shift_forward").tz_convert("UTC")

    client = TransparencyClient(token=token, cache_dir=cache_dir)

    if not quiet:
        print(f"Zone {area.code} ({area.name}, {area.eic}) in {area.tz}")
        print(f"Range {start_date.date()} to {end_date.date()} inclusive")

    specs = [
        ("Price", {"documentType": "A44",
                   "in_Domain": area.eic,
                   "out_Domain": area.eic},
         "price.amount", "first", _filter_day_ahead),
        ("Exogenous 1", {"documentType": "A65",
                         "processType": "A01",
                         "outBiddingZone_Domain": area.eic},
         "quantity", "first", None),
        ("Exogenous 2", {"documentType": "A69",
                         "processType": "A01",
                         "in_Domain": area.eic},
         "quantity", "sum", None),
    ]

    columns = {}
    for label, params, value_tag, aggregate, filter_fn in specs:
        series = _fetch_series(client, params, value_tag, aggregate,
                               start_utc, end_utc, label, filter_fn, quiet)
        if series.empty:
            raise ValueError(
                f"The platform returned no data for {label} in zone {area.code}. "
                f"Check that this zone publishes that data item for the requested range."
            )
        columns[label] = to_local_hourly_grid(
            series, area.tz, start_date, end_date, max_gap=max_gap
        )

    frame = pd.DataFrame(columns)
    # Naive local market time, not UTC -- see hourly.py. Named explicitly so the
    # CSV is unambiguous on its own, without requiring this module's docstring.
    frame.index.name = f"Date ({area.tz} local time, naive, ISO 8601)"
    assert_epftoolbox_grid(frame)

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
