"""
Report what the platform actually publishes over a date range, before building.

    python -m entsoe_tp.coverage --zone DK1 --start 2015-01-05 --end 2025-04-07 --exog load-wind-solar

``build_dataset`` refuses to interpolate across a long gap, and refuses to mix
market time unit resolutions. Both are the right call -- but both only surface
*after* a multi-year download, and neither tells you which start date would have
worked. This does: it reads the same queries (hitting the raw-XML cache, so a
re-run costs nothing), then reports per series

* the first and last hour actually published,
* every gap longer than ``--max-gap``, with its date and length,
* the first date from which every series is dense enough to build.

Nothing here writes a dataset; it only reports.
"""

import argparse
import sys

import pandas as pd

from .areas import lookup
from .build_dataset import (
    DEFAULT_CACHE, EXOG_LAYOUTS, MTU_SWITCHOVER, _column_specs, _fetch_frame,
)
from .client import TransparencyClient
from .parser import TransparencyError, UnexpectedResolution, to_series


def gap_runs(missing_index):
    """Collapse a DatetimeIndex of missing hours into (start, end, hours) runs."""
    if not len(missing_index):
        return []

    runs = []
    run_start = previous = missing_index[0]
    length = 1

    for timestamp in missing_index[1:]:
        if timestamp - previous == pd.Timedelta(hours=1):
            length += 1
        else:
            runs.append((run_start, previous, length))
            run_start, length = timestamp, 1
        previous = timestamp

    runs.append((run_start, previous, length))
    return runs


def analyse(series, tz, start_date, end_date, max_gap):
    """Coverage facts for one series against the local hourly grid it must fill."""
    if series.empty:
        return {"empty": True}

    local = series.tz_convert(tz).tz_localize(None)
    local = local.groupby(level=0).mean()

    grid = pd.date_range(
        pd.Timestamp(start_date).normalize(),
        pd.Timestamp(end_date).normalize() + pd.Timedelta(hours=23),
        freq="h",
    )
    aligned = local.reindex(grid)

    # Hours that do not exist locally (spring forward) are not real gaps.
    localized = grid.tz_localize(tz, ambiguous=True, nonexistent="NaT")
    dst_missing = grid[localized.isna()]
    missing = aligned.index[aligned.isna()].difference(dst_missing)

    runs = gap_runs(missing)
    oversized = [r for r in runs if r[2] > max_gap]

    return {
        "empty": False,
        "first": local.index.min(),
        "last": local.index.max(),
        "hours": int(local.notna().sum()),
        "missing": len(missing),
        "runs": runs,
        "oversized": oversized,
    }


def first_clean_date(reports, max_gap):
    """Earliest date from which no series has a gap longer than ``max_gap``.

    Starting the dataset here is the least-loss fix for sparse early data. A
    gap in the middle of the range cannot be fixed this way, so those are
    reported separately by the caller.
    """
    candidates = []
    for report in reports.values():
        if report["empty"]:
            return None
        candidates.append(pd.Timestamp(report["first"]).normalize())
        for _, run_end, _ in report["oversized"]:
            candidates.append(pd.Timestamp(run_end).normalize() + pd.Timedelta(days=1))
    return max(candidates) if candidates else None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Report ENTSO-E data coverage for a zone and range, without "
                    "building a dataset.")
    parser.add_argument("--zone", required=True, help="Bidding zone code, e.g. DK1")
    parser.add_argument("--start", required=True, help="First day, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Last day (inclusive), YYYY-MM-DD")
    parser.add_argument("--exog", choices=EXOG_LAYOUTS, default="load-windsolar",
                        help="Exogenous layout to check (default: load-windsolar)")
    parser.add_argument("--max-gap", type=int, default=3,
                        help="Gap length to flag, in hours (default 3)")
    parser.add_argument("--show-gaps", type=int, default=10,
                        help="Longest gaps to list per series (default 10)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the raw-XML cache and re-download")
    args = parser.parse_args(argv)

    area = lookup(args.zone)
    start_date = pd.Timestamp(args.start).normalize()
    end_date = pd.Timestamp(args.end).normalize()

    client = TransparencyClient(
        cache_dir=None if args.no_cache else DEFAULT_CACHE
    )

    start_utc = start_date.tz_localize(area.tz, nonexistent="shift_forward").tz_convert("UTC")
    end_utc = (end_date + pd.Timedelta(days=1)).tz_localize(
        area.tz, nonexistent="shift_forward").tz_convert("UTC")

    print(f"Zone {area.code} ({area.name}) in {area.tz}")
    print(f"Range {start_date.date()} to {end_date.date()}, layout {args.exog}\n")

    queries, column_specs = _column_specs(area, args.exog)

    raw = {}
    for key, (params, value_tag) in queries.items():
        try:
            raw[key] = _fetch_frame(client, params, value_tag,
                                    start_utc, end_utc, key, quiet=False)
        except UnexpectedResolution as exc:
            # The whole point of this tool is to report rather than abort, so a
            # resolution change is a finding, not a failure.
            print(f"\n  [{key}] resolution changes inside this range:\n    {exc}\n")
            print(f"  -> the hourly era for {key!r} ends before {end_date.date()}; "
                  f"lower --end until this clears.")
            return 2

    reports = {}
    for label, key, aggregate, filter_fn in column_specs:
        frame = raw[key]
        if filter_fn is not None:
            try:
                frame = filter_fn(frame)
            except ValueError as exc:
                print(f"\n{label}: {exc.args[0]}")
                reports[label] = {"empty": True}
                continue
        reports[label] = analyse(
            to_series(frame, aggregate=aggregate),
            area.tz, start_date, end_date, args.max_gap,
        )

    print()
    for label, report in reports.items():
        if report["empty"]:
            print(f"{label:14} NO DATA")
            continue
        total = (end_date - start_date).days * 24 + 24
        print(f"{label:14} {report['first']} -> {report['last']}")
        print(f"{'':14} {report['hours']:,} hours present, {report['missing']:,} "
              f"missing of {total:,} ({report['missing'] / total:.2%})")
        if report["oversized"]:
            print(f"{'':14} {len(report['oversized'])} gap(s) longer than "
                  f"{args.max_gap}h:")
            worst = sorted(report["oversized"], key=lambda r: -r[2])
            for run_start, _, hours in worst[:args.show_gaps]:
                print(f"{'':16} {run_start}  {hours}h")
        else:
            print(f"{'':14} no gaps longer than {args.max_gap}h")
        print()

    clean = first_clean_date(reports, args.max_gap)
    if clean is None:
        print("At least one series has no data at all; nothing to recommend.")
        return 1

    # Gaps well past the start cannot be fixed by moving --start without
    # throwing away good data, so call them out separately.
    midrange = []
    for label, report in reports.items():
        if report["empty"]:
            continue
        for run_start, run_end, hours in report["oversized"]:
            if pd.Timestamp(run_end).normalize() > clean:
                midrange.append((label, run_start, hours))

    if clean > start_date:
        remaining = (end_date - clean).days + 1
        print(f"Recommended --start {clean.date()}  "
              f"({remaining:,} days, {(end_date - clean).days // 364} full years)")
    else:
        print(f"--start {start_date.date()} is already clean.")

    if midrange:
        print(f"\nGaps remain after that date -- raise --max-gap to cover them, "
              f"or accept interpolation:")
        for label, run_start, hours in sorted(midrange, key=lambda r: -r[2])[:args.show_gaps]:
            print(f"  {label}: {run_start} ({hours}h)")
        longest = max(h for _, _, h in midrange)
        print(f"\n  Smallest --max-gap that covers them: {longest}")

    if end_date >= MTU_SWITCHOVER:
        print(f"\nNote: --end is at or past {MTU_SWITCHOVER.date()}, the EU-wide "
              f"deadline for the 15-minute market time unit. Zones switched well "
              f"before it and on different dates (DK1 in April 2025, NO2 in "
              f"February), so check the resolutions reported above per series.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, ValueError, TransparencyError) as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        sys.exit(1)
