"""
Run the LEAR backtest on a zone taken from the cleaned hourly panel.

``data_cleaning.ipynb`` writes ``datasets/nordic_baltic_clean_hourly.parquet``: every
Nordic/Baltic series in one long, UTC, uniformly hourly table. ``run_lear_dk1.py`` reads
the epftoolbox layout instead -- one CSV per zone, naive local time, price first and the
exogenous inputs after it. This script is the bridge, and nothing more: it projects one
zone out of the panel into that CSV and then hands over to ``run_lear_dk1.main()``, so
the argument handling, window checks, causal imputation and backtest are the same code
paths a normal run uses.

Smoke-test the pipeline on ten days:

    python run_lear_from_clean.py --smoke

Any other flag is passed straight through to run_lear_dk1.py:

    python run_lear_from_clean.py --zone NO1 --exog load-wind-hydro --windows 456 --smoke

Commands are given on one line: this project is developed from PowerShell, where a
trailing ``\`` is not a line continuation and truncates the command.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import run_lear_dk1
from entsoe_tp.areas import lookup
from entsoe_tp.hourly import assert_epftoolbox_grid, to_local_hourly_grid
from lear_dk1.compat import minimum_calibration_window

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PANEL = os.path.join(THIS_DIR, "datasets", "nordic_baltic_clean_hourly.parquet")

# Column layouts, mirroring entsoe_tp.build_dataset._column_specs. Each entry is
# (column name, variable, psr_types) -- psr_types None meaning "this variable has none".
# read_data renames columns positionally to Price/Exogenous N, so the *order* is what
# binds the data to the model; the names are for whoever opens the CSV.
PRICE = ("Day-ahead price", "price", None)
LOAD = ("Day-ahead load forecast", "load_forecast", None)
WIND = ("Day-ahead wind forecast (on- and offshore)", "generation_forecast",
        ("wind_onshore", "wind_offshore"))
SOLAR = ("Day-ahead solar forecast", "generation_forecast", ("solar",))
WINDSOLAR = ("Day-ahead wind and solar forecast", "generation_forecast",
             ("wind_onshore", "wind_offshore", "solar"))
HYDRO = ("Water reservoir and hydro storage", "reservoir", None)

LAYOUTS = {
    "load-windsolar": [PRICE, LOAD, WINDSOLAR],
    "load-wind-solar": [PRICE, LOAD, WIND, SOLAR],
    # The panel carries reservoir for the hydro zones, which the CSV-building path in
    # build_dataset offers as --include-reservoir. Hydro goes last so the columns before
    # it keep the positions read_data binds on.
    "load-wind-hydro": [PRICE, LOAD, WIND, HYDRO],
    "load-wind-solar-hydro": [PRICE, LOAD, WIND, SOLAR, HYDRO],
}


def series_for(panel, zone, variable, psr_types):
    """One UTC series for a column spec, summing the production types it names.

    Summing is done across columns rather than by grouping rows so that a missing
    component makes the sum missing. A wind total that silently means "onshore only"
    for the hours offshore was not published would be a quiet lie about the level;
    left as NaN it is a gap, which the imputation in lear_dk1 fills causally and counts.
    """
    rows = panel[(panel.zone == zone) & (panel.variable == variable)]
    if psr_types is not None:
        rows = rows[rows.psr_type.isin(psr_types)]
        missing = sorted(set(psr_types) - set(rows.psr_type.unique()))
    else:
        missing = []

    if rows.empty:
        return None, psr_types or [variable]

    wide = rows.pivot_table(index="timestamp_utc", columns="psr_type", values="value")
    total = wide.sum(axis=1, min_count=wide.shape[1])
    return total.sort_index(), missing


def build_csv(panel_path, zone, layout, out_dir, dataset_name, quiet=False):
    """Project one zone out of the panel into the epftoolbox CSV layout.

    Returns the local date span written, so the caller can default the test range to
    what actually exists rather than to a hard-coded date.
    """
    panel = pd.read_parquet(panel_path)
    for column in ("zone", "variable", "psr_type"):
        panel[column] = panel[column].astype("object").fillna("").astype(str)

    zones = sorted(panel.zone.unique())
    if zone not in zones:
        raise ValueError(f"{zone} is not in the panel. It holds: {', '.join(zones)}")

    tz = lookup(zone).tz
    specs = LAYOUTS[layout]

    columns, absent = {}, []
    for name, variable, psr_types in specs:
        series, missing = series_for(panel, zone, variable, psr_types)
        if series is None:
            absent.append((name, variable, psr_types))
            continue
        if missing:
            print(f"note: {name} covers only {', '.join(sorted(set(psr_types) - set(missing)))} "
                  f"-- {', '.join(missing)} is not in the panel for {zone}")
        columns[name] = series

    if absent:
        have = sorted(
            v + (f" [{p}]" if p else "")
            for v, p in panel.loc[panel.zone == zone, ["variable", "psr_type"]]
                             .drop_duplicates().itertuples(index=False)
        )
        wanted = ", ".join(name for name, _, _ in absent)
        raise ValueError(
            f"{zone} has no data for: {wanted}. The cleaning notebook drops series that are "
            f"constant or near-constant, which is why some zones lack solar or hydro entirely. "
            f"{zone} has: {'; '.join(have)}. Choose a layout that fits, from "
            f"{', '.join(LAYOUTS)}."
        )

    # The panel spans whole local days by construction -- the notebook cuts it at the last
    # hour before 15-minute prices begin -- but derive the range rather than assume it.
    # Naive on purpose: these bound a naive local grid, and a tz-aware bound would make
    # date_range build an aware one, which is exactly what the layout must not have.
    span = pd.concat(columns.values(), axis=1).index
    local = span.tz_convert(tz).tz_localize(None)
    start_date, end_date = local.min().normalize(), local.max().normalize()
    if local.max().hour != 23:
        end_date -= pd.Timedelta(days=1)

    # allow_gaps=True keeps this a pure projection: fall-back duplicates are averaged
    # because a naive local grid has one 02:00 slot, but nothing is invented. Filling is
    # lear_dk1's job, where the method is a stated choice and every filled value is counted.
    frame = pd.DataFrame({
        name: to_local_hourly_grid(series, tz, start_date, end_date, allow_gaps=True)
        for name, series in columns.items()
    })
    assert_epftoolbox_grid(frame, allow_nan=True)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset_name}.csv")
    frame.index.name = f"Date ({tz} local time, naive, ISO 8601)"
    frame.to_csv(path, date_format="%Y-%m-%dT%H:%M:%S")

    if not quiet:
        nan = frame.isna().sum()
        print(f"{zone}: {len(frame):,} hours, {start_date.date()} to {end_date.date()} "
              f"({len(frame) // 24:,} days), {len(frame.columns)} columns")
        for name in frame.columns:
            print(f"  {name:<45} {int(nan[name]):>6,} missing "
                  f"({100 * nan[name] / len(frame):.2f}%)")
        print(f"written: {path}\n")

    return path, start_date, end_date


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the LEAR backtest on a zone from the cleaned hourly panel.",
        epilog="Unrecognised flags are passed through to run_lear_dk1.py.")
    parser.add_argument("--panel", default=DEFAULT_PANEL,
                        help="Cleaned hourly panel written by data_cleaning.ipynb")
    parser.add_argument("--zone", default="DK1", help="Bidding zone (default: DK1)")
    parser.add_argument("--exog", default="load-wind-solar", choices=sorted(LAYOUTS),
                        help="Exogenous layout (default: load-wind-solar)")
    parser.add_argument("--datasets-dir", default=os.path.join(THIS_DIR, "datasets"),
                        help="Where the projected CSV is written")
    parser.add_argument("--csv-only", action="store_true",
                        help="Write the CSV and stop, without running the backtest")
    args, passthrough = parser.parse_known_args(argv)

    if not os.path.exists(args.panel):
        print(f"error: no cleaned panel at {args.panel}", file=sys.stderr)
        print("Run data_cleaning.ipynb to build it.", file=sys.stderr)
        return 1

    # A name of its own, so the projected CSV never overwrites a dataset that
    # build_dataset produced for the same zone.
    dataset_name = f"{args.zone}_clean"

    try:
        _, start_date, end_date = build_csv(
            args.panel, args.zone, args.exog, args.datasets_dir, dataset_name)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.csv_only:
        return 0

    # LassoLarsIC will not fit fewer samples than features, so check the window against
    # this layout's exogenous count before starting work rather than failing partway in.
    n_exog = len(LAYOUTS[args.exog]) - 1  # every column but the price
    floor = minimum_calibration_window(n_exog)
    if not any(flag.startswith("--windows") for flag in passthrough):
        feasible = [w for w in run_lear_dk1.DEFAULT_WINDOWS if w > floor]
        if not feasible:
            print(f"error: {args.exog} has {n_exog} exogenous inputs, needing a window "
                  f"longer than {floor} days; none of the defaults "
                  f"{run_lear_dk1.DEFAULT_WINDOWS} qualify. Pass --windows explicitly.",
                  file=sys.stderr)
            return 1
        if len(feasible) != len(run_lear_dk1.DEFAULT_WINDOWS):
            dropped = [w for w in run_lear_dk1.DEFAULT_WINDOWS if w <= floor]
            print(f"note: dropping window(s) {dropped} -- {n_exog} exogenous inputs need "
                  f"more than {floor} days\n")
        passthrough += ["--windows", ",".join(str(w) for w in feasible)]

    # Default the test range to what the panel actually holds. The last day is its last
    # complete local day; the first is one calibration window plus a margin after the
    # start, so the smallest window has something to train on.
    if not any(flag.startswith("--end-test") for flag in passthrough):
        passthrough += ["--end-test", str(end_date.date())]
    if not any(flag.startswith("--begin-test") for flag in passthrough):
        smallest = min(int(w) for w in _windows_from(passthrough, floor))
        begin = start_date + pd.Timedelta(days=smallest + 14)
        default = pd.Timestamp(run_lear_dk1.DEFAULT_BEGIN_TEST)
        passthrough += ["--begin-test", str(max(begin, default).date())]
    if not any(flag.startswith("--data-start") for flag in passthrough):
        passthrough += ["--data-start", str(start_date.date())]

    return run_lear_dk1.main(
        ["--dataset", dataset_name, "--datasets-dir", args.datasets_dir] + passthrough)


def _windows_from(flags, floor):
    """The calibration windows named in ``flags``, or the feasible defaults."""
    for i, flag in enumerate(flags):
        if flag == "--windows" and i + 1 < len(flags):
            return [w for w in flags[i + 1].split(",") if w.strip()]
        if flag.startswith("--windows="):
            return [w for w in flag.split("=", 1)[1].split(",") if w.strip()]
    return [str(w) for w in run_lear_dk1.DEFAULT_WINDOWS if w > floor]


if __name__ == "__main__":
    sys.exit(main())
