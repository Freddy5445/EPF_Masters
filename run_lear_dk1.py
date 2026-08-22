"""
Run the LEAR ensemble backtest on a bidding-zone dataset.

Build the dataset first (needs an ENTSO-E token; see entsoe_tp/README.md).
Commands are given on one line: this project is developed from PowerShell,
where a trailing ``\`` is not a line continuation and truncates the command.

    python -m entsoe_tp.build_dataset --zone DK1 --start 2015-01-05 --end 2025-09-30 --exog load-wind-solar

Then smoke-test the pipeline on a few days with one window before committing to
the full run:

    python run_lear_dk1.py --smoke

Then the real thing (hours -- it checkpoints, so it can be interrupted and
resumed with the same command):

    python run_lear_dk1.py

Missing values are imputed here rather than at download time, so that how gaps
are filled stays a stated modelling choice: short gaps linearly, longer ones
from the same hour in the nearest week. Every run records how much of its input
was imputed, and by which method, in run_metadata.json.

Per-day progress, timings and an ETA are printed as it goes, and every run
writes forecasts, per-day timings and a JSON manifest under experiments/.
"""

import argparse
import os
import sys

import pandas as pd

from lear_dk1.backtest import run_ensemble
from lear_dk1.compat import minimum_calibration_window

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASETS = os.path.join(THIS_DIR, "datasets")
DEFAULT_OUT = os.path.join(THIS_DIR, "experiments")

# Feasible analogue of the LEAR ensemble from Lago et al. (2021). Their 56- and
# 84-day windows cannot be used here: modern scikit-learn's LassoLarsIC refuses
# to fit when samples < features, and LEAR has 247 features with two exogenous
# inputs and 319 with three.
DEFAULT_WINDOWS = (364, 728, 1092, 1456)

# ENTSO-E publishes DK1 too sparsely before this to be worth imputing: the
# Transparency Platform went live on 2015-01-05 and took a couple of days to
# produce complete days.
DEFAULT_DATA_START = "2015-01-07"

# DK1 day-ahead moved to 15-minute market time units on 2025-04-08, so the
# hourly series the epftoolbox models require ends the day before.
DEFAULT_END_TEST = "2025-04-07"

# 728 days (two 364-day "years", the epftoolbox convention) ending there.
DEFAULT_BEGIN_TEST = "2023-04-11"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Backtest the LEAR model with daily recalibration.")
    parser.add_argument("--dataset", default="DK1",
                        help="Dataset name; reads <datasets-dir>/<dataset>.csv "
                             "(default: DK1)")
    parser.add_argument("--datasets-dir", default=DEFAULT_DATASETS,
                        help="Directory holding the dataset CSV")
    parser.add_argument("--data-start", default=DEFAULT_DATA_START,
                        help=f"Ignore data before this day, YYYY-MM-DD "
                             f"(default: {DEFAULT_DATA_START}). ENTSO-E coverage "
                             f"before this is too sparse to be worth imputing.")
    parser.add_argument("--begin-test", default=DEFAULT_BEGIN_TEST,
                        help=f"First test day, YYYY-MM-DD "
                             f"(default: {DEFAULT_BEGIN_TEST})")
    parser.add_argument("--end-test", default=DEFAULT_END_TEST,
                        help=f"Last test day inclusive, YYYY-MM-DD "
                             f"(default: {DEFAULT_END_TEST}, where DK1 hourly "
                             f"day-ahead data ends)")
    parser.add_argument("--no-impute", action="store_true",
                        help="Fail instead of filling missing values. LEAR cannot "
                             "be fitted on NaN, so this only reports the problem.")
    parser.add_argument("--max-linear", type=int, default=3,
                        help="Longest gap to fill by linear interpolation, in "
                             "hours (default 3). Longer gaps use the same hour "
                             "from the nearest available week.")
    parser.add_argument("--windows", default=",".join(str(w) for w in DEFAULT_WINDOWS),
                        help="Comma-separated calibration windows in days "
                             f"(default: {','.join(str(w) for w in DEFAULT_WINDOWS)})")
    parser.add_argument("--out-dir", default=DEFAULT_OUT,
                        help="Where run directories are created")
    parser.add_argument("--run-name", default=None,
                        help="Run directory name (default: derived from dataset "
                             "and test range). Reuse it to resume a run.")
    parser.add_argument("--smoke", action="store_true",
                        help="Short trial: 10 test days on the smallest window, "
                             "into a separate run directory. Use to validate the "
                             "pipeline before a multi-hour run.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-day progress output")
    args = parser.parse_args(argv)

    try:
        windows = sorted({int(w) for w in args.windows.split(",") if w.strip()})
    except ValueError:
        print(f"error: --windows must be comma-separated integers, got "
              f"{args.windows!r}", file=sys.stderr)
        return 1
    if not windows:
        print("error: no calibration windows given", file=sys.stderr)
        return 1

    begin_test = pd.Timestamp(args.begin_test).normalize()
    end_test = pd.Timestamp(args.end_test).normalize()

    if args.smoke:
        windows = windows[:1]
        end_test = begin_test + pd.Timedelta(days=9)
        print(f"Smoke run: {windows[0]}-day window, "
              f"{begin_test.date()} to {end_test.date()}\n")

    if begin_test > end_test:
        print(f"error: --begin-test {begin_test.date()} is after --end-test "
              f"{end_test.date()}", file=sys.stderr)
        return 1

    run_name = args.run_name or (
        f"{args.dataset}_{begin_test.date()}_{end_test.date()}"
        + ("_smoke" if args.smoke else "")
    )

    # read_data wants the test range as full days: 00:00 through 23:00.
    #
    # Pass Timestamps, never strings. read_data parses strings with
    # dayfirst=True (its docstring asks for "%d/%m/%Y %H:%M"), so an ISO string
    # like "2023-10-03" is silently read as 3 October -> 2023-03-10, shifting
    # the test period by months without any error. Datetime objects pass
    # through pd.to_datetime untouched.
    begin_arg = begin_test
    end_arg = end_test + pd.Timedelta(hours=23)

    try:
        run_ensemble(
            dataset=args.dataset,
            datasets_dir=args.datasets_dir,
            begin_test_date=begin_arg,
            end_test_date=end_arg,
            calibration_windows=windows,
            out_dir=args.out_dir,
            run_name=run_name,
            data_start=pd.Timestamp(args.data_start) if args.data_start else None,
            impute=not args.no_impute,
            max_linear=args.max_linear,
            quiet=args.quiet,
        )
    except (IOError, OSError) as exc:
        print(f"error: could not read the dataset -- {exc}", file=sys.stderr)
        print(f"Build it first with:\n  python -m entsoe_tp.build_dataset "
              f"--zone {args.dataset} --start 2015-01-05 --end {DEFAULT_END_TEST} "
              f"--exog load-wind-solar --allow-gaps", file=sys.stderr)
        return 1
    except ValueError as exc:
        message = exc.args[0] if exc.args else str(exc)
        print(f"error: {message}", file=sys.stderr)

        # Add a hint only when it fits the failure. Printing the calibration
        # window floor after, say, a NaN error sends the reader the wrong way.
        lowered = message.lower()
        if "contains nan" in lowered or "missing value" in lowered:
            print(f"\nThe dataset has gaps that were not filled. Imputation runs "
                  f"by default; if --no-impute was passed, drop it. Otherwise the "
                  f"installed code may predate imputation being added -- check "
                  f"that lear_dk1/impute.py exists and pull if not.", file=sys.stderr)
        elif "calibration_window" in lowered:
            print(f"\nMinimum window is {minimum_calibration_window(2)} days for 2 "
                  f"exogenous inputs, {minimum_calibration_window(3)} for 3.",
                  file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
