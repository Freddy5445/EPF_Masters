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

Missing values are not handled here. All cleaning -- the local grid, the
epftoolbox DST convention and the causal imputation -- happens once in
data_cleaning_v2.ipynb, so every model tested against this data provably sees
identically prepared inputs. This script refuses to run on a dataset with gaps.

Per-day progress, timings and an ETA are printed as it goes, and every run
writes forecasts, per-day timings and a JSON manifest under experiments/.
"""

import argparse
import glob
import os
import sys

import pandas as pd

from lear_dk1.backtest import run_ensemble
from lear_dk1.compat import minimum_calibration_window

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASETS = os.path.join(THIS_DIR, "datasets")
DEFAULT_OUT = os.path.join(THIS_DIR, "experiments")

# The LEAR ensemble of Lago et al. (2021): 8 weeks, 12 weeks, 3 years, 4 years.
# The two short windows have fewer samples than regressors, which scikit-learn's
# LassoLarsIC refuses by default; LEARCompat.recalibrate supplies the noise
# variance so they fit, as the published implementation did.
DEFAULT_WINDOWS = (56, 84, 1092, 1456)

# ENTSO-E publishes DK1 too sparsely before this to be worth imputing: the
# Transparency Platform went live on 2015-01-05 and took a couple of days to
# produce complete days.
DEFAULT_DATA_START = "2015-01-07"

# DK1 and DK2 day-ahead *prices* moved to 15-minute market time units on
# 2025-10-01, so the hourly series the epftoolbox models require ends
# 2025-09-30. (Not to be confused with 2025-04-08, when DK1/DK2 *generation
# forecasts* moved to 15 minutes -- a separate transition that does not bound
# the price series.)
DEFAULT_END_TEST = "2025-09-30"

# 728 days (two 364-day "years", the epftoolbox convention) ending there.
DEFAULT_BEGIN_TEST = "2023-10-04"


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

    run_name = args.run_name or f"{args.dataset}_{begin_test.date()}_{end_test.date()}"

    if args.smoke:
        # A smoke run exists to exercise the fitting path, so it must never resume: a
        # previous run of the same ten days would be skipped day by day and report a
        # 0:00:00 "success" that proves nothing about the pipeline it was meant to check.
        #
        # The suffix is forced on even when --run-name was given, so a smoke run can only
        # ever discard checkpoints from a directory that a smoke run created. Only the
        # per-window checkpoints go; run_metadata.json is rewritten by the run itself.
        if not run_name.endswith("_smoke"):
            run_name += "_smoke"
        stale = sorted(
            glob.glob(os.path.join(args.out_dir, run_name, "forecasts_cw*.csv"))
            + glob.glob(os.path.join(args.out_dir, run_name, "timings_cw*.csv"))
        )
        for path in stale:
            os.remove(path)
        if stale:
            print(f"Discarded {len(stale)} checkpoint file(s) from an earlier smoke run "
                  f"in {run_name}; starting from scratch.\n")

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
            print(f"\nRe-run data_cleaning_v2.ipynb to rebuild the cleaned panel, "
                  f"then rebuild this CSV with run_lear_from_clean.py.",
                  file=sys.stderr)
        elif "calibration_window" in lowered:
            print(f"\nMinimum window is {minimum_calibration_window()} days, the "
                  f"shortest window Lago et al. (2021) run LEAR on.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
