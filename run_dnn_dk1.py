"""
Smoke-test the DNN on DK1.

    python run_dnn_dk1.py --smoke

That runs a short hyperparameter search, then forecasts a few days with daily
recalibration. It exists to prove the pipeline runs end to end -- not to produce
a usable model. The paper searches 1500 hyperparameter evaluations; a smoke run
does five, and five is not a search.

The DNN needs a hyperparameter file before it can be built at all, because the
search chooses the input features as well as the network. That is why this
script runs the search itself rather than expecting one to exist.

Note on units: the DNN's calibration window is in **years** (upstream trains on
the last ``calibration_window * 364`` days), where LEAR's windows are in days.

Commands are given on one line: this project is developed from PowerShell, where
a trailing ``\`` is not a line continuation and truncates the command.
"""

import argparse
import json
import os
import sys
import time

# Keras is chatty on import and TensorFlow logs device probing at INFO; neither
# says anything useful here and both bury the progress output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lear_dk1.impute import (  # noqa: E402
    first_complete_day, format_report, impute_frame,
)
# Private, but shared on purpose: the resume semantics -- in particular that an
# incomplete final row is discarded rather than trusted -- must match the LEAR
# backtest's, or the two models would recover differently from an interruption.
from lear_dk1.backtest import _load_checkpoint  # noqa: E402

HOURS = [f"h{h}" for h in range(24)]


def _append_timing(path, row):
    """Append one row to a per-seed timing file, writing the header once."""
    header = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASETS = os.path.join(THIS_DIR, "datasets")
DEFAULT_OUT = os.path.join(THIS_DIR, "experiments")

# Matches run_lear_dk1.py, so the two models are scored on the same days.
DEFAULT_DATA_START = "2015-01-07"
DEFAULT_END_TEST = "2025-04-07"
DEFAULT_BEGIN_TEST = "2023-04-11"

# The paper's DNN: 4 networks differing only in random seed, 4-year window.
DEFAULT_SEEDS = (1, 2, 3, 4)
DEFAULT_CALIBRATION_YEARS = 4
DEFAULT_MAX_EVALS = 1500


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Backtest the epftoolbox DNN with daily recalibration.")
    parser.add_argument("--dataset", default="DK1")
    parser.add_argument("--datasets-dir", default=DEFAULT_DATASETS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--hyperparameter-dir", default=None,
                        help="Where the hyperopt trials file lives "
                             "(default: <out-dir>/hyperparameters)")
    parser.add_argument("--data-start", default=DEFAULT_DATA_START)
    parser.add_argument("--begin-test", default=DEFAULT_BEGIN_TEST)
    parser.add_argument("--end-test", default=DEFAULT_END_TEST)
    parser.add_argument("--max-evals", type=int, default=DEFAULT_MAX_EVALS,
                        help=f"Hyperopt evaluations (paper: {DEFAULT_MAX_EVALS})")
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--calibration-years", type=int,
                        default=DEFAULT_CALIBRATION_YEARS,
                        help="Training window in YEARS (the DNN's unit, not days)")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Comma-separated seeds; the ensemble averages them")
    parser.add_argument("--skip-hyperopt", action="store_true",
                        help="Reuse an existing trials file instead of searching")
    parser.add_argument("--no-impute", action="store_true",
                        help="Fail instead of filling missing values. The DNN "
                             "cannot be fitted on NaN, so this only reports them.")
    parser.add_argument("--max-linear", type=int, default=3,
                        help="Longest gap to forward-fill, in hours (default 3). "
                             "Matches run_lear_dk1.py, so both models see the "
                             "same inputs.")
    parser.add_argument("--no-evaluate", action="store_true",
                        help="Skip scoring; write the forecasts only")
    parser.add_argument("--smoke", action="store_true",
                        help="5 hyperopt evaluations, 3 forecast days, 1 seed. "
                             "Proves the pipeline runs; proves nothing about accuracy.")
    args = parser.parse_args(argv)

    from dnn_dk1 import hyperopt as dnn_hyperopt

    begin_test = pd.Timestamp(args.begin_test).normalize()
    end_test = pd.Timestamp(args.end_test).normalize()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    max_evals = args.max_evals

    if args.smoke:
        max_evals = min(max_evals, 5)
        end_test = begin_test + pd.Timedelta(days=2)
        seeds = seeds[:1]
        print(f"Smoke run: {max_evals} hyperopt evals, "
              f"{begin_test.date()} to {end_test.date()}, seed {seeds[0]}.")
        print("This validates the pipeline. It is not a model.\n")

    if begin_test > end_test:
        print(f"error: --begin-test {begin_test.date()} is after --end-test "
              f"{end_test.date()}", file=sys.stderr)
        return 1

    dataset_csv = os.path.join(args.datasets_dir, f"{args.dataset}.csv")
    if not os.path.exists(dataset_csv):
        print(f"error: no dataset at {dataset_csv}", file=sys.stderr)
        print(f"Build it with run_lear_from_clean.py --zone {args.dataset} "
              f"--csv-only, or entsoe_tp.build_dataset.", file=sys.stderr)
        return 1

    hyper_dir = args.hyperparameter_dir or os.path.join(args.out_dir, "hyperparameters")

    # Hyperopt must not see the test period. read_data's `begin_test_date`
    # splits the frame, and the search scores on what follows it -- so passing
    # the real test range here would select features on the days the model is
    # then evaluated on. The search gets the year before the test period
    # instead, and the test period stays untouched.
    hyperopt_end = begin_test - pd.Timedelta(hours=1)
    hyperopt_begin = hyperopt_end.normalize() - pd.Timedelta(days=363)

    if not args.skip_hyperopt:
        print(f"Hyperparameter search: {max_evals} evaluations on "
              f"{hyperopt_begin.date()}..{hyperopt_end.date()} "
              f"(before the test period, so no leakage)")
        started = time.time()
        path = dnn_hyperopt.optimize(
            path_datasets_folder=args.datasets_dir,
            path_hyperparameters_folder=hyper_dir,
            dataset=args.dataset,
            begin_test_date=hyperopt_begin,
            end_test_date=hyperopt_end,
            max_evals=max_evals, nlayers=args.nlayers,
            calibration_window=args.calibration_years,
            quiet=True)
        print(f"  {time.time() - started:.0f}s -> {path}\n")

    from dnn_dk1 import DNN
    from epftoolbox.data import read_data

    df_train, df_test = read_data(
        path=args.datasets_dir, dataset=args.dataset,
        begin_test_date=begin_test, end_test_date=end_test + pd.Timedelta(hours=23))
    data = pd.concat([df_train, df_test])
    if args.data_start:
        data = data.loc[pd.Timestamp(args.data_start):]

    # Impute exactly as the LEAR path does, with the same module and the same
    # default. Both models must see identically prepared inputs, or a difference
    # in their scores could be a difference in data handling rather than in the
    # models. Every filled value comes from earlier observations only.
    imputation, trimmed_no_history = None, 0
    if data.isna().any().any():
        if args.no_impute:
            counts = data.isna().sum()
            print(f"error: the dataset has missing values "
                  f"{counts[counts > 0].to_dict()} and --no-impute was given. "
                  f"The DNN cannot be fitted on NaN.", file=sys.stderr)
            return 1

        data, imputation = impute_frame(data, max_ffill=args.max_linear)
        print("Imputed missing values (past observations only):")
        print(format_report(imputation, len(data)))

        # Causal imputation cannot fill hours with no history behind them, so
        # those are dropped rather than invented.
        if data.isna().any().any():
            usable_from = first_complete_day(data)
            trimmed_no_history = int((data.index < usable_from).sum())
            data = data.loc[usable_from:]
            print(f"  Trimmed {trimmed_no_history:,} leading hour(s) with no "
                  f"history to impute from; data now starts {usable_from}")
            if data.empty or usable_from >= begin_test:
                print(f"error: after dropping unfillable leading hours the data "
                      f"starts at {usable_from}, at or after the test start "
                      f"{begin_test}. Raise --data-start.", file=sys.stderr)
                return 1
        print()

    days = pd.date_range(begin_test, end_test, freq="D")
    print(f"Forecasting {len(days)} day(s), {len(seeds)} seed(s), "
          f"{args.calibration_years}-year window, recalibrating daily")

    os.makedirs(args.out_dir, exist_ok=True)
    run_dir = os.path.join(
        args.out_dir, f"{args.dataset}_dnn_{begin_test.date()}_{end_test.date()}"
        + ("_smoke" if args.smoke else ""))
    os.makedirs(run_dir, exist_ok=True)

    forecast_paths = {s: os.path.join(run_dir, f"forecasts_seed{s}.csv") for s in seeds}
    timing_paths = {s: os.path.join(run_dir, f"timings_seed{s}.csv") for s in seeds}

    # Resume whatever a previous run of the same command finished. _load_checkpoint
    # is shared with the LEAR backtest rather than reimplemented, so both models
    # treat a half-written final row identically: it is dropped and recomputed.
    forecasts = {}
    for seed in seeds:
        frame = pd.DataFrame(index=days, columns=HOURS, dtype="float64")
        done = _load_checkpoint(forecast_paths[seed])
        if done is not None and len(done):
            common = frame.index.intersection(done.index)
            frame.loc[common, :] = done.loc[common, :]
        forecasts[seed] = frame

    already = sum(1 for d in days
                  if all(forecasts[s].loc[d].notna().all() for s in seeds))
    if already:
        print(f"Resuming: {already} of {len(days)} day(s) already done for every seed")

    models = {seed: DNN(path_hyperparameter_folder=hyper_dir, experiment_id=1,
                        nlayers=args.nlayers, dataset=args.dataset,
                        calibration_window=args.calibration_years, seed=seed)
              for seed in seeds}

    # Day outer, seed inner. Every seed advances together, so an interruption
    # always leaves a *balanced* ensemble: all members cover exactly the same days
    # and what has been computed can be scored straight away. Running the seeds
    # one after another instead leaves a ragged set, and build_ensemble intersects
    # the members' indices -- so a single lagging seed would drag the whole
    # ensemble back to its own last finished day.
    #
    # This costs nothing. recalibrate() builds a fresh network for every day
    # regardless, so holding one DNN per seed adds only their hyperparameter
    # dicts. Nor does it change the numbers -- but only because
    # recalibrate_and_forecast_next_day seeds the RNG from (seed, day) before
    # building the features. Without that, upstream draws the train/validation
    # split from the unseeded global RNG, and the order in which days and seeds
    # ran would change the forecasts. See that method for the detail.
    timings = []
    run_started = time.time()
    computed_days = 0

    for n, day in enumerate(days, 1):
        day_started = time.time()
        ran = []
        for seed in seeds:
            if forecasts[seed].loc[day].notna().all():
                continue
            started = time.time()
            prediction = models[seed].recalibrate_and_forecast_next_day(data, day)
            elapsed = time.time() - started

            # The forecast comes back as (1, 24) when scaleY is set (the scaler's
            # inverse_transform reshapes) and (24,) when it is not, so flatten
            # rather than assume either.
            forecasts[seed].loc[day, :] = np.asarray(
                prediction, dtype=float).reshape(-1)
            timings.append(elapsed)
            ran.append(seed)

            # Checkpoint immediately: a 60-hour run must never lose more than one
            # recalibration. Rows for days not yet reached are still all-NaN and
            # are dropped on read, by _load_checkpoint and by the evaluator alike.
            forecasts[seed].to_csv(forecast_paths[seed])
            _append_timing(timing_paths[seed], {
                "date": day.isoformat(), "seed": seed,
                "seconds": round(elapsed, 3),
                "calibration_window_years": args.calibration_years,
                "model": "DNN",
            })

        if not ran:
            continue

        computed_days += 1
        per_day = (time.time() - run_started) / computed_days
        eta = (len(days) - n) * per_day
        print(f"  [{n}/{len(days)}] {day.date()}  "
              f"seed(s) {','.join(str(s) for s in ran)}  "
              f"{time.time() - day_started:6.1f}s   eta {eta / 3600:5.1f}h")

    # The same manifest the LEAR runs write, so both models' runs describe
    # themselves the same way.
    manifest = {
        "model": "DNN",
        "dataset": args.dataset,
        "n_exogenous": len(data.columns) - 1,
        "test_start": str(begin_test), "test_end": str(end_test),
        "test_days": len(days),
        "seeds": seeds,
        "calibration_window_years": args.calibration_years,
        "nlayers": args.nlayers,
        "hyperopt_evals": max_evals,
        "hyperopt_range": [str(hyperopt_begin.date()), str(hyperopt_end.date())],
        "data_start": str(data.index.min()),
        "imputation": {
            "applied": imputation is not None,
            "causal": True,
            "max_forward_fill_hours": args.max_linear,
            "trimmed_leading_hours_no_history": trimmed_no_history,
            "columns": imputation or {},
        },
        "seconds_per_recalibration": round(
            sum(timings) / max(len(timings), 1), 1),
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8") as h:
        json.dump(manifest, h, indent=2, default=str)

    print(f"\nwritten: {run_dir}")

    if not args.no_evaluate:
        # Scored by the LEAR evaluator, not a parallel copy of it. The ensemble
        # mean, MAE, rMAE against a weekly naive, the DM/GW tests and
        # predictions.csv are all produced by the same code that scores LEAR, so
        # the two models' figures mean the same thing and can be compared
        # directly.
        from lear_dk1.evaluate import evaluate_run
        try:
            results = evaluate_run(run_dir, dataset=args.dataset,
                                   datasets_dir=args.datasets_dir,
                                   zone=args.dataset.split("_")[0], kind="seed")
            print(f"\npredictions: {results['predictions']}")
        except ValueError as exc:
            print(f"\nnot scored: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
