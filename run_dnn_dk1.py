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
import os
import sys
import time

# Keras is chatty on import and TensorFlow logs device probing at INFO; neither
# says anything useful here and both bury the progress output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

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

    if data.isna().any().any():
        counts = data.isna().sum()
        print(f"error: the dataset has missing values "
              f"{counts[counts > 0].to_dict()}. The DNN cannot be fitted on NaN.",
              file=sys.stderr)
        print("run_lear_dk1.py imputes causally before fitting; the DNN path does "
              "not do that yet.", file=sys.stderr)
        return 1

    days = pd.date_range(begin_test, end_test, freq="D")
    print(f"Forecasting {len(days)} day(s), {len(seeds)} seed(s), "
          f"{args.calibration_years}-year window, recalibrating daily")

    forecasts = {}
    for seed in seeds:
        model = DNN(path_hyperparameter_folder=hyper_dir, experiment_id=1,
                    nlayers=args.nlayers, dataset=args.dataset,
                    calibration_window=args.calibration_years, seed=seed)
        rows = []
        for n, day in enumerate(days, 1):
            started = time.time()
            prediction = model.recalibrate_and_forecast_next_day(data, day)
            # The forecast comes back as (1, 24) when scaleY is set (the scaler's
            # inverse_transform reshapes) and (24,) when it is not, so flatten
            # rather than assume either.
            rows.append(pd.Series(np.asarray(prediction, dtype=float).reshape(-1),
                                  name=day))
            print(f"  seed {seed}  [{n}/{len(days)}] {day.date()}  "
                  f"{time.time() - started:6.1f}s")
        forecasts[seed] = pd.DataFrame(rows)
        forecasts[seed].columns = [f"h{h}" for h in range(24)]

    ensemble = sum(forecasts.values()) / len(forecasts)

    os.makedirs(args.out_dir, exist_ok=True)
    run_dir = os.path.join(
        args.out_dir, f"{args.dataset}_dnn_{begin_test.date()}_{end_test.date()}"
        + ("_smoke" if args.smoke else ""))
    os.makedirs(run_dir, exist_ok=True)
    for seed, frame in forecasts.items():
        frame.to_csv(os.path.join(run_dir, f"forecasts_seed{seed}.csv"))
    ensemble.to_csv(os.path.join(run_dir, "forecasts_ensemble.csv"))

    print(f"\nwritten: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
