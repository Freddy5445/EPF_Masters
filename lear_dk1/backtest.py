"""
Walk-forward backtesting of LEAR with daily recalibration.

This replaces ``epftoolbox.models.evaluate_lear_in_test_dataset``, which cannot
run on this stack: it uses ``np.NaN``, removed in NumPy 2.0. The recalibration
protocol here is the same one -- for each test day, refit on the trailing
calibration window with that day's prices hidden, then forecast its 24 hours --
but it adds three things a multi-hour run needs:

* **Checkpoint and resume.** Forecasts and timings are flushed after every day.
  Re-running the same command picks up where it stopped instead of starting over.
* **Live progress.** Each day prints its own time, the running average, elapsed
  wall time, an ETA, and the running MAE, so a long run can be followed.
* **Timing telemetry.** Per-day fit times are recorded to CSV and summarised in
  a JSON manifest, so runtime can be compared against other models later.

Outputs land in ``<out_dir>/<run_name>/``:

===========================  ================================================
File                         Contents
===========================  ================================================
``forecasts_cw<N>.csv``      Forecast prices, one row per day, columns h0..h23
``timings_cw<N>.csv``        Per-day wall time and training-set size
``run_metadata.json``        Config, environment, and per-window summaries
===========================  ================================================
"""

import json
import os
import platform
import sys
import time
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

from .compat import LEARCompat, PROJECT_ROOT, minimum_calibration_window, n_features
from .impute import first_complete_day, format_report, impute_frame

# read_data lives in a TensorFlow-free subpackage, so this import is safe.
if os.path.join(PROJECT_ROOT, "epftoolbox") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "epftoolbox"))
from epftoolbox.data import read_data  # noqa: E402
from epftoolbox.evaluation import MAE  # noqa: E402

HOURS = [f"h{h}" for h in range(24)]

# ``run_id`` distinguishes measurements taken by different invocations. Timings
# are appended, never rewritten, so a resumed or repeated run adds rows rather
# than replacing them -- every row is a real measurement, and the id says which
# invocation produced it.
TIMING_COLUMNS = [
    "run_id", "date", "seconds", "n_train_days", "n_train_samples",
    "n_features", "calibration_window", "model",
]


def _fmt(seconds):
    """Human-readable duration, e.g. ``1:23:45``."""
    return str(timedelta(seconds=int(round(seconds))))


def _load_checkpoint(path):
    """Return previously computed forecasts, or None if there is no checkpoint."""
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    # A partially written final row would poison the resume; drop incomplete rows.
    return frame.dropna(how="any")


def _append_timing(path, row):
    """Append one timing record, writing the header only for a new file."""
    new_file = not os.path.exists(path)
    pd.DataFrame([row], columns=TIMING_COLUMNS).to_csv(
        path, mode="a", header=new_file, index=False
    )


def environment_metadata():
    """Environment facts worth recording alongside timings.

    Runtime is only comparable across models measured on the same machine, so
    the machine is part of the result.
    """
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def run_window(df_train, df_test, calibration_window, run_dir, run_id,
               quiet=False):
    """Backtest one calibration window, resuming any partial run.

    Returns ``(forecasts, summary)`` where ``forecasts`` is a DataFrame indexed
    by test day with columns ``h0..h23``, and ``summary`` holds timing totals.
    """
    tag = f"cw{calibration_window:04d}"
    forecast_path = os.path.join(run_dir, f"forecasts_{tag}.csv")
    timing_path = os.path.join(run_dir, f"timings_{tag}.csv")

    n_exogenous = len(df_train.columns) - 1
    n_feat = n_features(n_exogenous)
    minimum = minimum_calibration_window(n_exogenous)
    if calibration_window < minimum:
        raise ValueError(
            f"calibration_window={calibration_window} is too small for a dataset "
            f"with {n_exogenous} exogenous inputs ({n_feat} features). LassoLarsIC "
            f"needs more samples than features; use at least {minimum} days."
        )

    forecast_dates = df_test.index[::24]
    real_values = pd.DataFrame(
        df_test.loc[:, ["Price"]].values.reshape(-1, 24),
        index=forecast_dates, columns=HOURS,
    )

    forecasts = pd.DataFrame(index=forecast_dates, columns=HOURS, dtype="float64")
    done = _load_checkpoint(forecast_path)
    if done is not None and len(done):
        common = forecasts.index.intersection(done.index)
        forecasts.loc[common, :] = done.loc[common, :]
        if not quiet:
            print(f"  [{tag}] resuming: {len(common)} of {len(forecast_dates)} "
                  f"days already done")

    pending = forecasts.index[forecasts.isna().any(axis=1)]
    model = LEARCompat(calibration_window=calibration_window)

    started = time.time()
    spent = 0.0
    completed = 0

    for date in pending:
        # Simulate the information set of a forecaster on the morning of `date`:
        # everything up to and including the previous day, plus that day's
        # exogenous forecasts, but not its prices.
        data_available = pd.concat(
            [df_train, df_test.loc[:date + pd.Timedelta(hours=23), :]], axis=0
        )
        data_available.loc[date:date + pd.Timedelta(hours=23), "Price"] = np.nan

        day_started = time.time()
        with warnings.catch_warnings():
            # LassoLarsIC is noisy about convergence on some folds; upstream
            # suppresses the same warnings inside recalibrate().
            warnings.simplefilter("ignore")
            Yp = model.recalibrate_and_forecast_next_day(
                df=data_available,
                calibration_window=calibration_window,
                next_day_date=date,
            )
        day_seconds = time.time() - day_started

        forecasts.loc[date, :] = np.asarray(Yp).ravel()
        spent += day_seconds
        completed += 1

        n_train_days = min(calibration_window,
                           len(data_available.loc[:date]) // 24)
        _append_timing(timing_path, {
            "run_id": run_id,
            "date": date.isoformat(),
            "seconds": round(day_seconds, 4),
            "n_train_days": n_train_days,
            "n_train_samples": max(n_train_days - 7, 0),
            "n_features": n_feat,
            "calibration_window": calibration_window,
            "model": "LEAR",
        })

        # Checkpoint every day: a 6-hour run must never lose more than one day.
        forecasts.to_csv(forecast_path)

        if not quiet:
            filled = forecasts.notna().all(axis=1)
            mae = MAE(forecasts[filled].values.astype(float),
                      real_values[filled].values.astype(float))
            remaining = len(pending) - completed
            eta = (spent / completed) * remaining
            print(
                f"  [{tag}] {completed}/{len(pending)}  {date.date()}  "
                f"{day_seconds:5.1f}s  avg {spent / completed:5.1f}s  "
                f"elapsed {_fmt(time.time() - started)}  ETA {_fmt(eta)}  "
                f"MAE {mae:.3f}",
                flush=True,
            )

    filled = forecasts.notna().all(axis=1)
    summary = {
        "calibration_window": calibration_window,
        "n_features": n_feat,
        "forecast_days": int(filled.sum()),
        "days_computed_this_run": completed,
        "seconds_this_run": round(spent, 2),
        "seconds_per_day": round(spent / completed, 3) if completed else None,
        "forecast_file": os.path.basename(forecast_path),
        "timing_file": os.path.basename(timing_path),
    }
    if filled.any():
        summary["mae"] = float(MAE(forecasts[filled].values.astype(float),
                                   real_values[filled].values.astype(float)))

    return forecasts, summary


def run_ensemble(dataset, datasets_dir, begin_test_date, end_test_date,
                 calibration_windows, out_dir, run_name, data_start=None,
                 impute=True, max_linear=3, quiet=False):
    """Backtest several calibration windows and record the whole run.

    ``begin_test_date`` and ``end_test_date`` must be datetime-like, not strings:
    ``read_data`` parses strings with ``dayfirst=True``, which silently reads an
    ISO date like ``"2023-10-03"`` as 3 October and yields ``2023-03-10``.
    """
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Coerce here too, so a caller passing a string still gets correct dates.
    begin_test_date = pd.Timestamp(begin_test_date)
    end_test_date = pd.Timestamp(end_test_date)

    df_train, df_test = read_data(
        path=datasets_dir, dataset=dataset,
        begin_test_date=begin_test_date, end_test_date=end_test_date,
    )

    # Trimming and imputation are done on the whole series, then re-split at the
    # same boundary: filling from a neighbouring week needs to see across the
    # train/test join, and a gap at the very start of training would otherwise
    # be filled differently than one a few rows later.
    combined = pd.concat([df_train, df_test], axis=0)
    test_start = df_test.index[0]

    if data_start is not None:
        data_start = pd.Timestamp(data_start)
        dropped = int((combined.index < data_start).sum())
        combined = combined.loc[data_start:]
        if dropped and not quiet:
            print(f"Trimmed {dropped:,} hours before {data_start}")
        if combined.empty:
            raise ValueError(
                f"--data-start {data_start} leaves no data; the dataset ends at "
                f"{df_test.index[-1]}."
            )
        if test_start <= data_start:
            raise ValueError(
                f"--data-start {data_start} is at or after the test start "
                f"{test_start}; there would be no training data."
            )

    imputation = None
    trimmed_no_history = 0
    if impute and combined.isna().any().any():
        combined, imputation = impute_frame(combined, max_ffill=max_linear)
        if not quiet:
            print("Imputed missing values (past observations only):")
            print(format_report(imputation, len(combined)))

        # Causal imputation cannot fill hours with no history, so those are
        # dropped rather than invented.
        if combined.isna().any().any():
            usable_from = first_complete_day(combined)
            trimmed_no_history = int((combined.index < usable_from).sum())
            combined = combined.loc[usable_from:]
            if not quiet:
                print(f"  Trimmed {trimmed_no_history:,} leading hour(s) with no "
                      f"history to impute from; data now starts {usable_from}")
            if combined.empty or usable_from >= test_start:
                raise ValueError(
                    f"After dropping unfillable leading hours the data starts at "
                    f"{usable_from}, at or after the test start {test_start}. "
                    f"Raise --data-start."
                )
        if not quiet:
            print()
    elif combined.isna().any().any():
        counts = combined.isna().sum()
        raise ValueError(
            f"The dataset contains missing values and imputation is disabled: "
            f"{counts[counts > 0].to_dict()}. LEAR cannot be fitted on NaN."
        )

    df_train = combined.loc[:test_start - pd.Timedelta(hours=1)]
    df_test = combined.loc[test_start:]

    n_exogenous = len(df_train.columns) - 1
    test_days = len(df_test) // 24

    if not quiet:
        print(f"Dataset {dataset}: {n_exogenous} exogenous inputs, "
              f"{n_features(n_exogenous)} LEAR features")
        print(f"Train {df_train.index[0]} -> {df_train.index[-1]} "
              f"({len(df_train) // 24} days)")
        print(f"Test  {df_test.index[0]} -> {df_test.index[-1]} ({test_days} days)")
        print(f"Windows: {', '.join(str(w) for w in calibration_windows)}\n")

    # Identifies this invocation in the appended timing files.
    run_id = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")

    metadata = {
        "run_id": run_id,
        "model": "LEAR",
        "dataset": dataset,
        "n_exogenous": n_exogenous,
        "n_features": n_features(n_exogenous),
        "train_start": df_train.index[0].isoformat(),
        "train_end": df_train.index[-1].isoformat(),
        "test_start": df_test.index[0].isoformat(),
        "test_end": df_test.index[-1].isoformat(),
        "test_days": test_days,
        "calibration_windows": list(calibration_windows),
        "data_start": str(data_start) if data_start is not None else None,
        # What fraction of the input was invented, and how -- needed to report
        # the result honestly.
        "imputation": {
            "applied": imputation is not None,
            "causal": True,
            "max_forward_fill_hours": max_linear,
            "trimmed_leading_hours_no_history": trimmed_no_history,
            "columns": imputation,
        },
        "environment": environment_metadata(),
        "started_at": pd.Timestamp.now().isoformat(),
        "windows": [],
    }
    metadata_path = os.path.join(run_dir, "run_metadata.json")

    overall_started = time.time()
    results = {}

    for window in calibration_windows:
        if not quiet:
            print(f"Calibration window {window} days")
        forecasts, summary = run_window(
            df_train, df_test, window, run_dir, run_id, quiet=quiet
        )
        results[window] = forecasts
        metadata["windows"].append(summary)

        # Rewrite the manifest after each window so a crash still leaves a
        # readable record of what finished.
        metadata["elapsed_seconds"] = round(time.time() - overall_started, 2)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        if not quiet:
            print(f"  [{window}] done: {summary.get('mae', float('nan')):.3f} MAE, "
                  f"{_fmt(summary['seconds_this_run'])} this run\n")

    metadata["finished_at"] = pd.Timestamp.now().isoformat()
    metadata["elapsed_seconds"] = round(time.time() - overall_started, 2)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if not quiet:
        print(f"Run complete in {_fmt(metadata['elapsed_seconds'])}")
        print(f"Artifacts in {run_dir}")

    return results, metadata
