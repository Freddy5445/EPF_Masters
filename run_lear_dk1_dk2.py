"""
Thesis entry point for the LEAR benchmark: DK1 and DK2, one command, pinned config.

This is the single reproducible entry point for the LEAR results reported in the
thesis. It does not reimplement any of the machinery --
``run_lear_from_clean.py``, ``run_lear_dk1.py`` and ``lear_dk1/backtest.py``
already do the work. This file only *drives* them, with every parameter fixed, so
the reported numbers never depend on remembering a flag.

    python run_lear_dk1_dk2.py --smoke     # ten days, one window, both zones
    python run_lear_dk1_dk2.py             # the full reported benchmark (hours)

Layers beneath this file, unchanged:

===========================  ================================================
Layer                        Role
===========================  ================================================
``data_cleaning_v2.ipynb``   clean and normalize 24-hour local delivery days
``run_lear_from_clean.py``   project one zone into the epftoolbox CSV layout
``run_lear_dk1.py``          the per-zone LEAR ensemble backtest CLI
``lear_dk1/backtest.py``     walk-forward daily recalibration
``lear_dk1/evaluate.py``     ensemble mean (on the price scale) and scoring
===========================  ================================================

Before the runs start this script asserts five preconditions and refuses to
proceed on any failure (see :func:`preflight`):

1. the projected CSV has exactly 4 columns for each zone (price + 3 exogenous);
2. the LEAR design matrix has exactly 319 columns, checked by building it;
3. the test period is exactly 731 days and identical for both zones;
4. the panel covers the 1463-day burn-in before the first test day for both zones;
5. the clean panel records the expected DST transition days for every series.

Data cleaning and this model run are separate concerns: anything about how the
panel was built -- gaps, imputation, DST -- lives with ``data_cleaning_v2.ipynb``,
not here.

Outputs land in ``experiments/lear_dk1_dk2_thesis/``:

=============================  ==============================================
File                           Contents
=============================  ==============================================
``forecasts_<zone>.csv``       local date, hour, actual, ensemble, per-window
``ensemble_<zone>.csv``        local date, hour, actual, ensemble -- one zone
``ensemble_forecasts.csv``     the ensemble forecast, both zones, long format
``accuracy_summary.csv``       both zones x {ensemble, each window}: MAE/RMSE/rMAE
``run_manifest.json``          the five assertion results, the finding, the runtime
=============================  ==============================================

The per-zone working state lands in
``experiments/<zone>_clean_load-wind-solar_<begin>_<end>/`` as usual -- the four
``forecasts_cw<N>.csv`` members, ``forecasts_ensemble.csv`` (their mean, same
wide shape), per-day ``timings_cw<N>.csv`` and ``run_metadata.json``. A run
checkpoints after every day and resumes on the same command.

Progress: the backtest emits a per-day line internally, but this script throttles
it to roughly one update per minute (plus the first and last day of each window)
so the console stays readable -- each kept line still carries the day count,
running MAE and the window's own ETA. After each zone the measured per-window fit
time and a projection of what is left are printed. ``--verbose`` restores every
day; ``--quiet`` silences the backtest entirely.

Commands are given on one line: this project is developed from PowerShell, where
a trailing ``\`` is not a line continuation and truncates the command.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stdout
from datetime import timedelta

import numpy as np
import pandas as pd

import run_lear_dk1
import run_lear_from_clean
from lear_dk1.compat import LEARCompat, n_features
from lear_dk1.evaluate import (
    HOURS, build_ensemble, load_forecasts, naive_weekly_mae, real_prices,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PANEL = run_lear_from_clean.DEFAULT_PANEL
DEFAULT_DATASETS = os.path.join(THIS_DIR, "datasets")
DEFAULT_OUT = os.path.join(THIS_DIR, "experiments")
SUMMARY_DIRNAME = "lear_dk1_dk2_thesis"


# ---------------------------------------------------------------------------
# Pinned configuration. Do not parameterise any of this -- a fixed, named
# configuration is the entire reason this file exists.
# ---------------------------------------------------------------------------

ZONES = ("DK1", "DK2")

# "load-wind-solar", NOT "load-windsolar". THREE separate exogenous series:
#
#   1. day-ahead load forecast
#   2. day-ahead wind forecast, onshore + offshore summed into one series
#   3. day-ahead solar forecast
#
# giving a 319-feature LEAR design matrix. "load-windsolar" (no second hyphen)
# instead sums wind AND solar into a single exogenous series, which is the
# 247-feature canonical model of Lago et al. (2021). The two layout names differ
# by exactly one hyphen and select materially different models, and nothing
# downstream raises if the wrong one is chosen -- so it is hard-coded here and
# checked before the run (N_EXOG_EXPECTED / N_FEATURES_EXPECTED). This is the
# easiest thing in the project to get wrong silently.
EXOG_LAYOUT = "load-wind-solar"

# The full published ensemble: 8 weeks, 12 weeks, ~3 years, ~4 years. The
# reported forecast is the arithmetic mean of the four window forecasts, taken
# on the price (EUR/MWh) scale -- see check_back_transformation().
WINDOWS = (56, 84, 1092, 1456)

# The reported test period, identical for both zones. Passed explicitly as
# --begin-test / --end-test so it overrides the 728-day ("two 364-day years")
# derivation in run_lear_from_clean.py.
BEGIN_TEST = pd.Timestamp("2023-10-01")
END_TEST = pd.Timestamp("2025-09-30")
TEST_DAYS_EXPECTED = 731

# Daily recalibration throughout: lambda is re-selected from scratch every test
# day. That is what lear_dk1/backtest.run_window already does; there is no flag
# and nothing here changes it.

# Burn-in. The longest window must have its full 1456 days available on the
# first test day, and LEAR's longest lag is 7 days, so 1463 days of history must
# exist before BEGIN_TEST -- i.e. back to 2019-09-29.
MAX_LAG_DAYS = 7
BURN_IN_DAYS = max(WINDOWS) + MAX_LAG_DAYS
HISTORY_START_REQUIRED = BEGIN_TEST - pd.Timedelta(days=BURN_IN_DAYS)

N_EXOG_EXPECTED = 3
# 96 lagged prices (24 h x days D-1, D-2, D-3, D-7)
# + 216 exogenous (3 series x 3 lags {D, D-1, D-7} x 24 h)
# + 7 day-of-week dummies
N_FEATURES_EXPECTED = 319

# EU DST transitions inside the panel span (2019-01-01 .. 2025-09-30 local): the
# last Sunday of March every year 2019-2025 (7), and the last Sunday of October
# 2019-2024 (6) -- the 2025 autumn change, 2025-10-26, falls after the panel
# ends.
DST_SPRING_EXPECTED = 7
DST_AUTUMN_EXPECTED = 6


def _fmt(seconds):
    return str(timedelta(seconds=int(round(seconds))))


def dataset_name(zone):
    return run_lear_from_clean.dataset_name_for(zone, EXOG_LAYOUT)


def run_dir_for(zone, out_dir, smoke):
    """The directory run_lear_dk1 writes for this zone, derived the same way."""
    end = BEGIN_TEST + pd.Timedelta(days=9) if smoke else END_TEST
    name = f"{dataset_name(zone)}_{BEGIN_TEST.date()}_{end.date()}"
    if smoke:
        name += "_smoke"
    return os.path.join(out_dir, name)


def _eu_dst_transitions(first, last):
    """Last-Sunday-of-March / last-Sunday-of-October dates within [first, last]."""
    spring, autumn = [], []
    for year in range(first.year, last.year + 1):
        for month, bucket in ((3, spring), (10, autumn)):
            day = pd.Timestamp(year, month, 31)
            while day.dayofweek != 6:
                day -= pd.Timedelta(days=1)
            if first <= day <= last:
                bucket.append(day.normalize())
    return spring, autumn


# ---------------------------------------------------------------------------
# Pre-flight assertions
# ---------------------------------------------------------------------------

def _load_panel(panel_path):
    panel = pd.read_parquet(panel_path)
    for column in ("zone", "variable", "psr_type"):
        panel[column] = panel[column].astype("object").fillna("").astype(str)
    return panel


def preflight(panel_path, datasets_dir, quiet=False):
    """Run the five assertions. Returns (results, csv_info).

    ``results`` is a list of dicts with keys ``id``, ``name``, ``passed``,
    ``detail``. ``csv_info`` maps zone -> (csv_path, first_local_date,
    last_local_date). Raises nothing on assertion failure -- the caller inspects
    ``passed`` and aborts.
    """
    results = []

    def record(assertion_id, name, passed, detail):
        results.append({"id": assertion_id, "name": name,
                        "passed": bool(passed), "detail": detail})

    panel = _load_panel(panel_path)
    have_zones = sorted(panel.zone.unique())
    for zone in ZONES:
        if zone not in have_zones:
            record(0, f"{zone} present in panel", False,
                   f"panel holds {', '.join(have_zones)}")
    if any(not r["passed"] for r in results):
        return results, {}

    # --- 5. Cleaning-stage DST normalization -----------------------------
    required = {"timestamp_local", "dst_adjustment"}
    missing_columns = sorted(required - set(panel.columns))
    if missing_columns:
        record(5, "clean panel contains DST-normalized local hours", False,
               f"missing columns: {missing_columns}")
        return results, {}
    panel["timestamp_local"] = pd.to_datetime(panel["timestamp_local"])
    span_first = panel["timestamp_local"].min().normalize()
    span_last = panel["timestamp_local"].max().normalize()
    exp_spring, exp_autumn = _eu_dst_transitions(span_first, span_last)
    spring = sorted(panel.loc[
        panel.dst_adjustment == "spring_interpolation", "timestamp_local"
    ].dt.normalize().unique())
    autumn = sorted(panel.loc[
        panel.dst_adjustment == "autumn_average", "timestamp_local"
    ].dt.normalize().unique())
    transition_counts = panel.loc[
        panel.dst_adjustment != "none"
    ].groupby(["dst_adjustment", panel.timestamp_local.dt.normalize()]).size()
    per_day = panel.groupby(
        ["series", panel.timestamp_local.dt.normalize()], observed=True
    ).size()
    dst_ok = (
        len(spring) == DST_SPRING_EXPECTED
        and len(autumn) == DST_AUTUMN_EXPECTED
        and sorted(spring) == sorted(exp_spring)
        and sorted(autumn) == sorted(exp_autumn)
        and transition_counts.eq(panel.series.nunique()).all()
        and per_day.eq(24).all()
    )
    record(
        5, "clean panel DST transition days, identical across series", dst_ok,
        f"{len(spring)} spring {[str(d.date()) for d in sorted(spring)]}, "
        f"{len(autumn)} autumn {[str(d.date()) for d in sorted(autumn)]}, "
        f"identical across {panel.series.nunique()} series, 24 hours per local day "
        f"(expected {DST_SPRING_EXPECTED} spring / {DST_AUTUMN_EXPECTED} autumn "
        f"over the panel span {span_first.date()}..{span_last.date()})",
    )

    # --- 4. Burn-in coverage from the panel ------------------------------
    for zone in ZONES:
        zone_hours = panel.loc[panel.zone == zone, "timestamp_local"]
        first = zone_hours.min().normalize()
        last = zone_hours.max().normalize()
        covered = first <= HISTORY_START_REQUIRED and last >= END_TEST
        record(
            4, f"{zone} panel covers the {BURN_IN_DAYS}-day burn-in", covered,
            f"panel {first.date()}..{last.date()}; need <= "
            f"{HISTORY_START_REQUIRED.date()} and >= {END_TEST.date()} "
            f"({max(WINDOWS)}-day window + {MAX_LAG_DAYS}-day max lag)",
        )

    # --- 1. Projected CSV column count ----------------------------------
    csv_info = {}
    for zone in ZONES:
        buf = io.StringIO()
        with redirect_stdout(buf):
            path, first_date, last_date = run_lear_from_clean.build_csv(
                panel_path, zone, EXOG_LAYOUT, datasets_dir,
                dataset_name(zone), quiet=True,
            )
        csv_info[zone] = (path, first_date, last_date)
        ncols = len(pd.read_csv(path, index_col=0, nrows=8).columns)
        record(
            1, f"{zone} projected CSV has 4 columns (price + 3 exogenous)",
            ncols == 1 + N_EXOG_EXPECTED,
            f"{ncols} columns in {os.path.basename(path)}",
        )

    # --- 3. Test period is 731 days, identical for both zones ------------
    configured_days = (END_TEST - BEGIN_TEST).days + 1
    spans = {
        zone: (info[1] <= BEGIN_TEST and info[2] >= END_TEST)
        for zone, info in csv_info.items()
    }
    record(
        3, "test period is exactly 731 days, identical for both zones",
        configured_days == TEST_DAYS_EXPECTED and all(spans.values()),
        f"{BEGIN_TEST.date()}..{END_TEST.date()} = {configured_days} days; "
        f"both zone CSVs span it: {spans}",
    )

    # --- 2. LEAR design matrix has 319 columns (built, not derived) -----
    arithmetic_ok = 96 + 72 * N_EXOG_EXPECTED + 7 == N_FEATURES_EXPECTED
    for zone in ZONES:
        buf = io.StringIO()
        with redirect_stdout(buf):
            df_train, df_test = _read_dataset(zone, datasets_dir,
                                              BEGIN_TEST, BEGIN_TEST)
        combined = pd.concat([df_train, df_test])
        next_day = BEGIN_TEST
        train = combined.loc[:next_day - pd.Timedelta(hours=1)].iloc[-56 * 24:]
        test = combined.loc[next_day - pd.Timedelta(weeks=2):]
        model = LEARCompat(calibration_window=56)
        x_train, _, x_test = model._build_and_split_XYs(
            df_train=train, df_test=test, date_test=next_day)
        n_exog = combined.shape[1] - 1
        built_ok = (
            x_train.shape[1] == N_FEATURES_EXPECTED
            and x_test.shape[1] == N_FEATURES_EXPECTED
            and n_exog == N_EXOG_EXPECTED
            and n_features(n_exog) == N_FEATURES_EXPECTED
            and arithmetic_ok
        )
        record(
            2, f"{zone} LEAR design matrix has 319 columns", built_ok,
            f"built X is {x_train.shape[1]} wide (train) / {x_test.shape[1]} "
            f"(test); {n_exog} exogenous; "
            f"96 + 72*{n_exog} + 7 = {96 + 72 * n_exog + 7}",
        )

    if not quiet:
        _print_preflight(results)
    return results, csv_info


def _read_dataset(zone, datasets_dir, begin, end):
    from epftoolbox.data import read_data
    return read_data(
        path=datasets_dir, dataset=dataset_name(zone),
        begin_test_date=begin,
        end_test_date=end + pd.Timedelta(hours=23),
    )


def _print_preflight(results):
    print("\nPre-flight assertions")
    print("=" * 72)
    for r in sorted(results, key=lambda r: r["id"]):
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['id']}. {r['name']}")
        print(f"       {r['detail']}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Back-transformation finding
# ---------------------------------------------------------------------------

def check_back_transformation(run_dirs):
    """Verify the ensemble averages price-scale forecasts, not asinh-scale ones.

    Since ``asinh`` is nonlinear, ``mean_i sinh^-1_scale(z_i)`` and
    ``sinh^-1_scale(mean_i z_i)`` are different estimators; Lago et al. (2021)
    and the thesis report the former (a plain mean in EUR/MWh).

    The code path:

    * ``lear_dk1/compat.py`` ``LEARCompat.predict`` ends with
      ``self.scalerY.inverse_transform(Yp.reshape(1, -1))`` -- ``scalerY`` is the
      Invariant (asinh-median) scaler, so each window's daily forecast leaves
      the model already back-transformed to EUR/MWh.
    * ``lear_dk1/backtest.py`` ``run_window`` stores that array verbatim, so
      ``forecasts_cw<N>.csv`` is in EUR/MWh.
    * ``lear_dk1/evaluate.py`` ``build_ensemble`` stacks those CSVs and takes
      ``.mean(axis=0)`` -- an arithmetic mean of already-back-transformed values.

    Returns a dict with the finding and, where a run exists, a numeric check
    that the per-window files are in the price scale and that the recomputed
    mean matches ``build_ensemble``.
    """
    finding = {
        "conclusion": "averaged AFTER back-transformation to the price scale",
        "evidence": [
            "compat.py: LEARCompat.predict returns "
            "scalerY.inverse_transform(Yp) -> each window forecast is EUR/MWh",
            "backtest.py: run_window writes that array verbatim to "
            "forecasts_cw<N>.csv",
            "evaluate.py: build_ensemble takes np.stack([...]).mean(axis=0) "
            "over those price-scale files",
        ],
        "numeric_check": None,
    }
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
        forecasts, _ = load_forecasts(run_dir)
        if len(forecasts) < 1:
            continue
        ensemble = build_ensemble(forecasts)
        if ensemble is None or not len(ensemble):
            continue
        stacked = np.stack([
            forecasts[w].loc[ensemble.index, HOURS].to_numpy(float)
            for w in sorted(forecasts)
        ])
        recomputed = stacked.mean(axis=0)
        matches = bool(np.allclose(recomputed, ensemble.to_numpy(float),
                                   equal_nan=True))
        median_abs = float(np.nanmedian(np.abs(stacked)))
        finding["numeric_check"] = {
            "run_dir": os.path.basename(run_dir),
            "windows": sorted(forecasts),
            "recomputed_mean_matches_build_ensemble": matches,
            "median_abs_window_value": round(median_abs, 3),
            "comment": (
                "median |forecast| is in the tens of EUR/MWh, i.e. the price "
                "scale, not the single-digit asinh scale -- confirms the "
                "per-window files are back-transformed before averaging"
            ),
        }
        break
    return finding


# ---------------------------------------------------------------------------
# Scoring and outputs
# ---------------------------------------------------------------------------

def _score(forecast_df, real_df, naive_mae):
    """MAE, RMSE and rMAE on the hours with an observed price."""
    forecast = forecast_df.reindex(real_df.index)[HOURS].to_numpy(float)
    real = real_df[HOURS].to_numpy(float)
    mask = np.isfinite(real) & np.isfinite(forecast)
    err = forecast[mask] - real[mask]
    mae = float(np.mean(np.abs(err)))
    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "rmae": mae / naive_mae,
        "hours_scored": int(mask.sum()),
    }


def _flatten_hours(zone, frame, days):
    """A (day x 24) forecast frame -> long rows: timestamp, date, hour, value."""
    hours = np.arange(24)
    stamps = (days.to_numpy()[:, None]
              + (hours * np.timedelta64(1, "h"))[None, :]).reshape(-1)
    return pd.DataFrame({
        "timestamp_local": stamps,
        "date": np.repeat([d.date() for d in days], 24),
        "hour": np.tile(hours, len(days)),
        "zone": zone,
        "value": frame.reindex(days)[HOURS].to_numpy(float).reshape(-1),
    })


def _write_tidy(zone, forecasts, ensemble, real, path):
    """local date, hour, actual, ensemble, and each window forecast."""
    days = ensemble.index

    def flat(frame):
        return frame.reindex(days)[HOURS].to_numpy(float).reshape(-1)

    out = _flatten_hours(zone, real, days).drop(columns="value")
    out["actual"] = flat(real)
    out["ensemble"] = flat(ensemble)
    for w in sorted(forecasts):
        out[f"cw{w}"] = flat(forecasts[w])
    out.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
    return out


def score_zone(zone, run_dir, datasets_dir, summary_dir):
    """Build the ensemble, score every member, write the per-zone forecast files.

    Writes three things:

    * ``<summary_dir>/forecasts_<zone>.csv`` -- tidy, one row per hour, with the
      actual price, the ensemble forecast and every window forecast side by side;
    * ``<summary_dir>/ensemble_<zone>.csv`` -- the ensemble forecast alone, tidy,
      actual beside forecast -- the headline deliverable for the zone;
    * ``<run_dir>/forecasts_ensemble.csv`` -- the ensemble in the same wide
      ``date`` x ``h0..h23`` shape as the ``forecasts_cw<N>.csv`` members it is
      the mean of, so it sits alongside them.
    """
    forecasts, _ = load_forecasts(run_dir)
    if not forecasts:
        raise FileNotFoundError(f"no forecast files in {run_dir}")
    ensemble = build_ensemble(forecasts)
    if ensemble is None or not len(ensemble):
        raise ValueError(f"windows in {run_dir} share no complete forecast day")

    real = real_prices(dataset_name(zone), datasets_dir, ensemble.index)
    naive_mae = naive_weekly_mae(real)

    tidy_path = os.path.join(summary_dir, f"forecasts_{zone}.csv")
    _write_tidy(zone, forecasts, ensemble, real, tidy_path)

    # The ensemble in the members' own wide shape, next to forecasts_cw*.csv.
    wide_path = os.path.join(run_dir, "forecasts_ensemble.csv")
    wide = ensemble.copy()
    wide.index.name = "Date"
    wide.to_csv(wide_path)

    # The ensemble alone, tidy, actual beside forecast.
    ensemble_tidy = _flatten_hours(zone, real, ensemble.index).drop(columns="value")
    ensemble_tidy["actual"] = real.reindex(ensemble.index)[HOURS].to_numpy(
        float).reshape(-1)
    ensemble_tidy["ensemble"] = ensemble[HOURS].to_numpy(float).reshape(-1)
    ensemble_path = os.path.join(summary_dir, f"ensemble_{zone}.csv")
    ensemble_tidy.to_csv(ensemble_path, index=False,
                         date_format="%Y-%m-%dT%H:%M:%S")

    scores = {"ensemble": _score(ensemble, real, naive_mae)}
    for w in sorted(forecasts):
        scores[f"cw{w}"] = _score(forecasts[w].loc[ensemble.index], real,
                                  naive_mae)
    return {
        "zone": zone,
        "forecast_days": len(ensemble),
        "first_day": str(ensemble.index.min().date()),
        "last_day": str(ensemble.index.max().date()),
        "naive_weekly_mae": naive_mae,
        "scores": scores,
        "tidy_file": tidy_path,
        "ensemble_file": ensemble_path,
        "ensemble_tidy": ensemble_tidy,
    }


def accuracy_table(zone_results):
    rows = []
    for res in zone_results:
        for model, s in res["scores"].items():
            rows.append({
                "zone": res["zone"],
                "model": model,
                "mae": round(s["mae"], 4),
                "rmse": round(s["rmse"], 4),
                "rmae": round(s["rmae"], 5),
                "hours_scored": s["hours_scored"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

PROGRESS_INTERVAL_SECONDS = 60


class _ThrottledProgress(io.TextIOBase):
    """Pass stdout through, but thin out the backtest's per-day progress lines.

    ``lear_dk1/backtest.run_window`` prints one line per test day. Over four
    windows and two zones that is thousands of near-identical lines. This wrapper
    keeps every non-progress line verbatim (window headers, "done" lines, the
    dataset banner) and, of the per-day lines, keeps only the first and last of
    each window plus about one every ``interval`` seconds -- each kept line
    already carries the day count, running MAE and the window's ETA.
    """

    _DAY = re.compile(r"^\s*\[cw\d+\]\s+(\d+)/(\d+)\b")

    def __init__(self, target, interval=PROGRESS_INTERVAL_SECONDS):
        self._target = target
        self._interval = interval
        self._buf = ""
        self._last = 0.0

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line + "\n")
        return len(text)

    def _emit(self, line):
        match = self._DAY.match(line)
        if match is None:
            self._target.write(line)
            self._target.flush()
            return
        done, total = int(match.group(1)), int(match.group(2))
        now = time.time()
        if done <= 1 or done == total or now - self._last >= self._interval:
            self._target.write(line)
            self._target.flush()
            self._last = now

    def flush(self):
        self._target.flush()

    def drain(self):
        if self._buf:
            self._target.write(self._buf)
            self._buf = ""
        self._target.flush()


# Rough per-day fit cost in seconds, by window, for a first-pass runtime
# estimate only. Order-of-magnitude figures from earlier runs; the live
# per-window ETA the backtest prints is the number to trust.
_ROUGH_SECONDS_PER_DAY = {56: 0.6, 84: 0.9, 1092: 3.5, 1456: 4.0}


def _rough_runtime_estimate():
    per_zone = sum(_ROUGH_SECONDS_PER_DAY.get(w, 3.0) for w in WINDOWS) \
        * TEST_DAYS_EXPECTED
    return per_zone * len(ZONES)


def _measured_fit_time(run_dir):
    """(per-window seconds, total) actually spent so far, from timings_cw*.csv."""
    import glob
    per_window = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "timings_cw*.csv"))):
        try:
            t = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError):
            continue
        if not len(t):
            continue
        per_window[int(t["calibration_window"].iloc[0])] = float(t["seconds"].sum())
    return per_window, sum(per_window.values())


def run_zone(zone, panel_path, datasets_dir, out_dir, smoke, quiet):
    """Invoke the machinery for one zone with the pinned configuration."""
    argv = [
        "--panel", panel_path,
        "--zone", zone,
        "--exog", EXOG_LAYOUT,
        "--datasets-dir", datasets_dir,
        "--out-dir", out_dir,
        # Explicit test range: overrides run_lear_from_clean's 728-day derivation.
        "--begin-test", str(BEGIN_TEST.date()),
        "--end-test", str(END_TEST.date()),
        "--windows", ",".join(str(w) for w in WINDOWS),
        # Use the whole panel history; the burn-in was asserted in preflight.
        "--data-start", str(HISTORY_START_REQUIRED.date()),
    ]
    if smoke:
        argv.append("--smoke")
    if quiet:
        argv.append("--quiet")
    return run_lear_from_clean.main(argv)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Thesis LEAR benchmark: DK1 and DK2, pinned configuration.")
    parser.add_argument("--panel", default=DEFAULT_PANEL,
                        help="Cleaned hourly local panel from data_cleaning_v2.ipynb")
    parser.add_argument("--datasets-dir", default=DEFAULT_DATASETS,
                        help="Where the projected zone CSVs are written")
    parser.add_argument("--out-dir", default=DEFAULT_OUT,
                        help="Where per-zone run directories are created")
    parser.add_argument("--smoke", action="store_true",
                        help="Ten days, smallest window, both zones -- passed "
                             "straight through to the existing smoke path")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip the backtests; rebuild the summary artifacts "
                             "from existing run directories")
    parser.add_argument("--verbose", action="store_true",
                        help="Print every per-day backtest line, unthrottled")
    parser.add_argument("--quiet", action="store_true",
                        help="Silence the backtest entirely (no progress at all)")
    args = parser.parse_args(argv)
    if args.verbose and args.quiet:
        parser.error("--verbose and --quiet are mutually exclusive")

    if not os.path.exists(args.panel):
        print(f"error: no cleaned panel at {args.panel}", file=sys.stderr)
        print("Build it with data_cleaning_v2.ipynb.", file=sys.stderr)
        return 1

    summary_dir = os.path.join(args.out_dir, SUMMARY_DIRNAME)
    os.makedirs(summary_dir, exist_ok=True)

    print(f"Zones:   {', '.join(ZONES)}")
    print(f"Layout:  {EXOG_LAYOUT}  ({N_EXOG_EXPECTED} exogenous, "
          f"{N_FEATURES_EXPECTED} features)")
    print(f"Windows: {', '.join(str(w) for w in WINDOWS)}  (mean = the ensemble)")
    print(f"Test:    {BEGIN_TEST.date()} .. {END_TEST.date()}  "
          f"({TEST_DAYS_EXPECTED} days)"
          + ("   [SMOKE: 10 days, window 56]" if args.smoke else ""))

    # --- Pre-flight -----------------------------------------------------
    results, _ = preflight(args.panel, args.datasets_dir, quiet=args.quiet)
    if any(not r["passed"] for r in results):
        _print_preflight(results)
        print("\nPre-flight FAILED -- refusing to run.", file=sys.stderr)
        return 2

    # --- Backtests -----------------------------------------------------
    run_dirs = {zone: run_dir_for(zone, args.out_dir, args.smoke)
                for zone in ZONES}
    started = time.time()
    if not args.score_only:
        if not args.smoke and not args.quiet:
            print(f"\nRough total runtime estimate: ~{_fmt(_rough_runtime_estimate())} "
                  f"({len(ZONES)} zones x {len(WINDOWS)} windows x "
                  f"{TEST_DAYS_EXPECTED} days, daily recalibration).")
            print("Progress is thinned to ~1 line / "
                  f"{PROGRESS_INTERVAL_SECONDS}s per window; after each zone the "
                  "measured fit time and a projection are printed.")

        # Thin the backtest's per-day lines unless the user asked for all of them
        # (--verbose) or none (--quiet, handled by passing it straight through).
        throttle = None
        if not args.verbose and not args.quiet:
            throttle = _ThrottledProgress(sys.stdout)
            sys.stdout = throttle
        try:
            for index, zone in enumerate(ZONES, 1):
                print(f"\n{'=' * 72}\n{zone}   (zone {index}/{len(ZONES)})   "
                      f"elapsed {_fmt(time.time() - started)}\n{'=' * 72}",
                      flush=True)
                rc = run_zone(zone, args.panel, args.datasets_dir, args.out_dir,
                              args.smoke, args.quiet)
                if rc != 0:
                    print(f"error: {zone} backtest exited {rc}", file=sys.stderr)
                    return 1

                if not args.quiet:
                    per_window, total = _measured_fit_time(run_dirs[zone])
                    if per_window:
                        breakdown = ", ".join(
                            f"cw{w} {_fmt(s)}" for w, s in sorted(per_window.items()))
                        print(f"\n{zone} measured fit time: {breakdown}  "
                              f"(total {_fmt(total)})")
                        left = len(ZONES) - index
                        if left:
                            print(f"~{_fmt(total * left)} to go for the remaining "
                                  f"{left} zone(s), plus scoring.", flush=True)
        finally:
            if throttle is not None:
                throttle.drain()
                sys.stdout = throttle._target
    runtime = time.time() - started

    # --- Back-transformation finding -------------------------------
    finding = check_back_transformation(list(run_dirs.values()))
    print(f"\nEnsemble back-transformation: {finding['conclusion']}")
    if finding["numeric_check"]:
        nc = finding["numeric_check"]
        print(f"  recomputed mean == build_ensemble: "
              f"{nc['recomputed_mean_matches_build_ensemble']}; "
              f"median |window forecast| = {nc['median_abs_window_value']} EUR/MWh")

    # --- Scoring and the combined table ---------------------------
    zone_results = []
    for zone in ZONES:
        zone_results.append(
            score_zone(zone, run_dirs[zone], args.datasets_dir, summary_dir))

    table = accuracy_table(zone_results)
    table_path = os.path.join(summary_dir, "accuracy_summary.csv")
    table.to_csv(table_path, index=False)

    # One ensemble-forecast file across both zones (long format: zones cover the
    # same span here, but long keeps it consistent with the per-zone files).
    combined_ensemble = os.path.join(summary_dir, "ensemble_forecasts.csv")
    (pd.concat([res["ensemble_tidy"] for res in zone_results], ignore_index=True)
       .sort_values(["zone", "timestamp_local"])
       .to_csv(combined_ensemble, index=False, date_format="%Y-%m-%dT%H:%M:%S"))

    print(f"\nAccuracy -- ensemble and each window, both zones")
    print("=" * 72)
    print(table.to_string(index=False))
    print("=" * 72)
    for res in zone_results:
        print(f"  {res['zone']}: forecasts (all members) -> {res['tidy_file']}")
        print(f"       ensemble only              -> {res['ensemble_file']}")
        print(f"       ensemble (wide, in run dir) -> "
              f"{os.path.join(run_dirs[res['zone']], 'forecasts_ensemble.csv')}")
    print(f"  combined ensemble forecasts -> {combined_ensemble}")
    print(f"  combined accuracy table     -> {table_path}")

    # --- Manifest ------------------------------------------------
    manifest = {
        "smoke": args.smoke,
        "zones": list(ZONES),
        "exog_layout": EXOG_LAYOUT,
        "windows": list(WINDOWS),
        "begin_test": str(BEGIN_TEST.date()),
        "end_test": str(END_TEST.date()),
        "test_days_expected": TEST_DAYS_EXPECTED,
        "preflight": results,
        "back_transformation": finding,
        "runtime_seconds": round(runtime, 1),
        "runtime": _fmt(runtime),
        "run_dirs": {z: run_dirs[z] for z in ZONES},
        "scores": {res["zone"]: res["scores"] for res in zone_results},
        "outputs": {
            "accuracy_table": table_path,
            "combined_ensemble_forecasts": combined_ensemble,
            "per_zone_all_members": {r["zone"]: r["tidy_file"] for r in zone_results},
            "per_zone_ensemble": {r["zone"]: r["ensemble_file"] for r in zone_results},
        },
    }
    manifest_path = os.path.join(summary_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    print(f"\nTotal backtest runtime: {_fmt(runtime)}"
          + ("  (score-only: no backtests run)" if args.score_only else ""))
    print(f"Manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
