"""
Scoring a finished LEAR run the way Lago et al. (2021) score theirs.

Four things, in the order the paper reports them:

1. **The ensemble.** LEAR is not one model but the arithmetic mean of the
   forecasts from its four calibration windows. The individual windows are
   reported too, because the ensemble's advantage over its own members is part
   of the result.
2. **MAE**, in the price unit.
3. **rMAE**, MAE divided by the MAE of a weekly seasonal naive forecast
   (``p_hat_d = p_{d-7}``). Absolute MAE is not comparable across zones -- a
   volatile zone scores worse at equal skill -- so the relative figure is what
   cross-zone statements rest on.
4. **Diebold-Mariano and Giacomini-White tests**, on multivariate loss: one
   statistic per day over the whole 24-hour vector rather than 24 separate
   univariate tests, which is what the paper does and what keeps the number of
   comparisons honest.

Nothing here refits anything; it reads the forecasts a backtest already wrote.
"""

import contextlib
import glob
import io
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from .compat import PROJECT_ROOT

if os.path.join(PROJECT_ROOT, "epftoolbox") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "epftoolbox"))
from epftoolbox.data import read_data  # noqa: E402
from epftoolbox.evaluation import DM, GW  # noqa: E402

HOURS = [f"h{h}" for h in range(24)]

ENSEMBLE = "ensemble"


def load_forecasts(run_dir):
    """Every per-window forecast file in a run directory, keyed by window."""
    forecasts = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "forecasts_cw*.csv"))):
        match = re.search(r"forecasts_cw(\d+)\.csv$", os.path.basename(path))
        if not match:
            continue
        frame = pd.read_csv(path, index_col=0)
        frame.index = pd.to_datetime(frame.index)
        # Only complete days count: a half-written final row would otherwise
        # enter the ensemble mean as a partial forecast.
        forecasts[int(match.group(1))] = frame.dropna(how="any")
    return forecasts


def build_ensemble(forecasts):
    """Arithmetic mean across windows, over the days every window has.

    Restricting to the common index matters: averaging over whatever each window
    happens to have finished would make the ensemble a different model on
    different days.
    """
    if not forecasts:
        return None

    common = None
    for frame in forecasts.values():
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None or not len(common):
        return None

    stacked = np.stack([forecasts[w].loc[common, HOURS].to_numpy(dtype=float)
                        for w in sorted(forecasts)])
    return pd.DataFrame(stacked.mean(axis=0), index=common, columns=HOURS)


def real_prices(dataset, datasets_dir, index):
    """Observed prices for the forecast days, shaped like the forecasts."""
    # A test range wide enough to cover the forecasts; read_data needs both ends.
    begin = pd.Timestamp(index.min())
    end = pd.Timestamp(index.max()) + pd.Timedelta(hours=23)
    # read_data prints its own banner; swallow it, since scoring eleven zones would
    # otherwise print eleven of them between the caller's own output.
    with contextlib.redirect_stdout(io.StringIO()):
        _, df_test = read_data(path=datasets_dir, dataset=dataset,
                               begin_test_date=begin, end_test_date=end)

    values = df_test.loc[:, ["Price"]].to_numpy(dtype=float).reshape(-1, 24)
    frame = pd.DataFrame(values, index=df_test.index[::24], columns=HOURS)
    return frame.loc[index]


def naive_weekly_mae(real):
    """MAE of the weekly seasonal naive, ``p_hat_d = p_{d-7}``, ignoring gaps.

    This is what ``rMAE(..., m="W", freq="1h")`` computes, and it is checked
    against it in the tests. It is reimplemented here only because the upstream
    version propagates NaN: it takes a plain mean, so one missing observed price
    anywhere in the test period turns the whole denominator into NaN.

    ``real`` must be indexed by consecutive days for the 7-row offset to mean a
    week; :func:`evaluate_run` builds it that way.
    """
    values = real.to_numpy(dtype=float)
    if len(values) <= 7:
        return float("nan")
    return float(np.nanmean(np.abs(values[7:] - values[:-7])))


def score(real, predicted, naive_mae=None):
    """MAE and rMAE for one forecast, against the weekly seasonal naive.

    Hours whose observed price is missing are excluded rather than counted as
    error: there is nothing to be right or wrong about. The count is reported so
    that how much of the test period was actually scored is on the record.

    The training data is imputed (see lear_dk1.impute), but the *observed* prices
    a forecast is scored against never are -- scoring against a filled value
    would measure agreement with the filling rule, not with the market.
    """
    real_values = real.to_numpy(dtype=float)
    errors = np.abs(real_values - predicted.to_numpy(dtype=float))
    observed = np.isfinite(real_values)

    mae = float(np.nanmean(np.where(observed, errors, np.nan)))
    naive = naive_weekly_mae(real) if naive_mae is None else naive_mae
    return {
        "mae": mae,
        "rmae": mae / naive,
        "hours_scored": int(observed.sum()),
        "hours_unobserved": int((~observed).sum()),
    }


def compare(real, better, worse):
    """DM and GW p-values for the claim that ``better`` beats ``worse``.

    Both tests are one-sided, and upstream's convention is the opposite of what
    the argument names suggest: the loss differential is
    ``|p_real - p_pred_1| - |p_real - p_pred_2|`` and the p-value is the upper
    tail, so a *small* p-value rejects H0 in favour of ``p_pred_2``. The
    candidate therefore goes in ``p_pred_2`` and the baseline in ``p_pred_1``.

    The multivariate version tests one loss per day averaged across all 24
    hours, rather than running 24 separate tests whose joint size would be hard
    to interpret.

    GW needs more forecast days than hours in a day. Upstream takes the series
    length as ``np.max(d.shape)`` on a ``(n_days, 24)`` array, which is 24 rather
    than ``n_days`` whenever fewer than 24 days are being compared, and the
    regression it builds then has mismatched dimensions. That only bites on short
    trial runs -- the paper's test period is 728 days -- and it surfaces below as
    a recorded error rather than a crash.
    """
    real_values = real.to_numpy(dtype=float)
    out = {}
    for name, test in (("dm", DM), ("gw", GW)):
        try:
            # Identical forecasts give a loss differential of exactly zero, so the
            # test statistic is 0/0. That surfaces as a NaN and a RuntimeWarning
            # rather than an exception, so it has to be caught by value below.
            with np.errstate(invalid="ignore", divide="ignore"):
                value = float(test(
                    p_real=real_values,
                    p_pred_1=worse.to_numpy(dtype=float),
                    p_pred_2=better.to_numpy(dtype=float),
                    norm=1, version="multivariate",
                ))
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as exc:
            # Reported rather than raised: a degenerate comparison (identical
            # forecasts, or too few days) should not lose the rest of the table.
            out[name] = None
            out[f"{name}_error"] = f"{type(exc).__name__}: {exc}"[:120]
            continue

        if not np.isfinite(value):
            # NaN is not valid JSON, and json.dump writes it anyway, so a NaN left
            # here would produce an evaluation.json that nothing can read back.
            out[name] = None
            out[f"{name}_error"] = "degenerate: zero loss differential"
        else:
            out[name] = value
    return out


def evaluate_run(run_dir, dataset=None, datasets_dir=None, quiet=False):
    """Score one run directory. Returns a dict; also writes ``evaluation.json``."""
    manifest_path = os.path.join(run_dir, "run_metadata.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)

    dataset = dataset or manifest.get("dataset")
    datasets_dir = datasets_dir or os.path.join(PROJECT_ROOT, "datasets")
    if not dataset:
        raise ValueError(
            f"No dataset name in {manifest_path} and none given, so the observed "
            f"prices cannot be found. Pass dataset=..."
        )

    forecasts = load_forecasts(run_dir)
    if not forecasts:
        raise ValueError(f"No forecast files in {run_dir}")

    ensemble = build_ensemble(forecasts)
    if ensemble is None or not len(ensemble):
        raise ValueError(
            f"The windows in {run_dir} share no complete forecast day, so no "
            f"ensemble can be formed. Windows hold: "
            + ", ".join(f"cw{w}={len(f)}" for w, f in sorted(forecasts.items()))
        )

    index = ensemble.index
    real = real_prices(dataset, datasets_dir, index)

    # The naive denominator is computed once from the observed prices, so every
    # forecast in this run is divided by the same number.
    naive_mae = naive_weekly_mae(real)

    # DM and GW take no NaN, and unlike the point metrics they cannot simply skip
    # an hour: the loss differential is a per-day average over all 24. Days with
    # an unobserved price are therefore dropped from the tests, which is safe
    # because neither test uses a lagged value of the series.
    complete = real.notna().all(axis=1)

    results = {
        "run_dir": run_dir,
        "dataset": dataset,
        "forecast_days": len(index),
        "first_day": str(index.min().date()),
        "last_day": str(index.max().date()),
        "windows": sorted(forecasts),
        "unobserved_hours": int(real.isna().to_numpy().sum()),
        "days_tested": int(complete.sum()),
        "naive_weekly_mae": naive_mae,
        "scores": {},
        "tests": {},
    }

    results["scores"][ENSEMBLE] = score(real, ensemble, naive_mae)
    for window, frame in sorted(forecasts.items()):
        # Score every window on the ensemble's days, so the comparison is like
        # for like even when one window ran further than another.
        results["scores"][f"cw{window}"] = score(real, frame.loc[index, HOURS], naive_mae)

    # The claim worth testing is that the ensemble beats its own members. With a
    # single window there is no such claim: the ensemble *is* that window, the loss
    # differential is identically zero, and the test statistic is 0/0.
    if len(forecasts) > 1 and complete.any():
        days = index[complete.to_numpy()]
        for window, frame in sorted(forecasts.items()):
            results["tests"][f"{ENSEMBLE}_vs_cw{window}"] = compare(
                real.loc[days], ensemble.loc[days], frame.loc[days, HOURS])

    with open(os.path.join(run_dir, "evaluation.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    if not quiet:
        print(format_run(results))

    return results


def format_run(results):
    """A readable table for one run."""
    lines = [
        f"{results['dataset']}: {results['forecast_days']} days, "
        f"{results['first_day']} to {results['last_day']}",
    ]
    if results.get("unobserved_hours"):
        lines.append(f"  {results['unobserved_hours']} hour(s) have no observed "
                     f"price and are excluded; {results['days_tested']} complete "
                     f"day(s) enter the DM/GW tests")
    lines.append(f"  {'forecast':>10}  {'MAE':>8}  {'rMAE':>7}")
    for name, values in results["scores"].items():
        lines.append(f"  {name:>10}  {values['mae']:8.3f}  {values['rmae']:7.4f}")

    if results["tests"]:
        lines.append(f"  {'comparison':>22}  {'DM p':>8}  {'GW p':>8}")
        for name, values in results["tests"].items():
            dm = "n/a" if values.get("dm") is None else f"{values['dm']:.4f}"
            gw = "n/a" if values.get("gw") is None else f"{values['gw']:.4f}"
            lines.append(f"  {name:>22}  {dm:>8}  {gw:>8}")
    return "\n".join(lines)


def compare_zones(results_by_zone):
    """One row per zone, for the cross-zone table.

    rMAE is the column to read across zones. MAE is in the local price unit and
    scales with the zone's own volatility, so a quiet zone looks skilful on MAE
    at equal forecasting quality.
    """
    rows = []
    for zone, results in sorted(results_by_zone.items()):
        row = {"zone": zone, "days": results["forecast_days"],
               "unobserved_hours": results.get("unobserved_hours", 0)}
        for name, values in results["scores"].items():
            row[f"mae_{name}"] = round(values["mae"], 4)
            row[f"rmae_{name}"] = round(values["rmae"], 5)
        rows.append(row)
    return pd.DataFrame(rows).set_index("zone")
