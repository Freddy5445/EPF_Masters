"""
The four phase-2 pre-flight checks. Nothing is launched from here.

    python run_dnn_preflight.py

===  ========================================================================
 A   the search-window gate fires when handed a window touching the test period
 B   determinism -- the same day forecasts bit-identically in different orders
 C   round-trip -- PerZoneScaler.inverse_transform(fit_transform(Y)) recovers Y
 D   all ten configurations smoke -- 5 hyperopt evaluations, 3 days, 1 seed
===  ========================================================================

D is the substantial one: seven of the ten configurations have never executed at
all. Each must start, write a per-day checkpoint, write a timing row, and be
scored by ``lear_dk1.evaluate.evaluate_run`` without error.

Everything is written under ``experiments/preflight/`` -- a smoke run must not
land in the directory the real runs will resume from, or a three-day checkpoint
would be mistaken for progress on a 731-day backtest.

Results go to ``experiments/preflight/preflight.json``, which
``run_dnn_all.py`` reads and refuses to launch without.

Commands are given on one line: this project is developed from PowerShell, where
a trailing ``\`` is not a line continuation and truncates the command.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import timedelta

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(THIS_DIR, "experiments", "preflight")

SMOKE_EVALS = 5
SMOKE_DAYS = 3
SMOKE_SEED = 1


def _fmt(seconds):
    return str(timedelta(seconds=int(round(seconds))))


# ---------------------------------------------------------------------------
# A. The search-window gate
# ---------------------------------------------------------------------------

def check_gate():
    """The gate must accept the real search window and refuse every overlap.

    Three overlapping windows are tried, because they fail differently and only
    the first is obvious:

    * the whole test period -- what a copy-paste of the run's own dates gives;
    * a legitimate start with the test period's end -- the plausible slip, since
      the window still *looks* like a pre-test year;
    * a window ending exactly on the first test day -- the off-by-one.
    """
    import run_dnn_dk1 as R
    from dnn_dk1 import hyperopt as H
    from dnn_dk1 import zones as Z

    begin, end = R.hyperopt_window(Z.BEGIN_TEST)
    details = []

    try:
        H.assert_search_window_precedes_test(begin, end, "preflight")
        details.append(f"accepts the real window {begin.date()}..{end}")
        accepted = True
    except H.SearchWindowError as exc:
        details.append(f"REJECTED the legitimate window: {exc}")
        accepted = False

    overlapping = {
        "the test period itself": (Z.BEGIN_TEST, Z.END_TEST),
        "a valid start with the test end": (begin, Z.END_TEST),
        "ending on the first test day": (begin, Z.BEGIN_TEST),
    }
    refused = {}
    for name, (b, e) in overlapping.items():
        try:
            H.assert_search_window_precedes_test(b, e, "preflight")
            refused[name] = False
        except H.SearchWindowError:
            refused[name] = True
    details.append(f"refuses {sum(refused.values())}/{len(refused)} overlapping "
                   f"windows: {refused}")

    # The optimisers must call it, not merely have it available.
    import inspect

    wired = {
        "optimize": "assert_search_window_precedes_test" in
                    inspect.getsource(H.optimize),
        "optimize_multizone": "assert_search_window_precedes_test" in
                              inspect.getsource(H.optimize_multizone),
    }
    details.append(f"called by {wired}")

    passed = accepted and all(refused.values()) and all(wired.values())
    return passed, "; ".join(details)


# ---------------------------------------------------------------------------
# B. Determinism
# ---------------------------------------------------------------------------

def check_determinism(out_dir, hyper_dir, datasets_dir, configs=("own", "joint")):
    """The same day must forecast identically regardless of what ran before it.

    ``_build_and_split_XYs`` draws the train/validation split with

        if hyperoptimization:
            np.random.seed(7)
        np.random.shuffle(index_week)

    so during a backtest the split comes from the *unseeded global* RNG. Left
    alone, which validation days a model sees would depend on how many random
    draws happened earlier in the process -- the same day would forecast
    differently depending on what ran before it, no run could be reproduced, and
    a resumed run would not agree with the uninterrupted one it continued.
    Seeding from (seed, day) is what fixes it; this is the check that it did.

    Two fresh processes: one forecasts the target day alone, the other forecasts
    the day before it first. Bit-identical, not almost.
    """
    details, passed = [], True
    for config in configs:
        alone = _forecast_days(out_dir, hyper_dir, datasets_dir, config,
                               days=1, tag=f"det_{config}_alone")
        after = _forecast_days(out_dir, hyper_dir, datasets_dir, config,
                               days=2, tag=f"det_{config}_after")
        # The target day is the last of each run.
        a = alone["forecasts"][-1]
        b = after["forecasts"][-1]
        identical = a == b
        passed = passed and identical
        details.append(
            f"{config}: {'bit-identical' if identical else 'DIFFERS'} "
            f"({len(a)} values, max |diff| "
            f"{max((abs(x - y) for x, y in zip(a, b)), default=0.0):.3e})")
    return passed, "; ".join(details)


def _forecast_days(out_dir, hyper_dir, datasets_dir, config, days, tag):
    """Forecast ``days`` consecutive days in a fresh process; return the values."""
    result_path = os.path.join(out_dir, f"determinism_{tag}.json")
    command = [sys.executable, os.path.abspath(__file__), "--forecast-worker",
               "--config", config, "--days", str(days),
               "--hyperparameter-dir", hyper_dir,
               "--datasets-dir", datasets_dir, "--result", result_path]
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "4", "TF_NUM_INTRAOP_THREADS": "4",
                "TF_NUM_INTEROP_THREADS": "4", "TF_CPP_MIN_LOG_LEVEL": "3",
                "KERAS_BACKEND": "tensorflow", "PYTHONWARNINGS": "ignore"})
    completed = subprocess.run(command, cwd=THIS_DIR, env=env,
                               capture_output=True, text=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout[-3000:] + completed.stderr[-3000:])
        raise RuntimeError(f"determinism worker {tag} failed")
    with open(result_path, encoding="utf-8") as handle:
        return json.load(handle)


def forecast_worker(args):
    """Forecast ``--days`` consecutive days, the last of them BEGIN_TEST.

    The target day is fixed; only how much ran before it in this process varies.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import numpy as np
    import pandas as pd

    import run_dnn_dk1 as R
    from dnn_dk1 import zones as Z

    zone = "DK1"
    models = R.build_models(args.config, zone, [SMOKE_SEED], args.hyperparameter_dir,
                            2, Z.CALIBRATION_YEARS)
    model = models[SMOKE_SEED]

    # The target day is always the same calendar day; only how many days are
    # forecast before it in this process differs.
    last = Z.BEGIN_TEST
    first = last - pd.Timedelta(days=args.days - 1)
    days = pd.date_range(first, last, freq="D")

    if args.config == "own":
        import io as _io
        from contextlib import redirect_stdout

        from epftoolbox.data import read_data

        name = R.dataset_name(args.config, zone)
        with redirect_stdout(_io.StringIO()):
            df_train, df_test = read_data(
                path=args.datasets_dir, dataset=name, begin_test_date=first,
                end_test_date=last + pd.Timedelta(hours=23))
        source = pd.concat([df_train, df_test])
    else:
        source = Z.load_zone_matrices(Z.ZONES, args.datasets_dir)

    forecasts = []
    for day in days:
        prediction = model.recalibrate_and_forecast_next_day(source, day)
        # Full precision, not a rounded value: this check is about bits.
        forecasts.append(
            [float(v) for v in np.asarray(prediction, dtype=float).reshape(-1)])

    with open(args.result, "w", encoding="utf-8") as handle:
        json.dump({"config": args.config, "days": [str(d.date()) for d in days],
                   "forecasts": forecasts}, handle)
    return 0


# ---------------------------------------------------------------------------
# C. Scaler round-trip
# ---------------------------------------------------------------------------

def check_round_trip(datasets_dir):
    """``inverse_transform(fit_transform(Y))`` must give back exactly ``Y``.

    Also records the two facts the launch manifest carries: whether a pooled fit
    equals the per-zone one (it does -- every epftoolbox scaler fits per column),
    and the per-zone dispersion of the transformed targets, which is what makes
    "the 168 outputs are weighted equally" mean anything.
    """
    import numpy as np

    from dnn_dk1 import zones as Z

    matrices = Z.load_zone_matrices(Z.ZONES, datasets_dir)
    days = Z.available_days(matrices)
    days = days[(days >= Z.HISTORY_START_REQUIRED) & (days < Z.BEGIN_TEST)]
    Y = Z.build_Y(matrices, days, Z.ZONES)

    report = {}
    worst = 0.0
    for normalize in ("Invariant", "Median", "Std", "Norm", "Norm1"):
        scaler = Z.PerZoneScaler(normalize, Z.ZONES)
        transformed = scaler.fit_transform(Y)
        error = float(np.max(np.abs(scaler.inverse_transform(transformed) - Y)))
        worst = max(worst, error)
        report[normalize] = {
            "max_abs_round_trip_error": error,
            "equals_pooled_fit": scaler.equals_pooled_fit(Y),
        }
        if normalize == "Invariant":
            report[normalize]["transformed_dispersion"] = {
                z: round(v, 4) for z, v in scaler.dispersion(transformed).items()}

    invariant = report["Invariant"]
    dispersion = invariant["transformed_dispersion"]
    spread = max(dispersion.values()) / min(dispersion.values())
    passed = worst < 1e-6 and all(r["equals_pooled_fit"] for r in report.values())
    detail = (f"max round-trip error {worst:.2e} over "
              f"{len(days)} days x 168 columns, all five scaleY options; "
              f"equals_pooled_fit True for all; Invariant dispersion "
              f"{dispersion}, max/min {spread:.2f}")
    return passed, detail, report


# ---------------------------------------------------------------------------
# D. Smoke all ten
# ---------------------------------------------------------------------------

def smoke_run(run, out_dir, datasets_dir, timeout=3600):
    """One configuration, 5 hyperopt evaluations, 3 days, 1 seed.

    Checks the four things section 3 asks for -- it starts, it checkpoints per
    day, it writes a timing row, and the shared evaluator scores it -- by looking
    at the artifacts on disk rather than at the exit code, because a run that
    exits 0 having silently written nothing is exactly the failure worth
    catching.
    """
    import pandas as pd

    import run_dnn_dk1 as R
    from dnn_dk1 import runs as RS

    command = [
        sys.executable, os.path.join(THIS_DIR, "run_dnn_dk1.py"),
        "--config", run.config, "--zone", run.focal,
        "--datasets-dir", datasets_dir, "--out-dir", out_dir,
        "--max-evals", str(SMOKE_EVALS), "--seeds", str(SMOKE_SEED),
        "--nlayers", str(RS.NLAYERS), "--smoke",
    ]
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": str(run.threads),
                "TF_NUM_INTRAOP_THREADS": str(run.threads),
                "TF_NUM_INTEROP_THREADS": str(run.threads),
                "TF_CPP_MIN_LOG_LEVEL": "3", "KERAS_BACKEND": "tensorflow",
                "PYTHONWARNINGS": "ignore"})

    started = time.time()
    log_path = os.path.join(out_dir, "logs", f"smoke_{run.run_id}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    completed = subprocess.run(command, cwd=THIS_DIR, env=env,
                               capture_output=True, text=True, timeout=timeout)
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(completed.stdout + "\n--- stderr ---\n" + completed.stderr)
    elapsed = time.time() - started

    begin = pd.Timestamp(RS.BEGIN_TEST)
    end = begin + pd.Timedelta(days=SMOKE_DAYS - 1)
    run_dir = R.run_dir_for(run.config, run.focal, begin, end, out_dir, smoke=True)

    prefix = ("forecasts_joint_seed" if run.config in RS.JOINT_CONFIGS
              else "forecasts_seed")
    forecast_path = os.path.join(run_dir, f"{prefix}{SMOKE_SEED}.csv")
    timing_path = os.path.join(run_dir, f"timings_seed{SMOKE_SEED}.csv")

    checkpoint_days = 0
    if os.path.exists(forecast_path):
        frame = pd.read_csv(forecast_path, index_col=0).dropna(how="any")
        checkpoint_days = len(frame)
    timing_rows = 0
    if os.path.exists(timing_path):
        timing_rows = len(pd.read_csv(timing_path))

    # evaluate_run writes predictions.csv and evaluation.json; joint is scored
    # one zone at a time into zone_<Z>/ subdirectories, never pooled.
    scored = sorted(
        os.path.basename(os.path.dirname(p))
        for p in glob.glob(os.path.join(run_dir, "**", "predictions.csv"),
                           recursive=True))
    expected_scored = len(run.out_zones)

    passed = (completed.returncode == 0
              and checkpoint_days == SMOKE_DAYS
              and timing_rows == SMOKE_DAYS
              and len(scored) == expected_scored)

    return {
        "run_id": run.run_id, "config": run.config, "zone": run.zone,
        "n_inputs": run.n_inputs, "n_outputs": run.n_outputs,
        "threads": run.threads, "exit_code": completed.returncode,
        "seconds": round(elapsed, 1), "run_dir": run_dir,
        "checkpoint_days": checkpoint_days, "timing_rows": timing_rows,
        "zones_scored": len(scored), "zones_expected": expected_scored,
        "log": log_path, "passed": bool(passed),
        "stderr_tail": completed.stderr[-400:] if completed.returncode else "",
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 2 pre-flight. Launches nothing.")
    parser.add_argument("--forecast-worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--config", default="own")
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--result", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--datasets-dir",
                        default=os.path.join(THIS_DIR, "datasets"))
    parser.add_argument("--hyperparameter-dir", default=None)
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Run A, B and C only. B needs a trials file, so "
                             "this works only after a previous full pre-flight.")
    parser.add_argument("--only", default=None,
                        help="Comma-separated run IDs to smoke (default: all ten)")
    args = parser.parse_args(argv)

    if args.forecast_worker:
        return forecast_worker(args)

    os.makedirs(args.out_dir, exist_ok=True)
    hyper_dir = args.hyperparameter_dir or os.path.join(
        args.out_dir, "hyperparameters")
    os.makedirs(hyper_dir, exist_ok=True)

    from dnn_dk1 import runs as RS

    started = time.time()
    results = {}
    print("Phase 2 pre-flight -- nothing is launched from here.\n")

    # --- A -------------------------------------------------------------
    print("A. search-window gate ... ", end="", flush=True)
    passed_a, detail_a = check_gate()
    results["A_search_window_gate"] = {"passed": passed_a, "detail": detail_a}
    print("PASS" if passed_a else "FAIL")

    # --- C (before B: it is instant and B needs the smoke's trials file) --
    print("C. scaler round-trip ... ", end="", flush=True)
    passed_c, detail_c, scaler_report = check_round_trip(args.datasets_dir)
    results["C_scaler_round_trip"] = {"passed": passed_c, "detail": detail_c,
                                      "report": scaler_report}
    print("PASS" if passed_c else "FAIL")

    # --- D -------------------------------------------------------------
    smoke = []
    if not args.skip_smoke:
        wanted = ([RS.get(r.strip()) for r in args.only.split(",")]
                  if args.only else list(RS.RUNS))
        print(f"\nD. smoking {len(wanted)} configuration(s): "
              f"{SMOKE_EVALS} hyperopt evaluations, {SMOKE_DAYS} days, 1 seed")
        print("-" * 88)
        print(f"  {'run':<10} {'in':>5} {'out':>4} {'thr':>3}  {'exit':>4} "
              f"{'days':>4} {'timing':>6} {'scored':>6}  {'time':>7}  result")
        for run in wanted:
            outcome = smoke_run(run, args.out_dir, args.datasets_dir)
            smoke.append(outcome)
            print(f"  {outcome['run_id']:<10} {outcome['n_inputs']:>5} "
                  f"{outcome['n_outputs']:>4} {outcome['threads']:>3}  "
                  f"{outcome['exit_code']:>4} "
                  f"{outcome['checkpoint_days']:>2}/{SMOKE_DAYS} "
                  f"{outcome['timing_rows']:>4}/{SMOKE_DAYS} "
                  f"{outcome['zones_scored']:>3}/{outcome['zones_expected']:<2} "
                  f"{_fmt(outcome['seconds']):>9}  "
                  f"{'PASS' if outcome['passed'] else 'FAIL'}")
            if not outcome["passed"] and outcome["stderr_tail"]:
                print(f"      {outcome['stderr_tail'].strip()[:300]}")
        results["D_smoke_all_ten"] = {
            "passed": bool(smoke) and all(s["passed"] for s in smoke),
            "detail": f"{sum(s['passed'] for s in smoke)}/{len(smoke)} "
                      f"configurations started, checkpointed, timed and scored",
            "runs": smoke,
        }

    # --- B -------------------------------------------------------------
    passed_b, detail_b = True, "skipped"
    if not args.skip_smoke or os.path.exists(hyper_dir):
        print("\nB. determinism ... ", end="", flush=True)
        try:
            passed_b, detail_b = check_determinism(
                args.out_dir, hyper_dir, args.datasets_dir)
        except (RuntimeError, FileNotFoundError) as exc:
            passed_b, detail_b = False, str(exc)
        results["B_determinism"] = {"passed": passed_b, "detail": detail_b}
        print("PASS" if passed_b else "FAIL")

    # --- Report --------------------------------------------------------
    order = ["A_search_window_gate", "B_determinism", "C_scaler_round_trip",
             "D_smoke_all_ten"]
    print("\nPre-flight")
    print("=" * 88)
    for key in order:
        if key not in results:
            continue
        entry = results[key]
        print(f"[{'PASS' if entry['passed'] else 'FAIL'}] {key}")
        print(f"       {entry['detail']}")
    print("=" * 88)

    everything = all(entry["passed"] for entry in results.values())
    payload = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "passed": everything,
        "settled": RS.SETTLED,
        "checks": results,
        "seconds": round(time.time() - started, 1),
    }
    path = os.path.join(args.out_dir, "preflight.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print(f"\nwritten: {path}")
    print(f"pre-flight runtime: {_fmt(time.time() - started)}")
    print("\nALL CHECKS PASSED -- run_dnn_all.py --dry-run to see the commands."
          if everything else "\nPRE-FLIGHT FAILED -- do not launch.")
    return 0 if everything else 1


if __name__ == "__main__":
    sys.exit(main())
