"""
Launch the ten phase-2 runs as detached processes, and resume them if re-run.

    python run_dnn_all.py --dry-run     # print the ten command lines, run nothing
    python run_dnn_all.py               # launch

Each run becomes an **independent** OS process with its own log under
``experiments/logs/<run_id>.log``. Closing the terminal, ending the agent
session, or killing this launcher does not kill them: on Windows they are started
with ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP``, so they belong to no
console and inherit no job. One run crashing has no effect on the other nine.

Order is ``joint`` first, then the two ``wide``, then the seven ``own``.
``joint`` is the critical path -- roughly 21 hours against 5 for an ``own`` run --
so it starts first and is never left waiting behind cheaper work.

**Re-running this is safe and is the intended way to recover.** It resumes rather
than restarts: a run whose forecasts are complete is skipped, a partial run picks
up from its own per-day checkpoint through the same ``_load_checkpoint`` path
LEAR uses, and a run already alive is left alone. No finished day is ever
discarded.

Progress is visible at any moment with ``python dnn_status.py``, which reads the
checkpoints these runs are already writing. A heartbeat process started alongside
them appends one row per run per hour to
``experiments/dnn_phase2/progress_log.csv``, so a run that *slows down* shows up
as a trend rather than merely as a worse estimate.

Commands are given on one line: this project is developed from PowerShell, where
a trailing ``\`` is not a line continuation and truncates the command.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(THIS_DIR, "experiments")
DEFAULT_DATASETS = os.path.join(THIS_DIR, "datasets")
PHASE2_DIR = os.path.join(DEFAULT_OUT, "dnn_phase2")
LOG_DIR = os.path.join(DEFAULT_OUT, "logs")
PREFLIGHT_JSON = os.path.join(DEFAULT_OUT, "preflight", "preflight.json")
MANIFEST = os.path.join(PHASE2_DIR, "launch_manifest.json")

HEARTBEAT_INTERVAL_SECONDS = 3600


def run_command(run, out_dir, datasets_dir, python=None):
    """The exact command line for one run.

    Every parameter is taken from :mod:`dnn_dk1.runs` rather than typed here, so
    a run cannot silently differ from the settled configuration or from what the
    pre-flight smoked.
    """
    from dnn_dk1 import runs as RS

    return [
        python or sys.executable,
        os.path.join(THIS_DIR, "run_dnn_dk1.py"),
        "--config", run.config,
        "--zone", run.focal,
        "--datasets-dir", datasets_dir,
        "--out-dir", out_dir,
        "--max-evals", str(RS.MAX_EVALS),
        "--seeds", ",".join(str(s) for s in RS.SEEDS),
        "--nlayers", str(RS.NLAYERS),
        "--calibration-years", str(RS.CALIBRATION_YEARS),
        "--begin-test", str(RS.BEGIN_TEST.date()),
        "--end-test", str(RS.END_TEST.date()),
    ]


def run_env(run):
    """Thread pinning, set before the child's Python starts.

    TensorFlow reads these when it initialises. Setting them after
    ``import tensorflow`` does nothing at all, which is why they are put in the
    environment here rather than in the run script.
    """
    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": str(run.threads),
        "TF_NUM_INTRAOP_THREADS": str(run.threads),
        "TF_NUM_INTEROP_THREADS": str(run.threads),
        "MKL_NUM_THREADS": str(run.threads),
        "OPENBLAS_NUM_THREADS": str(run.threads),
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "KERAS_BACKEND": "tensorflow",
        "PYTHONUNBUFFERED": "1",
    })
    return env


def detach_flags():
    """Process-creation flags that outlive this launcher.

    On Windows a child started normally shares the parent's console and dies
    with it -- which would make a 21-hour run hostage to the terminal it was
    started from. ``DETACHED_PROCESS`` gives it no console at all and
    ``CREATE_NEW_PROCESS_GROUP`` keeps a Ctrl-C here from reaching it. Elsewhere,
    ``start_new_session`` does the same job.
    """
    if os.name == "nt":
        detached = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        return {"creationflags": detached | new_group}
    return {"start_new_session": True}


def spawn(run, out_dir, datasets_dir, log_dir):
    """Start one run detached, appending to its own log. Returns the PID."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run.run_id}.log")

    # Append, never truncate: a resumed run's log should read as one history.
    handle = open(log_path, "a", encoding="utf-8", buffering=1)
    handle.write(f"\n{'=' * 78}\n{time.strftime('%Y-%m-%dT%H:%M:%S')}  "
                 f"start {run.run_id} ({run.label}, {run.n_inputs} in -> "
                 f"{run.n_outputs} out, {run.threads} thread(s))\n{'=' * 78}\n")
    handle.flush()

    process = subprocess.Popen(
        run_command(run, out_dir, datasets_dir), cwd=THIS_DIR, env=run_env(run),
        stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        close_fds=True, **detach_flags())
    handle.close()
    return process.pid, log_path


def spawn_heartbeat(interval, log_dir):
    """Start the hourly reporter, detached and independent of the backtests.

    It only reads what the runs already write, so if it dies the runs continue
    and restarting it disturbs nothing.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "heartbeat.log")
    handle = open(log_path, "a", encoding="utf-8", buffering=1)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [sys.executable, os.path.join(THIS_DIR, "dnn_status.py"),
         "--heartbeat", "--interval", str(interval)],
        cwd=THIS_DIR, env=env, stdout=handle, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, close_fds=True, **detach_flags())
    handle.close()
    return process.pid, log_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Launch the ten phase-2 DNN runs as detached processes.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print each command line and launch nothing")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--datasets-dir", default=DEFAULT_DATASETS)
    parser.add_argument("--log-dir", default=LOG_DIR)
    parser.add_argument("--only", default=None,
                        help="Comma-separated run IDs (default: all ten)")
    parser.add_argument("--no-heartbeat", action="store_true",
                        help="Do not start the hourly progress reporter")
    parser.add_argument("--skip-preflight-check", action="store_true",
                        help="Launch without a passing preflight.json. The "
                             "pre-flight is the only thing standing between a "
                             "typo and a week of invalid runs; do not use this.")
    parser.add_argument("--stagger", type=float, default=20.0,
                        help="Seconds between launches, so ten TensorFlow "
                             "imports do not contend at once (default: 20)")
    args = parser.parse_args(argv)

    import dnn_status
    from dnn_dk1 import runs as RS

    runs = ([RS.get(r.strip()) for r in args.only.split(",")]
            if args.only else list(RS.RUNS))

    # --- the pre-flight is a precondition, not a suggestion -------------
    preflight = None
    if os.path.exists(PREFLIGHT_JSON):
        with open(PREFLIGHT_JSON, encoding="utf-8") as handle:
            preflight = json.load(handle)
    if not args.dry_run and not args.skip_preflight_check:
        if preflight is None:
            print(f"error: no pre-flight at {PREFLIGHT_JSON}. Run "
                  f"`python run_dnn_preflight.py` first -- seven of these ten "
                  f"configurations have never executed.", file=sys.stderr)
            return 1
        if not preflight.get("passed"):
            failed = [k for k, v in preflight.get("checks", {}).items()
                      if not v.get("passed")]
            print(f"error: the pre-flight did not pass ({', '.join(failed)}). "
                  f"Refusing to launch.", file=sys.stderr)
            return 1

    os.makedirs(PHASE2_DIR, exist_ok=True)

    # --- what is already done or already running ------------------------
    states = dnn_status.collect(out_dir=args.out_dir, log_dir=args.log_dir)
    by_id = {s["run_id"]: s for s in states}

    print(f"{'run':<10} {'config':<6} {'in':>5} {'out':>4} {'thr':>3}  "
          f"{'state':<9} {'days':>9}  action")
    print("-" * 78)

    launched, skipped = [], []
    for run in runs:
        state = by_id.get(run.run_id, {})
        status = state.get("state", "queued")
        days = f"{state.get('days_done', 0)}/{RS.TEST_DAYS}"

        if status == "done":
            action = "skip -- complete"
        elif status in ("running", "stalled"):
            action = f"skip -- alive (pid {state.get('pid')})"
        elif args.dry_run:
            action = "would launch" + (
                f", resuming {state.get('days_done', 0)} day(s)"
                if state.get("days_done") else "")
        else:
            action = "launch" + (f", resuming {state.get('days_done')} day(s)"
                                 if state.get("days_done") else "")

        print(f"{run.run_id:<10} {run.config:<6} {run.n_inputs:>5} "
              f"{run.n_outputs:>4} {run.threads:>3}  {status:<9} {days:>9}  "
              f"{action}")

        if status in ("done", "running", "stalled"):
            skipped.append(run)
        else:
            launched.append(run)

    if args.dry_run:
        print(f"\nCommand lines ({len(launched)} would launch; "
              f"{len(skipped)} skipped):")
        print("=" * 78)
        for run in launched:
            env = {k: str(run.threads) for k in
                   ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
                    "TF_NUM_INTEROP_THREADS")}
            command = run_command(run, args.out_dir, args.datasets_dir)
            print(f"\n# {run.run_id} -- {run.label}, {run.n_inputs} in -> "
                  f"{run.n_outputs} out")
            print("  env: " + " ".join(f"{k}={v}" for k, v in env.items()))
            print("  " + subprocess.list2cmdline(command))
            print(f"  log: {os.path.join(args.log_dir, run.run_id + '.log')}")
        print("\n" + "=" * 78)
        print("dry run -- nothing was launched.")
        return 0

    # --- launch ---------------------------------------------------------
    records = []
    for n, run in enumerate(launched):
        pid, log_path = spawn(run, args.out_dir, args.datasets_dir, args.log_dir)
        records.append({
            "run_id": run.run_id, "config": run.config, "zone": run.zone,
            "n_inputs": run.n_inputs, "n_outputs": run.n_outputs,
            "threads": run.threads, "pid": pid, "log": log_path,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "command": run_command(run, args.out_dir, args.datasets_dir),
            "resumed_from_days": by_id.get(run.run_id, {}).get("days_done", 0),
        })
        print(f"  started {run.run_id:<10} pid {pid:<8} -> {log_path}")
        # Ten simultaneous TensorFlow imports thrash the disk and each other;
        # the runs are hours long, so a short stagger costs nothing.
        if args.stagger and n < len(launched) - 1:
            time.sleep(args.stagger)

    heartbeat = None
    if not args.no_heartbeat:
        pid, log_path = spawn_heartbeat(HEARTBEAT_INTERVAL_SECONDS, args.log_dir)
        heartbeat = {"pid": pid, "log": log_path,
                     "interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
                     "csv": os.path.join(PHASE2_DIR, "progress_log.csv")}
        print(f"  started heartbeat pid {pid} -> {heartbeat['csv']} "
              f"(hourly)")

    manifest = _merge_manifest({
        "launched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settled": RS.SETTLED,
        "total_threads": sum(r.threads for r in RS.RUNS),
        "scaler_checks": _scaler_checks(preflight),
        "preflight": {
            "path": PREFLIGHT_JSON,
            "passed": bool(preflight and preflight.get("passed")),
            "checked_at": (preflight or {}).get("checked_at"),
        },
        "runs": records,
        "skipped": [{"run_id": r.run_id,
                     "state": by_id.get(r.run_id, {}).get("state"),
                     "days_done": by_id.get(r.run_id, {}).get("days_done", 0)}
                    for r in skipped],
        "heartbeat": heartbeat,
    })
    with open(MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    print(f"\nmanifest -> {MANIFEST}")
    print(f"progress  -> python dnn_status.py  (or --watch, or --json)")
    print(f"These processes are detached: closing this terminal will not stop "
          f"them.")
    return 0


def _scaler_checks(preflight):
    """``equals_pooled_fit`` and ``transformed_dispersion``, for the manifest.

    Recorded at launch because they are properties of the *data* the runs are
    about to train on. ``equals_pooled_fit`` being true is the phase 1 finding
    that epftoolbox's scalers already fit per column, so a pooled fit and a
    per-zone one coincide; ``transformed_dispersion`` is what makes weighting the
    168 outputs equally mean what it says.
    """
    checks = ((preflight or {}).get("checks", {})
              .get("C_scaler_round_trip", {}).get("report", {}))
    if not checks:
        return None
    invariant = checks.get("Invariant", {})
    dispersion = invariant.get("transformed_dispersion") or {}
    return {
        "equals_pooled_fit": {k: v.get("equals_pooled_fit")
                              for k, v in checks.items()},
        "transformed_dispersion": dispersion,
        "transformed_dispersion_max_over_min": (
            round(max(dispersion.values()) / min(dispersion.values()), 4)
            if dispersion else None),
        "max_abs_round_trip_error": {k: v.get("max_abs_round_trip_error")
                                     for k, v in checks.items()},
        "source": PREFLIGHT_JSON,
    }


def _merge_manifest(payload):
    """Keep earlier launches in the manifest rather than overwriting them.

    Re-running the launcher is the supported way to recover from a crash, so the
    manifest has to be a history: which runs were started when, and which were
    resumed from what. Overwriting it would erase the record of the first launch
    the moment the second one happened.
    """
    history = []
    if os.path.exists(MANIFEST):
        try:
            with open(MANIFEST, encoding="utf-8") as handle:
                previous = json.load(handle)
            history = previous.get("history", [])
            history.append({k: v for k, v in previous.items() if k != "history"})
        except (ValueError, OSError):
            pass
    payload["history"] = history
    return payload


if __name__ == "__main__":
    sys.exit(main())
