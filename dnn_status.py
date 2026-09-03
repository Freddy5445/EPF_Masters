"""
Where the ten phase-2 runs have got to. Read-only, works at any moment.

    python dnn_status.py                 # one snapshot
    python dnn_status.py --watch         # refresh every 60 seconds
    python dnn_status.py --json          # machine-readable
    python dnn_status.py --heartbeat     # append one row per run per hour

It reads the artifacts the runs are already writing -- the per-day forecast
checkpoints and the per-seed timing files -- and asks the operating system
whether each process is alive. The runs report nothing, are asked for nothing,
and are never written to, locked, or opened for anything but reading: a status
tool that could disturb a 21-hour backtest would not be worth having.

It therefore works whether the runs are alive, finished, or dead, and whether or
not this session launched them.

Two things it deliberately does not do the obvious way:

* **The rate is measured over the last 20 days, not the whole run.** Early days
  finish faster, before ten processes are all competing for eight cores, so a
  whole-run mean flatters the estimate exactly when the estimate matters most.
* **``stalled`` and ``failed`` are distinct and both visible without opening a
  log.** ``stalled`` is a live process that has completed no new day in 30
  minutes; ``failed`` is a process that is gone with days still to forecast.
  Silence looks identical to progress otherwise.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(THIS_DIR, "experiments")
PHASE2_DIR = os.path.join(DEFAULT_OUT, "dnn_phase2")
DEFAULT_LOG_DIR = os.path.join(DEFAULT_OUT, "logs")
MANIFEST = os.path.join(PHASE2_DIR, "launch_manifest.json")
PROGRESS_CSV = os.path.join(PHASE2_DIR, "progress_log.csv")
PHASE1_TIMINGS = os.path.join(DEFAULT_OUT, "timing", "phase1_timings.json")

# A live process with no new completed day for this long is stalled, not working.
STALL_SECONDS = 30 * 60

# The window the rate is measured over. Long enough to smooth a slow day, short
# enough to track contention as it builds.
RATE_WINDOW_DAYS = 20

WATCH_SECONDS = 60
# The schema section 5.1 specifies, plus `phase` and `hyperopt_trials_done`:
# without them the first hours of every run are a flat line at zero days, and the
# history cannot tell a long search from a stuck one.
PROGRESS_COLUMNS = ["timestamp", "run_id", "state", "days_done", "seeds_done",
                    "seconds_per_day_recent", "eta_utc", "phase",
                    "hyperopt_trials_done"]


def _fmt(seconds):
    if seconds is None or not (seconds == seconds) or seconds < 0:
        return "-"
    return str(timedelta(seconds=int(round(seconds))))


def _load_manifest():
    if not os.path.exists(MANIFEST):
        return {}
    try:
        with open(MANIFEST, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (ValueError, OSError):
        return {}
    records = {}
    # Latest launch wins, but earlier ones fill in runs this launch skipped.
    for block in list(manifest.get("history", [])) + [manifest]:
        for record in block.get("runs", []) or []:
            records[record["run_id"]] = record
    return records


def _phase1_rates():
    """Measured seconds per recalibration per configuration, if available.

    Used only to estimate a run that has not started yet and so has no rate of
    its own. Labelled as an estimate wherever it is shown.
    """
    if not os.path.exists(PHASE1_TIMINGS):
        return {}
    try:
        with open(PHASE1_TIMINGS, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (ValueError, OSError):
        return {}
    return {config: {"recalibration": measurement["recalibration"]["mean"],
                     "hyperopt": measurement.get("hyperopt", {}).get("mean")}
            for config, measurement in payload.get("per_configuration", {}).items()
            if measurement.get("recalibration")}


def _process_alive(pid, run_id):
    """Whether ``pid`` is a live process that is actually this run.

    A bare ``pid_exists`` is not enough: PIDs are reused, and after a reboot the
    number in the manifest may well belong to something else entirely. The
    command line is checked so a recycled PID cannot be reported as a healthy
    backtest.
    """
    if not pid:
        return False, None
    try:
        import psutil
    except ImportError:
        return None, "psutil not installed; liveness unknown"
    try:
        process = psutil.Process(int(pid))
        command = " ".join(process.cmdline())
    except Exception:
        return False, None
    if "run_dnn_dk1.py" not in command:
        return False, "pid reused by another process"
    return True, command


def _run_dir_for(run, out_dir):
    import run_dnn_dk1 as R
    from dnn_dk1 import runs as RS

    return R.run_dir_for(run.config, run.focal, RS.BEGIN_TEST, RS.END_TEST,
                         out_dir, smoke=False)


def _read_forecasts(run_dir, config):
    """Completed days per seed, from the checkpoints themselves."""
    prefix = "forecasts_joint_seed" if config == "joint" else "forecasts_seed"
    per_seed = {}
    for path in sorted(glob.glob(os.path.join(run_dir, f"{prefix}*.csv"))):
        match = re.search(rf"{prefix}(\d+)\.csv$", os.path.basename(path))
        if not match:
            continue
        try:
            frame = pd.read_csv(path, index_col=0)
        except (OSError, ValueError, pd.errors.EmptyDataError):
            # A checkpoint caught mid-write. Not an error -- try again next tick.
            continue
        frame.index = pd.to_datetime(frame.index, errors="coerce")
        # Only complete rows count, exactly as _load_checkpoint and the
        # evaluator treat them.
        per_seed[int(match.group(1))] = frame.dropna(how="any").index
    return per_seed


def _read_timings(run_dir):
    """Every per-seed timing row, concatenated."""
    frames = []
    for path in sorted(glob.glob(os.path.join(run_dir, "timings_seed*.csv"))):
        try:
            frames.append(pd.read_csv(path))
        except (OSError, ValueError, pd.errors.EmptyDataError):
            continue
    if not frames:
        return pd.DataFrame(columns=["date", "seed", "seconds"])
    return pd.concat(frames, ignore_index=True)


def _recent_seconds_per_day(timings, window=RATE_WINDOW_DAYS):
    """Mean seconds of work per test day, over the last ``window`` days only.

    Summed across seeds within a day, because the loop is day-outer/seed-inner:
    a "day" of wall clock is every seed's fit for that day.
    """
    if timings.empty or "date" not in timings:
        return None, 0
    per_day = timings.groupby("date")["seconds"].sum().sort_index()
    if per_day.empty:
        return None, 0
    recent = per_day.tail(window)
    return float(recent.mean()), int(len(recent))


def _last_progress_time(run_dir, trials_path=None):
    """When this run last wrote anything. The stall signal.

    The trials file counts: hyperopt checkpoints before every evaluation, and a
    run spends its first hours there with no forecast written yet. Without it, a
    hung search would look exactly like a healthy one.
    """
    newest = None
    paths = []
    for pattern in ("forecasts_*.csv", "timings_seed*.csv"):
        paths += glob.glob(os.path.join(run_dir, pattern))
    if trials_path:
        paths.append(trials_path)
    for path in paths:
        try:
            stamp = os.path.getmtime(path)
        except OSError:
            continue
        newest = stamp if newest is None else max(newest, stamp)
    return newest


def _hyperopt_progress(run, out_dir):
    """How far the hyperparameter search has got: ``(path, trials_done)``.

    A run does nothing visible for its first two or three hours -- 300 TPE
    evaluations at the 25-33 s phase 1 measured -- and during that time it has
    forecast no days at all. Reported as its own phase, because "0/731, 0%" for
    three hours is indistinguishable from a run that never started.

    The trials file is hyperopt's own checkpoint, rewritten before each
    evaluation, so counting it costs nothing and touches nothing.
    """
    import run_dnn_dk1 as R
    from dnn_dk1 import runs as RS
    from dnn_dk1.forecaster import hyperparameter_path

    path = hyperparameter_path(
        os.path.join(out_dir, "hyperparameters"), 1, RS.NLAYERS,
        R.dataset_name(run.config, run.focal), 2, 1, 0, RS.CALIBRATION_YEARS)
    if not os.path.exists(path):
        return None, 0
    try:
        import pickle as pc

        with open(path, "rb") as handle:
            trials = pc.load(handle)
        done = sum(1 for loss in trials.losses() if loss is not None)
    except Exception:
        # Caught mid-write, or written by a different hyperopt. The file's
        # existence still counts as progress; the count does not.
        return path, None
    return path, done


def collect(out_dir=DEFAULT_OUT, log_dir=DEFAULT_LOG_DIR, now=None):
    """One dict per run. The whole of this tool's knowledge."""
    from dnn_dk1 import runs as RS

    now = time.time() if now is None else now
    manifest = _load_manifest()
    fallback_rates = _phase1_rates()
    states = []

    for run in RS.RUNS:
        run_dir = _run_dir_for(run, out_dir)
        record = manifest.get(run.run_id, {})
        per_seed = _read_forecasts(run_dir, run.config) if os.path.isdir(run_dir) else {}
        timings = _read_timings(run_dir) if os.path.isdir(run_dir) else pd.DataFrame()

        seeds_expected = len(RS.SEEDS)
        if per_seed:
            complete = None
            for index in per_seed.values():
                complete = index if complete is None else complete.intersection(index)
            days_done = 0 if complete is None else len(complete)
            latest = max((index.max() for index in per_seed.values()
                          if len(index)), default=None)
            seeds_on_latest = sum(1 for index in per_seed.values()
                                  if latest is not None and latest in index)
        else:
            days_done, latest, seeds_on_latest = 0, None, 0

        seconds_per_day, window = _recent_seconds_per_day(timings)
        estimated = False
        if seconds_per_day is None and run.config in fallback_rates:
            seconds_per_day = (fallback_rates[run.config]["recalibration"]
                               * seeds_expected)
            estimated = True

        trials_path, trials_done = _hyperopt_progress(run, out_dir)
        in_hyperopt = days_done == 0 and (trials_done or 0) < RS.MAX_EVALS

        alive, command = _process_alive(record.get("pid"), run.run_id)
        last_progress = _last_progress_time(run_dir if os.path.isdir(run_dir)
                                            else out_dir, trials_path)
        idle = (now - last_progress) if last_progress else None

        # Remaining hyperparameter evaluations, at the phase 1 per-evaluation
        # mean. Only an estimate, but the alternative is showing nothing for the
        # first two or three hours of every run.
        hyperopt_seconds = None
        per_eval = (fallback_rates.get(run.config) or {}).get("hyperopt")
        if per_eval and (trials_done or 0) < RS.MAX_EVALS:
            hyperopt_seconds = (RS.MAX_EVALS - (trials_done or 0)) * per_eval

        if days_done >= RS.TEST_DAYS:
            state = "done"
        elif alive:
            state = ("stalled" if idle is not None and idle > STALL_SECONDS
                     else "running")
        elif record.get("pid"):
            state = "failed"
        elif days_done or os.path.isdir(run_dir):
            # Artifacts but no process this launcher knows of: an earlier run
            # that is no longer alive. Not queued -- something started it.
            state = "failed"
        else:
            state = "queued"

        remaining = max(RS.TEST_DAYS - days_done, 0)
        eta_seconds = (remaining * seconds_per_day
                       if seconds_per_day and remaining else
                       (0.0 if state == "done" else None))
        # A run still searching has its whole backtest ahead of it *and* the
        # rest of the search; an ETA counting only the backtest would be short
        # by hours at exactly the moment it is first consulted.
        if eta_seconds is not None and hyperopt_seconds and in_hyperopt:
            eta_seconds += hyperopt_seconds

        started_at = record.get("started_at")
        elapsed = None
        if started_at:
            try:
                elapsed = now - datetime.fromisoformat(started_at).timestamp()
            except ValueError:
                elapsed = None
        if elapsed is None and os.path.isdir(run_dir):
            oldest = min((os.path.getmtime(p) for p in
                          glob.glob(os.path.join(run_dir, "*"))), default=None)
            elapsed = (now - oldest) if oldest else None

        states.append({
            "run_id": run.run_id,
            "config": run.config,
            "zone": run.zone,
            "state": state,
            "phase": "hyperopt" if (in_hyperopt and state != "done")
                     else "backtest",
            "hyperopt_trials_done": trials_done,
            "hyperopt_trials_total": RS.MAX_EVALS,
            "days_done": days_done,
            "days_total": RS.TEST_DAYS,
            "pct": round(100.0 * days_done / RS.TEST_DAYS, 1),
            "seeds_done": f"{seeds_on_latest}/{seeds_expected}",
            "seeds_on_latest_day": seeds_on_latest,
            "latest_day": str(latest.date()) if latest is not None else None,
            "elapsed_seconds": elapsed,
            "seconds_per_day_recent": (round(seconds_per_day, 1)
                                       if seconds_per_day else None),
            "rate_window_days": window,
            "rate_is_estimate": estimated,
            "eta_seconds": eta_seconds,
            "eta_utc": (
                (datetime.now(timezone.utc) + timedelta(seconds=eta_seconds))
                .strftime("%Y-%m-%dT%H:%M:%SZ") if eta_seconds else None),
            "pid": record.get("pid"),
            "idle_seconds": idle,
            "run_dir": run_dir,
            "log": record.get("log") or os.path.join(log_dir, f"{run.run_id}.log"),
        })
    return states


def render(states):
    """The table, plus the footer that answers "when will this be finished?"."""
    from dnn_dk1 import runs as RS

    lines = [
        f"{'run':<10} {'state':<8} {'days':>9} {'pct':>6} {'seeds':>6} "
        f"{'elapsed':>10} {'s/day':>8} {'eta':>10}  finish (UTC)",
        "-" * 92,
    ]
    for row in states:
        rate = ("-" if row["seconds_per_day_recent"] is None
                else f"{row['seconds_per_day_recent']:.0f}"
                     + ("~" if row["rate_is_estimate"] else ""))
        # While the search is running there are no days yet, so the days column
        # shows what is actually happening instead of a motionless 0/731.
        if row["phase"] == "hyperopt" and row["state"] != "queued":
            done = row["hyperopt_trials_done"]
            progress = (f"hpo {done}/{row['hyperopt_trials_total']}"
                        if done is not None else "hpo ...")
        else:
            progress = f"{row['days_done']:>4}/{row['days_total']:<4}"
        lines.append(
            f"{row['run_id']:<10} {row['state']:<8} "
            f"{progress:>9} "
            f"{row['pct']:>5.1f}% {row['seeds_done']:>6} "
            f"{_fmt(row['elapsed_seconds']):>10} {rate:>8} "
            f"{_fmt(row['eta_seconds']):>10}  {row['eta_utc'] or '-'}")

    done = sum(row["days_done"] for row in states)
    total = sum(row["days_total"] for row in states)
    elapsed = [row["elapsed_seconds"] for row in states
               if row["elapsed_seconds"] is not None]
    etas = [row["eta_seconds"] for row in states if row["eta_seconds"]]
    counts = {}
    for row in states:
        counts[row["state"]] = counts.get(row["state"], 0) + 1

    lines.append("-" * 92)
    lines.append(
        f"{'ALL':<10} {'':8} {done:>4}/{total:<4} "
        f"{100.0 * done / total:>5.1f}% {'':>6} "
        f"{_fmt(max(elapsed)) if elapsed else '-':>10} {'':>8} "
        f"{_fmt(max(etas)) if etas else '-':>10}  "
        + (f"last run finishes "
           f"{(datetime.now(timezone.utc) + timedelta(seconds=max(etas))).strftime('%Y-%m-%dT%H:%M:%SZ')}"
           if etas else "no active run"))
    lines.append("  " + ", ".join(f"{n} {state}" for state, n
                                  in sorted(counts.items())))
    if any(row["rate_is_estimate"] for row in states):
        lines.append("  ~ = rate estimated from the phase 1 timings; this run "
                     "has not produced a day of its own yet")
    if any(row["state"] == "stalled" for row in states):
        lines.append(f"  stalled = process alive, no new completed day in "
                     f"{STALL_SECONDS // 60} minutes")
    if any(row["state"] == "failed" for row in states):
        lines.append("  failed  = process gone with days remaining; "
                     "`python run_dnn_all.py` resumes it from its checkpoint")
    if any(row["phase"] == "hyperopt" and row["state"] not in ("queued", "done")
           for row in states):
        lines.append("  hpo n/300 = still searching hyperparameters; the "
                     "backtest, and the days column, begin after it")
    return "\n".join(lines)


def heartbeat_row(states):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return pd.DataFrame([{
        "timestamp": now,
        "run_id": row["run_id"],
        "state": row["state"],
        "days_done": row["days_done"],
        "seeds_done": row["seeds_on_latest_day"],
        "seconds_per_day_recent": row["seconds_per_day_recent"],
        "eta_utc": row["eta_utc"],
        "phase": row["phase"],
        "hyperopt_trials_done": row["hyperopt_trials_done"],
    } for row in states], columns=PROGRESS_COLUMNS)


def append_heartbeat(states, path=PROGRESS_CSV):
    """One row per run, appended. History, not a snapshot.

    A point-in-time status cannot tell a run that is slowing down from one that
    was always slow. The log can, and it lets anything else read progress without
    going anywhere near the runs.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = not os.path.exists(path)
    heartbeat_row(states).to_csv(path, mode="a", header=header, index=False)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Progress of the ten phase-2 DNN runs. Read-only.")
    parser.add_argument("--watch", action="store_true",
                        help=f"Refresh every {WATCH_SECONDS} seconds")
    parser.add_argument("--json", action="store_true",
                        help="Machine-readable output")
    parser.add_argument("--heartbeat", action="store_true",
                        help="Append one row per run to progress_log.csv "
                             "every --interval seconds, forever")
    parser.add_argument("--interval", type=float, default=3600.0,
                        help="Heartbeat interval in seconds (default: hourly)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--csv", default=PROGRESS_CSV)
    args = parser.parse_args(argv)

    if args.heartbeat:
        print(f"heartbeat: appending to {args.csv} every {args.interval:.0f}s")
        while True:
            try:
                states = collect(args.out_dir, args.log_dir)
                append_heartbeat(states, args.csv)
                done = sum(row["days_done"] for row in states)
                total = sum(row["days_total"] for row in states)
                print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
                      f"  {done}/{total} days "
                      f"({100.0 * done / total:.1f}%)", flush=True)
                if all(row["state"] in ("done", "failed") for row in states):
                    print("all runs finished or failed; heartbeat exiting",
                          flush=True)
                    return 0
            except Exception as exc:  # never let the reporter die on the runs
                print(f"heartbeat error (continuing): {type(exc).__name__}: "
                      f"{exc}", flush=True)
            time.sleep(args.interval)

    while True:
        states = collect(args.out_dir, args.log_dir)
        if args.json:
            print(json.dumps(states, indent=2, default=str))
        else:
            if args.watch:
                print("\033[2J\033[H", end="")
                print(f"dnn_status  {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(render(states))
        if not args.watch:
            return 0
        time.sleep(WATCH_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
