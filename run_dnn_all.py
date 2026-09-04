"""
Launch the ten phase-2 runs as detached processes, and resume them if re-run.

    python run_dnn_all.py --dry-run     # print the ten command lines, run nothing
    python run_dnn_all.py               # launch

Each run becomes an **independent** OS process with its own log under
``experiments/logs/<run_id>.log``, and **no console window appears**. Closing a
terminal, ending the agent session, or killing this launcher does not kill them.
On Windows that is enforced four ways at once -- ``pythonw.exe``,
``DETACHED_PROCESS``, ``CREATE_NO_WINDOW`` and a hidden ``STARTUPINFO`` -- because
a launch relying on ``DETACHED_PROCESS`` alone did produce ten console windows,
and closing three of them killed three runs. See ``_popen_detached``.
One run crashing has no effect on the other nine.

Order is ``joint``, then ``joint_dk``, then the two ``wide``, then the seven
``own``.
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

# Mirrors run_dnn_dk1.CHUNK_INCOMPLETE, imported lazily there to keep this
# module free of TensorFlow.
CHUNK_INCOMPLETE = 75


def windowless_python(executable=None):
    """``pythonw.exe`` beside the current interpreter, if there is one.

    This is the fix for the ten console windows, and the reason the creation
    flags alone could not be.

    ``<venv>/Scripts/python.exe`` is not the interpreter. It is a redirector stub
    that starts the real interpreter as a **new process**, and that second
    ``CreateProcess`` is the stub's, not ours -- it carries none of the flags we
    passed. So the stub is correctly detached and consoleless, and then it
    launches a console-subsystem child with no flags at all; Windows duly gives
    that child a fresh console. One window per run, hosting a process we never
    created and whose PID we never recorded. Closing the window sends
    ``CTRL_CLOSE_EVENT`` to it and the run dies.

    Measured, not inferred::

        python.exe   stub 37632 -> re-exec 29036, conhost 39528   <- a window
        pythonw.exe  stub 32796 -> re-exec 13212, no conhost      <- none

    ``pythonw.exe`` is the same interpreter built for the GUI subsystem, and its
    stub re-execs ``pythonw.exe`` in turn. Windows never allocates a console for
    a GUI-subsystem process, so the property holds however many times the binary
    re-execs itself and whatever flags are lost on the way. That is what makes
    this a fix rather than another flag that has to survive a hop.

    The flags stay too -- see :func:`detach_flags` -- for the first hop.
    """
    executable = executable or sys.executable
    if os.name != "nt":
        return executable
    directory, name = os.path.split(executable)
    if name.lower() == "pythonw.exe":
        return executable
    if name.lower().startswith("python") and name.lower().endswith(".exe"):
        candidate = os.path.join(directory, "pythonw.exe")
        if os.path.exists(candidate):
            return candidate
    return executable


def run_command(run, out_dir, datasets_dir, python=None, cadence=None,
                days_per_process=None):
    """The exact command line for one run.

    Every parameter is taken from :mod:`dnn_dk1.runs` rather than typed here, so
    a run cannot silently differ from the settled configuration or from what the
    pre-flight smoked.

    Note what is *not* here: no ``cmd /c``, no ``start``, no shell. The per-run
    thread settings go through ``Popen(env=...)`` (see :func:`run_env`), because
    a wrapper that set them would own a console of its own and hand one to the
    run.
    """
    from dnn_dk1 import runs as RS

    return [
        python or windowless_python(),
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
        # Spelled out rather than left to the worker's defaults, so the manifest
        # records the cadence and the chunk size that actually ran.
        "--recalibration-days", str(cadence if cadence is not None
                                    else RS.RECALIBRATION_DAYS),
        "--max-days-per-process", str(days_per_process
                                      if days_per_process is not None
                                      else RS.DAYS_PER_PROCESS),
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
    started from. ``DETACHED_PROCESS`` gives it no console at all,
    ``CREATE_NO_WINDOW`` says the same thing a second way, and
    ``CREATE_NEW_PROCESS_GROUP`` keeps a Ctrl-C here from reaching it. Elsewhere,
    ``start_new_session`` does the same job.

    Windows ignores ``CREATE_NO_WINDOW`` when ``DETACHED_PROCESS`` takes effect,
    so asking for both costs nothing and they are not mutually exclusive (unlike
    ``CREATE_NEW_CONSOLE``, which would make ``CreateProcess`` fail outright).

    These flags govern the process *we* create. They say nothing about any
    process that one re-execs, which is exactly how the console windows got in
    last time -- see :func:`windowless_python`, which is what actually closes
    that hole.
    """
    if os.name == "nt":
        detached = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        new_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        return {"creationflags": detached | no_window | new_group}
    return {"start_new_session": True}


def hidden_startupinfo():
    """A ``STARTUPINFO`` asking for a hidden window, for anything that ignores flags."""
    if os.name != "nt":
        return None
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    return info


def _popen_detached(command, log_handle, env):
    """The one place a background process is created.

    Four reasons no console can appear, because the obvious one was not enough:

    * the binary is ``pythonw.exe``, which has no console subsystem -- the only
      one of the four that also covers the interpreter the venv stub re-execs,
      and therefore the one that actually fixes it;
    * ``DETACHED_PROCESS`` asks for no console;
    * ``CREATE_NO_WINDOW`` asks for no window;
    * ``STARTUPINFO`` asks for any window there might be to be hidden.

    stdout and stderr go to the run's own log file and stdin to DEVNULL, so the
    process needs no console to write to and none to read from. There is no
    shell and no wrapper anywhere in the chain: the environment reaches the
    child through ``env=``.
    """
    return subprocess.Popen(
        command, cwd=THIS_DIR, env=env,
        stdout=log_handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        close_fds=True, startupinfo=hidden_startupinfo(), **detach_flags())


def spawn(run, out_dir, datasets_dir, log_dir, popen=None, cadence=None,
          days_per_process=None):
    """Start one run detached, appending to its own log. Returns the PID."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run.run_id}.log")

    # Append, never truncate: a resumed run's log should read as one history.
    handle = open(log_path, "a", encoding="utf-8", buffering=1)
    handle.write(f"\n{'=' * 78}\n{time.strftime('%Y-%m-%dT%H:%M:%S')}  "
                 f"start {run.run_id} ({run.label}, {run.n_inputs} in -> "
                 f"{run.n_outputs} out, {run.threads} thread(s))\n{'=' * 78}\n")
    handle.flush()

    process = _popen_detached(
        run_command(run, out_dir, datasets_dir, cadence=cadence,
                    days_per_process=days_per_process), handle, run_env(run))
    handle.close()
    if popen is not None:
        # The scheduler keeps the handle so it can read the exit code and tell a
        # planned chunk handover (CHUNK_INCOMPLETE) from a crash.
        popen[run.run_id] = process
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
    process = _popen_detached(
        [windowless_python(), os.path.join(THIS_DIR, "dnn_status.py"),
         "--heartbeat", "--interval", str(interval)], handle, env)
    handle.close()
    return process.pid, log_path


def manifest_pids(path=MANIFEST):
    """``run_id -> pid`` from every launch the manifest records, newest first.

    Kept for the record and for reporting. It is no longer the liveness signal:
    a run is now a sequence of worker processes, one per chunk, so its PID
    changes every hundred days or so. :mod:`dnn_dk1.procs` answers liveness from
    the command line instead.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (ValueError, OSError):
        return {}
    pids = {}
    for block in list(manifest.get("history", [])) + [manifest]:
        for record in block.get("runs", []) or []:
            if record.get("pid"):
                pids[record["run_id"]] = record["pid"]
        beat = block.get("heartbeat") or {}
        if beat.get("pid"):
            pids["__heartbeat__"] = beat["pid"]
        if block.get("scheduler", {}).get("pid"):
            pids["__scheduler__"] = block["scheduler"]["pid"]
    return pids


def process_is_alive(pid, expect="run_dnn_dk1.py"):
    """See :func:`dnn_dk1.procs.is_alive`."""
    from dnn_dk1 import procs

    return procs.is_alive(pid, expect)


def live_processes_for(run):
    """See :func:`dnn_dk1.procs.live_workers`."""
    from dnn_dk1 import procs

    return procs.live_workers(run)


def _has_pair(argv, flag, value):
    from dnn_dk1 import procs

    return procs.has_pair(argv, flag, value)


# ---------------------------------------------------------------------------
# The scheduler
# ---------------------------------------------------------------------------

SCHEDULE_POLL_SECONDS = 30
MAX_CONSECUTIVE_FAILURES = 3


def _run_state(run, out_dir):
    """``(days_done, complete)`` for one run, read from its own checkpoint."""
    import dnn_status
    from dnn_dk1 import procs
    from dnn_dk1 import runs as RS

    run_dir = dnn_status._run_dir_for(run, out_dir)
    if not os.path.isdir(run_dir):
        return 0, False
    per_seed = dnn_status._read_forecasts(run_dir, run.config)
    if not per_seed:
        return 0, False
    common = None
    for index in per_seed.values():
        common = index if common is None else common.intersection(index)
    days = 0 if common is None else len(common)
    return days, days >= RS.TEST_DAYS


def schedule(args):
    """Start runs as slots free up, and restart each one after every chunk.

    Two jobs, both of which have to happen somewhere and neither of which a
    detached one-shot launcher can do:

    * **Admission.** Ten processes on eight physical cores is what made joint
      nine times slower than it was measured alone. At most
      ``MAX_OTHER_CONCURRENT`` single-threaded runs run beside the one resident
      joint-class run's four threads -- eight threads on eight cores -- and the
      rest wait. ``joint`` and ``joint_dk`` share that one slot.
    * **Chunking.** A worker exits with ``CHUNK_INCOMPLETE`` after about a
      hundred forecast days so its heap goes back to the operating system. The
      run is not finished; it is handed over. Restarting it here, rather than
      having the worker respawn itself, keeps one process responsible for what
      is running and keeps the admission limit honest -- a worker that respawned
      itself would take a slot without asking.

    Nothing here holds forecast state. Every restart resumes from the run's own
    per-day checkpoint through the same ``_load_checkpoint`` path LEAR uses, so
    the scheduler dying costs only the scheduling, and re-running
    ``run_dnn_all.py`` picks it all up again.
    """
    from dnn_dk1 import procs
    from dnn_dk1 import runs as RS

    order = ([RS.get(r.strip()) for r in args.only.split(",")]
             if args.only else list(RS.RUNS))
    children = {}      # run_id -> Popen we started
    failures = {}      # run_id -> consecutive non-chunk failures
    abandoned = set()

    chunk = (args.days_per_process if args.days_per_process is not None
             else RS.DAYS_PER_PROCESS)
    cadence = (args.recalibration_days if args.recalibration_days is not None
               else RS.RECALIBRATION_DAYS)
    print(f"scheduler: {len(order)} run(s), at most {RS.MAX_CONCURRENT} at once "
          f"(joint on {RS.JOINT_THREADS} threads + up to "
          f"{RS.MAX_OTHER_CONCURRENT} single-threaded), "
          f"{chunk} days per process, recalibrating every {cadence} day(s)",
          flush=True)

    while True:
        try:
            # --- reap ---------------------------------------------------
            for run_id, process in list(children.items()):
                code = process.poll()
                if code is None:
                    continue
                del children[run_id]
                if code == CHUNK_INCOMPLETE:
                    failures[run_id] = 0
                    print(f"{_stamp()} {run_id}: chunk done, will resume",
                          flush=True)
                elif code == 0:
                    failures[run_id] = 0
                    print(f"{_stamp()} {run_id}: exited 0", flush=True)
                else:
                    failures[run_id] = failures.get(run_id, 0) + 1
                    print(f"{_stamp()} {run_id}: exited {code} "
                          f"(failure {failures[run_id]}/"
                          f"{MAX_CONSECUTIVE_FAILURES})", flush=True)
                    if failures[run_id] >= MAX_CONSECUTIVE_FAILURES:
                        abandoned.add(run_id)
                        print(f"{_stamp()} {run_id}: giving up after "
                              f"{MAX_CONSECUTIVE_FAILURES} consecutive "
                              f"failures; the others continue", flush=True)

            # --- survey -------------------------------------------------
            running, done, queued = [], [], []
            for run in order:
                if run.run_id in abandoned:
                    continue
                pids, certain = procs.live_workers(run)
                if pids or not certain:
                    running.append(run)
                    continue
                days, complete = _run_state(run, args.out_dir)
                (done if complete else queued).append(run)

            if not queued and not running:
                print(f"{_stamp()} nothing left to run: {len(done)} complete, "
                      f"{len(abandoned)} abandoned", flush=True)
                return 0

            # --- admit --------------------------------------------------
            others_running = sum(1 for r in running if not r.heavy)
            joint_running = any(r.heavy for r in running)

            for run in queued:
                if run.heavy:
                    if joint_running:
                        continue
                    joint_running = True
                elif others_running >= RS.MAX_OTHER_CONCURRENT:
                    continue
                else:
                    others_running += 1

                days, _ = _run_state(run, args.out_dir)
                pid, log_path = spawn(
                    run, args.out_dir, args.datasets_dir, args.log_dir,
                    popen=children, cadence=args.recalibration_days,
                    days_per_process=args.days_per_process)
                print(f"{_stamp()} {run.run_id}: started pid {pid}"
                      + (f", resuming from {days} day(s)" if days else "")
                      + f" -> {os.path.basename(log_path)}", flush=True)
                _record_start(run, pid, log_path, days, args)

        except Exception as exc:                      # never let one run stop all
            print(f"{_stamp()} scheduler error (continuing): "
                  f"{type(exc).__name__}: {exc}", flush=True)

        time.sleep(args.poll)


def _stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _record_start(run, pid, log_path, resumed_from, args):
    """Append one start to the manifest. Best effort; never fails a launch."""
    try:
        manifest = {}
        if os.path.exists(MANIFEST):
            with open(MANIFEST, encoding="utf-8") as handle:
                manifest = json.load(handle)
        starts = manifest.setdefault("runs", [])
        starts = [r for r in starts if r.get("run_id") != run.run_id]
        starts.append({
            "run_id": run.run_id, "config": run.config, "zone": run.zone,
            "n_inputs": run.n_inputs, "n_outputs": run.n_outputs,
            "threads": run.threads, "pid": pid, "log": log_path,
            "started_at": _stamp(), "resumed_from_days": resumed_from,
            "command": run_command(run, args.out_dir, args.datasets_dir,
                               cadence=args.recalibration_days,
                               days_per_process=args.days_per_process),
        })
        manifest["runs"] = starts
        with open(MANIFEST, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, default=str)
    except (OSError, ValueError):
        pass


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
                        help="Seconds between launches, so several TensorFlow "
                             "imports do not contend at once (default: 20)")
    parser.add_argument("--recalibration-days", type=int, default=None,
                        help=f"Override the settled recalibration cadence "
                             f"(default: 2). For testing.")
    parser.add_argument("--days-per-process", type=int, default=None,
                        help="Override how many days a worker forecasts before "
                             "exiting for the scheduler to restart it. For "
                             "testing; production uses the settled value.")
    parser.add_argument("--schedule", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--poll", type=float, default=SCHEDULE_POLL_SECONDS,
                        help=argparse.SUPPRESS)
    parser.add_argument("--foreground", action="store_true",
                        help="Run the scheduler in this process instead of "
                             "detaching it. For debugging; closing the terminal "
                             "then does stop it.")
    args = parser.parse_args(argv)

    if args.schedule:
        return schedule(args)

    import dnn_status
    from dnn_dk1 import procs
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

    # --- what is already done or already running ------------------------
    states = dnn_status.collect(out_dir=args.out_dir, log_dir=args.log_dir)
    by_id = {s["run_id"]: s for s in states}
    pids = manifest_pids()

    print(f"{'run':<10} {'config':<6} {'in':>5} {'out':>4} {'thr':>3}  "
          f"{'state':<9} {'days':>9}  action")
    print("-" * 78)

    launched, skipped = [], []
    for run in runs:
        state = by_id.get(run.run_id, {})
        status = state.get("state", "queued")
        days = f"{state.get('days_done', 0)}/{RS.TEST_DAYS}"
        alive = process_is_alive(pids.get(run.run_id))
        others, certain = live_processes_for(run)
        if others or not certain:
            alive = True if others else (alive if alive is not False else None)

        if status == "done":
            action = "skip -- complete"
        elif alive is not False:
            where = others or [pids.get(run.run_id)]
            action = (f"skip -- alive (pid {where[0]})" if alive
                      else f"skip -- liveness unknown (pid {pids.get(run.run_id)})")
            status = "running" if alive else status
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

        if status == "done" or alive is not False:
            skipped.append(run)
        else:
            launched.append(run)

    if args.dry_run:
        admitted = _admission_plan(launched)
        print(f"\nScheduling plan: at most {RS.MAX_CONCURRENT} runs at once "
              f"(joint on {RS.JOINT_THREADS} threads + up to "
              f"{RS.MAX_OTHER_CONCURRENT} single-threaded = "
              f"{RS.CONCURRENT_THREADS} threads on 8 physical cores)")
        for wave, group in enumerate(admitted, 1):
            print(f"  wave {wave}: {', '.join(r.run_id for r in group)}")
        print(f"  each worker forecasts up to {RS.DAYS_PER_PROCESS} days, exits, "
              f"and is restarted from its checkpoint")
        print(f"  {RS.REFITS} refits cover {RS.TEST_DAYS} forecast days "
              f"(recalibrating every {RS.RECALIBRATION_DAYS} days)")

        print(f"\nCommand lines ({len(launched)} would launch; "
              f"{len(skipped)} skipped):")
        print("=" * 78)
        for run in launched:
            env = {k: str(run.threads) for k in
                   ("OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS",
                    "TF_NUM_INTEROP_THREADS")}
            command = run_command(
            run, args.out_dir, args.datasets_dir,
            cadence=args.recalibration_days,
            days_per_process=args.days_per_process)
            print(f"\n# {run.run_id} -- {run.label}, {run.n_inputs} in -> "
                  f"{run.n_outputs} out")
            print("  env: " + " ".join(f"{k}={v}" for k, v in env.items()))
            print("  " + subprocess.list2cmdline(command))
            print(f"  log: {os.path.join(args.log_dir, run.run_id + '.log')}")
        print("\n" + "=" * 78)
        print("dry run -- nothing was launched.")
        return 0

    # --- launch ---------------------------------------------------------
    # Created here, not before the dry-run branch: `--dry-run` must leave the
    # filesystem exactly as it found it, or "nothing was launched" is not quite
    # true and a cleaned slate quietly stops being clean.
    os.makedirs(PHASE2_DIR, exist_ok=True)

    # One scheduler, ever. A second would admit its own five runs beside the
    # first's, which is both the concurrency limit gone and two processes
    # writing one checkpoint.
    existing, certain = procs.live_scheduler()
    if existing or not certain:
        reason = (f"a scheduler is already running (pid {existing[0]})"
                  if existing else
                  "cannot tell whether a scheduler is running")
        print(f"\n{reason}. It is already resuming whatever is unfinished; "
              f"nothing to do.\nWatch it with `python dnn_status.py`.",
              file=sys.stderr)
        return 0 if existing else 1

    if args.foreground:
        print("running the scheduler in the foreground (Ctrl-C stops it)\n")
        return schedule(args)

    os.makedirs(args.log_dir, exist_ok=True)
    scheduler_log = os.path.join(args.log_dir, "scheduler.log")
    handle = open(scheduler_log, "a", encoding="utf-8", buffering=1)
    handle.write(f"\n{'=' * 78}\n{time.strftime('%Y-%m-%dT%H:%M:%S')}  "
                 f"scheduler start\n{'=' * 78}\n")
    handle.flush()
    command = [windowless_python(), os.path.abspath(__file__), "--schedule",
               "--out-dir", args.out_dir, "--datasets-dir", args.datasets_dir,
               "--log-dir", args.log_dir, "--poll", str(args.poll)]
    if args.only:
        command += ["--only", args.only]
    # Forward the overrides, or the detached scheduler silently falls back to
    # the settled defaults and the run does something other than what was asked.
    if args.recalibration_days is not None:
        command += ["--recalibration-days", str(args.recalibration_days)]
    if args.days_per_process is not None:
        command += ["--days-per-process", str(args.days_per_process)]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    scheduler = _popen_detached(command, handle, env)
    handle.close()
    print(f"\nscheduler started, pid {scheduler.pid} -> {scheduler_log}")

    heartbeat = None
    beat_alive, beat_certain = procs.live_heartbeat()
    if not args.no_heartbeat and (beat_alive or not beat_certain):
        print(f"  heartbeat already running (pid "
              f"{beat_alive[0] if beat_alive else '?'}); not starting another")
    elif not args.no_heartbeat:
        pid, log_path = spawn_heartbeat(HEARTBEAT_INTERVAL_SECONDS, args.log_dir)
        heartbeat = {"pid": pid, "log": log_path,
                     "interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
                     "csv": os.path.join(PHASE2_DIR, "progress_log.csv")}
        print(f"  heartbeat started, pid {pid} -> {heartbeat['csv']} (hourly)")

    manifest = _merge_manifest({
        "launched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "settled": RS.SETTLED,
        "concurrency": {
            "max_concurrent_runs": RS.MAX_CONCURRENT,
            "joint_threads": RS.JOINT_THREADS,
            "max_other_concurrent": RS.MAX_OTHER_CONCURRENT,
            "other_threads": RS.OTHER_THREADS,
            "threads_resident": RS.CONCURRENT_THREADS,
        },
        "days_per_process": RS.DAYS_PER_PROCESS,
        "scaler_checks": _scaler_checks(preflight),
        "preflight": {
            "path": PREFLIGHT_JSON,
            "passed": bool(preflight and preflight.get("passed")),
            "checked_at": (preflight or {}).get("checked_at"),
        },
        "scheduler": {"pid": scheduler.pid, "log": scheduler_log,
                      "poll_seconds": args.poll},
        "queued": [r.run_id for r in launched],
        "skipped": [{"run_id": r.run_id,
                     "state": by_id.get(r.run_id, {}).get("state"),
                     "days_done": by_id.get(r.run_id, {}).get("days_done", 0)}
                    for r in skipped],
        "interpreter": windowless_python(),
        "runs": [],
        "heartbeat": heartbeat,
    })
    with open(MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    print(f"\nmanifest -> {MANIFEST}")
    print(f"progress  -> python dnn_status.py  (or --watch, or --json)")
    print(f"The scheduler and its runs are detached: closing this terminal "
          f"will not stop them.")
    return 0


def _admission_plan(runs):
    """How the scheduler will let ``runs`` in, as waves. For the dry run only.

    The real schedule is dynamic -- a slot frees the moment a run finishes -- so
    this is an illustration of the limit, not a promise about the order.
    """
    from dnn_dk1 import runs as RS

    # One joint-class run and up to MAX_OTHER_CONCURRENT single-threaded runs
    # are resident at a time, so each wave takes one from each queue. The second
    # joint run waits for the first to finish rather than sharing the slot.
    joint = [r for r in runs if r.heavy]
    others = [r for r in runs if not r.heavy]

    waves = []
    while joint or others:
        wave = joint[:1] + others[:RS.MAX_OTHER_CONCURRENT]
        joint = joint[1:]
        others = others[RS.MAX_OTHER_CONCURRENT:]
        waves.append(wave)
    return waves


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
