"""
Finding the phase-2 processes, by command line rather than by recorded PID.

Both the launcher and the status tool need to answer "is this run alive?", and
they must answer it the same way. Neither can rely on a PID from the manifest
any more: a run is now a *sequence* of worker processes, each forecasting about
a hundred days and exiting so TensorFlow's heap goes back to the operating
system. The PID changes every chunk; the run does not.

So liveness is a property of the command line. ``live_workers`` finds the worker
for a run whatever chunk it is on, and ``live_scheduler`` finds the process that
restarts them.

Every function returns ``(pids, certain)``. ``certain`` is False when the scan
could not be completed -- psutil missing, or a process this user may not
inspect. A caller deciding whether to *start* something must treat that as
"possibly alive": refusing to start is recoverable, two processes writing one
checkpoint is not.
"""

from __future__ import annotations

import os

WORKER_SCRIPT = "run_dnn_dk1.py"
SCHEDULER_FLAG = "--schedule"
LAUNCHER_SCRIPT = "run_dnn_all.py"
HEARTBEAT_SCRIPT = "dnn_status.py"


# Only a Python process can be one of ours. Without this, a shell whose command
# line merely *mentions* run_dnn_dk1.py -- a grep, a status query, this very
# comment being searched for -- would be counted as a live worker and the
# launcher would refuse to start a run that is not running.
_PYTHON_NAMES = ("python.exe", "pythonw.exe", "python", "python3", "pythonw")


def _iter_cmdlines():
    """``(pid, argv)`` for every Python process we are allowed to look at.

    Yields a final ``(None, None)`` sentinel if the scan was incomplete, so the
    caller can tell "nothing is running" from "could not tell".
    """
    try:
        import psutil
    except ImportError:
        yield None, None
        return

    complete = True
    for process in psutil.process_iter(["cmdline", "name"]):
        try:
            argv = process.info["cmdline"] or []
            name = (process.info["name"] or "").lower()
        except psutil.Error:
            complete = False
            continue
        if argv and name in _PYTHON_NAMES:
            yield process.pid, argv
    if not complete:
        yield None, None


def has_pair(argv, flag, value):
    """``--flag value`` as two adjacent tokens, or as ``--flag=value``.

    Token-exact on purpose: matching the joined command line would let an
    ``--out-dir`` containing "DK1" satisfy ``--zone DK1``.
    """
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv) and argv[i + 1] == value:
            return True
        if token == f"{flag}={value}":
            return True
    return False


def _scan(match):
    pids, certain = [], True
    for pid, argv in _iter_cmdlines():
        if pid is None:
            certain = False
            continue
        if match(argv):
            pids.append(pid)
    return pids, certain


def live_workers(run):
    """Worker processes forecasting ``run``, on whichever chunk they are on."""
    def match(argv):
        if WORKER_SCRIPT not in [os.path.basename(a) for a in argv]:
            return False
        if run.config == "joint":
            return has_pair(argv, "--config", "joint")
        return (has_pair(argv, "--config", run.config)
                and has_pair(argv, "--zone", run.focal))

    return _scan(match)


def live_scheduler():
    """The scheduler process, if one is running."""
    def match(argv):
        names = [os.path.basename(a) for a in argv]
        return LAUNCHER_SCRIPT in names and SCHEDULER_FLAG in argv

    return _scan(match)


def live_heartbeat():
    """The hourly progress reporter, if one is running."""
    def match(argv):
        names = [os.path.basename(a) for a in argv]
        return HEARTBEAT_SCRIPT in names and "--heartbeat" in argv

    return _scan(match)


def is_alive(pid, expect=WORKER_SCRIPT):
    """Is ``pid`` a live process whose command line contains ``expect``?

    ``True`` / ``False`` / ``None`` for "cannot tell". The command line is
    checked as well as the number, because PIDs are reused and a recycled one
    would otherwise make a dead run look healthy -- or block a relaunch forever.
    """
    if not pid:
        return False
    try:
        import psutil
    except ImportError:
        return None
    try:
        process = psutil.Process(int(pid))
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return False
        command = " ".join(process.cmdline())
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        return None
    return expect in command


def started_at(pids):
    """The earliest start time among ``pids``, as a POSIX timestamp, or None."""
    try:
        import psutil
    except ImportError:
        return None
    times = []
    for pid in pids or []:
        try:
            times.append(float(psutil.Process(int(pid)).create_time()))
        except Exception:
            continue
    return min(times) if times else None


def rss_bytes(pids):
    """Summed resident set size of ``pids``, or None if it cannot be read.

    A run is a stub plus the interpreter it re-execs, so its footprint is the
    sum over the processes that carry its command line.
    """
    try:
        import psutil
    except ImportError:
        return None
    total, seen = 0, False
    for pid in pids:
        try:
            total += int(psutil.Process(int(pid)).memory_info().rss)
            seen = True
        except Exception:
            continue
    return total if seen else None
