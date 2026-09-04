"""
The ten phase-2 runs, and the settled parameters they all share.

One definition, imported by ``run_dnn_preflight.py``, ``run_dnn_all.py`` and
``dnn_status.py``, so the launcher cannot start a run the status tool does not
know how to find, and neither can drift from what was smoke-tested.

The parameters in :data:`SETTLED` are not tunable. An unequal hyperparameter
budget or ensemble size between configurations would confound the very effect the
three configurations exist to measure, so they are stated once, here, and every
command line is derived from them.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import zones as Z

# --- Settled parameters (spec section 1) -----------------------------------

# 300 TPE iterations: Olivares et al. (2023), p. 895 footnote. Phase 1 measured
# 25-33 s per evaluation with a standard deviation larger than the mean, so this
# is also the budget the timing table supports.
MAX_EVALS = 300

# The ensemble averages networks differing only in their random seed.
SEEDS = (1, 2, 3, 4)

NLAYERS = 2
CALIBRATION_YEARS = Z.CALIBRATION_YEARS       # 4 years = 1456 days
BEGIN_TEST = Z.BEGIN_TEST                     # 2023-10-01
END_TEST = Z.END_TEST                         # 2025-09-30
TEST_DAYS = Z.TEST_DAYS_EXPECTED              # 731

# Recalibration cadence, in days. The network is refitted every RECALIBRATION_DAYS
# days and the fit is reused for the days in between, so the 731-day test period
# costs 366 fits rather than 731.
#
# The test period itself does not change: all 731 days are still forecast, and
# the reported MAE, rMAE and DM/GW tests still cover exactly the days the LEAR
# benchmark covers. Only the *weights* age -- by at most one day -- and never the
# inputs, which are rebuilt for every forecast day from data known at that day's
# gate closure. A day-old model forecasting with today's features is what
# "recalibrate every two days" means in the literature; it is not a shortened
# test period and it is not leakage in either direction.
#
# Why at all: the first launch measured joint at ~770 s per forecast day against
# phase 1's 85, and projected 156 hours. Halving the fits halves that, and the
# fits are essentially the whole cost -- a reused-model day is a forward pass.
RECALIBRATION_DAYS = 2
REFITS = -(-TEST_DAYS // RECALIBRATION_DAYS)  # ceil -> 366

# A worker exits cleanly after this many newly forecast days and is restarted by
# the scheduler, resuming from its own checkpoint. TensorFlow's heap grows across
# hundreds of model builds and clear_session() does not give it all back; the
# first launch died on an ArrayMemoryError after ~6 hours. Phase 1 measured peak
# RSS over five recalibrations, which is far too few to see the trend.
DAYS_PER_PROCESS = 100

# How often a worker samples its own resident set size, in newly forecast days.
RSS_SAMPLE_DAYS = 25

# Concurrency. Ten processes on eight physical cores is what made joint 9x
# slower than it was measured alone, so the scheduler runs at most five at once:
# joint on four threads, and up to four single-threaded runs beside it. That is
# 4 + 4 = 8 threads on 8 physical cores, and the rest queue.
JOINT_THREADS = 4
OTHER_THREADS = 1
MAX_OTHER_CONCURRENT = 4
MAX_CONCURRENT = 1 + MAX_OTHER_CONCURRENT

SETTLED = {
    "hyperopt_evaluations": MAX_EVALS,
    "seeds": list(SEEDS),
    "recalibration": f"every {RECALIBRATION_DAYS} days ({REFITS} fits over "
                     f"{TEST_DAYS} forecast days)",
    "recalibration_days": RECALIBRATION_DAYS,
    "refits": REFITS,
    "calibration_window_years": CALIBRATION_YEARS,
    "calibration_window_days": CALIBRATION_YEARS * 364,
    "test_start": str(BEGIN_TEST.date()),
    "test_end": str(END_TEST.date()),
    "test_days": TEST_DAYS,
    "days_per_process": DAYS_PER_PROCESS,
    "max_concurrent_runs": MAX_CONCURRENT,
    "hyperopt_window": "the ~364 days before the test period",
    "reference": "Olivares et al. (2023) p. 895 footnote; Lago et al. (2021)",
}


def refit_days(begin=BEGIN_TEST, end=END_TEST, cadence=RECALIBRATION_DAYS):
    """The days on which the network is refitted.

    Counted from ``begin`` on an absolute grid, deliberately: which days get a
    fresh fit must not depend on where a process happened to start. A worker
    that resumes mid-run, or a chunk boundary, would otherwise shift the whole
    schedule and the resumed run would not reproduce the uninterrupted one.
    """
    import pandas as pd

    days = pd.date_range(begin, end, freq="D")
    return days[::cadence]


def days_served_by(refit_day, begin=BEGIN_TEST, end=END_TEST,
                   cadence=RECALIBRATION_DAYS):
    """The forecast days one fit covers: the refit day and the ``cadence-1`` after."""
    import pandas as pd

    last = min(pd.Timestamp(refit_day) + pd.Timedelta(days=cadence - 1),
               pd.Timestamp(end))
    return pd.date_range(refit_day, last, freq="D")


def refit_day_for(day, begin=BEGIN_TEST, cadence=RECALIBRATION_DAYS):
    """Which refit a given forecast day is served by."""
    import pandas as pd

    offset = (pd.Timestamp(day) - pd.Timestamp(begin)).days
    return pd.Timestamp(begin) + pd.Timedelta(days=(offset // cadence) * cadence)


@dataclass(frozen=True)
class Run:
    """One of the ten runs."""

    run_id: str
    config: str          # "own" | "wide" | "joint"
    zone: str | None     # focal zone; None for joint, which targets all of Z
    threads: int

    @property
    def focal(self) -> str:
        """The zone the run script needs on the command line."""
        return self.zone or Z.ZONES[0]

    @property
    def out_zones(self) -> tuple[str, ...]:
        return Z.ZONES if self.config == "joint" else (self.focal,)

    @property
    def n_inputs(self) -> int:
        if self.config == "own":
            return Z.own_input_width(self.focal)
        return Z.input_width(Z.ZONES)

    @property
    def n_outputs(self) -> int:
        return 24 * len(self.out_zones)

    @property
    def label(self) -> str:
        return f"DNN-{self.config}" + (f"/{self.zone}" if self.zone else "")


# Priority order for the scheduler: joint first -- it is the critical path and
# must never queue behind cheaper work -- then the two wide runs, then the seven
# own runs.
#
# Thread counts are no longer "as many as the sweep allows". The phase 1 sweep
# measured one process on an idle machine; running ten at once on eight physical
# cores cost joint a factor of nine, which no thread setting recovers. So joint
# takes four threads and every other run takes one, and the scheduler admits at
# most JOINT + 4 others at a time: 8 threads on 8 physical cores, nothing
# oversubscribed, the rest queued.
RUNS: tuple[Run, ...] = (
    Run("joint", "joint", None, threads=JOINT_THREADS),
    Run("wide_DK1", "wide", "DK1", threads=OTHER_THREADS),
    Run("wide_DK2", "wide", "DK2", threads=OTHER_THREADS),
    *(Run(f"own_{zone}", "own", zone, threads=OTHER_THREADS) for zone in Z.ZONES),
)

RUNS_BY_ID = {run.run_id: run for run in RUNS}

TOTAL_THREADS = sum(run.threads for run in RUNS)

# What is actually resident at once, which is the number that matters.
CONCURRENT_THREADS = JOINT_THREADS + MAX_OTHER_CONCURRENT * OTHER_THREADS


def get(run_id: str) -> Run:
    if run_id not in RUNS_BY_ID:
        raise KeyError(
            f"unknown run {run_id!r}; the ten are {', '.join(RUNS_BY_ID)}")
    return RUNS_BY_ID[run_id]
