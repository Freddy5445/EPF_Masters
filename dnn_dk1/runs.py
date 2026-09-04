"""
The eleven phase-2 runs, and the settled parameters they all share.

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

# 1500 TPE iterations: Lago et al. (2021), the budget the benchmark DNN this
# thesis reproduces was itself searched with. Olivares et al. (2023) p. 895
# footnote report 1000, which is the floor rather than the target.
#
# This is the single most expensive parameter in the design, and it is fixed
# rather than tunable for a reason that has nothing to do with cost: an unequal
# search budget between configurations would confound the effect the eleven runs
# exist to measure. It is 1500 for every run or it is not a comparison.
MAX_EVALS = 1500

# The ensemble averages networks differing only in their random seed.
SEEDS = (1, 2, 3, 4)

NLAYERS = 2
CALIBRATION_YEARS = Z.CALIBRATION_YEARS       # 4 years = 1456 days
BEGIN_TEST = Z.BEGIN_TEST                     # 2023-10-01
END_TEST = Z.END_TEST                         # 2025-09-30
TEST_DAYS = Z.TEST_DAYS_EXPECTED              # 731

# The two joint configurations. Both take every zone in Z on the input side and
# more than one zone on the output side; they differ only in how many, which is
# the whole point of having both -- ``wide -> joint_dk`` isolates the output
# effect on the two focal zones, and ``joint_dk -> joint`` asks whether the five
# less-related auxiliary zones add to it or dilute it.
#
# They are also the two expensive runs, so the scheduler treats them as one
# class: at most one of them resident at a time, on JOINT_THREADS threads. See
# `Run.heavy`.
JOINT_CONFIGS = ("joint", "joint_dk")


def out_zones_for(config, focal):
    """Which zones a configuration's output layer covers.

    Defined here rather than in the run script because ``run_dnn_dk1.py``,
    ``run_dnn_all.py``, ``dnn_status.py`` and ``run_dnn_preflight.py`` all need
    the same answer, and a second copy of it is how a run gets launched with one
    output width and scored against another.
    """
    if config == "joint":
        return Z.ZONES
    if config == "joint_dk":
        return Z.FOCAL_ZONES
    return (focal,)


# Recalibration cadence, in days. The network is refitted every RECALIBRATION_DAYS
# days and the fit is reused for the days in between, so the 731-day test period
# costs 105 fits rather than 731.
#
# The test period itself does not change: all 731 days are still forecast, and
# the reported MAE, rMAE and DM/GW tests still cover exactly the days the LEAR
# benchmark covers. Only the *weights* age -- by at most six days -- and never
# the inputs, which are rebuilt for every forecast day from data known at that
# day's gate closure. A week-old model forecasting with today's features is what
# "recalibrate weekly" means in the literature; it is not a shortened test period
# and it is not leakage in either direction.
#
# Why at all: this is the one parameter in the design chosen for compute rather
# than argued from the literature, and it is worth saying so plainly rather than
# dressing it up. The cheap partial defence is available and should be taken:
# DNN-own/DK1 at every-2-days is already complete, so re-running that single
# configuration weekly costs about two core-hours and turns "we did this because
# it was faster" into a measured accuracy difference that section 4.2.2 can report.
RECALIBRATION_DAYS = 7
REFITS = -(-TEST_DAYS // RECALIBRATION_DAYS)  # ceil -> 105

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
# one joint-class run on four threads, and up to four single-threaded runs
# beside it. That is 4 + 4 = 8 threads on 8 physical cores, and the rest queue.
# There are two joint-class runs and they share the one slot -- see RUNS.
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
        return out_zones_for(self.config, self.focal)

    @property
    def heavy(self) -> bool:
        """Whether the scheduler counts this against the single joint slot."""
        return self.config in JOINT_CONFIGS

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


# Priority order for the scheduler: the two joint runs first -- joint is the
# critical path and must never queue behind cheaper work, and joint_dk is the
# next most expensive -- then the two wide runs, then the seven own runs.
#
# The two joint runs share one admission slot rather than getting one each,
# which serialises them. That is deliberate: giving joint_dk four threads of its
# own would put 12 threads on 8 physical cores, and the phase 1 sweep showed
# exactly what oversubscription costs here. Serialised on four threads is worse
# for joint_dk's own wall-clock than four threads in parallel would be, and
# better than either run gets under contention.
#
# Thread counts are no longer "as many as the sweep allows". The phase 1 sweep
# measured one process on an idle machine; running ten at once on eight physical
# cores cost joint a factor of nine, which no thread setting recovers. So joint
# takes four threads and every other run takes one, and the scheduler admits at
# most JOINT + 4 others at a time: 8 threads on 8 physical cores, nothing
# oversubscribed, the rest queued.
RUNS: tuple[Run, ...] = (
    Run("joint", "joint", None, threads=JOINT_THREADS),
    Run("joint_dk", "joint_dk", None, threads=JOINT_THREADS),
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
            f"unknown run {run_id!r}; the eleven are {', '.join(RUNS_BY_ID)}")
    return RUNS_BY_ID[run_id]
