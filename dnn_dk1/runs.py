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

SETTLED = {
    "hyperopt_evaluations": MAX_EVALS,
    "seeds": list(SEEDS),
    "recalibration": "daily",
    "calibration_window_years": CALIBRATION_YEARS,
    "calibration_window_days": CALIBRATION_YEARS * 364,
    "test_start": str(BEGIN_TEST.date()),
    "test_end": str(END_TEST.date()),
    "test_days": TEST_DAYS,
    "hyperopt_window": "the ~364 days before the test period",
    "reference": "Olivares et al. (2023) p. 895 footnote; Lago et al. (2021)",
}


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


# Ordered as the launcher starts them: joint first -- it is the critical path at
# roughly 21 hours and must never be starved -- then the two wide runs, then the
# seven own runs. Thread counts come from the phase 1 sweep: threading scales
# badly here (8x threads buys 1.31x on joint, 1.04x on own), so the wider
# networks get what little they can use and the narrow ones get one each.
# 4 + 2 + 2 + 7 = 15 threads on 16 logical cores.
RUNS: tuple[Run, ...] = (
    Run("joint", "joint", None, threads=4),
    Run("wide_DK1", "wide", "DK1", threads=2),
    Run("wide_DK2", "wide", "DK2", threads=2),
    *(Run(f"own_{zone}", "own", zone, threads=1) for zone in Z.ZONES),
)

RUNS_BY_ID = {run.run_id: run for run in RUNS}

TOTAL_THREADS = sum(run.threads for run in RUNS)


def get(run_id: str) -> Run:
    if run_id not in RUNS_BY_ID:
        raise KeyError(
            f"unknown run {run_id!r}; the ten are {', '.join(RUNS_BY_ID)}")
    return RUNS_BY_ID[run_id]
