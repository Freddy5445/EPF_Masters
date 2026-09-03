# DNN phase 2 — launch the ten runs, with progress visible at any time

Spec for the coding agent. Read `docs/dnn_phase1_spec.md` first; everything there still holds.
Phase 1 is complete and its corrections (1969 inputs, no block toggles for wide/joint, the zero-MAD
guard) are already in the code.

---

## 0. Before launching — two changes and a pre-flight

Findings from a read of `zones.py`, `forecaster.py`, `model.py`, `scaling_compat.py` and
`hyperopt.py`. Everything else reviewed was sound: no leakage in the feature assembly, scalers
fitted on training data only, output ordering consistent forward and inverse, no double inverse
transform, per-(seed, day) seeding, zero-MAD guard on both paths, `clear_session` per day.

### 0.1 Gate the hyperopt window — mandatory

`optimize_multizone` (and `optimize`) take `begin_test_date` / `end_test_date` and use them to
define both the training days and the evaluation days of the *search*. These must be the ~364 days
**before** `zones.BEGIN_TEST`, never the real test period.

Add an assertion inside the optimiser that refuses to run when the search window touches
`BEGIN_TEST` or later. This is currently the only invariant in the pipeline enforced by convention
rather than by code, and it is the most dangerous one: if it is ever violated the study is invalid
**and the results look excellent**.

### 0.2 Normalise the early-stopping MAE per zone

`DNNModel._obtain_metrics` inverse-transforms the validation targets and takes
`np.mean(MAE(Y, Ybar))` pooled across all output columns, on the price scale. The loss is on the
standardised scale and is balanced across zones; this MAE is not. Upstream's rule keeps weights
whenever *either* metric improves, so the pooled MAE lets the most expensive zones drive weight
retention — the imbalance per-zone target scaling exists to remove, surviving in the stopping rule.

Divide each zone's MAE by that zone's normaliser before averaging, reusing
`hyperopt._scale_normalisers`. For a single output zone this is division by one constant and cannot
change any decision, so DNN-own stays bit-identical to upstream; only DNN-joint is affected.

### 0.3 Pre-flight — all four must pass, reported as a table

1. The 0.1 assertion fires when given an overlapping window.
2. **Determinism**: forecast the same day twice in different process orders; assert bit-identical.
3. **Round-trip**: `PerZoneScaler.inverse_transform(fit_transform(Y))` recovers `Y`.
4. **Smoke all ten configurations** — 5 hyperopt evaluations, 3 test days, 1 seed. Seven of the ten
   have never executed. Each must start, write a per-day checkpoint, write a timing row, and be
   scored by `lear_dk1.evaluate.evaluate_run`.

Record `equals_pooled_fit` and `transformed_dispersion` in the launch manifest.

Stop after the pre-flight and the dry run. The real launch is the user's call.

## 1. Settled parameters — do not vary these

| | |
|---|---|
| Hyperparameter search | **300 TPE iterations** (Olivares et al. 2023, p. 895 footnote) |
| Ensemble | **4 seeds** |
| Recalibration | **daily** |
| Calibration window | 4 years (1456 days) |
| Test period | 2023-10-01 … 2025-09-30, 731 days |
| Hyperopt validation | the ~364 days before the test period |

Identical for all ten runs. An unequal budget between configurations would confound the effect
being measured.

## 2. The ten runs

| ID | Config | Target(s) | Inputs | Outputs | Threads |
|---|---|---|---|---|---|
| `joint` | DNN-joint | all 7 zones | 1969 | 168 | **4** |
| `wide_DK1` | DNN-wide | DK1 | 1969 | 24 | **2** |
| `wide_DK2` | DNN-wide | DK2 | 1969 | 24 | **2** |
| `own_DK1` | DNN-own | DK1 | 313 | 24 | 1 |
| `own_DK2` | DNN-own | DK2 | 313 | 24 | 1 |
| `own_DE_LU` | DNN-own | DE_LU | 313 | 24 | 1 |
| `own_NL` | DNN-own | NL | 313 | 24 | 1 |
| `own_NO2` | DNN-own | NO2 | 241 | 24 | 1 |
| `own_SE3` | DNN-own | SE3 | 241 | 24 | 1 |
| `own_SE4` | DNN-own | SE4 | 241 | 24 | 1 |

Machine: 8 physical / 16 logical cores, 32 GB. Total 15 threads if all run at once; combined peak
RSS ~10.4 GB. `joint` is the critical path (~21 h) — start it first and never starve it.

Threading scales poorly here (8× threads buys 1.3× on joint, 1.04× on own), so more processes with
fewer threads each is correct. Set `OMP_NUM_THREADS`, `intra_op_parallelism_threads` and
`inter_op_parallelism_threads` **before importing TensorFlow**, not after.

## 3. Smoke all ten first — mandatory

Before the real launch, run every one of the ten with 5 hyperopt evaluations, 3 test days, 1 seed.
Seven of the ten have never executed. Confirm for each: it starts, writes a per-day checkpoint,
writes a timing row, and is scored by `lear_dk1.evaluate.evaluate_run` without error.

Report the ten results as a table. Do not launch the real runs until all ten pass.

## 4. Launcher — `run_dnn_all.py`

- Spawns each run as an **independent detached process** with its own log file under
  `experiments/logs/<run_id>.log`. Closing the terminal, or the agent session, must not kill them.
  On Windows use `Start-Process`-equivalent detachment (`CREATE_NEW_PROCESS_GROUP` /
  `DETACHED_PROCESS`), not a child that dies with the parent.
- Starts `joint` first, then the two `wide`, then the seven `own`.
- One run crashing must not affect the others.
- `--dry-run` prints the exact command line for each run without executing.
- Writes `experiments/dnn_phase2/launch_manifest.json`: run IDs, commands, thread counts, PIDs,
  start times, and the settled parameters above.
- **Idempotent.** Re-running it must resume rather than restart: skip runs already complete, and
  for partial runs rely on the existing `_load_checkpoint` resume path. Never silently discard
  finished days.

## 5. Status — `dnn_status.py`

This is the deliverable the user cares about most. It must work at any moment, whether the runs are
alive, finished, or dead, and must read the artifacts already being written rather than requiring
the runs to report anything new.

    python dnn_status.py

Prints one row per run:

| column | meaning |
|---|---|
| run | run ID |
| state | `running` / `done` / `stalled` / `queued` / `failed` |
| days | completed test days out of 731 |
| pct | percentage |
| seeds | seeds complete on the most recent day |
| elapsed | wall-clock since that run started |
| s/day | mean seconds per day over the **last 20 days only** |
| eta | projected finish from that recent rate |

Then a footer: overall percentage across all ten, total elapsed, and projected time until the last
run finishes.

Details that matter:

- **Use a recent-window rate, not a whole-run average.** Early days run faster before contention
  builds; a whole-run mean gives an optimistic ETA.
- `stalled` = process alive but no new completed day for 30 minutes. `failed` = process gone with
  days remaining. Both must be visible without reading logs.
- Add `--watch` to refresh every 60 seconds.
- Add `--json` for machine-readable output (used by the hourly reporter below).
- Must never write to, lock, or otherwise interfere with a running backtest — read-only.

### 5.1 Hourly heartbeat

The launcher also starts a lightweight reporter process that appends one row per run per hour to
`experiments/dnn_phase2/progress_log.csv`:

    timestamp, run_id, state, days_done, seeds_done, seconds_per_day_recent, eta_utc

This costs nothing and does two things a point-in-time status cannot: it preserves the *history*,
so a run that slows down is visible as a trend rather than a worse ETA; and it lets an external
process report progress without touching the runs.

The reporter must be independent of the backtests — if it dies, the runs continue, and restarting
it must not disturb anything.

## 6. When a run finishes

Score it with `lear_dk1.evaluate.evaluate_run` as each run completes, rather than waiting for all
ten. `joint` is sliced per zone into `zone_<Z>/` subdirectories and scored one zone at a time,
never pooled.

Write `experiments/dnn_phase2/accuracy_summary.csv` incrementally in the same schema as
`experiments/lear_dk1_dk2_thesis/accuracy_summary.csv`, so DNN and LEAR results are directly
comparable and partial results are usable before the set completes.

## 7. Out of scope

Chronos-2. Any change to the LEAR runs, the cleaning notebook, the shared evaluator's metrics, or
the vendored `epftoolbox` tree. Any change to the settled parameters in §1.
