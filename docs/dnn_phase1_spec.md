# DNN phase 1 — build the three configurations and measure them

Spec for the coding agent. **Phase 1 ends with timings, not with a backtest.** Do not launch a
full run; the hyperparameter-search budget and ensemble size are chosen from the numbers this
phase produces.

Read `CLAUDE.md` and `SETUP.md` first. Python 3.11, venv outside OneDrive, `epftoolbox` from the
local `./epftoolbox` source tree.

---

## 1. What already exists and must not be re-implemented

- `run_lear_dk1_dk2.py` — the completed LEAR thesis run. Its preflight-assertion pattern, run
  manifest schema and directory naming are the templates to follow.
- `run_dnn_dk1.py` — single-zone DNN backtest against `epftoolbox`. Its checkpoint/resume
  semantics (`lear_dk1.backtest._load_checkpoint`), day-outer/seed-inner loop, per-day CSV
  checkpointing and per-seed timing files are all correct. **Extend this file's structure; do not
  fork a parallel implementation.**
- `lear_dk1.evaluate.evaluate_run` — the shared scorer. Every configuration must be scored by it
  so LEAR and DNN figures are comparable. Do not write a second evaluator.
- `data_cleaning_v2.ipynb` → cleaned parquet → per-zone CSVs. Cleaning happens once, there. The
  run scripts must continue to **error on NaN rather than impute**.

## 2. Zone set and dimensions — assert these, do not infer them

```
Z = [DK1, DK2, DE_LU, NL, NO2, SE3, SE4]      # 7 zones, order fixed and canonical
```

Z is the union of DK1's and DK2's direct interconnections plus the two focal zones. It is not a
tunable list.

Exogenous variables per zone (Lago layout: price lags at d-1, d-2, d-3, d-7; each exogenous at
d, d-1, d-7; all 24 hours):

| Zone | Exogenous | Feature block |
|---|---|---|
| DK1, DK2, DE_LU, NL, SE3, SE4 | load, wind, solar | 96 + 3×72 = **312** |
| NO2 | load, wind (no solar series exists) | 96 + 2×72 = **240** |

Resulting input dimensions (+1 day-of-week categorical):

| Configuration | Inputs | Outputs |
|---|---|---|
| DNN-own, all zones except NO2 | **313** | 24 |
| DNN-own, NO2 | **241** | 24 |
| DNN-wide (DK1 and DK2 alike) | **2113** | 24 |
| DNN-joint | **2113** | **168** |

DNN-wide and DNN-joint see the *same* 2113 inputs — both focal zones get all of Z, not only their
own neighbours. This is deliberate: wide→joint must change the output layer and nothing else, or
the decomposition is confounded. Do not "optimise" this by giving each focal zone only its
neighbours.

## 3. Data preparation

`DE_LU` and `NL` are present in `datasets/nordic_baltic_raw.parquet` and in the cleaned hourly
parquet, but have no per-zone CSV yet. Build them the same way the existing
`*_clean_load-wind-solar.csv` files were built, over the same span.

- DE_LU coverage starts 2018-09-30, which clears the 2019-09-29 burn-in requirement. Assert it.
- Panel span, DST handling and the 1463-day burn-in must match the LEAR run exactly. Reuse
  `local_day_panel.py`.

## 4. Code changes required

### 4.1 Multi-zone feature assembly

Assemble the 2113-column matrix from the per-zone cleaned CSVs, in the canonical Z order, with
deterministic column names of the form `<ZONE>_<block>_<lag>_h<hour>`. The column order must be
stable across runs and reproducible from the Z list alone.

### 4.2 Hyperparameter search space

`epftoolbox`'s DNN hyperopt hard-codes 11 binary feature-selection toggles, at the granularity of
(variable, day-set): 4 for the price lag-days (d-1, d-2, d-3, d-7), 2 exogenous × 3 day-sets, and
one for the calendar. Generalised naively to Z, that granularity gives
`7 zones × (4 + 3·n_exo) + 1` = **89 binaries**, i.e. a 2^89 block-selection space. No realistic
evaluation budget explores that; TPE would degenerate to random sampling.

**Adopted rule — block toggles apply only where Lago's design intended them:**

- **DNN-own**: the zone's own blocks are toggled at full Lago granularity. For a 3-exogenous zone
  this is 4 + 9 + 1 = **13 binaries**; for NO2 (2 exogenous) it is **11**, exactly the benchmark.
  DNN-own therefore *is* the Lago et al. (2021) DNN, unmodified.
- **DNN-wide and DNN-joint**: **no per-block binaries at all.** All 2113 inputs are always present
  and are pruned by the L1 penalty on the first-layer kernel, whose coefficient is already a tuned
  hyperparameter. Only the calendar toggle remains, so the search space is the 8 architecture
  hyperparameters + 1 = **9**.

This keeps DNN-wide and DNN-joint on an *identical* search space, so the wide→joint comparison —
the one the thesis turns on — is perfectly matched. Feature selection is not removed; it moves
from block-level (binary) to weight-level (L1), which is the finer and more appropriate instrument
at this input width.

Also widen the **neuron-count and L1 ranges** for the wide/joint configurations. The upstream
ranges were chosen for ~240 inputs; at 2113 they are likely badly scaled — in particular the L1
range must reach high enough to prune 1800 cross-zonal inputs. Widening the search *space* is
legitimate because the same procedure still runs for every configuration. Widening the search
*budget* for one configuration is not — see §6.

**Record per-zone first-layer weight magnitudes after training**, for every configuration. With
block toggles gone for wide/joint, this is what §5.5 reports instead of toggle states — which zones
the model actually gave weight to.

### 4.3 168-output head, with per-zone scaling

The output layer becomes 24 × |Z|, ordered by the canonical Z order then hour.

Targets are standardised **per zone** — median/MAD then asinh VST, each zone's own parameters —
before the network sees them, and inverted per zone on the way out. All 168 outputs are then
weighted **equally** in the loss.

These two decisions are one decision. The loss is computed on the transformed scale and summed
over outputs, so if a single scaler is fitted across all seven zones, the zones with the widest
price dispersion retain the largest transformed values, produce the largest errors, and dominate
the gradient. "Equal weighting" would then be equal in name only, and the network would spend its
capacity on whichever zone is noisiest rather than on the zones being reported.

Fit `scalerY` per zone; assert that each zone's transformed targets have comparable dispersion
before training begins.

### 4.4 Sub-vector scoring

DNN-joint is scored on each zone's own 24-column slice, passed to `lear_dk1.evaluate.evaluate_run`
one zone at a time. **Never** compute a pooled loss across zones for reporting — a pooled figure
is dominated by whichever zones happen to be easiest.

## 5. Preflight assertions

Follow the LEAR run's pattern: assert, record the result in the manifest, fail loudly. At minimum:

1. Every zone in Z has a cleaned CSV covering 2019-01-01 … 2025-09-30.
2. Column counts are exactly 313 / 241 / 2113 / 2113 as tabulated above, checked on the built
   matrix, not assumed.
3. DST transition days are identical across all zones' series (7 spring, 6 autumn over the span).
4. The test period is exactly 731 days, 2023-10-01 … 2025-09-30, identical for every zone.
5. The hyperopt validation window (363 days ending the hour before the test period) does not
   overlap the test period.
6. No NaN anywhere in any assembled matrix.
7. DNN-joint's output column order round-trips: inverse-transforming a known input reproduces the
   per-zone price scale (median |value| in the tens of EUR/MWh, not the single-digit asinh scale).

## 6. Invariants that must hold across all ten runs

The ten runs are: DNN-own × 7 zones, DNN-wide × {DK1, DK2}, DNN-joint × 1.

- **Identical** hyperopt evaluation budget `E` and seed count `S` for every run. An unequal budget
  between DNN-own and DNN-joint confounds the effect being measured.
- **Identical** calibration window: 4 years (`calibration_window=4`, i.e. 1456 days — the same as
  LEAR's longest window).
- **Identical** test period and recalibration cadence.
- Seeds fixed and recorded. Day-outer/seed-inner loop preserved, so an interruption always leaves
  a balanced ensemble.

## 7. The timing harness — the actual deliverable of phase 1

Add `run_dnn_timing.py`. It must **not** run a full backtest.

### 7.1 Thread sweep

On DNN-own/DK1, measure seconds per recalibration at `OMP_NUM_THREADS` ∈ {1, 2, 4, 8}, 3
recalibrations each, single seed. Report per-fit seconds and implied total throughput for
N concurrent processes at each setting. This decides the process × thread split; the working
hypothesis is 4 processes × 4 threads on this machine (8 physical / 16 logical cores), but
measure it.

Set `intra_op_parallelism_threads` and `inter_op_parallelism_threads` explicitly, plus
`OMP_NUM_THREADS`, before importing TensorFlow — not after.

### 7.2 Per-configuration timings

For each of DNN-own/DK1 (313 in), DNN-wide/DK1 (2113 in), DNN-joint (2113 in, 168 out):

- 20 hyperopt evaluations → mean and standard deviation of seconds per evaluation
- 5 recalibration days, 1 seed → mean and standard deviation of seconds per recalibration
- peak RSS

Report the standard deviations, not only means: hyperopt evaluation time varies substantially with
the sampled architecture, and a mean over 20 draws is a rough estimate.

### 7.3 Output

Write `experiments/timing/phase1_timings.json` and print a table giving, for candidate
`E ∈ {300, 1500}` and `S ∈ {2, 4}`, the projected wall-clock for:

- each individual run,
- all ten runs serially,
- all ten runs at the chosen process × thread concurrency.

That table is what the budget decision is made from.

## 8. Out of scope for phase 1

Full backtests. Chronos-2. Any change to the LEAR runs or to the cleaning notebook. Any change to
the shared evaluator's metrics.
