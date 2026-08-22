# lear_dk1

Runs the `epftoolbox` LEAR model on ENTSO-E bidding-zone data, with daily
recalibration and a resumable, instrumented backtest.

Nothing in `epftoolbox/` is modified. The incompatibilities that would otherwise
stop LEAR from running on this project's pinned stack are worked around here.

## Usage

Build the dataset first (needs an ENTSO-E token — see `entsoe_tp/README.md`):

```
python -m entsoe_tp.build_dataset --zone DK1 --start 2015-01-05 --end 2025-09-30 --exog load-wind-solar
```

(Commands are written on one line throughout: this project is developed from
PowerShell, where `\` is not a line continuation and silently truncates the
command.)

Validate the pipeline on ten days before committing to a multi-hour run:

```bash
python run_lear_dk1.py --smoke
```

Then the full ensemble:

```
python run_lear_dk1.py
```

| Flag | Meaning |
|---|---|
| `--dataset` | Dataset name; reads `<datasets-dir>/<dataset>.csv` (default `DK1`) |
| `--data-start` | Ignore data before this day (default `2015-01-07`) |
| `--begin-test`, `--end-test` | Inclusive test days (default `2023-04-11` to `2025-04-07`) |
| `--windows` | Comma-separated calibration windows in days (default `364,728,1092,1456`) |
| `--max-linear` | Longest gap filled by interpolation, in hours (default 3) |
| `--no-impute` | Fail on missing values instead of filling them |
| `--run-name` | Run directory name — **reuse it to resume** |
| `--smoke` | Ten days on the smallest window, into a separate directory |

The end date defaults to 2025-04-07 because DK1 day-ahead moved to 15-minute
market time units on 2025-04-08, and the start to 2015-01-07 because ENTSO-E
coverage before that is too sparse to be worth imputing.

## Missing values

`build_dataset --allow-gaps` writes what the platform published and leaves the
rest as NaN. LEAR cannot be fitted on NaN, so gaps are filled **here**, at model
time, where the method is a stated choice rather than something baked invisibly
into the data.

**Every method is causal**: a value at time *t* is derived only from observations
strictly before *t*. This is not a detail. A backtest simulates forecasting day
D knowing only what was available beforehand, so a value imputed from day D+7
leaks future information into the training set — and where a gap falls inside the
test period, into the very target the model is scored against. Interpolating
across a gap and taking a median over the whole series both do this; neither is
used.

| Gap | Method | Why |
|---|---|---|
| Up to `--max-linear` hours | Carry last observation forward | Adjacent hours are highly correlated, and unlike interpolation it needs nothing from the far side |
| Longer | Same hour, an **earlier** week (−7d, −14d, …) | Keeps the daily shape and the weekday/weekend split; carrying one value forward for days would flatten both |
| No earlier week has that hour | Same hour, previous day | In practice only the first weeks of a series |
| Anything left | Expanding median for that hour of day, past only | Last resort |
| No earlier data at all | **Left as NaN and trimmed** | Cannot be filled causally; inventing it would be indistinguishable from look-ahead |

A quick sanity check of the causality claim: on a series whose level jumps from
50 to 500 exactly where a gap ends, this fills the gap with values up to 75 —
while linear interpolation reaches 473 and a same-hour-*next*-week fill reaches
525. Both of those have leaked the future.

Every filled value is counted and attributed to the method that produced it,
printed at run start and recorded in `run_metadata.json` under `imputation`, so a
run can state exactly how much of its input was imputed and how.

## Following a long run

The full ensemble takes hours. Every day prints its own time, the running
average, elapsed wall time, an ETA and the running MAE:

```
  [cw1092] 143/728  2024-02-23   5.8s  avg   5.9s  elapsed 0:14:04  ETA 0:57:31  MAE 8.412
```

Forecasts and timings are flushed **after every day**, so an interrupted run
loses at most one day. Re-running the same command resumes: completed days are
skipped and the ETA reflects only what is left.

## Outputs

Each run writes to `experiments/<run-name>/`:

| File | Contents |
|---|---|
| `forecasts_cw<N>.csv` | Forecast prices, one row per day, columns `h0`..`h23` |
| `timings_cw<N>.csv` | Per-day wall time, training-set size, feature count |
| `run_metadata.json` | Config, environment, and per-window timing summary |

`run_metadata.json` records the machine (platform, CPU count) and library
versions alongside the timings, because runtime is only comparable across models
measured on the same hardware. Timing rows carry a `run_id` so measurements from
a resumed or repeated run stay attributable.

## What this works around

**`LEAR.predict` crashes on NumPy 2.** Upstream does
`Yp[h] = self.models[h].predict(X)`, assigning a size-1 array into a scalar slot.
NumPy 1.25 deprecated that; 2.0 made it a `ValueError`. `LEARCompat` overrides
`predict` to call `.item()`. The arithmetic is otherwise untouched, so forecasts
match what upstream produces on NumPy 1.x.

**`evaluate_lear_in_test_dataset` uses `np.NaN`**, removed in NumPy 2.0. It is
unusable, so `backtest.py` reimplements the same recalibration protocol rather
than patching it.

**Importing LEAR drags in TensorFlow.** `epftoolbox/models/__init__.py` imports
the DNN modules, which do `import tensorflow.keras as kr` — a layout Keras 3 no
longer provides. `compat.load_lear_class()` loads `_lear.py` directly from its
path so the package `__init__` never runs.

**A constant feature makes the scaler emit NaN.** LEAR's asinh-median
("Invariant") scaler divides by the median absolute deviation. For a feature
that never varies, MAD is 0 and `data - median` is 0, so upstream computes
`0 / 0` and fills the column with NaN — which `LassoLarsIC` then rejects with a
message recommending imputer pipelines, pointing away from the real cause.

This is not hypothetical. LEAR builds one feature per (hour, lag), and a solar
generation forecast is **exactly zero every night, year-round**. A DK1 dataset
with solar as a separate exogenous input therefore has 39 constant columns —
13 night hours × 3 lags (D, D−1, D−7) — and upstream LEAR cannot be fitted on it
at all. `LEARCompat.recalibrate` substitutes 1 for a zero MAD, mapping such a
column to all zeros: a feature with no variation carries no information, LASSO
gives it a zero coefficient, and the inverse transform still recovers the
constant. Columns that do vary are scaled bit-identically to upstream. The count
is reported per window as `constant_features` in `run_metadata.json`.

**Short calibration windows are impossible.** `LassoLarsIC` refuses to fit when
samples < features, and LEAR has `96 + 7 + 72·n_exogenous` features:

| Exogenous inputs | Features | Minimum window |
|---|---|---|
| 1 | 175 | 183 days |
| 2 | 247 | 255 days |
| 3 | 319 | 327 days |

This rules out the 56- and 84-day windows of the original LEAR ensemble
(Lago et al., 2021), which is why the default ensemble starts at 364 days.
`run_lear_dk1.py` checks this before starting work rather than failing partway
through a long run.

**`read_data` silently shifts dates.** It parses date arguments with
`dayfirst=True`, so the ISO string `"2023-10-03"` is read as 3 October and
becomes `2023-03-10` — a seven-month shift, with no error. Datetime objects pass
through untouched, so this code never hands it raw strings. There is a
regression test for this in `tests/test_lear_dk1.py`.

## Tests

Offline and fast; they deliberately do not fit a model (the numerical path is
covered by `--smoke`):

```bash
python -m unittest discover -s tests
```
