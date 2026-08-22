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

```bash
python run_lear_dk1.py --begin-test 2023-10-03 --end-test 2025-09-30
```

| Flag | Meaning |
|---|---|
| `--dataset` | Dataset name; reads `<datasets-dir>/<dataset>.csv` (default `DK1`) |
| `--begin-test`, `--end-test` | Inclusive test days, `YYYY-MM-DD` |
| `--windows` | Comma-separated calibration windows in days (default `364,728,1092,1456`) |
| `--run-name` | Run directory name — **reuse it to resume** |
| `--smoke` | Ten days on the smallest window, into a separate directory |

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
