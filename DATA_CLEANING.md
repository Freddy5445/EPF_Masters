# Data cleaning

Everything that happens to the data between the ENTSO-E API and a model lives in
**`data_cleaning.ipynb`**. Nothing downstream cleans anything: `run_lear_dk1.py`
and `run_dnn_dk1.py` refuse to run on a dataset with gaps and say to re-run the
notebook.

That is the point. Two models are being compared, and if each held its own copy
of the imputation rules those copies would eventually disagree — at which point a
difference between LEAR's and the DNN's scores could be a difference in data
handling rather than in the models. One implementation, run once, before either
model sees anything, makes that impossible rather than unlikely.

---

## The pipeline

```
ENTSO-E API
   │  entsoe_tp/            acquisition only — nothing invented
   ▼
nordic_baltic_raw.parquet   long, UTC, mixed resolutions, duplicates, gaps
   │  data_cleaning.ipynb   sections 1–6: structural cleaning (UTC)
   │                        section 7:    the model-ready panel (local time)
   ▼
nordic_baltic_clean_hourly.parquet
   │  run_lear_from_clean.py   projection only — select, sum, order
   ▼
<ZONE>_clean_<layout>.csv   what epftoolbox reads
```

The split matters. Sections 1–6 work in **UTC**, where every day has 24 hours and
DST does not exist. Section 7 converts to local market time, which is where the
awkward parts appear.

---

## Sections 1–6: structural cleaning

### Which zones

The download covers the eleven Nordic bidding zones, Finland, and **DE-LU** —
Germany-Luxembourg, the large thermal market the Nordic zones are coupled to,
captured on the same footing as the rest rather than as an afterthought. Finland
is dropped during cleaning; DE-LU is kept.

**The Baltic zones (EE, LV, LT) are not downloaded.** Section 4 of the notebook
shows why: after the switch to 15-minute resolution, Lithuania published flat
quarter-hours for one to three and a half months, and Latvia never stabilised —
LV generation dropped back to 0% varying for whole months in 2025. Rather than
carry three zones that would have to be excluded downstream anyway, they were
removed from `entsoe_tp.raw_dump.DEFAULT_ZONES` and from `entsoe_tp.areas`. The
analysis rows are retained in the notebook as the record of that decision.

| Step | What | Why |
|---|---|---|
| 1 | Drop duplicate rows on the full key | Month-chunked downloads with inclusive endpoints overlap at the joins |
| 2 | Drop Finland, and any zone `entsoe_tp.areas` no longer defines | FI is out of scope. The Baltic zones were removed from the download entirely (see below), but a raw file captured earlier still contains them, and section 7 would fail looking up a timezone for a zone the package has forgotten |
| 3 | Drop constant and near-constant series | A series more than 90% exactly zero (e.g. `NO5 wind_onshore`) carries no information and produces a zero-MAD column that the model's scaler divides by |
| 4 | Find the cut-off | Per zone, the last hour whose day-ahead price is still genuinely hourly — a `PT60M` stamp, or a `PT15M` hour whose four values are identical |
| 5 | Truncate at the earliest such cut-off across zones | So no zone contributes a partial-15-minute tail |
| 6 | Collapse flows to hourly | Mean of the quarter-hours in each clock hour. For prices in the retained window the four values are identical, so the mean is exact, not an approximation. Section 5's sanity check asserts this |
| 7 | Step the weekly reservoir series out to hourly | A stock is held, never interpolated, and shifted forward by its publication period so no hour knows a level before it could have been published |

The reservoir treatment is worth stating explicitly for the write-up: a reservoir
level is a **level, not a rate**. Interpolating between weekly publications would
invent a smooth trajectory the data does not contain, and would do it by reading
the *next* level backwards into hours that precede it. It is therefore held
constant, and hours before the first available level are not emitted at all
rather than back-filled.

---

## Section 7: the model-ready panel

`epftoolbox` imposes a strict invariant: **exactly 24 rows per calendar day, in
local market time, with no gaps.** `_lear.py` reshapes the test set with
`reshape(-1, 24)`, and both models look features up by exact timestamp, so a
missing hour raises `KeyError` and a duplicated one corrupts the array shape.

Three things stand between the UTC panel and that invariant.

### 1. Local time

The market trades in local time and the models index by local hour-of-day, so
the panel is converted. Timezones come from `entsoe_tp.areas`, not a table
restated in the notebook, so there is one definition.

Every retained zone is CET/CEST (`Europe/Copenhagen`, `Europe/Oslo`,
`Europe/Stockholm`, `Europe/Berlin`), so per-zone conversion currently gives the
same result as a single market-time conversion would. Finland is EET/EEST and is
dropped; it would matter if it were ever restored, which is why the skipped-hour
detection asks the timezone rather than assuming 02:00.

### 2. Daylight saving

Two days a year are not 24 hours long:

| | Reality | Needed | Treatment |
|---|---|---|---|
| Last Sunday in March | **23 hours** — the clock jumps 02:00 → 03:00, so there is no 02:00 | 24 | The missing hour is **linearly interpolated** from its neighbours |
| Last Sunday in October | **25 hours** — 02:00 happens twice | 24 | The two observations sharing the slot are **averaged** |

This is the `epftoolbox` convention, not an invention. It was verified against
the NP dataset shipped with the toolbox:

- every one of the 17,472 test rows fits **728 × 24 exactly**, with no day having
  23 or 25 rows;
- on **2017-03-26** and **2018-03-25** the 02:00 price is the mean of its 01:00
  and 03:00 neighbours to the last decimal.

The skipped hour is found by asking the timezone which hour does not exist
(`tz_localize(..., nonexistent="NaT")`), never by hardcoding a label — it is
02:00 under CET but 03:00 under EET:

```
Europe/Copenhagen: 10 skipped hours in 10 years, always at [2]
Europe/Helsinki:   10 skipped hours in 10 years, always at [3]
```

`ambiguous=True` resolves the repeated autumn hour to the first of the pair, so a
fall-back hour — which exists, twice — is never mistaken for a missing one.

> **This is the one place the pipeline reads forward.** Filling 02:00 uses the
> value at 03:00. Everywhere else, filling is strictly backward-looking. The
> exception is deliberate: it is the published convention, it makes results
> comparable with the paper's, and it covers one hour per zone per year —
> measured at **0.0114%** of the panel. As in `epftoolbox`, the filled value is
> then ordinary data: nothing marks it, and nothing excludes it from scoring.

### 3. Imputation

ENTSO-E does not publish everything. In the raw panel a gap is an **absent row**;
once the series is reindexed onto a complete grid it becomes `NaN`, and neither
`LassoLarsIC` nor Keras will fit on `NaN`.

This matters more than the hour count suggests, because a price is an **input as
well as an output**. LEAR uses price at D-1, D-2, D-3 and D-7, so one missing
price hour on day *d*:

- makes day *d* unusable as a training target — its 24-vector has a hole; **and**
- puts a `NaN` regressor into days *d+1*, *d+2*, *d+3* and *d+7*.

**Five days lost per hole.** That is why gaps cannot simply be left alone.

Gaps are filled from **past observations only**, in this order, each method
handling what the previous one could not:

| Order | Method | Handles |
|---|---|---|
| 1 | Forward fill, runs of ≤ 3 hours only | Brief publication hiccups |
| 2 | Same hour in the nearest earlier week, up to 8 weeks back | Longer outages, preserving hour-of-day and day-of-week shape |
| 3 | Same hour on an earlier day, up to 6 days back | When no earlier week is available |
| 4 | Expanding median of earlier observations at that hour | Last resort, early in the series |

Forward fill is capped deliberately. `ffill(limit=n)` fills the first *n* hours of
*any* run, which would leave a long gap partly filled with a stale value; only
fills that closed a run entirely are kept, and longer gaps fall through to the
weekly method. The expanding median uses `shift(1)` within each hour-of-day group
so the current observation is excluded and the median is taken strictly over the
past.

Hours with **no earlier data at all** cannot be filled causally. They are dropped,
not invented: the panel simply starts at the first complete local day.

Imputation runs on each zone's series **before** they are summed into model
columns. A wind total is onshore plus offshore, and summing first would turn one
missing component into a missing total, discarding the component that *was*
published.

> **This has no `epftoolbox` precedent.** The published NP/BE/FR/DE datasets are
> complete — NP's real price has **zero** `NaN` across its entire 728-day test
> period — so Lago et al. never had to fill a genuinely missing price. This
> cascade is this project's own choice and has to be defended on its own terms.

---

## What is *not* done

**Imputed values are not flagged or excluded from scoring.** `epftoolbox` does not
distinguish a filled value from a real one, and neither does this. A forecast is
therefore scored against a filled value wherever one exists. The alternative — a
per-series mask carried through to the evaluator — was considered and rejected as
disproportionate given the volume involved, but it is a real methodological
choice and should be stated as such.

**Nothing is smoothed, deseasonalised or outlier-trimmed.** Price spikes are real
market events and the models are meant to face them.

**The panel is not scaled.** Both models scale internally, inside their own
calibration windows, which is what keeps scaling out of the information set of a
forecast.

---

## Output

`datasets/nordic_baltic_clean_hourly.parquet`, long format:

| Column | Meaning |
|---|---|
| `timestamp_local` | Naive local market time. Every calendar day holds exactly 24 rows |
| `zone` | Bidding zone |
| `variable` | `price`, `load_forecast`, `generation_forecast`, `reservoir` |
| `psr_type` | Production type where one applies (`wind_onshore`, `wind_offshore`, `solar`), else empty |
| `value` | Never missing |

Alongside it, `datasets/cleaning_report.json` records per zone how many hours
were DST-interpolated, how many were imputed and by which method, and how many
leading hours were trimmed — so a backtest can be traced back to how its inputs
were built.

The notebook asserts the invariants before saving: no `NaN`, no duplicate
`(series, hour)`, exactly 24 rows per series per day, a gapless hourly run, and
every spring-forward hour equal to the mean of its neighbours. These are
assertions rather than printed diagnostics because a violation produces a *wrong
model*, not a crash.

---

## What changed in the move

Cleaning used to happen inside each model runner. Moving it changed a few things,
so results produced before the move are **not guaranteed to reproduce**:

1. **The DST hour.** Previously left `NaN` by acquisition and then forward-filled
   by the runner, so 02:00 took the 01:00 value. It is now the mean of 01:00 and
   03:00.
2. **Component-level imputation.** Previously a missing wind component made the
   summed total missing, and the *total* was imputed. Components are now filled
   first, then summed.
3. **The imputation window.** The runner imputed the projected CSV, bounded by
   `--data-start`. The notebook imputes the full panel span, so the weekly and
   expanding-median methods see different history.
4. **Where the usable span starts.** The trim point is now computed per zone
   across all its series, not per layout across the selected columns.

A changed value anywhere inside a calibration window moves every LASSO fit that
sees it, so forecasts can differ on days whose own inputs are unchanged.

**To find out whether any of this touched your data**, compare the newly built CSV
against the one your existing results used — no model fitting required:

```
python compare_datasets.py datasets/DK1_clean_load-windsolar.csv old/DK1_clean.csv
```

It exits 0 and says so if they are identical, in which case the existing backtest
still stands. Otherwise it reports which columns and hours moved.
