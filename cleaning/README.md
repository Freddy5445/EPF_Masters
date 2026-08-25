# `cleaning` — all data preparation, in one place

Every model tested in this project reads the output of this package. That is the
point: a difference between LEAR and the DNN is then a difference between the
models, not between two imputation routines that drifted apart.

## What to add to `data_cleaning.ipynb`

One cell, immediately before the `to_parquet` call:

```python
from cleaning import clean_panel, format_report

panel, cleaning_report = clean_panel(panel)   # add zones=[...] to restrict
print(format_report(cleaning_report))
```

`clean_panel` takes the raw long panel — columns `timestamp_utc, zone, variable,
psr_type, value` — and returns a cleaned long panel plus a report. The cleaned
panel is indexed on **naive local market time** (`timestamp_local`), holds
**exactly 24 rows per calendar day**, and has **no missing values**.

Optionally save the report next to the parquet, so a run can be traced back to
how its inputs were built:

```python
import json
with open("datasets/cleaning_report.json", "w") as f:
    json.dump(cleaning_report, f, indent=2, default=str)
```

## What it does, in order

1. **Grid.** Each series is projected onto the naive local hourly grid. The two
   fall-back 02:00 observations share one slot and are averaged — real values
   aggregated, nothing invented. (`entsoe_tp.hourly`)

2. **DST.** The spring-forward hour does not exist, so it is linearly
   interpolated from its neighbours. This is the epftoolbox convention, verified
   against the NP dataset: on 2017-03-26 and 2018-03-25 the 02:00 price is the
   mean of 01:00 and 03:00 to the last decimal, and all 17,472 NP rows fit
   728 × 24 exactly. The skipped hour is found from the timezone rather than
   hardcoded, so it is 02:00 under CET and 03:00 under EET. (`cleaning.dst`)

3. **Imputation.** Whatever is still missing is filled from **past observations
   only**: forward fill up to 3 hours, then the same hour in the nearest earlier
   week, then the same hour the previous day, then an expanding median. Hours
   with no history behind them cannot be filled causally and are dropped.
   (`cleaning.impute`)

Imputation runs on each zone's series *before* they are summed into model
columns, so one missing wind component does not discard the one that was
published.

## Two things worth knowing for the write-up

**The DST fill looks forward.** Interpolating 02:00 reads 03:00. Everywhere else
this project fills strictly from the past. The exception is deliberate: it is the
published convention, and it affects one hour per zone per year — 0.011% of a
728-day test period. Like epftoolbox, the filled value is then treated as
ordinary data; nothing marks it and nothing excludes it from scoring.

**Imputation has no epftoolbox precedent.** The published NP/BE/FR/DE datasets
are complete — NP's real price has zero NaN across its entire 728-day test
period — so the paper never had to fill a genuinely missing price. The causal
cascade here is this project's own choice and has to be defended on its own
terms.

## Consequences elsewhere

`run_lear_dk1.py` and `run_dnn_dk1.py` no longer impute. They refuse to run on a
dataset with gaps and say to re-run the notebook. `--no-impute` and
`--max-linear` are gone.
