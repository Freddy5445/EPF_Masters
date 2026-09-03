# Treatment of missing values

This document describes the missing-data treatment currently implemented in
[`data_cleaning_v2.ipynb`](../data_cleaning_v2.ipynb). It complements the full
pipeline description in [`data-cleaning-v2.md`](data-cleaning-v2.md).

The central distinction is between two different phenomena:

- **Publication gaps** are absent physical UTC observations in the ENTSO-E data
  and are handled in the cleaning notebook.
- **DST clock irregularities** arise when complete UTC data are mapped to local
  delivery days and are normalized at the end of the cleaning notebook by
  [`local_day_panel.py`](../local_day_panel.py).

They are not treated as the same kind of missingness.

## 1. Dataset and observed missingness

The retained sample covers DK1, DK2, NL, DE-LU, NO2, SE3 and SE4. It contains
day-ahead prices, day-ahead load forecasts, and day-ahead solar and onshore or
offshore wind generation forecasts. The local delivery window is
2019-01-01 00:00 through 2025-09-30 23:00 in `Europe/Copenhagen`. Cleaning uses
the corresponding 2018-12-31 23:00 through 2025-09-30 21:00 UTC physical window
before exporting normalized local timestamps.

After hourly aggregation, the common-window filter and removal of NO2 offshore
wind, the notebook finds **91 gap runs containing 2,190 missing UTC hours** in
16 series. The longest runs contain 48 hours. All publication gaps are in
exogenous forecasts; the retained price series contain no gaps.

Missingness is diagnosed before filling through:

1. Observed versus expected hours for every series.
2. Every contiguous gap's start, end and duration.
3. Monthly coverage.
4. A daily availability raster with the period from 2023-10-01 highlighted.

An absent publication is represented in long form by a missing series-hour row,
not by a row whose `value` is null. The notebook detects it by comparing observed
timestamps with a complete hourly UTC grid.

## 2. Separation of acquisition, cleaning and model panels

The raw parquet preserves what ENTSO-E published. It is not imputed during
download. The cleaning notebook aggregates resolution, diagnoses gaps, inserts
synthetic rows under explicit rules, normalizes local delivery days, and writes
`datasets/nordic_baltic_clean_hourly_local.parquet`.

The cleaned parquet uses naive local market timestamps and carries separate audit
fields for publication-gap imputation and DST adjustment. Downstream model scripts
perform neither missing-value imputation nor DST normalization; they project the
already-complete local panel into the CSV layouts consumed by LEAR and the DNNs.

This separation keeps the raw data recoverable, gives every model the same
filled inputs, and prevents model-specific copies of the imputation rules from
drifting apart.

## 3. Information-set requirement

Publication-gap treatment is designed to avoid future target information. For a
missing observation at UTC hour $t$, the weekly method reads $t-168$ hours. A
wind model fitted for a gap beginning at $g$ is trained only on complete rows
whose timestamps satisfy $t<g$.

Wind prediction does use contemporaneous forecasts from neighboring zones during
the target gap. These are day-ahead exogenous forecasts for the same delivery
hours, not later observations of the missing target series. The historical
coefficient fit, predictor centering and predictor scaling remain strictly
pre-gap.

No bidirectional interpolation, future-week substitution, whole-sample mean or
whole-sample median is used to repair ENTSO-E publication gaps.

## 4. Variable-specific filling rules

There is no generic hierarchy and no silent fallback. The rule is selected from
the physical variable, and the notebook raises an error when its requirements
are not met.

### 4.1 Solar and load

Every missing solar or load forecast at hour $t$ is replaced by the same series
exactly one physical week earlier:

$$
\hat{y}_t = y_{t-168\text{ h}}.
$$

This preserves hour-of-day and weekday structure in UTC while using only prior
data. Missing timestamps are processed from oldest to newest, so a filled value
could support another fill more than one week later. The current gaps are at most
48 hours, so this cascading case does not arise.

The method does not search two or more weeks backward if $t-168$ hours is absent.
It fails explicitly instead. Because a fixed 168-hour UTC lag can cross a DST
offset change, the rebuilt dataset was checked separately: **0 of the 798
retained weekly fills change local clock hour relative to their source**.

### 4.2 Wind

Every contiguous wind gap is filled by a separate cross-zone ordinary least
squares model. Onshore wind uses only onshore forecasts from other zones;
offshore wind uses only offshore forecasts.

For a target zone and a gap beginning at $g$:

1. Candidate predictors are non-target zones whose same-type wind forecast is
   present for every hour of the gap.
2. Training rows require the target and every selected predictor to be observed.
3. Every training timestamp must be strictly earlier than $g$.
4. At least 672 complete historical hours, or 28 days, are required.

For the selected pre-gap training set $\mathcal{T}_g$,

$$
\mathcal{T}_g = \{t : t < g,\ y_t\text{ and all }x_t\text{ are observed}\}.
$$

Predictors are standardized using only their means and standard deviations in
$\mathcal{T}_g$. A constant predictor receives scale one. An intercept is added
and coefficients are estimated by least squares:

$$
\hat{\beta}_g = \arg\min_{\beta}\lVert y-X\beta\rVert_2^2.
$$

The fitted model is applied to contemporaneous neighbor-zone forecasts over the
gap. Negative predictions are clipped to zero:

$$
\hat{y}_t = \max(0, x_t^\top\hat{\beta}_g).
$$

There is no fallback to persistence or a weekly lag if no neighbor is complete
or the historical sample is too short; the pipeline stops with an error.

### 4.3 Prices and unsupported variables

The retained price series have no publication gaps and **no price value is
imputed**. If a price or any unsupported variable were missing, the notebook
would raise an error rather than apply an exogenous-forecast rule to the target.

## 5. Measured imputation totals

Before constant-series removal, all 2,190 gaps are filled:

| Method | Variable | Hours |
|---|---|---:|
| Causal cross-zone OLS | Wind | 1,342 |
| Same hour previous week | Solar | 798 |
| Same hour previous week | Load | 50 |
| **Total before series removal** | | **2,190** |

NO2 solar is then removed because the entire series is constant at zero. This
removes 50 weekly-filled rows together with the rest of that series. The exported
parquet therefore contains:

| Retained category | Imputed hours |
|---|---:|
| Wind offshore | 456 |
| Wind onshore | 886 |
| Solar | 748 |
| Load | 50 |
| Price | 0 |
| **Total** | **2,140** |

The 2,140 retained synthetic rows are **0.1247%** of the final 1,715,611-row
dataset.

## 6. Audit trail and validation

The canonical cleaned parquet records treatment row by row:

| Column | Meaning |
|---|---|
| `imputed` | `True` only for a row inserted by publication-gap treatment |
| `imputation_method` | `observed`, `same_hour_previous_week`, or `causal_cross_zone_ols` |
| `imputation_predictors` | Comma-separated wind predictor zones; empty otherwise |

Before export, the notebook asserts that no `(series, timestamp_utc)` pair is
duplicated, no `value` is null, and no hourly UTC gap remains. These checks apply
after filling and after the constant NO2 solar series is removed.

Imputation metadata belongs to the dataset, not to an individual backtest run.
LEAR and all DNN configurations read projections derived from this same cleaned
parquet.

## 7. DST normalization during cleaning

The UTC panel is first completed over physical time, but local market days contain
23 or 25 physical hours at DST transitions. Before writing the cleaned parquet,
`local_day_panel.py` converts every series to `Europe/Copenhagen` and applies the
standard 24-hour EPF convention:

- On each spring transition, nonexistent local 02:00 is inserted as the mean of
  local 01:00 and 03:00.
- On each autumn transition, the two physical observations labelled local 02:00
  are averaged into one value.

The spring rule reads the following delivery hour and is therefore not a causal
publication-gap imputation. It is a deliberate calendar normalization applied
after the physical UTC panel is complete, identically to prices and every exogenous
series. Autumn handling aggregates two observed values and does not
invent a missing publication.

The module rejects any other missing or repeated local hour. For the current
sample, all 29 series pass assertions for seven spring and six autumn transition
days. Each series becomes 2,465 complete local days, or 59,160 model rows, with
complete first and last days.

## 8. Limitations and robustness checks

**Weekly stability.** Solar and load substitution assumes the previous week's
same hour is representative. Holidays, rapid seasonal changes and structural
shifts can weaken that assumption.

**Stable spatial relationship.** Wind OLS assumes cross-zone relationships are
approximately linear and stable over the historical sample. Predictor sets can
differ by gap according to availability.

**Single imputation.** Every gap receives one point estimate. The additional
uncertainty is not propagated into model prediction intervals or error metrics.

**Possible informative missingness.** If publication outages coincide with
unusual system conditions, filled values may underrepresent extremes.

Useful robustness checks are to exclude model days whose inputs include an
imputed value, compare alternative minimum-history requirements for wind OLS,
and report results with the DST-normalized transition days excluded. Price-only
target filtering is unnecessary for the current sample because no price target
was imputed.
