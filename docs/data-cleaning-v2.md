# Data cleaning and missing-data handling

This document describes the cleaning implemented in
[`data_cleaning_v2.ipynb`](../data_cleaning_v2.ipynb). The notebook converts the
raw ENTSO-E extract into a complete, long-form hourly UTC dataset for electricity
price forecasting. Missing-data treatment is performed once in this shared
pipeline so downstream models receive the same observations and assumptions.

## Scope

The input is `datasets/nordic_baltic_raw.parquet`. The retained bidding zones
are DK1, DK2, NL, DE-LU, NO2, SE3 and SE4. The retained variables are day-ahead
price, day-ahead load forecast, and day-ahead solar and wind generation
forecasts. Reservoir data are excluded because they are weekly stock levels,
not hourly flows.

The cleaned sample runs from **2019-01-01 00:00 UTC** through
**2025-09-30 21:00 UTC**. The end is chosen because the next hour marks the
transition to genuinely quarter-hourly day-ahead prices in most retained zones.
Stopping there keeps the target price at hourly market resolution.

All cleaning remains in UTC. Consequently, daylight-saving transitions do not
create missing or duplicated clock hours in this dataset.

## Pipeline overview

```text
nordic_baltic_raw.parquet
    |
    |  select zones, variables and end time
    |  remove duplicate source rows
    |  aggregate PT15M observations to UTC hours
    |  prefer PT60M where both resolutions exist
    |  trim the common sample and remove unusable series
    |  diagnose and fill missing hours
    |  remove constant NO2 solar
    |  assert completeness and uniqueness
    v
nordic_baltic_clean_hourly_utc.parquet
```

### 1. Filter the raw extract

The parquet scan applies zone, variable and time predicates before materializing
the data. It includes all sub-hourly slots belonging to the final retained hour.
Categorical nulls are normalized to empty strings, and ENTSO-E production codes
are mapped to `solar`, `wind_offshore` and `wind_onshore`.

The notebook asserts that every requested zone is present, excluded variables
are absent, no timestamp exceeds the bound, and no retained price hour must be
constructed from differing quarter-hour prices without an hourly alternative.

### 2. Remove duplicate rows

A source row is uniquely identified by:

- zone;
- variable;
- production type;
- source resolution; and
- UTC timestamp.

Duplicate keys arise mainly at monthly download boundaries. They are analyzed
before the first copy is retained. In the documented run, 27,288 redundant rows
were removed and none of the duplicated copies disagreed in value.

### 3. Convert mixed resolutions to hourly data

PT15M values are averaged within each UTC clock hour. This is an hourly average
for MW forecasts. The sample boundary ensures that quarter-hour prices used in
the panel represent repeated hourly prices rather than distinct quarter-hour
products.

Aggregation is performed separately by source resolution. If an hour contains
both PT60M and PT15M data, PT60M takes precedence; the PT15M mean is used only
when no PT60M observation exists. Partial hours are counted and reported.
After this step, each series has at most one row per UTC hour.

### 4. Align the sample and remove unsuitable series

The initial common start is 2018-10-01, when DE-LU load is available. Series
beginning later than this point are removed rather than represented by years of
leading missing values; this removes SE3 and SE4 solar. The final modeling start
is then set to 2019-01-01.

NO2 offshore wind is removed because it is unsuitable for the retained panel.
After imputation, variance diagnostics identify NO2 solar as constant at zero,
so that series is also removed.

## Missing-data analysis

A missing observation is represented in the raw long-form data by an absent
series-hour row. For each series, the notebook constructs the complete hourly
UTC grid between its first and last timestamps and takes the difference from the
observed timestamps.

Missingness is reported in three forms:

1. coverage by series, including partial source hours;
2. contiguous gap runs with their start, end and duration; and
3. a daily availability raster, with the period from 2023-10-01 highlighted.

After the 2019 start-date filter and removal of NO2 offshore wind, the documented
run contains **91 gap runs and 2,190 missing hours** across 16 series. The longest
runs contain 48 consecutive hours. Prices have no gaps in the retained sample;
all gaps occur in exogenous load, solar or wind forecasts.

## Missing-data handling

The method depends on the physical variable. This avoids applying one generic
time-series rule to forecasts with different temporal and spatial behavior.
There is deliberately no fallback that silently substitutes an unrelated value:
if the required historical value, neighboring forecasts or training history are
unavailable, the notebook raises an error.

### Solar and load: same hour in the previous week

Every missing solar or load value at hour $t$ is replaced by the same series at
exactly $t-168$ hours:

$$
\hat{y}_t = y_{t-168}.
$$

This preserves hour-of-day and day-of-week effects while using an observation
that predates the missing hour. Missing timestamps are processed in ascending
order. Therefore, if a gap exceeded one week, a value filled earlier in the run
could become the one-week lag for a later missing hour. In the documented data,
the longest gap is only 48 hours.

The method does not search farther back when the previous-week value is absent.
Instead, it raises an error, making unsupported imputation visible.

### Wind: causal cross-zone ordinary least squares

Wind generation is spatially correlated, while persistence over a long outage
can erase meaningful weather variation. Each contiguous wind gap is therefore
filled from forecasts for the same wind type in other zones.

A separate OLS model is fitted for every target-series gap. Candidate predictors
are all non-target zones whose corresponding wind forecast is present for every
hour of that gap. Onshore wind predicts only onshore wind, and offshore wind
predicts only offshore wind.

For a gap beginning at $g$, the training sample contains only complete rows
strictly before the gap:

$$
\mathcal{T}_g = \{t : t < g,\ y_t\text{ and all selected }x_t\text{ are observed}\}.
$$

The predictor columns are standardized using means and standard deviations from
$\mathcal{T}_g$ only. An intercept is added and coefficients are estimated by
least squares:

$$
\hat{\beta}_g = \arg\min_{\beta}\lVert y-X\beta\rVert_2^2.
$$

At least 672 complete historical hours, equivalent to 28 days, are required.
A zero-variance predictor is assigned a scale of one to avoid division by zero.
Negative fitted wind values are physically implausible and are clipped to zero:

$$
\hat{y}_t = \max(0, x_t^\top\hat{\beta}_g).
$$

The model coefficients, centering and scaling use no target or predictor rows at
or after the gap start. Prediction does use the contemporaneous neighboring-zone
wind forecasts for each missing target hour. These are exogenous day-ahead
forecasts available for the same delivery period, not future observations of the
target series. Thus the procedure prevents future-data leakage in model fitting
while exploiting information available across zones at forecast time.

A gap fails explicitly if no neighboring zone is complete throughout it or if
fewer than 672 complete pre-gap training hours exist. The code also asserts that
the latest training timestamp is earlier than the gap start.

### Variables without a rule

The imputer handles only load, solar, onshore wind and offshore wind. A gap in a
price series or another variable raises an error. This is intentional: silently
applying an exogenous-forecast rule to a price target would change the empirical
meaning of the evaluation. No such price gaps occur in the retained sample.

## Imputation audit trail

Observed and synthetic rows remain distinguishable in the exported parquet:

| Column | Meaning |
|---|---|
| `imputed` | `True` for a row inserted by the missing-data procedure |
| `imputation_method` | `observed`, `same_hour_previous_week`, or `causal_cross_zone_ols` |
| `imputation_predictors` | Comma-separated predictor zones for wind OLS; empty otherwise |

Before the final constant-series removal, all 2,190 missing hours are filled:

| Method | Hours |
|---|---:|
| Causal cross-zone OLS for wind | 1,342 |
| Same hour in the previous week for solar and load | 848 |
| **Total** | **2,190** |

NO2 solar is subsequently removed because the complete series is constant at
zero. Its 50 previously filled hours leave the exported dataset. The final file
therefore contains **2,140 imputed rows**: 1,342 wind rows and 798 solar/load
rows.

## Validation and output

Before export, the notebook asserts that:

- no duplicate `(series, timestamp_utc)` pair exists;
- `value` contains no nulls; and
- every retained series is gapless between its first and last hour.

The output is `datasets/nordic_baltic_clean_hourly_utc.parquet`. In the documented
run it contains **1,715,582 rows, 20 columns and 29 series**. The data remain in
long format and retain source metadata, source resolution, slot counts and the
imputation audit fields.

## Methodological limitations

The weekly substitution assumes that load and solar conditions at the same hour
of the preceding week are representative. Public holidays, rapid seasonal
changes and structural shifts can violate that assumption.

Cross-zone OLS assumes a sufficiently stable linear spatial relationship. One
model is used throughout each gap, and predictions do not propagate coefficient
uncertainty. Predictor availability can also change which zones enter different
gap models.

Imputed observations are point estimates. They should not be interpreted as
observed ENTSO-E publications, and analyses sensitive to synthetic values should
use the audit columns for exclusion or robustness checks. Useful checks include
re-estimating results without periods containing imputed targets or inputs and
comparing alternative minimum-history requirements for the wind regressions.
