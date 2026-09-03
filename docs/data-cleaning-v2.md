# Data cleaning and missing-data handling

This document describes the cleaning implemented in
[`data_cleaning_v2.ipynb`](../data_cleaning_v2.ipynb). The notebook converts the
raw ENTSO-E extract into a complete, long-form hourly local dataset for electricity
price forecasting. Missing-data treatment is performed once in this shared
pipeline so downstream models receive the same observations and assumptions.

## Scope

The input is `datasets/nordic_baltic_raw.parquet`. The retained bidding zones
are DK1, DK2, NL, DE-LU, NO2, SE3 and SE4. The retained variables are day-ahead
price, day-ahead load forecast, and day-ahead solar and wind generation
forecasts. Reservoir data are excluded because they are weekly stock levels,
not hourly flows.

The sample bounds are defined as local delivery hours in `Europe/Copenhagen`:
**2019-01-01 00:00** through **2025-09-30 23:00**. They become
**2018-12-31 23:00 UTC** through **2025-09-30 21:00 UTC** for the parquet scan.
The end is chosen because the next local hour marks the transition to genuinely
quarter-hourly day-ahead prices in most retained zones.

Publication-gap detection and filling operate on the unambiguous physical UTC
grid. Before export, the cleaning notebook converts every retained series to
naive `Europe/Copenhagen` local market time and applies the 24-hour EPF DST
convention. Downstream analysis and model scripts consume this result directly.

## Pipeline overview

```text
nordic_baltic_raw.parquet
    |
    |  select zones, variables and local-derived UTC bounds
    |  retain only the DE-LU PT60M price product
    |  remove duplicate source rows
    |  aggregate PT15M observations to UTC hours
    |  prefer PT60M where both resolutions exist
    |  trim the common sample and remove unusable series
    |  diagnose and fill missing hours
    |  remove constant NO2 solar
    |  normalize DST to 24 local delivery hours per day
    |  assert completeness and uniqueness
    v
nordic_baltic_clean_hourly_local.parquet
```

### 1. Filter the raw extract

The parquet scan applies zone, variable and converted time predicates before
materializing the data. It includes all sub-hourly slots belonging to the final
retained hour.
Categorical nulls are normalized to empty strings, and ENTSO-E production codes
are mapped to `solar`, `wind_offshore` and `wind_onshore`.

The notebook asserts that every requested zone is present, excluded variables
are absent, timestamps stay inside the bounds, and no retained price hour must
be constructed from differing quarter-hour prices without an hourly alternative.

#### DE-LU dual day-ahead price products

DE-LU publishes two complete parallel price series under the same EIC. `PT60M`
is the hourly day-ahead auction used by this study. `PT15M` is the separate
quarter-hourly auction, not a finer sampling of the same cleared product.

The distinction is empirical as well as semantic. Across 61,368 overlapping
hours, the four quarter-hour prices vary within 61,364 hours. Their hourly mean
has a zero-lag correlation of 0.986 with the hourly auction but a mean absolute
difference of EUR 8.28/MWh. The two series are related, but not interchangeable.

The pipeline therefore filters DE-LU price to `PT60M` explicitly during loading
and asserts after hourly construction that no retained DE-LU price row has
`source_resolution == "PT15M"`. This prevents a silent product switch when the
hourly auction series ends while the quarter-hourly series continues.

### 2. Remove duplicate rows

A source row is uniquely identified by:

- zone;
- variable;
- production type;
- source resolution; and
- UTC timestamp.

Duplicate keys arise mainly at monthly download boundaries. They are analyzed
before the first copy is retained. In the documented run, 14,112 redundant rows
were removed and none of the duplicated copies disagreed in value.

### 3. Convert mixed resolutions to hourly data

PT15M values are averaged within each UTC clock hour. This is an hourly average
for MW forecasts. The sample boundary ensures that quarter-hour prices used in
the panel represent repeated hourly prices rather than distinct quarter-hour
products.

Aggregation is performed separately by source resolution. For non-price series,
if an hour contains both PT60M and PT15M data, PT60M takes precedence and the
PT15M mean is used only when no PT60M observation exists. DE-LU price has already
been restricted to its PT60M product. Partial hours are counted and reported.
After this step, each series has at most one row per UTC hour.

### 4. Align the sample and remove unsuitable series

The local modeling start is 2019-01-01 00:00, or 2018-12-31 23:00 UTC. Series
beginning later than this point are removed rather than represented by years of
leading missing values; this removes SE3 and SE4 solar. Including the preceding
UTC hour makes the first local delivery day complete.

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

The lag is 168 physical UTC hours, which can differ from the same local clock
hour across a DST transition. In the rebuilt dataset, none of the 798 retained
weekly fills crosses such a local-hour mismatch.

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

Before local-time conversion, the notebook asserts that:

- no duplicate `(series, timestamp_utc)` pair exists;
- `value` contains no nulls; and
- every retained series is gapless between its first and last hour.

The output is `datasets/nordic_baltic_clean_hourly_local.parquet`. In the
documented run it contains **1,715,640 rows, 21 columns and 29 series**. It uses
naive `timestamp_local`, contains exactly 24 rows per local delivery day and
series, and retains source metadata, source resolution, slot counts and the
imputation audit fields. `dst_adjustment` distinguishes `spring_interpolation`,
`autumn_average`, and ordinary rows (`none`).

## Local-day normalization

The cleaning notebook calls [`local_day_panel.py`](../local_day_panel.py) once
before export. The module converts UTC timestamps to `Europe/Copenhagen`, derives
local date and hour, and constructs one local-date-by-local-hour matrix per series.

The module applies the standard 24-hour EPF convention identically to prices and
all exogenous series:

- on spring transitions, nonexistent 02:00 is inserted as the mean of 01:00 and
    03:00;
- on autumn transitions, the two physical observations labelled 02:00 are
    averaged into one local-hour value.

It rejects any missing or repeated local hour not explained by those transitions
and asserts that every matrix has columns 00:00 through 23:00 with complete first
and last days. The current sample yields 2,465 complete local days for every one
of the 29 series, or 59,160 model hours per series. The pre-conversion UTC panel
has 59,159 physical hours per series: across this sample, seven inserted spring slots
and six collapsed autumn slots produce a net increase of one row in the normalized
local grid. The assertions cover seven spring dates and six autumn dates, all 13
transitions in the sample.

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
