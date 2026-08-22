# Treatment of missing values

Draft material for the methodology chapter. Figures marked `[X]` must be filled
from the run manifest (`experiments/<run>/run_metadata.json`, key `imputation`)
once the dataset has been built — they are not yet measured on the real DK1
series.

---

## 1. Origin and extent of missingness

The dataset is assembled from three ENTSO-E Transparency Platform data items for
bidding zone DK1: day-ahead prices (12.1.D, `documentType=A44`), the day-ahead
total load forecast (6.1.B, `A65`), and the day-ahead wind and solar generation
forecast (14.1.D, `A69`), the latter split by production type into wind
(on- and offshore, `psrType` B18 and B19) and solar (B16). All three are
*ex ante* forecasts published before gate closure, and are therefore genuinely
available to a forecaster at the time a day-ahead price forecast must be made.

Hourly observations are absent for several distinct reasons, which are worth
separating because they justify different treatment:

1. **Platform coverage ramp-up.** The Transparency Platform became operational
   on 5 January 2015, and publication was not immediately complete for every
   data item. Coverage in the first days of the sample is sparse.
2. **Publication outages and late submissions.** Isolated hours or short runs
   are absent where a TSO submission failed or arrived late.
3. **Production types not yet reported.** A zone may begin publishing a
   generation forecast for a given production type later than others.
4. **Daylight saving time.** On the spring transition the local hour 02:00 does
   not exist, so no observation can exist for it.

Across the sample period ([start] to [end], [N] hourly observations), missing
values account for [X]% of the price series, [X]% of the load forecast, [X]% of
the wind forecast and [X]% of the solar forecast.

## 2. Design principle: acquisition is separated from treatment

The download step stores exactly what the platform published and records every
unpublished hour as missing. No value is filled, smoothed or extrapolated at
this stage. Imputation is applied later, when the dataset is loaded for
modelling.

This separation is deliberate. If gaps were filled during download, the stored
dataset would silently contain synthetic values indistinguishable from observed
ones, the choice of method would be unrecoverable from the data, and any
sensitivity analysis would require re-downloading. Keeping the raw series
immutable means the imputation scheme is a stated modelling assumption that can
be varied and reported, rather than an invisible property of the input.

## 3. Requirement: imputation must be causal

The evaluation design is a walk-forward backtest with daily recalibration: to
forecast day *D*, the model is re-estimated on a trailing calibration window
using only information available before *D*. The credibility of the resulting
error metrics rests entirely on that information set being respected.

An imputed value therefore cannot be permitted to depend on observations later
than the hour it fills. Formally, the value assigned to a missing observation at
time *t* must be a function only of observations at times strictly less than *t*.
Violating this introduces look-ahead bias through two distinct channels:

- **Training contamination.** A model calibrated on a window containing values
  derived from future observations has been given information a real forecaster
  would not have had, biasing measured accuracy optimistically.
- **Target contamination.** Where a gap falls in the *price* series inside the
  test period, the model is scored against a value that itself encodes future
  information — corrupting the error metric directly rather than merely the
  model.

This requirement rules out several methods that are otherwise conventional for
time-series gap filling. **Linear interpolation is bidirectional by
construction**: it fills a gap by drawing a line between the last observation
before it and the first after it, so it cannot be made causal. **Global summary
statistics** — a series mean, a per-hour mean or median computed over the whole
sample — are contaminated for the same reason, since the statistic is computed
partly from observations later than the gap it fills. Both are common defaults,
and both were rejected here.

The practical magnitude of the distortion is easily demonstrated. On a synthetic
hourly series whose level steps from 50 to 500 exactly at the point a gap ends,
a causal scheme fills the gap with values reaching at most 75 — consistent with
the pre-gap level and the diurnal amplitude — whereas linear interpolation
returns values up to 473 and a "same hour, following week" rule up to 525. Both
non-causal methods have reproduced a level shift that had not yet occurred.

## 4. Imputation hierarchy

Missing values are filled by the first applicable rule in the following order.
Each rule uses only observations preceding the hour being filled.

| Rule | Applies to | Justification |
|---|---|---|
| Carry the last observation forward | Runs of at most 3 hours | Adjacent hourly observations are strongly correlated, so over a short interval persistence is close to harmless. Unlike interpolation it requires nothing from the far side of the gap. |
| Same hour, an earlier week (−7, −14, … days, up to 8 weeks) | Longer runs | Preserves both the diurnal profile and the weekday/weekend distinction, which are pronounced in prices, load and solar generation alike. |
| Same hour, an earlier day (−1 … −6 days) | Hours for which no earlier week is available | Relevant only in the first weeks of the sample. |
| Expanding median for that hour of day, over past observations only | Anything still missing | Last resort; the expanding window ensures the statistic is computed from past data alone. |
| No fill; observation dropped | Hours with no prior observation at all | See §5. |

The ordering reflects a preference for the most local information that remains
informative. Persistence is appropriate over a few hours but degrades quickly:
carried across a multi-day gap it would flatten the diurnal cycle entirely,
which is why the weekly rule takes over beyond the three-hour threshold. The
threshold itself is a parameter (`--max-linear`) and is a natural candidate for
sensitivity analysis.

## 5. Boundary handling

Observations at the very beginning of the sample have no prior data to draw on
and therefore cannot be imputed causally by any rule. Rather than fall back on a
non-causal estimate, these hours are left missing and the sample is truncated
forward to the first midnight from which all series are complete. Truncating to
a midnight boundary preserves the requirement of exactly 24 observations per
calendar day, which the LEAR feature construction assumes.

For this reason the sample begins on 7 January 2015 rather than 5 January, the
platform's first day of operation. [Adjust if the run reports further trimming.]

## 6. Daylight saving time

The index is naive local market time with exactly 24 rows per calendar day,
matching the convention of the reference datasets distributed with `epftoolbox`.
The two DST transitions are handled distinctly:

- **Autumn transition (25-hour day).** Two observations map to the single 02:00
  slot. They are averaged. This is an aggregation of two genuine observations,
  not an imputation, and cannot be avoided while maintaining a fixed 24-row day.
- **Spring transition (23-hour day).** Local 02:00 does not exist, so the slot
  is necessarily empty. It is treated as an ordinary one-hour gap and filled by
  the causal rules above.

## 7. Reporting

Each backtest run records, per series, the number of values filled by each rule
and the number left unfilled, in `run_metadata.json`. This makes the exact
extent and composition of imputation recoverable for any reported result rather
than being a matter of assertion.

## 8. Limitations

Several caveats should be stated explicitly rather than left implicit.

**Missingness may not be at random.** All rules assume the missing observation
resembles nearby or seasonally analogous observed ones. If publication failures
correlate with market conditions — for instance if outages cluster around
extreme events or system stress — imputed values will be systematically
unrepresentative, and in a direction that understates volatility. The extent of
this cannot be assessed from the data itself.

**The weekly rule assumes short-run seasonal stability.** Substituting the same
hour from an earlier week is reasonable under stable conditions but degrades
around public holidays and during regime shifts. The 2022 energy price episode
is the obvious case in this sample: week-on-week price levels moved sharply,
so a gap filled from the preceding week may be materially mis-levelled. Gaps
falling in that period deserve individual scrutiny.

**Persistence understates short-run variability.** Carrying an observation
forward across up to three hours produces a locally flat segment, marginally
reducing measured volatility.

**Imputed values enter both training and evaluation.** Where an imputed hour
falls in the test period's price series, the model is scored against a partly
synthetic target. If this affects a non-negligible share of test hours,
forecast accuracy should additionally be reported over the subset of test days
whose prices are fully observed, and the two figures compared. This is the
single most important robustness check for the scheme described here.

**Single imputation.** Each missing value is replaced by one point estimate, so
the additional uncertainty introduced by imputation is not propagated into the
reported error metrics. A multiple-imputation treatment would quantify it, at
substantial computational cost given daily recalibration; it was not attempted.

## 9. Suggested robustness checks

1. Report MAE and rMAE both over all test days and over the subset with fully
   observed prices.
2. Vary the persistence threshold (`--max-linear`, e.g. 0, 3, 6 hours) and
   confirm that reported accuracy is insensitive to it.
3. Report the share of imputed observations separately for the training and test
   portions of the sample, since their consequences differ.
