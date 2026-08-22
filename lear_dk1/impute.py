"""
Filling the gaps a raw dataset carries, using only past observations.

``build_dataset --allow-gaps`` writes exactly what the platform published and
leaves every unpublished hour as NaN. LEAR cannot be fitted on that: it looks
features up by timestamp and NaN propagates silently through the LASSO fit into
the forecasts. So the gaps are filled here, at model time, where the choice of
method is a modelling decision that can be stated and varied.

**Every method here is causal**: a value at time *t* is derived only from
observations strictly before *t*. This matters more than it might appear. A
backtest simulates forecasting day D knowing only what was available beforehand,
so an imputed value that peeked at day D+7 would leak future information into
the training set -- and where a gap falls inside the test period, into the
target the model is scored against. Interpolating across a gap, or taking a
median over the whole series, both do exactly that; neither is used.

Four methods apply in order, each handling what the previous one could not:

1. **Carry the last observation forward**, for runs of at most ``max_ffill``
   hours. Adjacent hours are highly correlated, so over a short gap this is
   close to harmless -- and unlike interpolation it needs no value from the far
   side of the gap.

2. **Same hour, an earlier week.** For longer gaps the value from the same hour
   seven days earlier (then fourteen, and so on) keeps both the daily shape and
   the weekday/weekend distinction, which matters for load and price alike.

3. **Same hour, the previous day**, when no earlier week has that hour --
   in practice only in the first weeks of a series.

4. **Expanding median for that hour of day**, over past observations only.

Hours with no earlier data at all -- the very start of a series -- cannot be
filled causally by any method, and are deliberately left as NaN for the caller
to trim rather than invented.
"""

import pandas as pd

# One week of hourly observations. The index is a complete hourly grid, so a
# positional shift of this many rows is exactly seven days.
DAY = 24
WEEK = 24 * 7

# How far back to look for a same-hour value: eight weeks. Beyond that the
# "same hour" is seasonally too distant to stand in.
MAX_WEEK_SEARCH = 8


def _forward_fill_short(series, max_ffill):
    """Carry the last observation forward across runs of at most ``max_ffill``."""
    missing = series.isna()
    if not missing.any():
        return series, 0

    filled = series.ffill(limit=max_ffill)

    # ffill(limit=n) fills the first n hours of *any* run, which would leave a
    # long gap partly filled with a stale value. Only keep fills that closed a
    # run entirely.
    groups = (missing != missing.shift()).cumsum()
    lengths = missing.groupby(groups).transform("size")
    filled = filled.where(~(missing & lengths.gt(max_ffill)), other=None)

    return filled, int((missing & filled.notna()).sum())


def _fill_same_hour_earlier_week(series):
    """Fill from the same hour in an earlier week. Never looks forward."""
    before = int(series.isna().sum())
    if not before:
        return series, 0

    filled = series
    for weeks in range(1, MAX_WEEK_SEARCH + 1):
        if not filled.isna().any():
            break
        filled = filled.fillna(filled.shift(WEEK * weeks))

    return filled, before - int(filled.isna().sum())


def _fill_same_hour_previous_day(series):
    """Fill from the same hour a day earlier. Never looks forward."""
    before = int(series.isna().sum())
    if not before:
        return series, 0

    filled = series
    for days in range(1, 7):
        if not filled.isna().any():
            break
        filled = filled.fillna(filled.shift(DAY * days))

    return filled, before - int(filled.isna().sum())


def _fill_expanding_hour_median(series):
    """Fill with the median of *earlier* observations at the same hour of day."""
    before = int(series.isna().sum())
    if not before:
        return series, 0

    # shift(1) inside each hour-of-day group excludes the current observation,
    # so the median is taken strictly over the past.
    medians = series.groupby(series.index.hour).transform(
        lambda group: group.shift(1).expanding().median()
    )
    filled = series.fillna(medians)

    return filled, before - int(filled.isna().sum())


def impute_frame(frame, max_ffill=3):
    """Fill NaN using only past observations. Returns ``(filled, report)``.

    Hours with no earlier data are left as NaN: they cannot be filled causally,
    and inventing them would be indistinguishable from look-ahead. ``report``
    records how many values each method supplied, and how many remain.
    """
    out = frame.copy()
    report = {}

    for column in out.columns:
        series = out[column]
        total_missing = int(series.isna().sum())
        if not total_missing:
            report[column] = {"missing": 0}
            continue

        series, n_ffill = _forward_fill_short(series, max_ffill)
        series, n_weekly = _fill_same_hour_earlier_week(series)
        series, n_daily = _fill_same_hour_previous_day(series)
        series, n_median = _fill_expanding_hour_median(series)

        out[column] = series
        report[column] = {
            "missing": total_missing,
            "share": round(total_missing / len(out), 6),
            "forward_fill": n_ffill,
            "same_hour_earlier_week": n_weekly,
            "same_hour_previous_day": n_daily,
            "expanding_hour_median": n_median,
            "unfilled_no_history": int(series.isna().sum()),
        }

    return out, report


def first_complete_day(frame):
    """First midnight from which every column is complete for the rest of the frame.

    After causal imputation the only remaining NaN sit at the very start, where
    there is no history to draw on. Trimming to this timestamp keeps the
    24-rows-per-day grid intact.
    """
    incomplete = frame.isna().any(axis=1)
    if not incomplete.any():
        return frame.index[0]

    last_bad = frame.index[incomplete][-1]
    candidate = (last_bad + pd.Timedelta(hours=1)).normalize()
    if candidate <= last_bad:
        candidate = candidate + pd.Timedelta(days=1)
    return candidate


def format_report(report, total_rows):
    """Readable summary of what :func:`impute_frame` filled."""
    lines = []
    for column, counts in report.items():
        if not counts["missing"]:
            lines.append(f"  {column}: complete")
            continue
        line = (
            f"  {column}: {counts['missing']:,} of {total_rows:,} hours "
            f"({counts['missing'] / total_rows:.2%}) imputed "
            f"[carry-forward {counts['forward_fill']:,}, "
            f"same-hour-earlier-week {counts['same_hour_earlier_week']:,}, "
            f"same-hour-previous-day {counts['same_hour_previous_day']:,}, "
            f"hour-median {counts['expanding_hour_median']:,}]"
        )
        if counts["unfilled_no_history"]:
            line += (f"\n      {counts['unfilled_no_history']:,} hour(s) had no "
                     f"earlier data and were left unfilled")
        lines.append(line)
    return "\n".join(lines)
