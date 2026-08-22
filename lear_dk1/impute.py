"""
Filling the gaps a raw dataset carries, so a model can be fitted on it.

``build_dataset --allow-gaps`` writes exactly what the platform published and
leaves every unpublished hour as NaN. LEAR cannot be fitted on that: it looks
features up by timestamp and NaN propagates silently through the LASSO fit into
the forecasts. So the gaps are filled here, at model time, where the choice of
method is a modelling decision that can be stated and varied -- rather than at
download time, where it would be baked invisibly into the data.

Three methods are applied in order, each handling what the previous one could
not:

1. **Linear interpolation**, but only across runs of at most ``max_linear``
   hours. Over a short gap the series barely bends, so this is close to
   harmless. Over a long one it would draw a straight line through a daily
   cycle, which is why it is bounded.

2. **Same hour, nearest week.** For longer gaps the value from the same hour
   seven days earlier (then seven days later, then fourteen, and so on) keeps
   both the daily shape and the weekday/weekend distinction, which matters for
   load and price alike. A straight line through a multi-day gap destroys both.

3. **Median for that hour of day**, over the whole series. This is the last
   resort for hours no neighbouring week can supply -- in practice the very
   start of the range, where there is nothing earlier to copy from.

Every filled value is counted and attributed to the method that produced it, so
a run can report exactly how much of its input was invented and how.
"""

import numpy as np
import pandas as pd

# One week of hourly observations. The index is a complete hourly grid, so a
# positional shift of this many rows is exactly seven days.
WEEK = 24 * 7

# How far to search for a same-hour value before giving up: eight weeks either
# way. Beyond that the "same hour" is seasonally too far away to stand in.
MAX_WEEK_SEARCH = 8


def _interpolate_short(series, max_linear):
    """Linearly interpolate interior gaps of at most ``max_linear`` hours."""
    missing = series.isna()
    if not missing.any():
        return series, 0

    groups = (missing != missing.shift()).cumsum()
    lengths = missing.groupby(groups).transform("size")
    too_long = missing & lengths.gt(max_linear)

    filled = series.interpolate(method="linear", limit_area="inside")
    filled[too_long] = np.nan

    return filled, int((missing & filled.notna()).sum())


def _fill_same_hour_other_week(series):
    """Fill from the same hour in the nearest available week, either direction."""
    before = series.isna().sum()
    if not before:
        return series, 0

    filled = series
    for weeks in range(1, MAX_WEEK_SEARCH + 1):
        if not filled.isna().any():
            break
        # Prefer the past: it is information a forecaster would actually have.
        filled = filled.fillna(filled.shift(WEEK * weeks))
        filled = filled.fillna(filled.shift(-WEEK * weeks))

    return filled, int(before - filled.isna().sum())


def _fill_hour_of_day_median(series):
    """Fill anything left with the median for that hour of day."""
    before = series.isna().sum()
    if not before:
        return series, 0

    medians = series.groupby(series.index.hour).transform("median")
    filled = series.fillna(medians)

    # A column that is entirely absent has no median; fall back to the overall
    # median, and to zero only if there is genuinely nothing to go on.
    if filled.isna().any():
        filled = filled.fillna(series.median())
    if filled.isna().any():
        filled = filled.fillna(0.0)

    return filled, int(before - filled.isna().sum())


def impute_frame(frame, max_linear=3):
    """Fill every NaN in ``frame``. Returns ``(filled, report)``.

    ``report`` maps each column to the number of values filled by each method,
    so a run can state how much of its input was imputed.
    """
    out = frame.copy()
    report = {}

    for column in out.columns:
        series = out[column]
        total_missing = int(series.isna().sum())
        if not total_missing:
            report[column] = {"missing": 0}
            continue

        series, n_linear = _interpolate_short(series, max_linear)
        series, n_weekly = _fill_same_hour_other_week(series)
        series, n_median = _fill_hour_of_day_median(series)

        out[column] = series
        report[column] = {
            "missing": total_missing,
            "share": round(total_missing / len(out), 6),
            "linear": n_linear,
            "same_hour_other_week": n_weekly,
            "hour_of_day_median": n_median,
            "unfilled": int(series.isna().sum()),
        }

    return out, report


def format_report(report, total_rows):
    """Readable summary of what :func:`impute_frame` filled."""
    lines = []
    for column, counts in report.items():
        if not counts["missing"]:
            lines.append(f"  {column}: complete")
            continue
        lines.append(
            f"  {column}: {counts['missing']:,} of {total_rows:,} hours "
            f"({counts['missing'] / total_rows:.2%}) imputed "
            f"[linear {counts['linear']:,}, "
            f"same-hour-other-week {counts['same_hour_other_week']:,}, "
            f"hour-median {counts['hour_of_day_median']:,}]"
        )
    return "\n".join(lines)
