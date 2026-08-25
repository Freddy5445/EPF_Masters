"""
Daylight-saving handling, following the epftoolbox datasets.

Both models require exactly 24 rows per calendar day: ``_lear.py`` reshapes with
``reshape(-1, 24)``, and features are looked up by exact timestamp label. Two
days a year are not 24 hours long in local market time:

* **spring forward** (last Sunday in March) -- the clock jumps 02:00 to 03:00,
  so the day has 23 hours and there is no 02:00;
* **fall back** (last Sunday in October) -- the clock repeats 02:00, so the day
  has 25 hours.

The published epftoolbox datasets resolve this by forcing 24 slots and patching
the two days. Inspecting ``NP.csv`` shows the convention, and it is confirmed by
the real prices shipped with the NP forecasts: every one of the 17,472 test rows
fits 728 x 24 exactly, and on 2017-03-26 and 2018-03-25 the 02:00 value is the
mean of its 01:00 and 03:00 neighbours to the last decimal.

* **fall back** -- the two 02:00 observations are averaged into one row. Done in
  :func:`entsoe_tp.hourly.to_local_hourly_grid`, which collapses the duplicate
  labels; it aggregates real values rather than inventing any.
* **spring forward** -- the missing hour is linearly interpolated from its
  neighbours. That is what this module adds.

Note what interpolation costs: it reads 03:00 to fill 02:00, so it looks two
hours forward. Everywhere else this project fills strictly from the past. The
exception is deliberate -- it is the published convention, it affects one hour
per zone per year (0.011% of a 728-day test period), and matching it keeps
results comparable with the paper's. The filled value is then treated as
ordinary data, exactly as epftoolbox treats it: nothing marks it and nothing
excludes it from scoring.
"""

import pandas as pd


def skipped_hours(index, tz):
    """Boolean mask of naive local timestamps the clock skipped.

    ``nonexistent="NaT"`` marks exactly the hours that do not exist in ``tz``.
    ``ambiguous=True`` resolves the autumn repeated hour to the first of the
    pair, so a fall-back hour -- which does exist, twice -- is never marked.

    The hour differs by zone: 02:00 under CET, 03:00 under EET. Detecting it
    from the timezone rather than hardcoding a label keeps the Baltic and
    Finnish zones correct if they are ever restored to the panel.
    """
    localised = pd.DatetimeIndex(index).tz_localize(
        tz, ambiguous=True, nonexistent="NaT")
    return pd.isna(localised)


def fill_skipped_hours(frame, tz):
    """Interpolate the spring-forward hour, the epftoolbox way.

    Only the skipped hours are touched. Every other gap is left alone for the
    causal imputation to handle, so this cannot quietly interpolate across a
    real outage -- which would be both acausal and much larger than one hour.

    Returns ``(filled, n_filled)``.
    """
    mask = skipped_hours(frame.index, tz)
    if not mask.any():
        return frame, 0

    # limit_area="inside" refuses to extrapolate, so a skipped hour at the very
    # start or end of the frame -- with no neighbour on one side -- stays NaN
    # rather than being filled from one side only.
    interpolated = frame.interpolate(method="linear", limit=1, limit_area="inside")

    filled = frame.copy()
    rows = frame.index[mask]
    was_missing = frame.loc[rows].isna()
    filled.loc[rows] = interpolated.loc[rows]

    n = int((was_missing & filled.loc[rows].notna()).to_numpy().sum())
    return filled, n
