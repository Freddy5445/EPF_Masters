"""
Turning UTC series into the hourly grid ``epftoolbox`` requires.

``epftoolbox.data.read_data`` does no DST, gap or duplicate handling whatsoever --
it calls ``pd.to_datetime`` on the index and stops. The invariant is enforced by
the *data files* instead. Inspecting the shipped ``datasets/NP.csv`` shows the
convention the Zenodo datasets follow:

* the index is **naive local market time**, formatted ``%Y-%m-%d %H:%M:%S``;
* there are **exactly 24 rows per calendar day**, always 00:00 through 23:00;
* on the spring-forward day the missing 02:00 is **interpolated in** (2018-03-25
  carries 38.365, the mean of the 01:00 and 03:00 values);
* on the fall-back day the two 02:00 hours are **averaged into one** row.

This module reproduces that convention exactly. Everything downstream depends on
it: ``_lear.py`` reshapes the test set with ``reshape(-1, 24)`` and both models
look features up by exact timestamp label, so a missing hour raises ``KeyError``
and a duplicated one corrupts the array shape.
"""

import warnings

import pandas as pd


class GridError(ValueError):
    """Raised when a series cannot be coerced to the required hourly grid."""


def to_local_hourly_grid(series_utc, tz, start_date, end_date, max_gap=3,
                         allow_gaps=False):
    """Convert a UTC-indexed series to a naive local hourly grid.

    ``start_date`` and ``end_date`` are dates (inclusive); the result spans
    ``start_date 00:00`` to ``end_date 23:00`` in ``tz``, giving exactly 24 rows
    per calendar day.

    Fall-back duplicates are averaged, the spring-forward hour is interpolated,
    and any remaining gap up to ``max_gap`` hours is interpolated with a warning.
    A longer gap raises :class:`GridError` rather than quietly inventing data.

    With ``allow_gaps=True`` **nothing is interpolated at all** and ``max_gap`` is
    ignored: every hour the platform did not publish is left as NaN, including
    the spring-forward hour. This turns the function into a pure acquisition
    step -- what came back, on the hourly grid, and nothing invented -- so that
    how to treat missing data stays a separate, later decision.

    The one transformation still applied is fall-back averaging: a naive local
    grid has a single 02:00 slot on the autumn DST day, so the two observations
    that share it are averaged. That aggregates real values rather than
    inventing any, and cannot be avoided while keeping 24 rows per day.

    The result violates the epftoolbox no-NaN invariant, so it cannot be fed to
    a model as-is.
    """
    if series_utc.empty:
        raise GridError("No data returned for the requested range")

    local = series_utc.tz_convert(tz)

    # Dropping the offset collapses the two fall-back 02:00 hours onto one label;
    # the groupby then averages them, matching the NP.csv convention.
    naive = local.tz_localize(None)
    naive = naive.groupby(level=0).mean()

    grid = pd.date_range(
        pd.Timestamp(start_date).normalize(),
        pd.Timestamp(end_date).normalize() + pd.Timedelta(hours=23),
        freq="h",
    )

    aligned = naive.reindex(grid)

    if allow_gaps:
        # Pure acquisition: exactly what was published, on the grid, nothing filled.
        return aligned

    # Spring-forward hours do not exist in local time, so no amount of data would
    # fill them; they are expected NaNs. Anything else is a real gap.
    localized = grid.tz_localize(tz, ambiguous=True, nonexistent="NaT")
    dst_missing = grid[localized.isna()]

    missing = aligned.index[aligned.isna()]
    real_gaps = missing.difference(dst_missing)

    if len(real_gaps):
        _check_gap_lengths(real_gaps, max_gap)
        warnings.warn(
            f"Interpolating {len(real_gaps)} missing hour(s) not explained by DST, "
            f"first at {real_gaps[0]}, last at {real_gaps[-1]}",
            stacklevel=2,
        )

    filled = aligned.interpolate(method="linear", limit_area="inside")

    if filled.isna().any():
        edge = filled.index[filled.isna()]
        raise GridError(
            f"{len(edge)} hour(s) at the edges of the range have no data and cannot "
            f"be interpolated (first {edge[0]}, last {edge[-1]}). Narrow the "
            f"requested range to where the platform actually publishes."
        )

    return filled



def to_local_hourly_step(series_utc, tz, start_date, end_date,
                         available_after=pd.Timedelta(days=7), allow_gaps=False):
    """Place a coarse *stock* series onto the hourly grid as a step function.

    Water reservoir filling [16.1.D] is published weekly, and is a stock rather
    than a flow: a level, not a rate. Two things follow.

    First, it must be held constant between publications, not interpolated.
    Interpolating would invent a smooth trajectory the data does not contain --
    and, worse, would read from the *next* observation, which is future
    information.

    Second, an observation covering a week cannot be known before that week is
    over. ``available_after`` shifts each observation forward by that much
    before it is allowed to influence the grid, so hours only ever see levels
    that had actually been published by then. It defaults to the ``P7D``
    publication period; add more to model the platform's own reporting delay.

    Hours before the first available observation are left as NaN: there is no
    earlier level to carry forward, and back-filling would be look-ahead.
    """
    if series_utc.empty:
        raise GridError("No data returned for the requested range")

    # Delay availability, then move to the naive local grid.
    delayed = series_utc.copy()
    delayed.index = delayed.index + available_after

    local = delayed.tz_convert(tz).tz_localize(None)
    local = local.groupby(level=0).mean().sort_index()

    grid = pd.date_range(
        pd.Timestamp(start_date).normalize(),
        pd.Timestamp(end_date).normalize() + pd.Timedelta(hours=23),
        freq="h",
    )

    # Carry each level forward to every hour until the next publication. Union
    # first so observations landing between grid points are not lost, then
    # forward-fill and drop back to the grid. ffill only -- never bfill.
    combined = local.reindex(local.index.union(grid)).sort_index().ffill()
    stepped = combined.reindex(grid)

    if stepped.isna().any() and not allow_gaps:
        missing = stepped.index[stepped.isna()]
        raise GridError(
            f"{len(missing)} hour(s) precede the first published value that is "
            f"available by then (first {missing[0]}, last {missing[-1]}). Start "
            f"the range later, or widen it so an earlier publication exists."
        )

    return stepped


def _check_gap_lengths(gap_index, max_gap):
    """Raise if any run of consecutive missing hours exceeds ``max_gap``."""
    run_start = previous = gap_index[0]
    run_length = 1

    for timestamp in gap_index[1:]:
        if timestamp - previous == pd.Timedelta(hours=1):
            run_length += 1
        else:
            run_start, run_length = timestamp, 1
        if run_length > max_gap:
            raise GridError(
                f"Gap of more than {max_gap} consecutive hours starting at "
                f"{run_start}; refusing to interpolate across it"
            )
        previous = timestamp


def assert_epftoolbox_grid(frame, allow_nan=False):
    """Validate the invariant ``read_data`` and the models silently assume.

    Raises :class:`GridError` on violation. Failing here is far cheaper than the
    downstream symptoms -- a ``KeyError`` deep in feature construction or a
    reshape error on a non-multiple-of-24 test set.
    """
    index = frame.index

    if not isinstance(index, pd.DatetimeIndex):
        raise GridError(f"Index is {type(index).__name__}, expected DatetimeIndex")
    if index.tz is not None:
        raise GridError("Index must be timezone-naive local market time")
    if not index.is_monotonic_increasing:
        raise GridError("Index is not sorted")
    if index.has_duplicates:
        dupes = index[index.duplicated()]
        raise GridError(f"Index has {len(dupes)} duplicate timestamps, first {dupes[0]}")

    deltas = index.to_series().diff().dropna().unique()
    if len(deltas) and set(deltas) != {pd.Timedelta(hours=1).to_timedelta64()}:
        raise GridError(f"Index is not strictly hourly (found steps {list(deltas)})")

    if len(index) % 24:
        raise GridError(f"{len(index)} rows is not a whole number of 24-hour days")
    if index[0].hour or index[-1].hour != 23:
        raise GridError(
            f"Range must start at 00:00 and end at 23:00, got {index[0]} to {index[-1]}"
        )

    # The shape invariants above always hold. NaN is the one thing a caller may
    # deliberately accept, to inspect partial data before deciding what to do.
    if not allow_nan and frame.isna().any().any():
        counts = frame.isna().sum()
        raise GridError(f"Frame still contains NaN values: {counts.to_dict()}")
