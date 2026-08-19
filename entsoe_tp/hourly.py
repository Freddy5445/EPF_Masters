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


def to_local_hourly_grid(series_utc, tz, start_date, end_date, max_gap=3):
    """Convert a UTC-indexed series to a naive local hourly grid.

    ``start_date`` and ``end_date`` are dates (inclusive); the result spans
    ``start_date 00:00`` to ``end_date 23:00`` in ``tz``, giving exactly 24 rows
    per calendar day.

    Fall-back duplicates are averaged, the spring-forward hour is interpolated,
    and any remaining gap up to ``max_gap`` hours is interpolated with a warning.
    A longer gap raises :class:`GridError` rather than quietly inventing data.
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


def assert_epftoolbox_grid(frame):
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

    if frame.isna().any().any():
        counts = frame.isna().sum()
        raise GridError(f"Frame still contains NaN values: {counts.to_dict()}")
