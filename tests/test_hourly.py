"""
DST handling tests.

These pin the convention the shipped ``datasets/NP.csv`` uses, because that is
what ``epftoolbox`` silently assumes: exactly 24 rows per calendar day, with the
spring-forward hour interpolated and the two fall-back hours averaged.
"""

import os
import sys
import unittest
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entsoe_tp.hourly import (  # noqa: E402
    GridError, assert_epftoolbox_grid, to_local_hourly_grid,
)

TZ = "Europe/Copenhagen"


def utc_series(start, hours, value=1.0):
    """A dense hourly UTC series of ``hours`` points."""
    index = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    return pd.Series([value] * hours, index=index)


class TestSpringForward(unittest.TestCase):
    """2016-03-27: local time jumps 02:00 -> 03:00, so the day has 23 hours."""

    def setUp(self):
        # 23:00Z on the 26th is 00:00 local on the 27th.
        index = pd.date_range("2016-03-26T23:00Z", periods=23, freq="h", tz="UTC")
        # Ramp so interpolation is checkable.
        self.series = pd.Series(range(23), index=index, dtype="float64")

    def test_day_has_24_rows(self):
        grid = to_local_hourly_grid(self.series, TZ, "2016-03-27", "2016-03-27")
        self.assertEqual(len(grid), 24)
        self.assertEqual(grid.index[0], pd.Timestamp("2016-03-27 00:00:00"))
        self.assertEqual(grid.index[-1], pd.Timestamp("2016-03-27 23:00:00"))

    def test_missing_hour_is_interpolated(self):
        grid = to_local_hourly_grid(self.series, TZ, "2016-03-27", "2016-03-27")
        # 02:00 local never happened; it must be the mean of 01:00 and 03:00,
        # matching how NP.csv fills 2018-03-25 02:00.
        self.assertAlmostEqual(
            grid.loc["2016-03-27 02:00:00"],
            (grid.loc["2016-03-27 01:00:00"] + grid.loc["2016-03-27 03:00:00"]) / 2,
        )

    def test_no_spurious_warning(self):
        """The DST hour is expected, so it must not warn about a data gap."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            to_local_hourly_grid(self.series, TZ, "2016-03-27", "2016-03-27")


class TestFallBack(unittest.TestCase):
    """2016-10-30: local 02:00 happens twice, so the day has 25 hours."""

    def setUp(self):
        index = pd.date_range("2016-10-29T22:00Z", periods=25, freq="h", tz="UTC")
        self.series = pd.Series([10.0] * 25, index=index)
        # The two repeated local 02:00 hours are UTC 00:00 and 01:00 on the 30th.
        self.series.loc["2016-10-30T00:00Z"] = 40.0
        self.series.loc["2016-10-30T01:00Z"] = 60.0

    def test_day_has_24_rows(self):
        grid = to_local_hourly_grid(self.series, TZ, "2016-10-30", "2016-10-30")
        self.assertEqual(len(grid), 24)

    def test_duplicate_hour_is_averaged(self):
        grid = to_local_hourly_grid(self.series, TZ, "2016-10-30", "2016-10-30")
        # NP.csv averages the pair: 2018-10-28 02:00 holds 41.605.
        self.assertAlmostEqual(grid.loc["2016-10-30 02:00:00"], 50.0)

    def test_index_has_no_duplicates(self):
        grid = to_local_hourly_grid(self.series, TZ, "2016-10-30", "2016-10-30")
        self.assertFalse(grid.index.has_duplicates)


class TestGaps(unittest.TestCase):
    def test_short_gap_is_interpolated_with_a_warning(self):
        series = utc_series("2016-06-01T22:00Z", 24, value=5.0)
        series.iloc[5] = float("nan")
        series = series.dropna()

        with self.assertWarns(UserWarning):
            grid = to_local_hourly_grid(series, TZ, "2016-06-02", "2016-06-02")

        self.assertEqual(len(grid), 24)
        self.assertFalse(grid.isna().any())

    def test_long_gap_raises(self):
        series = utc_series("2016-06-01T22:00Z", 24, value=5.0)
        series = series.drop(series.index[5:12])

        with self.assertRaises(GridError):
            to_local_hourly_grid(series, TZ, "2016-06-02", "2016-06-02")

    def test_missing_edge_raises(self):
        """Interpolation must not extrapolate past the end of the data."""
        series = utc_series("2016-06-01T22:00Z", 12, value=5.0)
        with self.assertRaises(GridError):
            to_local_hourly_grid(series, TZ, "2016-06-02", "2016-06-02")


class TestInvariant(unittest.TestCase):
    def _frame(self, periods):
        index = pd.date_range("2016-01-01", periods=periods, freq="h")
        return pd.DataFrame({"Price": range(periods)}, index=index, dtype="float64")

    def test_valid_frame_passes(self):
        assert_epftoolbox_grid(self._frame(48))

    def test_partial_day_raises(self):
        with self.assertRaises(GridError):
            assert_epftoolbox_grid(self._frame(30))

    def test_tz_aware_index_raises(self):
        frame = self._frame(24)
        frame.index = frame.index.tz_localize("UTC")
        with self.assertRaises(GridError):
            assert_epftoolbox_grid(frame)

    def test_nan_raises(self):
        frame = self._frame(24)
        frame.iloc[3, 0] = float("nan")
        with self.assertRaises(GridError):
            assert_epftoolbox_grid(frame)

    def test_duplicate_index_raises(self):
        frame = self._frame(24)
        frame = pd.concat([frame, frame.iloc[[0]]]).sort_index()
        with self.assertRaises(GridError):
            assert_epftoolbox_grid(frame)

    def test_non_nanosecond_index_passes(self):
        """A microsecond-resolution index -- what a parquet-backed panel

        produces on some pandas builds -- must not spuriously fail the hourly
        step check just because its numpy dtype differs from the nanosecond
        default used to build the comparison value.
        """
        frame = self._frame(24)
        frame.index = frame.index.as_unit("us")
        assert_epftoolbox_grid(frame)

    def test_irregular_step_raises_regardless_of_resolution(self):
        index = pd.date_range("2016-01-01", periods=24, freq="h").as_unit("us")
        frame = pd.DataFrame(
            {"Price": range(23)}, index=index.delete(5), dtype="float64"
        )
        with self.assertRaises(GridError):
            assert_epftoolbox_grid(frame)


if __name__ == "__main__":
    unittest.main()
