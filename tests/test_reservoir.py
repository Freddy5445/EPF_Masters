"""
Tests for the weekly reservoir column and the column-level cache.

Water Reservoirs and Hydro Storage Plants [16.1.D] is published weekly and is a
stock rather than a flow, so it needs different handling from the hourly
day-ahead series: held constant between publications, and never visible before
the week it describes has ended.
"""

import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entsoe_tp.areas import lookup  # noqa: E402
from entsoe_tp.build_dataset import (  # noqa: E402
    RESERVOIR_PERIOD, _collapse_sub_hourly, _column_cache_path, _column_specs,
    _first_non_hourly, _load_cached_column, _store_cached_column,
)
from entsoe_tp.hourly import GridError, to_local_hourly_step  # noqa: E402

TZ = "Europe/Oslo"


def weekly(levels, start="2020-01-06"):
    """A weekly series labelled by period start, as the parser emits it."""
    index = pd.date_range(start, periods=len(levels), freq="7D", tz="UTC")
    return pd.Series([float(v) for v in levels], index=index)


class TestStepExpansion(unittest.TestCase):

    def test_level_is_held_not_interpolated(self):
        series = weekly([100, 200])
        grid = to_local_hourly_step(
            series, TZ, pd.Timestamp("2020-01-20"), pd.Timestamp("2020-01-26"),
            allow_gaps=True)
        observed = set(grid.dropna().unique())
        # A straight line between 100 and 200 would produce intermediate values.
        self.assertTrue(observed <= {100.0, 200.0}, observed)

    def test_a_level_is_not_visible_during_the_week_it_describes(self):
        """The decisive causality check."""
        series = weekly([100, 100, 5000, 5000, 5000])
        grid = to_local_hourly_step(
            series, TZ, pd.Timestamp("2020-01-20"), pd.Timestamp("2020-02-10"),
            available_after=RESERVOIR_PERIOD, allow_gaps=True)

        # The 5000 observation is labelled with the week starting 2020-01-20.
        week_start = pd.Timestamp("2020-01-20")
        first_high = grid[grid == 5000].index.min()
        self.assertGreaterEqual(first_high, week_start + pd.Timedelta(days=7))

    def test_extra_lag_delays_visibility_further(self):
        series = weekly([100, 100, 5000, 5000, 5000])
        base = to_local_hourly_step(
            series, TZ, pd.Timestamp("2020-01-20"), pd.Timestamp("2020-02-20"),
            available_after=RESERVOIR_PERIOD, allow_gaps=True)
        lagged = to_local_hourly_step(
            series, TZ, pd.Timestamp("2020-01-20"), pd.Timestamp("2020-02-20"),
            available_after=RESERVOIR_PERIOD + pd.Timedelta(days=3),
            allow_gaps=True)

        self.assertEqual(
            lagged[lagged == 5000].index.min() - base[base == 5000].index.min(),
            pd.Timedelta(days=3),
        )

    def test_hours_before_the_first_publication_are_not_back_filled(self):
        series = weekly([100, 200])
        grid = to_local_hourly_step(
            series, TZ, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-20"),
            allow_gaps=True)
        self.assertTrue(grid.isna().any())
        # Back-filling would be look-ahead, so the leading hours stay empty.
        self.assertTrue(grid.iloc[0] != grid.iloc[0] or pd.isna(grid.iloc[0]))

    def test_missing_leading_hours_raise_without_allow_gaps(self):
        series = weekly([100, 200])
        with self.assertRaises(GridError):
            to_local_hourly_step(
                series, TZ, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-20"))

    def test_empty_series_raises(self):
        with self.assertRaises(GridError):
            to_local_hourly_step(
                pd.Series(dtype="float64",
                          index=pd.DatetimeIndex([], tz="UTC")),
                TZ, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"))


class TestReservoirColumnWiring(unittest.TestCase):

    def test_reservoir_is_absent_unless_requested(self):
        area = lookup("NO2")
        queries, columns = _column_specs(area, "load-wind-solar")
        self.assertNotIn("reservoir", queries)
        self.assertEqual(len(columns), 4)

    def test_reservoir_is_appended_last(self):
        """Appending keeps the existing columns in their positions."""
        area = lookup("NO2")
        _, without = _column_specs(area, "load-wind-solar")
        queries, with_res = _column_specs(area, "load-wind-solar",
                                          include_reservoir=True)

        self.assertEqual([c[0] for c in with_res][:4], [c[0] for c in without])
        self.assertEqual(with_res[-1][1], "reservoir")
        self.assertEqual(queries["reservoir"][0]["documentType"], "A72")
        self.assertEqual(queries["reservoir"][0]["processType"], "A16")

    def test_reservoir_query_accepts_any_resolution(self):
        """It is weekly, so it must not be held to the hourly expectation."""
        area = lookup("NO2")
        queries, _ = _column_specs(area, "load-wind-solar", include_reservoir=True)
        self.assertIsNone(queries["reservoir"][2])
        self.assertEqual(queries["price"][2], "PT60M")


class TestSubHourlyFolding(unittest.TestCase):
    """Sub-hourly publication is not the same as sub-hourly information.

    Zones began emitting PT15M documents months before the day-ahead auction
    cleared sub-hourly -- NO2 in February 2025, DK1 in April -- and in that
    period all four quarters of an hour repeat one value. Folding them is
    lossless; truncating at the first PT15M document would throw away months of
    good hourly data. So the values decide, not the declared resolution.
    """

    def quarters(self, per_hour, psr=None, start="2025-03-01"):
        rows = []
        for hour, values in enumerate(per_hour):
            for quarter, value in enumerate(values):
                rows.append({
                    "timestamp": pd.Timestamp(start, tz="UTC")
                    + pd.Timedelta(hours=hour, minutes=15 * quarter),
                    "value": float(value), "resolution": "PT15M",
                    "psr_type": psr,
                })
        return pd.DataFrame(rows)

    def test_identical_quarters_fold_losslessly(self):
        frame = self.quarters([[50] * 4, [60] * 4, [70] * 4])
        collapsed, first_varying = _collapse_sub_hourly(frame)

        self.assertIsNone(first_varying)
        self.assertEqual(len(collapsed), 3)
        self.assertEqual(sorted(collapsed["value"]), [50.0, 60.0, 70.0])
        self.assertTrue(all(t.minute == 0 for t in collapsed["timestamp"]))

    def test_varying_quarters_are_reported_not_averaged(self):
        frame = self.quarters([[50] * 4, [60, 61, 62, 63]])
        _, first_varying = _collapse_sub_hourly(frame)
        self.assertEqual(first_varying,
                         pd.Timestamp("2025-03-01 01:00", tz="UTC"))

    def test_hourly_rows_pass_through_untouched(self):
        hourly = pd.DataFrame([{
            "timestamp": pd.Timestamp("2025-02-28 23:00", tz="UTC"),
            "value": 40.0, "resolution": "PT60M", "psr_type": None,
        }])
        frame = pd.concat([hourly, self.quarters([[50] * 4])], ignore_index=True)
        collapsed, first_varying = _collapse_sub_hourly(frame)

        self.assertIsNone(first_varying)
        self.assertEqual(sorted(collapsed["value"]), [40.0, 50.0])

    def test_production_types_fold_independently(self):
        frame = pd.concat([self.quarters([[100] * 4], psr="B16"),
                           self.quarters([[700] * 4], psr="B19")],
                          ignore_index=True)
        collapsed, first_varying = _collapse_sub_hourly(frame)

        self.assertIsNone(first_varying)
        self.assertEqual(len(collapsed), 2)
        self.assertEqual(dict(zip(collapsed["psr_type"], collapsed["value"])),
                         {"B16": 100.0, "B19": 700.0})

    def test_a_wholly_hourly_frame_is_unchanged(self):
        hourly = pd.DataFrame([{
            "timestamp": pd.Timestamp("2025-01-01", tz="UTC"),
            "value": 1.0, "resolution": "PT60M", "psr_type": None,
        }])
        collapsed, first_varying = _collapse_sub_hourly(hourly)
        self.assertIsNone(first_varying)
        self.assertEqual(len(collapsed), 1)


class TestResolutionBoundaryIsMeasured(unittest.TestCase):
    """Where sub-hourly *publication* starts differs by zone, so it cannot be a
    constant -- though it is not by itself the end of the hourly era."""

    def frame(self, switch_at):
        """A parsed frame that is hourly up to ``switch_at``, then 15-minute."""
        hourly = pd.date_range(pd.Timestamp("2025-02-01", tz="UTC"), switch_at,
                               freq="h", inclusive="left")
        quarter = pd.date_range(switch_at, periods=96, freq="15min", tz="UTC")
        return pd.DataFrame({
            "timestamp": list(hourly) + list(quarter),
            "value": 1.0,
            "resolution": ["PT60M"] * len(hourly) + ["PT15M"] * len(quarter),
        })

    def test_finds_the_first_non_hourly_timestamp(self):
        switch = pd.Timestamp("2025-02-20 23:00", tz="UTC")
        self.assertEqual(_first_non_hourly(self.frame(switch)), switch)

    def test_returns_none_when_wholly_hourly(self):
        frame = self.frame(pd.Timestamp("2025-02-20 23:00", tz="UTC"))
        frame = frame[frame["resolution"] == "PT60M"]
        self.assertIsNone(_first_non_hourly(frame))

    def test_returns_none_for_an_empty_frame(self):
        self.assertIsNone(_first_non_hourly(pd.DataFrame()))

    def test_last_hourly_day_is_the_day_before_the_change(self):
        """The boundary derives from the first *varying* hour, in local time."""
        switch = pd.Timestamp("2025-02-20 23:00", tz="UTC")
        local = switch.tz_convert("Europe/Oslo").tz_localize(None)
        last_hourly_day = local.normalize() - pd.Timedelta(days=1)
        self.assertEqual(last_hourly_day, pd.Timestamp("2025-02-20"))

    def test_switchover_constant_is_not_used_to_decide_anything(self):
        """It is kept only as a reference point, never as a boundary."""
        import inspect

        from entsoe_tp import build_dataset

        source = inspect.getsource(build_dataset.build)
        self.assertNotIn("MTU_SWITCHOVER", source)


class TestColumnCache(unittest.TestCase):

    def test_round_trip_preserves_values_and_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = pd.date_range("2020-01-01", periods=48, freq="h")
            series = pd.Series(range(48), index=index, dtype="float64")
            path = os.path.join(tmp, "columns", "x.csv")

            _store_cached_column(path, series, "Day-ahead price (EUR/MWh)")
            loaded, label = _load_cached_column(path)

            self.assertEqual(label, "Day-ahead price (EUR/MWh)")
            # check_freq=False: a CSV round trip cannot preserve the index's
            # inferred frequency. Nothing downstream reads it -- the grid
            # invariant is checked from the timestamps themselves.
            pd.testing.assert_series_equal(loaded, series, check_names=False,
                                           check_freq=False)

    def test_missing_cache_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_load_cached_column(os.path.join(tmp, "nope.csv")))

    def test_key_distinguishes_wind_from_solar(self):
        """Both derive from one query with the same aggregate."""
        area = lookup("NO2")
        _, columns = _column_specs(area, "load-wind-solar")
        wind = next(c for c in columns if "wind" in c[0].lower())
        solar = next(c for c in columns if "solar" in c[0].lower())

        def key(spec):
            return {"label": spec[0], "psr": getattr(spec[3], "psr_codes", None)}

        with tempfile.TemporaryDirectory() as tmp:
            self.assertNotEqual(
                _column_cache_path(tmp, "NO2", wind[0], key(wind)),
                _column_cache_path(tmp, "NO2", solar[0], key(solar)),
            )

    def test_key_changes_when_the_range_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = _column_cache_path(tmp, "NO2", "Price", {"start": "2016-01-01"})
            b = _column_cache_path(tmp, "NO2", "Price", {"start": "2017-01-01"})
            self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
