"""
Tests for the LEAR compatibility layer and backtest driver.

These are all offline and fast: they deliberately avoid fitting a real LEAR
model, which takes seconds per day. The numerical path is exercised by the
smoke run (``python run_lear_dk1.py --smoke``) instead.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lear_dk1.backtest import (  # noqa: E402
    TIMING_COLUMNS, _append_timing, _fmt, _load_checkpoint,
)
from lear_dk1.compat import (  # noqa: E402
    LEARCompat, minimum_calibration_window, n_features,
)

HOURS = [f"h{h}" for h in range(24)]


def synthetic_frame(days, start="2020-01-01", n_exogenous=3, seed=0):
    """An epftoolbox-shaped hourly frame: Price plus N exogenous columns."""
    idx = pd.date_range(start, periods=days * 24, freq="h")
    rng = np.random.default_rng(seed)
    hour = idx.hour.values
    data = {"Price": 40 + 20 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 5, len(idx))}
    for n in range(1, n_exogenous + 1):
        data[f"Exogenous {n}"] = 1000 * n + rng.normal(0, 50, len(idx))
    return pd.DataFrame(data, index=idx)


class TestFeatureCounts(unittest.TestCase):
    """LEAR's feature count drives the minimum usable calibration window."""

    def test_feature_count_matches_lear_construction(self):
        # 96 price lags + 72 per exogenous + 7 weekday dummies
        self.assertEqual(n_features(1), 175)
        self.assertEqual(n_features(2), 247)
        self.assertEqual(n_features(3), 319)

    def test_minimum_window_exceeds_feature_count(self):
        for n_exog in (1, 2, 3):
            self.assertGreater(minimum_calibration_window(n_exog), n_features(n_exog))

    def test_paper_short_windows_are_rejected(self):
        """The 56/84-day windows of the original LEAR ensemble cannot be used.

        LassoLarsIC needs more samples than features, and LEAR has at least 175
        features. This is why the ensemble here starts at 364 days.
        """
        for window in (56, 84):
            self.assertLess(window, minimum_calibration_window(1))


class TestCompatSubclass(unittest.TestCase):

    def test_lear_is_loaded_without_importing_tensorflow(self):
        """The compat loader must not pull in TensorFlow.

        epftoolbox/models/__init__.py imports the DNN modules, which do
        ``import tensorflow.keras as kr`` -- a layout Keras 3 removed.
        """
        self.assertNotIn("tensorflow", sys.modules)
        self.assertTrue(issubclass(LEARCompat, object))
        self.assertTrue(hasattr(LEARCompat, "recalibrate_and_forecast_next_day"))

    def test_predict_override_is_ours_not_upstream(self):
        upstream = LEARCompat.__mro__[1]
        self.assertIsNot(LEARCompat.predict, upstream.predict)


class TestCheckpointing(unittest.TestCase):

    def test_missing_checkpoint_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_load_checkpoint(os.path.join(tmp, "nope.csv")))

    def test_partial_rows_are_dropped_on_resume(self):
        """A half-written final row must not be treated as a completed day."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "forecasts.csv")
            frame = pd.DataFrame(
                np.arange(48, dtype="float64").reshape(2, 24),
                index=pd.to_datetime(["2024-01-01", "2024-01-02"]), columns=HOURS,
            )
            frame.loc[pd.Timestamp("2024-01-02"), "h5"] = np.nan
            frame.to_csv(path)

            resumed = _load_checkpoint(path)
            self.assertEqual(len(resumed), 1)
            self.assertEqual(resumed.index[0], pd.Timestamp("2024-01-01"))

    def test_timings_append_with_header_written_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "timings.csv")
            row = {c: 1 for c in TIMING_COLUMNS}
            _append_timing(path, dict(row, run_id="a"))
            _append_timing(path, dict(row, run_id="b"))

            written = pd.read_csv(path)
            self.assertEqual(len(written), 2)
            self.assertEqual(list(written.columns), TIMING_COLUMNS)
            self.assertEqual(list(written["run_id"]), ["a", "b"])


class TestReadDataDateHandling(unittest.TestCase):
    """Regression tests for a silent date-shifting bug.

    ``read_data`` parses date arguments with ``dayfirst=True``. An ISO string
    like ``"2023-10-03"`` is therefore read as 3 October -> ``2023-03-10``,
    silently moving the test period by months. Passing datetime objects avoids
    it, so the backtest must never hand it raw strings.
    """

    def test_iso_string_is_misparsed_by_dayfirst(self):
        # Documents the upstream behaviour this code works around.
        self.assertEqual(
            pd.to_datetime("2023-10-03", dayfirst=True), pd.Timestamp("2023-03-10")
        )

    def test_timestamps_pass_through_unchanged(self):
        for value in ("2023-10-03", "2023-12-01", "2025-09-30"):
            self.assertEqual(
                pd.to_datetime(pd.Timestamp(value), dayfirst=True), pd.Timestamp(value)
            )

    def test_backtest_splits_on_the_requested_day(self):
        """End to end through read_data, with the dates the real run uses."""
        from epftoolbox.data import read_data

        with tempfile.TemporaryDirectory() as tmp:
            frame = synthetic_frame(days=40, start="2023-11-01", n_exogenous=3)
            frame.to_csv(os.path.join(tmp, "T.csv"), date_format="%Y-%m-%dT%H:%M:%S")

            begin = pd.Timestamp("2023-12-01")
            df_train, df_test = read_data(
                path=tmp, dataset="T",
                begin_test_date=begin,
                end_test_date=begin + pd.Timedelta(days=4, hours=23),
            )

            self.assertEqual(df_test.index[0], begin)
            self.assertEqual(len(df_test) // 24, 5)
            self.assertLess(df_train.index[-1], begin)


class TestImputation(unittest.TestCase):
    """Gaps are filled at model time, so the method is a stated choice."""

    def periodic_frame(self, days=90):
        """A series with a strong daily and weekly shape, so a bad fill shows."""
        idx = pd.date_range("2015-01-07", periods=days * 24, freq="h")
        hour, dow = idx.hour.values, idx.dayofweek.values
        base = 50 + 25 * np.sin(2 * np.pi * (hour - 8) / 24) - 12 * (dow >= 5)
        return pd.DataFrame({"Price": base, "Load": 3000 + base * 10}, index=idx)

    def test_everything_is_filled(self):
        from lear_dk1.impute import impute_frame

        truth = self.periodic_frame()
        gappy = truth.copy()
        gappy.iloc[100:102] = np.nan
        gappy.iloc[1000:1072] = np.nan

        filled, report = impute_frame(gappy)
        self.assertFalse(filled.isna().any().any())
        self.assertEqual(report["Price"]["unfilled"], 0)

    def test_short_gaps_use_linear_and_long_gaps_do_not(self):
        from lear_dk1.impute import impute_frame

        gappy = self.periodic_frame()
        gappy.iloc[100:102] = np.nan     # 2h  -> linear
        gappy.iloc[1000:1072] = np.nan   # 72h -> same hour, other week

        _, report = impute_frame(gappy, max_linear=3)
        self.assertEqual(report["Price"]["linear"], 2)
        self.assertEqual(report["Price"]["same_hour_other_week"], 72)

    def test_long_gap_preserves_the_daily_shape(self):
        """The point of the weekly fill: a straight line would flatten the cycle."""
        from lear_dk1.impute import impute_frame

        truth = self.periodic_frame()
        gappy = truth.copy()
        gappy.iloc[1000:1072] = np.nan

        filled, _ = impute_frame(gappy)
        weekly_error = (filled["Price"].iloc[1000:1072]
                        - truth["Price"].iloc[1000:1072]).abs().max()
        linear_error = (gappy["Price"].interpolate("linear").iloc[1000:1072]
                        - truth["Price"].iloc[1000:1072]).abs().max()

        self.assertLess(weekly_error, 1.0)
        self.assertGreater(linear_error, 20.0)

    def test_leading_gap_is_filled_from_a_later_week(self):
        from lear_dk1.impute import impute_frame

        truth = self.periodic_frame()
        gappy = truth.copy()
        gappy.iloc[0:5] = np.nan

        filled, report = impute_frame(gappy)
        self.assertFalse(filled.isna().any().any())
        # limit_area="inside" cannot touch a leading gap, so linear fills none.
        self.assertEqual(report["Price"]["linear"], 0)

    def test_complete_column_is_left_alone(self):
        from lear_dk1.impute import impute_frame

        truth = self.periodic_frame(days=30)
        filled, report = impute_frame(truth)
        pd.testing.assert_frame_equal(filled, truth)
        self.assertEqual(report["Price"]["missing"], 0)


class TestFormatting(unittest.TestCase):

    def test_duration_formatting(self):
        self.assertEqual(_fmt(0), "0:00:00")
        self.assertEqual(_fmt(65), "0:01:05")
        self.assertEqual(_fmt(3600), "1:00:00")


if __name__ == "__main__":
    unittest.main()
