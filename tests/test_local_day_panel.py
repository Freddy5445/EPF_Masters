"""Tests for the shared UTC-to-local delivery-day conversion."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from local_day_panel import (
    LocalDayPanelError,
    build_local_day_matrices,
    flatten_local_day_matrix,
    normalize_local_hourly_panel,
)


class TestLocalDayPanel(unittest.TestCase):
    def _panel(self, start="2024-03-30 23:00", end="2024-04-01 21:00"):
        timestamps = pd.date_range(start, end, freq="h", tz="UTC")
        return pd.DataFrame(
            {
                "series": "DK1 | price",
                "timestamp_utc": timestamps,
                "value": np.arange(len(timestamps), dtype=float),
            }
        )

    def test_spring_hour_is_interpolated(self):
        matrices, report = build_local_day_matrices(self._panel())
        matrix = matrices["DK1 | price"]
        day = pd.Timestamp("2024-03-31")

        self.assertEqual(report.spring_days, (day,))
        self.assertAlmostEqual(matrix.loc[day, 2],
                               (matrix.loc[day, 1] + matrix.loc[day, 3]) / 2)
        self.assertEqual(matrix.shape, (2, 24))

    def test_autumn_repeated_hour_is_averaged(self):
        panel = self._panel("2024-10-25 22:00", "2024-10-27 22:00")
        local = panel.timestamp_utc.dt.tz_convert("Europe/Copenhagen")
        repeated_values = panel.loc[
            (local.dt.tz_localize(None).dt.normalize() == pd.Timestamp("2024-10-27"))
            & (local.dt.hour == 2),
            "value",
        ]

        matrices, report = build_local_day_matrices(panel)
        day = pd.Timestamp("2024-10-27")

        self.assertEqual(report.autumn_days, (day,))
        self.assertAlmostEqual(matrices["DK1 | price"].loc[day, 2],
                               repeated_values.mean())
        self.assertEqual(matrices["DK1 | price"].shape, (2, 24))

    def test_real_gap_is_rejected(self):
        panel = self._panel().drop(index=10)
        with self.assertRaisesRegex(LocalDayPanelError, "not explained by spring DST"):
            build_local_day_matrices(panel)

    def test_flattened_matrix_is_gapless_naive_local_time(self):
        matrix = build_local_day_matrices(self._panel())[0]["DK1 | price"]
        flattened = flatten_local_day_matrix(matrix)

        self.assertIsNone(flattened.index.tz)
        self.assertTrue(flattened.index.to_series().diff().dropna()
                        .eq(pd.Timedelta(hours=1)).all())
        self.assertEqual(len(flattened), len(matrix) * 24)

    def test_normalized_long_panel_records_dst_adjustments(self):
        local, report = normalize_local_hourly_panel(self._panel())

        self.assertNotIn("timestamp_utc", local.columns)
        self.assertIsNone(local["timestamp_local"].dt.tz)
        self.assertEqual((local["dst_adjustment"] == "spring_interpolation").sum(), 1)
        self.assertEqual(report.spring_days, (pd.Timestamp("2024-03-31"),))

    def test_autumn_average_preserves_either_source_imputation(self):
        panel = self._panel("2024-10-25 22:00", "2024-10-27 22:00")
        panel["imputed"] = False
        panel["imputation_method"] = "observed"
        panel["imputation_predictors"] = ""
        local_time = panel.timestamp_utc.dt.tz_convert("Europe/Copenhagen")
        repeated = (local_time.dt.tz_localize(None) == pd.Timestamp("2024-10-27 02:00"))
        second = panel.index[repeated][-1]
        panel.loc[second, ["imputed", "imputation_method"]] = [True, "causal_cross_zone_ols"]

        local, _ = normalize_local_hourly_panel(panel)
        row = local.loc[local.timestamp_local == pd.Timestamp("2024-10-27 02:00")].iloc[0]

        self.assertTrue(row.imputed)
        self.assertEqual(row.imputation_method, "causal_cross_zone_ols")
        self.assertEqual(row.dst_adjustment, "autumn_average")

    def test_full_dataset_has_all_transition_days_and_complete_edges(self):
        path = Path(__file__).parents[1] / "datasets" / "nordic_baltic_clean_hourly_local.parquet"
        panel = pd.read_parquet(path)

        expected_spring = tuple(pd.to_datetime([
            "2019-03-31", "2020-03-29", "2021-03-28", "2022-03-27",
            "2023-03-26", "2024-03-31", "2025-03-30",
        ]))
        expected_autumn = tuple(pd.to_datetime([
            "2019-10-27", "2020-10-25", "2021-10-31", "2022-10-30",
            "2023-10-29", "2024-10-27",
        ]))

        spring = tuple(pd.DatetimeIndex(panel.loc[
            panel.dst_adjustment == "spring_interpolation", "timestamp_local"
        ].dt.normalize().unique()).sort_values())
        autumn = tuple(pd.DatetimeIndex(panel.loc[
            panel.dst_adjustment == "autumn_average", "timestamp_local"
        ].dt.normalize().unique()).sort_values())
        per_day = panel.groupby(
            ["series", panel.timestamp_local.dt.normalize()], observed=True
        ).size()

        self.assertEqual(spring, expected_spring)
        self.assertEqual(autumn, expected_autumn)
        self.assertEqual(panel.series.nunique(), 29)
        self.assertTrue(per_day.eq(24).all())
        self.assertFalse(panel.duplicated(["series", "timestamp_local"]).any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
