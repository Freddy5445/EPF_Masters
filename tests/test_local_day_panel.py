"""Tests for the shared UTC-to-local delivery-day conversion."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from local_day_panel import (
    LocalDayPanelError,
    build_local_day_matrices,
    flatten_local_day_matrix,
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

    def test_full_dataset_has_all_transition_days_and_complete_edges(self):
        path = Path(__file__).parents[1] / "datasets" / "nordic_baltic_clean_hourly_utc.parquet"
        panel = pd.read_parquet(path)
        matrices, report = build_local_day_matrices(panel)

        expected_spring = tuple(pd.to_datetime([
            "2019-03-31", "2020-03-29", "2021-03-28", "2022-03-27",
            "2023-03-26", "2024-03-31", "2025-03-30",
        ]))
        expected_autumn = tuple(pd.to_datetime([
            "2019-10-27", "2020-10-25", "2021-10-31", "2022-10-30",
            "2023-10-29", "2024-10-27",
        ]))

        self.assertEqual(report.spring_days, expected_spring)
        self.assertEqual(report.autumn_days, expected_autumn)
        self.assertEqual(report.series_count, 29)
        for matrix in matrices.values():
            self.assertEqual(matrix.shape, (2465, 24))
            self.assertTrue(matrix.iloc[[0, -1]].notna().all(axis=None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
