"""Tests for the single cleaning implementation.

The point of the `cleaning` package is that every model reads identically
prepared inputs, so what is pinned here is the *convention*: 24 rows per day
always, the epftoolbox DST treatment, and imputation that never looks forward.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cleaning import clean_panel, clean_zone, format_report  # noqa: E402
from cleaning.dst import fill_skipped_hours, skipped_hours  # noqa: E402


def raw_panel(zones=("DK1",), start="2024-01-01", end="2024-12-31 23:00", seed=0):
    """A raw-dump-shaped panel spanning both 2024 DST transitions."""
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    rows = []
    for zone in zones:
        for variable, psr in (("price", ""), ("load_forecast", ""),
                              ("generation_forecast", "wind_onshore")):
            rows.append(pd.DataFrame({
                "timestamp_utc": idx, "zone": zone, "variable": variable,
                "psr_type": psr,
                "value": 40 + 10 * np.sin(2 * np.pi * idx.hour.to_numpy() / 24)
                         + rng.normal(0, 3, len(idx)),
            }))
    return pd.concat(rows, ignore_index=True)


class TestSkippedHourDetection(unittest.TestCase):
    """Which hour the clock skips depends on the zone, so it is derived."""

    def test_cet_skips_0200(self):
        grid = pd.date_range("2024-03-31", periods=24, freq="h")
        mask = skipped_hours(grid, "Europe/Copenhagen")
        self.assertEqual(list(grid[mask].hour), [2])

    def test_eet_skips_0300(self):
        """A rule hardcoded to 02:00 would silently miss FI and the Baltics."""
        grid = pd.date_range("2024-03-31", periods=24, freq="h")
        mask = skipped_hours(grid, "Europe/Helsinki")
        self.assertEqual(list(grid[mask].hour), [3])

    def test_fall_back_is_not_marked(self):
        """The repeated autumn hour exists twice; it is averaged, not filled."""
        grid = pd.date_range("2024-10-27", periods=24, freq="h")
        self.assertEqual(int(skipped_hours(grid, "Europe/Copenhagen").sum()), 0)

    def test_ordinary_days_are_not_marked(self):
        grid = pd.date_range("2024-06-01", periods=24, freq="h")
        self.assertEqual(int(skipped_hours(grid, "Europe/Copenhagen").sum()), 0)

    def test_one_skipped_hour_per_year(self):
        grid = pd.date_range("2015-01-01", "2024-12-31 23:00", freq="h")
        self.assertEqual(int(skipped_hours(grid, "Europe/Copenhagen").sum()), 10)


class TestDSTFill(unittest.TestCase):
    """epftoolbox interpolates the skipped hour from its neighbours."""

    def _frame(self):
        idx = pd.date_range("2024-03-30", "2024-04-01 23:00", freq="h")
        f = pd.DataFrame({"Price": np.arange(len(idx), dtype=float)}, index=idx)
        f.loc[skipped_hours(idx, "Europe/Copenhagen"), "Price"] = np.nan
        return f

    def test_skipped_hour_is_the_mean_of_its_neighbours(self):
        """The convention verified against NP.csv: 02:00 = mean(01:00, 03:00)."""
        filled, n = fill_skipped_hours(self._frame(), "Europe/Copenhagen")
        self.assertEqual(n, 1)
        hour = pd.Timestamp("2024-03-31 02:00")
        self.assertAlmostEqual(
            filled.loc[hour, "Price"],
            np.mean([filled.loc[pd.Timestamp("2024-03-31 01:00"), "Price"],
                     filled.loc[pd.Timestamp("2024-03-31 03:00"), "Price"]]),
            places=10)

    def test_other_gaps_are_left_alone(self):
        """Only the clock artefact is interpolated; a real outage is not.

        Interpolation looks forward, so applying it to genuine gaps would be
        both acausal and unbounded in size.
        """
        frame = self._frame()
        frame.iloc[10:14] = np.nan
        filled, _ = fill_skipped_hours(frame, "Europe/Copenhagen")
        self.assertEqual(int(filled.iloc[10:14].isna().sum().sum()), 4)

    def test_nothing_to_do_is_a_no_op(self):
        idx = pd.date_range("2024-06-01", periods=48, freq="h")
        frame = pd.DataFrame({"Price": np.arange(48, dtype=float)}, index=idx)
        filled, n = fill_skipped_hours(frame, "Europe/Copenhagen")
        self.assertEqual(n, 0)
        pd.testing.assert_frame_equal(filled, frame)


class TestCleanZone(unittest.TestCase):

    def test_every_day_has_exactly_24_rows(self):
        """Both models reshape(-1, 24); a 23- or 25-hour day breaks them."""
        panel = raw_panel()
        frame, _ = clean_zone(panel[panel.zone == "DK1"], "DK1")
        per_day = frame.groupby(frame.index.normalize()).size()
        self.assertEqual(sorted(per_day.unique()), [24])

    def test_result_has_no_gaps(self):
        panel = raw_panel()
        frame, report = clean_zone(panel[panel.zone == "DK1"], "DK1")
        self.assertFalse(frame.isna().any().any())
        self.assertTrue(report["complete"])

    def test_long_gaps_are_filled_and_counted(self):
        panel = raw_panel()
        price = panel[(panel.zone == "DK1") & (panel.variable == "price")]
        order = price.sort_values("timestamp_utc").index
        panel.loc[order[5000:5030], "value"] = np.nan

        frame, report = clean_zone(panel[panel.zone == "DK1"], "DK1")
        self.assertFalse(frame.isna().any().any())
        self.assertGreaterEqual(report["imputation"]["price"]["missing"], 30)

    def test_dst_hours_are_reported_separately_from_imputation(self):
        """The write-up has to distinguish a clock artefact from missing data."""
        panel = raw_panel()
        _, report = clean_zone(panel[panel.zone == "DK1"], "DK1")
        # One skipped hour in 2024, times three series.
        self.assertEqual(report["dst_hours_interpolated"], 3)


class TestCleanPanel(unittest.TestCase):

    def test_shape_and_columns_survive(self):
        cleaned, report = clean_panel(raw_panel(zones=("DK1", "NO1")), quiet=True)
        self.assertEqual(sorted(cleaned.zone.unique()), ["DK1", "NO1"])
        self.assertEqual(list(cleaned.columns),
                         ["timestamp_local", "zone", "variable", "psr_type", "value"])
        self.assertFalse(cleaned.value.isna().any())
        self.assertEqual(len(report["zones"]), 2)

    def test_psr_type_round_trips(self):
        """The wind components must stay distinguishable after cleaning."""
        cleaned, _ = clean_panel(raw_panel(), quiet=True)
        wind = cleaned[cleaned.variable == "generation_forecast"]
        self.assertEqual(sorted(wind.psr_type.unique()), ["wind_onshore"])

    def test_missing_columns_are_named(self):
        bad = raw_panel().drop(columns=["psr_type"])
        with self.assertRaises(ValueError) as caught:
            clean_panel(bad, quiet=True)
        self.assertIn("psr_type", str(caught.exception))

    def test_unknown_zone_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            clean_panel(raw_panel(), zones=["XX9"], quiet=True)
        self.assertIn("XX9", str(caught.exception))

    def test_report_formats(self):
        _, report = clean_panel(raw_panel(), quiet=True)
        text = format_report(report)
        self.assertIn("DK1", text)


class TestNoLookAhead(unittest.TestCase):
    """Imputation must never read the future. DST interpolation is the one
    documented exception, and is confined to the skipped hour."""

    def test_a_gap_is_filled_only_from_the_past(self):
        panel = raw_panel(end="2024-06-30 23:00")
        price = panel[(panel.zone == "DK1") & (panel.variable == "price")]
        order = price.sort_values("timestamp_utc").index

        # The hours that will be blanked, as naive local labels.
        gap_utc = panel.loc[order[3000:3010], "timestamp_utc"]
        gap_local = gap_utc.dt.tz_convert("Europe/Copenhagen").dt.tz_localize(None)

        panel.loc[order[3000:3010], "value"] = np.nan
        frame_a, _ = clean_zone(panel[panel.zone == "DK1"], "DK1")

        # Now move every observation *after* the gap by a large amount. A causal
        # filler cannot see them, so the values it put in the gap must not budge.
        panel.loc[order[3010:], "value"] += 1000.0
        frame_b, _ = clean_zone(panel[panel.zone == "DK1"], "DK1")

        labels = [t for t in gap_local if t in frame_a.index]
        self.assertTrue(labels, "the gap did not land inside the cleaned frame")
        np.testing.assert_allclose(frame_a.loc[labels, "price"].to_numpy(),
                                   frame_b.loc[labels, "price"].to_numpy())

    def test_dst_interpolation_is_the_documented_exception(self):
        """It does read forward -- one hour a year, by design, matching the paper."""
        idx = pd.date_range("2024-03-30", "2024-04-01 23:00", freq="h")
        frame = pd.DataFrame({"Price": np.arange(len(idx), dtype=float)}, index=idx)
        frame.loc[skipped_hours(idx, "Europe/Copenhagen"), "Price"] = np.nan

        moved = frame.copy()
        after = moved.index > pd.Timestamp("2024-03-31 02:00")
        moved.loc[after, "Price"] += 1000.0

        a, _ = fill_skipped_hours(frame, "Europe/Copenhagen")
        b, _ = fill_skipped_hours(moved, "Europe/Copenhagen")
        hour = pd.Timestamp("2024-03-31 02:00")
        self.assertNotAlmostEqual(a.loc[hour, "Price"], b.loc[hour, "Price"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
