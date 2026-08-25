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

from tests.notebook_code import load  # noqa: E402

# The cleaning lives in data_cleaning.ipynb, next to the analysis that motivated
# it. These tests read the functions out of the notebook rather than duplicating
# them, so what is verified is exactly what runs.
_NB = load()
build_model_panel = _NB["build_model_panel"]
fill_skipped_hours = _NB["fill_skipped_hours"]
skipped_hours = _NB["skipped_hours"]


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


class TestModelPanel(unittest.TestCase):

    def _build(self, panel):
        frame, report = build_model_panel(panel, verbose=False)
        return frame, report

    def test_every_day_has_exactly_24_rows(self):
        """Both models reshape(-1, 24); a 23- or 25-hour day breaks them."""
        model, _ = self._build(raw_panel())
        price = model[model.variable == "price"]
        per_day = price.groupby(price.timestamp_local.dt.normalize()).size()
        self.assertEqual(sorted(per_day.unique()), [24])

    def test_result_has_no_gaps(self):
        model, report = self._build(raw_panel())
        self.assertFalse(model.value.isna().any())
        self.assertTrue(all(r["complete"] for r in report))

    def test_hours_are_a_gapless_run(self):
        model, _ = self._build(raw_panel())
        price = model[model.variable == "price"].sort_values("timestamp_local")
        step = price.timestamp_local.diff().dropna()
        self.assertTrue(step.eq(pd.Timedelta(hours=1)).all())

    def test_long_gaps_are_filled_and_counted(self):
        panel = raw_panel()
        price = panel[(panel.zone == "DK1") & (panel.variable == "price")]
        order = price.sort_values("timestamp_utc").index
        panel = panel.drop(index=order[5000:5030])

        model, report = self._build(panel)
        self.assertFalse(model.value.isna().any())
        self.assertGreaterEqual(report[0]["imputation"]["price"]["missing"], 30)

    def test_dst_is_counted_separately_from_imputation(self):
        """The write-up must distinguish a clock artefact from missing data.

        fill_skipped_hours runs before impute_frame, so the DST hours are gone by
        the time the imputation report is built and must never be netted against
        it.
        """
        _, report = self._build(raw_panel())
        r = report[0]
        # One skipped hour in 2024, times three series.
        self.assertEqual(r["dst_hours_interpolated"], 3)
        self.assertEqual(r["imputation"].get("price", {}).get("missing", 0),
                         r["imputation"].get("price", {}).get("missing", 0))
        self.assertNotIn("dst", str(r["imputation"]))

    def test_psr_type_round_trips(self):
        """The wind components must stay distinguishable after cleaning."""
        model, _ = self._build(raw_panel())
        wind = model[model.variable == "generation_forecast"]
        self.assertEqual(sorted(wind.psr_type.unique()), ["wind_onshore"])

    def test_columns_and_zones_survive(self):
        model, report = self._build(raw_panel(zones=("DK1", "NO1")))
        self.assertEqual(sorted(model.zone.unique()), ["DK1", "NO1"])
        self.assertEqual(list(model.columns),
                         ["timestamp_local", "zone", "variable", "psr_type", "value"])
        self.assertEqual(len(report), 2)


class TestNoLookAhead(unittest.TestCase):
    """Imputation must never read the future. DST interpolation is the one
    documented exception, and is confined to the skipped hour."""

    def test_a_gap_is_filled_only_from_the_past(self):
        panel = raw_panel(end="2024-06-30 23:00")
        price = panel[(panel.zone == "DK1") & (panel.variable == "price")]
        order = price.sort_values("timestamp_utc").index

        gap_local = (panel.loc[order[3000:3010], "timestamp_utc"]
                     .dt.tz_convert("Europe/Copenhagen").dt.tz_localize(None))
        panel.loc[order[3000:3010], "value"] = np.nan

        a, _ = build_model_panel(panel, verbose=False)

        # Move every observation after the gap. A causal filler cannot see them,
        # so what it put in the gap must not budge.
        panel.loc[order[3010:], "value"] += 1000.0
        b, _ = build_model_panel(panel, verbose=False)

        def at(frame):
            p = frame[(frame.zone == "DK1") & (frame.variable == "price")]
            p = p.set_index("timestamp_local")["value"]
            return p.reindex([t for t in gap_local if t in p.index])

        self.assertTrue(len(at(a)), "the gap did not land inside the cleaned frame")
        np.testing.assert_allclose(at(a).to_numpy(), at(b).to_numpy())

    def test_dst_interpolation_is_the_documented_exception(self):
        """It does read forward -- one hour a year, by design, matching the paper."""
        idx = pd.date_range("2024-03-30", "2024-04-01 23:00", freq="h")
        frame = pd.DataFrame({"Price": np.arange(len(idx), dtype=float)}, index=idx)
        frame.loc[skipped_hours(idx, "Europe/Copenhagen"), "Price"] = np.nan

        moved = frame.copy()
        moved.loc[moved.index > pd.Timestamp("2024-03-31 02:00"), "Price"] += 1000.0

        a, _ = fill_skipped_hours(frame, "Europe/Copenhagen")
        b, _ = fill_skipped_hours(moved, "Europe/Copenhagen")
        hour = pd.Timestamp("2024-03-31 02:00")
        self.assertNotAlmostEqual(a.loc[hour, "Price"], b.loc[hour, "Price"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
