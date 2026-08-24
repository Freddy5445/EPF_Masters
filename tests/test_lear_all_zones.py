"""Tests for the cross-zone sweep and its scoring.

These exercise the parts that are cheap to check and easy to get quietly wrong:
the ensemble mean, the metrics under missing observations, the orientation of
the DM/GW tests, and the bookkeeping that decides which run directory gets
scored. Fitting LEAR is not exercised here -- ``python run_lear_all_zones.py
--smoke`` does that.
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run_lear_all_zones  # noqa: E402
import run_lear_from_clean  # noqa: E402
from lear_dk1.evaluate import (  # noqa: E402
    HOURS, build_ensemble, compare, evaluate_run, load_forecasts,
    naive_weekly_mae, score,
)

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "epftoolbox"))
from epftoolbox.evaluation import MAE, rMAE  # noqa: E402


def prices(days=120, seed=7, start="2023-01-01"):
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame(50 + rng.normal(0, 15, (days, 24)), index=index, columns=HOURS)


class TestLayouts(unittest.TestCase):
    """The sweep names layouts that run_lear_from_clean has to know about."""

    def test_layouts_referenced_by_the_sweep_exist(self):
        for layout in (run_lear_all_zones.BASE_LAYOUT, run_lear_all_zones.HYDRO_LAYOUT):
            self.assertIn(layout, run_lear_from_clean.LAYOUTS)

    def test_base_layout_is_the_papers_two_exogenous_specification(self):
        columns = run_lear_from_clean.LAYOUTS[run_lear_all_zones.BASE_LAYOUT]
        self.assertEqual(len(columns) - 1, 2)  # 247 regressors

    def test_hydro_layout_adds_reservoir_after_the_base_columns(self):
        base = run_lear_from_clean.LAYOUTS[run_lear_all_zones.BASE_LAYOUT]
        hydro = run_lear_from_clean.LAYOUTS[run_lear_all_zones.HYDRO_LAYOUT]
        # read_data binds columns by position, so the shared ones must not move.
        self.assertEqual(hydro[:len(base)], base)
        self.assertEqual(hydro[-1], run_lear_from_clean.HYDRO)

    def test_dataset_name_separates_layouts(self):
        """Two layouts for one zone must not write to the same run directory."""
        base = run_lear_from_clean.dataset_name_for("NO1", "load-windsolar")
        hydro = run_lear_from_clean.dataset_name_for("NO1", "load-windsolar-hydro")
        self.assertNotEqual(base, hydro)
        self.assertTrue(base.startswith("NO1"))


class TestEnsemble(unittest.TestCase):

    def test_ensemble_is_the_arithmetic_mean(self):
        index = pd.date_range("2023-01-01", periods=5, freq="D")
        a = pd.DataFrame(np.ones((5, 24)), index=index, columns=HOURS)
        b = pd.DataFrame(3 * np.ones((5, 24)), index=index, columns=HOURS)
        ensemble = build_ensemble({56: a, 84: b})
        np.testing.assert_allclose(ensemble.to_numpy(), 2.0)

    def test_ensemble_uses_only_days_every_window_has(self):
        """Otherwise the ensemble would be a different model on different days."""
        index = pd.date_range("2023-01-01", periods=5, freq="D")
        a = pd.DataFrame(np.ones((5, 24)), index=index, columns=HOURS)
        b = pd.DataFrame(3 * np.ones((3, 24)), index=index[:3], columns=HOURS)
        ensemble = build_ensemble({56: a, 84: b})
        self.assertEqual(len(ensemble), 3)
        np.testing.assert_allclose(ensemble.to_numpy(), 2.0)

    def test_no_common_days_gives_none(self):
        a = pd.DataFrame(np.ones((2, 24)),
                         index=pd.date_range("2023-01-01", periods=2), columns=HOURS)
        b = pd.DataFrame(np.ones((2, 24)),
                         index=pd.date_range("2024-01-01", periods=2), columns=HOURS)
        self.assertIsNone(build_ensemble({56: a, 84: b}))

    def test_partial_days_are_dropped_when_loading(self):
        """A half-written final row must not enter the mean as a real forecast."""
        with tempfile.TemporaryDirectory() as tmp:
            frame = pd.DataFrame(np.ones((3, 24)),
                                 index=pd.date_range("2023-01-01", periods=3),
                                 columns=HOURS)
            frame.iloc[-1, 5:] = np.nan
            frame.to_csv(os.path.join(tmp, "forecasts_cw0056.csv"))
            loaded = load_forecasts(tmp)
            self.assertEqual(sorted(loaded), [56])
            self.assertEqual(len(loaded[56]), 2)


class TestMetrics(unittest.TestCase):
    """The metrics are reimplemented to tolerate gaps; they must still be the
    same estimators epftoolbox computes."""

    def test_mae_and_rmae_match_epftoolbox_on_complete_data(self):
        real = prices()
        pred = real + np.random.default_rng(1).normal(0, 4, real.shape)
        mine = score(real, pred)
        self.assertAlmostEqual(
            mine["mae"], float(MAE(real.to_numpy(float), pred.to_numpy(float))), places=12)
        self.assertAlmostEqual(
            mine["rmae"],
            float(rMAE(real.to_numpy(float), pred.to_numpy(float), m="W", freq="1h")),
            places=12)

    def test_naive_denominator_is_the_weekly_lag(self):
        real = prices(days=30)
        expected = np.mean(np.abs(real.to_numpy()[7:] - real.to_numpy()[:-7]))
        self.assertAlmostEqual(naive_weekly_mae(real), float(expected), places=12)

    def test_unobserved_hours_are_excluded_not_scored_as_error(self):
        """One missing observed price used to turn every metric into NaN."""
        real = prices()
        pred = real + np.random.default_rng(1).normal(0, 4, real.shape)
        holed = real.copy()
        holed.iloc[40, 2] = np.nan

        result = score(holed, pred)
        self.assertTrue(np.isfinite(result["mae"]))
        self.assertTrue(np.isfinite(result["rmae"]))
        self.assertEqual(result["hours_unobserved"], 1)
        self.assertEqual(result["hours_scored"], real.size - 1)
        # epftoolbox's own metric is NaN on the same input, which is why this exists.
        self.assertTrue(np.isnan(float(MAE(holed.to_numpy(float), pred.to_numpy(float)))))

    def test_every_forecast_is_divided_by_the_same_denominator(self):
        real = prices()
        pred = real + np.random.default_rng(1).normal(0, 4, real.shape)
        naive = naive_weekly_mae(real)
        result = score(real, pred, naive_mae=naive)
        self.assertAlmostEqual(result["rmae"], result["mae"] / naive, places=12)


class TestDMOrientation(unittest.TestCase):
    """epftoolbox's DM/GW reject H0 in favour of ``p_pred_2``, not ``p_pred_1``.

    Passing the candidate as ``p_pred_1`` inverts every p-value in the table
    while still producing plausible-looking numbers, so this is pinned down.
    """

    def setUp(self):
        rng = np.random.default_rng(0)
        self.real = prices(days=200, seed=3)
        self.good = self.real + rng.normal(0, 1, self.real.shape)
        self.bad = self.real + rng.normal(0, 5, self.real.shape)

    def test_small_p_value_means_the_first_argument_is_better(self):
        result = compare(self.real, self.good, self.bad)
        self.assertLess(result["dm"], 0.01)
        self.assertLess(result["gw"], 0.01)

    def test_large_p_value_when_the_first_argument_is_worse(self):
        result = compare(self.real, self.bad, self.good)
        self.assertGreater(result["dm"], 0.99)
        self.assertGreater(result["gw"], 0.99)

    def test_identical_forecasts_report_none_not_nan(self):
        """NaN is not valid JSON, and json.dump writes it regardless."""
        result = compare(self.real, self.good, self.good)
        self.assertIsNone(result["dm"])
        self.assertIn("dm_error", result)


class TestEvaluateRun(unittest.TestCase):

    def _write_run(self, run_dir, windows=(56, 84), days=40):
        os.makedirs(run_dir, exist_ok=True)
        real = prices(days=days, seed=11)
        rng = np.random.default_rng(2)
        for window in windows:
            (real + rng.normal(0, 3, real.shape)).to_csv(
                os.path.join(run_dir, f"forecasts_cw{window:04d}.csv"))
        return real

    def _write_dataset(self, datasets_dir, name, real):
        """The epftoolbox CSV the observed prices are read back from."""
        os.makedirs(datasets_dir, exist_ok=True)
        index = pd.date_range(real.index[0], periods=len(real) * 24, freq="h")
        frame = pd.DataFrame(
            {"Price": real.to_numpy(float).reshape(-1),
             "Exogenous 1": np.arange(len(index), dtype=float),
             "Exogenous 2": np.arange(len(index), dtype=float)},
            index=index)
        frame.index.name = "Date"
        frame.to_csv(os.path.join(datasets_dir, f"{name}.csv"),
                     date_format="%Y-%m-%dT%H:%M:%S")

    def test_evaluation_json_is_valid_json(self):
        """A NaN written into evaluation.json makes it unreadable by anything else."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            datasets = os.path.join(tmp, "datasets")
            real = self._write_run(run_dir)
            self._write_dataset(datasets, "ZZ1", real)

            results = evaluate_run(run_dir, dataset="ZZ1", datasets_dir=datasets,
                                   quiet=True)
            self.assertEqual(results["windows"], [56, 84])

            with open(os.path.join(run_dir, "evaluation.json"), encoding="utf-8") as f:
                text = f.read()

            def reject(constant):
                raise ValueError(f"{constant} is not valid JSON")

            parsed = json.loads(text, parse_constant=reject)
            self.assertIn("ensemble", parsed["scores"])

    def test_a_single_window_runs_no_self_comparison(self):
        """The ensemble of one window *is* that window; DM on it is 0/0."""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            datasets = os.path.join(tmp, "datasets")
            real = self._write_run(run_dir, windows=(56,))
            self._write_dataset(datasets, "ZZ1", real)

            results = evaluate_run(run_dir, dataset="ZZ1", datasets_dir=datasets,
                                   quiet=True)
            self.assertEqual(results["tests"], {})

    def test_missing_dataset_name_is_reported_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "run")
            self._write_run(run_dir)
            with self.assertRaises(ValueError) as caught:
                evaluate_run(run_dir, quiet=True)
            self.assertIn("dataset", str(caught.exception))


class TestRunDirSelection(unittest.TestCase):
    """Which directory gets scored must not be decided by sort order."""

    def test_picks_the_directory_written_by_this_run(self):
        import time

        with tempfile.TemporaryDirectory() as tmp:
            prefix = run_lear_from_clean.dataset_name_for("NO1", "load-windsolar")

            # Written first, but sorts last -- the trap the old code fell into.
            stale = os.path.join(tmp, f"{prefix}_2023-04-01_2023-04-21")
            fresh = os.path.join(tmp, f"{prefix}_2023-03-01_2023-04-14")
            for path in (stale, fresh):
                os.makedirs(path)
            with open(os.path.join(stale, "run_metadata.json"), "w") as handle:
                handle.write("{}")
            os.utime(os.path.join(stale, "run_metadata.json"), (1, 1))

            started = time.time()
            with open(os.path.join(fresh, "run_metadata.json"), "w") as handle:
                handle.write("{}")

            found = run_lear_all_zones._run_dir_written(
                tmp, "NO1", "load-windsolar", started)
            self.assertEqual(found, fresh)

    def test_other_layouts_are_not_picked_up(self):
        import time

        with tempfile.TemporaryDirectory() as tmp:
            other = os.path.join(
                tmp,
                run_lear_from_clean.dataset_name_for("NO1", "load-windsolar-hydro")
                + "_2023-03-01_2023-04-14")
            os.makedirs(other)
            with open(os.path.join(other, "run_metadata.json"), "w") as handle:
                handle.write("{}")

            found = run_lear_all_zones._run_dir_written(
                tmp, "NO1", "load-windsolar", time.time() - 1)
            self.assertIsNone(found)


class TestPanelZones(unittest.TestCase):

    def test_hydro_zones_are_those_with_reservoir(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = pd.DataFrame({
                "timestamp_utc": pd.date_range("2023-01-01", periods=6, freq="h",
                                               tz="UTC"),
                "zone": ["DK1", "DK1", "NO1", "NO1", "SE2", "SE2"],
                "variable": ["price", "load_forecast", "price", "reservoir",
                             "price", "reservoir"],
                "psr_type": [None] * 6,
                "value": [1.0] * 6,
            })
            path = os.path.join(tmp, "panel.parquet")
            panel.to_parquet(path, index=False)

            zones, with_hydro = run_lear_all_zones.panel_zones(path)
            self.assertEqual(zones, ["DK1", "NO1", "SE2"])
            self.assertEqual(with_hydro, ["NO1", "SE2"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
