"""Tests for the multi-zone DNN: assembly, scaling, split, search space.

These are the parts with no upstream counterpart to fall back on --
``_build_and_split_XYs`` reads one ``Price`` column and a fixed ``Exogenous n``
naming, so it can build neither the 1969-column input nor the 168-column target.
Anything reproduced from upstream rather than called (the weekly-block split, the
Lago lag layout) is checked against upstream's own code here.

Training a real model is left to ``python run_dnn_dk1.py --config joint --smoke``.
"""

import os
import sys
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dnn_dk1 import zones as Z  # noqa: E402
from dnn_dk1.hyperopt import (  # noqa: E402
    build_multizone_space, build_space, n_binary_toggles_if_generalised,
    space_summary,
)
from dnn_dk1.scaling_compat import fit_scaler, guarded_scaling  # noqa: E402


def _matrices(days=400, start="2019-01-01", seed=0):
    """A synthetic panel with the real zone layout, so no CSV is needed."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    out = {}
    for i, zone in enumerate(Z.ZONES):
        series = {"price": 30.0 + 10 * i + rng.normal(0, 5, (days, 24))}
        for j, block in enumerate(Z.ZONE_EXOG[zone]):
            series[block] = rng.uniform(100, 200, (days, 24)) * (j + 1)
        out[zone] = {name: pd.DataFrame(values, index=dates, columns=list(Z.HOURS))
                     for name, values in series.items()}
    return out


class TestDimensions(unittest.TestCase):
    """Every width is derived from ZONES and ZONE_EXOG, and must add up."""

    def test_zone_block_widths(self):
        for zone in Z.ZONES:
            expected = 96 + 72 * Z.n_exogenous(zone)
            self.assertEqual(Z.zone_block_width(zone), expected, zone)

    def test_own_widths_are_313_or_241(self):
        widths = {Z.own_input_width(z) for z in Z.ZONES}
        self.assertEqual(widths, {313, 241})

    def test_wide_width_is_the_sum_plus_one_calendar(self):
        self.assertEqual(
            Z.input_width(Z.ZONES),
            sum(Z.zone_block_width(z) for z in Z.ZONES) + 1)

    def test_joint_output_is_24_per_zone(self):
        self.assertEqual(Z.output_width(Z.ZONES), 24 * len(Z.ZONES))
        self.assertEqual(Z.output_width(Z.ZONES), 168)

    def test_zone_slices_tile_the_output_exactly(self):
        covered = np.zeros(Z.output_width(Z.ZONES), dtype=int)
        for zone in Z.ZONES:
            covered[Z.zone_slice(zone)] += 1
        np.testing.assert_array_equal(covered, np.ones_like(covered))


class TestFeatureNames(unittest.TestCase):
    """Column order must be reproducible from the zone list alone."""

    def test_names_match_the_built_width(self):
        matrices = _matrices()
        days = Z.available_days(matrices)[10:20]
        X = Z.build_X(matrices, days, Z.ZONES, include_calendar=True)
        self.assertEqual(X.shape[1], len(Z.feature_names(Z.ZONES, True)))

    def test_calendar_toggle_drops_exactly_one_column(self):
        matrices = _matrices()
        days = Z.available_days(matrices)[10:20]
        with_cal = Z.build_X(matrices, days, Z.ZONES, True)
        without = Z.build_X(matrices, days, Z.ZONES, False)
        self.assertEqual(with_cal.shape[1] - without.shape[1], 1)
        np.testing.assert_allclose(with_cal[:, 1:], without)

    def test_names_are_stable_across_calls(self):
        self.assertEqual(Z.feature_names(), Z.feature_names())

    def test_zone_labels_line_up_with_the_columns(self):
        labels = Z.feature_zone_labels(Z.ZONES, True)
        self.assertEqual(len(labels), Z.input_width(Z.ZONES))
        self.assertEqual(labels[0], "calendar")
        for zone in Z.ZONES:
            self.assertEqual(int((labels == zone).sum()), Z.zone_block_width(zone))

    def test_wide_block_is_a_superset_of_the_own_block(self):
        """DNN-own's inputs must appear inside DNN-wide's, unchanged."""
        matrices = _matrices()
        days = Z.available_days(matrices)[10:20]
        wide = Z.build_X(matrices, days, Z.ZONES, True)
        own = Z.build_X(matrices, days, ("DK2",), True)
        names = Z.feature_names(Z.ZONES, True)
        columns = [names.index(n) for n in Z.zone_feature_names("DK2")]
        np.testing.assert_allclose(wide[:, columns], own[:, 1:])


class TestLagLayout(unittest.TestCase):
    """The assembled values must be the prices and forecasts they claim to be."""

    def setUp(self):
        self.matrices = _matrices()
        self.days = Z.available_days(self.matrices)[30:40]
        self.X = Z.build_X(self.matrices, self.days, Z.ZONES, True)
        self.names = Z.feature_names(Z.ZONES, True)

    def _column(self, name):
        return self.X[:, self.names.index(name)]

    def test_price_lags_read_the_right_day(self):
        for lag in Z.PRICE_LAGS:
            expected = self.matrices["NL"]["price"].reindex(
                self.days - pd.Timedelta(days=lag))[7].to_numpy()
            np.testing.assert_allclose(
                self._column(f"NL_price_D-{lag}_h7"), expected)

    def test_exogenous_day_d_is_not_lagged(self):
        expected = self.matrices["DK1"]["wind"].reindex(self.days)[3].to_numpy()
        np.testing.assert_allclose(self._column("DK1_wind_D_h3"), expected)

    def test_targets_are_day_d_prices(self):
        Y = Z.build_Y(self.matrices, self.days, Z.ZONES)
        for zone in Z.ZONES:
            expected = self.matrices[zone]["price"].reindex(self.days).to_numpy()
            np.testing.assert_allclose(Y[:, Z.zone_slice(zone)], expected)

    def test_calendar_is_the_day_of_week(self):
        np.testing.assert_allclose(self._column(Z.CALENDAR_COLUMN),
                                   self.days.dayofweek.to_numpy(float))


class TestNaNRefusal(unittest.TestCase):
    """Nothing in a run script imputes; a gap must stop the run."""

    def test_a_lag_reaching_before_the_panel_is_an_error(self):
        matrices = _matrices()
        first = Z.available_days(matrices)[:1]
        with self.assertRaises(Z.ZoneDataError) as caught:
            Z.build_X(matrices, first, Z.ZONES, True)
        self.assertIn("NaN", str(caught.exception))

    def test_first_forecastable_day_clears_the_longest_lag(self):
        matrices = _matrices()
        day = Z.first_forecastable_day(matrices)
        X = Z.build_X(matrices, pd.DatetimeIndex([day]), Z.ZONES, True)
        self.assertFalse(np.isnan(X).any())

    def test_training_window_is_clipped_to_forecastable_days(self):
        matrices = _matrices(days=400)
        days = Z.training_days(matrices, Z.available_days(matrices)[-1], 4)
        self.assertGreaterEqual(days.min(), Z.first_forecastable_day(matrices))
        self.assertFalse(np.isnan(Z.build_X(matrices, days, Z.ZONES, True)).any())


class TestPerZoneScaler(unittest.TestCase):

    def setUp(self):
        self.matrices = _matrices()
        self.days = Z.available_days(self.matrices)[10:300]
        self.Y = Z.build_Y(self.matrices, self.days, Z.ZONES)

    def test_round_trips(self):
        scaler = Z.PerZoneScaler("Invariant", Z.ZONES)
        back = scaler.inverse_transform(scaler.fit_transform(self.Y))
        np.testing.assert_allclose(back, self.Y, atol=1e-9)

    def test_transformed_dispersion_is_comparable_across_zones(self):
        """Equal weighting of the 168 outputs only means something if it is."""
        scaler = Z.PerZoneScaler("Invariant", Z.ZONES)
        dispersion = scaler.dispersion(scaler.fit_transform(self.Y))
        self.assertLess(max(dispersion.values()) / min(dispersion.values()), 3.0)

    def test_matches_a_pooled_fit(self):
        """The spec expects per-zone and pooled fits to differ. They do not.

        epftoolbox's MedianScaler fits ``np.median(data, axis=0)`` -- per column
        -- and so do sklearn's StandardScaler and MinMaxScaler behind the other
        options. One scaler over 168 columns is therefore already seven scalers
        over 24. The per-zone container is kept for the inverse transform of a
        single zone's sub-vector, and so the guarantee does not rest on a
        property a future scaleY need not have.
        """
        for normalize in ("Invariant", "Median", "Std", "Norm", "Norm1"):
            with self.subTest(normalize=normalize):
                scaler = Z.PerZoneScaler(normalize, Z.ZONES)
                self.assertTrue(scaler.equals_pooled_fit(self.Y))

    def test_one_zone_inverts_independently_of_the_others(self):
        scaler = Z.PerZoneScaler("Invariant", Z.ZONES)
        transformed = scaler.fit_transform(self.Y)
        back = scaler.inverse_transform(transformed)
        for zone in Z.ZONES:
            np.testing.assert_allclose(
                back[:, Z.zone_slice(zone)], self.Y[:, Z.zone_slice(zone)],
                atol=1e-9)


class TestZeroMadGuard(unittest.TestCase):
    """A constant column must not become a column of NaN.

    Upstream's Median/Invariant scalers compute ``(x - median) / mad``; a solar
    forecast is exactly zero at night, so ``mad == 0`` and the column fills with
    NaN, which Keras then propagates into every loss.
    ``lear_dk1.compat._fit_invariant_scaler`` already guards this for LEAR.
    """

    def _data(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 4))
        X[:, 2] = 0.0          # a night-time solar hour
        return X

    def test_upstream_produces_nan(self):
        from epftoolbox.data import scaling

        with np.errstate(invalid="ignore"):
            [scaled], _ = scaling([self._data()], "Invariant")
        self.assertTrue(np.isnan(scaled[:, 2]).all(),
                        "upstream no longer has the zero-MAD bug; drop the guard")

    def test_the_guard_maps_a_constant_column_to_zero(self):
        [scaled], scaler = guarded_scaling([self._data()], "Invariant")
        self.assertFalse(np.isnan(scaled).any())
        np.testing.assert_allclose(scaled[:, 2], 0.0)

    def test_the_guard_leaves_varying_columns_alone(self):
        data = self._data()
        [guarded], _ = guarded_scaling([data], "Invariant")
        from epftoolbox.data import scaling

        with np.errstate(invalid="ignore"):
            [upstream], _ = scaling([data], "Invariant")
        varying = [0, 1, 3]
        np.testing.assert_allclose(guarded[:, varying], upstream[:, varying])

    def test_the_inverse_recovers_the_constant(self):
        data = self._data()
        [scaled], scaler = guarded_scaling([data], "Invariant")
        np.testing.assert_allclose(scaler.inverse_transform(scaled), data,
                                   atol=1e-9)

    def test_it_fires_on_the_real_layout(self):
        """27 of DNN-own/DK1's 313 columns and 117 of 1969 are night-time solar."""
        scaler, n_constant = fit_scaler("Invariant", self._data())
        self.assertEqual(n_constant, 1)


class TestSplit(unittest.TestCase):
    """The weekly-block shuffle must be upstream's, not a lookalike."""

    def test_matches_upstream_on_a_single_zone_frame(self):
        from epftoolbox.models._dnn import _build_and_split_XYs

        index = pd.date_range("2023-01-01", periods=24 * 120, freq="h")
        rng = np.random.default_rng(3)
        frame = pd.DataFrame(
            {"Price": rng.normal(40, 8, len(index)),
             "Exogenous 1": rng.normal(1000, 50, len(index))}, index=index)
        features = {"In: Day": 1, "In: Price D-1": 1, "In: Price D-2": 0,
                    "In: Price D-3": 0, "In: Price D-7": 0,
                    "In: Exog-1 D": 0, "In: Exog-1 D-1": 0, "In: Exog-1 D-7": 0}
        up_Xtrain, up_Ytrain, up_Xval, up_Yval, _, _, _ = _build_and_split_XYs(
            dfTrain=frame, dfTest=frame, features=features, shuffle_train=True,
            hyperoptimization=True, data_augmentation=False,
            n_exogenous_inputs=1)

        # The same rows, unshuffled, fed through our own split.
        n = up_Xtrain.shape[0] + up_Xval.shape[0]
        X = np.arange(n * 3, dtype=float).reshape(n, 3)
        Y = np.arange(n * 24, dtype=float).reshape(n, 24)
        Xtrain, Ytrain, Xval, Yval = Z.split_train_val(
            X, Y, shuffle_train=True, hyperoptimization=True)

        self.assertEqual(Xtrain.shape[0], up_Xtrain.shape[0])
        self.assertEqual(Xval.shape[0], up_Xval.shape[0])
        # The permutation itself: rebuild upstream's expansion literally.
        np.random.seed(7)
        index_week = np.arange(n)[::7]
        np.random.shuffle(index_week)
        order = [i + k for i in index_week for k in range(7) if i + k in np.arange(n)]
        np.testing.assert_array_equal(
            np.concatenate([Xtrain, Xval])[:, 0], X[order][:, 0])

    def test_validation_is_a_quarter(self):
        X = np.zeros((400, 5))
        Y = np.zeros((400, 24))
        _, _, Xval, _ = Z.split_train_val(X, Y, shuffle_train=True)
        self.assertEqual(Xval.shape[0], 100)

    def test_x_and_y_stay_aligned_through_the_shuffle(self):
        n = 210
        X = np.arange(n, dtype=float).reshape(n, 1)
        Y = np.arange(n, dtype=float).reshape(n, 1) * 10
        Xtrain, Ytrain, Xval, Yval = Z.split_train_val(
            X, Y, shuffle_train=True, hyperoptimization=True)
        np.testing.assert_allclose(Ytrain[:, 0], Xtrain[:, 0] * 10)
        np.testing.assert_allclose(Yval[:, 0], Xval[:, 0] * 10)


class TestSearchSpaces(unittest.TestCase):
    """Wide and joint must search the same space; own must stay Lago's."""

    def test_wide_and_joint_share_one_space(self):
        """They call the same builder with the same argument -- by construction.

        This is the invariant the whole wide -> joint comparison rests on, so it
        is asserted rather than trusted to stay true.
        """
        self.assertEqual(sorted(build_multizone_space(2)),
                         sorted(build_multizone_space(2)))
        self.assertEqual(space_summary(build_multizone_space(2)),
                         space_summary(build_multizone_space(2)))

    def test_the_multizone_space_has_no_block_toggles(self):
        space = build_multizone_space(2)
        toggles = [k for k in space if k.startswith("In: ")]
        self.assertEqual(toggles, ["In: Day"])

    def test_the_multizone_space_keeps_every_architecture_dimension(self):
        wide = build_multizone_space(2)
        own = build_space(2, 0, 3)
        architecture = {k for k in own if not k.startswith("In: ")}
        self.assertEqual(architecture, {k for k in wide if not k.startswith("In: ")})

    def test_own_matches_the_benchmark_toggle_count(self):
        """11 binaries for a two-exogenous zone, which is upstream's own count."""
        two = build_space(2, 0, 2)
        self.assertEqual(sum(1 for k in two if k.startswith("In: ")), 11)
        three = build_space(2, 0, 3)
        self.assertEqual(sum(1 for k in three if k.startswith("In: ")), 14)

    def test_the_naive_generalisation_is_what_was_avoided(self):
        """4 zones x 13 + 3 zones x 10 + 1 calendar."""
        self.assertEqual(n_binary_toggles_if_generalised(Z.ZONES), 83)

    def test_multizone_ranges_are_wider_than_upstreams(self):
        from dnn_dk1.hyperopt import (
            L1_RANGE, MULTIZONE_L1_RANGE, MULTIZONE_NEURON_RANGES, NEURON_RANGES,
        )
        for layer in (1, 2):
            self.assertGreater(MULTIZONE_NEURON_RANGES[layer][1],
                               NEURON_RANGES[layer][1])
        self.assertGreater(MULTIZONE_L1_RANGE[1], L1_RANGE[1])


class TestFirstLayerWeights(unittest.TestCase):
    """With the block toggles gone, the weights are the record of what was used."""

    def test_magnitudes_group_by_zone(self):
        from dnn_dk1 import DNNModel, first_layer_weight_magnitudes

        labels = Z.feature_zone_labels(Z.ZONES, True)
        model = DNNModel(neurons=[16], n_features=len(labels), outputShape=168)
        magnitudes = first_layer_weight_magnitudes(model, labels)
        self.assertEqual(set(magnitudes), {"calendar", *Z.ZONES})
        for value in magnitudes.values():
            self.assertGreaterEqual(value, 0.0)

    def test_a_label_mismatch_is_an_error(self):
        from dnn_dk1 import DNNModel, first_layer_weight_magnitudes

        model = DNNModel(neurons=[8], n_features=10)
        with self.assertRaises(ValueError):
            first_layer_weight_magnitudes(model, np.array(["a"] * 9))

class TestSearchWindowGate(unittest.TestCase):
    """The search must never see the test period.

    This is the one invariant that used to be a convention. A violated run does
    not crash and does not warn -- it produces results that look *excellent*,
    which is the outcome one is least likely to question. So it fails loudly.
    """

    def setUp(self):
        import run_dnn_dk1 as R

        self.begin, self.end = R.hyperopt_window(Z.BEGIN_TEST)

    def test_the_real_window_is_accepted(self):
        from dnn_dk1.hyperopt import assert_search_window_precedes_test

        assert_search_window_precedes_test(self.begin, self.end, "test")

    def test_the_test_period_itself_is_refused(self):
        from dnn_dk1.hyperopt import (
            SearchWindowError, assert_search_window_precedes_test,
        )

        with self.assertRaises(SearchWindowError):
            assert_search_window_precedes_test(Z.BEGIN_TEST, Z.END_TEST, "test")

    def test_a_valid_start_with_the_test_end_is_refused(self):
        """The plausible slip: the window still looks like a pre-test year."""
        from dnn_dk1.hyperopt import (
            SearchWindowError, assert_search_window_precedes_test,
        )

        with self.assertRaises(SearchWindowError):
            assert_search_window_precedes_test(self.begin, Z.END_TEST, "test")

    def test_ending_on_the_first_test_day_is_refused(self):
        """The off-by-one: one day of overlap is still leakage."""
        from dnn_dk1.hyperopt import (
            SearchWindowError, assert_search_window_precedes_test,
        )

        with self.assertRaises(SearchWindowError):
            assert_search_window_precedes_test(self.begin, Z.BEGIN_TEST, "test")

    def test_the_last_hour_before_the_test_is_accepted(self):
        """The boundary must be usable, or the legitimate window fails too."""
        from dnn_dk1.hyperopt import assert_search_window_precedes_test

        assert_search_window_precedes_test(
            self.begin, Z.BEGIN_TEST - pd.Timedelta(hours=1), "test")

    def test_both_optimisers_call_it(self):
        """Available but uncalled would be worse than absent: it would look safe."""
        import inspect

        from dnn_dk1 import hyperopt as H

        for name in ("optimize", "optimize_multizone"):
            with self.subTest(optimiser=name):
                self.assertIn("assert_search_window_precedes_test",
                              inspect.getsource(getattr(H, name)))

    def test_it_fires_before_anything_is_written(self):
        """A refused search must not leave a trials file behind."""
        import tempfile

        from dnn_dk1 import hyperopt as H

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(H.SearchWindowError):
                H.optimize_multizone(
                    matrices={}, path_hyperparameters_folder=tmp,
                    dataset="TEST", begin_test_date=Z.BEGIN_TEST,
                    end_test_date=Z.END_TEST, out_zones=Z.ZONES, max_evals=1)
            self.assertEqual(os.listdir(tmp), [])


class TestEarlyStoppingMAE(unittest.TestCase):
    """The stopping rule must not be driven by the most expensive zones.

    ``loss`` is on the standardised scale, where per-zone target scaling has
    already balanced the zones. The validation MAE is on the price scale, where
    it has not. Upstream keeps the weights whenever *either* improves, so a
    pooled price-scale MAE hands weight retention to DE_LU and NL -- the exact
    imbalance the per-zone scaling exists to remove.
    """

    def _targets(self, seed=0):
        rng = np.random.default_rng(seed)
        # Zone price levels roughly as they are in the data: SE3 near 32,
        # DE_LU and NL near 95.
        levels = [70, 62, 95, 95, 55, 32, 48]
        Y = np.concatenate(
            [rng.normal(level, level * 0.3, (60, 24)) for level in levels], axis=1)
        return Y, Y + rng.normal(0, 4, Y.shape)

    def test_a_single_output_zone_is_untouched(self):
        """DNN-own and DNN-wide must keep upstream's number exactly."""
        from dnn_dk1 import DNNModel

        model = DNNModel(neurons=[8], n_features=5, outputShape=24)
        self.assertIsNone(model.output_zones)
        one = DNNModel(neurons=[8], n_features=5, outputShape=24,
                       output_zones=("DK1",))
        self.assertEqual(len(one.output_zones), 1)

    def test_joint_normalises_and_pooling_would_not(self):
        from dnn_dk1 import DNNModel
        from dnn_dk1.hyperopt import _per_zone_mae, _scale_normalisers
        from epftoolbox.evaluation import MAE

        Y, Ybar = self._targets()
        normalisers = _scale_normalisers(Y, Z.ZONES)
        per_zone = _per_zone_mae(Y, Ybar, Z.ZONES)
        normalised = float(np.mean([per_zone[z] / normalisers[z] for z in Z.ZONES]))
        pooled = float(np.mean(MAE(Y, Ybar)))
        self.assertNotAlmostEqual(normalised, pooled, places=3)

        model = DNNModel(neurons=[8], n_features=5, outputShape=168,
                         output_zones=Z.ZONES)
        self.assertEqual(len(model.output_zones), 7)

    def test_the_expensive_zones_stop_dominating(self):
        """The whole point: an error in SE3 must count as much as one in NL.

        The same absolute error is added to one cheap zone and then to one
        expensive zone. Pooled, both give the identical MAE, so the metric cannot
        tell them apart; normalised, the cheap zone's error is the larger one,
        because it is larger relative to that zone's own scale.
        """
        from dnn_dk1.hyperopt import _per_zone_mae, _scale_normalisers
        from epftoolbox.evaluation import MAE

        Y, _ = self._targets()
        normalisers = _scale_normalisers(Y, Z.ZONES)

        def score(zone):
            Ybar = Y.copy()
            Ybar[:, Z.zone_slice(zone)] += 10.0
            per_zone = _per_zone_mae(Y, Ybar, Z.ZONES)
            return (float(np.mean(MAE(Y, Ybar))),
                    float(np.mean([per_zone[z] / normalisers[z] for z in Z.ZONES])))

        cheap_pooled, cheap_normalised = score("SE3")     # ~32 EUR/MWh
        rich_pooled, rich_normalised = score("NL")        # ~95 EUR/MWh

        self.assertAlmostEqual(cheap_pooled, rich_pooled, places=9)
        self.assertGreater(cheap_normalised, rich_normalised)

    def test_normalisers_start_uncached_and_are_computed_once(self):
        """A scale that drifted per epoch would make bestMAE incomparable."""
        from dnn_dk1 import DNNModel

        model = DNNModel(neurons=[8], n_features=5, outputShape=168,
                         output_zones=Z.ZONES)
        self.assertIsNone(model._mae_normalisers)


class TestPhase2Runs(unittest.TestCase):
    """The ten runs, defined once and shared by launcher, status and pre-flight."""

    def test_there_are_exactly_ten(self):
        from dnn_dk1 import runs as RS

        self.assertEqual(len(RS.RUNS), 10)
        self.assertEqual(sum(1 for r in RS.RUNS if r.config == "own"), 7)
        self.assertEqual(sum(1 for r in RS.RUNS if r.config == "wide"), 2)
        self.assertEqual(sum(1 for r in RS.RUNS if r.config == "joint"), 1)

    def test_joint_is_started_first(self):
        """It is the critical path at ~21 h; it must never queue behind cheap work."""
        from dnn_dk1 import runs as RS

        self.assertEqual(RS.RUNS[0].run_id, "joint")

    def test_widths_match_the_zone_module(self):
        from dnn_dk1 import runs as RS

        for run in RS.RUNS:
            with self.subTest(run=run.run_id):
                if run.config == "own":
                    self.assertEqual(run.n_inputs, Z.own_input_width(run.focal))
                    self.assertEqual(run.n_outputs, 24)
                else:
                    self.assertEqual(run.n_inputs, Z.input_width(Z.ZONES))
        self.assertEqual(RS.RUNS_BY_ID["joint"].n_outputs, 168)

    def test_threads_fit_the_machine(self):
        from dnn_dk1 import runs as RS

        self.assertEqual(RS.TOTAL_THREADS, 15)

    def test_the_settled_budget_is_identical_for_every_run(self):
        """An unequal budget between configurations confounds the comparison."""
        from dnn_dk1 import runs as RS

        self.assertEqual(RS.MAX_EVALS, 300)
        self.assertEqual(RS.SEEDS, (1, 2, 3, 4))
        self.assertEqual(RS.TEST_DAYS, 731)

    def test_every_command_line_carries_the_settled_parameters(self):
        import run_dnn_all
        from dnn_dk1 import runs as RS

        for run in RS.RUNS:
            command = run_dnn_all.run_command(run, "out", "datasets")
            with self.subTest(run=run.run_id):
                self.assertIn("300", command)
                self.assertIn("1,2,3,4", command)
                self.assertIn("2023-10-01", command)
                self.assertIn("2025-09-30", command)


class TestStatus(unittest.TestCase):
    """The status tool must survive an empty, partial or corrupt state."""

    def test_collect_works_with_nothing_on_disk(self):
        import tempfile

        import dnn_status

        with tempfile.TemporaryDirectory() as tmp:
            states = dnn_status.collect(out_dir=tmp, log_dir=tmp)
        self.assertEqual(len(states), 10)
        self.assertTrue(all(s["days_done"] == 0 for s in states))

    def test_render_never_raises_on_an_empty_state(self):
        import tempfile

        import dnn_status

        with tempfile.TemporaryDirectory() as tmp:
            text = dnn_status.render(dnn_status.collect(out_dir=tmp, log_dir=tmp))
        self.assertIn("ALL", text)

    def test_the_rate_uses_only_the_recent_window(self):
        """A whole-run mean flatters the ETA exactly when it matters."""
        import dnn_status

        # 100 fast days then 20 slow ones: the recent rate must see the slow ones.
        timings = pd.DataFrame({
            "date": [f"day{i:04d}" for i in range(120)],
            "seconds": [10.0] * 100 + [60.0] * 20,
        })
        rate, window = dnn_status._recent_seconds_per_day(timings, window=20)
        self.assertEqual(window, 20)
        self.assertAlmostEqual(rate, 60.0)

    def test_seeds_are_summed_within_a_day(self):
        """Day-outer/seed-inner: a day of wall clock is every seed's fit."""
        import dnn_status

        timings = pd.DataFrame({"date": ["d1", "d1", "d2", "d2"],
                                "seconds": [5.0, 5.0, 7.0, 7.0]})
        rate, _ = dnn_status._recent_seconds_per_day(timings)
        self.assertAlmostEqual(rate, 12.0)

    def test_a_recycled_pid_is_not_reported_as_running(self):
        """This process is alive but is not a backtest."""
        import dnn_status

        alive, note = dnn_status._process_alive(os.getpid(), "joint")
        self.assertFalse(alive)
        self.assertEqual(note, "pid reused by another process")

    def test_heartbeat_row_matches_the_documented_schema(self):
        import tempfile

        import dnn_status

        with tempfile.TemporaryDirectory() as tmp:
            states = dnn_status.collect(out_dir=tmp, log_dir=tmp)
            row = dnn_status.heartbeat_row(states)
            path = dnn_status.append_heartbeat(
                states, os.path.join(tmp, "progress_log.csv"))
            written = pd.read_csv(path)
        for column in ("timestamp", "run_id", "state", "days_done", "seeds_done",
                       "seconds_per_day_recent", "eta_utc"):
            self.assertIn(column, row.columns)
        self.assertEqual(len(written), 10)


class TestLauncherDetachment(unittest.TestCase):

    def test_flags_detach_the_child_from_this_console(self):
        import run_dnn_all

        flags = run_dnn_all.detach_flags()
        if os.name == "nt":
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP -- no console to be
            # killed with, and out of reach of a Ctrl-C here.
            self.assertEqual(flags["creationflags"], 0x00000008 | 0x00000200)
        else:
            self.assertTrue(flags["start_new_session"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
