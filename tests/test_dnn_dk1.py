"""Tests for the Keras 3 rebuild of the epftoolbox DNN.

The point of these is fidelity: the network this project builds must be the
network epftoolbox describes, differing only where Keras 3 forced a change.
Training a real model is left to ``python run_dnn_dk1.py --smoke``.
"""

import os
import pickle
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dnn_dk1.forecaster import DNN, MIN_NEURONS, hyperparameter_path  # noqa: E402
from dnn_dk1.hyperopt import build_space  # noqa: E402
from dnn_dk1.model import BATCH_SIZE, GRADIENT_CLIP, MAX_EPOCHS, DNNModel  # noqa: E402


class TestArchitecture(unittest.TestCase):
    """The layer stack must match _dnn.py's _build_model."""

    def test_output_is_24_hours(self):
        model = DNNModel(neurons=[64], n_features=30)
        self.assertEqual(model.model.output_shape[-1], 24)

    def test_input_width_is_the_feature_count(self):
        model = DNNModel(neurons=[64], n_features=37)
        self.assertEqual(model.model.input_shape[-1], 37)

    def test_hidden_layers_follow_the_neuron_list(self):
        model = DNNModel(neurons=[128, 64], n_features=30)
        dense = [l for l in model.model.layers if l.__class__.__name__ == "Dense"]
        # Two hidden layers plus the 24-wide output.
        self.assertEqual([l.units for l in dense], [128, 64, 24])

    def test_every_searchable_activation_builds(self):
        """hyperopt can pick any of these, so all seven must construct."""
        for activation in ("relu", "softplus", "tanh", "selu", "LeakyReLU",
                           "PReLU", "sigmoid"):
            with self.subTest(activation=activation):
                model = DNNModel(neurons=[32], n_features=20, activation=activation,
                                 dropout=0.1)
                self.assertEqual(model.model.output_shape[-1], 24)

    def test_selu_forces_lecun_normal_and_alpha_dropout(self):
        """Self-normalising nets need this pairing; upstream hardcodes it."""
        model = DNNModel(neurons=[32], n_features=20, activation="selu", dropout=0.2)
        self.assertEqual(model.initializer, "lecun_normal")
        names = [l.__class__.__name__ for l in model.model.layers]
        self.assertIn("AlphaDropout", names)
        self.assertNotIn("Dropout", names)

    def test_non_selu_uses_plain_dropout(self):
        model = DNNModel(neurons=[32], n_features=20, activation="relu", dropout=0.2)
        names = [l.__class__.__name__ for l in model.model.layers]
        self.assertIn("Dropout", names)
        self.assertNotIn("AlphaDropout", names)

    def test_batch_normalization_is_optional(self):
        on = DNNModel(neurons=[32], n_features=20, batch_normalization=True)
        off = DNNModel(neurons=[32], n_features=20, batch_normalization=False)
        self.assertIn("BatchNormalization",
                      [l.__class__.__name__ for l in on.model.layers])
        self.assertNotIn("BatchNormalization",
                         [l.__class__.__name__ for l in off.model.layers])

    def test_regularizer_choices(self):
        self.assertIsNone(DNNModel(neurons=[8], n_features=5)._reg(0.1))
        for kind in ("l1", "l2"):
            model = DNNModel(neurons=[8], n_features=5, regularization=kind)
            self.assertIsNotNone(model._reg(0.1))

    def test_dropout_outside_zero_one_is_rejected(self):
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                DNNModel(neurons=[8], n_features=5, dropout=bad)


class TestKeras3Fixes(unittest.TestCase):
    """The two calls Keras 3 rejects, and what replaced them."""

    def test_learning_rate_reaches_the_optimizer(self):
        """Upstream passes `lr=`, which Keras 3 raises on."""
        model = DNNModel(neurons=[16], n_features=10, lr=0.0025, optimizer="adam")
        self.assertAlmostEqual(
            float(model.model.optimizer.learning_rate.numpy()), 0.0025, places=7)

    def test_every_searchable_optimizer_builds(self):
        for name in ("adam", "RMSprop", "adagrad", "adadelta"):
            with self.subTest(optimizer=name):
                model = DNNModel(neurons=[16], n_features=10, lr=0.001, optimizer=name)
                self.assertIsNotNone(model.model.optimizer)

    def test_gradient_clipping_is_kept(self):
        model = DNNModel(neurons=[16], n_features=10, lr=0.001)
        self.assertEqual(model.model.optimizer.clipvalue, GRADIENT_CLIP)

    def test_no_lr_means_the_default_adam(self):
        model = DNNModel(neurons=[16], n_features=10, lr=None)
        self.assertIsNotNone(model.model.optimizer)

    def test_unknown_optimizer_is_named_in_the_error(self):
        """Upstream's if/if/if leaves `opt` unbound and raises UnboundLocalError."""
        with self.assertRaises(ValueError) as caught:
            DNNModel(neurons=[16], n_features=10, lr=0.001, optimizer="nadam")
        self.assertIn("nadam", str(caught.exception))

    def test_training_constants_match_upstream(self):
        self.assertEqual(BATCH_SIZE, 192)
        self.assertEqual(MAX_EPOCHS, 1000)
        self.assertEqual(GRADIENT_CLIP, 10000)


class TestTraining(unittest.TestCase):
    """The training loop's mechanics.

    MAX_EPOCHS is patched down to keep these quick. The real cap is upstream's
    1000, asserted in TestKeras3Fixes; what is under test here is that the loop
    trains, early-stops and restores weights -- none of which needs 1000 epochs,
    and each epoch costs a fit, an evaluate and a predict.
    """

    def _data(self, n=64, features=12, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(0, 1, (n, features))
        Y = rng.normal(50, 10, (n, 24))
        return X, Y

    def test_fit_then_predict_gives_a_24_hour_vector(self):
        X, Y = self._data()
        model = DNNModel(neurons=[16], n_features=X.shape[1], lr=0.01,
                         epochs_early_stopping=2)
        with mock.patch("dnn_dk1.model.MAX_EPOCHS", 5):
            model.fit(X, Y, X, Y)
        self.assertEqual(model.predict(X[:1]).shape, (1, 24))

    def test_training_changes_the_weights(self):
        X, Y = self._data()
        model = DNNModel(neurons=[16], n_features=X.shape[1], lr=0.01,
                         epochs_early_stopping=2)
        before = [w.copy() for w in model.model.get_weights()]
        with mock.patch("dnn_dk1.model.MAX_EPOCHS", 5):
            model.fit(X, Y, X, Y)
        after = model.model.get_weights()
        self.assertFalse(all(np.allclose(a, b) for a, b in zip(before, after)),
                         "fit() left the weights untouched")

    def test_early_stopping_installs_the_kept_weights(self):
        """fit() must end with the kept weights, not whatever the last epoch left.

        The loop tracks bestWeights and calls set_weights at the end; if that
        final call were dropped, the model would silently keep the last epoch's
        weights instead of the ones early stopping selected.
        """
        X, Y = self._data()
        model = DNNModel(neurons=[16], n_features=X.shape[1], lr=0.01,
                         epochs_early_stopping=2)

        captured = {}
        real_set = model.model.set_weights

        def spy(weights):
            captured["weights"] = [w.copy() for w in weights]
            return real_set(weights)

        model.model.set_weights = spy
        with mock.patch("dnn_dk1.model.MAX_EPOCHS", 5):
            model.fit(X, Y, X, Y)

        self.assertIn("weights", captured, "fit() never restored any weights")
        for kept, installed in zip(captured["weights"], model.model.get_weights()):
            np.testing.assert_allclose(kept, installed)


class TestSeeding(unittest.TestCase):
    """The paper's DNN ensemble differs only by seed, so seeding must bite."""

    def test_same_seed_gives_identical_initial_weights(self):
        import keras
        keras.utils.set_random_seed(7)
        a = DNNModel(neurons=[16], n_features=10).model.get_weights()
        keras.utils.set_random_seed(7)
        b = DNNModel(neurons=[16], n_features=10).model.get_weights()
        for wa, wb in zip(a, b):
            np.testing.assert_allclose(wa, wb)

    def test_different_seeds_give_different_initial_weights(self):
        """Upstream seeds only numpy; Keras 3 draws from its own generator, so
        seeding numpy alone would make every ensemble member identical."""
        import keras
        keras.utils.set_random_seed(1)
        a = DNNModel(neurons=[16], n_features=10).model.get_weights()
        keras.utils.set_random_seed(2)
        b = DNNModel(neurons=[16], n_features=10).model.get_weights()
        self.assertFalse(
            all(np.allclose(wa, wb) for wa, wb in zip(a, b)),
            "different seeds produced identical weights -- the ensemble would be "
            "four copies of one model")


class TestDrawSeed(unittest.TestCase):
    """The per-(seed, day) RNG seed that makes a run reproducible.

    epftoolbox draws the train/validation split with

        if hyperoptimization:
            np.random.seed(7)
        np.random.shuffle(index_week)

    so in a backtest -- hyperoptimization False -- the split comes from the
    unseeded global RNG, before recalibrate() seeds anything. Which validation
    days a model sees would then depend on how many draws happened earlier in
    the process: the same day would forecast differently depending on what ran
    before it, no run could be reproduced, and a resumed run would not match the
    uninterrupted one it continued.
    """

    def _forecaster(self, seed):
        obj = DNN.__new__(DNN)
        obj.seed = seed
        obj.best_hyperparameters = {"seed": 999}
        return obj

    def test_same_seed_and_day_gives_the_same_draw(self):
        day = pd.Timestamp("2023-12-01")
        self.assertEqual(self._forecaster(1)._draw_seed(day),
                         self._forecaster(1)._draw_seed(day))

    def test_different_days_draw_differently(self):
        """Otherwise every day would get the identical weekly permutation."""
        f = self._forecaster(1)
        seeds = {f._draw_seed(d) for d in pd.date_range("2023-12-01", periods=50)}
        self.assertEqual(len(seeds), 50)

    def test_different_seeds_draw_differently(self):
        day = pd.Timestamp("2023-12-01")
        values = {self._forecaster(s)._draw_seed(day) for s in (1, 2, 3, 4)}
        self.assertEqual(len(values), 4)

    def test_falls_back_to_the_hyperparameter_seed(self):
        obj = DNN.__new__(DNN)
        obj.seed = None
        obj.best_hyperparameters = {"seed": 7}
        explicit = self._forecaster(7)
        day = pd.Timestamp("2023-12-01")
        self.assertEqual(obj._draw_seed(day), explicit._draw_seed(day))

    def test_stays_within_numpys_accepted_range(self):
        """numpy's legacy seeding rejects anything outside 32 bits."""
        f = self._forecaster(1000)
        for day in pd.date_range("2015-01-01", periods=200, freq="17D"):
            value = f._draw_seed(day)
            self.assertGreaterEqual(value, 0)
            self.assertLess(value, 2 ** 31 - 1)


class TestHyperparameterFile(unittest.TestCase):

    def test_path_matches_upstream_naming(self):
        path = hyperparameter_path("/tmp", experiment_id=1, nlayers=2,
                                   dataset="DK1", years_test=2, shuffle_train=1,
                                   data_augmentation=0, calibration_window=4)
        self.assertEqual(os.path.basename(path),
                         "DNN_hyperparameters_nl2_datDK1_YT2_SF_CW4_1")

    def test_flags_drop_out_of_the_name_when_off(self):
        path = hyperparameter_path("/tmp", experiment_id=3, nlayers=3,
                                   dataset="NO1", years_test=2, shuffle_train=0,
                                   data_augmentation=0, calibration_window=2)
        self.assertEqual(os.path.basename(path),
                         "DNN_hyperparameters_nl3_datNO1_YT2_CW2_3")

    def test_missing_file_explains_how_to_make_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as caught:
                DNN(path_hyperparameter_folder=tmp, dataset="DK1")
            message = str(caught.exception)
            self.assertIn("hyperopt", message.lower())

    def test_no_source_of_hyperparameters_is_rejected(self):
        with self.assertRaises(ValueError):
            DNN()


class TestSearchSpace(unittest.TestCase):
    """The space must match _dnn_hyperopt.py's, including the feature flags."""

    def test_feature_flags_cover_price_lags_and_every_exogenous(self):
        space = build_space(nlayers=2, data_augmentation=0, n_exogenous_inputs=2)
        for name in ("In: Day", "In: Price D-1", "In: Price D-2", "In: Price D-3",
                     "In: Price D-7"):
            self.assertIn(name, space)
        for n in (1, 2):
            for lag in ("D", "D-1", "D-7"):
                self.assertIn(f"In: Exog-{n} {lag}", space)
        self.assertNotIn("In: Exog-3 D", space)

    def test_one_neuron_entry_per_layer(self):
        for nlayers in (1, 2, 3, 4, 5):
            space = build_space(nlayers, 0, 1)
            present = [k for k in space if k.startswith("neurons")]
            self.assertEqual(len(present), nlayers)

    def test_network_hyperparameters_are_all_searched(self):
        space = build_space(nlayers=2, data_augmentation=0, n_exogenous_inputs=1)
        for name in ("batch_normalization", "dropout", "lr", "seed", "activation",
                     "init", "reg", "scaleX", "scaleY"):
            self.assertIn(name, space)

    def test_minimum_neuron_width_matches_upstream(self):
        self.assertEqual(MIN_NEURONS, 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
