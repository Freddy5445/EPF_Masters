"""
The daily-recalibration wrapper around :class:`dnn_dk1.model.DNNModel`.

Mirrors ``epftoolbox.models.DNN``. The differences are deliberate and small:

* it builds our Keras 3 :class:`DNNModel` instead of the Keras 2 one;
* the hyperparameter file can be given by path, so a caller is not forced into
  upstream's ``DNN_hyperparameters_nl2_datPJM_YT2_SF_CW4_1`` naming scheme;
* a missing hyperparameter file says what is missing and how to produce it,
  rather than raising a bare ``FileNotFoundError`` from inside ``pickle``.

Feature construction is upstream's ``_build_and_split_XYs`` and the scalers are
upstream's, reached through :mod:`dnn_dk1.scaling_compat` -- a thin wrapper that
guards the zero-MAD divide, exactly as ``lear_dk1.compat`` does for LEAR. Both are pure
numpy/pandas with no Keras in them, and they are where the model's actual
specification lives -- which lags of which series enter, how the validation
split is drawn, how each column is normalised. Rewriting that by hand is how a
reimplementation silently stops matching the paper.
"""

import os
import sys

import numpy as np
import pandas as pd

from .model import DNNModel, PROJECT_ROOT

if os.path.join(PROJECT_ROOT, "epftoolbox") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "epftoolbox"))

from epftoolbox.models._dnn import _build_and_split_XYs, format_best_trial  # noqa: E402

from .scaling_compat import guarded_scaling  # noqa: E402

import pickle as pc  # noqa: E402

# Upstream's recalibration settings, fixed rather than searched.
EPOCHS_EARLY_STOPPING = 20
LOSS = 'mae'
OPTIMIZER = 'adam'

# A hidden layer narrower than this is dropped entirely, so the network can
# choose its own depth within the layer budget. Upstream's rule.
MIN_NEURONS = 50


def hyperparameter_path(folder, experiment_id, nlayers=2, dataset='DK1',
                        years_test=2, shuffle_train=1, data_augmentation=0,
                        calibration_window=4):
    """Upstream's trials-file name, so files stay interchangeable with it."""
    name = ('DNN_hyperparameters_nl' + str(nlayers) +
            '_dat' + str(dataset) + '_YT' + str(years_test) +
            '_SF' * int(shuffle_train) + '_DA' * int(data_augmentation) +
            '_CW' + str(calibration_window) + '_' + str(experiment_id))
    return os.path.join(folder, name)


def draw_seed(base, day):
    """A stable seed for one (seed, day) pair.

    Mixed with the day so consecutive days do not all draw the identical weekly
    permutation, and kept inside 32 bits because that is what numpy's legacy
    seeding accepts.
    """
    return (int(base) * 100_003 + day.toordinal()) % (2 ** 31 - 1)


class DNN:
    """A DNN that recalibrates daily and forecasts one day at a time.

    ``calibration_window`` is in **years**, not days -- upstream trains on the
    last ``calibration_window * 364`` days. This is the one place where the DNN
    and LEAR use the same word for different units, so it is worth stating: LEAR's
    windows are 56/84/1092/1456 days, the DNN's is 4 (years).
    """

    def __init__(self, hyperparameter_file=None, experiment_id=1,
                 path_hyperparameter_folder=None, nlayers=2, dataset='DK1',
                 years_test=2, shuffle_train=1, data_augmentation=0,
                 calibration_window=4, seed=None):
        if hyperparameter_file is None:
            if path_hyperparameter_folder is None:
                raise ValueError(
                    "Give either hyperparameter_file or path_hyperparameter_folder")
            hyperparameter_file = hyperparameter_path(
                path_hyperparameter_folder, experiment_id, nlayers, dataset,
                years_test, shuffle_train, data_augmentation, calibration_window)

        self.hyperparameter_file = hyperparameter_file
        self.experiment_id = experiment_id
        self.nlayers = nlayers
        self.dataset = dataset
        self.years_test = years_test
        self.shuffle_train = shuffle_train
        self.data_augmentation = data_augmentation
        self.calibration_window = calibration_window

        # The paper's DNN ensemble averages runs that differ only in their random
        # seed. `seed=None` keeps the seed hyperopt selected, which is what a
        # single run uses.
        self.seed = seed
        self.scaler = None
        self.model = None

        self._read_best_hyperparameters()

    def _read_best_hyperparameters(self):
        if not os.path.exists(self.hyperparameter_file):
            raise FileNotFoundError(
                f"No hyperparameter file at {self.hyperparameter_file}. The DNN "
                f"cannot be built without one: hyperopt selects the input features "
                f"as well as the layer sizes, so there is no sensible default. "
                f"Produce one with dnn_dk1.hyperopt.optimize (or "
                f"`python run_dnn_dk1.py --smoke`, which runs a short search first)."
            )
        with open(self.hyperparameter_file, "rb") as handle:
            trials = pc.load(handle)
        self.best_hyperparameters = format_best_trial(trials.best_trial)

    def _draw_seed(self, day):
        """A stable seed for one (seed, day) pair."""
        return draw_seed(
            self.best_hyperparameters['seed'] if self.seed is None else self.seed,
            day)

    def _regularize_data(self, Xtrain, Xval, Xtest, Ytrain, Yval):
        """Scale inputs and outputs by whatever hyperopt chose."""
        if self.best_hyperparameters['scaleX'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
            [Xtrain, Xval, Xtest], _ = guarded_scaling(
                [Xtrain, Xval, Xtest], self.best_hyperparameters['scaleX'])

        if self.best_hyperparameters['scaleY'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
            [Ytrain, Yval], self.scaler = guarded_scaling(
                [Ytrain, Yval], self.best_hyperparameters['scaleY'])
        else:
            self.scaler = None

        return Xtrain, Xval, Xtest, Ytrain, Yval

    def recalibrate(self, Xtrain, Ytrain, Xval, Yval):
        """Train a fresh network from scratch on this window."""
        neurons = [int(self.best_hyperparameters['neurons' + str(k)])
                   for k in range(1, self.nlayers + 1)
                   if int(self.best_hyperparameters['neurons' + str(k)]) >= MIN_NEURONS]

        seed = int(self.best_hyperparameters['seed'] if self.seed is None else self.seed)
        np.random.seed(seed)
        # Upstream seeds only numpy, which was enough for Keras 2's initialisers.
        # Keras 3 draws from its own generator, so without this the ensemble
        # members would be identical and averaging them would do nothing.
        import keras
        keras.utils.set_random_seed(seed)

        self.model = DNNModel(
            neurons=neurons, n_features=Xtrain.shape[-1],
            dropout=self.best_hyperparameters['dropout'],
            batch_normalization=self.best_hyperparameters['batch_normalization'],
            lr=self.best_hyperparameters['lr'], verbose=False,
            optimizer=OPTIMIZER, activation=self.best_hyperparameters['activation'],
            epochs_early_stopping=EPOCHS_EARLY_STOPPING, scaler=self.scaler,
            loss=LOSS, regularization=self.best_hyperparameters['reg'],
            lambda_reg=self.best_hyperparameters['lambdal1'],
            initializer=self.best_hyperparameters['init'])

        self.model.fit(Xtrain, Ytrain, Xval, Yval)

    def recalibrate_predict(self, Xtrain, Ytrain, Xval, Yval, Xtest):
        self.recalibrate(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval, Yval=Yval)
        Yp = self.predict(X=Xtest)
        self.model.clear_session()
        return Yp

    def predict(self, X):
        Yp = self.model.predict(X).squeeze()
        if self.best_hyperparameters['scaleY'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
            Yp = self.scaler.inverse_transform(Yp.reshape(1, -1))
        return Yp

    def recalibrate_and_forecast_next_day(self, df, next_day_date):
        """Retrain on everything before ``next_day_date``, then forecast that day.

        Only data strictly before the forecast day enters training, and the test
        frame reaches back two weeks solely so the lagged features of the
        forecast day can be built. Nothing after the forecast day is read.

        The RNG is seeded from (seed, day) first. This matters more than it
        looks: ``_build_and_split_XYs`` draws the train/validation split with

            if hyperoptimization:
                np.random.seed(7)
            np.random.shuffle(index_week)

        so during a backtest -- where ``hyperoptimization`` is False -- the split
        is drawn from the *unseeded global* RNG, and it is drawn before
        ``recalibrate`` gets to call ``set_random_seed``. Left alone, which
        validation days a model sees would depend on how many random draws
        happened earlier in the process, so the same day would forecast
        differently depending on what ran before it. A run could not be
        reproduced, and an interrupted run could not be resumed without changing
        the forecasts it had already made.

        Seeding here fixes each (day, seed) pair independently of execution
        history. It does not change what is drawn from -- the split is still a
        uniform random weekly permutation, as upstream intends -- only that the
        draw is repeatable.
        """
        import keras
        keras.utils.set_random_seed(self._draw_seed(next_day_date))

        df_train = df.loc[:next_day_date - pd.Timedelta(hours=1)]
        df_train = df_train.loc[
            next_day_date - pd.Timedelta(hours=self.calibration_window * 364 * 24):]

        df_test = df.loc[next_day_date - pd.Timedelta(weeks=2):, :]

        Xtrain, Ytrain, Xval, Yval, Xtest, _, _ = _build_and_split_XYs(
            dfTrain=df_train, features=self.best_hyperparameters,
            shuffle_train=True, dfTest=df_test, date_test=next_day_date,
            data_augmentation=self.data_augmentation,
            n_exogenous_inputs=len(df_train.columns) - 1)

        Xtrain, Xval, Xtest, Ytrain, Yval = self._regularize_data(
            Xtrain=Xtrain, Xval=Xval, Xtest=Xtest, Ytrain=Ytrain, Yval=Yval)

        return self.recalibrate_predict(Xtrain=Xtrain, Ytrain=Ytrain, Xval=Xval,
                                        Yval=Yval, Xtest=Xtest)


# ---------------------------------------------------------------------------
# DNN-wide and DNN-joint
# ---------------------------------------------------------------------------

def first_layer_weight_magnitudes(model, labels):
    """Mean absolute first-layer weight, grouped by ``labels``.

    With the block toggles gone for DNN-wide and DNN-joint, there are no toggle
    states to report: which inputs the model actually used is visible only in the
    weights. ``labels[i]`` names the group input column ``i`` belongs to -- a zone
    (see :func:`dnn_dk1.zones.feature_zone_labels`) or a ``<zone>/<block>``.

    Averaged per input column rather than summed, so groups of different width --
    a 312-column zone and a 240-column one -- stay comparable.
    """
    kernel = None
    for layer in model.model.layers:
        weights = layer.get_weights()
        if weights and weights[0].ndim == 2:
            kernel = weights[0]
            break
    if kernel is None:
        raise ValueError("no dense kernel found in the first layers of the model")
    if kernel.shape[0] != len(labels):
        raise ValueError(
            f"first-layer kernel has {kernel.shape[0]} input rows but "
            f"{len(labels)} labels were given")

    magnitudes = np.abs(kernel).mean(axis=1)
    labels = np.asarray(labels)
    return {str(group): float(magnitudes[labels == group].mean())
            for group in dict.fromkeys(labels.tolist())}


class MultiZoneDNN:
    """DNN-wide and DNN-joint: one network over every zone's inputs.

    The two configurations are the *same object* with a different ``out_zones``:

    * **DNN-wide** -- one focal zone, a 24-neuron output layer;
    * **DNN-joint** -- every zone, a 24*|Z|-neuron output layer.

    Inputs, search space, calibration window, seeds and recalibration cadence are
    identical between them. That is the point: wide to joint must change the
    output layer and nothing else, or the input effect and the output effect
    cannot be separated.

    ``calibration_window`` is in **years**, as in :class:`DNN`.
    """

    def __init__(self, hyperparameter_file=None, experiment_id=1,
                 path_hyperparameter_folder=None, nlayers=2, dataset=None,
                 years_test=2, shuffle_train=1, data_augmentation=0,
                 calibration_window=4, seed=None, zones=None, out_zones=None):
        from . import zones as _zones

        self.zones = tuple(zones if zones is not None else _zones.ZONES)
        if out_zones is None:
            raise ValueError(
                "out_zones must be given: it is what distinguishes DNN-wide "
                "(one focal zone) from DNN-joint (the whole zone set)")
        self.out_zones = tuple(out_zones)
        unknown = [z for z in self.out_zones if z not in self.zones]
        if unknown:
            raise ValueError(f"out_zones {unknown} are not in the input zone set")

        if hyperparameter_file is None:
            if path_hyperparameter_folder is None or dataset is None:
                raise ValueError(
                    "Give either hyperparameter_file or "
                    "path_hyperparameter_folder together with dataset")
            hyperparameter_file = hyperparameter_path(
                path_hyperparameter_folder, experiment_id, nlayers, dataset,
                years_test, shuffle_train, data_augmentation, calibration_window)

        self.hyperparameter_file = hyperparameter_file
        self.experiment_id = experiment_id
        self.nlayers = nlayers
        self.dataset = dataset
        self.years_test = years_test
        self.shuffle_train = shuffle_train
        self.data_augmentation = data_augmentation
        self.calibration_window = calibration_window
        self.seed = seed
        self.scaler = None
        self.model = None
        self.zone_weights = None
        self.transformed_dispersion = None

        DNN._read_best_hyperparameters(self)

    # The output width is the only structural difference between wide and joint.
    @property
    def output_shape(self):
        return 24 * len(self.out_zones)

    @property
    def include_calendar(self):
        return bool(self.best_hyperparameters["In: Day"])

    def _draw_seed(self, day):
        return draw_seed(
            self.best_hyperparameters["seed"] if self.seed is None else self.seed,
            day)

    def training_days(self, matrices, next_day_date):
        """The days the network trains on: the calibration window before the day."""
        from . import zones as _zones

        return _zones.training_days(
            matrices, next_day_date, self.calibration_window)

    def recalibrate_and_forecast_next_day(self, matrices, next_day_date):
        """Retrain on the window before ``next_day_date``, then forecast that day.

        Returns a ``(1, 24 * len(out_zones))`` array on the price scale, ordered
        zone-major then hour -- so DNN-joint's output slices apart by zone with
        :func:`dnn_dk1.zones.zone_slice`.

        The RNG is seeded from (seed, day) before anything is built, for the same
        reason as in :meth:`DNN.recalibrate_and_forecast_next_day`: the
        train/validation split is a random weekly permutation drawn outside any
        seeding of its own, so without this the forecast for a given day would
        depend on what ran before it in the process.
        """
        from . import zones as _zones

        import keras
        keras.utils.set_random_seed(self._draw_seed(next_day_date))

        train_days = self.training_days(matrices, next_day_date)
        if not len(train_days):
            raise ValueError(
                f"no training days before {next_day_date.date()} inside the "
                f"{self.calibration_window}-year window")

        include_calendar = self.include_calendar
        X = _zones.build_X(matrices, train_days, self.zones, include_calendar)
        Y = _zones.build_Y(matrices, train_days, self.out_zones)
        Xtrain, Ytrain, Xval, Yval = _zones.split_train_val(
            X, Y, shuffle_train=True, hyperoptimization=False)
        Xtest = _zones.build_X(
            matrices, pd.DatetimeIndex([next_day_date]), self.zones,
            include_calendar)

        scale_x = self.best_hyperparameters["scaleX"]
        if scale_x in ["Norm", "Norm1", "Std", "Median", "Invariant"]:
            [Xtrain, Xval, Xtest], _ = guarded_scaling([Xtrain, Xval, Xtest], scale_x)

        scale_y = self.best_hyperparameters["scaleY"]
        if scale_y in ["Norm", "Norm1", "Std", "Median", "Invariant"]:
            self.scaler = _zones.PerZoneScaler(scale_y, self.out_zones)
            y_train_s = self.scaler.fit_transform(Ytrain)
            y_val_s = self.scaler.transform(Yval)
            self.transformed_dispersion = self.scaler.dispersion(y_train_s)
        else:
            self.scaler = None
            y_train_s, y_val_s = Ytrain, Yval
            self.transformed_dispersion = {
                z: float(np.std(Ytrain[:, _zones.zone_slice(z, self.out_zones)]))
                for z in self.out_zones}

        self.recalibrate(Xtrain, y_train_s, Xval, y_val_s)

        # Which zones the model gave weight to. With no block toggles left this
        # is the only record of what the network actually used, so it is taken
        # before the session is cleared.
        self.zone_weights = first_layer_weight_magnitudes(
            self.model, _zones.feature_zone_labels(self.zones, include_calendar))

        Yp = self.model.predict(Xtest)
        if self.scaler is not None:
            Yp = self.scaler.inverse_transform(Yp)
        self.model.clear_session()
        return np.asarray(Yp, dtype=float).reshape(1, -1)

    def recalibrate(self, Xtrain, Ytrain, Xval, Yval):
        """Train a fresh network from scratch on this window."""
        neurons = [int(self.best_hyperparameters["neurons" + str(k)])
                   for k in range(1, self.nlayers + 1)
                   if int(self.best_hyperparameters["neurons" + str(k)]) >= MIN_NEURONS]

        seed = int(self.best_hyperparameters["seed"] if self.seed is None else self.seed)
        np.random.seed(seed)
        import keras
        keras.utils.set_random_seed(seed)

        self.model = DNNModel(
            neurons=neurons, n_features=Xtrain.shape[-1],
            outputShape=self.output_shape, output_zones=self.out_zones,
            dropout=self.best_hyperparameters["dropout"],
            batch_normalization=self.best_hyperparameters["batch_normalization"],
            lr=self.best_hyperparameters["lr"], verbose=False,
            optimizer=OPTIMIZER, activation=self.best_hyperparameters["activation"],
            epochs_early_stopping=EPOCHS_EARLY_STOPPING, scaler=self.scaler,
            loss=LOSS, regularization=self.best_hyperparameters["reg"],
            lambda_reg=self.best_hyperparameters["lambdal1"],
            initializer=self.best_hyperparameters["init"])

        self.model.fit(Xtrain, Ytrain, Xval, Yval)
