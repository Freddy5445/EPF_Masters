"""
Hyperparameter and feature selection for the DNN.

Two search spaces live here.

**DNN-own** (:func:`build_space`, :func:`optimize`) mirrors
``epftoolbox.models.hyperparameter_optimizer`` exactly: the same 11 binary
feature toggles, the same architecture ranges, the same TPE algorithm and the
same trials-file format, so the file this writes is readable by upstream's
``format_best_trial`` -- and by upstream itself, on a stack where it runs. It has
to be reimplemented rather than called for one reason: upstream's objective
builds ``epftoolbox.models.DNNModel`` directly, which is the Keras 2 class that
Keras 3 rejects. DNN-own therefore *is* the Lago et al. (2021) DNN.

**DNN-wide / DNN-joint** (:func:`build_multizone_space`,
:func:`optimize_multizone`) drop the block toggles entirely. Generalised naively
to the seven zones of :data:`dnn_dk1.zones.ZONES`, upstream's (variable, day-set)
granularity gives ``7 zones x (4 + 3*n_exo) + 1`` = 83 binaries -- a 2^83
block-selection space that no realistic budget explores and in which TPE
degenerates to random sampling. All inputs are always present instead, and
selection moves from block-level (binary) to weight-level: the L1 penalty on the
first-layer kernel, whose coefficient is already a tuned hyperparameter. Only the
calendar toggle survives.

That leaves wide and joint on an *identical* space, differing in the output layer
alone -- which is the whole point of the wide -> joint comparison.

Worth knowing before running the DNN-own search: it chooses the *input features*
as well as the network -- whether price at D-1, D-2, D-3, D-7 enters, and which
lags of each exogenous series. That is why the DNN cannot simply be handed a
sensible default architecture the way LEAR can be handed a calibration window.
"""

import os
import pickle as pc
import sys
import time
from functools import partial

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from .model import DNNModel, PROJECT_ROOT

if os.path.join(PROJECT_ROOT, "epftoolbox") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "epftoolbox"))

from epftoolbox.evaluation import MAE, sMAPE  # noqa: E402
from epftoolbox.models._dnn import _build_and_split_XYs  # noqa: E402

from . import zones as Z  # noqa: E402
from .forecaster import MIN_NEURONS, hyperparameter_path  # noqa: E402
from .scaling_compat import guarded_scaling  # noqa: E402

# Widths searched per layer, upstream's ranges. Chosen for a ~240-input network.
NEURON_RANGES = {1: (50, 500), 2: (25, 400), 3: (25, 300), 4: (25, 200), 5: (25, 200)}

# Widened for the 1969-input configurations. Upstream's ceiling of 500 first-layer
# neurons was picked for an input eight times narrower; at this width the first
# layer is the bottleneck, so the ceiling is doubled. Widening the search *space*
# is legitimate -- the same procedure still runs for every configuration. Widening
# the *budget* for one configuration is not (see the spec's section 6), and the
# evaluation count is held identical across all ten runs.
MULTIZONE_NEURON_RANGES = {1: (50, 1000), 2: (25, 800), 3: (25, 600),
                           4: (25, 400), 5: (25, 400)}

# L1 range. Upstream searches 1e-5..1; with block toggles gone, L1 is the *only*
# instrument that can prune the ~1650 cross-zonal columns, so the ceiling is
# raised an order of magnitude to make a strongly-pruning solution reachable.
L1_RANGE = (1e-5, 1.0)
MULTIZONE_L1_RANGE = (1e-5, 10.0)

# Hyperparameters common to both spaces -- everything that is not a feature
# toggle. This is what "identical search space" means for wide vs joint.
ARCHITECTURE_KEYS = (
    "batch_normalization", "dropout", "lr", "seed", "activation", "init",
    "reg", "scaleX", "scaleY",
)


def _architecture_space(neuron_ranges, l1_range, nlayers):
    """The non-feature part of the space: identical for every configuration."""
    space = {
        'batch_normalization': hp.choice('batch_normalization', [False, True]),
        'dropout': hp.uniform('dropout', 0, 1),
        'lr': hp.loguniform('lr', np.log(5e-4), np.log(0.1)),
        'seed': hp.quniform('seed', 1, 1000, 1),
        'activation': hp.choice('activation', ["relu", "softplus", "tanh", 'selu',
                                               'LeakyReLU', 'PReLU', 'sigmoid']),
        'init': hp.choice('init', ['Orthogonal', 'lecun_uniform', 'glorot_uniform',
                                   'glorot_normal', 'he_uniform', 'he_normal']),
        'reg': hp.choice('reg', [
            {'val': None, 'lambda': 0},
            {'val': 'l1', 'lambda': hp.loguniform(
                'lambdal1', np.log(l1_range[0]), np.log(l1_range[1]))}]),
        'scaleX': hp.choice('scaleX', ['No', 'Norm', 'Norm1', 'Std', 'Median', 'Invariant']),
        'scaleY': hp.choice('scaleY', ['No', 'Norm', 'Norm1', 'Std', 'Median', 'Invariant']),
    }
    for layer in range(1, nlayers + 1):
        low, high = neuron_ranges[layer]
        space['neurons' + str(layer)] = hp.quniform(
            'neurons' + str(layer), low, high, 1)
    return space


def build_space(nlayers, data_augmentation, n_exogenous_inputs):
    """Upstream's search space: network, scaling, and which inputs to use."""
    space = _architecture_space(NEURON_RANGES, L1_RANGE, nlayers)

    # The input features are searched too, not fixed.
    for name in ('In: Day', 'In: Price D-1', 'In: Price D-2', 'In: Price D-3',
                 'In: Price D-7'):
        space[name] = hp.choice(name, [False, True])

    for n_ex in range(1, n_exogenous_inputs + 1):
        for lag in ('D', 'D-1', 'D-7'):
            name = f'In: Exog-{n_ex} {lag}'
            space[name] = hp.choice(name, [False, True])

    return space


def build_multizone_space(nlayers):
    """The DNN-wide / DNN-joint space: architecture plus the calendar toggle only.

    Identical for both configurations by construction -- they call this with the
    same ``nlayers`` and differ only in how many output neurons the network is
    built with.
    """
    space = _architecture_space(MULTIZONE_NEURON_RANGES, MULTIZONE_L1_RANGE, nlayers)
    space['In: Day'] = hp.choice('In: Day', [False, True])
    return space


def space_summary(space):
    """The searched dimensions, for the run manifest.

    ``reg`` carries a nested ``lambdal1`` that is only sampled on the l1 branch,
    so it counts as two dimensions.
    """
    keys = sorted(space)
    return {
        "dimensions": keys,
        "n_dimensions": len(keys) + 1,   # + lambdal1, nested inside reg
        "n_feature_toggles": sum(1 for k in keys if k.startswith("In: ")),
    }


class SearchWindowError(ValueError):
    """Raised when a hyperparameter search would see the test period."""


def assert_search_window_precedes_test(begin_test_date, end_test_date, dataset,
                                       test_start=None):
    """Refuse a search window that touches the test period.

    Both optimisers take ``begin_test_date`` / ``end_test_date`` and use them to
    define the *search's* own training and evaluation days. Those must be the
    ~364 days before :data:`dnn_dk1.zones.BEGIN_TEST`. Passing the real test
    range instead selects the architecture, the features and the scalers on the
    very days the model is then evaluated on.

    This is the one invariant in the pipeline that was enforced by convention
    rather than by code, and it is the most dangerous one to lose: a violated
    run does not crash, does not warn, and produces results that look
    *excellent*. Nothing downstream would catch it -- the forecasts are
    well-formed, the evaluator scores them happily, and the only symptom is an
    implausibly low MAE, which is exactly the outcome one is least inclined to
    question.

    The search's training days always precede ``begin_test_date``, so requiring
    the evaluation window to end strictly before the test period is sufficient.
    """
    test_start = pd.Timestamp(Z.BEGIN_TEST if test_start is None else test_start)
    begin = pd.Timestamp(begin_test_date)
    end = pd.Timestamp(end_test_date)

    if begin > end:
        raise SearchWindowError(
            f"search window for {dataset} starts {begin} after it ends {end}")
    if end >= test_start:
        raise SearchWindowError(
            f"the hyperparameter search for {dataset} was given the window "
            f"{begin}..{end}, which reaches into the test period beginning "
            f"{test_start}. The search selects the architecture, the input "
            f"features and the scalers, so a window overlapping the test period "
            f"invalidates every number the run produces -- and does so "
            f"silently, by making them look better. Pass the ~"
            f"{(test_start - pd.Timedelta(days=364)).date()}.."
            f"{(test_start - pd.Timedelta(hours=1))} window instead "
            f"(run_dnn_dk1.hyperopt_window derives it).")


def n_binary_toggles_if_generalised(zones=Z.ZONES):
    """How many binaries upstream's granularity would give over ``zones``.

    Recorded in the manifest as the justification for dropping them: this is the
    exponent of the block-selection space the adopted rule avoids.
    """
    return sum(len(Z.PRICE_LAGS) + len(Z.EXOG_LAGS) * Z.n_exogenous(z)
               for z in zones) + 1


# ---------------------------------------------------------------------------
# DNN-own: upstream's objective, on upstream's features
# ---------------------------------------------------------------------------

def _objective(hyperparameters, trials, trials_file_path, max_evals, nlayers,
               dfTrain, dfTest, shuffle_train, dataset, data_augmentation,
               calibration_window, n_exogenous_inputs, quiet=False,
               record=None):
    """One hyperopt evaluation: build the features, train a net, score it."""
    started = time.time()
    dfTrain_cw = dfTrain.loc[
        dfTrain.index[-1] - pd.Timedelta(weeks=52) * calibration_window
        + pd.Timedelta(hours=1):]

    # Checkpoint before training, so an interrupted search keeps its trials.
    with open(trials_file_path, "wb") as handle:
        pc.dump(trials, handle)

    if not quiet and trials.losses()[0] is not None:
        best = trials.best_trial['result']
        print('\nTested {}/{} iterations.'.format(len(trials.losses()) - 1, max_evals))
        print('Best MAE - Validation Dataset')
        print("  MAE: {:.1f} | sMAPE: {:.2f} %".format(best['MAE Val'], best['sMAPE Val']))
        print('Best MAE - Test Dataset')
        print("  MAE: {:.1f} | sMAPE: {:.2f} %".format(best['MAE Test'], best['sMAPE Test']))

    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, _ = _build_and_split_XYs(
        dfTrain=dfTrain_cw, dfTest=dfTest, features=hyperparameters,
        shuffle_train=shuffle_train, hyperoptimization=True,
        data_augmentation=data_augmentation, n_exogenous_inputs=n_exogenous_inputs)

    if hyperparameters['scaleX'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        [Xtrain, Xval, Xtest], _ = guarded_scaling([Xtrain, Xval, Xtest], hyperparameters['scaleX'])

    if hyperparameters['scaleY'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        [Ytrain, Yval], scaler = guarded_scaling([Ytrain, Yval], hyperparameters['scaleY'])
    else:
        scaler = None

    neurons = [int(hyperparameters['neurons' + str(k)]) for k in range(1, nlayers + 1)
               if int(hyperparameters['neurons' + str(k)]) >= MIN_NEURONS]

    seed = int(hyperparameters['seed'])
    np.random.seed(seed)
    import keras
    keras.utils.set_random_seed(seed)

    forecaster = DNNModel(
        neurons=neurons, n_features=Xtrain.shape[-1],
        dropout=hyperparameters['dropout'],
        batch_normalization=hyperparameters['batch_normalization'],
        lr=hyperparameters['lr'], verbose=False, optimizer='adam',
        activation=hyperparameters['activation'], epochs_early_stopping=20,
        scaler=scaler, loss='mae', regularization=hyperparameters['reg']['val'],
        lambda_reg=hyperparameters['reg']['lambda'],
        initializer=hyperparameters['init'])

    forecaster.fit(Xtrain, Ytrain, Xval, Yval)

    Yp_val = forecaster.predict(Xval).squeeze()
    Yp_test = forecaster.predict(Xtest).squeeze()
    if scaler is not None:
        Yval_unscaled = scaler.inverse_transform(Yval)
        Yp_val = scaler.inverse_transform(Yp_val)
    else:
        Yval_unscaled = Yval

    mae_validation = np.mean(MAE(Yval_unscaled, Yp_val))
    smape_validation = np.mean(sMAPE(Yval_unscaled, Yp_val)) * 100

    # The test scores are recorded but never optimised on: hyperopt minimises the
    # validation MAE alone. They are here because upstream reports them, and
    # because a search whose test score diverges from its validation score is
    # worth seeing.
    mae_test = np.mean(MAE(Ytest, Yp_test))
    smape_test = np.mean(sMAPE(Ytest, Yp_test)) * 100

    forecaster.clear_session()

    elapsed = time.time() - started
    if record is not None:
        record.append({"seconds": elapsed, "neurons": neurons,
                       "n_features": int(Xtrain.shape[-1]),
                       "loss": float(mae_validation)})

    return {'loss': mae_validation, 'MAE Val': mae_validation, 'MAE Test': mae_test,
            'sMAPE Val': smape_validation, 'sMAPE Test': smape_test,
            'seconds': elapsed, 'status': STATUS_OK}


def optimize(path_datasets_folder, path_hyperparameters_folder, dataset,
             begin_test_date, end_test_date, max_evals=1500, nlayers=2,
             years_test=2, calibration_window=4, shuffle_train=1,
             data_augmentation=0, experiment_id=1, new_hyperopt=1, quiet=False,
             record=None):
    """Run the DNN-own search and write the trials file. Returns its path.

    ``max_evals`` is 1500 in the paper. A smoke run uses a handful -- enough to
    produce a usable file, not enough to produce a good model.
    """
    from epftoolbox.data import read_data

    # Before anything else, and before any file is written: a search window that
    # touches the test period invalidates the run silently.
    assert_search_window_precedes_test(begin_test_date, end_test_date, dataset)

    os.makedirs(path_hyperparameters_folder, exist_ok=True)
    trials_file_path = hyperparameter_path(
        path_hyperparameters_folder, experiment_id, nlayers, dataset, years_test,
        shuffle_train, data_augmentation, calibration_window)

    if new_hyperopt or not os.path.exists(trials_file_path):
        trials = Trials()
    else:
        with open(trials_file_path, "rb") as handle:
            trials = pc.load(handle)

    dfTrain, dfTest = read_data(
        dataset=dataset, years_test=years_test, path=path_datasets_folder,
        begin_test_date=begin_test_date, end_test_date=end_test_date)

    # Cleaning happens once, in data_cleaning_v2.ipynb. This search used to
    # impute here, which meant the search and the backtest could fill a gap
    # differently; that path is gone and a NaN is now an error, exactly as in
    # run_dnn_dk1.py. (The imputation module it called was removed with the
    # v2 refactor, so this import-time dependency was in fact already broken.)
    combined = pd.concat([dfTrain, dfTest], axis=0)
    if combined.isna().any().any():
        counts = combined.isna().sum()
        raise ValueError(
            f"{dataset} has missing values {counts[counts > 0].to_dict()}. The "
            f"DNN cannot be fitted on NaN and nothing here imputes. Re-run "
            f"data_cleaning_v2.ipynb, then rebuild the CSV with "
            f"run_lear_from_clean.py.")

    n_exogenous_inputs = len(dfTrain.columns) - 1
    space = build_space(nlayers, data_augmentation, n_exogenous_inputs)

    objective = partial(
        _objective, trials=trials, trials_file_path=trials_file_path,
        max_evals=max_evals, nlayers=nlayers, dfTrain=dfTrain, dfTest=dfTest,
        shuffle_train=shuffle_train, dataset=dataset,
        data_augmentation=data_augmentation, calibration_window=calibration_window,
        n_exogenous_inputs=n_exogenous_inputs, quiet=quiet, record=record)

    fmin(objective, space=space, algo=tpe.suggest, max_evals=max_evals,
         trials=trials, verbose=False)

    # fmin's own checkpoint is written before each trial, so the final trial's
    # result would be missing from the file without this.
    with open(trials_file_path, "wb") as handle:
        pc.dump(trials, handle)

    return trials_file_path


# ---------------------------------------------------------------------------
# DNN-wide / DNN-joint
# ---------------------------------------------------------------------------

def _scale_normalisers(Y, out_zones):
    """A fixed per-zone scale for the objective, so zones are weighted equally.

    The objective has to be one number. A plain mean of the per-zone MAEs on the
    price scale is dominated by whichever zones are most expensive -- the same
    argument that makes the targets scale per zone before training. Each zone's
    MAE is therefore divided by that zone's own median absolute deviation, taken
    once from the validation targets and so constant across trials.

    Being constant across trials is what makes this safe: for DNN-wide, which has
    a single output zone, dividing by a constant cannot change which trial wins,
    so wide still selects on exactly upstream's criterion. For DNN-joint it turns
    "average the zones" into something that means what it says.
    """
    from statsmodels.robust import mad

    values = np.asarray(Y, dtype=float)
    out = {}
    for zone in out_zones:
        block = values[:, Z.zone_slice(zone, out_zones)]
        scale = float(np.mean(mad(block, axis=0)))
        out[zone] = scale if scale > 0 else 1.0
    return out


def _per_zone_mae(Yreal, Ypred, out_zones):
    real = np.asarray(Yreal, dtype=float)
    pred = np.asarray(Ypred, dtype=float).reshape(real.shape)
    return {zone: float(np.mean(np.abs(
        real[:, Z.zone_slice(zone, out_zones)]
        - pred[:, Z.zone_slice(zone, out_zones)])))
        for zone in out_zones}


def _multizone_objective(hyperparameters, trials, trials_file_path, max_evals,
                         nlayers, matrices, train_days, test_days, zones,
                         out_zones, quiet=False, record=None):
    """One evaluation of the wide/joint search.

    The only feature decision left is the calendar toggle, so the input matrix is
    rebuilt per trial only because that toggle can drop its one column.
    """
    started = time.time()

    with open(trials_file_path, "wb") as handle:
        pc.dump(trials, handle)

    if not quiet and trials.losses()[0] is not None:
        best = trials.best_trial['result']
        print('\nTested {}/{} iterations.'.format(len(trials.losses()) - 1, max_evals))
        print("  best loss {:.4f} | MAE Val {:.2f} | MAE Test {:.2f}".format(
            best['loss'], best['MAE Val'], best['MAE Test']))

    include_calendar = bool(hyperparameters['In: Day'])
    Xall = Z.build_X(matrices, train_days, zones, include_calendar)
    Yall = Z.build_Y(matrices, train_days, out_zones)
    Xtrain, Ytrain, Xval, Yval = Z.split_train_val(
        Xall, Yall, shuffle_train=True, hyperoptimization=True)
    Xtest = Z.build_X(matrices, test_days, zones, include_calendar)
    Ytest = Z.build_Y(matrices, test_days, out_zones)

    if hyperparameters['scaleX'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        [Xtrain, Xval, Xtest], _ = guarded_scaling(
            [Xtrain, Xval, Xtest], hyperparameters['scaleX'])

    if hyperparameters['scaleY'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        scaler = Z.PerZoneScaler(hyperparameters['scaleY'], out_zones)
        Ytrain_s = scaler.fit_transform(Ytrain)
        Yval_s = scaler.transform(Yval)
    else:
        scaler = None
        Ytrain_s, Yval_s = Ytrain, Yval

    neurons = [int(hyperparameters['neurons' + str(k)]) for k in range(1, nlayers + 1)
               if int(hyperparameters['neurons' + str(k)]) >= MIN_NEURONS]

    seed = int(hyperparameters['seed'])
    np.random.seed(seed)
    import keras
    keras.utils.set_random_seed(seed)

    forecaster = DNNModel(
        neurons=neurons, n_features=Xtrain.shape[-1],
        outputShape=Z.output_width(out_zones), output_zones=out_zones,
        dropout=hyperparameters['dropout'],
        batch_normalization=hyperparameters['batch_normalization'],
        lr=hyperparameters['lr'], verbose=False, optimizer='adam',
        activation=hyperparameters['activation'], epochs_early_stopping=20,
        scaler=scaler, loss='mae', regularization=hyperparameters['reg']['val'],
        lambda_reg=hyperparameters['reg']['lambda'],
        initializer=hyperparameters['init'])

    forecaster.fit(Xtrain, Ytrain_s, Xval, Yval_s)

    Yp_val = forecaster.predict(Xval)
    Yp_test = forecaster.predict(Xtest)
    if scaler is not None:
        Yp_val = scaler.inverse_transform(Yp_val)
        Yp_test = scaler.inverse_transform(Yp_test)

    normalisers = _scale_normalisers(Yval, out_zones)
    mae_val_zone = _per_zone_mae(Yval, Yp_val, out_zones)
    mae_test_zone = _per_zone_mae(Ytest, Yp_test, out_zones)
    loss = float(np.mean([mae_val_zone[z] / normalisers[z] for z in out_zones]))

    forecaster.clear_session()

    elapsed = time.time() - started
    if record is not None:
        record.append({"seconds": elapsed, "neurons": neurons,
                       "n_features": int(Xtrain.shape[-1]),
                       "n_outputs": Z.output_width(out_zones),
                       "loss": loss})

    return {
        'loss': loss,
        # Reported per zone as well as pooled: a pooled price-scale figure is
        # dominated by whichever zones are easiest, so it is recorded but never
        # optimised on and never reported as the result (spec section 4.4).
        'MAE Val': float(np.mean(list(mae_val_zone.values()))),
        'MAE Test': float(np.mean(list(mae_test_zone.values()))),
        'MAE Val per zone': mae_val_zone,
        'MAE Test per zone': mae_test_zone,
        'seconds': elapsed,
        'status': STATUS_OK,
    }


def optimize_multizone(matrices, path_hyperparameters_folder, dataset,
                       begin_test_date, end_test_date, out_zones,
                       zones=Z.ZONES, max_evals=1500, nlayers=2, years_test=2,
                       calibration_window=4, shuffle_train=1, data_augmentation=0,
                       experiment_id=1, new_hyperopt=1, quiet=False, record=None):
    """Run the DNN-wide / DNN-joint search. Returns the trials-file path.

    ``out_zones`` is the only difference between the two configurations: one
    focal zone for DNN-wide, all of ``zones`` for DNN-joint. The space, the
    inputs and the budget are identical.
    """
    # Before anything else, and before any file is written: a search window that
    # touches the test period invalidates the run silently.
    assert_search_window_precedes_test(begin_test_date, end_test_date, dataset)

    os.makedirs(path_hyperparameters_folder, exist_ok=True)
    trials_file_path = hyperparameter_path(
        path_hyperparameters_folder, experiment_id, nlayers, dataset, years_test,
        shuffle_train, data_augmentation, calibration_window)

    if new_hyperopt or not os.path.exists(trials_file_path):
        trials = Trials()
    else:
        with open(trials_file_path, "rb") as handle:
            trials = pc.load(handle)

    begin_test_date = pd.Timestamp(begin_test_date).normalize()
    end_test_date = pd.Timestamp(end_test_date).normalize()

    days = Z.available_days(matrices)
    test_days = days[(days >= begin_test_date) & (days <= end_test_date)]
    train_days = Z.training_days(matrices, begin_test_date, calibration_window)
    if not len(test_days) or not len(train_days):
        raise ValueError(
            f"the search window {begin_test_date.date()}..{end_test_date.date()} "
            f"leaves {len(train_days)} training and {len(test_days)} evaluation "
            f"days in a panel spanning {days.min().date()}..{days.max().date()}")

    space = build_multizone_space(nlayers)

    objective = partial(
        _multizone_objective, trials=trials, trials_file_path=trials_file_path,
        max_evals=max_evals, nlayers=nlayers, matrices=matrices,
        train_days=train_days, test_days=test_days, zones=zones,
        out_zones=out_zones, quiet=quiet, record=record)

    fmin(objective, space=space, algo=tpe.suggest, max_evals=max_evals,
         trials=trials, verbose=False)

    with open(trials_file_path, "wb") as handle:
        pc.dump(trials, handle)

    return trials_file_path
