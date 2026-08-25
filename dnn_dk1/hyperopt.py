"""
Hyperparameter and feature selection for the DNN.

Mirrors ``epftoolbox.models.hyperparameter_optimizer``. It has to be
reimplemented rather than called, for one reason: upstream's objective builds
``epftoolbox.models.DNNModel`` directly, which is the Keras 2 class that Keras 3
rejects. The search space, the objective, the TPE algorithm and the trials-file
format are all upstream's, so the file this writes is readable by upstream's
``format_best_trial`` -- and by upstream itself, on a stack where it runs.

Worth knowing before running it: the search chooses the *input features* as well
as the network -- whether price at D-1, D-2, D-3, D-7 enters, and which lags of
each exogenous series. That is why the DNN cannot simply be handed a sensible
default architecture the way LEAR can be handed a calibration window.
"""

import os
import pickle as pc
import sys
from functools import partial

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from .model import DNNModel, PROJECT_ROOT

if os.path.join(PROJECT_ROOT, "epftoolbox") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "epftoolbox"))

from epftoolbox.data import scaling  # noqa: E402
from epftoolbox.evaluation import MAE, sMAPE  # noqa: E402
from epftoolbox.models._dnn import _build_and_split_XYs  # noqa: E402

from .forecaster import MIN_NEURONS, hyperparameter_path  # noqa: E402
from lear_dk1.impute import first_complete_day, impute_frame  # noqa: E402

# Widths searched per layer, upstream's ranges.
NEURON_RANGES = {1: (50, 500), 2: (25, 400), 3: (25, 300), 4: (25, 200), 5: (25, 200)}


def build_space(nlayers, data_augmentation, n_exogenous_inputs):
    """Upstream's search space: network, scaling, and which inputs to use."""
    space = {
        'batch_normalization': hp.choice('batch_normalization', [False, True]),
        'dropout': hp.uniform('dropout', 0, 1),
        'lr': hp.loguniform('lr', np.log(5e-4), np.log(0.1)),
        'seed': hp.quniform('seed', 1, 1000, 1),
        'neurons1': hp.quniform('neurons1', 50, 500, 1),
        'activation': hp.choice('activation', ["relu", "softplus", "tanh", 'selu',
                                               'LeakyReLU', 'PReLU', 'sigmoid']),
        'init': hp.choice('init', ['Orthogonal', 'lecun_uniform', 'glorot_uniform',
                                   'glorot_normal', 'he_uniform', 'he_normal']),
        'reg': hp.choice('reg', [
            {'val': None, 'lambda': 0},
            {'val': 'l1', 'lambda': hp.loguniform('lambdal1', np.log(1e-5), np.log(1))}]),
        'scaleX': hp.choice('scaleX', ['No', 'Norm', 'Norm1', 'Std', 'Median', 'Invariant']),
        'scaleY': hp.choice('scaleY', ['No', 'Norm', 'Norm1', 'Std', 'Median', 'Invariant']),
    }

    for layer in range(2, nlayers + 1):
        low, high = NEURON_RANGES[layer]
        space['neurons' + str(layer)] = hp.quniform('neurons' + str(layer), low, high, 1)

    # The input features are searched too, not fixed.
    for name in ('In: Day', 'In: Price D-1', 'In: Price D-2', 'In: Price D-3',
                 'In: Price D-7'):
        space[name] = hp.choice(name, [False, True])

    for n_ex in range(1, n_exogenous_inputs + 1):
        for lag in ('D', 'D-1', 'D-7'):
            name = f'In: Exog-{n_ex} {lag}'
            space[name] = hp.choice(name, [False, True])

    return space


def _objective(hyperparameters, trials, trials_file_path, max_evals, nlayers,
               dfTrain, dfTest, shuffle_train, dataset, data_augmentation,
               calibration_window, n_exogenous_inputs, quiet=False):
    """One hyperopt evaluation: build the features, train a net, score it."""
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
        [Xtrain, Xval, Xtest], _ = scaling([Xtrain, Xval, Xtest], hyperparameters['scaleX'])

    if hyperparameters['scaleY'] in ['Norm', 'Norm1', 'Std', 'Median', 'Invariant']:
        [Ytrain, Yval], scaler = scaling([Ytrain, Yval], hyperparameters['scaleY'])
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

    return {'loss': mae_validation, 'MAE Val': mae_validation, 'MAE Test': mae_test,
            'sMAPE Val': smape_validation, 'sMAPE Test': smape_test, 'status': STATUS_OK}


def optimize(path_datasets_folder, path_hyperparameters_folder, dataset,
             begin_test_date, end_test_date, max_evals=1500, nlayers=2,
             years_test=2, calibration_window=4, shuffle_train=1,
             data_augmentation=0, experiment_id=1, new_hyperopt=1, quiet=False,
             max_linear=3):
    """Run the search and write the trials file. Returns its path.

    ``max_evals`` is 1500 in the paper. A smoke run uses a handful -- enough to
    produce a usable file, not enough to produce a good model.
    """
    from epftoolbox.data import read_data

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

    # A raw dataset carries real gaps (ENTSO-E does not publish everything), and
    # unlike the backtest path this search never went through lear_dk1.impute --
    # it fit straight on NaN, which Keras propagates into every loss, so *every*
    # trial scored `nan` regardless of hyperparameters. Impute train+test as one
    # frame, exactly as the backtest does, so the boundary is filled consistently.
    test_start = dfTest.index[0]
    combined = pd.concat([dfTrain, dfTest], axis=0)
    if combined.isna().any().any():
        combined, _ = impute_frame(combined, max_ffill=max_linear)
        if combined.isna().any().any():
            combined = combined.loc[first_complete_day(combined):]
    dfTrain = combined.loc[:test_start - pd.Timedelta(hours=1)]
    dfTest = combined.loc[test_start:]

    n_exogenous_inputs = len(dfTrain.columns) - 1
    space = build_space(nlayers, data_augmentation, n_exogenous_inputs)

    objective = partial(
        _objective, trials=trials, trials_file_path=trials_file_path,
        max_evals=max_evals, nlayers=nlayers, dfTrain=dfTrain, dfTest=dfTest,
        shuffle_train=shuffle_train, dataset=dataset,
        data_augmentation=data_augmentation, calibration_window=calibration_window,
        n_exogenous_inputs=n_exogenous_inputs, quiet=quiet)

    fmin(objective, space=space, algo=tpe.suggest, max_evals=max_evals,
         trials=trials, verbose=False)

    # fmin's own checkpoint is written before each trial, so the final trial's
    # result would be missing from the file without this.
    with open(trials_file_path, "wb") as handle:
        pc.dump(trials, handle)

    return trials_file_path
