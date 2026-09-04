"""Profile ``DNNModel._obtain_metrics`` -- the per-epoch validation cost.

Handover section 11.1: ``_obtain_metrics`` calls BOTH ``model.evaluate(valX, valY)``
and ``model.predict(valX)`` every epoch -- two complete forward passes over the
validation set where one would do. What share of a fit is that?

    python profile_obtain_metrics.py --config own   --zone DK1
    python profile_obtain_metrics.py --config wide  --zone DK1
    python profile_obtain_metrics.py --config joint

The fit reproduced is the real one: the hyperparameter file the E=300 search
wrote, the same data, the same 4-year window before 2023-10-01, the same code
path (the forecaster's own data prep, then ``DNNModel.fit``). Five variants of
the validation step are timed on that identical fit:

  A  baseline           upstream: evaluate() + predict()
  B  predict + numpy    one predict(); the loss recomputed from Ybar in numpy
  C  baseline, one batch     as A, but evaluate/predict at batch_size = len(valX)
  D  predict + numpy, one batch
  E  __call__ + numpy   ``model(X, training=False)`` instead of predict()

Every variant must reach the same epoch count and the same best metrics as A; the
script prints both so that a variant which is merely stopping earlier cannot be
mistaken for one that is faster.

Nothing under ``dnn_dk1/`` or ``epftoolbox/`` is modified -- the instrumentation
is a subclass, confined to this file.
"""

import io
import json
import os
import sys
import time
from contextlib import redirect_stdout

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "epftoolbox"))

import keras
from epftoolbox.data import read_data
from epftoolbox.evaluation import MAE

from dnn_dk1 import DNN
from dnn_dk1 import zones as Z
from dnn_dk1.model import BATCH_SIZE, MAX_EPOCHS, DNNModel
from dnn_dk1.scaling_compat import guarded_scaling

DATASET = "DK1_clean_load-wind-solar"
HYPER_DIR = os.path.join(HERE, "experiments", "hyperparameters")
DATA_DIR = os.path.join(HERE, "datasets")
REFIT_DAY = pd.Timestamp("2023-10-01")
BEGIN_TEST = pd.Timestamp("2023-10-01")
END_TEST = pd.Timestamp("2025-09-30")
SEED = 1


# ---------------------------------------------------------------------------
# Instrumented model
# ---------------------------------------------------------------------------

class ProfiledDNNModel(DNNModel):
    """DNNModel with per-call timing, and a switchable validation path.

    mode='baseline'  -- upstream: evaluate() + predict(), both full forward passes
    mode='predict'   -- one predict(), loss recomputed from Ybar in numpy
    mode='bigbatch'  -- upstream, but evaluate/predict at batch_size=len(valX)
    """

    mode = "baseline"
    eval_batch = None

    def reset_timers(self):
        self.t_train = 0.0
        self.t_evaluate = 0.0
        self.t_predict = 0.0
        self.t_numpy = 0.0
        self.t_bookkeeping = 0.0
        self.epochs = 0
        self.loss_pairs = []

    def _obtain_metrics(self, X, Y):
        kw = {} if self.eval_batch is None else {"batch_size": self.eval_batch}

        if self.mode in ("predict", "call"):
            t0 = time.perf_counter()
            if self.mode == "call":
                # A direct __call__ instead of predict(). predict() runs the
                # whole Trainer machinery -- data adapter, per-batch callback
                # dispatch, output concatenation -- for one small array; calling
                # the model runs the graph. training=False is what puts dropout
                # and batch norm in inference mode, which is what predict() does
                # for us.
                # The model was built as Model(inputs=[t], outputs=[t]), so
                # its input structure is a one-element list; passing a bare
                # array works but warns on every epoch.
                out = self.model([X], training=False)
                Ybar = np.asarray(out[0] if isinstance(out, (list, tuple)) else out)
            else:
                Ybar = self.model.predict(X, verbose=0, **kw)
            self.t_predict += time.perf_counter() - t0

            t0 = time.perf_counter()
            # Keras' compiled 'mae' loss, recomputed: mean over every element of
            # |y - yhat| on the *training* (scaled) scale, PLUS the regularisation
            # penalty, which evaluate() includes and which is a function of the
            # weights alone. Without the second term this is a different early-
            # stopping criterion whenever hyperopt draws reg='l1'/'l2'.
            error = float(np.mean(np.abs(np.asarray(Y, float) - np.asarray(Ybar, float))))
            if self.model.losses:
                error += float(sum(float(l) for l in self.model.losses))
            self.t_numpy += time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            error = self.model.evaluate(X, Y, verbose=0, **kw)
            self.t_evaluate += time.perf_counter() - t0

            t0 = time.perf_counter()
            Ybar = self.model.predict(X, verbose=0, **kw)
            self.t_predict += time.perf_counter() - t0

        t0 = time.perf_counter()
        if self.scaler is not None:
            Yi = Y.reshape(-1, 1) if len(Y.shape) == 1 else Y
            Bi = Ybar.reshape(-1, 1) if len(Y.shape) == 1 else Ybar
            Yi = self.scaler.inverse_transform(Yi)
            Bi = self.scaler.inverse_transform(Bi)
        else:
            Yi, Bi = Y, Ybar

        if self.output_zones is None or len(self.output_zones) < 2:
            mae = np.mean(MAE(Yi, Bi))
        else:
            from dnn_dk1.hyperopt import _per_zone_mae, _scale_normalisers
            if self._mae_normalisers is None:
                self._mae_normalisers = _scale_normalisers(Yi, self.output_zones)
            per_zone = _per_zone_mae(Yi, Bi, self.output_zones)
            mae = float(np.mean([per_zone[z] / self._mae_normalisers[z]
                                 for z in self.output_zones]))
        self.t_numpy += time.perf_counter() - t0

        self.epochs += 1
        return error, mae

    def fit(self, trainX, trainY, valX, valY):
        self.reset_timers()
        bestError = 1e20
        bestMAE = 1e20
        countNoImprovement = 0

        t0 = time.perf_counter()
        bestWeights = self.model.get_weights()
        self.t_bookkeeping += time.perf_counter() - t0

        for epoch in range(MAX_EPOCHS):
            t0 = time.perf_counter()
            self.model.fit(trainX, trainY, batch_size=BATCH_SIZE, epochs=1,
                           verbose=False, shuffle=True)
            self.t_train += time.perf_counter() - t0

            valError, valMAE = self._obtain_metrics(valX, valY)

            t0 = time.perf_counter()
            if valError < bestError:
                countNoImprovement = 0
                bestWeights = self.model.get_weights()
                bestError = valError
                bestMAE = valMAE
                if valMAE < bestMAE:
                    bestMAE = valMAE
            elif valMAE < bestMAE:
                countNoImprovement = 0
                bestWeights = self.model.get_weights()
                bestMAE = valMAE
            else:
                countNoImprovement += 1
            self.t_bookkeeping += time.perf_counter() - t0

            if countNoImprovement >= self.epochs_early_stopping:
                break

        t0 = time.perf_counter()
        self.model.set_weights(bestWeights)
        self.t_bookkeeping += time.perf_counter() - t0
        self.best = (bestError, bestMAE)


# ---------------------------------------------------------------------------
# Data prep -- lifted verbatim from DNN.recalibrate_and_forecast_days
# ---------------------------------------------------------------------------

def prepare(forecaster, df, refit_day, days):
    from epftoolbox.models._dnn import _build_and_split_XYs

    keras.utils.set_random_seed(forecaster._draw_seed(refit_day))
    days = pd.DatetimeIndex(days)

    df_train = df.loc[:refit_day - pd.Timedelta(hours=1)]
    df_train = df_train.loc[
        refit_day - pd.Timedelta(hours=forecaster.calibration_window * 364 * 24):]
    df_test = df.loc[days[0] - pd.Timedelta(weeks=2):
                     days[-1] + pd.Timedelta(hours=23), :]

    Xtrain, Ytrain, Xval, Yval, Xtest, _, index_test = _build_and_split_XYs(
        dfTrain=df_train, features=forecaster.best_hyperparameters,
        shuffle_train=True, dfTest=df_test, date_test=None,
        data_augmentation=forecaster.data_augmentation,
        n_exogenous_inputs=len(df_train.columns) - 1)

    rows = pd.DatetimeIndex(index_test).get_indexer(days)
    Xtest = Xtest[rows]
    Xtrain, Xval, Xtest, Ytrain, Yval = forecaster._regularize_data(
        Xtrain=Xtrain, Xval=Xval, Xtest=Xtest, Ytrain=Ytrain, Yval=Yval)
    return Xtrain, Ytrain, Xval, Yval, Xtest


def build(forecaster, n_features, mode, eval_batch=None):
    hp = forecaster.best_hyperparameters
    neurons = [int(hp['neurons' + str(k)]) for k in range(1, forecaster.nlayers + 1)
               if int(hp['neurons' + str(k)]) >= 50]
    seed = int(hp['seed'] if forecaster.seed is None else forecaster.seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

    m = ProfiledDNNModel(
        neurons=neurons, n_features=n_features, dropout=hp['dropout'],
        batch_normalization=hp['batch_normalization'], lr=hp['lr'], verbose=False,
        optimizer='adam', activation=hp['activation'], epochs_early_stopping=20,
        scaler=forecaster.scaler, loss='mae', regularization=hp['reg'],
        lambda_reg=hp['lambdal1'], initializer=hp['init'])
    m.mode = mode
    m.eval_batch = eval_batch
    return m, neurons


def summarise(tag, m, wall):
    total = m.t_train + m.t_evaluate + m.t_predict + m.t_numpy + m.t_bookkeeping
    return {
        "tag": tag,
        "epochs": m.epochs,
        "wall_seconds": round(wall, 3),
        "train_seconds": round(m.t_train, 3),
        "evaluate_seconds": round(m.t_evaluate, 3),
        "predict_seconds": round(m.t_predict, 3),
        "numpy_seconds": round(m.t_numpy, 3),
        "bookkeeping_seconds": round(m.t_bookkeeping, 3),
        "accounted_seconds": round(total, 3),
        "validation_share_of_wall": round((m.t_evaluate + m.t_predict) / wall, 4),
        "evaluate_share_of_wall": round(m.t_evaluate / wall, 4),
        "ms_per_epoch_train": round(1000 * m.t_train / max(m.epochs, 1), 2),
        "ms_per_epoch_evaluate": round(1000 * m.t_evaluate / max(m.epochs, 1), 2),
        "ms_per_epoch_predict": round(1000 * m.t_predict / max(m.epochs, 1), 2),
        "best": [float(m.best[0]), float(m.best[1])],
    }



def prepare_multizone(f, matrices, refit_day):
    keras.utils.set_random_seed(f._draw_seed(refit_day))
    train_days = f.training_days(matrices, refit_day)
    inc = f.include_calendar
    X = Z.build_X(matrices, train_days, f.zones, inc)
    Y = Z.build_Y(matrices, train_days, f.out_zones)
    Xtrain, Ytrain, Xval, Yval = Z.split_train_val(X, Y, shuffle_train=True,
                                                   hyperoptimization=False)
    Xtest = Z.build_X(matrices, pd.DatetimeIndex([refit_day]), f.zones, inc)

    sx = f.best_hyperparameters["scaleX"]
    if sx in ["Norm", "Norm1", "Std", "Median", "Invariant"]:
        [Xtrain, Xval, Xtest], _ = guarded_scaling([Xtrain, Xval, Xtest], sx)

    sy = f.best_hyperparameters["scaleY"]
    if sy in ["Norm", "Norm1", "Std", "Median", "Invariant"]:
        f.scaler = Z.PerZoneScaler(sy, f.out_zones)
        Ytr = f.scaler.fit_transform(Ytrain)
        Yv = f.scaler.transform(Yval)
    else:
        f.scaler = None
        Ytr, Yv = Ytrain, Yval
    return Xtrain, Ytr, Xval, Yv


def build_multizone(f, n_features, mode, eval_batch=None):
    hp = f.best_hyperparameters
    neurons = [int(hp["neurons" + str(k)]) for k in range(1, f.nlayers + 1)
               if int(hp["neurons" + str(k)]) >= 50]
    seed = int(hp["seed"] if f.seed is None else f.seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    m = ProfiledDNNModel(
        neurons=neurons, n_features=n_features, outputShape=f.output_shape,
        output_zones=f.out_zones, dropout=hp["dropout"],
        batch_normalization=hp["batch_normalization"], lr=hp["lr"], verbose=False,
        optimizer="adam", activation=hp["activation"], epochs_early_stopping=20,
        scaler=f.scaler, loss="mae", regularization=hp["reg"],
        lambda_reg=hp["lambdal1"], initializer=hp["init"])
    m.mode = mode
    m.eval_batch = eval_batch
    return m, neurons



RUNS = [
    ("A_baseline_evaluate+predict", "baseline", None),
    ("B_predict_only_numpy_loss", "predict", None),
    ("C_baseline_valbatch=all", "baseline", "all"),
    ("D_predict_only_valbatch=all", "predict", "all"),
    ("E_model_call_numpy_loss", "call", None),
    ("A2_baseline_repeat", "baseline", None),
]

MULTIZONE_DATASET = {"wide": "dnnwide_{zone}", "joint": "dnnjoint"}


def load_own(zone):
    dataset = Z.dataset_name(zone)
    forecaster = DNN(path_hyperparameter_folder=HYPER_DIR, experiment_id=1,
                     nlayers=2, dataset=dataset, calibration_window=4, seed=SEED)
    with redirect_stdout(io.StringIO()):
        df_train, df_test = read_data(path=DATA_DIR, dataset=dataset,
                                      begin_test_date=BEGIN_TEST,
                                      end_test_date=END_TEST + pd.Timedelta(hours=23))
    data = pd.concat([df_train, df_test])
    days = pd.date_range(REFIT_DAY, periods=7, freq="D")
    Xtrain, Ytrain, Xval, Yval, _ = prepare(forecaster, data, REFIT_DAY, days)
    return forecaster, (Xtrain, Ytrain, Xval, Yval), build


def load_multizone(config, zone):
    from dnn_dk1 import MultiZoneDNN

    out_zones = (zone,) if config == "wide" else Z.ZONES
    matrices = Z.load_zone_matrices(Z.ZONES, DATA_DIR)
    forecaster = MultiZoneDNN(
        path_hyperparameter_folder=HYPER_DIR, experiment_id=1, nlayers=2,
        dataset=MULTIZONE_DATASET[config].format(zone=zone),
        calibration_window=4, seed=SEED, zones=Z.ZONES, out_zones=out_zones)
    data = prepare_multizone(forecaster, matrices, REFIT_DAY)
    return forecaster, data, build_multizone


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="own", choices=["own", "wide", "joint"])
    p.add_argument("--zone", default="DK1")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    label = args.config if args.config == "joint" else f"{args.config}_{args.zone}"
    print(f"threads: intra={os.environ['TF_NUM_INTRAOP_THREADS']} "
          f"inter={os.environ['TF_NUM_INTEROP_THREADS']}  config={label}")

    if args.config == "own":
        forecaster, (Xtrain, Ytrain, Xval, Yval), builder = load_own(args.zone)
    else:
        forecaster, (Xtrain, Ytrain, Xval, Yval), builder = load_multizone(
            args.config, args.zone)

    hp = forecaster.best_hyperparameters
    print("hyperparameters:",
          {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
           for k, v in hp.items() if not str(k).startswith("In:")})
    print(f"Xtrain {Xtrain.shape}  Ytrain {Ytrain.shape}  Xval {Xval.shape}")
    print(f"train batches/epoch @{BATCH_SIZE}: {int(np.ceil(len(Xtrain)/BATCH_SIZE))}; "
          f"validation batches/epoch @32 (keras default): "
          f"{int(np.ceil(len(Xval)/32))} per pass, two passes\n", flush=True)

    results = []
    for tag, mode, eb in RUNS:
        m, neurons = builder(forecaster, Xtrain.shape[-1], mode,
                             len(Xval) if eb == "all" else None)
        t0 = time.perf_counter()
        m.fit(Xtrain, Ytrain, Xval, Yval)
        row = summarise(tag, m, time.perf_counter() - t0)
        row["neurons"] = neurons
        row["config"] = label
        results.append(row)
        print(json.dumps(row), flush=True)
        m.clear_session()

    base = results[0]["wall_seconds"]
    print("\nseconds per fit, and the saving against A:")
    for r in results:
        print(f"  {r['tag']:<30s} {r['wall_seconds']:8.2f}s  "
              f"{100 * (1 - r['wall_seconds'] / base):+6.1f}%   "
              f"epochs {r['epochs']}  best {r['best'][0]:.9g}")
    print("\nEvery row must show the same epoch count and the same best value as A. "
          "A variant that differs is stopping somewhere else, not computing faster.")

    out = args.out or os.path.join(HERE, f"profile_obtain_metrics_{label}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
