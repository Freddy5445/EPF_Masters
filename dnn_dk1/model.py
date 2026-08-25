"""
The epftoolbox DNN, rebuilt against the Keras 3 API.

``epftoolbox/models/_dnn.py`` was written for Keras 2 and makes two calls that
Keras 3 rejects outright:

1. ``kr.optimizers.Adam(lr=...)`` -- renamed to ``learning_rate``. Keras 3 raises
   ``ValueError: Argument(s) not recognized: {'lr': ...}`` rather than warning.
2. ``Dense(..., batch_input_shape=...)`` -- removed. It was redundant anyway: the
   ``Input`` layer above it already fixes the shape, and Keras 2 ignored it on a
   layer that was not first in the model.

Everything else in the upstream file imports and runs on Keras 3.15 unchanged --
``AlphaDropout``, ``LeakyReLU(alpha=)`` and ``Input(batch_shape=)`` all still
work. This module therefore reproduces the same architecture and the same
training procedure rather than inventing anything: same layer stack, same
custom early-stopping loop, same batch size, same epoch cap, same metrics.

The vendored ``epftoolbox/`` tree is not modified. Where a piece of upstream is
pure numpy/pandas -- the feature construction and the scalers -- it is imported
and reused, because a hand-rewritten copy of that is exactly how a model quietly
stops being the model the paper describes.
"""

import os
import sys
import time

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

if os.path.join(PROJECT_ROOT, "epftoolbox") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "epftoolbox"))

from epftoolbox.evaluation import MAE  # noqa: E402

import keras  # noqa: E402
from keras.layers import (  # noqa: E402
    AlphaDropout, BatchNormalization, Dense, Dropout, Input, LeakyReLU, PReLU,
)
from keras.models import Model  # noqa: E402
from keras.regularizers import l1, l2  # noqa: E402

# Upstream's training constants, kept explicit so they are visible rather than
# buried in the loop.
BATCH_SIZE = 192
MAX_EPOCHS = 1000
GRADIENT_CLIP = 10000


class DNNModel:
    """One deep neural network, trained with early stopping on a validation set.

    Mirrors ``epftoolbox.models.DNNModel``. The constructor signature is the same,
    including ``lr`` -- kept under upstream's name because the hyperparameter
    dictionaries hyperopt produces use that key.
    """

    def __init__(self, neurons, n_features, outputShape=24, dropout=0,
                 batch_normalization=False, lr=None, verbose=False,
                 epochs_early_stopping=40, scaler=None, loss='mae',
                 optimizer='adam', activation='relu', initializer='glorot_uniform',
                 regularization=None, lambda_reg=0):
        self.neurons = neurons
        self.dropout = dropout

        if self.dropout > 1 or self.dropout < 0:
            raise ValueError('Dropout parameter must be between 0 and 1')

        self.batch_normalization = batch_normalization
        self.verbose = verbose
        self.epochs_early_stopping = epochs_early_stopping
        self.n_features = n_features
        self.scaler = scaler
        self.outputShape = outputShape
        self.activation = activation
        self.initializer = initializer
        self.regularization = regularization
        self.lambda_reg = lambda_reg

        self.model = self._build_model()

        if lr is None:
            opt = 'adam'
        else:
            # Keras 3 renamed `lr` to `learning_rate` and raises on the old name.
            # The upstream branches are if/if/if, not elif, so an unrecognised
            # optimizer name leaves `opt` undefined and raises UnboundLocalError;
            # a dict lookup says what was actually wrong instead.
            optimizers = {
                'adam': keras.optimizers.Adam,
                'RMSprop': keras.optimizers.RMSprop,
                'adagrad': keras.optimizers.Adagrad,
                'adadelta': keras.optimizers.Adadelta,
            }
            if optimizer not in optimizers:
                raise ValueError(
                    f"Unknown optimizer {optimizer!r}; expected one of "
                    f"{', '.join(sorted(optimizers))}")
            opt = optimizers[optimizer](learning_rate=lr, clipvalue=GRADIENT_CLIP)

        self.model.compile(loss=loss, optimizer=opt)

    def _reg(self, lambda_reg):
        """An l1 or l2 regularizer, or None. Same semantics as upstream.

        ``lambda_reg`` comes from hyperopt's own sampling as a numpy float64.
        Some Keras 3 point releases store that on the regularizer as-is and
        later multiply it against a layer's float32 weights with no cast,
        which TF's strict op typing rejects (`Mul` on float64 x float32). A
        plain Python float is weakly typed and combines with any tensor
        dtype, so casting here is version-proof rather than relying on a
        particular Keras release's own guard.
        """
        if self.regularization == 'l2':
            return l2(float(lambda_reg))
        if self.regularization == 'l1':
            return l1(float(lambda_reg))
        return None

    def _build_model(self):
        """The layer stack: `neurons` hidden layers, then a 24-wide linear output."""
        past_data = Input(batch_shape=(None, self.n_features))

        past_Dense = past_data
        if self.activation == 'selu':
            # Self-normalising networks need this pairing to work at all.
            self.initializer = 'lecun_normal'

        for neurons in self.neurons:
            # `batch_input_shape` is dropped here: Keras 3 rejects it on Dense,
            # and the Input layer above already fixes the shape.
            if self.activation == 'LeakyReLU':
                past_Dense = Dense(neurons, activation='linear',
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._reg(self.lambda_reg))(past_Dense)
                past_Dense = LeakyReLU(negative_slope=.001)(past_Dense)

            elif self.activation == 'PReLU':
                past_Dense = Dense(neurons, activation='linear',
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._reg(self.lambda_reg))(past_Dense)
                past_Dense = PReLU()(past_Dense)

            else:
                past_Dense = Dense(neurons, activation=self.activation,
                                   kernel_initializer=self.initializer,
                                   kernel_regularizer=self._reg(self.lambda_reg))(past_Dense)

            if self.batch_normalization:
                past_Dense = BatchNormalization()(past_Dense)

            if self.dropout > 0:
                if self.activation == 'selu':
                    past_Dense = AlphaDropout(self.dropout)(past_Dense)
                else:
                    past_Dense = Dropout(self.dropout)(past_Dense)

        output_layer = Dense(self.outputShape, kernel_initializer=self.initializer,
                             kernel_regularizer=self._reg(self.lambda_reg))(past_Dense)

        return Model(inputs=[past_data], outputs=[output_layer])

    def _obtain_metrics(self, X, Y):
        """Validation loss and MAE, the latter in the original price unit."""
        error = self.model.evaluate(X, Y, verbose=0)
        Ybar = self.model.predict(X, verbose=0)

        if self.scaler is not None:
            if len(Y.shape) == 1:
                Y = Y.reshape(-1, 1)
                Ybar = Ybar.reshape(-1, 1)
            Y = self.scaler.inverse_transform(Y)
            Ybar = self.scaler.inverse_transform(Ybar)

        return error, np.mean(MAE(Y, Ybar))

    def _display_info_training(self, bestError, bestMAE, countNoImprovement):
        print(" Best error:\t\t{:.1e}".format(bestError))
        print(" Best MAE:\t\t{:.2f}".format(bestMAE))
        print(" Epochs without improvement:\t{}\n".format(countNoImprovement))

    def fit(self, trainX, trainY, valX, valY):
        """Train with early stopping on the validation set.

        The stopping rule is upstream's, reproduced as written: the weights are
        kept whenever *either* the validation loss or the validation MAE
        improves, and training stops after ``epochs_early_stopping`` epochs in
        which neither did. Note that the weights restored at the end are the ones
        from the last epoch that improved *either* metric, which need not be the
        best-loss epoch. That is upstream's behaviour and is left alone, since
        changing it would make these forecasts something other than the paper's.
        """
        bestError = 1e20
        bestMAE = 1e20
        countNoImprovement = 0
        bestWeights = self.model.get_weights()

        for epoch in range(MAX_EPOCHS):
            start_time = time.time()

            self.model.fit(trainX, trainY, batch_size=BATCH_SIZE, epochs=1,
                           verbose=False, shuffle=True)

            if self.verbose:
                print("\nEpoch {} of {} took {:.3f}s".format(
                    epoch + 1, MAX_EPOCHS, time.time() - start_time))

            valError, valMAE = self._obtain_metrics(valX, valY)

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

            if countNoImprovement >= self.epochs_early_stopping:
                if self.verbose:
                    self._display_info_training(bestError, bestMAE, countNoImprovement)
                break

            if self.verbose:
                self._display_info_training(bestError, bestMAE, countNoImprovement)

        self.model.set_weights(bestWeights)

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def clear_session(self):
        """Release the Keras graph.

        Called after every daily recalibration: without it, retraining in a loop
        leaks memory steadily until the run dies.
        """
        keras.backend.clear_session()
