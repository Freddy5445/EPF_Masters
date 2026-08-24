"""
Compatibility shim for running ``epftoolbox``'s LEAR on a modern stack.

The vendored ``epftoolbox`` predates NumPy 2 and Keras 3, and breaks against the
versions this project pins (numpy 2.4, keras 3.15). Everything here works around
that from the outside; the ``epftoolbox/`` tree itself is never modified.

Three problems are handled:

1. **``LEAR.predict`` crashes on NumPy 2.** ``_lear.py`` does
   ``Yp[h] = self.models[h].predict(X)``, assigning a size-1 array into a scalar
   slot. NumPy 1.25 deprecated that and NumPy 2.0 made it an error
   (``ValueError: setting an array element with a sequence``). :class:`LEARCompat`
   overrides ``predict`` to call ``.item()`` explicitly. The arithmetic is
   otherwise identical to upstream.

2. **Importing LEAR drags in TensorFlow.** ``epftoolbox/models/__init__.py``
   imports the DNN modules, which do ``import tensorflow.keras as kr`` -- a
   layout Keras 3 no longer provides, and an import that costs seconds and
   hundreds of MB even when it works. :func:`load_lear_class` loads ``_lear.py``
   directly from its file path, so the package ``__init__`` never runs.
   ``_lear.py``'s own ``from epftoolbox.data import ...`` imports still resolve
   normally, because those subpackages are TensorFlow-free.

3. **``evaluate_lear_in_test_dataset`` uses ``np.NaN``**, removed in NumPy 2.0.
   That function is unusable; :mod:`lear_dk1.backtest` replaces it rather than
   patching it.
"""

import importlib.util
import os
import sys

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoLarsIC
from sklearn.utils._testing import ignore_warnings

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

# The package root is the outer folder: <root>/epftoolbox/epftoolbox/...
EPFTOOLBOX_ROOT = os.path.join(PROJECT_ROOT, "epftoolbox")
LEAR_SOURCE = os.path.join(EPFTOOLBOX_ROOT, "epftoolbox", "models", "_lear.py")

_MODULE_NAME = "_epftoolbox_lear_direct"


def load_lear_class():
    """Import the upstream ``LEAR`` class without executing ``models/__init__``.

    Loading the module under a private name keeps it out of the ``epftoolbox``
    package namespace, so importing it never triggers the DNN/TensorFlow imports
    that ``epftoolbox.models`` performs at package level.
    """
    if EPFTOOLBOX_ROOT not in sys.path:
        sys.path.insert(0, EPFTOOLBOX_ROOT)

    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME].LEAR

    if not os.path.isfile(LEAR_SOURCE):
        raise FileNotFoundError(
            f"Cannot find the vendored LEAR implementation at {LEAR_SOURCE}. "
            f"Is the epftoolbox/ source tree present?"
        )

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, LEAR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)

    return module.LEAR


_LEAR = load_lear_class()


def _fit_invariant_scaler(data):
    """Fit the asinh-median scaler, guarding against a zero MAD.

    The scaler divides by the median absolute deviation. A feature that never
    varies has ``mad == 0`` and ``data - median == 0``, so upstream computes
    ``0 / 0`` and fills the whole column with NaN, which ``LassoLarsIC`` then
    rejects with a message about imputers that has nothing to do with the cause.

    This is not a hypothetical: LEAR builds one feature per (hour, lag), so a
    solar generation forecast -- exactly zero at night, every night -- produces
    several genuinely constant columns in any real dataset that includes solar.

    Substituting 1 for a zero MAD maps such a column to all-zeros, which is the
    sensible reading: a feature with no variation carries no information, and
    LASSO gives it a zero coefficient. The inverse transform still recovers the
    constant, since ``0 * 1 + median == median``.

    Returns ``(scaler, transformed, n_constant)``.
    """
    from epftoolbox.data import DataScaler

    scaler = DataScaler("Invariant")
    scaler.scaler.fit(data)

    constant = scaler.scaler.mad == 0
    n_constant = int(constant.sum())
    if n_constant:
        scaler.scaler.mad = np.where(constant, 1.0, scaler.scaler.mad)

    return scaler, scaler.transform(data), n_constant


class LEARCompat(_LEAR):
    """LEAR with the NumPy 2 and zero-MAD breaks fixed.

    Two methods differ from upstream; the LASSO/LARS estimation itself, the
    feature construction and the recalibration protocol are unchanged, so
    forecasts match what upstream would produce on a stack where it ran.

    ``constant_features_`` records how many input features had no variation in
    the most recent calibration window.
    """

    constant_features_ = 0

    @ignore_warnings(category=ConvergenceWarning)
    def recalibrate(self, Xtrain, Ytrain):
        # Same as upstream, except the two scalers guard against a zero MAD.
        self.scalerY, Ytrain, _ = _fit_invariant_scaler(Ytrain)
        self.scalerX, Xtrain_no_dummies, self.constant_features_ = \
            _fit_invariant_scaler(Xtrain[:, :-7])
        Xtrain[:, :-7] = Xtrain_no_dummies

        self.models = {}
        for h in range(24):
            # Estimate lambda with LARS under an AIC criterion, then refit with
            # coordinate-descent LASSO at that lambda, as upstream does.
            #
            # noise_variance is supplied rather than left to scikit-learn. Its
            # default estimator is an OLS residual variance, which needs more
            # samples than features and so refuses outright on the paper's 8-
            # and 12-week windows (49 and 77 samples against 247 regressors).
            # Passing the variance of the target restores the behaviour LEAR
            # was written against and makes those windows fitable.
            #
            # This is not confined to the short windows. AIC here is
            # ``n log(2 pi s2) + RSS/s2 + 2 df``: the first term is flat along
            # the path, but s2 sets how RSS trades against sparsity, so it moves
            # the argmin at every window length. Measured on synthetic data,
            # alpha roughly doubles at n=1092. Results are therefore not
            # comparable with runs made before this change.
            param = LassoLarsIC(
                criterion="aic", max_iter=2500,
                noise_variance=float(np.var(Ytrain[:, h])),
            ).fit(Xtrain, Ytrain[:, h]).alpha_
            model = Lasso(max_iter=2500, alpha=param)
            model.fit(Xtrain, Ytrain[:, h])
            self.models[h] = model

    def predict(self, X):
        Yp = np.zeros(24)

        # Rescale everything except the 7 weekday dummies, as upstream does.
        X[:, :-7] = self.scalerX.transform(X[:, :-7])

        for h in range(24):
            # Upstream assigns the raw (1,) array here, which NumPy 2 rejects.
            Yp[h] = self.models[h].predict(X).item()

        return self.scalerY.inverse_transform(Yp.reshape(1, -1))


def n_features(n_exogenous):
    """Feature count LEAR builds for a dataset with ``n_exogenous`` inputs.

    96 price lags (24 hours x days D-1, D-2, D-3, D-7)
    + 72 per exogenous input (24 hours x days D, D-1, D-7)
    + 7 weekday dummies.
    """
    return 96 + 7 + 72 * n_exogenous


# Lago et al. (2021) take 8 weeks as the shortest window LEAR is run on. Below
# that there is too little data to identify the model however lambda is chosen,
# so it is a modelling floor rather than a numerical one.
MINIMUM_WINDOW_DAYS = 56


def minimum_calibration_window(n_exogenous=None):
    """Shortest calibration window (days) LEAR is run on.

    This used to be ``n_features + 8``, because scikit-learn's ``LassoLarsIC``
    refuses to fit when samples are fewer than features -- which ruled out the
    8- and 12-week windows of the published ensemble. Supplying the noise
    variance explicitly (see :meth:`LEARCompat.recalibrate`) removes that
    restriction, so the binding constraint is now the paper's own floor.

    ``n_exogenous`` is accepted and ignored, so existing callers keep working.
    """
    return MINIMUM_WINDOW_DAYS
