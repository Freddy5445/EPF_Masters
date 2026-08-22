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


class LEARCompat(_LEAR):
    """LEAR with the NumPy 2 scalar-assignment break fixed.

    Only :meth:`predict` differs from upstream. Recalibration, the asinh-median
    ("Invariant") scaling and the LASSO/LARS hyperparameter search are inherited
    untouched, so forecasts match what upstream would produce on NumPy 1.x.
    """

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


def minimum_calibration_window(n_exogenous):
    """Smallest calibration window (days) that ``LassoLarsIC`` will accept.

    ``LassoLarsIC`` refuses to fit when there are fewer samples than features,
    because it cannot estimate the noise variance. One training sample is one
    day, and the first week is consumed building lagged features, so the window
    must exceed ``n_features + 7``. A little headroom is added because the
    estimator needs strictly more samples than features, not merely equal.

    Note this rules out the 56- and 84-day windows used by the original LEAR
    ensemble: with even one exogenous input the model has 175 features.
    """
    return n_features(n_exogenous) + 8
