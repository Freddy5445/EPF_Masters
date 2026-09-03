"""
``epftoolbox.data.scaling``, with the zero-MAD divide guarded.

The ``Median`` and ``Invariant`` scalers divide by the median absolute
deviation. A feature that never varies has ``mad == 0`` and
``data - median == 0``, so upstream computes ``0 / 0`` and fills the whole
column with NaN. Keras then propagates the NaN into the loss, and every
hyperopt trial that happens to sample ``scaleX='Median'`` or
``scaleX='Invariant'`` scores ``nan`` for a reason that has nothing to do with
its hyperparameters.

This is not hypothetical here. The DNN builds one input per (series, lag, hour),
so a **solar** forecast -- exactly zero at night, every night -- makes several
columns genuinely constant:

===============================  ==================  =====================
Configuration                    Input columns       Zero-MAD columns
===============================  ==================  =====================
DNN-own, DK1                     313                 27
DNN-wide / DNN-joint             1969                117
===============================  ==================  =====================

Upstream never hits this because none of the five markets it ships separates
solar from wind -- the ``load-wind-solar`` layout this thesis uses does, which is
also why its LEAR design matrix is 319 features rather than 247.

``lear_dk1.compat._fit_invariant_scaler`` already fixes exactly this for LEAR,
and the reasoning there applies unchanged: substituting 1 for a zero MAD maps a
constant column to all-zeros, which is the sensible reading -- a feature with no
variation carries no information -- and the inverse transform still recovers the
constant, since ``0 * 1 + median == median``. The two models are guarded the same
way so that "LEAR and the DNN read identically prepared inputs" stays true
through the scaler as well as through the cleaning.

The vendored ``epftoolbox/`` tree is not modified; this is a wrapper, in the same
spirit as :mod:`lear_dk1.compat`.
"""

import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

if os.path.join(PROJECT_ROOT, "epftoolbox") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "epftoolbox"))

# The scalers that divide by a MAD. sklearn's MinMaxScaler and StandardScaler,
# behind 'Norm', 'Norm1' and 'Std', already substitute 1 for a zero range or a
# zero variance themselves, so only these two need the guard.
MAD_SCALERS = ("Median", "Invariant")
SCALED = ("Norm", "Norm1", "Std", "Median", "Invariant")


def fit_scaler(normalize, reference):
    """A ``DataScaler`` fitted on ``reference``, with any zero MAD set to 1.

    Returns ``(scaler, n_constant)``.
    """
    from epftoolbox.data import DataScaler

    scaler = DataScaler(normalize)
    scaler.scaler.fit(np.asarray(reference, dtype=float))

    n_constant = 0
    if normalize in MAD_SCALERS:
        constant = scaler.scaler.mad == 0
        n_constant = int(constant.sum())
        if n_constant:
            scaler.scaler.mad = np.where(constant, 1.0, scaler.scaler.mad)
    return scaler, n_constant


def guarded_scaling(datasets, normalize):
    """Drop-in for ``epftoolbox.data.scaling`` with the zero-MAD guard.

    Same contract: the first dataset in the list is the reference the scaler is
    estimated from, every dataset is transformed with it, and the scaler comes
    back so the caller can invert later.
    """
    scaler, _ = fit_scaler(normalize, datasets[0])
    return [scaler.transform(np.asarray(d, dtype=float)) for d in datasets], scaler
