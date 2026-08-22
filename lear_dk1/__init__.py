"""
Running the epftoolbox LEAR model on ENTSO-E bidding-zone data.

The vendored ``epftoolbox`` tree is used as-is and never modified; the
NumPy 2 / Keras 3 incompatibilities that would otherwise stop LEAR from running
are worked around in :mod:`lear_dk1.compat`.
"""

from .compat import LEARCompat, minimum_calibration_window, n_features

__all__ = [
    "LEARCompat",
    "minimum_calibration_window",
    "n_features",
]
