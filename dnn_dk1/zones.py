"""
The zone set Z, and the multi-zone feature assembly built from it.

``Z`` is DK1 and DK2 plus their direct interconnections. It is a fixed list, not
a tunable one, and every dimension in this module is *derived* from it rather
than written down twice -- so the assertions in ``run_dnn_dk1.preflight`` check a
built matrix against an arithmetic identity, not against a magic number copied
from a spec.

Layout, per zone, is Lago et al. (2021)'s: the price at days D-1, D-2, D-3 and
D-7, and each exogenous series at D, D-1 and D-7, all 24 hours of each. A zone
with three exogenous series therefore contributes ``96 + 3*72 = 312`` columns and
one with two contributes ``96 + 2*72 = 240``. The day-of-week calendar feature is
shared, so it is counted once for the whole matrix.

Three of the seven zones have no usable solar series -- see :data:`ZONE_EXOG`.

Column names are deterministic and reproducible from :data:`ZONES` alone:
``<ZONE>_<block>_<lag>_h<hour>``, ordered zone-major, then block, then lag, then
hour, with the calendar column first. Nothing about the order depends on the
order files happen to be read in.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
DEFAULT_DATASETS = os.path.join(PROJECT_ROOT, "datasets")

# Z = the union of DK1's and DK2's direct interconnections plus the two focal
# zones. The order is canonical and fixed: it fixes the column order of the
# assembled matrix and the block order of the 168-wide output head.
ZONES: tuple[str, ...] = ("DK1", "DK2", "DE_LU", "NL", "NO2", "SE3", "SE4")

FOCAL_ZONES: tuple[str, ...] = ("DK1", "DK2")

# Exogenous series per zone, in the order they sit in that zone's cleaned CSV.
#
# NO2, SE3 and SE4 have two, not three. This is a property of the data, not a
# choice:
#
#   * NO2's day-ahead solar forecast is published but is identically zero over
#     the whole span, so the cleaning notebook drops it as constant;
#   * SE3's and SE4's solar forecasts begin 2021-12-01, three years into the
#     2019-01-01 panel. Starting the panel there would leave no room for the
#     1463-day burn-in before the 2023-10-01 test period.
#
# Nothing here can recover a series that does not exist, so these zones carry
# load and wind. See ``docs/dnn_phase1_spec.md`` -- its table lists SE3 and SE4
# as three-exogenous zones, which the data does not support.
ZONE_EXOG: dict[str, tuple[str, ...]] = {
    "DK1": ("load", "wind", "solar"),
    "DK2": ("load", "wind", "solar"),
    "DE_LU": ("load", "wind", "solar"),
    "NL": ("load", "wind", "solar"),
    "NO2": ("load", "wind"),
    "SE3": ("load", "wind"),
    "SE4": ("load", "wind"),
}

# The ``run_lear_from_clean`` layout that produced each zone's cleaned CSV. Both
# layouts keep price first and load second, so the exogenous order above is the
# CSV's own column order.
EXOG_LAYOUTS = {3: "load-wind-solar", 2: "load-wind"}

PRICE_LAGS: tuple[int, ...] = (1, 2, 3, 7)   # days D-1, D-2, D-3, D-7
EXOG_LAGS: tuple[int, ...] = (0, 1, 7)       # days D, D-1, D-7
MAX_LAG_DAYS = max(max(PRICE_LAGS), max(EXOG_LAGS))

HOURS = tuple(range(24))
CALENDAR_COLUMN = "calendar_dayofweek"

# The reported test period and burn-in, identical to the LEAR thesis run.
BEGIN_TEST = pd.Timestamp("2023-10-01")
END_TEST = pd.Timestamp("2025-09-30")
TEST_DAYS_EXPECTED = 731
CALIBRATION_YEARS = 4
CALIBRATION_DAYS = CALIBRATION_YEARS * 364          # the DNN's window, in days
BURN_IN_DAYS = CALIBRATION_DAYS + MAX_LAG_DAYS      # 1463
HISTORY_START_REQUIRED = BEGIN_TEST - pd.Timedelta(days=BURN_IN_DAYS)

# The panel the cleaned CSVs are projected from.
PANEL_START = pd.Timestamp("2019-01-01")
PANEL_END = END_TEST

# EU DST transitions inside 2019-01-01..2025-09-30: the last Sunday of March
# 2019-2025 (7) and the last Sunday of October 2019-2024 (6); 2025-10-26 falls
# after the panel ends.
DST_SPRING_EXPECTED = 7
DST_AUTUMN_EXPECTED = 6


class ZoneDataError(ValueError):
    """Raised when a zone's cleaned CSV is missing, short, or holds NaN."""


# ---------------------------------------------------------------------------
# Dimensions -- derived from ZONES and ZONE_EXOG, never written down twice
# ---------------------------------------------------------------------------

def n_exogenous(zone: str) -> int:
    return len(ZONE_EXOG[zone])


def zone_block_width(zone: str) -> int:
    """Columns one zone contributes: 24*4 price lags + 24*3 per exogenous."""
    return 24 * len(PRICE_LAGS) + 24 * len(EXOG_LAGS) * n_exogenous(zone)


def own_input_width(zone: str) -> int:
    """DNN-own input width for ``zone``: its own block plus the calendar."""
    return zone_block_width(zone) + 1


def input_width(zones: tuple[str, ...] = ZONES) -> int:
    """DNN-wide / DNN-joint input width: every zone's block plus one calendar."""
    return sum(zone_block_width(z) for z in zones) + 1


def output_width(zones: tuple[str, ...]) -> int:
    return 24 * len(zones)


def dataset_name(zone: str) -> str:
    """The cleaned per-zone CSV's dataset name, as ``run_lear_from_clean`` names it."""
    return f"{zone}_clean_{EXOG_LAYOUTS[n_exogenous(zone)]}"


def dataset_path(zone: str, datasets_dir: str = DEFAULT_DATASETS) -> str:
    return os.path.join(datasets_dir, f"{dataset_name(zone)}.csv")


def zone_slice(zone: str, zones: tuple[str, ...] = ZONES) -> slice:
    """The 24 output columns belonging to ``zone`` in a zone-major target vector."""
    position = zones.index(zone)
    return slice(position * 24, (position + 1) * 24)


# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------

def _lag_label(lag: int) -> str:
    return "D" if lag == 0 else f"D-{lag}"


def zone_feature_names(zone: str) -> list[str]:
    """One zone's block, in assembly order: price lags, then each exogenous."""
    names = [f"{zone}_price_{_lag_label(lag)}_h{hour}"
             for lag in PRICE_LAGS for hour in HOURS]
    for block in ZONE_EXOG[zone]:
        names += [f"{zone}_{block}_{_lag_label(lag)}_h{hour}"
                  for lag in EXOG_LAGS for hour in HOURS]
    return names


def feature_names(zones: tuple[str, ...] = ZONES,
                  include_calendar: bool = True) -> list[str]:
    """Every input column name, in the matrix's own order."""
    names = [CALENDAR_COLUMN] if include_calendar else []
    for zone in zones:
        names += zone_feature_names(zone)
    return names


def target_names(zones: tuple[str, ...]) -> list[str]:
    """Output column names: zone-major, then hour."""
    return [f"{zone}_price_D_h{hour}" for zone in zones for hour in HOURS]


def feature_zone_labels(zones: tuple[str, ...] = ZONES,
                        include_calendar: bool = True) -> np.ndarray:
    """Which zone each input column belongs to (``"calendar"`` for the toggle).

    Used to group first-layer weights by zone after training.
    """
    labels = ["calendar"] if include_calendar else []
    for zone in zones:
        labels += [zone] * zone_block_width(zone)
    return np.array(labels)


def feature_block_labels(zones: tuple[str, ...] = ZONES,
                         include_calendar: bool = True) -> np.ndarray:
    """Which ``<zone>/<block>`` each input column belongs to."""
    labels = ["calendar"] if include_calendar else []
    for zone in zones:
        labels += [f"{zone}/price"] * (24 * len(PRICE_LAGS))
        for block in ZONE_EXOG[zone]:
            labels += [f"{zone}/{block}"] * (24 * len(EXOG_LAGS))
    return np.array(labels)


# ---------------------------------------------------------------------------
# Loading -- one day x hour matrix per (zone, series)
# ---------------------------------------------------------------------------

def load_zone_matrices(zones: tuple[str, ...] = ZONES,
                       datasets_dir: str = DEFAULT_DATASETS,
                       data_start=None, data_end=None) -> dict:
    """Read the cleaned per-zone CSVs into day x hour matrices.

    Returns ``{zone: {series: DataFrame(index=local dates, columns=0..23)}}``
    where ``series`` is ``"price"`` plus that zone's entries in
    :data:`ZONE_EXOG`.

    Cleaning happens once, in ``data_cleaning_v2.ipynb``. This function refuses
    to see a NaN rather than fill one: an imputation rule living in a run script
    is how two models stop reading identically prepared inputs.
    """
    matrices: dict[str, dict[str, pd.DataFrame]] = {}
    common_dates = None

    for zone in zones:
        path = dataset_path(zone, datasets_dir)
        if not os.path.exists(path):
            raise ZoneDataError(
                f"no cleaned CSV for {zone} at {path}. Build it with "
                f"`python run_lear_from_clean.py --zone {zone} --exog "
                f"{EXOG_LAYOUTS[n_exogenous(zone)]} --csv-only`."
            )
        frame = pd.read_csv(path, index_col=0)
        frame.index = pd.to_datetime(frame.index)

        expected = 1 + n_exogenous(zone)
        if frame.shape[1] != expected:
            raise ZoneDataError(
                f"{os.path.basename(path)} has {frame.shape[1]} columns; "
                f"{zone} is declared as price + {n_exogenous(zone)} exogenous "
                f"({', '.join(ZONE_EXOG[zone])}), i.e. {expected}."
            )
        frame.columns = ["price", *ZONE_EXOG[zone]]

        if data_start is not None:
            frame = frame.loc[pd.Timestamp(data_start):]
        if data_end is not None:
            frame = frame.loc[:pd.Timestamp(data_end) + pd.Timedelta(hours=23)]

        if frame.isna().any().any():
            counts = frame.isna().sum()
            raise ZoneDataError(
                f"{zone} has missing values {counts[counts > 0].to_dict()}. The "
                f"DNN cannot be fitted on NaN and nothing here imputes. Re-run "
                f"data_cleaning_v2.ipynb, then rebuild the CSV with "
                f"run_lear_from_clean.py."
            )

        dates = frame.index.normalize()
        hours = frame.index.hour
        per_day = pd.Series(1, index=dates).groupby(level=0).sum()
        if not per_day.eq(24).all():
            bad = per_day[per_day != 24]
            raise ZoneDataError(
                f"{zone} is not on the 24-hour local delivery grid: "
                f"{bad.head(3).to_dict()}"
            )

        zone_matrices = {}
        for series in ("price", *ZONE_EXOG[zone]):
            matrix = pd.DataFrame(
                frame[series].to_numpy(float).reshape(-1, 24),
                index=pd.DatetimeIndex(dates[::24]), columns=list(HOURS))
            zone_matrices[series] = matrix
        matrices[zone] = zone_matrices

        if common_dates is None:
            common_dates = zone_matrices["price"].index
        elif not common_dates.equals(zone_matrices["price"].index):
            raise ZoneDataError(
                f"{zone} spans {zone_matrices['price'].index.min().date()}.."
                f"{zone_matrices['price'].index.max().date()}, which differs "
                f"from the other zones' {common_dates.min().date()}.."
                f"{common_dates.max().date()}. Every zone must cover the same "
                f"local days or the assembled matrix is not one panel."
            )
        _ = hours  # the 24-per-day check above is what the hour column is for

    return matrices


def available_days(matrices: dict) -> pd.DatetimeIndex:
    """The local dates every zone covers."""
    return next(iter(matrices.values()))["price"].index


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _lagged(matrix: pd.DataFrame, days: pd.DatetimeIndex, lag: int) -> np.ndarray:
    """``matrix`` read at ``days - lag`` days, as an (n_days, 24) array."""
    return matrix.reindex(days - pd.Timedelta(days=lag)).to_numpy(float)


def build_X(matrices: dict, days: pd.DatetimeIndex,
            zones: tuple[str, ...] = ZONES,
            include_calendar: bool = True) -> np.ndarray:
    """The input matrix for ``days``: (n_days, :func:`input_width`).

    Column order is exactly :func:`feature_names`.
    """
    days = pd.DatetimeIndex(days)
    blocks = []
    if include_calendar:
        blocks.append(days.dayofweek.to_numpy(float).reshape(-1, 1))
    for zone in zones:
        zone_matrices = matrices[zone]
        for lag in PRICE_LAGS:
            blocks.append(_lagged(zone_matrices["price"], days, lag))
        for block in ZONE_EXOG[zone]:
            for lag in EXOG_LAGS:
                blocks.append(_lagged(zone_matrices[block], days, lag))
    X = np.concatenate(blocks, axis=1)

    expected = input_width(zones) - (0 if include_calendar else 1)
    if X.shape[1] != expected:
        raise ZoneDataError(
            f"assembled {X.shape[1]} input columns, expected {expected}")
    _reject_nan(X, days, feature_names(zones, include_calendar), "input")
    return X


def build_Y(matrices: dict, days: pd.DatetimeIndex,
            zones: tuple[str, ...]) -> np.ndarray:
    """The target matrix for ``days``: (n_days, 24 * len(zones)), zone-major."""
    days = pd.DatetimeIndex(days)
    Y = np.concatenate(
        [_lagged(matrices[zone]["price"], days, 0) for zone in zones], axis=1)
    _reject_nan(Y, days, target_names(zones), "target")
    return Y


def _reject_nan(values: np.ndarray, days, names, what: str) -> None:
    if not np.isnan(values).any():
        return
    rows, cols = np.nonzero(np.isnan(values))
    first = [(str(pd.Timestamp(days[r]).date()), names[c])
             for r, c in list(zip(rows, cols))[:5]]
    raise ZoneDataError(
        f"{int(np.isnan(values).sum())} NaN in the assembled {what} matrix, "
        f"first at {first}. Nothing here imputes: a lag reaching before the "
        f"panel starts, or a gap the cleaning notebook left, has to be fixed "
        f"upstream."
    )


def first_forecastable_day(matrices: dict) -> pd.Timestamp:
    """The earliest day whose 7-day lags are all inside the panel."""
    return available_days(matrices).min() + pd.Timedelta(days=MAX_LAG_DAYS)


def training_days(matrices: dict, next_day: pd.Timestamp,
                  calibration_window: int) -> pd.DatetimeIndex:
    """The days a network trains on before ``next_day``.

    The window is ``calibration_window`` *years* of 364 days, clipped at the
    front to the first day whose 7-day lags exist. Upstream arrives at the same
    days by a different route -- it slices the frame to the last ``52 *
    calibration_window`` weeks and then ``_build_and_split_XYs`` drops the first
    week of whatever remains -- so a window reaching before the panel starts is
    silently short there too, rather than an error. Kept identical here so the
    configurations train on the same days.
    """
    days = available_days(matrices)
    first = max(next_day - pd.Timedelta(days=calibration_window * 364),
                first_forecastable_day(matrices))
    return days[(days < next_day) & (days >= first)]


# ---------------------------------------------------------------------------
# Per-zone target scaling
# ---------------------------------------------------------------------------

class PerZoneScaler:
    """One :class:`epftoolbox.data.DataScaler` per zone, over its own 24 columns.

    The loss is summed over the output columns on the *transformed* scale, so
    what the scaler does decides how the gradient is shared between zones. Each
    zone therefore gets its own median/MAD and its own asinh, and the 168
    outputs are then weighted equally in a loss where "equal" means something.

    A note on what this buys, because it is not what one might assume:
    epftoolbox's ``MedianScaler`` (and so ``InvariantScaler``, and sklearn's
    ``StandardScaler``/``MinMaxScaler``) already fits *per column* --
    ``np.median(data, axis=0)``. Fitting one scaler across all 168 columns is
    therefore numerically identical to fitting seven across 24 each, and the
    failure mode ``docs/dnn_phase1_spec.md`` §4.3 warns about cannot arise with
    any ``scaleY`` upstream offers. :meth:`equals_pooled_fit` checks that
    equality rather than asserting it.

    It is still built this way, for two reasons that survive the correction: the
    inverse transform for one zone's sub-vector is then a first-class operation
    (§4.4 scores each zone separately), and the guarantee stops depending on a
    property of a scaler class that a future ``scaleY`` need not share.
    """

    def __init__(self, normalize: str, zones: tuple[str, ...]):
        self.normalize = normalize
        self.zones = tuple(zones)
        self.scalers: dict = {}
        self.constant_columns = 0

    def _slice(self, zone: str) -> slice:
        return zone_slice(zone, self.zones)

    def fit_transform(self, Y: np.ndarray) -> np.ndarray:
        from .scaling_compat import fit_scaler

        values = np.asarray(Y, dtype=float)
        out = np.zeros_like(values)
        self.constant_columns = 0
        for zone in self.zones:
            block = self._slice(zone)
            scaler, n_constant = fit_scaler(self.normalize, values[:, block])
            self.scalers[zone] = scaler
            self.constant_columns += n_constant
            out[:, block] = scaler.transform(values[:, block])
        return out

    def transform(self, Y: np.ndarray) -> np.ndarray:
        out = np.zeros_like(np.asarray(Y, dtype=float))
        for zone in self.zones:
            block = self._slice(zone)
            out[:, block] = self.scalers[zone].transform(
                np.asarray(Y, dtype=float)[:, block])
        return out

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        values = np.asarray(Y, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        out = np.zeros_like(values)
        for zone in self.zones:
            block = self._slice(zone)
            out[:, block] = self.scalers[zone].inverse_transform(values[:, block])
        return out

    def dispersion(self, Yt: np.ndarray) -> dict[str, float]:
        """Per-zone standard deviation of the transformed targets."""
        values = np.asarray(Yt, dtype=float)
        return {zone: float(np.std(values[:, self._slice(zone)]))
                for zone in self.zones}

    def equals_pooled_fit(self, Y: np.ndarray) -> bool:
        """Whether a single scaler over all columns would give the same numbers.

        True for every ``scaleY`` epftoolbox offers, because all of them fit
        column-wise. Recorded in the manifest rather than assumed.
        """
        from .scaling_compat import fit_scaler

        values = np.asarray(Y, dtype=float)
        pooled, _ = fit_scaler(self.normalize, values)
        return bool(np.allclose(pooled.transform(values),
                                self.fit_transform(values), equal_nan=True))


# ---------------------------------------------------------------------------
# Train / validation split -- upstream's, reproduced for the multi-zone case
# ---------------------------------------------------------------------------

def split_train_val(X: np.ndarray, Y: np.ndarray, shuffle_train: bool,
                    hyperoptimization: bool = False, percentage_val: float = 0.25):
    """Upstream's weekly-block shuffle and 75/25 split.

    ``epftoolbox.models._dnn._build_and_split_XYs`` ends with exactly this, but
    it cannot be called here: it reads a single ``Price`` column and a fixed
    ``Exogenous n`` naming, so it can build neither the 1969-column input nor the
    168-column target. The split is reproduced rather than reinvented -- same
    weekly blocks, same ``np.random.seed(7)`` during hyperopt, same fractions --
    because which days land in validation is part of the model's specification.

    Upstream writes the block expansion as
    ``[ind + i for ind in index_week for i in range(7) if ind + i in index]``
    over ``index = np.arange(n)``; membership in that array is exactly
    ``ind + i < n``, which is what is written below.
    """
    n = X.shape[0]
    if shuffle_train:
        if hyperoptimization:
            np.random.seed(7)
        index_week = np.arange(n)[::7]
        np.random.shuffle(index_week)
        order = [ind + i for ind in index_week for i in range(7) if ind + i < n]
        X = X[order]
        Y = Y[order]

    n_val = int(percentage_val * n)
    n_train = n - n_val
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]
