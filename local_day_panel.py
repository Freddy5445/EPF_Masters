"""Convert the canonical hourly UTC panel to 24-hour local delivery days."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

MARKET_TIMEZONE = "Europe/Copenhagen"
LOCAL_HOURS = pd.Index(range(24), name="local_hour")


class LocalDayPanelError(ValueError):
    """Raised when the UTC panel cannot form complete local delivery days."""


@dataclass(frozen=True)
class DSTReport:
    """DST dates normalized identically across every retained series."""

    timezone: str
    spring_days: tuple[pd.Timestamp, ...]
    autumn_days: tuple[pd.Timestamp, ...]
    series_count: int


def build_local_day_matrices(
    panel: pd.DataFrame,
    timezone: str = MARKET_TIMEZONE,
) -> tuple[dict[str, pd.DataFrame], DSTReport]:
    """Return one local-date by local-hour value matrix per series.

    The input remains the canonical UTC representation. On conversion to local
    delivery time, the nonexistent spring 02:00 is interpolated from 01:00 and
    03:00, while the two autumn 02:00 observations are averaged. Any missing or
    repeated hour not explained by those conventions is rejected.
    """
    required = {"series", "timestamp_utc", "value"}
    missing_columns = sorted(required - set(panel.columns))
    if missing_columns:
        raise LocalDayPanelError(f"UTC panel is missing columns: {missing_columns}")
    if panel.empty:
        raise LocalDayPanelError("UTC panel is empty")
    if panel["value"].isna().any():
        raise LocalDayPanelError("UTC panel contains missing values")
    if panel.duplicated(["series", "timestamp_utc"]).any():
        raise LocalDayPanelError("UTC panel contains duplicate series-hours")

    timestamps = pd.to_datetime(panel["timestamp_utc"])
    if timestamps.dt.tz is None:
        raise LocalDayPanelError("timestamp_utc must be timezone-aware")

    matrices: dict[str, pd.DataFrame] = {}
    common_spring: tuple[pd.Timestamp, ...] | None = None
    common_autumn: tuple[pd.Timestamp, ...] | None = None

    for series, group in panel.groupby("series", observed=True, sort=True):
        local = group["timestamp_utc"].dt.tz_convert(timezone)
        coordinates = pd.DataFrame(
            {
                "local_date": local.dt.tz_localize(None).dt.normalize(),
                "local_hour": local.dt.hour,
                "value": group["value"].to_numpy(),
            }
        )
        counts = coordinates.groupby(["local_date", "local_hour"]).size()
        repeated = counts[counts > 1]
        invalid_repeated = repeated[
            (repeated.index.get_level_values("local_hour") != 2) | (repeated != 2)
        ]
        if not invalid_repeated.empty:
            raise LocalDayPanelError(
                f"{series!r} has repeated local hours not explained by autumn DST"
            )
        autumn_days = tuple(
            pd.DatetimeIndex(repeated.index.get_level_values("local_date")).unique()
        )

        matrix = coordinates.pivot_table(
            index="local_date", columns="local_hour", values="value", aggfunc="mean"
        )
        dates = pd.date_range(matrix.index.min(), matrix.index.max(), freq="D")
        matrix = matrix.reindex(index=dates, columns=LOCAL_HOURS)
        matrix.index.name = "local_date"

        missing = matrix.isna()
        missing_positions = missing.stack()[lambda values: values]
        spring_days = tuple(pd.DatetimeIndex(matrix.index[matrix[2].isna()]))
        allowed_positions = {(day, 2) for day in spring_days}
        actual_positions = set(missing_positions.index)
        if actual_positions != allowed_positions:
            unexpected = sorted(actual_positions - allowed_positions)[:3]
            raise LocalDayPanelError(
                f"{series!r} has missing local hours not explained by spring DST: "
                f"{unexpected}"
            )

        if spring_days:
            skipped = pd.DatetimeIndex(
                [day + pd.Timedelta(hours=2) for day in spring_days]
            ).tz_localize(timezone, ambiguous=True, nonexistent="NaT")
            if not skipped.isna().all():
                raise LocalDayPanelError(
                    f"{series!r} is missing local 02:00 outside a spring transition"
                )
            matrix.loc[list(spring_days), 2] = (
                matrix.loc[list(spring_days), 1].to_numpy()
                + matrix.loc[list(spring_days), 3].to_numpy()
            ) / 2

        if matrix.isna().any().any():
            raise LocalDayPanelError(f"{series!r} is incomplete after DST normalization")
        if not matrix.iloc[[0, -1]].notna().all(axis=None):
            raise LocalDayPanelError(f"{series!r} has an incomplete boundary day")
        if matrix.shape[1] != 24:
            raise LocalDayPanelError(f"{series!r} does not have 24 local-hour columns")

        if common_spring is None:
            common_spring, common_autumn = spring_days, autumn_days
        elif spring_days != common_spring or autumn_days != common_autumn:
            raise LocalDayPanelError(f"{series!r} has inconsistent DST transition days")
        matrices[str(series)] = matrix

    report = DSTReport(
        timezone=timezone,
        spring_days=common_spring or (),
        autumn_days=common_autumn or (),
        series_count=len(matrices),
    )
    return matrices, report


def flatten_local_day_matrix(matrix: pd.DataFrame) -> pd.Series:
    """Flatten a local-date by local-hour matrix to a naive hourly series."""
    if not matrix.columns.equals(LOCAL_HOURS):
        raise LocalDayPanelError("matrix columns must be local hours 0 through 23")
    if matrix.isna().any().any():
        raise LocalDayPanelError("matrix contains missing values")

    index = pd.DatetimeIndex(
        [day + pd.Timedelta(hours=hour) for day in matrix.index for hour in LOCAL_HOURS]
    )
    return pd.Series(matrix.to_numpy().reshape(-1), index=index, name="value")
