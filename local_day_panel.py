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


def normalize_local_hourly_panel(
    panel: pd.DataFrame,
    timezone: str = MARKET_TIMEZONE,
) -> tuple[pd.DataFrame, DSTReport]:
    """Convert a complete UTC long panel to 24-hour naive local delivery days."""
    matrices, report = build_local_day_matrices(panel, timezone)
    working = panel.copy()
    local = working["timestamp_utc"].dt.tz_convert(timezone)
    working["timestamp_local"] = local.dt.tz_localize(None)
    working["local_date"] = working["timestamp_local"].dt.normalize()
    working["local_hour"] = working["timestamp_local"].dt.hour

    value_and_time = {"timestamp_utc", "timestamp_local", "local_date", "local_hour", "value"}
    metadata_columns = [column for column in working.columns if column not in value_and_time]
    rows = []
    for series, group in working.groupby("series", observed=True, sort=True):
        matrix = matrices[str(series)]
        local_keys = pd.MultiIndex.from_product(
            [matrix.index, LOCAL_HOURS], names=["local_date", "local_hour"]
        )
        local_index = pd.DatetimeIndex(
            [day + pd.Timedelta(hours=hour) for day in matrix.index for hour in LOCAL_HOURS]
        )
        metadata = (
            group.sort_values("timestamp_utc")
            .groupby(["local_date", "local_hour"], as_index=False, sort=True)
            .first()
            .set_index(["local_date", "local_hour"])[metadata_columns]
            .reindex(local_keys)
            .ffill()
            .bfill()
            .reset_index(drop=True)
        )
        if "imputed" in group:
            audit = group.groupby(["local_date", "local_hour"], sort=True).agg(
                imputed=("imputed", "any"),
                imputation_method=("imputation_method", lambda values: next(
                    (value for value in values if value not in {"", "observed"}), "observed"
                )),
                imputation_predictors=("imputation_predictors", lambda values: next(
                    (value for value in values if value), ""
                )),
            ).reindex(local_keys)
            metadata[["imputed", "imputation_method", "imputation_predictors"]] = (
                audit.reset_index(drop=True)
            )
        metadata["timestamp_local"] = local_index
        metadata["value"] = matrix.to_numpy().reshape(-1)
        metadata["dst_adjustment"] = "none"
        metadata.loc[metadata["timestamp_local"].dt.normalize().isin(report.spring_days)
                     & metadata["timestamp_local"].dt.hour.eq(2),
                     "dst_adjustment"] = "spring_interpolation"
        metadata.loc[metadata["timestamp_local"].dt.normalize().isin(report.autumn_days)
                     & metadata["timestamp_local"].dt.hour.eq(2),
                     "dst_adjustment"] = "autumn_average"
        spring_rows = metadata["dst_adjustment"].eq("spring_interpolation")
        if "imputed" in metadata:
            metadata.loc[spring_rows, "imputed"] = False
        if "imputation_method" in metadata:
            metadata.loc[spring_rows, "imputation_method"] = "observed"
        if "imputation_predictors" in metadata:
            metadata.loc[spring_rows, "imputation_predictors"] = ""
        rows.append(metadata)

    result = pd.concat(rows, ignore_index=True)
    result = result[[
        *[column for column in panel.columns if column != "timestamp_utc"],
        "timestamp_local",
        "dst_adjustment",
    ]].sort_values(["series", "timestamp_local"]).reset_index(drop=True)
    if result.duplicated(["series", "timestamp_local"]).any():
        raise LocalDayPanelError("local panel contains duplicate series-hours")
    if result["value"].isna().any():
        raise LocalDayPanelError("local panel contains missing values")
    return result, report


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
