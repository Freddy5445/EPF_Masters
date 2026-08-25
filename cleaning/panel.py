"""
Turn the raw multi-zone panel into the gap-free panel the models read.

One implementation, run once in ``data_cleaning.ipynb``, so that every model
tested against this data provably sees identically prepared inputs. A difference
between LEAR and the DNN can then be a difference between the models, which is
the whole point of comparing them.

The order matters and is not arbitrary:

1. **Project each series onto the naive local hourly grid.** Fall-back duplicates
   are averaged here (:mod:`entsoe_tp.hourly`); nothing else is invented.
2. **Fill the spring-forward hour** by interpolation, the epftoolbox convention
   (:mod:`cleaning.dst`). Done before imputation so the causal filler never sees
   a hole that is an artefact of the clock rather than of the data.
3. **Impute what is genuinely missing**, from past observations only
   (:mod:`cleaning.impute`).

Step 3 has no epftoolbox precedent: the published NP/BE/FR/DE datasets are
complete -- NP's real price has zero NaN across its whole 728-day test period --
so the paper never had to fill a genuinely missing price. The causal cascade
here is this project's own choice and is documented as such.

Imputation runs on each zone's series *before* they are summed into model
columns. A wind total is the sum of onshore and offshore, and summing first
would turn one missing component into a missing total, discarding the component
that was published. Filling first keeps it.
"""

import numpy as np
import pandas as pd

from entsoe_tp.areas import lookup
from entsoe_tp.hourly import to_local_hourly_grid

from .dst import fill_skipped_hours
from .impute import first_complete_day, impute_frame

# Columns the raw panel is expected to carry.
PANEL_COLUMNS = ["timestamp_utc", "zone", "variable", "psr_type", "value"]


def _series_key(variable, psr_type):
    """A stable column name for one (variable, psr_type) pair."""
    return f"{variable}|{psr_type}" if psr_type else variable


def _split_key(key):
    return tuple(key.split("|", 1)) if "|" in key else (key, "")


def clean_zone(rows, zone, max_ffill=3):
    """Clean every series of one zone. Returns ``(wide_frame, report)``.

    ``wide_frame`` is indexed by naive local time with one column per series and
    exactly 24 rows per calendar day.
    """
    tz = lookup(zone).tz

    keys, series = [], []
    for (variable, psr), group in rows.groupby(["variable", "psr_type"], dropna=False):
        psr = "" if (psr is None or (isinstance(psr, float) and np.isnan(psr))) else str(psr)
        s = group.set_index("timestamp_utc")["value"].sort_index()
        keys.append(_series_key(variable, psr))
        series.append(s)

    if not series:
        raise ValueError(f"{zone} has no series in the panel")

    # A single span for the whole zone, so its series share one index and the
    # model columns line up without a later reindex.
    spans = [s.index for s in series]
    local_min = min(idx.min() for idx in spans).tz_convert(tz).tz_localize(None)
    local_max = max(idx.max() for idx in spans).tz_convert(tz).tz_localize(None)
    start = local_min.normalize()
    end = local_max.normalize()
    if local_max.hour != 23:
        end -= pd.Timedelta(days=1)

    frame = pd.DataFrame({
        key: to_local_hourly_grid(s, tz, start, end, allow_gaps=True)
        for key, s in zip(keys, series)
    })

    report = {"zone": zone, "hours": len(frame), "timezone": str(tz),
              "missing_before": {k: int(v) for k, v in frame.isna().sum().items()}}

    frame, n_dst = fill_skipped_hours(frame, tz)
    report["dst_hours_interpolated"] = n_dst

    if frame.isna().any().any():
        frame, imputation = impute_frame(frame, max_ffill=max_ffill)
        report["imputation"] = imputation
    else:
        report["imputation"] = {}

    trimmed = 0
    if frame.isna().any().any():
        # Causal imputation cannot fill hours with nothing behind them.
        usable = first_complete_day(frame)
        trimmed = int((frame.index < usable).sum())
        frame = frame.loc[usable:]
    report["trimmed_leading_hours_no_history"] = trimmed
    report["complete"] = not bool(frame.isna().any().any())
    report["hours_after"] = len(frame)

    return frame, report


def clean_panel(panel, max_ffill=3, zones=None, quiet=False):
    """Clean every zone. Returns ``(cleaned_panel, report)``.

    The cleaned panel keeps the raw panel's long shape -- one row per
    (timestamp, zone, variable, psr_type) -- but its timestamps are naive local
    market time, every calendar day holds exactly 24 hours, and no value is
    missing.
    """
    missing = [c for c in PANEL_COLUMNS if c not in panel.columns]
    if missing:
        raise ValueError(
            f"The panel is missing column(s) {', '.join(missing)}. Expected "
            f"{', '.join(PANEL_COLUMNS)}, as written by the raw dump.")

    panel = panel.copy()
    panel["psr_type"] = panel["psr_type"].astype("object").where(
        panel["psr_type"].notna(), "")

    available = sorted(panel["zone"].dropna().unique())
    zones = zones or available
    unknown = [z for z in zones if z not in available]
    if unknown:
        raise ValueError(f"Not in the panel: {', '.join(unknown)}. "
                         f"It holds {', '.join(available)}.")

    frames, reports = [], []
    for zone in zones:
        wide, report = clean_zone(panel[panel.zone == zone], zone, max_ffill=max_ffill)
        reports.append(report)

        long = wide.stack().rename("value").reset_index()
        long.columns = ["timestamp_local", "series", "value"]
        long[["variable", "psr_type"]] = pd.DataFrame(
            [_split_key(k) for k in long["series"]], index=long.index)
        long["zone"] = zone
        frames.append(long[["timestamp_local", "zone", "variable", "psr_type", "value"]])

        if not quiet:
            print(f"{zone:5} {report['hours_after']:>8,} hours  "
                  f"DST filled {report['dst_hours_interpolated']:>2}  "
                  f"imputed {sum(v['missing'] for v in report['imputation'].values()):>6,}  "
                  f"trimmed {report['trimmed_leading_hours_no_history']:>4}  "
                  f"{'complete' if report['complete'] else 'STILL HAS GAPS'}")

    cleaned = pd.concat(frames, ignore_index=True)
    return cleaned, {"zones": reports, "max_ffill": max_ffill}


def format_report(report):
    """A readable summary of what cleaning did."""
    lines = [f"Cleaned {len(report['zones'])} zone(s), "
             f"forward-fill limit {report['max_ffill']}h"]
    for z in report["zones"]:
        imputed = sum(v["missing"] for v in z["imputation"].values())
        lines.append(
            f"  {z['zone']:5} {z['hours_after']:>8,} hours  "
            f"DST {z['dst_hours_interpolated']:>2}  imputed {imputed:>6,}  "
            f"trimmed {z['trimmed_leading_hours_no_history']:>4}"
            + ("" if z["complete"] else "   <-- STILL HAS GAPS"))
    return "\n".join(lines)
