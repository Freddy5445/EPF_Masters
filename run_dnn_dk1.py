"""
Backtest the DNN in one of four configurations, with daily recalibration.

    python run_dnn_dk1.py --config own      --zone DK1 --smoke
    python run_dnn_dk1.py --config wide     --zone DK1 --smoke
    python run_dnn_dk1.py --config joint_dk --smoke
    python run_dnn_dk1.py --config joint    --smoke

The four configurations decompose the effect of cross-zonal information into an
*input* effect and an *output* effect, and then ask how far the output effect
extends:

============  ==========================  ==========  =========  ==============
Config        Inputs                      Width       Outputs    Runs
============  ==========================  ==========  =========  ==============
``own``       the zone's own series        313 / 241  24         one per zone (7)
``wide``      every zone in Z              1969       24         DK1, DK2 (2)
``joint_dk``  every zone in Z              1969       48         one (1)
``joint``     every zone in Z              1969       168        one (1)
============  ==========================  ==========  =========  ==============

``own -> wide`` is the input effect: the same 24-output network, given every
zone's series instead of its own. ``wide -> joint_dk`` is the output effect: the
same inputs and the same search space, forecasting DK1 and DK2 from one head
instead of two networks -- Lago et al. (2018b)'s dual-market forecaster.
``joint_dk -> joint`` asks whether the five less-related auxiliary zones add to
that or dilute it, which is this thesis's own extension and the empirical test of
the interconnection criterion.

``own`` is Lago et al. (2021) unmodified -- the same 11 (or 14) binary feature
toggles, the same architecture ranges, the same objective. ``wide``, ``joint_dk``
and ``joint`` share an **identical** input space and an **identical** search
space; only the width of the output layer differs. That is what makes each step
above a clean measurement rather than a confounded one. See
``dnn_dk1/hyperopt.py`` for why the block toggles are dropped there and replaced
by the L1 penalty.

``--smoke`` runs a short search and a few days. It proves the pipeline runs; it
proves nothing about accuracy. The paper searches 1500 hyperparameter
evaluations, and five is not a search.

Note on units: the DNN's calibration window is in **years** (upstream trains on
the last ``calibration_window * 364`` days), where LEAR's windows are in days.

Commands are given on one line: this project is developed from PowerShell, where
a trailing ``\`` is not a line continuation and truncates the command.
"""

import argparse
import gc
import io
import json
import os
import sys
import time
from contextlib import redirect_stdout

# Keras is chatty on import and TensorFlow logs device probing at INFO; neither
# says anything useful here and both bury the progress output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Private, but shared on purpose: the resume semantics -- in particular that an
# incomplete final row is discarded rather than trusted -- must match the LEAR
# backtest's, or the two models would recover differently from an interruption.
from lear_dk1.backtest import _load_checkpoint  # noqa: E402

from dnn_dk1 import runs as RS  # noqa: E402
from dnn_dk1 import zones as Z  # noqa: E402

# Returned when this process stopped on its day budget with days still to go.
# Distinct from 0 (finished) and from a crash, so the scheduler can tell a
# planned handover from a failure without parsing a log.
CHUNK_INCOMPLETE = 75

HOURS = [f"h{h}" for h in range(24)]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASETS = os.path.join(THIS_DIR, "datasets")
DEFAULT_OUT = os.path.join(THIS_DIR, "experiments")
DEFAULT_PANEL = os.path.join(
    THIS_DIR, "datasets", "nordic_baltic_clean_hourly_local.parquet")

CONFIGS = ("own", "wide", "joint_dk", "joint")

# The reported test period, identical to the LEAR thesis run and to every
# configuration here -- an unequal test period would confound the comparison
# just as an unequal budget would.
DEFAULT_DATA_START = str(Z.PANEL_START.date())
DEFAULT_BEGIN_TEST = str(Z.BEGIN_TEST.date())
DEFAULT_END_TEST = str(Z.END_TEST.date())

# The paper's DNN: networks differing only in their random seed, 4-year window.
DEFAULT_SEEDS = (1, 2, 3, 4)
DEFAULT_CALIBRATION_YEARS = Z.CALIBRATION_YEARS
DEFAULT_MAX_EVALS = 1500

# Hyperopt sees the year before the test period and never the test period
# itself. 363 days back from the last day before the test start.
HYPEROPT_DAYS = 363


def _append_timing(path, row):
    """Append one row to a per-seed timing file, writing the header once."""
    header = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)


def _sample_rss(path, run_label, chunk_days, days_done, note):
    """Record this process's resident set size in its own run directory.

    Written here rather than straight into ``progress_log.csv`` because up to
    five runs are alive at once and interleaved appends from five processes
    would corrupt a shared file. Each run owns this one; ``dnn_status`` reads
    them and puts the number into ``progress_log.csv``, which has a single
    writer.

    The growth is the thing to watch: the first launch died on an
    ArrayMemoryError after about six hours, and phase 1's peak RSS was measured
    over five recalibrations, which cannot show a trend.
    """
    try:
        import psutil

        info = psutil.Process().memory_info()
        rss = int(getattr(info, "peak_wset", info.rss))
        current = int(info.rss)
    except Exception:
        return
    header = not os.path.exists(path)
    try:
        pd.DataFrame([{
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "run": run_label,
            "pid": os.getpid(),
            "chunk_days": chunk_days,
            "days_done": days_done,
            "rss_bytes": current,
            "peak_rss_bytes": rss,
            "note": note,
        }]).to_csv(path, mode="a", header=header, index=False)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def dataset_name(config, zone):
    """The name a configuration's trials file and run directory are built from.

    ``own`` uses the zone's cleaned CSV name, so its trials file is
    interchangeable with upstream's. ``wide`` and ``joint`` are not tied to a
    single CSV, so they get names of their own.
    """
    if config == "own":
        return Z.dataset_name(zone)
    if config == "wide":
        return f"dnnwide_{zone}"
    if config == "joint_dk":
        return "dnnjointdk"
    return "dnnjoint"


def out_zones_for(config, zone):
    """Which zones the output layer covers.

    Delegated to ``dnn_dk1.runs`` so the launcher, the status tool and the
    pre-flight cannot disagree with this script about how wide a run's output
    head is.
    """
    return RS.out_zones_for(config, zone)


def input_zones_for(config, zone):
    """Which zones the input matrix covers."""
    if config == "own":
        return (zone,)
    return Z.ZONES


def expected_input_width(config, zone):
    if config == "own":
        return Z.own_input_width(zone)
    return Z.input_width(Z.ZONES)


def run_dir_for(config, zone, begin_test, end_test, out_dir, smoke=False):
    name = f"{dataset_name(config, zone)}_dnn_{begin_test.date()}_{end_test.date()}"
    if smoke:
        name += "_smoke"
    return os.path.join(out_dir, name)


def hyperopt_window(begin_test):
    """The search window: ``HYPEROPT_DAYS`` days ending the hour before the test.

    ``read_data`` splits on ``begin_test_date`` and the search scores on what
    follows it, so passing the real test range would select features on the very
    days the model is then evaluated on.
    """
    end = pd.Timestamp(begin_test).normalize() - pd.Timedelta(hours=1)
    begin = end.normalize() - pd.Timedelta(days=HYPEROPT_DAYS)
    return begin, end


# ---------------------------------------------------------------------------
# Pre-flight assertions (spec section 5)
# ---------------------------------------------------------------------------

def preflight(config, zone, datasets_dir, begin_test, end_test,
              panel_path=DEFAULT_PANEL, matrices=None, quiet=False):
    """The seven assertions. Returns ``(results, matrices)``.

    Follows the LEAR run's pattern: each assertion records ``id``, ``name``,
    ``passed`` and ``detail``; the caller inspects ``passed`` and refuses to
    proceed. Nothing here is inferred -- the column counts are checked on a
    matrix that was actually built.
    """
    results = []

    def record(assertion_id, name, passed, detail):
        results.append({"id": assertion_id, "name": name,
                        "passed": bool(passed), "detail": detail})

    # --- 1. Every zone in Z has a cleaned CSV covering the panel span ----
    spans, missing = {}, []
    for z in Z.ZONES:
        path = Z.dataset_path(z, datasets_dir)
        if not os.path.exists(path):
            missing.append(f"{z} -> {os.path.basename(path)}")
            continue
        index = pd.to_datetime(pd.read_csv(path, index_col=0, usecols=[0]).index)
        spans[z] = (index.min().normalize(), index.max().normalize())
    covered = (not missing) and all(
        first <= Z.PANEL_START and last >= Z.PANEL_END
        for first, last in spans.values())
    record(
        1, f"every zone in Z has a cleaned CSV covering "
           f"{Z.PANEL_START.date()}..{Z.PANEL_END.date()}",
        covered,
        (f"missing: {', '.join(missing)}; " if missing else "")
        + "; ".join(f"{z} {a.date()}..{b.date()}" for z, (a, b) in spans.items()),
    )
    if missing:
        return results, None

    if matrices is None:
        matrices = Z.load_zone_matrices(Z.ZONES, datasets_dir)

    # --- 6. No NaN anywhere in any assembled matrix ----------------------
    # Checked before the width assertions because those build matrices too, and
    # the builders raise on NaN -- so this has to be the thing that reports it.
    burn_in_days = Z.available_days(matrices)
    check_days = burn_in_days[burn_in_days >= Z.HISTORY_START_REQUIRED]
    try:
        X_all = Z.build_X(matrices, check_days, Z.ZONES, include_calendar=True)
        Y_all = Z.build_Y(matrices, check_days, Z.ZONES)
        nan_detail = (f"{X_all.shape[0]} days x {X_all.shape[1]} inputs and "
                      f"{Y_all.shape[1]} targets, "
                      f"{check_days.min().date()}..{check_days.max().date()}: "
                      f"no NaN")
        nan_ok = True
    except Z.ZoneDataError as exc:
        X_all = Y_all = None
        nan_detail = str(exc)
        nan_ok = False
    record(6, "no NaN anywhere in any assembled matrix", nan_ok, nan_detail)
    if not nan_ok:
        return results, matrices

    # --- 2. Column counts, checked on the built matrix -------------------
    own_widths = {}
    for z in Z.ZONES:
        X_own = Z.build_X(matrices, check_days[:8], (z,), include_calendar=True)
        own_widths[z] = int(X_own.shape[1])
    derived_own = {z: Z.own_input_width(z) for z in Z.ZONES}
    derived_wide = Z.input_width(Z.ZONES)
    widths_ok = (
        own_widths == derived_own
        and int(X_all.shape[1]) == derived_wide
        and int(Y_all.shape[1]) == Z.output_width(Z.ZONES) == 168
        # the arithmetic the widths are supposed to satisfy, spelled out
        and all(derived_own[z] == 96 + 72 * Z.n_exogenous(z) + 1 for z in Z.ZONES)
        and derived_wide == sum(96 + 72 * Z.n_exogenous(z) for z in Z.ZONES) + 1
    )
    record(
        2, "input and output widths, checked on the built matrix", widths_ok,
        f"DNN-own built {own_widths}; DNN-wide/joint built "
        f"{int(X_all.shape[1])} inputs; DNN-joint built {int(Y_all.shape[1])} "
        f"outputs. 96 + 72*n_exog + 1 per zone with n_exog="
        f"{ {z: Z.n_exogenous(z) for z in Z.ZONES} }",
    )

    # --- 3. DST transition days, identical across all zones' series -------
    dst_ok, dst_detail = _dst_assertion(panel_path)
    record(3, "DST transition days identical across all zones' series",
           dst_ok, dst_detail)

    # --- 4. Test period is exactly 731 days, identical for every zone -----
    configured = (end_test - begin_test).days + 1
    covers = {z: bool(spans[z][0] <= begin_test and spans[z][1] >= end_test)
              for z in Z.ZONES}
    record(
        4, f"test period is exactly {Z.TEST_DAYS_EXPECTED} days, identical for "
           f"every zone",
        configured == Z.TEST_DAYS_EXPECTED and all(covers.values()),
        f"{begin_test.date()}..{end_test.date()} = {configured} days; every "
        f"zone CSV spans it: {all(covers.values())}",
    )

    # --- 5. Hyperopt validation window does not touch the test period -----
    hyper_begin, hyper_end = hyperopt_window(begin_test)
    record(
        5, "hyperopt validation window does not overlap the test period",
        hyper_end < begin_test,
        f"search on {hyper_begin.date()}..{hyper_end.date()} "
        f"({(hyper_end.normalize() - hyper_begin).days + 1} days), test from "
        f"{begin_test.date()}; gap {begin_test - hyper_end}",
    )

    # --- 7. Output column order round-trips through the per-zone scaler ---
    record(7, *_round_trip_assertion(matrices, check_days))

    if not quiet:
        print_preflight(results)
    return results, matrices


def _dst_assertion(panel_path):
    """The cleaning stage's DST record, read from the panel it wrote."""
    if not os.path.exists(panel_path):
        return False, f"no cleaned panel at {panel_path}"
    panel = pd.read_parquet(
        panel_path, columns=["series", "zone", "timestamp_local", "dst_adjustment"])
    panel["timestamp_local"] = pd.to_datetime(panel["timestamp_local"])
    panel["zone"] = panel["zone"].astype("object").fillna("").astype(str)
    panel = panel[panel.zone.isin(Z.ZONES)]

    days = panel["timestamp_local"].dt.normalize()
    spring = sorted(days[panel.dst_adjustment == "spring_interpolation"].unique())
    autumn = sorted(days[panel.dst_adjustment == "autumn_average"].unique())
    per_series = panel[panel.dst_adjustment != "none"].groupby(
        ["dst_adjustment", days]).series.nunique()
    n_series = panel.series.nunique()
    ok = (len(spring) == Z.DST_SPRING_EXPECTED
          and len(autumn) == Z.DST_AUTUMN_EXPECTED
          and bool(per_series.eq(n_series).all()))
    return ok, (
        f"{len(spring)} spring {[str(pd.Timestamp(d).date()) for d in spring]}, "
        f"{len(autumn)} autumn {[str(pd.Timestamp(d).date()) for d in autumn]}, "
        f"each affecting all {n_series} series of the {len(Z.ZONES)} zones "
        f"(expected {Z.DST_SPRING_EXPECTED} spring / {Z.DST_AUTUMN_EXPECTED} "
        f"autumn)")


def _round_trip_assertion(matrices, days):
    """Assertion 7: the 168 outputs slice and invert back to per-zone prices.

    Inverse-transforming the transformed targets must give each zone's own price
    scale back -- medians in the tens of EUR/MWh, not the single-digit asinh
    scale -- and the zone-major column order must line up with
    :func:`dnn_dk1.zones.zone_slice`, checked against each zone's own CSV rather
    than against the assembled matrix it came from.
    """
    sample = days[-364:]
    Y = Z.build_Y(matrices, sample, Z.ZONES)
    scaler = Z.PerZoneScaler("Invariant", Z.ZONES)
    Yt = scaler.fit_transform(Y)
    back = scaler.inverse_transform(Yt)

    round_trip = float(np.max(np.abs(back - Y)))
    per_zone = {z: float(np.median(np.abs(back[:, Z.zone_slice(z)])))
                for z in Z.ZONES}
    # Independently: each zone's own day x hour matrix over the same days.
    direct = {z: float(np.median(np.abs(
        matrices[z]["price"].reindex(sample).to_numpy(float)))) for z in Z.ZONES}

    transformed_median = float(np.median(np.abs(Yt)))
    dispersion = scaler.dispersion(Yt)
    spread = max(dispersion.values()) / min(dispersion.values())

    ok = (round_trip < 1e-6
          and all(abs(per_zone[z] - direct[z]) < 1e-9 for z in Z.ZONES)
          and all(5.0 < v < 1000.0 for v in per_zone.values())
          and transformed_median < 5.0
          and spread < 3.0)
    return (
        "DNN-joint output order round-trips to the per-zone price scale",
        ok,
        f"round-trip max |error| {round_trip:.2e}; median |value| per zone "
        f"{ {z: round(v, 1) for z, v in per_zone.items()} } EUR/MWh, matching "
        f"each zone's own CSV; median |transformed| {transformed_median:.3f} "
        f"(asinh scale); transformed dispersion "
        f"{ {z: round(v, 3) for z, v in dispersion.items()} }, max/min "
        f"{spread:.2f}",
    )


def print_preflight(results):
    print("\nPre-flight assertions")
    print("=" * 78)
    for r in sorted(results, key=lambda r: r["id"]):
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['id']}. {r['name']}")
        print(f"       {r['detail']}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# The backtest
# ---------------------------------------------------------------------------

def build_models(config, zone, seeds, hyper_dir, nlayers, calibration_years):
    """One forecaster per seed. ``wide`` and ``joint`` differ only in out_zones."""
    name = dataset_name(config, zone)
    if config == "own":
        from dnn_dk1 import DNN
        return {seed: DNN(path_hyperparameter_folder=hyper_dir, experiment_id=1,
                          nlayers=nlayers, dataset=name,
                          calibration_window=calibration_years, seed=seed)
                for seed in seeds}

    from dnn_dk1 import MultiZoneDNN
    return {seed: MultiZoneDNN(
        path_hyperparameter_folder=hyper_dir, experiment_id=1, nlayers=nlayers,
        dataset=name, calibration_window=calibration_years, seed=seed,
        zones=input_zones_for(config, zone), out_zones=out_zones_for(config, zone))
        for seed in seeds}


def forecast_columns(config, zone):
    """Column names of a seed's forecast file.

    A multi-zone head writes one column per (zone, hour) so the file says which
    zone each number belongs to; a 24-output run writes bare hours.
    """
    if config in RS.JOINT_CONFIGS:
        return Z.target_names(out_zones_for(config, zone))
    return HOURS


def write_zone_slices(run_dir, seeds, zones):
    """Split a DNN-joint forecast file into one 24-column file per zone.

    ``lear_dk1.evaluate.evaluate_run`` scores a 24-column forecast against one
    zone's observed prices -- which is exactly what section 4.4 asks for, one zone
    at a time and never pooled. Rather than write a second evaluator, the joint
    forecast is sliced into per-zone run directories that the existing one reads.
    """
    written = {}
    names = Z.target_names(zones)
    for zone in zones:
        zone_dir = os.path.join(run_dir, f"zone_{zone}")
        os.makedirs(zone_dir, exist_ok=True)
        columns = names[Z.zone_slice(zone, zones)]
        for seed in seeds:
            source = os.path.join(run_dir, f"forecasts_joint_seed{seed}.csv")
            if not os.path.exists(source):
                continue
            frame = pd.read_csv(source, index_col=0)
            frame.index = pd.to_datetime(frame.index)
            slice_ = frame[columns].dropna(how="any")
            slice_.columns = HOURS
            slice_.to_csv(os.path.join(zone_dir, f"forecasts_seed{seed}.csv"))
        written[zone] = zone_dir
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Backtest the epftoolbox DNN with daily recalibration.")
    parser.add_argument("--config", default="own", choices=CONFIGS,
                        help="own: the zone's own inputs, 24 outputs (Lago "
                             "unmodified). wide: every zone's inputs, 24 "
                             "outputs. joint_dk: every zone's inputs, 48 "
                             "outputs (DK1+DK2). joint: every zone's inputs, "
                             "168 outputs (all of Z).")
    parser.add_argument("--zone", "--dataset", dest="zone", default="DK1",
                        help="Focal zone. Ignored by the joint configurations.")
    parser.add_argument("--datasets-dir", default=DEFAULT_DATASETS)
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--hyperparameter-dir", default=None,
                        help="Where the hyperopt trials file lives "
                             "(default: <out-dir>/hyperparameters)")
    parser.add_argument("--data-start", default=DEFAULT_DATA_START)
    parser.add_argument("--begin-test", default=DEFAULT_BEGIN_TEST)
    parser.add_argument("--end-test", default=DEFAULT_END_TEST)
    parser.add_argument("--max-evals", type=int, default=DEFAULT_MAX_EVALS,
                        help=f"Hyperopt evaluations (paper: {DEFAULT_MAX_EVALS}). "
                             f"Must be identical across all ten runs.")
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--recalibration-days", type=int,
                        default=RS.RECALIBRATION_DAYS,
                        help=f"Refit every N days and reuse the fit for the "
                             f"days between (default: {RS.RECALIBRATION_DAYS}). "
                             f"Every day is still forecast, with inputs built "
                             f"for itself; only the weights age.")
    parser.add_argument("--max-days-per-process", type=int,
                        default=RS.DAYS_PER_PROCESS,
                        help=f"Exit cleanly after this many newly forecast days "
                             f"so the scheduler can restart from the checkpoint "
                             f"(default: {RS.DAYS_PER_PROCESS}; 0 = no limit). "
                             f"TensorFlow's heap grows across hundreds of model "
                             f"builds and clear_session() does not give it back.")
    parser.add_argument("--calibration-years", type=int,
                        default=DEFAULT_CALIBRATION_YEARS,
                        help="Training window in YEARS (the DNN's unit, not days)")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Comma-separated seeds; the ensemble averages them")
    parser.add_argument("--skip-hyperopt", action="store_true",
                        help="Reuse an existing trials file instead of searching")
    parser.add_argument("--no-evaluate", action="store_true",
                        help="Skip scoring; write the forecasts only")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Do not run the section 5 assertions (they are "
                             "cheap; this exists for debugging only)")
    parser.add_argument("--smoke", action="store_true",
                        help="5 hyperopt evaluations, 3 forecast days, 1 seed. "
                             "Proves the pipeline runs; proves nothing about "
                             "accuracy.")
    args = parser.parse_args(argv)

    from dnn_dk1 import hyperopt as dnn_hyperopt

    config = args.config
    zone = args.zone if config not in RS.JOINT_CONFIGS else Z.ZONES[0]
    if config not in RS.JOINT_CONFIGS and zone not in Z.ZONES:
        print(f"error: {zone} is not in Z = {', '.join(Z.ZONES)}", file=sys.stderr)
        return 1

    begin_test = pd.Timestamp(args.begin_test).normalize()
    end_test = pd.Timestamp(args.end_test).normalize()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    max_evals = args.max_evals

    if args.smoke:
        max_evals = min(max_evals, 5)
        end_test = begin_test + pd.Timedelta(days=2)
        seeds = seeds[:1]
        print(f"Smoke run: {max_evals} hyperopt evals, "
              f"{begin_test.date()} to {end_test.date()}, seed {seeds[0]}.")
        print("This validates the pipeline. It is not a model.\n")

    if begin_test > end_test:
        print(f"error: --begin-test {begin_test.date()} is after --end-test "
              f"{end_test.date()}", file=sys.stderr)
        return 1

    name = dataset_name(config, zone)
    hyper_dir = args.hyperparameter_dir or os.path.join(args.out_dir,
                                                        "hyperparameters")
    hyperopt_begin, hyperopt_end = hyperopt_window(begin_test)

    # --- Pre-flight ---------------------------------------------------
    matrices = None
    preflight_results = []
    if not args.skip_preflight:
        preflight_results, matrices = preflight(
            config, zone, args.datasets_dir, begin_test,
            begin_test + pd.Timedelta(days=Z.TEST_DAYS_EXPECTED - 1),
            panel_path=args.panel)
        failed = [r for r in preflight_results if not r["passed"]]
        if failed:
            print("\nerror: refusing to run -- "
                  + "; ".join(f"assertion {r['id']} failed" for r in failed),
                  file=sys.stderr)
            return 1

    if matrices is None:
        matrices = Z.load_zone_matrices(Z.ZONES, args.datasets_dir)

    # --- Hyperparameter search ----------------------------------------
    # A search already on disk is resumed, never restarted. This is not only
    # about not repeating two or three hours of work: `optimize` defaults to
    # new_hyperopt=1, so a relaunched run would re-search from scratch, overwrite
    # the trials file, and very likely select *different* hyperparameters -- and
    # the days already forecast would then have come from a different model than
    # the days still to come, with nothing in the output saying so. Resuming a
    # backtest has to resume the search that defined it.
    trials_path, trials_done = _existing_trials(
        hyper_dir, config, zone, args.nlayers, args.calibration_years)
    search_complete = trials_done >= max_evals

    if args.skip_hyperopt:
        pass
    elif search_complete:
        print(f"Hyperparameter search ({config}): reusing the completed search "
              f"in {os.path.basename(trials_path)} ({trials_done} evaluations)\n")
    else:
        if trials_done:
            print(f"Hyperparameter search ({config}): resuming at "
                  f"{trials_done}/{max_evals} evaluations")

    if not args.skip_hyperopt and not search_complete:
        print(f"Hyperparameter search ({config}): {max_evals} evaluations on "
              f"{hyperopt_begin.date()}..{hyperopt_end.date()} "
              f"(before the test period, so no leakage)")
        started = time.time()
        if config == "own":
            path = dnn_hyperopt.optimize(
                path_datasets_folder=args.datasets_dir,
                path_hyperparameters_folder=hyper_dir, dataset=name,
                begin_test_date=hyperopt_begin, end_test_date=hyperopt_end,
                max_evals=max_evals, nlayers=args.nlayers,
                calibration_window=args.calibration_years, quiet=True,
                new_hyperopt=not trials_done)
        else:
            path = dnn_hyperopt.optimize_multizone(
                matrices=matrices, path_hyperparameters_folder=hyper_dir,
                dataset=name, begin_test_date=hyperopt_begin,
                end_test_date=hyperopt_end.normalize(),
                zones=input_zones_for(config, zone),
                out_zones=out_zones_for(config, zone),
                max_evals=max_evals, nlayers=args.nlayers,
                calibration_window=args.calibration_years, quiet=True,
                new_hyperopt=not trials_done)
        print(f"  {time.time() - started:.0f}s -> {path}\n")

    # --- Data ---------------------------------------------------------
    if config == "own":
        from epftoolbox.data import read_data
        with redirect_stdout(io.StringIO()):
            df_train, df_test = read_data(
                path=args.datasets_dir, dataset=name, begin_test_date=begin_test,
                end_test_date=end_test + pd.Timedelta(hours=23))
        data = pd.concat([df_train, df_test])
        if args.data_start:
            data = data.loc[pd.Timestamp(args.data_start):]
        # Cleaning is not this script's job: it happens once in
        # data_cleaning_v2.ipynb, so the DNN and LEAR provably read identically
        # prepared inputs.
        if data.isna().any().any():
            counts = data.isna().sum()
            print(f"error: the dataset has missing values "
                  f"{counts[counts > 0].to_dict()}. The DNN cannot be fitted on "
                  f"NaN, and this script does not impute.", file=sys.stderr)
            return 1
        source = data
        n_exogenous = len(data.columns) - 1
    else:
        source = matrices
        n_exogenous = None

    days = pd.date_range(begin_test, end_test, freq="D")
    day_set = set(days)
    cadence = args.recalibration_days
    refits = RS.refit_days(begin_test, end_test, cadence)
    max_days_per_process = args.max_days_per_process
    run_id_label = f"{config}" + (f"_{zone}" if config not in RS.JOINT_CONFIGS
                                  else "")
    out_zones = out_zones_for(config, zone)
    print(f"[{config}] {len(days)} forecast day(s) from {len(refits)} refit(s) "
          f"(recalibrating every {cadence} day(s)), {len(seeds)} seed(s), "
          f"{args.calibration_years}-year window; "
          f"{expected_input_width(config, zone)} inputs -> "
          f"{24 * len(out_zones)} outputs"
          + (f"; this process stops after {max_days_per_process} new day(s)"
             if max_days_per_process else ""))

    os.makedirs(args.out_dir, exist_ok=True)
    run_dir = run_dir_for(config, zone, begin_test, end_test, args.out_dir,
                          args.smoke)
    os.makedirs(run_dir, exist_ok=True)

    columns = forecast_columns(config, zone)
    prefix = ("forecasts_joint_seed" if config in RS.JOINT_CONFIGS
              else "forecasts_seed")
    forecast_paths = {s: os.path.join(run_dir, f"{prefix}{s}.csv") for s in seeds}
    timing_paths = {s: os.path.join(run_dir, f"timings_seed{s}.csv") for s in seeds}

    # Resume whatever a previous run of the same command finished.
    # _load_checkpoint is shared with the LEAR backtest rather than
    # reimplemented, so both models treat a half-written final row identically:
    # it is dropped and recomputed.
    forecasts = {}
    for seed in seeds:
        frame = pd.DataFrame(index=days, columns=columns, dtype="float64")
        done = _load_checkpoint(forecast_paths[seed])
        if done is not None and len(done):
            common = frame.index.intersection(done.index)
            frame.loc[common, :] = done.loc[common, columns]
        forecasts[seed] = frame

    already = sum(1 for d in days
                  if all(forecasts[s].loc[d].notna().all() for s in seeds))
    if already:
        print(f"Resuming: {already} of {len(days)} day(s) already done for "
              f"every seed")

    models = build_models(config, zone, seeds, hyper_dir, args.nlayers,
                          args.calibration_years)

    # Day outer, seed inner. Every seed advances together, so an interruption
    # always leaves a *balanced* ensemble: all members cover exactly the same
    # days and what has been computed can be scored straight away. Running the
    # seeds one after another instead leaves a ragged set, and build_ensemble
    # intersects the members' indices -- so a single lagging seed would drag the
    # whole ensemble back to its own last finished day.
    #
    # This costs nothing. recalibrate() builds a fresh network for every day
    # regardless. Nor does it change the numbers -- but only because
    # recalibrate_and_forecast_next_day seeds the RNG from (seed, day) before
    # building the features. Without that, the train/validation split is drawn
    # from the unseeded global RNG and the order in which days and seeds ran
    # would change the forecasts.
    timings = []
    zone_weights = {}
    run_started = time.time()
    computed_days = 0

    chunk_days = 0
    chunk_exhausted = False
    sampled_bucket = 0
    memory_path = os.path.join(run_dir, "memory.csv")
    _sample_rss(memory_path, run_id_label, 0, already, "chunk start")

    for n, refit_day in enumerate(refits, 1):
        covered = [d for d in RS.days_served_by(
            refit_day, begin_test, end_test, cadence) if d in day_set]
        # Only fit if some seed still owes one of the days this fit serves.
        pending = {seed: [d for d in covered
                          if not forecasts[seed].loc[d].notna().all()]
                   for seed in seeds}
        if not any(pending.values()):
            continue

        if chunk_exhausted:
            break

        day_started = time.time()
        ran = []
        for seed in seeds:
            need = pending[seed]
            if not need:
                continue

            # Always fit at the grid's refit day, never at the first day that
            # happens to be missing. The schedule is a function of the calendar,
            # so a resumed run refits exactly where an uninterrupted one did and
            # reproduces it -- which is what makes chunking safe.
            predictions, seconds = models[seed].recalibrate_and_forecast_days(
                source, refit_day, need)

            for row, day in enumerate(need):
                forecasts[seed].loc[day, :] = np.asarray(
                    predictions[row], dtype=float).reshape(-1)

            weights = getattr(models[seed], "zone_weights", None)
            if weights:
                zone_weights.setdefault(str(seed), []).append(weights)

            # Checkpoint immediately: a long run must never lose more than one
            # recalibration. Rows for days not yet reached are still all-NaN and
            # are dropped on read, by _load_checkpoint and by the evaluator alike.
            forecasts[seed].to_csv(forecast_paths[seed])

            # One row per forecast day. The fit is charged to the day it was
            # made for and the reused days carry only their forward pass, so the
            # file shows what the cadence actually costs rather than an average
            # that hides it.
            for row, day in enumerate(need):
                is_refit = day == refit_day
                elapsed = (seconds["fit_seconds"] if is_refit else 0.0) \
                    + seconds["predict_seconds"][row]
                timings.append(elapsed)
                _append_timing(timing_paths[seed], {
                    "date": day.isoformat(), "seed": seed,
                    "seconds": round(elapsed, 3),
                    "refit": bool(is_refit),
                    "refit_day": refit_day.isoformat(),
                    "fit_seconds": round(seconds["fit_seconds"], 3)
                                   if is_refit else 0.0,
                    "predict_seconds": round(seconds["predict_seconds"][row], 3),
                    "calibration_window_years": args.calibration_years,
                    "recalibration_days": cadence,
                    "config": config, "model": "DNN",
                })
            ran.append(seed)

        if not ran:
            continue

        # TensorFlow does not hand back everything clear_session() releases;
        # collecting here does not fix that, but it does return what Python can
        # and it costs milliseconds against a fit of tens of seconds.
        gc.collect()

        newly_done = len({d for seed in seeds for d in pending[seed]})
        chunk_days += newly_done
        computed_days += 1
        done_days = sum(1 for d in days
                        if all(forecasts[s].loc[d].notna().all() for s in seeds))
        per_refit = (time.time() - run_started) / computed_days
        remaining_refits = max(len(refits) - n, 0)
        eta = remaining_refits * per_refit
        print(f"  [{n}/{len(refits)} refits, {done_days}/{len(days)} days] "
              f"{refit_day.date()}  seed(s) {','.join(str(s) for s in ran)}  "
              f"{time.time() - day_started:6.1f}s   eta {eta / 3600:5.1f}h")

        # One sample per RSS_SAMPLE_DAYS crossed, not one per day that happens
        # to land near a multiple: a refit adds `cadence` days at once, so a
        # modulo test fires twice at every boundary.
        bucket = chunk_days // RS.RSS_SAMPLE_DAYS
        if bucket > sampled_bucket:
            sampled_bucket = bucket
            _sample_rss(memory_path, run_id_label, chunk_days, done_days,
                        "periodic")

        if max_days_per_process and chunk_days >= max_days_per_process:
            chunk_exhausted = True

    _sample_rss(memory_path, run_id_label, chunk_days,
                sum(1 for d in days
                    if all(forecasts[s].loc[d].notna().all() for s in seeds)),
                "chunk end")

    # Mean first-layer weight magnitude per zone, across the days this process
    # computed. With the block toggles gone for wide/joint this is what stands in
    # for the toggle states: which zones the model actually gave weight to.
    weight_summary = {
        seed: {group: float(np.mean([d[group] for d in daily]))
               for group in daily[0]}
        for seed, daily in zone_weights.items() if daily
    }
    if weight_summary:
        with open(os.path.join(run_dir, "first_layer_zone_weights.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(weight_summary, handle, indent=2)

    manifest = {
        "model": "DNN",
        "config": config,
        "dataset": name,
        "zone": zone if config not in RS.JOINT_CONFIGS else None,
        "input_zones": list(input_zones_for(config, zone)),
        "output_zones": list(out_zones),
        "n_inputs": expected_input_width(config, zone),
        "n_outputs": 24 * len(out_zones),
        "n_exogenous": n_exogenous,
        "test_start": str(begin_test), "test_end": str(end_test),
        "test_days": len(days),
        "seeds": seeds,
        "calibration_window_years": args.calibration_years,
        "recalibration_days": cadence,
        "refits": len(refits),
        "days_per_process": max_days_per_process,
        "nlayers": args.nlayers,
        "hyperopt_evals": max_evals,
        "hyperopt_range": [str(hyperopt_begin.date()), str(hyperopt_end.date())],
        "search_space": _space_summary(config, zone, args.nlayers),
        "data_start": args.data_start,
        "cleaning": "data_cleaning_v2.ipynb",
        "preflight": preflight_results,
        "first_layer_zone_weights": weight_summary or None,
        "seconds_per_recalibration": round(
            sum(timings) / max(len(timings), 1), 1),
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w",
              encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)

    print(f"\nwritten: {run_dir}")

    if chunk_exhausted:
        done_days = sum(1 for d in days
                        if all(forecasts[s].loc[d].notna().all() for s in seeds))
        print(f"\nstopping after {chunk_days} new day(s) in this process; "
              f"{done_days}/{len(days)} done. The scheduler restarts from the "
              f"checkpoint -- this is not a failure.")
        return CHUNK_INCOMPLETE

    if args.no_evaluate:
        return 0

    # Scored by the LEAR evaluator, not a parallel copy of it. The ensemble mean,
    # MAE, rMAE against a weekly naive, the DM/GW tests and predictions.csv are
    # all produced by the same code that scores LEAR, so the two models' figures
    # mean the same thing and can be compared directly.
    #
    # DNN-joint is scored one zone at a time on that zone's own 24-column slice.
    # A pooled loss across zones is never computed for reporting: it would be
    # dominated by whichever zones happen to be easiest.
    from lear_dk1.evaluate import evaluate_run

    targets = ({z: d for z, d in
                write_zone_slices(run_dir, seeds, out_zones).items()}
               if config in RS.JOINT_CONFIGS else {zone: run_dir})
    failures = []
    for scored_zone, directory in targets.items():
        try:
            results = evaluate_run(directory, dataset=Z.dataset_name(scored_zone),
                                   datasets_dir=args.datasets_dir,
                                   zone=scored_zone, kind="seed")
            print(f"  {scored_zone}: {results['predictions']}")
        except ValueError as exc:
            failures.append(f"{scored_zone}: {exc}")
    if failures:
        print("\nnot scored: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


def _existing_trials(hyper_dir, config, zone, nlayers, calibration_years):
    """The trials file for this run, and how many evaluations it already holds.

    Returns ``(path, completed)``. A file that cannot be read counts as zero
    completed evaluations rather than raising: a corrupt checkpoint should cost a
    fresh search, not the whole run.
    """
    import pickle as pc

    from dnn_dk1.forecaster import hyperparameter_path

    path = hyperparameter_path(hyper_dir, 1, nlayers, dataset_name(config, zone),
                               2, 1, 0, calibration_years)
    if not os.path.exists(path):
        return path, 0
    try:
        with open(path, "rb") as handle:
            trials = pc.load(handle)
        return path, sum(1 for loss in trials.losses() if loss is not None)
    except Exception:
        return path, 0


def _space_summary(config, zone, nlayers):
    from dnn_dk1 import hyperopt as dnn_hyperopt

    if config == "own":
        space = dnn_hyperopt.build_space(nlayers, 0, Z.n_exogenous(zone))
    else:
        space = dnn_hyperopt.build_multizone_space(nlayers)
    summary = dnn_hyperopt.space_summary(space)
    summary["binaries_if_generalised_naively"] = (
        dnn_hyperopt.n_binary_toggles_if_generalised())
    return summary


if __name__ == "__main__":
    sys.exit(main())
