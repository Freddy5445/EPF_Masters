"""
Run the LEAR ensemble on every price zone in the cleaned panel, in parallel.

    python run_lear_all_zones.py --smoke      validate the pipeline, ten days
    python run_lear_all_zones.py              the real run

One process per zone, each running that zone's four calibration windows in
sequence. Zones are independent -- separate data, separate models, separate
output files -- so nothing is shared and nothing needs locking.

Specification follows Lago, Marcjasz, De Schutter & Weron (2021):

* **Windows** 56, 84, 1092 and 1456 days, plus the ensemble, which is the
  arithmetic mean of the four forecasts.
* **Rolling**, not expanding: a fixed-length window that slides one day at a
  time, which is what ``recalibrate_and_forecast_next_day`` already does with
  ``df_train.iloc[-calibration_window * 24:]``.
* **Daily recalibration**, lambda re-selected from scratch every day.
* **Two exogenous series** (247 regressors), so the specification matches the
  paper and is identical across zones -- which is what makes the cross-zone
  rMAE table and the DM/GW tests comparable.

``--with-hydro`` runs a second variant that adds reservoir level as a third
exogenous series. Only the Norwegian and Swedish zones have it; Denmark has no
hydro storage, so those zones are skipped for that variant rather than run under
a different specification.

**Test period.** The last 728 days each zone has, which is the paper's 104 weeks.
Zones are cut at their own switch to 15-minute prices, so this is derived per
zone rather than fixed. A zone whose history is too short for the 1456-day window
to be fully available on its first test day gets a shorter test period, and says
so.

**Output.** Under ``--out-dir`` (default ``experiments/``):

* ``predictions_<layout>.csv`` -- **the deliverable.** One row per zone per hour:
  ``timestamp_local, zone, forecast, observed``. The forecast is the ensemble
  mean, which is what LEAR predicts; ``observed`` is the price that actually
  cleared, never imputed.
* ``<zone>_clean_<layout>_<begin>_<end>/`` -- ``predictions.csv`` for that zone
  alone, plus the working state: one forecast file per calibration window,
  per-day timings, ``run_metadata.json`` and ``evaluation.json``. The per-window
  files exist so a run can resume and so the ensemble can be rebuilt. The layout
  is in the directory name because two layouts for one zone would otherwise
  overwrite each other.
* ``summary_<layout>.csv`` -- MAE and rMAE for every zone, the cross-zone table.
* ``logs/<zone>_<layout>.log`` -- what that zone's worker printed.
* ``sweep_results.json`` -- per-zone status and wall clock.

Runs checkpoint per day and per window, so an interrupted sweep can be restarted
with the same command and picks up where it stopped. Budget roughly an hour per
zone for the four-window ensemble over 728 days, divided by ``--workers``.

Commands are given on one line: this project is developed from PowerShell, where
a trailing ``\`` is not a line continuation and truncates the command.
"""

import os

# Pin the BLAS thread pools before NumPy is imported anywhere. Each worker is
# already a whole core's worth of work, and a threaded BLAS underneath N worker
# processes oversubscribes the machine badly -- typically slower than running
# sequentially. Set here rather than in the workers because the children inherit
# this environment, and on Windows they re-import this module from scratch.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import concurrent.futures as futures  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import contextlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import timedelta  # noqa: E402

import pandas as pd  # noqa: E402

import run_lear_dk1  # noqa: E402
import run_lear_from_clean  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PANEL = os.path.join(THIS_DIR, "datasets", "nordic_baltic_clean_hourly.parquet")

# The paper's specification: two exogenous series, 247 regressors.
BASE_LAYOUT = "load-windsolar"
# The same, plus reservoir level. Denmark has no hydro storage and is skipped.
HYDRO_LAYOUT = "load-windsolar-hydro"


def panel_zones(panel_path):
    """Zones present in the cleaned panel, and which of them carry reservoir."""
    panel = pd.read_parquet(panel_path, columns=["zone", "variable"])
    zones = sorted(panel["zone"].astype("string").dropna().unique())
    with_hydro = sorted(
        panel.loc[panel["variable"].astype("string") == "reservoir", "zone"]
        .astype("string").dropna().unique()
    )
    return zones, with_hydro


def run_zone(zone, layout, panel, datasets_dir, out_dir, passthrough, log_dir):
    """Backtest one zone. Runs in its own process; returns a summary dict.

    Output is captured to a per-zone log rather than printed: eleven concurrent
    per-day progress streams interleaved on one console are unreadable, and the
    parent prints a line per zone as it finishes.
    """
    started = time.time()
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{zone}_{layout}.log")

    buffer = io.StringIO()
    status, detail = "ok", None
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = run_lear_from_clean.main(
                ["--panel", panel, "--zone", zone, "--exog", layout,
                 "--datasets-dir", datasets_dir, "--out-dir", out_dir]
                + list(passthrough)
            )
        if code != 0:
            status, detail = "failed", f"exit code {code}"
    except Exception as exc:  # noqa: BLE001 - a zone must not take the sweep down
        status = "failed"
        detail = f"{type(exc).__name__}: {exc}"[:200]
        buffer.write(f"\n{type(exc).__name__}: {exc}\n")

    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write(buffer.getvalue())

    return {
        "zone": zone, "layout": layout, "status": status, "detail": detail,
        "seconds": round(time.time() - started, 1), "log": log_path,
        "run_dir": _run_dir_written(out_dir, zone, layout, started),
    }


def _run_dir_written(out_dir, zone, layout, started):
    """The run directory this worker just wrote, or None.

    The name cannot be reconstructed here: run_lear_from_clean derives the test
    range from the panel, so the dates in it are not known until the run has read
    the data. Picking the alphabetically last matching directory instead is wrong
    the moment a zone has more than one run -- "..._2023-04-01" sorts after
    "..._2023-03-01" whichever was written first -- so identify it by the manifest
    the run itself wrote, taking the newest one touched since the run began.
    """
    # run_lear_dk1 names the directory "<dataset>_<begin>_<end>", so match on the
    # separator too. A bare startswith would let "load-windsolar" claim the
    # "load-windsolar-hydro" directories, and score the wrong specification.
    prefix = run_lear_from_clean.dataset_name_for(zone, layout)
    newest, newest_mtime = None, started - 60  # a little slack for clock coarseness
    for name in os.listdir(out_dir) if os.path.isdir(out_dir) else []:
        if name != prefix and not name.startswith(prefix + "_"):
            continue
        manifest = os.path.join(out_dir, name, "run_metadata.json")
        if not os.path.exists(manifest):
            continue
        mtime = os.path.getmtime(manifest)
        if mtime >= newest_mtime:
            newest, newest_mtime = os.path.join(out_dir, name), mtime
    return newest


def _fmt(seconds):
    return str(timedelta(seconds=int(round(seconds))))


def sweep(zones, layout, panel, datasets_dir, out_dir, log_dir, passthrough,
          workers):
    """Run every zone for one layout, in parallel. Returns the summaries."""
    print(f"\n{'=' * 66}")
    print(f"{layout}: {len(zones)} zone(s) on {workers} worker(s) -- "
          f"{', '.join(zones)}")
    print(f"{'=' * 66}\n")

    results = []
    started = time.time()

    with futures.ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(run_zone, zone, layout, panel, datasets_dir, out_dir,
                        passthrough, log_dir): zone
            for zone in zones
        }
        for number, future in enumerate(futures.as_completed(pending), 1):
            result = future.result()
            results.append(result)
            mark = "ok    " if result["status"] == "ok" else "FAILED"
            note = f"  {result['detail']}" if result["detail"] else ""
            print(f"[{number}/{len(zones)}] {result['zone']:5} {mark} "
                  f"{_fmt(result['seconds']):>9}{note}")

    print(f"\n{layout}: {sum(r['status'] == 'ok' for r in results)}/{len(zones)} "
          f"zones in {_fmt(time.time() - started)} wall clock")
    return results


def evaluate(sweep_results, layout, out_dir, datasets_dir, quiet=False):
    """Score every zone that produced forecasts, and build the cross-zone outputs."""
    from lear_dk1.evaluate import PREDICTIONS_FILE, compare_zones, evaluate_run

    results, predictions = {}, []
    for result in sorted(sweep_results, key=lambda r: r["zone"]):
        if result["status"] != "ok" or not result.get("run_dir"):
            continue
        zone = result["zone"]
        try:
            results[zone] = evaluate_run(
                result["run_dir"],
                dataset=run_lear_from_clean.dataset_name_for(zone, layout),
                datasets_dir=datasets_dir, quiet=True, zone=zone)
        except (ValueError, OSError, KeyError) as exc:
            if not quiet:
                print(f"  {zone}: not scored -- {type(exc).__name__}: {exc}"[:160])
            continue
        predictions.append(pd.read_csv(
            os.path.join(result["run_dir"], PREDICTIONS_FILE),
            parse_dates=["timestamp_local"]))

    if not results:
        return None

    # One tidy table across zones: a row per zone-hour, forecast beside observed.
    # Zones cover different date ranges -- each is cut at its own switch to
    # 15-minute prices -- so this is a long format rather than a wide one, which
    # would have to pad the ends with blanks.
    combined = os.path.join(out_dir, f"predictions_{layout}.csv")
    (pd.concat(predictions, ignore_index=True)
       .sort_values(["zone", "timestamp_local"])
       .to_csv(combined, index=False, date_format="%Y-%m-%dT%H:%M:%S"))

    table = compare_zones(results)
    path = os.path.join(out_dir, f"summary_{layout}.csv")
    table.to_csv(path)

    if not quiet:
        print(f"\nrMAE by zone ({layout}) -- lower is better, 1.0 means no better "
              f"than a weekly naive:")
        columns = [c for c in table.columns if c.startswith("rmae_")]
        print(table[columns].to_string())
        print(f"\nscores:      {path}")
        print(f"predictions: {combined}")
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the LEAR ensemble on every zone in the cleaned panel.")
    parser.add_argument("--panel", default=DEFAULT_PANEL)
    parser.add_argument("--zones", default=None,
                        help="Comma-separated zones (default: every zone in the panel)")
    parser.add_argument("--with-hydro", action="store_true",
                        help="Also run a reservoir-augmented variant on the zones "
                             "that have hydro storage")
    parser.add_argument("--hydro-only", action="store_true",
                        help="Run only the reservoir-augmented variant")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel zones (default: min(zone count, CPU count))")
    parser.add_argument("--datasets-dir", default=os.path.join(THIS_DIR, "datasets"))
    parser.add_argument("--out-dir", default=os.path.join(THIS_DIR, "experiments"))
    parser.add_argument("--log-dir", default=None,
                        help="Per-zone logs (default: <out-dir>/logs)")
    parser.add_argument("--no-evaluate", action="store_true",
                        help="Skip scoring; run the backtests only")
    args, passthrough = parser.parse_known_args(argv)

    if not os.path.exists(args.panel):
        print(f"error: no cleaned panel at {args.panel}", file=sys.stderr)
        print("Run data_cleaning.ipynb to build it.", file=sys.stderr)
        return 1

    available, with_hydro = panel_zones(args.panel)
    zones = ([z.strip().upper() for z in args.zones.split(",") if z.strip()]
             if args.zones else available)

    unknown = [z for z in zones if z not in available]
    if unknown:
        print(f"error: not in the panel: {', '.join(unknown)}. It holds "
              f"{', '.join(available)}.", file=sys.stderr)
        return 1

    log_dir = args.log_dir or os.path.join(args.out_dir, "logs")
    workers = args.workers or min(len(zones), os.cpu_count() or 1)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Panel: {args.panel}")
    print(f"Zones: {', '.join(zones)}")
    print(f"Windows: {', '.join(str(w) for w in run_lear_dk1.DEFAULT_WINDOWS)} "
          f"(Lago et al. 2021) plus their ensemble mean")
    print(f"Logs:  {log_dir}")

    plan = []
    if not args.hydro_only:
        plan.append((BASE_LAYOUT, zones))
    if args.with_hydro or args.hydro_only:
        hydro_zones = [z for z in zones if z in with_hydro]
        skipped = [z for z in zones if z not in with_hydro]
        if skipped:
            print(f"\nNo reservoir data for {', '.join(skipped)}; those zones are "
                  f"skipped in the hydro variant rather than run under a "
                  f"different specification.")
        if hydro_zones:
            plan.append((HYDRO_LAYOUT, hydro_zones))

    if not plan:
        print(f"error: nothing to run. --hydro-only was given, but none of "
              f"{', '.join(zones)} has reservoir data in the panel; the zones that "
              f"do are {', '.join(with_hydro) or 'none'}.", file=sys.stderr)
        return 1

    all_results = {}
    for layout, layout_zones in plan:
        results = sweep(layout_zones, layout, args.panel, args.datasets_dir,
                        args.out_dir, log_dir, passthrough, workers)
        all_results[layout] = results

        if not args.no_evaluate:
            evaluate(results, layout, args.out_dir, args.datasets_dir)

    manifest = os.path.join(args.out_dir, "sweep_results.json")
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)
    print(f"\nPer-zone outcomes: {manifest}")

    failed = [r for results in all_results.values() for r in results
              if r["status"] != "ok"]
    if failed:
        print(f"\n{len(failed)} zone run(s) failed:")
        for result in failed:
            print(f"  {result['zone']:5} {result['layout']:22} {result['detail']}")
            print(f"        log: {result['log']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
