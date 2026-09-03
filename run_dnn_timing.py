"""
Phase 1's deliverable: how long the three DNN configurations actually take.

    python run_dnn_timing.py

This does **not** run a backtest. It measures, and the hyperparameter-search
budget ``E`` and the ensemble size ``S`` for the ten real runs are chosen from
what it prints.

What it measures
================

**A thread sweep** (spec section 7.1). Three recalibrations at
``OMP_NUM_THREADS`` in {1, 2, 4, 8}, one seed -- on DNN-own/DK1 as the spec asks,
and on the other two configurations as well, because the split the sweep picks is
applied to all ten runs and DNN-own/DK1 is the narrowest network of the three.
TensorFlow reads its thread
settings when it initialises, so a setting cannot be changed inside a process
that has already imported it -- each point in the sweep therefore runs in its own
subprocess, with ``OMP_NUM_THREADS``, ``TF_NUM_INTRAOP_THREADS`` and
``TF_NUM_INTEROP_THREADS`` set in the environment *before* Python starts, and
``tf.config.threading.set_{intra,inter}_op_parallelism_threads`` called in the
child immediately after the import and before any op runs.

A thread count does not translate into throughput on its own: what matters is
how many fits per hour ``P`` concurrent processes finish at ``T`` threads each.
The sweep reports that, subject to ``P * T <= logical cores``.

**Per-configuration timings** (section 7.2), for DNN-own/DK1 (313 inputs),
DNN-wide/DK1 (1969 inputs) and DNN-joint (1969 inputs, 168 outputs):

* 20 hyperopt evaluations -- mean *and standard deviation* of seconds per
  evaluation. The spread is the point: evaluation time varies with the sampled
  architecture by more than the mean of 20 draws pins down, and a budget picked
  off the mean alone will be wrong.
* 5 recalibration days, 1 seed -- mean and standard deviation of seconds per
  recalibration.
* peak resident set size.

**A projected wall-clock table** (section 7.3) for E in {300, 1500} and S in
{2, 4}: each individual run, all ten serially, and all ten at the concurrency the
sweep picks.

The ten runs are DNN-own x 7 zones, DNN-wide x {DK1, DK2}, and DNN-joint x 1.

Results land in ``experiments/timing/phase1_timings.json``.

Commands are given on one line: this project is developed from PowerShell, where
a trailing ``\`` is not a line continuation and truncates the command.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import timedelta

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(THIS_DIR, "experiments", "timing")

# --- What is measured ------------------------------------------------------

THREAD_SWEEP = (1, 2, 4, 8)
SWEEP_RECALIBRATIONS = 3
HYPEROPT_EVALS = 20
RECALIBRATION_DAYS = 5
SWEEP_SEED = 1

# The thread count the per-configuration timings are taken at. The working
# hypothesis is 4 processes x 4 threads on an 8-physical / 16-logical machine;
# the sweep is what tests it, and section 7.3 rescales if it disagrees.
MEASURE_THREADS = 4

# The three configurations to time, and the ten runs they stand for.
MEASURED = (("own", "DK1"), ("wide", "DK1"), ("joint", None))

# Section 7.1 sweeps DNN-own/DK1. All three are swept, because the split the
# sweep picks is applied to all ten runs and DNN-own/DK1 is the *narrowest*
# network of the three: at 313 inputs the per-epoch Python and Keras overhead
# dominates the matmuls, so a 313-input fit can look thread-insensitive while a
# 1969-input one does not. Extrapolating from the narrow case would be assuming
# the answer. DNN-own/DK1 stays the reported sweep; where the wider
# configurations peak at a different thread count, the widest wins.
SWEPT = MEASURED

E_CANDIDATES = (300, 1500)
S_CANDIDATES = (2, 4)

TEST_DAYS = 731


def planned_runs():
    """The ten runs, as (config, zone) pairs."""
    from dnn_dk1 import zones as Z

    return ([("own", z) for z in Z.ZONES]
            + [("wide", z) for z in Z.FOCAL_ZONES]
            + [("joint", None)])


def _fmt(seconds):
    return str(timedelta(seconds=int(round(seconds))))


def _stats(values):
    values = [float(v) for v in values]
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 3) if values else None,
        "sd": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
        "min": round(min(values), 3) if values else None,
        "max": round(max(values), 3) if values else None,
        "values": [round(v, 3) for v in values],
    }


# ---------------------------------------------------------------------------
# The child process
# ---------------------------------------------------------------------------

def _peak_rss_bytes():
    """Peak resident set of this process, or None if it cannot be read."""
    try:
        import psutil
    except ImportError:
        return None
    info = psutil.Process().memory_info()
    # Windows exposes the peak working set; elsewhere fall back to current RSS.
    return int(getattr(info, "peak_wset", info.rss))


def _configure_threads(threads):
    """Pin TensorFlow's thread pools. Must run before any op executes."""
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    return {
        "intra_op": tf.config.threading.get_intra_op_parallelism_threads(),
        "inter_op": tf.config.threading.get_inter_op_parallelism_threads(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "TF_NUM_INTRAOP_THREADS": os.environ.get("TF_NUM_INTRAOP_THREADS"),
        "TF_NUM_INTEROP_THREADS": os.environ.get("TF_NUM_INTEROP_THREADS"),
        "tensorflow": tf.__version__,
    }


def worker(args):
    """One measurement, in a process whose threads were pinned before start."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")

    threading_info = _configure_threads(args.threads)

    import pandas as pd

    import run_dnn_dk1 as R
    from dnn_dk1 import hyperopt as dnn_hyperopt
    from dnn_dk1 import zones as Z

    zone = args.zone if args.config != "joint" else Z.ZONES[0]
    name = R.dataset_name(args.config, zone)
    hyper_dir = args.hyperparameter_dir
    begin_test = Z.BEGIN_TEST
    hyperopt_begin, hyperopt_end = R.hyperopt_window(begin_test)

    result = {
        "config": args.config, "zone": zone, "dataset": name,
        "threads": args.threads, "threading": threading_info,
        "n_inputs": R.expected_input_width(args.config, zone),
        "n_outputs": 24 * len(R.out_zones_for(args.config, zone)),
    }

    matrices = None
    if args.config != "own":
        matrices = Z.load_zone_matrices(Z.ZONES, args.datasets_dir)

    # --- hyperopt evaluations -----------------------------------------
    if args.evals:
        record = []
        started = time.time()
        if args.config == "own":
            dnn_hyperopt.optimize(
                path_datasets_folder=args.datasets_dir,
                path_hyperparameters_folder=hyper_dir, dataset=name,
                begin_test_date=hyperopt_begin, end_test_date=hyperopt_end,
                max_evals=args.evals, nlayers=args.nlayers,
                calibration_window=Z.CALIBRATION_YEARS, quiet=True, record=record)
        else:
            dnn_hyperopt.optimize_multizone(
                matrices=matrices, path_hyperparameters_folder=hyper_dir,
                dataset=name, begin_test_date=hyperopt_begin,
                end_test_date=hyperopt_end.normalize(),
                zones=R.input_zones_for(args.config, zone),
                out_zones=R.out_zones_for(args.config, zone),
                max_evals=args.evals, nlayers=args.nlayers,
                calibration_window=Z.CALIBRATION_YEARS, quiet=True, record=record)
        result["hyperopt"] = _stats([r["seconds"] for r in record])
        result["hyperopt"]["wall_seconds"] = round(time.time() - started, 3)
        result["hyperopt"]["architectures"] = [r["neurons"] for r in record]

    # --- recalibrations -----------------------------------------------
    if args.days:
        models = R.build_models(args.config, zone, [args.seed], hyper_dir,
                                args.nlayers, Z.CALIBRATION_YEARS)
        model = models[args.seed]

        if args.config == "own":
            from contextlib import redirect_stdout
            import io as _io
            from epftoolbox.data import read_data

            end = begin_test + pd.Timedelta(days=args.days - 1)
            with redirect_stdout(_io.StringIO()):
                df_train, df_test = read_data(
                    path=args.datasets_dir, dataset=name,
                    begin_test_date=begin_test,
                    end_test_date=end + pd.Timedelta(hours=23))
            source = pd.concat([df_train, df_test])
            if source.isna().any().any():
                raise ValueError(f"{name} holds NaN; nothing here imputes")
        else:
            source = matrices

        seconds = []
        for day in pd.date_range(begin_test, periods=args.days, freq="D"):
            started = time.time()
            model.recalibrate_and_forecast_next_day(source, day)
            seconds.append(time.time() - started)
        result["recalibration"] = _stats(seconds)
        weights = getattr(model, "zone_weights", None)
        if weights:
            result["first_layer_zone_weights"] = weights

    result["peak_rss_bytes"] = _peak_rss_bytes()
    with open(args.result, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return 0


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

def spawn(out_dir, tag, threads, reuse=True, **kwargs):
    """Run one worker in a fresh process with its thread pools pinned.

    The environment is set here, before the child's Python starts, because
    TensorFlow reads these at import: setting them after ``import tensorflow``
    has no effect at all.

    A measurement already on disk is reused unless ``reuse`` is false, so a
    harness that was interrupted -- or extended with a new sweep -- does not
    re-measure what it already has.
    """
    result_path = os.path.join(out_dir, f"measurement_{tag}.json")
    if reuse and os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as handle:
            result = json.load(handle)
        result["reused"] = True
        return result

    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": str(threads),
        "TF_NUM_INTRAOP_THREADS": str(threads),
        "TF_NUM_INTEROP_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "KERAS_BACKEND": "tensorflow",
        "PYTHONWARNINGS": "ignore",
    })
    command = [sys.executable, os.path.abspath(__file__), "--worker",
               "--threads", str(threads), "--result", result_path]
    for key, value in kwargs.items():
        if value is None or value is False:
            continue
        command += [f"--{key.replace('_', '-')}", str(value)]

    started = time.time()
    completed = subprocess.run(command, cwd=THIS_DIR, env=env,
                               capture_output=True, text=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout[-4000:])
        sys.stderr.write(completed.stderr[-4000:])
        raise RuntimeError(f"measurement {tag} failed (exit {completed.returncode})")

    with open(result_path, encoding="utf-8") as handle:
        result = json.load(handle)
    result["process_wall_seconds"] = round(time.time() - started, 3)
    return result


def environment_metadata():
    import numpy as np
    import pandas as pd

    logical = os.cpu_count()
    physical = logical
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or logical
        total_ram = int(psutil.virtual_memory().total)
    except ImportError:
        total_ram = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical_cores": logical,
        "physical_cores": physical,
        "total_ram_bytes": total_ram,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def concurrency_options(sweep, logical_cores, run_peak_rss=None):
    """Fits per hour for ``P`` processes at ``T`` threads, from the sweep.

    ``P`` is capped so that ``P * T`` does not exceed the logical core count:
    oversubscribing beyond that trades a faster single fit for a slower machine.

    ``fits_per_hour`` is ``P`` times a *single* process's rate, measured on an
    otherwise idle machine. It is an upper bound, not a prediction: ``P``
    concurrent processes share 8 physical cores and one memory bus, so real
    throughput lands below it. It is still the right quantity for *ranking*
    thread counts, which is what the sweep is for.

    ``run_peak_rss`` is the peak resident set of a full measurement for this
    configuration -- hyperopt included. The sweep's own workers skip hyperopt and
    so use less, which would understate what ``P`` concurrent real runs need.
    """
    options = []
    for threads, measurement in sorted(sweep.items()):
        seconds = measurement["recalibration"]["mean"]
        processes = max(1, logical_cores // threads)
        options.append({
            "threads": threads,
            "seconds_per_fit": seconds,
            "speedup_vs_1_thread": round(
                sweep[1]["recalibration"]["mean"] / seconds, 3)
            if 1 in sweep else None,
            "processes": processes,
            # An upper bound, per the note above -- named plainly because it is
            # what the ranking uses, and qualified everywhere it is printed.
            "fits_per_hour": round(3600.0 * processes / seconds, 1),
            "peak_rss_bytes": run_peak_rss or measurement.get("peak_rss_bytes"),
            "peak_rss_source": "full run" if run_peak_rss else "sweep worker",
        })
    return options


def project(per_config, runs, best, thread_scaling):
    """The section 7.3 table: projected wall clock for each (E, S) candidate.

    ``thread_scaling`` rescales the per-configuration timings, which are taken at
    :data:`MEASURE_THREADS`, to the thread count the sweep picked. It is measured
    on DNN-own/DK1 only and applied to the other two configurations, which is an
    approximation and is labelled as one -- a 1969-input network need not
    parallelise exactly like a 313-input one.
    """
    table = []
    for evals in E_CANDIDATES:
        for seeds in S_CANDIDATES:
            per_run, serial = {}, 0.0
            for config, zone in runs:
                timing = per_config[config]
                seconds = (evals * timing["hyperopt"]["mean"]
                           + TEST_DAYS * seeds * timing["recalibration"]["mean"])
                per_run[f"{config}/{zone or 'Z'}"] = round(seconds, 1)
                serial += seconds
            at_threads = serial * thread_scaling
            table.append({
                "E": evals,
                "S": seeds,
                "per_run_seconds": per_run,
                "serial_seconds": round(serial, 1),
                "serial": _fmt(serial),
                "concurrency": f"{best['processes']}x{best['threads']}",
                "concurrent_seconds": round(at_threads / best["processes"], 1),
                "concurrent": _fmt(at_threads / best["processes"]),
            })
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure the three DNN configurations. Runs no backtest.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", default="own")
    parser.add_argument("--zone", default="DK1")
    parser.add_argument("--threads", type=int, default=MEASURE_THREADS)
    parser.add_argument("--evals", type=int, default=0)
    parser.add_argument("--days", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SWEEP_SEED)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--result", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--datasets-dir",
                        default=os.path.join(THIS_DIR, "datasets"))
    parser.add_argument("--hyperparameter-dir", default=None)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args(argv)

    if args.worker:
        return worker(args)

    os.makedirs(args.out_dir, exist_ok=True)
    hyper_dir = args.hyperparameter_dir or os.path.join(
        args.out_dir, "hyperparameters")
    os.makedirs(hyper_dir, exist_ok=True)

    import pandas as pd  # noqa: F401  (imported for the version record)

    import run_dnn_dk1 as R
    from dnn_dk1 import zones as Z

    environment = environment_metadata()
    print("Phase 1 timing harness -- no backtest is run.")
    print(f"  {environment['platform']}, {environment['physical_cores']} "
          f"physical / {environment['logical_cores']} logical cores, "
          f"{(environment['total_ram_bytes'] or 0) / 1e9:.0f} GB RAM\n")

    # --- Pre-flight, once, before anything is timed -------------------
    preflight_results = []
    if not args.skip_preflight:
        preflight_results, _ = R.preflight(
            "joint", "DK1", args.datasets_dir, Z.BEGIN_TEST, Z.END_TEST)
        if any(not r["passed"] for r in preflight_results):
            print("\nerror: refusing to measure -- pre-flight failed",
                  file=sys.stderr)
            return 1

    started = time.time()

    # --- 7.2 Per-configuration timings --------------------------------
    print(f"\nPer-configuration timings at {MEASURE_THREADS} threads: "
          f"{HYPEROPT_EVALS} hyperopt evaluations, then {RECALIBRATION_DAYS} "
          f"recalibrations, 1 seed")
    print("-" * 78)
    per_config = {}
    for config, zone in MEASURED:
        tag = f"{config}_{zone or 'Z'}_t{MEASURE_THREADS}"
        print(f"  {config:>5} ... ", end="", flush=True)
        measurement = spawn(
            args.out_dir, tag, MEASURE_THREADS, config=config,
            zone=zone or "DK1", evals=HYPEROPT_EVALS, days=RECALIBRATION_DAYS,
            seed=SWEEP_SEED, datasets_dir=args.datasets_dir,
            hyperparameter_dir=hyper_dir)
        per_config[config] = measurement
        hyp, rec = measurement["hyperopt"], measurement["recalibration"]
        print(f"{measurement['n_inputs']:>5} in -> "
              f"{measurement['n_outputs']:>3} out   "
              f"hyperopt {hyp['mean']:7.2f} +/- {hyp['sd']:6.2f} s/eval   "
              f"recalibration {rec['mean']:7.2f} +/- {rec['sd']:5.2f} s   "
              f"peak RSS {(measurement['peak_rss_bytes'] or 0) / 1e9:5.2f} GB")

    # --- 7.1 Thread sweep ---------------------------------------------
    print(f"\nThread sweep: {SWEEP_RECALIBRATIONS} recalibrations per setting, "
          f"1 seed")
    print("-" * 78)
    sweeps = {}
    for config, zone in SWEPT:
        sweeps[config] = {}
        for threads in THREAD_SWEEP:
            label = f"{config}/{zone or 'Z'}"
            print(f"  {label:>10}  OMP_NUM_THREADS={threads:>2} ... ",
                  end="", flush=True)
            sweeps[config][threads] = spawn(
                args.out_dir, f"sweep_{config}_t{threads}", threads,
                config=config, zone=zone or "DK1", days=SWEEP_RECALIBRATIONS,
                seed=SWEEP_SEED, datasets_dir=args.datasets_dir,
                hyperparameter_dir=hyper_dir)
            rec = sweeps[config][threads]["recalibration"]
            print(f"{rec['mean']:7.2f} +/- {rec['sd']:5.2f} s per recalibration")

    options_by_config = {
        config: concurrency_options(
            sweep, environment["logical_cores"],
            run_peak_rss=(per_config[config].get("peak_rss_bytes")
                          if config in per_config else None))
        for config, sweep in sweeps.items()}
    options = options_by_config["own"]

    print(f"\n  {'config':>7}  {'threads':>7}  {'s/fit':>8}  {'speedup':>8}  "
          f"{'processes':>9}  {'fits/h max':>10}  {'run RSS x P':>13}")
    best_by_config = {}
    for config, config_options in options_by_config.items():
        best_by_config[config] = max(config_options,
                                     key=lambda o: o["fits_per_hour"])
        for option in config_options:
            rss = (option["peak_rss_bytes"] or 0) * option["processes"] / 1e9
            mark = "  <--" if option is best_by_config[config] else ""
            print(f"  {config:>7}  {option['threads']:>7}  "
                  f"{option['seconds_per_fit']:>8.2f}  "
                  f"{option['speedup_vs_1_thread'] or 0:>8.2f}  "
                  f"{option['processes']:>9}  {option['fits_per_hour']:>10.1f}  "
                  f"{rss:>10.1f} GB{mark}")

    # DNN-own/DK1 is the sweep section 7.1 asks for, and it is the one reported.
    # The split, though, is applied to all ten runs, so where the wider
    # configurations disagree the widest thread count wins: a split that starves
    # the 1969-input runs costs more than one that under-uses threads on the
    # 313-input ones.
    best = best_by_config["own"]
    disagree = {c: o["threads"] for c, o in best_by_config.items()
                if o["threads"] != best["threads"]}
    if disagree:
        print(f"  note: {disagree} peak at a different thread count from "
              f"DNN-own/DK1's {best['threads']}; taking the widest")
        best = max(best_by_config.values(), key=lambda o: o["threads"])

    # Peak RSS is the other constraint on P, and the thread sweep cannot see it:
    # 16 concurrent DNN-joint processes need more memory than this machine has.
    heaviest = max((m.get("peak_rss_bytes") or 0) for m in per_config.values())
    ram = environment["total_ram_bytes"] or 0
    if ram and heaviest * best["processes"] > 0.8 * ram:
        affordable = max(1, int(0.8 * ram // heaviest))
        print(f"  note: {best['processes']} concurrent processes would need "
              f"{heaviest * best['processes'] / 1e9:.0f} GB at DNN-joint's peak "
              f"RSS, against {ram / 1e9:.0f} GB installed; capping at "
              f"{affordable}")
        best = dict(best, processes=affordable, capped_by="memory",
                    fits_per_hour=round(
                        3600.0 * affordable / best["seconds_per_fit"], 1))

    sweep = sweeps["own"]
    thread_scaling = (sweep[best["threads"]]["recalibration"]["mean"]
                      / sweep[MEASURE_THREADS]["recalibration"]["mean"])

    # --- 7.3 Projection -----------------------------------------------
    runs = planned_runs()
    table = project(per_config, runs, best, thread_scaling)

    print(f"\nProjected wall clock for the ten runs "
          f"({TEST_DAYS} test days, daily recalibration)")
    print("=" * 78)
    print(f"  {'E':>5}  {'S':>2}  {'own/zone':>10}  {'wide/zone':>10}  "
          f"{'joint':>10}  {'10 serial':>11}  "
          f"{'10 at ' + best_label(best):>13}")
    for row in table:
        print(f"  {row['E']:>5}  {row['S']:>2}  "
              f"{_fmt(row['per_run_seconds']['own/DK1']):>10}  "
              f"{_fmt(row['per_run_seconds']['wide/DK1']):>10}  "
              f"{_fmt(row['per_run_seconds']['joint/Z']):>10}  "
              f"{row['serial']:>11}  {row['concurrent']:>13}")
    print("=" * 78)
    if abs(thread_scaling - 1.0) > 1e-9:
        print(f"  the concurrent column rescales the {MEASURE_THREADS}-thread "
              f"timings by {thread_scaling:.3f} to {best['threads']} threads, "
              f"measured on DNN-own/DK1 only")
    print(f"  the concurrent column is a LOWER BOUND on wall clock: it divides "
          f"the serial figure by {best['processes']}, which assumes perfect "
          f"scaling across {best['processes']} processes on "
          f"{environment['physical_cores']} physical cores. Budget for more.")

    payload = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "environment": environment,
        "harness": {
            "hyperopt_evaluations": HYPEROPT_EVALS,
            "recalibration_days": RECALIBRATION_DAYS,
            "sweep_recalibrations": SWEEP_RECALIBRATIONS,
            "swept_configurations": [c for c, _ in SWEPT],
            "measure_threads": MEASURE_THREADS,
            "thread_sweep": list(THREAD_SWEEP),
            "seed": SWEEP_SEED,
            "test_days": TEST_DAYS,
            "runs": [{"config": c, "zone": z} for c, z in runs],
        },
        "preflight": preflight_results,
        "per_configuration": per_config,
        "thread_sweep": {str(k): v for k, v in sweep.items()},
        "thread_sweep_all_configs": {
            config: {str(k): v for k, v in points.items()}
            for config, points in sweeps.items()},
        "concurrency_options": options,
        "concurrency_options_all_configs": options_by_config,
        "chosen_concurrency": best,
        "thread_scaling_applied": round(thread_scaling, 4),
        "projection": table,
        "harness_wall_seconds": round(time.time() - started, 1),
    }
    path = os.path.join(args.out_dir, "phase1_timings.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwritten: {path}")
    print(f"harness runtime: {_fmt(time.time() - started)}")
    return 0


def best_label(best):
    return f"{best['processes']}x{best['threads']}"


if __name__ == "__main__":
    sys.exit(main())
