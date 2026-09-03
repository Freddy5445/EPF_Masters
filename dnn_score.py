"""
Score whichever phase-2 runs have finished, into one table beside LEAR's.

    python dnn_score.py            # score every complete run, rewrite the table
    python dnn_score.py --partial  # also score runs still in progress

Written to be run repeatedly while the set is still going: a run is scored as it
completes rather than after all ten, so partial results are usable days before
the last run lands.

``experiments/dnn_phase2/accuracy_summary.csv`` carries the same columns as
``experiments/lear_dk1_dk2_thesis/accuracy_summary.csv`` -- zone, model, mae,
rmse, rmae, hours_scored -- so DNN and LEAR rows can be concatenated and read as
one table. ``model`` names the configuration and the ensemble member:
``dnn_own``, ``dnn_own_seed1``, ``dnn_wide``, ``dnn_joint``, and so on.

Every number comes from ``lear_dk1.evaluate.evaluate_run``, the same scorer that
produced the LEAR figures. DNN-joint is passed to it one zone at a time, on that
zone's own 24-column slice; a pooled figure across zones is never computed,
because it would be dominated by whichever zones happen to be easiest.

Read-only with respect to a running backtest: it reads the checkpoints and writes
only into ``experiments/dnn_phase2/``, plus the ``predictions.csv`` /
``evaluation.json`` the evaluator itself writes inside each run directory.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(THIS_DIR, "experiments")
PHASE2_DIR = os.path.join(DEFAULT_OUT, "dnn_phase2")
SUMMARY = os.path.join(PHASE2_DIR, "accuracy_summary.csv")

COLUMNS = ["zone", "model", "mae", "rmse", "rmae", "hours_scored"]


def _rmse(run_dir, zone, dataset, datasets_dir, member_frames, index):
    """RMSE, which the shared evaluator does not report but LEAR's table does."""
    from lear_dk1.evaluate import HOURS, real_prices

    real = real_prices(dataset, datasets_dir, index)[HOURS].to_numpy(float)
    out = {}
    for name, frame in member_frames.items():
        predicted = frame.reindex(index)[HOURS].to_numpy(float)
        mask = np.isfinite(real) & np.isfinite(predicted)
        out[name] = float(np.sqrt(np.mean((predicted[mask] - real[mask]) ** 2)))
    return out


def score_run(run, out_dir, datasets_dir):
    """Score one finished run. Returns rows in the LEAR table's schema.

    DNN-joint is scored per zone, on the ``zone_<Z>/`` slices ``run_dnn_dk1.py``
    writes; the others are scored on the run directory itself.
    """
    import run_dnn_dk1 as R
    from dnn_dk1 import runs as RS
    from lear_dk1.evaluate import HOURS, build_ensemble, evaluate_run, load_forecasts
    from dnn_dk1 import zones as Z

    run_dir = R.run_dir_for(run.config, run.focal, RS.BEGIN_TEST, RS.END_TEST,
                            out_dir, smoke=False)
    if not os.path.isdir(run_dir):
        return [], f"no run directory at {run_dir}"

    if run.config == "joint":
        # Refresh the per-zone slices from the joint checkpoint, so scoring a
        # run in progress sees the days it has actually finished.
        targets = R.write_zone_slices(run_dir, list(RS.SEEDS), Z.ZONES)
    else:
        targets = {run.focal: run_dir}

    rows, notes = [], []
    for zone, directory in targets.items():
        dataset = Z.dataset_name(zone)
        try:
            results = evaluate_run(directory, dataset=dataset,
                                   datasets_dir=datasets_dir, zone=zone,
                                   kind="seed", quiet=True)
        except (ValueError, FileNotFoundError) as exc:
            notes.append(f"{zone}: {exc}")
            continue

        forecasts, label = load_forecasts(directory, kind="seed")
        ensemble = build_ensemble(forecasts)
        members = {"ensemble": ensemble}
        members.update({f"{label}{m}": f.loc[ensemble.index, HOURS]
                        for m, f in sorted(forecasts.items())})
        rmse = _rmse(directory, zone, dataset, datasets_dir, members,
                     ensemble.index)

        prefix = f"dnn_{run.config}"
        for name, scores in results["scores"].items():
            rows.append({
                "zone": zone,
                "model": prefix if name == "ensemble" else f"{prefix}_{name}",
                "mae": round(scores["mae"], 4),
                "rmse": round(rmse.get(name, float("nan")), 4),
                "rmae": round(scores["rmae"], 5),
                "hours_scored": scores["hours_scored"],
            })
        notes.append(f"{zone}: {results['forecast_days']} days, "
                     f"MAE {results['scores']['ensemble']['mae']:.3f}, "
                     f"rMAE {results['scores']['ensemble']['rmae']:.4f}")
    return rows, "; ".join(notes)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score the finished phase-2 runs into one table.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--datasets-dir",
                        default=os.path.join(THIS_DIR, "datasets"))
    parser.add_argument("--summary", default=SUMMARY)
    parser.add_argument("--partial", action="store_true",
                        help="Score runs still in progress, on the days they "
                             "have finished. Those rows are NOT comparable with "
                             "a complete run's -- fewer days, different days.")
    parser.add_argument("--only", default=None,
                        help="Comma-separated run IDs (default: all ten)")
    args = parser.parse_args(argv)

    import dnn_status
    from dnn_dk1 import runs as RS

    wanted = ([RS.get(r.strip()) for r in args.only.split(",")]
              if args.only else list(RS.RUNS))
    states = {s["run_id"]: s for s in dnn_status.collect(args.out_dir)}

    os.makedirs(PHASE2_DIR, exist_ok=True)
    all_rows, scored, skipped = [], [], []
    for run in wanted:
        state = states.get(run.run_id, {})
        complete = state.get("state") == "done"
        if not complete and not args.partial:
            skipped.append(f"{run.run_id} ({state.get('state', 'unknown')}, "
                           f"{state.get('days_done', 0)}/{RS.TEST_DAYS})")
            continue
        rows, note = score_run(run, args.out_dir, args.datasets_dir)
        if rows:
            for row in rows:
                row["run_id"] = run.run_id
                row["complete"] = complete
            all_rows += rows
            scored.append(f"{run.run_id}: {note}")
        else:
            skipped.append(f"{run.run_id} ({note})")

    if not all_rows:
        print("nothing to score yet.")
        for line in skipped:
            print(f"  skipped {line}")
        return 0

    table = pd.DataFrame(all_rows)[COLUMNS + ["run_id", "complete"]]
    table.to_csv(args.summary, index=False)

    print(table[COLUMNS].to_string(index=False))
    print(f"\nwritten: {args.summary}")
    for line in scored:
        print(f"  scored  {line}")
    for line in skipped:
        print(f"  skipped {line}")
    if args.partial:
        print("\nnote: --partial rows cover only the days finished so far and "
              "are not comparable across runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
