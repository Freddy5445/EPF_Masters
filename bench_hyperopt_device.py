"""Time hyperopt evaluations for one configuration on one device.

Run once per device in a *fresh process*: TensorFlow decides what it can see at
import time, so CPU-only has to be arranged before anything imports it.

    python bench_hyperopt_device.py --config own  --device gpu --draws 6
    python bench_hyperopt_device.py --config own  --device cpu --draws 6

What is timed is one call to ``dnn_dk1.hyperopt._objective`` (own) or
``_multizone_objective`` (wide/joint) -- the same function the real search
calls, on the same window, with the same data. Not a proxy for an evaluation:
the evaluation.

Per-evaluation time varies by an order of magnitude across architectures -- a
1000-neuron net that early-stops at 300 epochs against a 60-neuron one that stops
at 40 -- so timing *one* evaluation measures the draw, not the device. The
architectures to time are therefore fixed in advance, and there are two ways to
fix them:

``--replay <trials file>`` (preferred) replays actual trials from a completed
search. Each trial in those pickles records the seconds it took **on the local
machine**, so this pairs Colab against the laptop trial for trial, on the search's
own distribution of architectures, with no local re-run needed.

Otherwise draws are sampled from the search space with a fixed seed. Still paired
across devices, but drawn from the prior rather than from what TPE actually
explored -- uniform draws include many degenerate architectures that early-stop
almost immediately, so their mean cost understates a real search's.

Nothing is written to the real hyperparameters folder: the objective's own
checkpoint goes to a scratch file, and the trials object is thrown away.
"""
import argparse
import json
import os
import sys
import time

# --- everything that must precede `import tensorflow` --------------------
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
_pre.add_argument("--threads", type=int, default=0,
                  help="intra/inter-op threads; 0 = leave TensorFlow's default")
_known, _ = _pre.parse_known_args()

if _known.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
if _known.threads:
    os.environ["OMP_NUM_THREADS"] = str(_known.threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(_known.threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(_known.threads)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
from hyperopt import Trials                                 # noqa: E402
from hyperopt.pyll.stochastic import sample                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "epftoolbox"))

import tensorflow as tf                                     # noqa: E402
from dnn_dk1 import hyperopt as H                           # noqa: E402
from dnn_dk1 import zones as Z                              # noqa: E402
from dnn_dk1.model import DNNModel                          # noqa: E402

# 363 days back from the last day before the test period -- run_dnn_dk1's
# `hyperopt_window`, restated so this script does not depend on argparse
# defaults there. assert_search_window_precedes_test re-checks it anyway.
HYPEROPT_DAYS = 363


# --- record how many epochs each fit actually ran ------------------------
# Seconds per evaluation confounds device speed with epoch count: the same draw
# can early-stop at a different epoch on two devices, because the float
# arithmetic differs. Seconds per epoch is the quantity that transfers.
_EPOCHS = []


class _CountingDNNModel(DNNModel):
    def fit(self, trainX, trainY, valX, valY):
        self._n_epochs = 0
        original = self._obtain_metrics

        def counted(X, Y):
            self._n_epochs += 1
            return original(X, Y)

        self._obtain_metrics = counted
        try:
            super().fit(trainX, trainY, valX, valY)
        finally:
            _EPOCHS.append(self._n_epochs)


H.DNNModel = _CountingDNNModel   # what _objective / _multizone_objective build


def _sample_draws(space, seed, n):
    """``n`` draws from ``space``, reproducibly.

    hyperopt's pyll sampler wants whatever numpy RNG the installed version was
    written against: releases up to 0.2.x call ``rng.randint`` (legacy
    ``RandomState``), later ones ``rng.integers`` (``Generator``). Both are
    tried rather than pinned, so the same seed gives the same draws on the
    laptop and on Colab whichever is installed. Fails loudly if neither works --
    silently falling back to an unseeded RNG would unpair the comparison, which
    is the one thing this script exists to guarantee.
    """
    errors = []
    for factory in (lambda: np.random.default_rng(seed),
                    lambda: np.random.RandomState(seed)):
        rng = factory()
        try:
            return [sample(space, rng=rng) for _ in range(n)]
        except (AttributeError, TypeError) as exc:
            errors.append(f"{type(rng).__name__}: {exc}")
    raise RuntimeError("hyperopt's sampler accepted neither numpy RNG type: "
                       + "; ".join(errors))


def _replay_draws(space, path, seed, n):
    """``n`` hyperparameter dicts replayed from a completed search.

    Returns ``(draws, local_seconds)``. ``local_seconds[i]`` is what that exact
    evaluation cost on the machine that ran the search -- both objectives record
    it in the trial result -- so the comparison needs no second local run.

    The sample is drawn without replacement at a fixed seed, so it is the same
    set of trials on every device, and its mean is an unbiased estimate of the
    search's mean evaluation cost (evenly-spaced ranks would over-weight the
    tails).
    """
    import pickle as pc
    from hyperopt import space_eval

    with open(path, "rb") as handle:
        trials = pc.load(handle)

    ok = [t for t in trials.trials
          if t.get("result", {}).get("status") == "ok"
          and t["result"].get("seconds") is not None]
    if not ok:
        raise RuntimeError(f"{path} holds no completed trial with a recorded "
                           f"duration; use prior sampling instead (drop --replay)")

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ok), size=min(n, len(ok)), replace=False)

    draws, local = [], []
    for i in pick:
        vals = {k: v[0] for k, v in ok[i]["misc"]["vals"].items() if v}
        draws.append(space_eval(space, vals))
        local.append(float(ok[i]["result"]["seconds"]))
    return draws, local, len(ok)


def own_setup(args):
    from epftoolbox.data import read_data

    begin, end = args.window
    H.assert_search_window_precedes_test(begin, end, args.dataset)
    dfTrain, dfTest = read_data(dataset=args.dataset, years_test=2,
                                path=args.datasets_dir,
                                begin_test_date=begin, end_test_date=end)
    n_exo = len(dfTrain.columns) - 1
    space = H.build_space(args.nlayers, 0, n_exo)
    kwargs = dict(nlayers=args.nlayers, dfTrain=dfTrain, dfTest=dfTest,
                  shuffle_train=1, dataset=args.dataset, data_augmentation=0,
                  calibration_window=args.calibration_years,
                  n_exogenous_inputs=n_exo)
    return H._objective, space, kwargs, {"n_exogenous": n_exo}


def multizone_setup(args):
    begin, end = args.window
    H.assert_search_window_precedes_test(begin, end, args.dataset)
    matrices = Z.load_zone_matrices(Z.ZONES, args.datasets_dir)

    out_zones = (args.zone,) if args.config == "wide" else Z.ZONES
    begin_n = pd.Timestamp(begin).normalize()
    end_n = pd.Timestamp(end).normalize()
    days = Z.available_days(matrices)
    test_days = days[(days >= begin_n) & (days <= end_n)]
    train_days = Z.training_days(matrices, begin_n, args.calibration_years)

    space = H.build_multizone_space(args.nlayers)
    kwargs = dict(nlayers=args.nlayers, matrices=matrices,
                  train_days=train_days, test_days=test_days, zones=Z.ZONES,
                  out_zones=out_zones)
    return (H._multizone_objective, space, kwargs,
            {"out_zones": list(out_zones), "n_train_days": int(len(train_days)),
             "n_test_days": int(len(test_days))})


def main():
    p = argparse.ArgumentParser(parents=[_pre])
    p.add_argument("--config", required=True, choices=["own", "wide", "joint"])
    p.add_argument("--zone", default="DK1")
    p.add_argument("--datasets-dir", default=os.path.join(HERE, "datasets"))
    p.add_argument("--draws", type=int, default=6)
    p.add_argument("--replay", default=None,
                   help="a completed trials pickle to replay evaluations from. "
                        "Pairs against the local seconds recorded in it, on the "
                        "search's own distribution of architectures.")
    p.add_argument("--draw-seed", type=int, default=20260904,
                   help="fixes WHICH architectures are timed; keep it identical "
                        "across devices or the comparison is not paired")
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--calibration-years", type=int, default=4)
    p.add_argument("--begin-test", default=str(Z.BEGIN_TEST.date()))
    p.add_argument("--out", default=None)
    p.add_argument("--label", default="")
    args = p.parse_args()

    begin_test = pd.Timestamp(args.begin_test)
    end = begin_test.normalize() - pd.Timedelta(hours=1)
    args.window = (end.normalize() - pd.Timedelta(days=HYPEROPT_DAYS), end)

    if args.config == "own":
        args.dataset = Z.dataset_name(args.zone)
        objective, space, kwargs, extra = own_setup(args)
    else:
        args.dataset = ("dnnjoint" if args.config == "joint"
                        else f"dnnwide_{args.zone}")
        objective, space, kwargs, extra = multizone_setup(args)

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:                      # a search process should not seize 15 GB
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    env = {
        "label": args.label,
        "config": args.config,
        "zone": args.zone,
        "dataset": args.dataset,
        "device_requested": args.device,
        "gpus_visible": [g.name for g in gpus],
        "gpu_details": ([tf.config.experimental.get_device_details(g) for g in gpus]
                        if gpus else []),
        "tensorflow": tf.__version__,
        "threads_requested": args.threads or None,
        "cpu_count": os.cpu_count(),
        "search_window": [str(args.window[0].date()), str(args.window[1])],
        "draw_seed": args.draw_seed,
        **extra,
    }
    print(json.dumps({"environment": env}, default=str), flush=True)

    if args.device == "gpu" and not gpus:
        print("!! asked for the GPU and TensorFlow sees none -- "
              "Runtime > Change runtime type > GPU, then Restart runtime.",
              flush=True)

    if args.replay:
        replayed, local_seconds, n_ok = _replay_draws(
            space, args.replay, args.draw_seed, args.draws)
        # A warm-up in front, which is not one of the replayed trials and is
        # never scored: it pays for CUDA context creation and the first graph
        # trace, which a 1000-evaluation search pays once.
        draws = _sample_draws(space, args.draw_seed, 1) + replayed
        local_seconds = [None] + local_seconds
        env["source"] = {"replay": os.path.basename(args.replay),
                         "completed_trials_in_file": n_ok}
    else:
        draws = _sample_draws(space, args.draw_seed, args.draws + 1)
        local_seconds = [None] * len(draws)
        env["source"] = {"prior_sampling": True}
    print(json.dumps({"source": env["source"]}), flush=True)

    scratch = os.path.join("/tmp" if os.path.isdir("/tmp") else HERE,
                           f"bench_trials_{args.config}_{args.device}")
    rows = []
    for i, hp in enumerate(draws):
        # Draw 0 is always the warm-up and is never scored.
        trials = Trials()
        _EPOCHS.clear()
        started = time.time()
        try:
            result = objective(hp, trials=trials, trials_file_path=scratch,
                               max_evals=args.draws, quiet=True, **kwargs)
            loss = float(result["loss"])
            failed = False
        except Exception as exc:                          # noqa: BLE001
            loss, failed = float("nan"), repr(exc)
        seconds = time.time() - started
        epochs = _EPOCHS[-1] if _EPOCHS else None

        neurons = [int(hp["neurons" + str(k)]) for k in range(1, args.nlayers + 1)
                   if int(hp["neurons" + str(k)]) >= 50]
        row = {
            "draw": i, "warmup": i == 0, "seconds": round(seconds, 3),
            "epochs": epochs,
            "seconds_per_epoch": round(seconds / epochs, 4) if epochs else None,
            "neurons": neurons, "activation": hp["activation"],
            "scaleX": hp["scaleX"], "scaleY": hp["scaleY"],
            "reg": hp["reg"]["val"], "batch_norm": bool(hp["batch_normalization"]),
            "loss": loss, "error": failed,
            "local_seconds": local_seconds[i],
            "speedup_vs_local": (round(local_seconds[i] / seconds, 2)
                                 if local_seconds[i] and seconds else None),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    timed = [r for r in rows if not r["warmup"] and not r["error"]]
    secs = [r["seconds"] for r in timed]
    per_epoch = [r["seconds_per_epoch"] for r in timed if r["seconds_per_epoch"]]
    summary = {
        "summary": True, "config": args.config, "device": args.device,
        "label": args.label, "n_timed": len(timed),
        "mean_seconds": round(float(np.mean(secs)), 2) if secs else None,
        "median_seconds": round(float(np.median(secs)), 2) if secs else None,
        "total_seconds": round(float(np.sum(secs)), 1) if secs else None,
        "median_seconds_per_epoch": (round(float(np.median(per_epoch)), 4)
                                     if per_epoch else None),
        "mean_epochs": round(float(np.mean([r["epochs"] for r in timed])), 1) if timed else None,
        "warmup_seconds": rows[0]["seconds"],
    }
    paired = [r for r in timed if r["local_seconds"]]
    if paired:
        # Two ratios, because they answer different questions. The total ratio is
        # what a whole search costs (long evaluations dominate it, and they are
        # most of the wall time). The median ratio says whether the device helps
        # a typical evaluation, and is not moved by one outlier.
        summary["paired_n"] = len(paired)
        summary["local_total_seconds"] = round(sum(r["local_seconds"] for r in paired), 1)
        summary["total_speedup_vs_local"] = round(
            sum(r["local_seconds"] for r in paired)
            / sum(r["seconds"] for r in paired), 2)
        summary["median_speedup_vs_local"] = round(float(np.median(
            [r["local_seconds"] / r["seconds"] for r in paired])), 2)
    print(json.dumps(summary), flush=True)

    out = args.out or os.path.join(
        HERE, f"bench_{args.config}_{args.zone}_{args.device}.json")
    with open(out, "w") as fh:
        json.dump({"environment": env, "rows": rows, "summary": summary}, fh,
                  indent=2, default=str)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
