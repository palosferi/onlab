"""Deep Fingerprinting evaluation on the spring (t0) dataset.

Runs the same four scenarios as evaluate_all_rf.py / evaluate_all_dl.py --
closed world baseline, closed world obfs4, zero-shot baseline->obfs4, and open
world -- but on raw packet-direction sequences instead of the 21 aggregated
features, and with three methodology fixes:

  * the train/test split is chronological, not random stratified;
  * the open world scenario reports TPR/FPR over a confidence sweep rather
    than plain accuracy, which is misleading under class imbalance;
  * every scenario runs under several seeds and is reported as mean +- sd.

The last one is not decoration. With roughly 30 traces per class a single run
is not a measurement: the same code and seed gave 73% and 28% on obfs4 on two
machines, because a noisy validation signal stopped training at different
points. Spread across seeds is the honest error bar, and a wide one is itself
the finding that the dataset is too small to tune a deep model on.
"""

import argparse
import json
import os
import platform
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from df_dataset import SEQUENCE_LENGTH, chronological_split, load_sequences
from df_model import predict_proba, train_df

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tor_dataset", "extracted_features")
)
BASELINE_DIR = os.path.join(BASE_DIR, "baseline_features")
OBFS4_DIR = os.path.join(BASE_DIR, "obfs4_features")
OTHER_DIR = os.path.join(BASE_DIR, "other_features")
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))

UNMONITORED_LABEL = "unmonitored"

AGGREGATE_KEYS = (
    "accuracy",
    "macro_f1",
    "tpr_at_zero_threshold",
    "fpr_at_zero_threshold",
)
CARRY_KEYS = (
    "n_train",
    "n_fit",
    "n_val",
    "n_test",
    "n_classes",
    "n_monitored_classes",
    "n_monitored_traces",
    "n_unmonitored_traces",
    "val_traces_per_class_median",
)


def provenance():
    """Facts that explain why two runs of the same seed can differ.

    Reduction order in the convolution kernels depends on the thread count and
    the CPU, so results are only bit-comparable between machines when these
    match. Recording them keeps a surprising number honest.
    """
    import torch

    return {
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "host": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def cap_per_class(train_idx, y, stamps, max_per_class):
    """Keep only the earliest `max_per_class` training traces of each class.

    Used to draw a learning curve: DF was designed for 800+ traces per class
    and we have roughly 30, so the question is whether the attack is limited by
    the architecture or simply starved of data.
    """
    if not max_per_class:
        return train_idx
    kept = []
    for label in np.unique(y[train_idx]):
        idx = train_idx[y[train_idx] == label]
        idx = idx[np.argsort(stamps[idx], kind="stable")]
        kept.append(idx[:max_per_class])
    return np.concatenate(kept) if kept else train_idx


def fit(X, y_enc, y, stamps, train_idx, n_classes, args, seed):
    """Hold out the latest traces of each class in train as validation, then train.

    Validation is chronological for the same reason the test split is: the most
    recent training traces are the closest stand-in for unseen future ones.

    `val_size` stays small on purpose. The validation set is already only two or
    three traces per class, and taking more would starve training further, so
    the noise is handled by watching validation loss -- which uses every
    validation trace's probability, not just whether its argmax was right --
    rather than by enlarging the set.
    """
    if args.val_size > 0:
        rel_fit, rel_val = chronological_split(
            y[train_idx], stamps[train_idx], test_size=args.val_size
        )
        fit_idx, val_idx = train_idx[rel_fit], train_idx[rel_val]
    else:
        fit_idx, val_idx = train_idx, train_idx[:0]

    model, info = train_df(
        X[fit_idx],
        y_enc[fit_idx],
        n_classes=n_classes,
        epochs=args.epochs,
        seed=seed,
        X_val=X[val_idx] if len(val_idx) else None,
        y_val=y_enc[val_idx],
        patience=args.patience,
        min_epochs=args.min_epochs,
        threads=args.threads or None,
    )
    info["n_fit"] = int(len(fit_idx))
    info["n_val"] = int(len(val_idx))
    if len(val_idx):
        _, counts = np.unique(y[val_idx], return_counts=True)
        info["val_traces_per_class_median"] = int(np.median(counts))
    return model, info


def encode(labels, classes):
    lookup = {c: i for i, c in enumerate(classes)}
    return np.array([lookup[l] for l in labels], dtype=np.int64)


def run_summary(info):
    """One line saying whether the model had actually converged when it stopped."""
    return (
        f"  -> best epoch {info['best_epoch']} of {info['stopped_at']} run "
        f"(cap {info['epoch_cap']}), train acc {info['final_train_acc']:.4f}, "
        f"val loss {info['best_val_loss']}"
    )


def closed_world(name, X, y, stamps, args, seed):
    classes = sorted(set(y))
    train_idx, test_idx = chronological_split(
        y, stamps, test_size=args.test_size, per_class=not args.global_split
    )
    train_idx = cap_per_class(train_idx, y, stamps, args.max_per_class)
    y_enc = encode(y, classes)

    model, info = fit(X, y_enc, y, stamps, train_idx, len(classes), args, seed)
    pred = predict_proba(model, X[test_idx]).argmax(1)
    truth = y_enc[test_idx]

    result = {
        "scenario": name,
        "seed": seed,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_classes": len(classes),
        "max_per_class": int(args.max_per_class),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
    }
    result.update(info)
    print(run_summary(info))
    print(f"  -> accuracy {result['accuracy']:.4f}  macro-F1 {result['macro_f1']:.4f}")
    return result, model, classes


def zero_shot(model, classes, X, y, name, seed):
    """Train on one transport, test on another -- no retraining."""
    keep = np.isin(y, classes)
    if keep.sum() == 0:
        return None
    pred = predict_proba(model, X[keep]).argmax(1)
    truth = encode(y[keep], classes)

    result = {
        "scenario": name,
        "seed": seed,
        "n_test": int(keep.sum()),
        "n_classes": len(classes),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
    }
    print(f"  -> accuracy {result['accuracy']:.4f}  macro-F1 {result['macro_f1']:.4f}")
    return result


def open_world(args, seed, monitored_data, unmonitored_data):
    """Monitored sites plus an unmonitored pool, scored as TPR against FPR.

    Accuracy is not reported as the headline number here on purpose: with a
    heavily imbalanced monitored/unmonitored ratio it is dominated by whichever
    side is larger and hides what the attacker actually cares about.
    """
    Xm, ym, tm = monitored_data
    Xu, yu, tu = unmonitored_data

    X = np.concatenate([Xm, Xu])
    y = np.concatenate([ym, yu])
    stamps = np.concatenate([tm, tu])

    monitored = sorted(set(ym))
    classes = monitored + [UNMONITORED_LABEL]
    train_idx, test_idx = chronological_split(
        y, stamps, test_size=args.test_size, per_class=not args.global_split
    )
    y_enc = encode(y, classes)
    unmon_id = classes.index(UNMONITORED_LABEL)

    model, info = fit(X, y_enc, y, stamps, train_idx, len(classes), args, seed)

    proba = predict_proba(model, X[test_idx])
    truth = y_enc[test_idx]
    pred = proba.argmax(1)
    confidence = proba.max(1)

    is_mon_true = truth != unmon_id
    is_mon_pred = pred != unmon_id

    sweep = []
    for thr in np.round(np.arange(0.0, 1.0, 0.05), 2):
        accept = is_mon_pred & (confidence >= thr)
        # TPR: monitored trace accepted AND attributed to the right site.
        tpr = float((accept & is_mon_true & (pred == truth)).sum() / max(is_mon_true.sum(), 1))
        # FPR: unmonitored trace wrongly claimed as some monitored site.
        fpr = float((accept & ~is_mon_true).sum() / max((~is_mon_true).sum(), 1))
        sweep.append({"threshold": float(thr), "tpr": tpr, "fpr": fpr})

    result = {
        "scenario": "open_world",
        "seed": seed,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_monitored_classes": len(monitored),
        "n_monitored_traces": int(len(Xm)),
        "n_unmonitored_traces": int(len(Xu)),
        "accuracy": float(accuracy_score(truth, pred)),
        "tpr_at_zero_threshold": sweep[0]["tpr"],
        "fpr_at_zero_threshold": sweep[0]["fpr"],
        "sweep": sweep,
    }
    result.update(info)
    print(run_summary(info))
    print(
        f"  -> TPR {result['tpr_at_zero_threshold']:.4f} @ "
        f"FPR {result['fpr_at_zero_threshold']:.4f} (no threshold); "
        f"accuracy {result['accuracy']:.4f} (reported for comparison only)"
    )
    return result


def aggregate(name, runs):
    """Mean, spread and per-seed detail for one scenario."""
    out = {"scenario": name, "n_seeds": len(runs), "seeds": [r["seed"] for r in runs]}
    for key in AGGREGATE_KEYS:
        vals = [r[key] for r in runs if key in r]
        if not vals:
            continue
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[f"{key}_min"] = float(min(vals))
        out[f"{key}_max"] = float(max(vals))
    for key in CARRY_KEYS:
        if key in runs[0]:
            out[key] = runs[0][key]
    out["best_epochs"] = [r["best_epoch"] for r in runs if "best_epoch" in r]
    out["runs"] = runs
    return out


def print_table(scenarios):
    print("\n=== summary over seeds ===")
    header = f"{'scenario':26s} {'accuracy':>18s} {'macro-F1':>18s}  {'best epochs':>14s}"
    print(header)
    print("-" * len(header))
    for name, agg in scenarios.items():
        acc = (
            f"{agg['accuracy_mean'] * 100:.2f} +- {agg['accuracy_sd'] * 100:.2f}"
            if "accuracy_mean" in agg
            else "-"
        )
        f1 = (
            f"{agg['macro_f1_mean'] * 100:.2f} +- {agg['macro_f1_sd'] * 100:.2f}"
            if "macro_f1_mean" in agg
            else "-"
        )
        epochs = ",".join(str(e) for e in agg.get("best_epochs", []))
        print(f"{name:26s} {acc:>18s} {f1:>18s}  {epochs:>14s}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="direction", choices=["direction", "tiktok", "iat"])
    p.add_argument("--length", type=int, default=SEQUENCE_LENGTH)
    p.add_argument(
        "--epochs", type=int, default=150, help="upper bound; early stopping picks the epoch"
    )
    p.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="latest share of each class's training traces used for early stopping (0 = off)",
    )
    p.add_argument(
        "--patience", type=int, default=30, help="epochs without a better validation loss"
    )
    p.add_argument(
        "--min-epochs",
        type=int,
        default=50,
        help="never stop before this epoch, however the validation signal looks",
    )
    p.add_argument(
        "--seeds",
        default="42,43,44",
        help="comma separated; every scenario runs once per seed and is reported as mean +- sd",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=0,
        help="torch CPU threads (0 = leave default); set it equal on two machines "
        "to make their numbers comparable",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="cap training traces per class (0 = use all); for learning curves",
    )
    p.add_argument(
        "--global-split",
        action="store_true",
        help="one global time cut instead of a per-class one",
    )
    p.add_argument(
        "--scenarios",
        default="all",
        help="comma separated: baseline,obfs4,zero_shot,open_world (or 'all')",
    )
    p.add_argument("--out", default=os.path.join(FIGURES_DIR, "metrics_df.json"))
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    wanted = (
        {"baseline", "obfs4", "zero_shot", "open_world"}
        if args.scenarios == "all"
        else set(args.scenarios.split(","))
    )

    # Loaded once and reused for every seed: reading ~2200 CSVs per pass would
    # otherwise dominate the runtime.
    baseline_data = obfs4_data = other_data = None
    if wanted & {"baseline", "zero_shot", "open_world"}:
        print(f"[loading baseline sequences, mode={args.mode}, length={args.length}]")
        baseline_data = load_sequences(BASELINE_DIR, mode=args.mode, length=args.length)
    if wanted & {"obfs4", "zero_shot"}:
        print("[loading obfs4 sequences]")
        obfs4_data = load_sequences(OBFS4_DIR, mode=args.mode, length=args.length)
    if "open_world" in wanted:
        print("[loading unmonitored sequences]")
        other_data = load_sequences(
            OTHER_DIR, force_label=UNMONITORED_LABEL, mode=args.mode, length=args.length
        )

    runs = {}

    def record(name, result):
        if result:
            runs.setdefault(name, []).append(result)

    for seed in seeds:
        print(f"\n########## seed {seed} ##########")
        baseline_model = baseline_classes = None

        if wanted & {"baseline", "zero_shot"}:
            print("[closed world: baseline tor]")
            res, baseline_model, baseline_classes = closed_world(
                "closed_world_baseline", *baseline_data, args, seed
            )
            if "baseline" in wanted:
                record("closed_world_baseline", res)

        if "obfs4" in wanted:
            print("[closed world: obfs4]")
            res, _, _ = closed_world("closed_world_obfs4", *obfs4_data, args, seed)
            record("closed_world_obfs4", res)

        if "zero_shot" in wanted and baseline_model is not None:
            print("[zero shot: train baseline -> test obfs4]")
            Xo, yo, _ = obfs4_data
            record("zero_shot", zero_shot(baseline_model, baseline_classes, Xo, yo,
                                          "zero_shot", seed))

        if "open_world" in wanted:
            print("[open world: monitored + unmonitored]")
            record("open_world", open_world(args, seed, baseline_data, other_data))

    results = {
        "config": vars(args),
        "provenance": provenance(),
        "scenarios": {name: aggregate(name, rs) for name, rs in runs.items()},
    }

    print_table(results["scenarios"])

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
