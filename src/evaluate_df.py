"""Deep Fingerprinting evaluation on the spring (t0) dataset.

Runs the same four scenarios as evaluate_all_rf.py / evaluate_all_dl.py --
closed world baseline, closed world obfs4, zero-shot baseline->obfs4, and open
world -- but on raw packet-direction sequences instead of the 21 aggregated
features, and with two methodology fixes:

  * the train/test split is chronological, not random stratified;
  * the open world scenario reports TPR/FPR over a confidence sweep rather
    than plain accuracy, which is misleading under class imbalance.
"""

import argparse
import json
import os
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


def fit(X, y_enc, y, stamps, train_idx, n_classes, args):
    """Hold out the latest traces of each class in train as validation, then train.

    Validation is chronological for the same reason the test split is: the
    most recent training traces are the closest stand-in for unseen future
    ones. Returns (model, best_epoch).
    """
    if args.val_size > 0:
        rel_fit, rel_val = chronological_split(
            y[train_idx], stamps[train_idx], test_size=args.val_size
        )
        fit_idx, val_idx = train_idx[rel_fit], train_idx[rel_val]
    else:
        fit_idx, val_idx = train_idx, train_idx[:0]
    return train_df(
        X[fit_idx],
        y_enc[fit_idx],
        n_classes=n_classes,
        epochs=args.epochs,
        seed=args.seed,
        X_val=X[val_idx] if len(val_idx) else None,
        y_val=y_enc[val_idx],
        patience=args.patience,
    )


def encode(labels, classes):
    lookup = {c: i for i, c in enumerate(classes)}
    return np.array([lookup[l] for l in labels], dtype=np.int64)


def closed_world(name, X, y, stamps, args):
    classes = sorted(set(y))
    train_idx, test_idx = chronological_split(
        y, stamps, test_size=args.test_size, per_class=not args.global_split
    )
    train_idx = cap_per_class(train_idx, y, stamps, args.max_per_class)
    y_enc = encode(y, classes)

    print(f"  {len(train_idx)} train / {len(test_idx)} test, {len(classes)} classes")
    model, best_epoch = fit(X, y_enc, y, stamps, train_idx, len(classes), args)
    pred = predict_proba(model, X[test_idx]).argmax(1)
    truth = y_enc[test_idx]

    result = {
        "scenario": name,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_classes": len(classes),
        "max_per_class": int(args.max_per_class),
        "best_epoch": int(best_epoch),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
    }
    print(f"  -> accuracy {result['accuracy']:.4f}  macro-F1 {result['macro_f1']:.4f}")
    return result, model, classes


def zero_shot(model, classes, X, y, name, args):
    """Train on one transport, test on another -- no retraining."""
    keep = np.isin(y, classes)
    if keep.sum() == 0:
        return None
    pred = predict_proba(model, X[keep]).argmax(1)
    truth = encode(y[keep], classes)

    result = {
        "scenario": name,
        "n_test": int(keep.sum()),
        "n_classes": len(classes),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
    }
    print(f"  -> accuracy {result['accuracy']:.4f}  macro-F1 {result['macro_f1']:.4f}")
    return result


def open_world(args):
    """Monitored sites plus an unmonitored pool, scored as TPR against FPR.

    Accuracy is not reported as the headline number here on purpose: with a
    heavily imbalanced monitored/unmonitored ratio it is dominated by whichever
    side is larger and hides what the attacker actually cares about.
    """
    Xm, ym, tm = load_sequences(BASELINE_DIR, mode=args.mode, length=args.length)
    Xu, yu, tu = load_sequences(
        OTHER_DIR, force_label=UNMONITORED_LABEL, mode=args.mode, length=args.length
    )

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

    print(
        f"  {len(train_idx)} train / {len(test_idx)} test, "
        f"{len(monitored)} monitored + 1 unmonitored "
        f"({len(Xm)} vs {len(Xu)} traces, {len(Xm) / max(len(Xu), 1):.1f}:1)"
    )
    model, best_epoch = fit(X, y_enc, y, stamps, train_idx, len(classes), args)

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
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_monitored_classes": len(monitored),
        "n_monitored_traces": int(len(Xm)),
        "n_unmonitored_traces": int(len(Xu)),
        "best_epoch": int(best_epoch),
        "accuracy": float(accuracy_score(truth, pred)),
        "tpr_at_zero_threshold": sweep[0]["tpr"],
        "fpr_at_zero_threshold": sweep[0]["fpr"],
        "sweep": sweep,
    }
    print(
        f"  -> TPR {result['tpr_at_zero_threshold']:.4f} @ "
        f"FPR {result['fpr_at_zero_threshold']:.4f} (no threshold); "
        f"accuracy {result['accuracy']:.4f} (reported for comparison only)"
    )
    return result


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
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
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

    wanted = (
        {"baseline", "obfs4", "zero_shot", "open_world"}
        if args.scenarios == "all"
        else set(args.scenarios.split(","))
    )
    results = {"config": vars(args), "scenarios": {}}

    baseline_model = baseline_classes = None
    if wanted & {"baseline", "zero_shot"}:
        print(f"[closed world: baseline tor, mode={args.mode}]")
        X, y, t = load_sequences(BASELINE_DIR, mode=args.mode, length=args.length)
        res, baseline_model, baseline_classes = closed_world(
            "closed_world_baseline", X, y, t, args
        )
        if "baseline" in wanted:
            results["scenarios"]["closed_world_baseline"] = res

    if wanted & {"obfs4", "zero_shot"}:
        Xo, yo, to = load_sequences(OBFS4_DIR, mode=args.mode, length=args.length)

    if "obfs4" in wanted:
        print("[closed world: obfs4]")
        res, _, _ = closed_world("closed_world_obfs4", Xo, yo, to, args)
        results["scenarios"]["closed_world_obfs4"] = res

    if "zero_shot" in wanted:
        print("[zero shot: train baseline -> test obfs4]")
        res = zero_shot(baseline_model, baseline_classes, Xo, yo, "zero_shot", args)
        if res:
            results["scenarios"]["zero_shot"] = res

    if "open_world" in wanted:
        print("[open world: monitored + unmonitored]")
        results["scenarios"]["open_world"] = open_world(args)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
