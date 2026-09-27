"""Cross-transport evaluation of Deep Fingerprinting on a Tor Browser collection.

The thesis question is whether a DF attacker trained on one transport still
works on another. On tor_dataset/tb/<collection>/features this runs, per seed:

  closed_world   train and test within each arm (chronological split);
  cross          the 3x3 matrix: the model trained on arm A, tested on arm B's
                 test split -- the diagonal is closed_world;
  leave_one_out  train on the other two arms, test on the held-out one: does
                 transport diversity in training buy generalisation?
  pooled         train on every arm's training split, test on each arm;
  background     the clean baseline model tested on background-tab visits,
                 and a model trained with background visits in its training
                 data (earliest share of them) tested on the rest;
  open_world     per arm, monitored + unmonitored, TPR/FPR over a threshold sweep.

Every test split is the same set of latest traces whichever model is scored on
it, so the numbers in one row of the matrix are directly comparable.

    python src/evaluate_transport.py --collection tb-2026-10 --mode cells --length 10000
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from df_dataset import chronological_split, load_sequences  # noqa: E402
from df_model import predict_proba  # noqa: E402
from evaluate_df import aggregate, encode, fit, print_table, provenance  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TB_ROOT = os.getenv("TOR_WF_TB_OUT", os.path.join(REPO_ROOT, "tor_dataset", "tb"))
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")
ARMS = ("baseline", "obfs4", "snowflake")
UNMONITORED = "unmonitored"
ALL_SCENARIOS = {"closed_world", "cross", "leave_one_out", "pooled", "background", "open_world"}


class Split:
    """One arm's traces of one kind, with a fixed chronological split."""

    def __init__(self, X, y, t, test_size):
        self.X, self.y, self.t = X, y, t
        if len(y):
            self.train, self.test = chronological_split(y, t, test_size=test_size)
        else:
            self.train = self.test = np.array([], dtype=int)

    def __len__(self):
        return len(self.y)


def load_arm(features, arm, kind, args, force_label=None):
    d = os.path.join(features, arm, kind)
    if not os.path.isdir(d):
        return None
    X, y, t = load_sequences(d, force_label=force_label, mode=args.mode, length=args.length)
    return Split(X, y, t, args.test_size) if len(y) else None


def score(model, classes, X, y):
    keep = np.isin(y, classes)
    if not keep.any():
        return None
    pred = predict_proba(model, X[keep]).argmax(1)
    truth = encode(y[keep], classes)
    return {
        "n_test": int(keep.sum()),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
    }


def train_on(parts, args, seed):
    """Train one DF model on the training splits of several Split objects."""
    X = np.concatenate([p.X[p.train] for p in parts])
    y = np.concatenate([p.y[p.train] for p in parts])
    t = np.concatenate([p.t[p.train] for p in parts])
    classes = sorted(set(y))
    y_enc = encode(y, classes)
    model, info = fit(X, y_enc, y, t, np.arange(len(y)), len(classes), args, seed)
    return model, classes, info


def open_world(mon, unmon, args, seed):
    X = np.concatenate([mon.X, unmon.X])
    y = np.concatenate([mon.y, unmon.y])
    t = np.concatenate([mon.t, unmon.t])
    offset = len(mon)
    # Each unmonitored site is visited once, so the time split of the
    # unmonitored class also keeps its train and test sites disjoint.
    train_idx = np.concatenate([mon.train, unmon.train + offset])
    test_idx = np.concatenate([mon.test, unmon.test + offset])

    classes = sorted(set(mon.y)) + [UNMONITORED]
    y_enc = encode(y, classes)
    unmon_id = classes.index(UNMONITORED)
    model, info = fit(X, y_enc, y, t, train_idx, len(classes), args, seed)

    proba = predict_proba(model, X[test_idx])
    truth, pred, conf = y_enc[test_idx], proba.argmax(1), proba.max(1)
    is_mon, said_mon = truth != unmon_id, pred != unmon_id
    sweep = []
    for thr in np.round(np.arange(0.0, 1.0, 0.05), 2):
        accept = said_mon & (conf >= thr)
        sweep.append({
            "threshold": float(thr),
            "tpr": float((accept & is_mon & (pred == truth)).sum() / max(is_mon.sum(), 1)),
            "fpr": float((accept & ~is_mon).sum() / max((~is_mon).sum(), 1)),
        })
    return {
        "seed": seed,
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "n_monitored_classes": len(classes) - 1,
        "n_monitored_traces": int(len(mon)), "n_unmonitored_traces": int(len(unmon)),
        "n_unmonitored_test": int((~is_mon).sum()),
        "tpr_at_zero_threshold": sweep[0]["tpr"], "fpr_at_zero_threshold": sweep[0]["fpr"],
        "sweep": sweep, **info,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--mode", default="cells", choices=["direction", "cells", "tiktok", "iat"])
    p.add_argument("--length", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--min-epochs", type=int, default=50)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--scenarios", default="all", help=f"comma separated from {sorted(ALL_SCENARIOS)}")
    p.add_argument("--out", default="")
    args = p.parse_args()
    # evaluate_df.fit reads these.
    args.max_per_class, args.global_split = 0, False

    wanted = ALL_SCENARIOS if args.scenarios == "all" else set(args.scenarios.split(","))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    features = os.path.join(TB_ROOT, args.collection, "features")
    out = args.out or os.path.join(FIGURES_DIR, f"metrics_transport_{args.collection}_{args.mode}.json")

    mon = {a: load_arm(features, a, "monitored", args) for a in ARMS}
    mon = {a: s for a, s in mon.items() if s is not None}
    if not mon:
        sys.exit(f"no monitored features under {features}")
    unmon = {}
    if "open_world" in wanted:
        unmon = {a: load_arm(features, a, "unmonitored", args, force_label=UNMONITORED) for a in mon}
        unmon = {a: s for a, s in unmon.items() if s is not None and len(s.test)}
    bg = load_arm(features, "baseline", "background", args) if "background" in wanted else None
    for a, s in mon.items():
        print(f"  {a:10s} monitored {len(s):5d} traces, {len(set(s.y))} sites"
              + (f", unmonitored {len(unmon[a])}" if a in unmon else ""))
    if bg is not None:
        print(f"  baseline   background {len(bg):5d} traces")

    runs, matrix_runs = {}, []

    def record(name, result, seed):
        if result:
            result.setdefault("seed", seed)
            runs.setdefault(name, []).append(result)

    for seed in seeds:
        print(f"\n########## seed {seed} ##########")
        models = {}
        if wanted & {"closed_world", "cross", "background"}:
            for a, s in mon.items():
                print(f"[train {a}]")
                model, classes, info = train_on([s], args, seed)
                models[a] = (model, classes)
                res = score(model, classes, s.X[s.test], s.y[s.test])
                res.update(info)
                print(f"  -> {a} closed world {res['accuracy']:.4f}")
                record(f"closed_world_{a}", res, seed)

        if "cross" in wanted:
            matrix = {}
            for a, (model, classes) in models.items():
                for b, s in mon.items():
                    res = score(model, classes, s.X[s.test], s.y[s.test])
                    matrix[f"{a}->{b}"] = res["accuracy"]
                    if a != b:
                        record(f"cross_{a}_to_{b}", res, seed)
                        print(f"  {a} -> {b}: {res['accuracy']:.4f}")
            matrix_runs.append(matrix)

        if "leave_one_out" in wanted and len(mon) > 2:
            for held in mon:
                print(f"[train all but {held}]")
                model, classes, info = train_on([s for a, s in mon.items() if a != held], args, seed)
                res = score(model, classes, mon[held].X[mon[held].test], mon[held].y[mon[held].test])
                res.update(info)
                print(f"  -> held-out {held}: {res['accuracy']:.4f}")
                record(f"leave_out_{held}", res, seed)

        if "pooled" in wanted and len(mon) > 1:
            print("[train pooled]")
            model, classes, info = train_on(list(mon.values()), args, seed)
            for a, s in mon.items():
                res = score(model, classes, s.X[s.test], s.y[s.test])
                print(f"  -> pooled on {a}: {res['accuracy']:.4f}")
                record(f"pooled_on_{a}", res, seed)

        if "background" in wanted and bg is not None and "baseline" in models:
            model, classes = models["baseline"]
            res = score(model, classes, bg.X[bg.test], bg.y[bg.test])
            print(f"  clean baseline model on background test: {res['accuracy']:.4f}")
            record("background_clean_model", res, seed)
            print("[train baseline + background]")
            model, classes, info = train_on([mon["baseline"], bg], args, seed)
            res = score(model, classes, bg.X[bg.test], bg.y[bg.test])
            res.update(info)
            print(f"  -> background-aware model on background test: {res['accuracy']:.4f}")
            record("background_aware_model", res, seed)

        if "open_world" in wanted:
            for a in unmon:
                print(f"[open world {a}]")
                res = open_world(mon[a], unmon[a], args, seed)
                print(f"  -> TPR {res['tpr_at_zero_threshold']:.4f} FPR {res['fpr_at_zero_threshold']:.4f}")
                record(f"open_world_{a}", res, seed)

    results = {
        "config": vars(args),
        "provenance": provenance(),
        "scenarios": {name: aggregate(name, rs) for name, rs in runs.items()},
    }
    if matrix_runs:
        keys = matrix_runs[0].keys()
        results["cross_matrix"] = {
            k: {"mean": float(np.mean([m[k] for m in matrix_runs])),
                "sd": float(np.std([m[k] for m in matrix_runs], ddof=1)) if len(matrix_runs) > 1 else 0.0}
            for k in keys
        }
        print("\n=== train (row) -> test (column) accuracy, mean over seeds ===")
        cols = list(mon)
        print(f"{'':12s}" + "".join(f"{c:>12s}" for c in cols))
        for a in cols:
            print(f"{a:12s}" + "".join(f"{results['cross_matrix'][f'{a}->{b}']['mean'] * 100:>11.1f}%"
                                       for b in cols))
    print_table(results["scenarios"])
    for name, agg in results["scenarios"].items():
        if "tpr_at_zero_threshold_mean" in agg:
            print(f"{name:26s} TPR {agg['tpr_at_zero_threshold_mean'] * 100:.2f} +- "
                  f"{agg['tpr_at_zero_threshold_sd'] * 100:.2f}  FPR "
                  f"{agg['fpr_at_zero_threshold_mean'] * 100:.2f} +- "
                  f"{agg['fpr_at_zero_threshold_sd'] * 100:.2f}  "
                  f"({agg['runs'][0]['n_unmonitored_test']} unmonitored test traces)")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
