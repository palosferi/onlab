"""What the cross-transport attack costs the attacker: R1, R2 and R4, plus R3.

evaluate_transport.py answers whether DF transfers between transports. This
script asks what it takes to make it work, on the same features, splits and
training procedure:

  budget     (R1) train on the source arm plus the k earliest training traces
                  per site from a target arm, test on the target's test split.
                  k = 0 is the zero-shot cross result, k = all is the pair pooled.
  set_size   (R2) open world with only k monitored sites, chosen at random per
                  seed; the other monitored sites join the unmonitored class. The
                  FPR is reported separately on the unmonitored pool (sites never
                  seen in training) and on the demoted sites (seen in training,
                  labelled unmonitored).
  curve      (R4) closed world on the n earliest training traces per site.

R3 needs no training: --precision-from reads the open-world threshold sweeps of
a results file (this script's or evaluate_transport.py's) and turns TPR/FPR into
precision at a base rate b, the share of page loads that go to a monitored site:

    precision = b * TPR / (b * (TPR + WPR) + (1 - b) * FPR)

WPR is the rate of monitored traces accepted under the wrong site; the sweeps
of evaluate_transport.py do not record it, and it is taken as 0 for them.

Every test split is the one evaluate_transport.py uses (the latest traces of
each site), so budget k = 0 and k = all, set_size k = all and curve n = all
repeat numbers from that run and check this one against it.

    python src/evaluate_budget.py --collection tb-main --min-visits 30 --scenarios budget,curve
    python src/evaluate_budget.py --precision-from figures/metrics_transport_tb-main_cells.json
"""

import argparse
from collections import Counter
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from df_dataset import chronological_split  # noqa: E402
from df_model import predict_proba  # noqa: E402
from evaluate_df import aggregate, encode, fit, provenance  # noqa: E402
from evaluate_transport import (ARMS, FIGURES_DIR, RESUME_KEYS, TB_ROOT, UNMONITORED,  # noqa: E402
                                Split, load_arm, score)

ALL_SCENARIOS = {"budget", "set_size", "curve"}
BUDGET_RESUME_KEYS = RESUME_KEYS + ("source", "targets", "budget_ks", "set_size_arms",
                                    "set_size_ks", "curve_arms", "curve_ns", "limit_sites")
BASE_RATES = (0.5, 0.2, 0.05, 0.01, 0.001)
THRESHOLDS = np.round(np.arange(0.0, 1.0, 0.05), 2)


def parse_counts(text):
    """'0,1,3,all' -> [0, 1, 3, None]; None means every available trace or site."""
    return [None if s.strip() == "all" else int(s) for s in text.split(",") if s.strip()]


def earliest_per_site(split, n):
    """Indices of the n earliest training traces of every site (all of them if n is None)."""
    if n is None:
        return split.train
    out = []
    for site in np.unique(split.y[split.train]):
        idx = split.train[split.y[split.train] == site]
        out.append(idx[np.argsort(split.t[idx], kind="stable")][:n])
    return np.concatenate(out) if out else split.train[:0]


def train_subsets(parts, args, seed):
    """Train one DF model on chosen training indices of several Splits."""
    X = np.concatenate([s.X[i] for s, i in parts])
    y = np.concatenate([s.y[i] for s, i in parts])
    t = np.concatenate([s.t[i] for s, i in parts])
    classes = sorted(set(y))
    model, info = fit(X, encode(y, classes), y, t, np.arange(len(y)), len(classes), args, seed)
    info["n_train"] = int(len(y))
    return model, classes, info


def subset(split, keep_sites, test_size):
    m = np.isin(split.y, list(keep_sites))
    return Split(split.X[m], split.y[m], split.t[m], test_size)


def open_world_sweep(proba, truth, unmon_id, groups):
    """TPR, WPR and one FPR per unmonitored group over the threshold sweep."""
    pred, conf = proba.argmax(1), proba.max(1)
    is_mon, said_mon = truth != unmon_id, pred != unmon_id
    sweep = []
    for thr in THRESHOLDS:
        accept = said_mon & (conf >= thr)
        row = {"threshold": float(thr),
               "tpr": float((accept & is_mon & (pred == truth)).sum() / max(is_mon.sum(), 1)),
               "wpr": float((accept & is_mon & (pred != truth)).sum() / max(is_mon.sum(), 1))}
        for name, mask in groups.items():
            row[f"fpr_{name}"] = float((accept & mask).sum() / max(mask.sum(), 1))
        row["fpr"] = float((accept & ~is_mon).sum() / max((~is_mon).sum(), 1))
        sweep.append(row)
    return sweep


def set_size_run(mon, unmon, k, args, seed):
    """Open world with k monitored sites; the remaining monitored sites are demoted."""
    sites = sorted(set(mon.y))
    rng = np.random.default_rng(seed)
    chosen = set(sites) if k is None or k >= len(sites) else set(rng.choice(sites, k, replace=False))
    m = np.isin(mon.y, list(chosen))
    # The demoted sites keep their own chronological split, so each one has
    # earlier visits in training and later ones in the test, like any site.
    rest = Split(mon.X[~m], mon.y[~m], mon.t[~m], args.test_size)
    kept = Split(mon.X[m], mon.y[m], mon.t[m], args.test_size)

    parts = [(kept, "mon"), (unmon, "pool"), (rest, "demoted")]
    X = np.concatenate([s.X for s, _ in parts])
    y = np.concatenate([s.y if g == "mon" else np.full(len(s), UNMONITORED) for s, g in parts])
    t = np.concatenate([s.t for s, _ in parts])
    group = np.concatenate([np.full(len(s), g) for s, g in parts])
    offsets = np.cumsum([0] + [len(s) for s, _ in parts[:-1]])
    train_idx = np.concatenate([s.train + o for (s, _), o in zip(parts, offsets)])
    test_idx = np.concatenate([s.test + o for (s, _), o in zip(parts, offsets)])

    classes = sorted(chosen) + [UNMONITORED]
    y_enc = encode(y, classes)
    model, info = fit(X, y_enc, y, t, train_idx, len(classes), args, seed)
    proba = predict_proba(model, X[test_idx])
    g = group[test_idx]
    groups = {"pool": g == "pool"}
    if len(rest):
        groups["demoted"] = g == "demoted"
    sweep = open_world_sweep(proba, y_enc[test_idx], classes.index(UNMONITORED), groups)
    return {
        "seed": seed, "k": len(chosen), "monitored_sites": sorted(chosen),
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "n_monitored_classes": len(chosen),
        "n_unmonitored_test": int((g != "mon").sum()),
        "tpr_at_zero_threshold": sweep[0]["tpr"], "fpr_at_zero_threshold": sweep[0]["fpr"],
        "sweep": sweep, **info,
    }


def precision_table(path, rates=BASE_RATES):
    """R3: precision at base rates from every open-world sweep in a results file."""
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    # A .partial.json of a run still going keeps its runs unaggregated.
    scenarios = results.get("scenarios") or {n: {"runs": rs} for n, rs in results["runs"].items()}
    table = {}
    for name, agg in scenarios.items():
        runs = [r for r in agg.get("runs", []) if "sweep" in r]
        if not runs:
            continue
        entry = {}
        for b in rates:
            at_zero, best = [], []
            for r in runs:
                prec = []
                for row in r["sweep"]:
                    denom = b * (row["tpr"] + row.get("wpr", 0.0)) + (1 - b) * row["fpr"]
                    prec.append((b * row["tpr"] / denom if denom else 0.0, row["tpr"], row["threshold"]))
                at_zero.append(prec[0][0])
                # The best threshold that still finds at least half the monitored visits.
                usable = [p for p in prec if p[1] >= 0.5] or prec[:1]
                best.append(max(usable))
            entry[str(b)] = {
                "precision_at_zero_mean": float(np.mean(at_zero)),
                "precision_at_zero_sd": float(np.std(at_zero, ddof=1)) if len(at_zero) > 1 else 0.0,
                "best_precision_mean": float(np.mean([p for p, _, _ in best])),
                "best_precision_sd": float(np.std([p for p, _, _ in best], ddof=1)) if len(best) > 1 else 0.0,
                "best_tpr_mean": float(np.mean([t for _, t, _ in best])),
                "best_thresholds": [th for _, _, th in best],
            }
        table[name] = {"n_seeds": len(runs), "wpr_recorded": "wpr" in runs[0]["sweep"][0],
                       "base_rates": entry}
    print(f"\n=== precision at base rate b (mean over seeds): threshold 0 / best with TPR >= 50% ===")
    print(f"{'scenario':28s}" + "".join(f"{'b=' + str(b):>16s}" for b in rates))
    for name, e in table.items():
        cells = "".join(f"{e['base_rates'][str(b)]['precision_at_zero_mean'] * 100:>7.1f}"
                        f"/{e['base_rates'][str(b)]['best_precision_mean'] * 100:>5.1f}%  " for b in rates)
        print(f"{name:28s}{cells}" + ("" if e["wpr_recorded"] else "  (WPR not recorded)"))
    out = os.path.splitext(path)[0] + "_precision.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"source": os.path.basename(path), "formula": "b*TPR / (b*(TPR+WPR) + (1-b)*FPR)",
                   "base_rates": list(rates), "scenarios": table}, f, indent=2)
    print(f"\nwrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--precision-from", default="", help="R3 only: a results JSON with open-world sweeps")
    p.add_argument("--collection")
    p.add_argument("--mode", default="cells", choices=["direction", "cells", "tiktok", "iat"])
    p.add_argument("--length", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--min-epochs", type=int, default=50)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--min-visits", type=int, default=0,
                   help="drop monitored sites with fewer ok visits than this in any arm")
    p.add_argument("--scenarios", default="all", help=f"comma separated from {sorted(ALL_SCENARIOS)}")
    p.add_argument("--source", default="baseline", help="budget: the arm the attacker has plenty of")
    p.add_argument("--targets", default="obfs4,snowflake")
    p.add_argument("--budget-ks", default="0,1,3,5,10,all", help="target traces per site")
    p.add_argument("--set-size-arms", default="baseline")
    p.add_argument("--set-size-ks", default="5,10,20,all", help="monitored sites")
    p.add_argument("--curve-arms", default="baseline")
    p.add_argument("--curve-ns", default="5,10,15,20,all", help="training traces per site")
    p.add_argument("--limit-sites", type=int, default=0, help="smoke tests: keep only the first N sites")
    p.add_argument("--out", default="")
    args = p.parse_args()
    if args.precision_from:
        precision_table(args.precision_from)
        return
    if not args.collection:
        p.error("--collection is required unless --precision-from")
    args.max_per_class, args.global_split = 0, False

    wanted = ALL_SCENARIOS if args.scenarios == "all" else set(args.scenarios.split(","))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    features = os.path.join(TB_ROOT, args.collection, "features")
    out = args.out or os.path.join(FIGURES_DIR, f"metrics_budget_{args.collection}_{args.mode}.json")

    mon = {a: load_arm(features, a, "monitored", args) for a in ARMS}
    mon = {a: s for a, s in mon.items() if s is not None}
    if not mon:
        sys.exit(f"no monitored features under {features}")
    counts = [Counter(s.y) for s in mon.values()]
    keep = set.intersection(*(set(c) for c in counts))
    if args.min_visits:
        keep = {x for x in keep if all(c[x] >= args.min_visits for c in counts)}
    if args.limit_sites:
        keep = set(sorted(keep)[:args.limit_sites])
    mon = {a: subset(s, keep, args.test_size) for a, s in mon.items()}
    print(f"  {len(keep)} sites in every arm: " + ", ".join(f"{a} {len(s)}" for a, s in mon.items()))
    unmon = {}
    if "set_size" in wanted:
        for a in args.set_size_arms.split(","):
            s = load_arm(features, a, "unmonitored", args, force_label=UNMONITORED)
            if s is not None and len(s.test):
                unmon[a] = s

    runs, done = {}, []
    partial = os.path.splitext(out)[0] + ".partial.json"
    if os.path.exists(partial):
        with open(partial, encoding="utf-8") as f:
            saved = json.load(f)
        changed = [k for k in BUDGET_RESUME_KEYS if saved["config"].get(k) != vars(args).get(k)]
        if changed:
            sys.exit(f"{partial} was made with different {', '.join(changed)}; move it away to start over")
        runs, done = saved["runs"], saved["seeds_done"]
        print(f"  resuming from {partial}: seeds {done} already done")

    def record(name, result, seed):
        result.setdefault("seed", seed)
        runs.setdefault(name, []).append(result)

    label = lambda n: "all" if n is None else str(n)  # noqa: E731
    for seed in seeds:
        if seed in done:
            continue
        print(f"\n########## seed {seed} ##########")

        if "budget" in wanted:
            src = mon[args.source]
            targets = [a for a in args.targets.split(",") if a in mon and a != args.source]
            for k in parse_counts(args.budget_ks):
                # With k = 0 every target is scored on the same source-only model.
                shared = None
                for tgt in targets:
                    if k == 0 and shared is not None:
                        model, classes, info = shared
                    else:
                        print(f"[train {args.source} + {label(k)} {tgt} traces per site]")
                        parts = [(src, src.train)]
                        if k != 0:
                            parts.append((mon[tgt], earliest_per_site(mon[tgt], k)))
                        model, classes, info = train_subsets(parts, args, seed)
                        if k == 0:
                            shared = (model, classes, info)
                    t = mon[tgt]
                    res = {**score(model, classes, t.X[t.test], t.y[t.test]), **info, "k": label(k)}
                    print(f"  -> {args.source} + {label(k)} -> {tgt}: {res['accuracy']:.4f}")
                    record(f"budget_{args.source}_to_{tgt}_k{label(k)}", res, seed)

        if "set_size" in wanted:
            for a in unmon:
                for k in parse_counts(args.set_size_ks):
                    print(f"[open world {a}, {label(k)} monitored sites]")
                    res = set_size_run(mon[a], unmon[a], k, args, seed)
                    fprs = {key: v for key, v in res["sweep"][0].items() if key.startswith("fpr_")}
                    print(f"  -> TPR {res['tpr_at_zero_threshold']:.4f} "
                          + " ".join(f"{key.upper()} {v:.4f}" for key, v in fprs.items()))
                    record(f"set_size_{a}_k{label(k)}", res, seed)

        if "curve" in wanted:
            for a in args.curve_arms.split(","):
                for n in parse_counts(args.curve_ns):
                    print(f"[train {a} on {label(n)} traces per site]")
                    s = mon[a]
                    model, classes, info = train_subsets([(s, earliest_per_site(s, n))], args, seed)
                    res = {**score(model, classes, s.X[s.test], s.y[s.test]), **info, "n": label(n)}
                    print(f"  -> {a} closed world with {label(n)} per site: {res['accuracy']:.4f}")
                    record(f"curve_{a}_n{label(n)}", res, seed)

        done.append(seed)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(partial + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "seeds_done": done, "runs": runs}, f, indent=2)
        os.replace(partial + ".tmp", partial)

    results = {"config": vars(args), "provenance": provenance(),
               "scenarios": {name: aggregate(name, rs) for name, rs in runs.items()}}
    print("\n=== summary over seeds ===")
    for name, agg in results["scenarios"].items():
        if "accuracy_mean" in agg:
            print(f"{name:34s} accuracy {agg['accuracy_mean'] * 100:6.2f} +- {agg['accuracy_sd'] * 100:.2f}")
        else:
            print(f"{name:34s} TPR {agg['tpr_at_zero_threshold_mean'] * 100:6.2f} +- "
                  f"{agg['tpr_at_zero_threshold_sd'] * 100:.2f}  FPR "
                  f"{agg['fpr_at_zero_threshold_mean'] * 100:6.2f} +- {agg['fpr_at_zero_threshold_sd'] * 100:.2f}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"\nwrote {out}")
    if "set_size" in wanted:
        precision_table(out)


if __name__ == "__main__":
    main()
