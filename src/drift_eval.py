"""Concept drift evaluation for Tor website fingerprinting.

Measures how a WF classifier's accuracy decays as the gap between training data
and test data grows, and compares maintenance policies that an adversary could
actually run.

Time origin (t0) is the spring 2026 collection.  Every longitudinal round adds
one later point.  With only t0 present the script still runs and reports the
within-t0 ceiling, so it is useful before the first round completes.

    python src/drift_eval.py
    python src/drift_eval.py --arm obfs4
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_pool import FEATURE_NAMES, extract_aggregated_features, select_top_features_mutual_info  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPRING_DIR = os.path.join(REPO_ROOT, "tor_dataset", "extracted_features")
LONGITUDINAL_DIR = os.getenv(
    "TOR_WF_LONGITUDINAL_DIR", os.path.join(REPO_ROOT, "tor_dataset", "longitudinal")
)
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")

RF_N_ESTIMATORS = 300
TOP_K_FEATURES = 10
SEEDS = [42, 52, 62]

# Site groups from the original target list design.  The interesting question
# is not the average decay but whether it splits by content type.
SITE_GROUPS = {
    "static": ["wikipedia", "duckduckgo", "github", "stackoverflow", "mit_ocw",
               "gnu", "debian", "w3c", "wordpress", "mozilla", "archive", "w3c"],
    "news": ["bbc", "cnn", "reuters", "theguardian", "aljazeera", "hacker_news",
             "wired", "techcrunch", "telex", "index", "hvg", "origo", "hwsw"],
    "commerce_media": ["amazon", "ebay", "aliexpress", "imdb", "reddit", "twitch",
                       "vimeo", "soundcloud", "medium", "quora", "coursera", "bme"],
}
GROUP_OF = {site: g for g, sites in SITE_GROUPS.items() for site in sites}

_DATE_RE = re.compile(r"_(\d{8})_\d{6}")


def _dir_median_date(directory):
    """Median capture date of a feature directory, from the filenames."""
    dates = []
    for name in os.listdir(directory):
        m = _DATE_RE.search(name)
        if m:
            try:
                dates.append(datetime.strptime(m.group(1), "%Y%m%d"))
            except ValueError:
                pass
    if not dates:
        return None
    dates.sort()
    return dates[len(dates) // 2]


class Snapshot:
    """One point in time: features, labels, and when they were captured."""

    def __init__(self, name, X, y, date, arm, source):
        self.name = name
        self.X = X
        self.y = y
        self.date = date
        self.arm = arm
        self.source = source

    def __len__(self):
        return len(self.y)

    def __repr__(self):
        d = self.date.date().isoformat() if self.date else "?"
        return f"<Snapshot {self.name} arm={self.arm} n={len(self)} date={d}>"


def load_snapshot(name, directory, arm):
    if not os.path.isdir(directory):
        return None
    csvs = [f for f in os.listdir(directory) if f.endswith(".csv")]
    if not csvs:
        return None
    X, y = extract_aggregated_features(directory, fail_on_high_skip=False)
    if X.empty:
        return None
    return Snapshot(name, X, y, _dir_median_date(directory), arm, directory)


def load_series(arm):
    """t0 from the spring collection, then one snapshot per longitudinal round."""
    snapshots = []
    spring = os.path.join(SPRING_DIR, f"{arm}_features")
    t0 = load_snapshot("t0-spring", spring, arm)
    if t0 is not None:
        snapshots.append(t0)

    if os.path.isdir(LONGITUDINAL_DIR):
        for rid in sorted(os.listdir(LONGITUDINAL_DIR)):
            d = os.path.join(LONGITUDINAL_DIR, rid, f"{arm}_features")
            snap = load_snapshot(rid, d, arm)
            if snap is not None:
                snapshots.append(snap)

    snapshots.sort(key=lambda s: s.date or datetime.min)
    return snapshots


def weeks_between(a, b):
    if a is None or b is None:
        return None
    return round((b - a).days / 7.0, 2)


def align_classes(train_y, test_X, test_y):
    """Restrict the test set to classes the model was actually trained on.

    A site that disappears or starts failing in a later round would otherwise be
    counted as a drift-induced error when it is really a collection gap.
    """
    known = set(np.unique(train_y))
    mask = np.array([label in known for label in test_y])
    return test_X[mask], test_y[mask], int((~mask).sum())


def fit_rf(X, y, seed):
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, random_state=seed, n_jobs=-1
    )
    model.fit(X, y)
    return model


def score(model, X, y):
    pred = model.predict(X)
    labels = np.sort(np.unique(y))
    recalls = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    per_class = {str(l): float(r) for l, r in zip(labels, recalls)}
    per_group = {}
    for group, sites in SITE_GROUPS.items():
        vals = [per_class[s] for s in sites if s in per_class]
        if vals:
            per_group[group] = round(float(np.mean(vals)) * 100, 2)
    return {
        "accuracy": round(accuracy_score(y, pred) * 100, 2),
        "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0) * 100, 2),
        "per_class_recall": {k: round(v * 100, 2) for k, v in per_class.items()},
        "per_group_recall": per_group,
        "n_test": int(len(y)),
    }


def decay_trend(rows):
    """Fit accuracy against weeks elapsed and report whether the slope is real.

    A decay curve drawn through three or four noisy points invites
    over-reading. This gives the slope in accuracy points per week with a
    confidence interval, so the thesis can state whether a decline is
    statistically supported rather than merely visible.
    """
    points = [(r["weeks_since_train"], r["accuracy"]) for r in rows
              if r.get("weeks_since_train") is not None]
    if len(points) < 3:
        return {"note": f"need at least 3 time points to fit a trend, have {len(points)}"}

    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    n = len(xs)
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    resid = ys - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    sxx = float(np.sum((xs - xs.mean()) ** 2))
    slope = float(slope)
    if n > 2 and sxx > 0:
        from scipy import stats as _st

        se = float(np.sqrt(ss_res / (n - 2) / sxx))
        tcrit = float(_st.t.ppf(0.975, n - 2))
        ci = [float(slope - tcrit * se), float(slope + tcrit * se)]
        if se > 0:
            t_stat = slope / se
            p_value = float(2 * (1 - _st.t.cdf(abs(t_stat), n - 2)))
        else:
            # An exact fit: the residuals are zero, so the slope is as well
            # determined as it can be. Undefined is the wrong answer here.
            p_value = 0.0 if slope != 0 else 1.0
    else:
        se, p_value, ci = float("nan"), float("nan"), [None, None]

    has_p = p_value == p_value  # False only for NaN
    return {
        "points": int(n),
        "slope_accuracy_points_per_week": round(slope, 4),
        # Plain floats, not numpy scalars: this dict is written straight to JSON.
        "slope_95ci": [None if c is None else round(float(c), 4) for c in ci],
        "p_value": round(p_value, 6) if has_p else None,
        "r_squared": round(float(r2), 4) if r2 == r2 else None,
        "significant_at_0.05": bool(has_p and p_value < 0.05),
    }


def mean_std(runs, key):
    vals = [r[key] for r in runs]
    return round(float(np.mean(vals)), 2), round(float(np.std(vals)), 2)


def frozen_features(t0):
    """Select the feature subset on t0 only.

    Selecting on the pooled series would leak future data into the model that
    is supposed to be ignorant of it, which flatters the no-retraining policy.
    """
    feats, _ = select_top_features_mutual_info(t0.X, t0.y, top_k=TOP_K_FEATURES, random_state=42)
    return feats or list(FEATURE_NAMES)


def policy_no_retrain(snapshots, feats):
    """Train once on t0, never update.  The decay curve of interest."""
    t0 = snapshots[0]
    results = []
    for target in snapshots[1:]:
        runs = []
        for seed in SEEDS:
            model = fit_rf(t0.X[feats], t0.y, seed)
            Xt, yt, dropped = align_classes(t0.y, target.X[feats], target.y)
            if len(yt) == 0:
                continue
            m = score(model, Xt, yt)
            m["dropped_unknown_class_samples"] = dropped
            runs.append(m)
        if not runs:
            continue
        acc, acc_sd = mean_std(runs, "accuracy")
        f1, f1_sd = mean_std(runs, "macro_f1")
        results.append({
            "round": target.name,
            "weeks_since_train": weeks_between(t0.date, target.date),
            "accuracy": acc, "accuracy_std": acc_sd,
            "macro_f1": f1, "macro_f1_std": f1_sd,
            "per_group_recall": runs[0]["per_group_recall"],
            "per_class_recall": runs[0]["per_class_recall"],
            "n_test": runs[0]["n_test"],
            "train_samples_used": int(len(t0)),
        })
    return results


def policy_within_round(snapshots, feats):
    """Train and test inside the same snapshot: the no-drift ceiling.

    Every no-retraining number has to be read against this, otherwise a low
    accuracy could just mean the classifier was never good.
    """
    results = []
    for snap in snapshots:
        if len(snap) < 20:
            print(f"    [skip ceiling] {snap.name}: only {len(snap)} samples")
            continue
        counts = pd.Series(snap.y).value_counts()
        if counts.min() < 2:
            thin = counts[counts < 2].index.tolist()
            print(f"    [skip ceiling] {snap.name}: {len(thin)} class(es) with a single "
                  f"sample, cannot stratify ({', '.join(map(str, thin[:5]))})")
            continue
        runs = []
        for seed in SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(
                snap.X[feats], snap.y, test_size=0.2, random_state=seed, stratify=snap.y
            )
            runs.append(score(fit_rf(Xtr, ytr, seed), Xte, yte))
        acc, acc_sd = mean_std(runs, "accuracy")
        f1, _ = mean_std(runs, "macro_f1")
        results.append({
            "round": snap.name,
            "date": snap.date.date().isoformat() if snap.date else None,
            "accuracy": acc, "accuracy_std": acc_sd, "macro_f1": f1,
            "n_samples": len(snap),
        })
    return results


def policy_periodic_retrain(snapshots, feats, every_k):
    """Retrain every k rounds on everything collected so far.

    Reports the labelling cost alongside the accuracy, since the practical
    question is not whether retraining helps but what it costs.
    """
    results = []
    model_cache = {}
    for i in range(1, len(snapshots)):
        # Index of the newest snapshot the model is allowed to have seen: the
        # most recent retraining point at or before the previous round.
        train_upto = ((i - 1) // every_k) * every_k
        train_snaps = snapshots[: train_upto + 1]
        key = train_upto
        if key not in model_cache:
            Xtr = pd.concat([s.X[feats] for s in train_snaps], ignore_index=True)
            ytr = np.concatenate([s.y for s in train_snaps])
            model_cache[key] = (fit_rf(Xtr, ytr, 42), ytr, len(ytr))
        model, ytr, n_train = model_cache[key]
        target = snapshots[i]
        Xt, yt, dropped = align_classes(ytr, target.X[feats], target.y)
        if len(yt) == 0:
            continue
        m = score(model, Xt, yt)
        results.append({
            "round": target.name,
            "weeks_since_train": weeks_between(snapshots[train_upto].date, target.date),
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "train_samples_used": int(n_train),
            "trained_through": snapshots[train_upto].name,
            "dropped_unknown_class_samples": dropped,
        })
    return results


def policy_sliding_window(snapshots, feats, window):
    """Retrain every round on the most recent `window` rounds only."""
    results = []
    for i in range(1, len(snapshots)):
        train_snaps = snapshots[max(0, i - window): i]
        Xtr = pd.concat([s.X[feats] for s in train_snaps], ignore_index=True)
        ytr = np.concatenate([s.y for s in train_snaps])
        target = snapshots[i]
        Xt, yt, _ = align_classes(ytr, target.X[feats], target.y)
        if len(yt) == 0:
            continue
        m = score(fit_rf(Xtr, ytr, 42), Xt, yt)
        results.append({
            "round": target.name,
            "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            "train_samples_used": int(len(ytr)),
            "window_rounds": len(train_snaps),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="baseline", choices=["baseline", "obfs4"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    snapshots = load_series(args.arm)
    if not snapshots:
        sys.exit(f"No feature snapshots found for arm '{args.arm}'.")

    print(f"=== drift evaluation, arm={args.arm} ===")
    for s in snapshots:
        print(f"    {s}")

    t0 = snapshots[0]
    feats = frozen_features(t0)
    print(f"    frozen feature set (selected on t0 only): {feats}")

    payload = {
        "arm": args.arm,
        "generated_at": datetime.now().isoformat(),
        "frozen_features": feats,
        "snapshots": [
            {"name": s.name, "n": len(s), "classes": int(len(set(s.y))),
             "date": s.date.date().isoformat() if s.date else None}
            for s in snapshots
        ],
        "within_round_ceiling": policy_within_round(snapshots, feats),
    }

    if len(snapshots) < 2:
        payload["note"] = (
            "Only t0 is present, so no drift can be measured yet. The within-round "
            "ceiling is the reference the first longitudinal round will be read against."
        )
        print("\n[!] Only one snapshot. Collect a round, then rerun for the decay curve.")
    else:
        payload["no_retrain"] = policy_no_retrain(snapshots, feats)
        payload["decay_trend"] = decay_trend(payload["no_retrain"])
        payload["periodic_retrain"] = {
            f"every_{k}_rounds": policy_periodic_retrain(snapshots, feats, k)
            for k in (1, 2, 4) if len(snapshots) > k
        }
        payload["sliding_window"] = {
            f"window_{w}": policy_sliding_window(snapshots, feats, w)
            for w in (1, 3) if len(snapshots) > w
        }

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = args.out or os.path.join(FIGURES_DIR, f"drift_metrics_{args.arm}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n--- within-round ceiling (no drift) ---")
    for r in payload["within_round_ceiling"]:
        print(f"    {r['round']:14s} acc={r['accuracy']:6.2f}%  n={r['n_samples']}")
    if payload.get("no_retrain"):
        print("\n--- t0 model, no retraining ---")
        for r in payload["no_retrain"]:
            print(f"    {r['round']:14s} +{r['weeks_since_train']:>5} weeks  "
                  f"acc={r['accuracy']:6.2f}%  macroF1={r['macro_f1']:6.2f}%  "
                  f"groups={r['per_group_recall']}")
    trend = payload.get("decay_trend") or {}
    if trend.get("points"):
        sig = "significant" if trend["significant_at_0.05"] else "NOT significant"
        print(f"\n--- decay trend over {trend['points']} time points ---")
        print(f"    slope: {trend['slope_accuracy_points_per_week']} accuracy points per week")
        print(f"    95% CI: {trend['slope_95ci']}   p={trend['p_value']}   "
              f"R2={trend['r_squared']}   ({sig})")
    elif trend.get("note"):
        print(f"\n--- decay trend: {trend['note']} ---")

    for label, block in (("periodic retraining", payload.get("periodic_retrain")),
                         ("sliding window", payload.get("sliding_window"))):
        if not block:
            continue
        print(f"\n--- {label} ---")
        for policy, rows in block.items():
            if not rows:
                continue
            last = rows[-1]
            mean_acc = round(float(np.mean([r["accuracy"] for r in rows])), 2)
            print(f"    {policy:18s} mean acc={mean_acc:6.2f}%  "
                  f"final acc={last['accuracy']:6.2f}%  "
                  f"labelled traces used={last['train_samples_used']}")

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
