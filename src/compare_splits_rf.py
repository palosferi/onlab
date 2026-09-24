"""How much does the random stratified split inflate the RF result?

evaluate_all_rf.py splits with train_test_split(..., stratify=y), so traces
captured minutes apart on the same circuit can land on both sides of the
boundary. This script runs the same Random Forest twice -- once with that
random split, once with a chronological one -- so the gap between them can be
quoted as a number instead of asserted.
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from df_dataset import chronological_split
from feature_pool import extract_aggregated_features, select_stable_top_features

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tor_dataset", "extracted_features")
)
FIGURES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "figures")
)

ARMS = {
    "baseline": os.path.join(BASE_DIR, "baseline_features"),
    "obfs4": os.path.join(BASE_DIR, "obfs4_features"),
}


def score(X, y, train_idx, test_idx, seed, n_estimators):
    rf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=seed, n_jobs=-1
    )
    rf.fit(X.iloc[train_idx], y[train_idx])
    pred = rf.predict(X.iloc[test_idx])
    return {
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "macro_f1": float(
            f1_score(y[test_idx], pred, average="macro", zero_division=0)
        ),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out", default=os.path.join(FIGURES_DIR, "metrics_split_rf.json"))
    args = p.parse_args()

    results = {"config": vars(args), "arms": {}}

    for arm, directory in ARMS.items():
        X, y, stamps = extract_aggregated_features(directory, return_timestamps=True)
        top_features, _ = select_stable_top_features(X, y, top_k=args.top_k)
        Xk = X[top_features]

        idx = np.arange(len(y))
        rand_train, rand_test = train_test_split(
            idx, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        chrono_train, chrono_test = chronological_split(
            y, stamps, test_size=args.test_size, per_class=True
        )

        random_res = score(Xk, y, rand_train, rand_test, args.seed, args.n_estimators)
        chrono_res = score(
            Xk, y, chrono_train, chrono_test, args.seed, args.n_estimators
        )
        delta = random_res["accuracy"] - chrono_res["accuracy"]

        results["arms"][arm] = {
            "n_traces": int(len(y)),
            "n_classes": int(len(set(y))),
            "top_features": top_features,
            "random_split": random_res,
            "chronological_split": chrono_res,
            "accuracy_inflation": float(delta),
        }

        print(f"[{arm}] {len(y)} traces, {len(set(y))} classes")
        print(
            f"  random split        acc {random_res['accuracy']:.4f}  "
            f"macro-F1 {random_res['macro_f1']:.4f}"
        )
        print(
            f"  chronological split acc {chrono_res['accuracy']:.4f}  "
            f"macro-F1 {chrono_res['macro_f1']:.4f}"
        )
        print(f"  -> random split inflates accuracy by {delta:+.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
