import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

from feature_pool import (
    extract_aggregated_features,
    load_locked_top_features,
    select_stable_top_features,
)

# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tor_dataset", "extracted_features")
)
BASELINE_DIR = os.path.join(BASE_DIR, "baseline_features")
OBFS4_DIR = os.path.join(BASE_DIR, "obfs4_features")
OTHER_DIR = os.path.join(BASE_DIR, "other_features")
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
RF_METRICS_PATH = os.path.join(FIGURES_DIR, "metrics_rf.json")
TOP_FEATURES_PATH = os.path.join(FIGURES_DIR, "selected_top10_features.json")
RF_N_ESTIMATORS = 300
TOP_K_FEATURES = 10


def evaluate_metrics(y_true, y_pred):
    labels = np.sort(np.unique(y_true))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        "per_class_recall": {
            str(lbl): round(float(rec) * 100, 2) for lbl, rec in zip(labels, recalls)
        },
    }


def print_metrics(name, metrics):
    print(f"[{name}] Accuracy: {metrics['accuracy']:.2f}% | Macro-F1: {metrics['macro_f1']:.2f}%")


def select_top_features(X_base, y_base, X_obfs, y_obfs, X_other):
    locked = load_locked_top_features(TOP_FEATURES_PATH)
    if locked:
        print("Using locked top-10 features from previous run.")
        return locked

    feature_frames = [X_base, X_obfs]
    labels = [y_base, y_obfs]

    if not X_other.empty:
        feature_frames.append(X_other)
        labels.append(np.array(["other"] * len(X_other)))

    X_all = pd.concat(feature_frames, ignore_index=True)
    y_all = np.concatenate(labels)

    top_features, ranking_df = select_stable_top_features(
        X_all,
        y_all,
        top_k=TOP_K_FEATURES,
        n_runs=8,
        n_estimators=RF_N_ESTIMATORS,
        out_path=TOP_FEATURES_PATH,
    )
    print("Selected stable top-10 features:")
    for i, feat in enumerate(top_features, start=1):
        print(f"  {i}. {feat}")
    ranking_df.to_json(
        os.path.join(FIGURES_DIR, "feature_stability_rf.json"), orient="records", indent=2
    )
    return top_features


def main():
    print("Extracting features from Baseline traffic...")
    X_base, y_base = extract_aggregated_features(BASELINE_DIR)

    print("Extracting features from Obfs4 traffic...")
    X_obfs, y_obfs = extract_aggregated_features(OBFS4_DIR)

    X_other = pd.DataFrame()
    if os.path.isdir(OTHER_DIR):
        print("Extracting features from Other traffic...")
        X_other, _ = extract_aggregated_features(OTHER_DIR, force_label="other")

    if X_base.empty or X_obfs.empty:
        print("Error: Could not find baseline/obfs4 dataset files.")
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)
    top_features = select_top_features(X_base, y_base, X_obfs, y_obfs, X_other)

    X_base = X_base[top_features]
    X_obfs = X_obfs[top_features]
    if not X_other.empty:
        X_other = X_other[top_features]

    print("\n" + "=" * 65)
    print(" RANDOM FOREST RESULTS (Top-10 Stable Features + Open-World)")
    print("=" * 65)

    scenarios = {}

    # 1. BASELINE
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_base, y_base, test_size=0.2, random_state=42, stratify=y_base
    )
    rf_base = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, random_state=42, n_jobs=-1
    )
    rf_base.fit(X_train_b, y_train_b)
    y_pred_b = rf_base.predict(X_test_b)
    scenarios["baseline"] = evaluate_metrics(y_test_b, y_pred_b)
    print_metrics("1. Baseline", scenarios["baseline"])

    # 2. OBFS4
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
        X_obfs, y_obfs, test_size=0.2, random_state=42, stratify=y_obfs
    )
    rf_obfs = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, random_state=42, n_jobs=-1
    )
    rf_obfs.fit(X_train_o, y_train_o)
    y_pred_o = rf_obfs.predict(X_test_o)
    scenarios["obfs4"] = evaluate_metrics(y_test_o, y_pred_o)
    print_metrics("2. Obfs4", scenarios["obfs4"])

    # 3. ZERO-SHOT (Train: Baseline -> Test: Obfs4)
    y_pred_z = rf_base.predict(X_obfs)
    scenarios["zero_shot"] = evaluate_metrics(y_obfs, y_pred_z)
    print_metrics("3. Zero-Shot", scenarios["zero_shot"])

    # 4-6. OPEN-WORLD (if Other exists)
    if not X_other.empty and len(X_other) > 4:
        y_other = np.array(["other"] * len(X_other))

        X_open_b = pd.concat([X_base, X_other], ignore_index=True)
        y_open_b = np.concatenate([y_base, y_other])
        X_train_ob, X_test_ob, y_train_ob, y_test_ob = train_test_split(
            X_open_b, y_open_b, test_size=0.2, random_state=42, stratify=y_open_b
        )
        rf_open_b = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, random_state=42, n_jobs=-1
        )
        rf_open_b.fit(X_train_ob, y_train_ob)
        scenarios["open_world_baseline"] = evaluate_metrics(
            y_test_ob, rf_open_b.predict(X_test_ob)
        )
        print_metrics("4. Open-World Baseline", scenarios["open_world_baseline"])

        X_open_o = pd.concat([X_obfs, X_other], ignore_index=True)
        y_open_o = np.concatenate([y_obfs, y_other])
        X_train_oo, X_test_oo, y_train_oo, y_test_oo = train_test_split(
            X_open_o, y_open_o, test_size=0.2, random_state=42, stratify=y_open_o
        )
        rf_open_o = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, random_state=42, n_jobs=-1
        )
        rf_open_o.fit(X_train_oo, y_train_oo)
        scenarios["open_world_obfs4"] = evaluate_metrics(
            y_test_oo, rf_open_o.predict(X_test_oo)
        )
        print_metrics("5. Open-World Obfs4", scenarios["open_world_obfs4"])

        X_other_train, X_other_test = train_test_split(
            X_other, test_size=0.5, random_state=42
        )
        y_other_train = np.array(["other"] * len(X_other_train))
        y_other_test = np.array(["other"] * len(X_other_test))

        X_train_oz = pd.concat([X_base, X_other_train], ignore_index=True)
        y_train_oz = np.concatenate([y_base, y_other_train])
        X_test_oz = pd.concat([X_obfs, X_other_test], ignore_index=True)
        y_test_oz = np.concatenate([y_obfs, y_other_test])

        rf_open_z = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, random_state=42, n_jobs=-1
        )
        rf_open_z.fit(X_train_oz, y_train_oz)
        scenarios["open_world_zero_shot"] = evaluate_metrics(
            y_test_oz, rf_open_z.predict(X_test_oz)
        )
        print_metrics("6. Open-World Zero-Shot", scenarios["open_world_zero_shot"])
    else:
        print("[Open-World] Skipped: no Other dataset found in extracted_features/other_features")

    print("=" * 65)

    with open(RF_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": scenarios["baseline"]["accuracy"],
                "obfs4": scenarios["obfs4"]["accuracy"],
                "zero_shot": scenarios["zero_shot"]["accuracy"],
                "top_features": top_features,
                "scenarios": scenarios,
                "n_estimators": RF_N_ESTIMATORS,
            },
            f,
            indent=2,
        )
    print(f"Saved RF metrics to: {RF_METRICS_PATH}")


if __name__ == "__main__":
    main()
