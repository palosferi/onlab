import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split

from feature_pool import (
    extract_aggregated_features,
    select_stable_top_features,
    select_top_features_mutual_info,
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

RF_N_ESTIMATORS = 300
TOP_K_FEATURES = 10
SEEDS = [42, 52, 62]
PRIMARY_SEED = 42


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


def evaluate_rf_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_metrics(y_test, y_pred), y_pred


def select_shared_features(X_pool_train, y_pool_train):
    top_features, _ = select_top_features_mutual_info(
        X_pool_train,
        y_pool_train,
        top_k=TOP_K_FEATURES,
        random_state=42,
    )
    return top_features


def select_rf_optimized_features(X_train, y_train):
    top_features, _ = select_stable_top_features(
        X_train,
        y_train,
        top_k=TOP_K_FEATURES,
        n_runs=4,
        n_estimators=RF_N_ESTIMATORS,
        out_path=None,
    )
    return top_features


def split_other_dataset(X_other, seed):
    if X_other.empty or len(X_other) < 2:
        return None, None, None, None

    X_train, X_test = train_test_split(X_other, test_size=0.5, random_state=seed)
    y_train = np.array(["other"] * len(X_train))
    y_test = np.array(["other"] * len(X_test))
    return X_train, X_test, y_train, y_test


def scenario_stats_from_runs(runs):
    scenario_names = sorted({name for run in runs for name in run.keys()})
    stats = {}

    for scenario in scenario_names:
        acc_vals = [run[scenario]["accuracy"] for run in runs if scenario in run]
        f1_vals = [run[scenario]["macro_f1"] for run in runs if scenario in run]
        stats[scenario] = {
            "accuracy_mean": round(float(np.mean(acc_vals)), 2),
            "accuracy_std": round(float(np.std(acc_vals)), 2),
            "macro_f1_mean": round(float(np.mean(f1_vals)), 2),
            "macro_f1_std": round(float(np.std(f1_vals)), 2),
            "n_runs": len(acc_vals),
        }

    return stats


def build_confusion_payload(y_true, y_pred):
    labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": [str(x) for x in labels.tolist()],
        "matrix": cm.tolist(),
    }


def run_track_for_seed(
    seed,
    track_name,
    X_base,
    y_base,
    X_obfs,
    y_obfs,
    X_other,
):
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_base, y_base, test_size=0.2, random_state=seed, stratify=y_base
    )
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
        X_obfs, y_obfs, test_size=0.2, random_state=seed, stratify=y_obfs
    )

    X_other_train, X_other_test, y_other_train, y_other_test = split_other_dataset(X_other, seed)

    if X_other_train is not None:
        X_pool_train = pd.concat([X_train_b, X_train_o, X_other_train], ignore_index=True)
        y_pool_train = np.concatenate([y_train_b, y_train_o, y_other_train])
    else:
        X_pool_train = pd.concat([X_train_b, X_train_o], ignore_index=True)
        y_pool_train = np.concatenate([y_train_b, y_train_o])

    if track_name == "shared":
        shared_features = select_shared_features(X_pool_train, y_pool_train)
        feature_map = {
            "baseline": shared_features,
            "obfs4": shared_features,
            "zero_shot": shared_features,
            "open_world_baseline": shared_features,
            "open_world_obfs4": shared_features,
            "open_world_zero_shot": shared_features,
        }
    else:
        feature_map = {
            "baseline": select_rf_optimized_features(X_train_b, y_train_b),
            "obfs4": select_rf_optimized_features(X_train_o, y_train_o),
            "zero_shot": select_rf_optimized_features(X_train_b, y_train_b),
        }
        if X_other_train is not None:
            X_train_ob = pd.concat([X_train_b, X_other_train], ignore_index=True)
            y_train_ob = np.concatenate([y_train_b, y_other_train])
            X_train_oo = pd.concat([X_train_o, X_other_train], ignore_index=True)
            y_train_oo = np.concatenate([y_train_o, y_other_train])
            feature_map["open_world_baseline"] = select_rf_optimized_features(X_train_ob, y_train_ob)
            feature_map["open_world_obfs4"] = select_rf_optimized_features(X_train_oo, y_train_oo)
            feature_map["open_world_zero_shot"] = select_rf_optimized_features(X_train_ob, y_train_ob)

    scenarios = {}
    confusion_payload = {}

    # 1. BASELINE
    feats = feature_map["baseline"]
    m_base, pred_base = evaluate_rf_model(
        X_train_b[feats], y_train_b, X_test_b[feats], y_test_b
    )
    scenarios["baseline"] = m_base
    confusion_payload["baseline"] = build_confusion_payload(y_test_b, pred_base)

    # 2. OBFS4
    feats = feature_map["obfs4"]
    m_obfs, pred_obfs = evaluate_rf_model(
        X_train_o[feats], y_train_o, X_test_o[feats], y_test_o
    )
    scenarios["obfs4"] = m_obfs
    confusion_payload["obfs4"] = build_confusion_payload(y_test_o, pred_obfs)

    # 3. ZERO-SHOT (Train: Baseline -> Test: Obfs4 test split)
    feats = feature_map["zero_shot"]
    rf_zero = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=42,
        n_jobs=-1,
    )
    rf_zero.fit(X_train_b[feats], y_train_b)
    pred_zero = rf_zero.predict(X_test_o[feats])
    scenarios["zero_shot"] = evaluate_metrics(y_test_o, pred_zero)
    confusion_payload["zero_shot"] = build_confusion_payload(y_test_o, pred_zero)

    if X_other_train is not None and len(X_other_test) > 0:
        # 4. OPEN-WORLD BASELINE
        feats = feature_map["open_world_baseline"]
        X_train_ob = pd.concat([X_train_b, X_other_train], ignore_index=True)
        y_train_ob = np.concatenate([y_train_b, y_other_train])
        X_test_ob = pd.concat([X_test_b, X_other_test], ignore_index=True)
        y_test_ob = np.concatenate([y_test_b, y_other_test])
        m_ob, pred_ob = evaluate_rf_model(
            X_train_ob[feats], y_train_ob, X_test_ob[feats], y_test_ob
        )
        scenarios["open_world_baseline"] = m_ob
        confusion_payload["open_world_baseline"] = build_confusion_payload(y_test_ob, pred_ob)

        # 5. OPEN-WORLD OBFS4
        feats = feature_map["open_world_obfs4"]
        X_train_oo = pd.concat([X_train_o, X_other_train], ignore_index=True)
        y_train_oo = np.concatenate([y_train_o, y_other_train])
        X_test_oo = pd.concat([X_test_o, X_other_test], ignore_index=True)
        y_test_oo = np.concatenate([y_test_o, y_other_test])
        m_oo, pred_oo = evaluate_rf_model(
            X_train_oo[feats], y_train_oo, X_test_oo[feats], y_test_oo
        )
        scenarios["open_world_obfs4"] = m_oo
        confusion_payload["open_world_obfs4"] = build_confusion_payload(y_test_oo, pred_oo)

        # 6. OPEN-WORLD ZERO-SHOT
        feats = feature_map["open_world_zero_shot"]
        rf_oz = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=42,
            n_jobs=-1,
        )
        rf_oz.fit(X_train_ob[feats], y_train_ob)
        X_test_oz = pd.concat([X_test_o, X_other_test], ignore_index=True)
        y_test_oz = np.concatenate([y_test_o, y_other_test])
        pred_oz = rf_oz.predict(X_test_oz[feats])
        scenarios["open_world_zero_shot"] = evaluate_metrics(y_test_oz, pred_oz)
        confusion_payload["open_world_zero_shot"] = build_confusion_payload(y_test_oz, pred_oz)

    return scenarios, feature_map, confusion_payload


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

    tracks = {}
    for track_name in ["shared", "optimized"]:
        print(f"\nRunning RF track: {track_name}")
        run_metrics = []
        primary_features = None
        primary_scenarios = None
        primary_confusions = None

        for seed in SEEDS:
            scenarios, feature_map, confusions = run_track_for_seed(
                seed,
                track_name,
                X_base,
                y_base,
                X_obfs,
                y_obfs,
                X_other,
            )
            run_metrics.append(scenarios)

            if seed == PRIMARY_SEED:
                primary_scenarios = scenarios
                primary_confusions = confusions
                if track_name == "shared":
                    primary_features = feature_map["baseline"]
                else:
                    primary_features = {
                        "baseline": feature_map.get("baseline", []),
                        "obfs4": feature_map.get("obfs4", []),
                        "zero_shot": feature_map.get("zero_shot", []),
                        "open_world_baseline": feature_map.get("open_world_baseline", []),
                        "open_world_obfs4": feature_map.get("open_world_obfs4", []),
                        "open_world_zero_shot": feature_map.get("open_world_zero_shot", []),
                    }

        tracks[track_name] = {
            "scenario_stats": scenario_stats_from_runs(run_metrics),
            "primary_seed": PRIMARY_SEED,
            "primary_seed_scenarios": primary_scenarios,
            "primary_seed_confusion_matrices": primary_confusions,
            "primary_seed_features": primary_features,
        }

    primary_shared = tracks["shared"]["primary_seed_scenarios"]

    payload = {
        "primary_track": "shared",
        "seeds": SEEDS,
        "n_estimators": RF_N_ESTIMATORS,
        "tracks": tracks,
        # Backward-compatible fields used by figure scripts.
        "baseline": primary_shared["baseline"]["accuracy"],
        "obfs4": primary_shared["obfs4"]["accuracy"],
        "zero_shot": primary_shared["zero_shot"]["accuracy"],
        "top_features": tracks["shared"]["primary_seed_features"],
        "scenarios": primary_shared,
    }

    with open(RF_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved RF metrics to: {RF_METRICS_PATH}")


if __name__ == "__main__":
    main()
