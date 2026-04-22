import json
import os
from glob import glob

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

FEATURE_NAMES = [
    "total_packets",
    "incoming_packets",
    "outgoing_packets",
    "in_out_packet_ratio",
    "total_bytes",
    "incoming_bytes",
    "outgoing_bytes",
    "in_out_byte_ratio",
    "duration",
    "mean_inter_arrival",
    "std_inter_arrival",
    "median_inter_arrival",
    "p90_inter_arrival",
    "max_packet_size",
    "mean_packet_size",
    "std_packet_size",
    "incoming_mean_packet_size",
    "outgoing_mean_packet_size",
    "incoming_std_packet_size",
    "outgoing_std_packet_size",
    "direction_changes",
]


def parse_site_name(file_path):
    base = os.path.splitext(os.path.basename(file_path))[0]
    parts = base.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else base.split("_")[0]


def extract_aggregated_features(directory, force_label=None):
    X, y = [], []
    csv_files = glob(os.path.join(directory, "*.csv"))

    for file_path in csv_files:
        site_name = force_label if force_label else parse_site_name(file_path)
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue

            dirs = df["direction_size"].values
            times = df["time_offset"].values
            iats = df["inter_arrival_time"].values
            incoming = dirs[dirs < 0]
            outgoing = dirs[dirs > 0]

            features = [
                len(dirs),
                len(incoming),
                len(outgoing),
                len(incoming) / (len(outgoing) + 1e-5),
                np.sum(np.abs(dirs)),
                np.abs(np.sum(incoming)) if len(incoming) > 0 else 0,
                np.sum(outgoing) if len(outgoing) > 0 else 0,
                np.abs(np.sum(incoming)) / (np.sum(outgoing) + 1e-5),
                times[-1] if len(times) > 0 else 0,
                np.mean(iats) if len(iats) > 0 else 0,
                np.std(iats) if len(iats) > 0 else 0,
                np.median(iats) if len(iats) > 0 else 0,
                np.percentile(iats, 90) if len(iats) > 0 else 0,
                np.max(np.abs(dirs)),
                np.mean(np.abs(dirs)),
                np.std(np.abs(dirs)),
                np.mean(np.abs(incoming)) if len(incoming) > 0 else 0,
                np.mean(outgoing) if len(outgoing) > 0 else 0,
                np.std(np.abs(incoming)) if len(incoming) > 1 else 0,
                np.std(outgoing) if len(outgoing) > 1 else 0,
                np.sum(np.diff(np.sign(dirs)) != 0) if len(dirs) > 1 else 0,
            ]
            X.append(features)
            y.append(site_name)
        except Exception:
            pass

    return pd.DataFrame(X, columns=FEATURE_NAMES), np.array(y)


def select_stable_top_features(
    X,
    y,
    top_k=10,
    n_runs=8,
    n_estimators=300,
    out_path=None,
):
    importances = []

    unique_labels, counts = np.unique(y, return_counts=True)
    _ = unique_labels
    min_count = np.min(counts) if len(counts) > 0 else 0

    if min_count < 2:
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X, y)
        importances.append(rf.feature_importances_)
    else:
        for seed in range(n_runs):
            X_train, _, y_train, _ = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42 + seed,
                stratify=y,
            )
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42 + seed,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            importances.append(rf.feature_importances_)

    imp = np.array(importances)
    mean_imp = imp.mean(axis=0)
    std_imp = imp.std(axis=0)

    ranking_df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": mean_imp,
            "importance_std": std_imp,
        }
    ).sort_values("importance_mean", ascending=False)

    top_features = ranking_df.head(top_k)["feature"].tolist()

    if out_path:
        payload = {
            "top_features": top_features,
            "ranking": ranking_df.to_dict(orient="records"),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return top_features, ranking_df


def load_locked_top_features(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    features = payload.get("top_features", [])
    return features if features else None
