import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from feature_pool import extract_aggregated_features, select_top_features_mutual_info

# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tor_dataset", "extracted_features")
)
BASELINE_DIR = os.path.join(BASE_DIR, "baseline_features")
OBFS4_DIR = os.path.join(BASE_DIR, "obfs4_features")
OTHER_DIR = os.path.join(BASE_DIR, "other_features")
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
DL_METRICS_PATH = os.path.join(FIGURES_DIR, "metrics_dl.json")

TOP_K_FEATURES = 10
SEEDS = [42, 52, 62]
PRIMARY_SEED = 42
DL_EPOCHS = 30


class TripletDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = np.array(y)
        self.labels_set = set(self.y)
        self.label_to_indices = {
            label: np.where(self.y == label)[0] for label in self.labels_set
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        anchor = self.X[index]
        anchor_label = self.y[index]

        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[anchor_label])
        positive = self.X[positive_index]

        negative_label = np.random.choice(list(self.labels_set - {anchor_label}))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        negative = self.X[negative_index]

        return anchor, positive, negative


class TripletMLP(nn.Module):
    def __init__(self, input_size):
        super(TripletMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
        )

    def forward(self, x):
        return nn.functional.normalize(self.fc(x), p=2, dim=1)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


def build_confusion_payload(y_true, y_pred):
    labels = np.sort(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "labels": [str(x) for x in labels.tolist()],
        "matrix": cm.tolist(),
    }


def train_embedding_model(X_train, y_train, seed):
    set_global_seed(seed)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripletMLP(input_size=X_train_scaled.shape[1]).to(device)

    dataset = TripletDataset(X_train_scaled, y_train)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=generator)

    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(DL_EPOCHS):
        for anchor, positive, negative in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            loss = criterion(model(anchor), model(positive), model(negative))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_emb = model(torch.FloatTensor(X_train_scaled).to(device)).cpu().numpy()

    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(train_emb, y_train)
    return scaler, model, device, knn


def predict_labels(scaler, model, device, knn, X_test):
    model.eval()
    with torch.no_grad():
        X_scaled = scaler.transform(X_test)
        emb = model(torch.FloatTensor(X_scaled).to(device)).cpu().numpy()
    return knn.predict(emb)


def select_shared_features(X_pool_train, y_pool_train):
    top_features, _ = select_top_features_mutual_info(
        X_pool_train,
        y_pool_train,
        top_k=TOP_K_FEATURES,
        random_state=42,
    )
    return top_features


def select_dl_optimized_features(X_train, y_train):
    top_features, _ = select_top_features_mutual_info(
        X_train,
        y_train,
        top_k=TOP_K_FEATURES,
        random_state=42,
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


def evaluate_dl_model(X_train, y_train, X_test, y_test, seed):
    scaler, model, device, knn = train_embedding_model(X_train, y_train, seed=seed)
    pred = predict_labels(scaler, model, device, knn, X_test)
    return evaluate_metrics(y_test, pred), pred


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
            "baseline": select_dl_optimized_features(X_train_b, y_train_b),
            "obfs4": select_dl_optimized_features(X_train_o, y_train_o),
            "zero_shot": select_dl_optimized_features(X_train_b, y_train_b),
        }
        if X_other_train is not None:
            X_train_ob = pd.concat([X_train_b, X_other_train], ignore_index=True)
            y_train_ob = np.concatenate([y_train_b, y_other_train])
            X_train_oo = pd.concat([X_train_o, X_other_train], ignore_index=True)
            y_train_oo = np.concatenate([y_train_o, y_other_train])
            feature_map["open_world_baseline"] = select_dl_optimized_features(X_train_ob, y_train_ob)
            feature_map["open_world_obfs4"] = select_dl_optimized_features(X_train_oo, y_train_oo)
            feature_map["open_world_zero_shot"] = select_dl_optimized_features(X_train_ob, y_train_ob)

    scenarios = {}
    confusion_payload = {}

    # 1. BASELINE
    feats = feature_map["baseline"]
    m_base, pred_base = evaluate_dl_model(
        X_train_b[feats].values, y_train_b, X_test_b[feats].values, y_test_b, seed
    )
    scenarios["baseline"] = m_base
    confusion_payload["baseline"] = build_confusion_payload(y_test_b, pred_base)

    # 2. OBFS4
    feats = feature_map["obfs4"]
    m_obfs, pred_obfs = evaluate_dl_model(
        X_train_o[feats].values, y_train_o, X_test_o[feats].values, y_test_o, seed
    )
    scenarios["obfs4"] = m_obfs
    confusion_payload["obfs4"] = build_confusion_payload(y_test_o, pred_obfs)

    # 3. ZERO-SHOT (Train: Baseline -> Test: Obfs4 test split)
    feats = feature_map["zero_shot"]
    scaler_z, model_z, device_z, knn_z = train_embedding_model(X_train_b[feats].values, y_train_b, seed=seed)
    pred_zero = predict_labels(scaler_z, model_z, device_z, knn_z, X_test_o[feats].values)
    scenarios["zero_shot"] = evaluate_metrics(y_test_o, pred_zero)
    confusion_payload["zero_shot"] = build_confusion_payload(y_test_o, pred_zero)

    if X_other_train is not None and len(X_other_test) > 0:
        # 4. OPEN-WORLD BASELINE
        feats = feature_map["open_world_baseline"]
        X_train_ob = pd.concat([X_train_b, X_other_train], ignore_index=True)
        y_train_ob = np.concatenate([y_train_b, y_other_train])
        X_test_ob = pd.concat([X_test_b, X_other_test], ignore_index=True)
        y_test_ob = np.concatenate([y_test_b, y_other_test])
        m_ob, pred_ob = evaluate_dl_model(
            X_train_ob[feats].values,
            y_train_ob,
            X_test_ob[feats].values,
            y_test_ob,
            seed,
        )
        scenarios["open_world_baseline"] = m_ob
        confusion_payload["open_world_baseline"] = build_confusion_payload(y_test_ob, pred_ob)

        # 5. OPEN-WORLD OBFS4
        feats = feature_map["open_world_obfs4"]
        X_train_oo = pd.concat([X_train_o, X_other_train], ignore_index=True)
        y_train_oo = np.concatenate([y_train_o, y_other_train])
        X_test_oo = pd.concat([X_test_o, X_other_test], ignore_index=True)
        y_test_oo = np.concatenate([y_test_o, y_other_test])
        m_oo, pred_oo = evaluate_dl_model(
            X_train_oo[feats].values,
            y_train_oo,
            X_test_oo[feats].values,
            y_test_oo,
            seed,
        )
        scenarios["open_world_obfs4"] = m_oo
        confusion_payload["open_world_obfs4"] = build_confusion_payload(y_test_oo, pred_oo)

        # 6. OPEN-WORLD ZERO-SHOT
        feats = feature_map["open_world_zero_shot"]
        X_train_oz = pd.concat([X_train_b, X_other_train], ignore_index=True)
        y_train_oz = np.concatenate([y_train_b, y_other_train])
        X_test_oz = pd.concat([X_test_o, X_other_test], ignore_index=True)
        y_test_oz = np.concatenate([y_test_o, y_other_test])
        scaler_oz, model_oz, device_oz, knn_oz = train_embedding_model(
            X_train_oz[feats].values,
            y_train_oz,
            seed=seed,
        )
        pred_oz = predict_labels(scaler_oz, model_oz, device_oz, knn_oz, X_test_oz[feats].values)
        scenarios["open_world_zero_shot"] = evaluate_metrics(y_test_oz, pred_oz)
        confusion_payload["open_world_zero_shot"] = build_confusion_payload(y_test_oz, pred_oz)

    return scenarios, feature_map, confusion_payload


def main():
    print("Loading and extracting features for Deep Learning...")
    X_base, y_base = extract_aggregated_features(BASELINE_DIR)
    X_obfs, y_obfs = extract_aggregated_features(OBFS4_DIR)

    X_other = pd.DataFrame()
    if os.path.isdir(OTHER_DIR):
        X_other, _ = extract_aggregated_features(OTHER_DIR, force_label="other")

    if X_base.empty or X_obfs.empty:
        print("Error: Missing baseline/obfs4 data.")
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)

    tracks = {}
    for track_name in ["shared", "optimized"]:
        print(f"\nRunning DL track: {track_name}")
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
        "dl_epochs": DL_EPOCHS,
        "tracks": tracks,
        # Backward-compatible fields used by figure scripts.
        "baseline": primary_shared["baseline"]["accuracy"],
        "obfs4": primary_shared["obfs4"]["accuracy"],
        "zero_shot": primary_shared["zero_shot"]["accuracy"],
        "top_features": tracks["shared"]["primary_seed_features"],
        "scenarios": primary_shared,
    }

    with open(DL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved DL metrics to: {DL_METRICS_PATH}")


if __name__ == "__main__":
    main()
