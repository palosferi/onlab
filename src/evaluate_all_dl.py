import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

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
DL_METRICS_PATH = os.path.join(FIGURES_DIR, "metrics_dl.json")
TOP_FEATURES_PATH = os.path.join(FIGURES_DIR, "selected_top10_features.json")
TOP_K_FEATURES = 10


# --- 1. PYTORCH DATASET ---
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

        return anchor, positive, negative, anchor_label


# --- 2. NEURAL NETWORK (Triplet MLP) ---
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


# --- 3. TRAINING & EVALUATION ---
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


def train_embedding_model(X_train, y_train, epochs=50):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripletMLP(input_size=X_train_scaled.shape[1]).to(device)

    dataset = TripletDataset(X_train_scaled, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for _ in range(epochs):
        for anchor, positive, negative, _ in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            loss = criterion(model(anchor), model(positive), model(negative))
            loss.backward()
            optimizer.step()

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
        n_estimators=300,
        out_path=TOP_FEATURES_PATH,
    )
    ranking_df.to_json(
        os.path.join(FIGURES_DIR, "feature_stability_dl.json"), orient="records", indent=2
    )
    return top_features


def main():
    print("Loading and extracting features for Deep Learning...")
    X_base, y_base = extract_aggregated_features(BASELINE_DIR)
    X_obfs, y_obfs = extract_aggregated_features(OBFS4_DIR)

    X_other = pd.DataFrame()
    if os.path.isdir(OTHER_DIR):
        X_other, _ = extract_aggregated_features(OTHER_DIR, force_label="other")

    if len(X_base) == 0 or len(X_obfs) == 0:
        print("Error: Missing baseline/obfs4 data.")
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)
    top_features = select_top_features(X_base, y_base, X_obfs, y_obfs, X_other)

    X_base = X_base[top_features].values
    X_obfs = X_obfs[top_features].values
    if not X_other.empty:
        X_other = X_other[top_features].values

    print("\n" + "=" * 70)
    print(" HYBRID DEEP LEARNING RESULTS (Triplet MLP, Top-10 Stable Features)")
    print("=" * 70)

    scenarios = {}

    # 1. BASELINE
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_base, y_base, test_size=0.2, random_state=42, stratify=y_base
    )
    scaler_b, model_b, device_b, knn_b = train_embedding_model(X_train_b, y_train_b)
    pred_b = predict_labels(scaler_b, model_b, device_b, knn_b, X_test_b)
    scenarios["baseline"] = evaluate_metrics(y_test_b, pred_b)
    print_metrics("1. Baseline", scenarios["baseline"])

    # 2. OBFS4
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
        X_obfs, y_obfs, test_size=0.2, random_state=42, stratify=y_obfs
    )
    scaler_o, model_o, device_o, knn_o = train_embedding_model(X_train_o, y_train_o)
    pred_o = predict_labels(scaler_o, model_o, device_o, knn_o, X_test_o)
    scenarios["obfs4"] = evaluate_metrics(y_test_o, pred_o)
    print_metrics("2. Obfs4", scenarios["obfs4"])

    # 3. ZERO-SHOT (Train: Baseline -> Test: Obfs4)
    pred_z = predict_labels(scaler_b, model_b, device_b, knn_b, X_obfs)
    scenarios["zero_shot"] = evaluate_metrics(y_obfs, pred_z)
    print_metrics("3. Zero-Shot", scenarios["zero_shot"])

    # 4-6. OPEN-WORLD (if Other exists)
    if isinstance(X_other, np.ndarray) and len(X_other) > 4:
        y_other = np.array(["other"] * len(X_other))

        X_open_b = np.vstack([X_base, X_other])
        y_open_b = np.concatenate([y_base, y_other])
        X_train_ob, X_test_ob, y_train_ob, y_test_ob = train_test_split(
            X_open_b, y_open_b, test_size=0.2, random_state=42, stratify=y_open_b
        )
        scaler_ob, model_ob, device_ob, knn_ob = train_embedding_model(X_train_ob, y_train_ob)
        pred_ob = predict_labels(scaler_ob, model_ob, device_ob, knn_ob, X_test_ob)
        scenarios["open_world_baseline"] = evaluate_metrics(y_test_ob, pred_ob)
        print_metrics("4. Open-World Baseline", scenarios["open_world_baseline"])

        X_open_o = np.vstack([X_obfs, X_other])
        y_open_o = np.concatenate([y_obfs, y_other])
        X_train_oo, X_test_oo, y_train_oo, y_test_oo = train_test_split(
            X_open_o, y_open_o, test_size=0.2, random_state=42, stratify=y_open_o
        )
        scaler_oo, model_oo, device_oo, knn_oo = train_embedding_model(X_train_oo, y_train_oo)
        pred_oo = predict_labels(scaler_oo, model_oo, device_oo, knn_oo, X_test_oo)
        scenarios["open_world_obfs4"] = evaluate_metrics(y_test_oo, pred_oo)
        print_metrics("5. Open-World Obfs4", scenarios["open_world_obfs4"])

        X_other_train, X_other_test = train_test_split(
            X_other, test_size=0.5, random_state=42
        )
        y_other_train = np.array(["other"] * len(X_other_train))
        y_other_test = np.array(["other"] * len(X_other_test))

        X_train_oz = np.vstack([X_base, X_other_train])
        y_train_oz = np.concatenate([y_base, y_other_train])
        X_test_oz = np.vstack([X_obfs, X_other_test])
        y_test_oz = np.concatenate([y_obfs, y_other_test])

        scaler_oz, model_oz, device_oz, knn_oz = train_embedding_model(X_train_oz, y_train_oz)
        pred_oz = predict_labels(scaler_oz, model_oz, device_oz, knn_oz, X_test_oz)
        scenarios["open_world_zero_shot"] = evaluate_metrics(y_test_oz, pred_oz)
        print_metrics("6. Open-World Zero-Shot", scenarios["open_world_zero_shot"])
    else:
        print("[Open-World] Skipped: no Other dataset found in extracted_features/other_features")

    print("=" * 70)

    with open(DL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": scenarios["baseline"]["accuracy"],
                "obfs4": scenarios["obfs4"]["accuracy"],
                "zero_shot": scenarios["zero_shot"]["accuracy"],
                "top_features": top_features,
                "scenarios": scenarios,
            },
            f,
            indent=2,
        )
    print(f"Saved DL metrics to: {DL_METRICS_PATH}")


if __name__ == "__main__":
    main()
