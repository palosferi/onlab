import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tor_dataset", "extracted_features")
)
BASELINE_DIR = os.path.join(BASE_DIR, "baseline_features")
OBFS4_DIR = os.path.join(BASE_DIR, "obfs4_features")
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
DL_METRICS_PATH = os.path.join(FIGURES_DIR, "metrics_dl.json")


# --- 1. FEATURE EXTRACTION (same as RF) ---
def extract_aggregated_features(directory):
    X, y = [], []
    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    for file_path in csv_files:
        site_name = os.path.basename(file_path).split("_")[0]
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
                len(dirs),  # total_packets
                len(incoming),  # incoming_packets
                len(outgoing),  # outgoing_packets
                len(incoming) / (len(outgoing) + 1e-5),  # in_out_packet_ratio
                np.sum(np.abs(dirs)),  # total_bytes
                np.abs(np.sum(incoming)) if len(incoming) > 0 else 0,  # incoming_bytes
                np.sum(outgoing) if len(outgoing) > 0 else 0,  # outgoing_bytes
                np.abs(np.sum(incoming))
                / (np.sum(outgoing) + 1e-5),  # in_out_byte_ratio
                times[-1] if len(times) > 0 else 0,  # duration
                np.mean(iats) if len(iats) > 0 else 0,  # mean_inter_arrival
                np.std(iats) if len(iats) > 0 else 0,  # std_inter_arrival
                np.max(np.abs(dirs)),  # max_packet_size
                np.mean(np.abs(dirs)),  # mean_packet_size
            ]
            X.append(features)
            y.append(site_name)
        except Exception:
            pass

    return np.array(X), np.array(y)


# --- 2. PYTORCH DATASET ---
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


# --- 3. NEURAL NETWORK (Multi-Layer Perceptron) ---
class TripletMLP(nn.Module):
    def __init__(self, input_size=13):
        super(TripletMLP, self).__init__()
        # Dense network to process the 13 features
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
        )

    def forward(self, x):
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)


# --- 4. TRAINING & EVALUATION ---
def train_and_extract_embeddings(X_train, y_train, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripletMLP().to(device)

    dataset = TripletDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for anchor, positive, negative, _ in loader:
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )
            optimizer.zero_grad()
            loss = criterion(model(anchor), model(positive), model(negative))
            loss.backward()
            optimizer.step()

    return model, device


def evaluate_scenario(name, model, device, knn, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        embeddings = model(X_tensor).cpu().numpy()

    y_pred = knn.predict(embeddings)
    acc = accuracy_score(y_test, y_pred)
    print(f"[{name}] Accuracy: {acc * 100:.2f}%")
    return acc


def main():
    print("Loading and extracting features for Deep Learning...")
    X_base, y_base = extract_aggregated_features(BASELINE_DIR)
    X_obfs, y_obfs = extract_aggregated_features(OBFS4_DIR)

    if len(X_base) == 0 or len(X_obfs) == 0:
        print("Error: Missing data.")
        return

    # Feature scaling
    scaler = StandardScaler()
    X_base_scaled = scaler.fit_transform(X_base)
    X_obfs_scaled = scaler.transform(X_obfs)

    print("\n" + "=" * 50)
    print(" NEW HYBRID DEEP LEARNING RESULTS (Triplet MLP)")
    print("=" * 50)

    # 1. BASELINE SCENARIO
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_base_scaled,
        y_base,
        test_size=0.2,
        random_state=42,
        stratify=y_base,
    )
    model_base, device = train_and_extract_embeddings(X_train_b, y_train_b, epochs=50)

    # Train k-NN on Baseline
    model_base.eval()
    with torch.no_grad():
        train_emb_b = model_base(torch.FloatTensor(X_train_b).to(device)).cpu().numpy()
    knn_base = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn_base.fit(train_emb_b, y_train_b)

    acc_base = evaluate_scenario(
        "1. Baseline", model_base, device, knn_base, X_test_b, y_test_b
    )

    # 2. OBFS4 SCENARIO
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
        X_obfs_scaled,
        y_obfs,
        test_size=0.2,
        random_state=42,
        stratify=y_obfs,
    )
    model_obfs, _ = train_and_extract_embeddings(X_train_o, y_train_o, epochs=50)

    model_obfs.eval()
    with torch.no_grad():
        train_emb_o = model_obfs(torch.FloatTensor(X_train_o).to(device)).cpu().numpy()
    knn_obfs = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn_obfs.fit(train_emb_o, y_train_o)

    acc_obfs = evaluate_scenario(
        "2. Obfs4", model_obfs, device, knn_obfs, X_test_o, y_test_o
    )

    # 3. ZERO-SHOT SCENARIO (Train: Baseline -> Test: Obfs4)
    acc_zero = evaluate_scenario(
        "3. Zero-Shot", model_base, device, knn_base, X_obfs_scaled, y_obfs
    )
    print("=" * 50)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    with open(DL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": round(acc_base * 100, 2),
                "obfs4": round(acc_obfs * 100, 2),
                "zero_shot": round(acc_zero * 100, 2),
            },
            f,
            indent=2,
        )
    print(f"Saved DL metrics to: {DL_METRICS_PATH}")


if __name__ == "__main__":
    main()
