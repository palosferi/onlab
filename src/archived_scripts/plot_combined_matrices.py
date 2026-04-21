import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Reuse existing data loaders and network definitions
from train_model import load_data, BASELINE_DIR, OBFS4_DIR
from train_triplet import TripletNet, TripletDataset


def train_dl(X_train, y_train, epochs=15):
    """Train a Triplet CNN and build a k-NN embedding database."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripletNet().to(device)
    dataset = TripletDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for a, p, n, _ in loader:
            a, p, n = a.to(device), p.to(device), n.to(device)
            optimizer.zero_grad()
            loss = criterion(model(a), model(p), model(n))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(device)
        embeddings = model(X_tensor).cpu().numpy()

    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(embeddings, y_train)
    return model, knn, device


def predict_dl(model, knn, device, X_test):
    """Extract embeddings and predict labels with k-NN."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)
        emb = model(X_tensor).cpu().numpy()
    return knn.predict(emb)


def plot_matrix_pair(y_true, y_pred_rf, y_pred_dl, classes, title, filename, cmap):
    """Generate side-by-side (1x2) confusion-matrix heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Random Forest (left)
    cm_rf = confusion_matrix(y_true, y_pred_rf, labels=classes)
    sns.heatmap(
        cm_rf,
        annot=False,
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[0],
    )
    axes[0].set_title("Random Forest (classic ML)", fontsize=14, pad=10)
    axes[0].set_ylabel("True website", fontsize=12)
    axes[0].set_xlabel("Predicted website", fontsize=12)

    # 2. Triplet network (right)
    cm_dl = confusion_matrix(y_true, y_pred_dl, labels=classes)
    sns.heatmap(
        cm_dl,
        annot=False,
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[1],
    )
    axes[1].set_title("Triplet network (N-shot DL)", fontsize=14, pad=10)
    axes[1].set_ylabel("True website", fontsize=12)
    axes[1].set_xlabel("Predicted website", fontsize=12)

    # Main title and save
    plt.suptitle(title, fontsize=18, y=1.05, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Paired figure saved: {filename}")


def main():
    print("Loading data...")
    X_base, y_base = load_data(BASELINE_DIR)
    X_obfs, y_obfs = load_data(OBFS4_DIR)

    labels = np.unique(y_base)
    rf_params = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 5,
        "random_state": 42,
        "n_jobs": -1,
    }

    # ==========================================
    # 1. SCENARIO: BASELINE
    # ==========================================
    print("\n[1/3] Running baseline models...")
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_base, y_base, test_size=0.2, random_state=42
    )

    rf_base = RandomForestClassifier(**rf_params).fit(X_train_b, y_train_b)
    pred_rf_base = rf_base.predict(X_test_b)

    dl_base_model, dl_base_knn, device = train_dl(X_train_b, y_train_b, epochs=15)
    pred_dl_base = predict_dl(dl_base_model, dl_base_knn, device, X_test_b)

    plot_matrix_pair(
        y_test_b,
        pred_rf_base,
        pred_dl_base,
        labels,
        "1. Regular Tor Traffic (Baseline)",
        "cm_pair_baseline.png",
        "Blues",
    )

    # ==========================================
    # 2. SCENARIO: OBFS4
    # ==========================================
    print("\n[2/3] Running obfs4 models...")
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
        X_obfs, y_obfs, test_size=0.2, random_state=42
    )

    rf_obfs = RandomForestClassifier(**rf_params).fit(X_train_o, y_train_o)
    pred_rf_obfs = rf_obfs.predict(X_test_o)

    dl_obfs_model, dl_obfs_knn, _ = train_dl(X_train_o, y_train_o, epochs=15)
    pred_dl_obfs = predict_dl(dl_obfs_model, dl_obfs_knn, device, X_test_o)

    plot_matrix_pair(
        y_test_o,
        pred_rf_obfs,
        pred_dl_obfs,
        labels,
        "2. Obfuscated Traffic (Obfs4)",
        "cm_pair_obfs4.png",
        "YlOrBr",
    )

    # ==========================================
    # 3. SCENARIO: ZERO-SHOT
    # ==========================================
    print("\n[3/3] Running zero-shot evaluation...")
    # Baseline-trained models predict on obfs4 traffic.
    pred_rf_zero = rf_base.predict(X_obfs)
    pred_dl_zero = predict_dl(dl_base_model, dl_base_knn, device, X_obfs)

    plot_matrix_pair(
        y_obfs,
        pred_rf_zero,
        pred_dl_zero,
        labels,
        "3. Zero-Shot Test (baseline models on obfs4 traffic)",
        "cm_pair_zeroshot.png",
        "Reds",
    )

    print("\nDone. All 3 paired figures generated.")


if __name__ == "__main__":
    main()
