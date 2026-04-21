import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Reuse existing data loaders and model classes
from train_model import load_data, BASELINE_DIR, OBFS4_DIR
from train_triplet import TripletNet, TripletDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


def run_zeroshot_triplet():
    print("--- DL ZERO-SHOT EVALUATION STARTED ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load baseline and train a quick demo model
    print("[1/3] Loading baseline data and training model (10 epochs)...")
    X_base, y_base = load_data(BASELINE_DIR)

    # Train CNN on baseline traffic
    train_dataset = TripletDataset(X_base, y_base)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = TripletNet().to(device)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        for anchor, positive, negative, _ in train_loader:
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )
            optimizer.zero_grad()
            loss = criterion(model(anchor), model(positive), model(negative))
            loss.backward()
            optimizer.step()

    # 2. Train k-NN on baseline embeddings
    print("[2/3] Building k-NN database from baseline samples...")
    model.eval()
    with torch.no_grad():
        X_base_tensor = torch.FloatTensor(X_base).unsqueeze(1).to(device)
        base_embeddings = model(X_base_tensor).cpu().numpy()

    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(base_embeddings, y_base)

    # 3. Test on obfs4 traffic
    print("[3/3] Loading obfs4 data and testing...")
    X_obfs, y_obfs = load_data(OBFS4_DIR)

    with torch.no_grad():
        X_obfs_tensor = torch.FloatTensor(X_obfs).unsqueeze(1).to(device)
        obfs_embeddings = model(X_obfs_tensor).cpu().numpy()

    y_pred = knn.predict(obfs_embeddings)
    accuracy = accuracy_score(y_obfs, y_pred)

    print("-" * 50)
    print(f"DL zero-shot accuracy: {accuracy * 100:.2f}%")
    print("-" * 50)

    # Generate confusion matrix for DL zero-shot run
    print("Generating confusion matrix...")
    labels = np.unique(y_obfs)
    cm = confusion_matrix(y_obfs, y_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.title("DL Zero-Shot Matrix (TripletNet)", fontsize=16)
    plt.ylabel("True website", fontsize=12)
    plt.xlabel("Predicted website", fontsize=12)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("cm_zeroshot_dl.png", dpi=300)
    print("Chart saved: cm_zeroshot_dl.png")


if __name__ == "__main__":
    run_zeroshot_triplet()
