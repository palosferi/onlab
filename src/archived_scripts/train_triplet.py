import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from train_model import load_data, BASELINE_DIR, OBFS4_DIR, SEQUENCE_LENGTH


# --- 1. PYTORCH DATASET FOR TRIPLET SAMPLING ---
class TripletDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(
            1
        )  # [batch, channel, length] -> [N, 1, 500]
        self.y = np.array(y)
        self.labels_set = set(self.y)
        self.label_to_indices = {
            label: np.where(self.y == label)[0] for label in self.labels_set
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Anchor
        anchor = self.X[index]
        anchor_label = self.y[index]

        # Positive sample: same class, different example
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[anchor_label])
        positive = self.X[positive_index]

        # Negative sample: different class
        negative_label = np.random.choice(list(self.labels_set - {anchor_label}))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        negative = self.X[negative_index]

        return anchor, positive, negative, anchor_label


# --- 2. 1D CNN FEATURE EXTRACTOR ---
class TripletNet(nn.Module):
    def __init__(self):
        super(TripletNet, self).__init__()
        # Simple CNN for packet direction/size sequence modeling
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (SEQUENCE_LENGTH // 4), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        # Normalize embeddings for stable distance-based matching
        return nn.functional.normalize(x, p=2, dim=1)


# --- 3. TRAINING AND EVALUATION ---
def train_and_eval_triplet(X, y, dataset_name, epochs=10):
    print(f"\n--- {dataset_name.upper()} TRIPLET MODEL ---")

    if len(X) == 0:
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = TripletDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripletNet().to(device)

    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training started on {device} ({epochs} epochs)...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for anchor, positive, negative, _ in train_loader:
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )

            optimizer.zero_grad()
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"    Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}"
        )

    # --- EVALUATION (N-SHOT k-NN) ---
    print("Extracting embeddings for evaluation...")
    model.eval()
    with torch.no_grad():
        # Build embedding database from training split
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(device)
        train_embeddings = model(X_train_tensor).cpu().numpy()

        # Extract embeddings for test split
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)
        test_embeddings = model(X_test_tensor).cpu().numpy()

    # N-shot classification with k-NN on learned embeddings
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(train_embeddings, y_train)

    y_pred = knn.predict(test_embeddings)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Triplet N-shot accuracy ({dataset_name}): {accuracy * 100:.2f}%\n")
    return model, knn


def main():
    X_base, y_base = load_data(BASELINE_DIR)
    train_and_eval_triplet(X_base, y_base, "Baseline (regular Tor)", epochs=15)

    print("-" * 50)

    X_obfs, y_obfs = load_data(OBFS4_DIR)
    train_and_eval_triplet(X_obfs, y_obfs, "Obfs4 (obfuscated Tor)", epochs=15)


if __name__ == "__main__":
    main()
