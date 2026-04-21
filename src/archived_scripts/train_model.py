import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
BASE_DIR = os.path.expanduser("~/Desktop/tor_dataset/extracted_features")
BASELINE_DIR = os.path.join(BASE_DIR, "baseline_features")
OBFS4_DIR = os.path.join(BASE_DIR, "obfs4_features")

# Number of packets used per trace (standard WF setup)
SEQUENCE_LENGTH = 500


def load_data(directory):
    print(f"Loading data from: {directory}...")
    X = []  # Input features
    y = []  # Labels (website names)

    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {directory}")
        return np.array([]), np.array([])

    for file_path in csv_files:
        # Parse website label from filename, e.g. "telex_20260404_1200.csv" -> "telex"
        filename = os.path.basename(file_path)
        site_name = filename.split("_")[0]

        try:
            df = pd.read_csv(file_path)

            # Extract packet direction/size sequence
            directions = df["direction_size"].values

            # Truncate or zero-pad to fixed length
            if len(directions) >= SEQUENCE_LENGTH:
                features = directions[:SEQUENCE_LENGTH]
            else:
                features = np.pad(
                    directions, (0, SEQUENCE_LENGTH - len(directions)), "constant"
                )

            X.append(features)
            y.append(site_name)

        except Exception:
            pass

    return np.array(X), np.array(y)


def train_and_evaluate(X, y, dataset_name):
    print(f"\n--- {dataset_name.upper()} MODEL ---")

    if len(X) == 0:
        print("Not enough data for training.")
        return

    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy ({dataset_name}): {accuracy * 100:.2f}%")
    return clf


def main():
    print("Tor Website Fingerprinting - ML evaluation started\n")

    # 1. Baseline training and evaluation
    X_base, y_base = load_data(BASELINE_DIR)
    train_and_evaluate(X_base, y_base, "Baseline (regular Tor)")

    print("-" * 50)

    # 2. Obfs4 training and evaluation
    X_obfs, y_obfs = load_data(OBFS4_DIR)
    train_and_evaluate(X_obfs, y_obfs, "Obfs4 (obfuscated Tor)")


if __name__ == "__main__":
    main()
