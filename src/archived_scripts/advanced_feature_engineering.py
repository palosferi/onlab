import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tor_dataset", "extracted_features")
)
BASELINE_DIR = os.path.join(BASE_DIR, "baseline_features")


def extract_aggregated_features(directory):
    print(f"Extracting aggregated features from: {directory}...")
    X = []
    y = []

    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return pd.DataFrame(), []

    for file_path in csv_files:
        site_name = os.path.basename(file_path).split("_")[0]

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue

            # Split packet directions and timing values
            dirs = df["direction_size"].values
            times = df["time_offset"].values
            iats = df["inter_arrival_time"].values

            incoming = dirs[dirs < 0]
            outgoing = dirs[dirs > 0]

            # --- FEATURE ENGINEERING (Aggregated Statistics) ---
            features = {
                "total_packets": len(dirs),
                "incoming_packets": len(incoming),
                "outgoing_packets": len(outgoing),
                "in_out_packet_ratio": len(incoming) / (len(outgoing) + 1e-5),
                "total_bytes": np.sum(np.abs(dirs)),
                "incoming_bytes": np.abs(np.sum(incoming)) if len(incoming) > 0 else 0,
                "outgoing_bytes": np.sum(outgoing) if len(outgoing) > 0 else 0,
                "in_out_byte_ratio": np.abs(np.sum(incoming))
                / (np.sum(outgoing) + 1e-5),
                "duration": times[-1] if len(times) > 0 else 0,
                "mean_inter_arrival": np.mean(iats) if len(iats) > 0 else 0,
                "std_inter_arrival": np.std(iats) if len(iats) > 0 else 0,
                "max_packet_size": np.max(np.abs(dirs)),
                "mean_packet_size": np.mean(np.abs(dirs)),
            }

            X.append(features)
            y.append(site_name)

        except Exception:
            pass

    return pd.DataFrame(X), np.array(y)


def main():
    print("Feature engineering and model training started.\n")

    # 1. Feature extraction
    X_df, y = extract_aggregated_features(BASELINE_DIR)

    if X_df.empty:
        return

    # 2. Model training
    print("\nTraining Random Forest on aggregated features...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("-" * 50)
    print(f"New RF accuracy (aggregated features): {acc * 100:.2f}%")
    print("-" * 50)


if __name__ == "__main__":
    main()
