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
OBFS4_DIR = os.path.join(BASE_DIR, "obfs4_features")


def extract_aggregated_features(directory):
    """Extracts macro-features from the CSV files."""
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

            # --- FEATURE ENGINEERING ---
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
    print("Extracting features from Baseline traffic...")
    X_base, y_base = extract_aggregated_features(BASELINE_DIR)

    print("Extracting features from Obfs4 traffic...")
    X_obfs, y_obfs = extract_aggregated_features(OBFS4_DIR)

    if X_base.empty or X_obfs.empty:
        print("Error: Couldnt find dataset files.")
        return

    print("\n" + "=" * 50)
    print(" NEW RANDOM FOREST RESULTS (With Aggregated Features)")
    print("=" * 50)

    # 1. SCENARIO: BASELINE
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_base, y_base, test_size=0.2, random_state=42
    )
    rf_base = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_base.fit(X_train_b, y_train_b)
    acc_base = accuracy_score(y_test_b, rf_base.predict(X_test_b))
    print(f"[1] Baseline Accuracy:   {acc_base * 100:.2f}%")

    # 2. SCENARIO: OBFS4
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
        X_obfs, y_obfs, test_size=0.2, random_state=42
    )
    rf_obfs = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf_obfs.fit(X_train_o, y_train_o)
    acc_obfs = accuracy_score(y_test_o, rf_obfs.predict(X_test_o))
    print(f"[2] Obfs4 Accuracy:      {acc_obfs * 100:.2f}%")

    # 3. SCENARIO: ZERO-SHOT (Train: Baseline -> Test: Obfs4)
    acc_zero = accuracy_score(y_obfs, rf_base.predict(X_obfs))
    print(f"[3] Zero-Shot Accuracy:  {acc_zero * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
