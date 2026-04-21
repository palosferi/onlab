import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# --- CONFIGURATION ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "tor_dataset", "extracted_features")
)
BASELINE_DIR = os.path.join(BASE_DIR, "baseline_features")
OBFS4_DIR = os.path.join(BASE_DIR, "obfs4_features")
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))

os.makedirs(FIGURES_DIR, exist_ok=True)


# --- 1. FEATURE EXTRACTION ---
def extract_aggregated_features(directory):
    X, y = [], []
    feature_names = [
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
        "max_packet_size",
        "mean_packet_size",
    ]

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
                np.max(np.abs(dirs)),
                np.mean(np.abs(dirs)),
            ]
            X.append(features)
            y.append(site_name)
        except Exception:
            pass
    return pd.DataFrame(X, columns=feature_names), np.array(y)


# --- CHART 1: FINAL COMPARISON BAR CHART ---
def plot_final_bar_chart():
    labels = ["Baseline", "Obfs4", "Zero-Shot"]
    rf_acc = [76.44, 73.81, 16.95]
    dl_acc = [61.78, 51.43, 20.48]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(
        x - width / 2, rf_acc, width, label="Random Forest", color="#2980b9"
    )
    rects2 = ax.bar(
        x + width / 2, dl_acc, width, label="Triplet MLP (Hybrid DL)", color="#27ae60"
    )

    ax.set_ylabel("Classification Accuracy (%)", fontsize=12)
    ax.set_title(
        "Website Fingerprinting Models vs. Obfs4 Obfuscation", fontsize=14, pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(
            f"{height}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "01_final_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Chart 1/3 Generated: 01_final_comparison.png")


# --- CHART 2: FEATURE IMPORTANCE ---
def plot_feature_importance(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_

    # Sort features by importance
    indices = np.argsort(importances)
    sorted_features = [X.columns[i] for i in indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=sorted_features, color="#3498db")
    plt.title("Random Forest: Feature Importance", fontsize=14)
    plt.xlabel("Relative Importance (%)")
    plt.ylabel("Engineered Features")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "02_feature_importance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Chart 2/3 Generated: 02_feature_importance.png")


# --- CHART 3: CONFUSION MATRICES ---
def plot_confusion_matrices(X_base, y_base, X_obfs, y_obfs):
    labels = np.unique(y_base)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_obfs = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_base, y_base, test_size=0.2, random_state=42
    )
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
        X_obfs, y_obfs, test_size=0.2, random_state=42
    )

    rf_base.fit(X_train_b, y_train_b)
    rf_obfs.fit(X_train_o, y_train_o)

    # 1. Baseline
    sns.heatmap(
        confusion_matrix(y_test_b, rf_base.predict(X_test_b), labels=labels),
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
    )
    axes[0].set_title("1. Baseline", fontsize=14)

    # 2. Obfs4
    sns.heatmap(
        confusion_matrix(y_test_o, rf_obfs.predict(X_test_o), labels=labels),
        cmap="Oranges",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
    )
    axes[1].set_title("2. Obfs4", fontsize=14)

    # 3. Zero-Shot
    sns.heatmap(
        confusion_matrix(y_obfs, rf_base.predict(X_obfs), labels=labels),
        cmap="Reds",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[2],
    )
    axes[2].set_title("3. Zero-Shot", fontsize=14)

    for ax in axes:
        ax.set_ylabel("True Website")
        ax.set_xlabel("Predicted Website")
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "03_confusion_matrices.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Chart 3/3 Generated: 03_confusion_matrices.png")


def main():
    print("Generating all final figures for the report...")
    plot_final_bar_chart()

    print("Extracting data for ML charts...")
    X_base, y_base = extract_aggregated_features(BASELINE_DIR)
    X_obfs, y_obfs = extract_aggregated_features(OBFS4_DIR)

    if not X_base.empty:
        plot_feature_importance(X_base, y_base)
        plot_confusion_matrices(X_base, y_base, X_obfs, y_obfs)

    print("\nSuccess! All charts are in 'figures/' folder.")


if __name__ == "__main__":
    main()
