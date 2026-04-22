import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

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
RF_METRICS_PATH = os.path.join(FIGURES_DIR, "metrics_rf.json")
DL_METRICS_PATH = os.path.join(FIGURES_DIR, "metrics_dl.json")
TOP_FEATURES_PATH = os.path.join(FIGURES_DIR, "selected_top10_features.json")
RF_N_ESTIMATORS = 300
TOP_K_FEATURES = 10

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_metrics(metrics_path):
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Missing metrics file: {metrics_path}. Run the corresponding evaluation script first."
        )

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return [metrics["baseline"], metrics["obfs4"], metrics["zero_shot"]]


def select_top_features(X_base, y_base, X_obfs, y_obfs, X_other):
    locked = load_locked_top_features(TOP_FEATURES_PATH)
    if locked:
        print("Using locked top-10 features from previous run.")
        return locked

    frames = [X_base, X_obfs]
    labels = [y_base, y_obfs]

    if not X_other.empty:
        frames.append(X_other)
        labels.append(np.array(["other"] * len(X_other)))

    X_all = pd.concat(frames, ignore_index=True)
    y_all = np.concatenate(labels)

    top_features, ranking_df = select_stable_top_features(
        X_all,
        y_all,
        top_k=TOP_K_FEATURES,
        n_runs=8,
        n_estimators=RF_N_ESTIMATORS,
        out_path=TOP_FEATURES_PATH,
    )
    ranking_df.to_json(
        os.path.join(FIGURES_DIR, "feature_stability_for_figures.json"),
        orient="records",
        indent=2,
    )
    return top_features


def plot_final_bar_chart():
    labels = ["Baseline", "Obfs4", "Zero-Shot"]
    rf_acc = load_metrics(RF_METRICS_PATH)
    dl_acc = load_metrics(DL_METRICS_PATH)

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


def get_feature_importance(X, y):
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns)


def plot_feature_importance(X_base, y_base, X_obfs, y_obfs, top_features):
    imp_base = get_feature_importance(X_base, y_base)
    imp_obfs = get_feature_importance(X_obfs, y_obfs)

    plot_df = pd.DataFrame(
        {
            "feature": top_features,
            "baseline": imp_base[top_features].values,
            "obfs4": imp_obfs[top_features].values,
        }
    )
    plot_df = plot_df.sort_values("baseline", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    sns.barplot(
        data=plot_df,
        x="baseline",
        y="feature",
        color="#2980b9",
        ax=axes[0],
    )
    axes[0].set_title("Baseline", fontsize=13)
    axes[0].set_xlabel("Relative Importance")
    axes[0].set_ylabel("Engineered Features")

    sns.barplot(
        data=plot_df,
        x="obfs4",
        y="feature",
        color="#e67e22",
        ax=axes[1],
    )
    axes[1].set_title("Obfs4", fontsize=13)
    axes[1].set_xlabel("Relative Importance")
    axes[1].set_ylabel("")

    fig.suptitle("Random Forest Feature Importance (Stable Top 10)", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "02_feature_importance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    top_table = plot_df.sort_values("baseline", ascending=False)
    top_table.insert(0, "rank", np.arange(1, len(top_table) + 1))
    top_table.to_json(
        os.path.join(FIGURES_DIR, "02_top10_features_rf.json"),
        orient="records",
        indent=2,
    )
    print("Chart 2/3 Generated: 02_feature_importance.png (+ top10 json)")


def plot_confusion_matrices(X_base, y_base, X_obfs, y_obfs):
    labels = np.sort(np.unique(np.concatenate([y_base, y_obfs])))
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    rf_base = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, random_state=42, n_jobs=-1
    )
    rf_obfs = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, random_state=42, n_jobs=-1
    )

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_base, y_base, test_size=0.2, random_state=42, stratify=y_base
    )
    X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
        X_obfs, y_obfs, test_size=0.2, random_state=42, stratify=y_obfs
    )

    rf_base.fit(X_train_b, y_train_b)
    rf_obfs.fit(X_train_o, y_train_o)

    sns.heatmap(
        confusion_matrix(y_test_b, rf_base.predict(X_test_b), labels=labels),
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
    )
    axes[0].set_title("1. Baseline", fontsize=14)

    sns.heatmap(
        confusion_matrix(y_test_o, rf_obfs.predict(X_test_o), labels=labels),
        cmap="Oranges",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
    )
    axes[1].set_title("2. Obfs4", fontsize=14)

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

    X_other = pd.DataFrame()
    if os.path.isdir(OTHER_DIR):
        X_other, _ = extract_aggregated_features(OTHER_DIR, force_label="other")

    if not X_base.empty and not X_obfs.empty:
        top_features = select_top_features(X_base, y_base, X_obfs, y_obfs, X_other)
        X_base_sel = X_base[top_features]
        X_obfs_sel = X_obfs[top_features]

        plot_feature_importance(X_base_sel, y_base, X_obfs_sel, y_obfs, top_features)
        plot_confusion_matrices(X_base_sel, y_base, X_obfs_sel, y_obfs)

    print("\nSuccess! All charts are in 'figures/' folder.")


if __name__ == "__main__":
    main()
