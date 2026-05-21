import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

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

# Set global font sizes (approx. 2x larger)
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 26
})


def load_metrics(metrics_path):
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Missing metrics file: {metrics_path}. Run the corresponding evaluation script first."
        )

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return [metrics["baseline"], metrics["obfs4"], metrics["zero_shot"]]


def load_scenario_metric(metrics_path, scenario_keys, metric_key):
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            f"Missing metrics file: {metrics_path}. Run the corresponding evaluation script first."
        )

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    if "tracks" in metrics:
        primary_track = metrics.get("primary_track", "shared")
        scenarios = (
            metrics.get("tracks", {})
            .get(primary_track, {})
            .get("primary_seed_scenarios", {})
        )
    else:
        scenarios = metrics.get("scenarios", {})

    values = []
    for key in scenario_keys:
        if key in scenarios and metric_key in scenarios[key]:
            values.append(scenarios[key][metric_key])
        elif metric_key == "accuracy" and key in metrics:
            values.append(metrics[key])
        else:
            values.append(np.nan)
    return values


def load_primary_confusions(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    if "tracks" in metrics:
        primary_track = metrics.get("primary_track", "shared")
        return (
            metrics.get("tracks", {})
            .get(primary_track, {})
            .get("primary_seed_confusion_matrices", {})
        )
    return metrics.get("confusion_matrices", {})


def load_other_recall(metrics_path, scenario_key):
    payload = load_metrics_payload(metrics_path)
    primary_track = payload.get("primary_track", "shared")
    scenarios = (
        payload.get("tracks", {})
        .get(primary_track, {})
        .get("primary_seed_scenarios", {})
    )
    per_class = scenarios.get(scenario_key, {}).get("per_class_recall", {})
    return per_class.get("other", np.nan)


def load_metrics_payload(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_track_scenario_metrics(metrics_path, track_name, scenario_keys, metric_key):
    payload = load_metrics_payload(metrics_path)
    tracks = payload.get("tracks", {})
    scenarios = tracks.get(track_name, {}).get("primary_seed_scenarios", {})
    values = []
    for key in scenario_keys:
        values.append(scenarios.get(key, {}).get(metric_key, np.nan))
    return values


def load_shared_primary_features_from_metrics(metrics_path):
    payload = load_metrics_payload(metrics_path)
    tracks = payload.get("tracks", {})
    features = tracks.get("shared", {}).get("primary_seed_features", [])
    if isinstance(features, list) and features:
        return features
    return None


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
    return top_features


def _plot_grouped_bars(ax, labels, left_vals, right_vals, left_name, right_name, title, y_label):
    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax.bar(x - width / 2, left_vals, width, label=left_name, color="#2980b9")
    rects2 = ax.bar(x + width / 2, right_vals, width, label=right_name, color="#27ae60")

    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            f"{height:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=16,
        )


def plot_final_bar_chart():
    scenario_keys = ["baseline", "obfs4", "zero_shot"]
    labels = ["Baseline", "Obfs4", "Zero-Shot"]

    rf_acc = load_scenario_metric(RF_METRICS_PATH, scenario_keys, "accuracy")
    dl_acc = load_scenario_metric(DL_METRICS_PATH, scenario_keys, "accuracy")
    rf_f1 = load_scenario_metric(RF_METRICS_PATH, scenario_keys, "macro_f1")
    dl_f1 = load_scenario_metric(DL_METRICS_PATH, scenario_keys, "macro_f1")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    _plot_grouped_bars(
        axes[0],
        labels,
        rf_acc,
        dl_acc,
        "Random Forest",
        "Triplet MLP",
        "",
        "Pontosság (%)",
    )
    _plot_grouped_bars(
        axes[1],
        labels,
        rf_f1,
        dl_f1,
        "Random Forest",
        "Triplet MLP",
        "",
        "",
    )
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "01_closed_world_summary.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Generated: 01_closed_world_summary.pdf")


def plot_shared_vs_optimized_delta_chart():
    scenario_pairs = [
        ("baseline", "Baseline"),
        ("obfs4", "Obfs4"),
        ("zero_shot", "Zero-Shot"),
        ("open_world_baseline", "OW-Baseline"),
        ("open_world_obfs4", "OW-Obfs4"),
        ("open_world_zero_shot", "OW-Zero-Shot"),
    ]
    scenario_keys = [k for k, _ in scenario_pairs]
    labels = [lbl for _, lbl in scenario_pairs]

    rf_acc_shared = np.array(load_track_scenario_metrics(RF_METRICS_PATH, "shared", scenario_keys, "accuracy"))
    rf_acc_opt = np.array(load_track_scenario_metrics(RF_METRICS_PATH, "optimized", scenario_keys, "accuracy"))
    rf_f1_shared = np.array(load_track_scenario_metrics(RF_METRICS_PATH, "shared", scenario_keys, "macro_f1"))
    rf_f1_opt = np.array(load_track_scenario_metrics(RF_METRICS_PATH, "optimized", scenario_keys, "macro_f1"))

    dl_acc_shared = np.array(load_track_scenario_metrics(DL_METRICS_PATH, "shared", scenario_keys, "accuracy"))
    dl_acc_opt = np.array(load_track_scenario_metrics(DL_METRICS_PATH, "optimized", scenario_keys, "accuracy"))
    dl_f1_shared = np.array(load_track_scenario_metrics(DL_METRICS_PATH, "shared", scenario_keys, "macro_f1"))
    dl_f1_opt = np.array(load_track_scenario_metrics(DL_METRICS_PATH, "optimized", scenario_keys, "macro_f1"))

    rf_acc_delta = rf_acc_opt - rf_acc_shared
    rf_f1_delta = rf_f1_opt - rf_f1_shared
    dl_acc_delta = dl_acc_opt - dl_acc_shared
    dl_f1_delta = dl_f1_opt - dl_f1_shared

    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=True)
    x = np.arange(len(labels))
    width = 0.38

    axes[0].bar(x - width / 2, rf_acc_delta, width, label="Pontosság különbség", color="#2e86de")
    axes[0].bar(x + width / 2, rf_f1_delta, width, label="Macro-F1 különbség", color="#17a589")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_ylabel("Különbség (százalékpont)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    axes[0].legend()

    axes[1].bar(x - width / 2, dl_acc_delta, width, label="Pontosság különbség", color="#5dade2")
    axes[1].bar(x + width / 2, dl_f1_delta, width, label="Macro-F1 különbség", color="#48c9b0")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Különbség (százalékpont)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15)
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "03_shared_vs_optimized_delta.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Generated: 03_shared_vs_optimized_delta.pdf")


def plot_open_world_focus_chart():
    scenario_pairs = [
        ("open_world_baseline", "OW-Baseline"),
        ("open_world_obfs4", "OW-Obfs4"),
        ("open_world_zero_shot", "OW-Zero-Shot"),
    ]
    scenario_keys = [k for k, _ in scenario_pairs]
    labels = [lbl for _, lbl in scenario_pairs]

    rf_acc = load_scenario_metric(RF_METRICS_PATH, scenario_keys, "accuracy")
    dl_acc = load_scenario_metric(DL_METRICS_PATH, scenario_keys, "accuracy")
    rf_f1 = load_scenario_metric(RF_METRICS_PATH, scenario_keys, "macro_f1")
    dl_f1 = load_scenario_metric(DL_METRICS_PATH, scenario_keys, "macro_f1")

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    _plot_grouped_bars(
        axes[0],
        labels,
        rf_acc,
        dl_acc,
        "Random Forest",
        "Triplet MLP",
        "",
        "Pontosság (%)",
    )
    _plot_grouped_bars(
        axes[1],
        labels,
        rf_f1,
        dl_f1,
        "Random Forest",
        "Triplet MLP",
        "",
        "",
    )
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "04_open_world_focus.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Generated: 04_open_world_focus.pdf")


def get_feature_importance(X, y):
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns)


def plot_feature_importance(
    X_base,
    y_base,
    X_obfs,
    y_obfs,
    top_features,
    X_base_full=None,
    X_obfs_full=None,
):
    imp_base = get_feature_importance(X_base, y_base)
    imp_obfs = get_feature_importance(X_obfs, y_obfs)

    plot_base = pd.DataFrame(
        {
            "feature": top_features,
            "importance": imp_base[top_features].values,
        }
    ).sort_values("importance", ascending=True)
    plot_obfs = pd.DataFrame(
        {
            "feature": top_features,
            "importance": imp_obfs[top_features].values,
        }
    ).sort_values("importance", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharex=False, sharey=False)

    sns.barplot(
        data=plot_base,
        x="importance",
        y="feature",
        color="#2980b9",
        ax=axes[0],
    )
    axes[0].set_xlabel("Relatív fontosság")
    axes[0].set_ylabel("Kinyert jellemzők")

    sns.barplot(
        data=plot_obfs,
        x="importance",
        y="feature",
        color="#e67e22",
        ax=axes[1],
    )
    axes[1].set_xlabel("Relatív fontosság")
    axes[1].set_ylabel("Kinyert jellemzők")

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "05_feature_importance_shared.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    imp_base_full = get_feature_importance(
        X_base_full if X_base_full is not None else X_base,
        y_base,
    )
    imp_obfs_full = get_feature_importance(
        X_obfs_full if X_obfs_full is not None else X_obfs,
        y_obfs,
    )
    top_base = imp_base_full.sort_values(ascending=False).head(10)
    top_obfs = imp_obfs_full.sort_values(ascending=False).head(10)

    top_base_df = (
        pd.DataFrame(
            {
                "feature": top_base.index.tolist(),
                "importance": top_base.values,
            }
        )
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )
    top_obfs_df = (
        pd.DataFrame(
            {
                "feature": top_obfs.index.tolist(),
                "importance": top_obfs.values,
            }
        )
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharex=False, sharey=False)
    sns.barplot(
        data=top_base_df,
        x="importance",
        y="feature",
        color="#2471a3",
        ax=axes[0],
    )
    axes[0].set_xlabel("Relatív fontosság")
    axes[0].set_ylabel("Kinyert jellemzők")

    sns.barplot(
        data=top_obfs_df,
        x="importance",
        y="feature",
        color="#ca6f1e",
        ax=axes[1],
    )
    axes[1].set_xlabel("Relatív fontosság")
    axes[1].set_ylabel("Kinyert jellemzők")

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "06_feature_importance_separate.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Generated: 05_feature_importance_shared.pdf (+ 06_feature_importance_separate.pdf)")


def plot_confusion_matrices_recall():
    confusions = load_primary_confusions(RF_METRICS_PATH)
    if not confusions:
        print("Skipping recall confusion matrices: no persisted confusion payload found.")
        return

    if all(k in confusions for k in ["open_world_baseline", "open_world_obfs4", "open_world_zero_shot"]):
        scenario_keys = ["open_world_baseline", "open_world_obfs4", "open_world_zero_shot"]
        cmaps = ["Blues", "Oranges", "Reds"]
    else:
        scenario_keys = ["baseline", "obfs4", "zero_shot"]
        cmaps = ["Blues", "Oranges", "Reds"]

    fig, axes = plt.subplots(1, 3, figsize=(32, 10))

    for ax, scenario_key, cmap in zip(axes, scenario_keys, cmaps):
        payload = confusions.get(scenario_key)
        if not payload:
            ax.axis("off")
            continue

        labels = payload.get("labels", [])
        matrix = np.array(payload.get("matrix", []), dtype=float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums
        sns.heatmap(
            matrix,
            cmap=cmap,
            cbar=False,
            vmin=0.0,
            vmax=1.0,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        
    for ax in axes:
        ax.set_ylabel("Valós webhely")
        ax.tick_params(axis="x", rotation=90)
        
    fig.supxlabel("Prediktált webhely", fontsize=28)

    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "08_confusion_matrices_recall.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Generated: 08_confusion_matrices_recall.pdf")


def plot_other_recall_open_world():
    scenario_pairs = [
        ("open_world_baseline", "OW-Baseline"),
        ("open_world_obfs4", "OW-Obfs4"),
        ("open_world_zero_shot", "OW-Zero-Shot"),
    ]
    scenario_keys = [k for k, _ in scenario_pairs]
    labels = [lbl for _, lbl in scenario_pairs]

    rf_vals = [load_other_recall(RF_METRICS_PATH, key) for key in scenario_keys]
    dl_vals = [load_other_recall(DL_METRICS_PATH, key) for key in scenario_keys]

    if all(np.isnan(rf_vals)) and all(np.isnan(dl_vals)):
        print("Skipping other recall chart: no 'other' class found in metrics.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    _plot_grouped_bars(
        ax,
        labels,
        rf_vals,
        dl_vals,
        "Random Forest",
        "Triplet MLP",
        "",
        "Visszahívás (%)",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "09_other_recall_open_world.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Generated: 09_other_recall_open_world.pdf")


def main():
    print("Generating all final figures for the report...")
    plot_final_bar_chart()
    plot_shared_vs_optimized_delta_chart()
    plot_open_world_focus_chart()

    print("Extracting data for ML charts...")
    X_base, y_base = extract_aggregated_features(BASELINE_DIR)
    X_obfs, y_obfs = extract_aggregated_features(OBFS4_DIR)

    X_other = pd.DataFrame()
    y_other = np.array([])
    if os.path.isdir(OTHER_DIR):
        X_other, y_other = extract_aggregated_features(OTHER_DIR, force_label="other")

    if not X_base.empty and not X_obfs.empty:
        X_base_full = X_base.copy()
        X_obfs_full = X_obfs.copy()
        top_features = load_shared_primary_features_from_metrics(RF_METRICS_PATH)
        if top_features:
            print("Using RF shared primary features from metrics payload.")
        else:
            top_features = select_top_features(X_base, y_base, X_obfs, y_obfs, X_other)
        X_base_sel = X_base[top_features]
        X_obfs_sel = X_obfs[top_features]

        plot_feature_importance(
            X_base_sel,
            y_base,
            X_obfs_sel,
            y_obfs,
            top_features,
            X_base_full=X_base_full,
            X_obfs_full=X_obfs_full,
        )
        _ = y_other
    else:
        print("Skipping feature importance: missing extracted features.")

    plot_confusion_matrices_recall()
    plot_other_recall_open_world()

    print("\nSuccess! All charts are in 'figures/' folder.")


if __name__ == "__main__":
    main()