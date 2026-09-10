"""Figures for the concept drift study, built from drift_metrics_<arm>.json.

    python src/drift_eval.py --arm baseline
    python src/drift_figures.py --arm baseline
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
GROUP_COLORS = {"static": "#4C72B0", "news": "#C44E52", "commerce_media": "#55A868"}


def plot_decay(data, arm):
    rows = data.get("no_retrain") or []
    if not rows:
        return None
    ceiling = {r["round"]: r["accuracy"] for r in data.get("within_round_ceiling", [])}
    weeks = [r["weeks_since_train"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    sd = [r.get("accuracy_std", 0) for r in rows]
    f1 = [r["macro_f1"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(weeks, acc, yerr=sd, marker="o", capsize=3,
                color="#C44E52", label="accuracy, trained on t0 only")
    ax.plot(weeks, f1, marker="s", linestyle="--", color="#8172B2",
            label="macro-F1, trained on t0 only")

    ceil_w = [r["weeks_since_train"] for r in rows if r["round"] in ceiling]
    ceil_a = [ceiling[r["round"]] for r in rows if r["round"] in ceiling]
    if ceil_a:
        ax.plot(ceil_w, ceil_a, marker="^", linestyle=":", color="#4C72B0",
                label="same-round ceiling, no drift")

    # 1/36 is what a classifier that has stopped working entirely would score.
    ax.axhline(100 / 36, color="#999999", linestyle="-.", linewidth=1,
               label="random guess, 36 classes")
    ax.set_xlabel("weeks between training data and test data")
    ax.set_ylabel("percent")
    ax.set_title(f"Website fingerprinting accuracy decay over time, {arm} arm")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 100)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, f"drift_01_decay_{arm}.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_groups(data, arm):
    rows = data.get("no_retrain") or []
    if not rows or not rows[0].get("per_group_recall"):
        return None
    weeks = [r["weeks_since_train"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for group, color in GROUP_COLORS.items():
        vals = [r["per_group_recall"].get(group) for r in rows]
        if any(v is not None for v in vals):
            ax.plot(weeks, vals, marker="o", color=color, label=group)
    ax.set_xlabel("weeks between training data and test data")
    ax.set_ylabel("mean per-class recall, percent")
    ax.set_title(f"Does drift hit news sites harder than static sites? {arm} arm")
    ax.grid(alpha=0.3)
    ax.legend(title="site group")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, f"drift_02_groups_{arm}.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_cost(data, arm):
    """Accuracy against labelling cost, which is the actual decision an
    adversary faces: every retraining round is traces someone has to collect."""
    series = {}
    if data.get("no_retrain"):
        last = data["no_retrain"][-1]
        series["no retraining"] = (last["train_samples_used"], last["accuracy"])
    for policy, rows in (data.get("periodic_retrain") or {}).items():
        if rows:
            series[policy.replace("_", " ")] = (rows[-1]["train_samples_used"], rows[-1]["accuracy"])
    for policy, rows in (data.get("sliding_window") or {}).items():
        if rows:
            series[policy.replace("_", " ")] = (rows[-1]["train_samples_used"], rows[-1]["accuracy"])
    if len(series) < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (cost, acc) in series.items():
        ax.scatter(cost, acc, s=90)
        ax.annotate(name, (cost, acc), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_xlabel("labelled traces used for training")
    ax.set_ylabel("accuracy on the final round, percent")
    ax.set_title(f"Cost of staying accurate, {arm} arm")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, f"drift_03_cost_{arm}.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="baseline", choices=["baseline", "obfs4"])
    args = parser.parse_args()

    path = os.path.join(FIGURES_DIR, f"drift_metrics_{args.arm}.json")
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Run: python src/drift_eval.py --arm {args.arm}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    made = [p for p in (plot_decay(data, args.arm), plot_groups(data, args.arm),
                        plot_cost(data, args.arm)) if p]
    if not made:
        print("[!] Not enough snapshots yet for drift figures. "
              "Collect at least one longitudinal round.")
    for p in made:
        print(f"[+] wrote {p}")


if __name__ == "__main__":
    main()
