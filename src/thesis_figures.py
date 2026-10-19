"""Thesis figures for one Tor Browser collection, from its GPU result files.

Reads figures/metrics_transport_<c>_cells_gpu.json, the budget file next to it
and their _precision.json (R3), and writes vector PDFs to figures/thesis/:

  <c>_cross_matrix.pdf   train transport (rows) x test transport (columns), accuracy
  <c>_budget.pdf         R1: baseline plus k target traces per site, per target
  <c>_curve.pdf          R4: baseline closed world on n training traces per site
  <c>_precision.pdf      R3: best precision (TPR >= 50%) against the base rate, per arm

Every point is the mean over seeds with its sd as an error bar. Labels are in
Hungarian, as the thesis is. A figure whose input file is missing is skipped.

    python src/thesis_figures.py --collection tb-main
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES = os.path.join(REPO_ROOT, "figures")

ARMS = ("baseline", "obfs4", "snowflake")
ARM_LABEL = {"baseline": "baseline", "obfs4": "obfs4", "snowflake": "Snowflake"}
# One fixed colour per transport in every figure (categorical slots 1-3, which
# stay apart for colour-blind readers in any pairing).
ARM_COLOR = {"baseline": "#2a78d6", "obfs4": "#eb6834", "snowflake": "#1baf7a"}
ARM_MARKER = {"baseline": "o", "obfs4": "s", "snowflake": "^"}
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "xtick.color": MUTED,
        "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
        "pdf.fonttype": 42, "savefig.bbox": "tight",
    })


def load(name):
    path = os.path.join(FIGURES, name)
    if not os.path.exists(path):
        print(f"  skip: no {name}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def acc(entry):
    return 100 * entry["accuracy_mean"], 100 * entry["accuracy_sd"]


def save(fig, collection, name):
    os.makedirs(os.path.join(FIGURES, "thesis"), exist_ok=True)
    path = os.path.join(FIGURES, "thesis", f"{collection}_{name}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)}")


def cross_matrix(transport, collection):
    m = transport["cross_matrix"]
    vals = np.array([[100 * m[f"{a}->{b}"]["mean"] for b in ARMS] for a in ARMS])
    sds = np.array([[100 * m[f"{a}->{b}"]["sd"] for b in ARMS] for a in ARMS])
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    im = ax.imshow(vals, cmap=cmap, vmin=0, vmax=100)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            dark = vals[i, j] > 60
            ax.text(j, i, f"{vals[i, j]:.1f}\n± {sds[i, j]:.1f}", ha="center", va="center",
                    fontsize=8.5, color="white" if dark else INK)
    labels = [ARM_LABEL[a] for a in ARMS]
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("teszt transzport")
    ax.set_ylabel("tanító transzport")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("pontosság (%)")
    cb.outline.set_visible(False)
    save(fig, collection, "cross_matrix")


def budget(scen, transport, collection):
    ks = ["0", "1", "3", "5", "10", "all"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = np.arange(len(ks))
    for target in ("obfs4", "snowflake"):
        keys = [f"budget_baseline_to_{target}_k{k}" for k in ks]
        if not all(k in scen for k in keys):
            continue
        mean, sd = zip(*(acc(scen[k]) for k in keys))
        ax.errorbar(x, mean, yerr=sd, color=ARM_COLOR[target], marker=ARM_MARKER[target],
                    markersize=5, linewidth=2, capsize=3, elinewidth=1,
                    label=f"→ {ARM_LABEL[target]}")
        cw = transport["scenarios"].get(f"closed_world_{target}") if transport else None
        if cw:
            ax.axhline(acc(cw)[0], color=ARM_COLOR[target], linewidth=1, linestyle=":")
    ax.set_xticks(x, ["0\n(zero-shot)", "1", "3", "5", "10", "összes\n(pooled)"])
    ax.set_xlabel("célforgalmi tanító minták oldalanként (k)")
    ax.set_ylabel("pontosság a cél transzporton (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", title="baseline + k minta", title_fontsize=8)
    ax.text(0.01, 0.02, "pontozott: a cél transzport zárt világú pontossága",
            transform=ax.transAxes, fontsize=7, color=MUTED)
    save(fig, collection, "budget")


def curve(scen, collection):
    ns = ["5", "10", "15", "20", "all"]
    keys = [f"curve_baseline_n{n}" for n in ns]
    if not all(k in scen for k in keys):
        print("  skip: curve scenarios incomplete")
        return
    mean, sd = zip(*(acc(scen[k]) for k in keys))
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    x = np.arange(len(ns))
    ax.errorbar(x, mean, yerr=sd, color=ARM_COLOR["baseline"], marker="o", markersize=5,
                linewidth=2, capsize=3, elinewidth=1)
    for xi, m in zip(x, mean):
        ax.annotate(f"{m:.1f}", (xi, m), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, ns[:-1] + ["összes"])
    ax.set_xlabel("tanító minták oldalanként (n)")
    ax.set_ylabel("zárt világú pontosság, baseline (%)")
    ax.set_ylim(min(mean) - 15, 100)
    save(fig, collection, "curve")


def precision(prec, collection):
    rates = [b for b in prec["base_rates"] if b <= 0.2]
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    for arm in ARMS:
        e = prec["scenarios"].get(f"open_world_{arm}")
        if not e:
            continue
        mean = [100 * e["base_rates"][str(b)]["best_precision_mean"] for b in rates]
        sd = [100 * e["base_rates"][str(b)]["best_precision_sd"] for b in rates]
        ax.errorbar(rates, mean, yerr=sd, color=ARM_COLOR[arm], marker=ARM_MARKER[arm],
                    markersize=5, linewidth=2, capsize=3, elinewidth=1, label=ARM_LABEL[arm])
    ax.set_xscale("log")
    ax.set_xticks(rates, [f"{100 * b:g}%" for b in rates])
    ax.minorticks_off()
    ax.invert_xaxis()
    ax.set_xlabel("alapráta: a monitorozott oldalakra eső oldalbetöltések aránya")
    ax.set_ylabel("legjobb precizitás, TPR ≥ 50% (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    save(fig, collection, "precision")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--suffix", default="cells_gpu", help="results file suffix after the collection")
    args = p.parse_args()
    style()
    c, s = args.collection, args.suffix
    transport = load(f"metrics_transport_{c}_{s}.json")
    budget_run = load(f"metrics_budget_{c}_{s}.json")
    prec = load(f"metrics_transport_{c}_{s}_precision.json")
    if transport:
        cross_matrix(transport, c)
    if budget_run:
        budget(budget_run["scenarios"], transport, c)
        curve(budget_run["scenarios"], c)
    if prec:
        precision(prec, c)


if __name__ == "__main__":
    main()
