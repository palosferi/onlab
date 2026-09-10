"""Model-free evidence of drift: per-site distribution shift between snapshots.

Answers "did the traffic change?" without involving a classifier at all, using a
two-sample Kolmogorov-Smirnov test per feature per site.  This matters because a
drop in classifier accuracy has two possible causes, the traffic changing or the
model being fragile, and only this separates them.

It also produces a usable figure from the very first round, when there are still
too few snapshots for a decay curve.

    python src/drift_shift.py --arm baseline --reference t0-spring --target latest
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from drift_eval import FIGURES_DIR, SITE_GROUPS, load_series  # noqa: E402
from feature_pool import FEATURE_NAMES  # noqa: E402

GROUP_OF = {site: g for g, sites in SITE_GROUPS.items() for site in sites}
MIN_SAMPLES_PER_SITE = 3


def per_site_shift(ref, tgt, features):
    """KS statistic per site per feature, plus a per-site mean across features."""
    rows = []
    ref_df = ref.X.copy()
    ref_df["_site"] = ref.y
    tgt_df = tgt.X.copy()
    tgt_df["_site"] = tgt.y

    for site in sorted(set(ref.y) & set(tgt.y)):
        a = ref_df[ref_df["_site"] == site]
        b = tgt_df[tgt_df["_site"] == site]
        if len(a) < MIN_SAMPLES_PER_SITE or len(b) < MIN_SAMPLES_PER_SITE:
            continue
        entry = {"site": site, "group": GROUP_OF.get(site, "other"),
                 "n_ref": len(a), "n_target": len(b)}
        stats = []
        for feat in features:
            try:
                res = ks_2samp(a[feat].values, b[feat].values)
                entry[f"ks_{feat}"] = round(float(res.statistic), 4)
                entry[f"p_{feat}"] = round(float(res.pvalue), 6)
                stats.append(float(res.statistic))
            except Exception:
                pass
        if not stats:
            continue
        entry["ks_mean"] = round(float(np.mean(stats)), 4)
        # Benjamini-Hochberg would be stricter, but with 21 features per site a
        # plain Bonferroni threshold is enough to call a site "shifted".
        pvals = [entry[f"p_{f}"] for f in features if f"p_{f}" in entry]
        entry["n_features_shifted"] = int(sum(p < 0.05 / len(pvals) for p in pvals))
        rows.append(entry)
    return pd.DataFrame(rows).sort_values("ks_mean", ascending=False)


def plot_shift(df, ref_name, tgt_name, arm, out_path):
    if df.empty:
        print("[-] nothing to plot")
        return
    colors = {"static": "#4C72B0", "news": "#C44E52", "commerce_media": "#55A868",
              "other": "#8172B2"}
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(df))))
    bars = ax.barh(df["site"], df["ks_mean"],
                   color=[colors.get(g, "#999999") for g in df["group"]])
    ax.invert_yaxis()
    ax.set_xlabel("mean Kolmogorov-Smirnov statistic across 21 flow features")
    ax.set_title(f"Per-site traffic distribution shift, {arm} arm\n{ref_name} vs {tgt_name}")
    ax.grid(axis="x", alpha=0.3)
    for group, color in colors.items():
        if (df["group"] == group).any():
            ax.barh([], [], color=color, label=group)
    ax.legend(title="site group", loc="lower right")
    for bar, n in zip(bars, df["n_features_shifted"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{n}/21", va="center", fontsize=7, color="#444444")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[+] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="baseline", choices=["baseline", "obfs4"])
    parser.add_argument("--reference", default="t0-spring")
    parser.add_argument("--target", default="latest")
    args = parser.parse_args()

    snapshots = load_series(args.arm)
    by_name = {s.name: s for s in snapshots}
    if args.reference not in by_name:
        sys.exit(f"Reference '{args.reference}' not found. Have: {', '.join(by_name)}")
    if len(snapshots) < 2:
        sys.exit("Need at least two snapshots. Collect a longitudinal round first.")

    target_name = snapshots[-1].name if args.target == "latest" else args.target
    if target_name not in by_name:
        sys.exit(f"Target '{target_name}' not found. Have: {', '.join(by_name)}")

    ref, tgt = by_name[args.reference], by_name[target_name]
    print(f"=== shift {ref.name} -> {tgt.name} (arm {args.arm}) ===")
    df = per_site_shift(ref, tgt, list(FEATURE_NAMES))
    if df.empty:
        sys.exit(f"No site had at least {MIN_SAMPLES_PER_SITE} samples in both snapshots.")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    stem = f"drift_shift_{args.arm}_{ref.name}_to_{tgt.name}".replace("/", "_")
    df.to_csv(os.path.join(FIGURES_DIR, f"{stem}.csv"), index=False)
    plot_shift(df, ref.name, tgt.name, args.arm, os.path.join(FIGURES_DIR, f"{stem}.pdf"))

    group_means = df.groupby("group")["ks_mean"].mean().round(4).to_dict()
    summary = {
        "arm": args.arm, "reference": ref.name, "target": tgt.name,
        "sites_compared": int(len(df)),
        "mean_ks": round(float(df["ks_mean"].mean()), 4),
        "group_mean_ks": group_means,
        "most_shifted": df.head(5)[["site", "ks_mean", "n_features_shifted"]].to_dict("records"),
        "least_shifted": df.tail(5)[["site", "ks_mean", "n_features_shifted"]].to_dict("records"),
    }
    with open(os.path.join(FIGURES_DIR, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"    sites compared: {len(df)}   mean KS: {summary['mean_ks']}")
    print(f"    by group: {group_means}")
    print("    most shifted:")
    for r in summary["most_shifted"]:
        print(f"       {r['site']:14s} KS={r['ks_mean']:.3f}  {r['n_features_shifted']}/21 features")


if __name__ == "__main__":
    main()
