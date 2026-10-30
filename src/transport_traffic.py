"""Why DF transfers badly between transports: what each transport does to a trace.

The kiírás asks for the causes of the cross-transport drop "a forgalom
jellemzői alapján". This reads every monitored trace of a Tor Browser
collection, in the same cells view DF gets (514-byte units, ACKs dropped,
first --length cells), and reports per transport:

  - trace shape: cells in/out, outgoing share, duration, direction runs
    ("bursts"), and the packet size distribution behind the cell counts;
  - the per-site shift against baseline (median of the site's obfs4 or
    Snowflake traces over the median of its baseline traces), and whether
    the transports keep the sites' order (Spearman over site medians);
  - within-site spread, which is what makes a transport hard to learn from;
  - transfer of a random forest on eight such coarse features, as is and
    after rescaling every transport by its own median and sd. The rescaling
    uses no labels of the target transport, only its traffic, so it tests
    whether the drop is a systematic shift or lost information.

Sites are kept as in evaluate_transport.py (--min-visits on every arm), and the
forest uses the same chronological per-site split. Output:
figures/traffic_<c>.json, figures/thesis/<c>_traffic_shift.pdf and
<c>_traffic_transfer.pdf (the latter with DF's zero-shot accuracy beside it
when figures/metrics_transport_<c>_cells_gpu.json exists).

    python src/transport_traffic.py --collection s2
"""

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from df_dataset import CELL_BYTES, chronological_split, parse_trace_name  # noqa: E402
from thesis_figures import (ARM_COLOR, ARM_LABEL, ARM_MARKER, ARMS, FIGURES, INK,  # noqa: E402
                            INK_2, MUTED, load, save, style)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TB_ROOT = os.getenv("TOR_WF_TB_OUT", os.path.join(REPO_ROOT, "tor_dataset", "tb"))

# Packet size bins in bytes, chosen by what a packet becomes in cells mode:
# under ~100 is a bare ACK, under 257 rounds to zero cells, then 1, 2, 3, 4+.
SIZE_BINS = (0, 100, 257, 771, 1285, 1799, np.inf)
SIZE_BIN_LABELS = ("<100 (ACK)", "100-256 (0 cells)", "257-770 (1)", "771-1284 (2)",
                   "1285-1798 (3)", ">=1799 (4+)")
# The forest's features: what is left of a trace without its order.
COARSE = ("cells_in", "cells_out", "out_share", "duration", "bursts", "in_burst_med",
          "out_burst_med", "in_burst_max")
LOGGED = ("cells_in", "cells_out", "duration", "bursts", "in_burst_max")
SHIFT = ("cells_in", "cells_out", "bytes_in", "bytes_out", "duration", "bursts",
         "in_burst_med", "out_burst_med", "out_share")
SHIFT_LABEL = {"cells_in": "bejövő cellák", "cells_out": "kimenő cellák",
               "bursts": "irányváltások", "in_burst_med": "bejövő löket\n(medián)"}


def trace_stats(df, length=10000):
    """One trace's shape in DF's cells view, or None for an empty one."""
    sizes = df["direction_size"].to_numpy(dtype=np.float64)
    if not len(sizes):
        return None
    mag, out = np.abs(sizes), sizes > 0
    cells = np.repeat(np.sign(sizes), np.rint(mag / CELL_BYTES).astype(np.int64))
    if not len(cells):
        return None
    # Bursts are runs of one direction in the window DF reads.
    w = cells[:length]
    starts = np.r_[0, np.flatnonzero(np.diff(w) != 0) + 1]
    runs, run_dir = np.diff(np.r_[starts, len(w)]), w[starts]
    in_runs, out_runs = runs[run_dir < 0], runs[run_dir > 0]
    return {
        "packets": int(len(sizes)),
        "bytes_in": float(mag[~out].sum()), "bytes_out": float(mag[out].sum()),
        "cells_in": int((cells < 0).sum()), "cells_out": int((cells > 0).sum()),
        "out_share": float((w > 0).mean()),
        "duration": float(df["time_offset"].iloc[-1]),
        "bursts": int(len(runs)),
        "in_burst_med": float(np.median(in_runs)) if len(in_runs) else 0.0,
        "out_burst_med": float(np.median(out_runs)) if len(out_runs) else 0.0,
        "in_burst_max": int(in_runs.max()) if len(in_runs) else 0,
        "hist_in": np.histogram(mag[~out], SIZE_BINS)[0].tolist(),
        "hist_out": np.histogram(mag[out], SIZE_BINS)[0].tolist(),
    }


def _one(task):
    arm, path, length = task
    try:
        df = pd.read_csv(path, usecols=["time_offset", "direction_size"])
    except Exception:
        return None
    row = trace_stats(df, length)
    if row is None:
        return None
    site, stamp = parse_trace_name(path)
    return {"arm": arm, "site": site, "stamp": stamp, **row}


def load_traces(collection, length, workers):
    tasks = [(arm, p, length) for arm in ARMS
             for p in sorted(glob.glob(os.path.join(TB_ROOT, collection, "features", arm,
                                                    "monitored", "*.csv")))]
    with ProcessPoolExecutor(workers or None) as ex:
        rows = [r for r in ex.map(_one, tasks, chunksize=32) if r]
    return pd.DataFrame(rows)


def keep_sites(df, min_visits):
    """The sites evaluate_transport.py keeps: on every arm, min_visits or more."""
    counts = df.groupby(["site", "arm"]).size().unstack().reindex(columns=list(ARMS))
    keep = counts.index[(counts.fillna(0) >= min_visits).all(axis=1)]
    return df[df["site"].isin(keep)].copy(), sorted(set(counts.index) - set(keep))


def summary(df):
    out = {}
    for arm, d in df.groupby("arm"):
        entry = {f"{k}_median": round(float(d[k].median()), 4)
                 for k in ("packets", "cells_in", "cells_out", "out_share", "duration",
                           "bursts", "in_burst_med", "out_burst_med", "in_burst_max")}
        for direction in ("in", "out"):
            h = np.sum(d[f"hist_{direction}"].tolist(), axis=0)
            entry[f"packet_sizes_{direction}_pct"] = dict(zip(
                SIZE_BIN_LABELS, (100 * h / h.sum()).round(2).tolist()))
            nbytes = d[f"bytes_{direction}"].sum()
            # How many cells the reconstruction finds per 514 bytes sent: under 1
            # means small packets round away, which is where Snowflake loses some.
            entry[f"cells_per_cell_bytes_{direction}"] = round(
                float(d[f"cells_{direction}"].sum() / (nbytes / CELL_BYTES)), 3)
        spread = d.groupby("site")
        entry["within_site_cv_cells_median"] = round(float(spread.apply(
            lambda s: (s.cells_in + s.cells_out).std() / (s.cells_in + s.cells_out).mean(),
            include_groups=False).median()), 4)
        entry["within_site_sd_out_share_median"] = round(float(
            spread["out_share"].std().median()), 4)
        entry["n_traces"] = int(len(d))
        out[arm] = entry
    return out


def site_shift(df):
    """Per site: target-arm median over baseline median, and rank agreement."""
    med = df.groupby(["site", "arm"])[list(SHIFT)].median()
    base = med.xs("baseline", level="arm")
    out = {}
    for arm in ARMS[1:]:
        target = med.xs(arm, level="arm")
        out[arm] = {}
        for k in SHIFT:
            r = target[k] / base[k]
            # Snowflake's outgoing runs are one cell at every site: no order to compare.
            rho = (spearmanr(target[k], base[k]).statistic
                   if target[k].nunique() > 1 and base[k].nunique() > 1 else None)
            out[arm][k] = {
                "median_ratio": round(float(r.median()), 3),
                "q25": round(float(r.quantile(0.25)), 3), "q75": round(float(r.quantile(0.75)), 3),
                "spearman": None if rho is None else round(float(rho), 3),
                "per_site": {s: round(float(v), 4) for s, v in r.items()},
            }
    return out


def coarse_transfer(df, seeds, test_size):
    """Forest on COARSE features, trained per arm, tested on every arm."""
    X = df[list(COARSE)].astype(float).copy()
    for k in LOGGED:
        X[k] = np.log1p(X[k])
    scaled = X.copy()
    for arm in ARMS:
        m = (df["arm"] == arm).to_numpy()
        scaled.loc[m] = (X.loc[m] - X.loc[m].median()) / X.loc[m].std()
    split = {}
    for arm in ARMS:
        idx = np.flatnonzero((df["arm"] == arm).to_numpy())
        tr, te = chronological_split(df["site"].to_numpy()[idx],
                                     df["stamp"].to_numpy(dtype="datetime64[ns]")[idx], test_size)
        split[arm] = (idx[tr], idx[te])
    y = df["site"].to_numpy()
    out = {}
    for name, feats in (("raw", X), ("per_arm_scaled", scaled)):
        F = feats.to_numpy()
        for a in ARMS:
            for seed in seeds:
                rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
                rf.fit(F[split[a][0]], y[split[a][0]])
                for b in ARMS:
                    te = split[b][1]
                    out.setdefault(name, {}).setdefault(f"{a}->{b}", []).append(
                        float(rf.score(F[te], y[te])))
        for pair, accs in out[name].items():
            out[name][pair] = {"mean": float(np.mean(accs)), "sd": float(np.std(accs, ddof=1)),
                               "runs": accs}
    return out


def shift_figure(shift, collection):
    import matplotlib.pyplot as plt

    feats = list(SHIFT_LABEL)
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    rng = np.random.default_rng(0)
    for i, k in enumerate(feats):
        for j, arm in enumerate(ARMS[1:]):
            x0 = i + (j - 0.5) * 0.36
            vals = np.array(list(shift[arm][k]["per_site"].values()))
            ax.scatter(x0 + rng.uniform(-0.09, 0.09, len(vals)), vals, s=9,
                       color=ARM_COLOR[arm], alpha=0.55, linewidths=0,
                       label=ARM_LABEL[arm] if i == 0 else None)
            med = shift[arm][k]["median_ratio"]
            ax.plot([x0 - 0.14, x0 + 0.14], [med, med], color=INK, lw=2, solid_capstyle="round")
            ax.annotate(f"×{med:.2f}", (x0, vals.max()), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=7, color=INK_2)
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.set_yscale("log", base=2)
    ax.set_ylim(0.4, 3)
    ax.set_yticks([0.5, 1, 2])
    ax.set_yticklabels(["×½", "×1", "×2"])
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels([SHIFT_LABEL[k] for k in feats])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("oldalanként, a baseline-hoz képest")
    ax.legend(loc="upper left", ncol=2, markerscale=1.6, handletextpad=0.2)
    save(fig, collection, "traffic_shift")


def transfer_figure(transfer, collection):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    df_run = load(f"metrics_transport_{collection}_cells_gpu.json")
    pairs = [f"{a}->{b}" for a in ARMS for b in ARMS if a != b]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for i, pair in enumerate(pairs):
        y = len(pairs) - 1 - i
        raw = 100 * transfer["raw"][pair]["mean"]
        scaled = 100 * transfer["per_arm_scaled"][pair]["mean"]
        target = pair.split("->")[1]
        ax.plot([raw, scaled], [y, y], color=ARM_COLOR[target], lw=2, alpha=0.5,
                solid_capstyle="round")
        ax.scatter(raw, y, s=40, facecolor="white", edgecolor=ARM_COLOR[target],
                   linewidths=1.6, marker=ARM_MARKER[target], zorder=3)
        ax.scatter(scaled, y, s=40, color=ARM_COLOR[target], marker=ARM_MARKER[target],
                   edgecolor=ARM_COLOR[target], linewidths=1.6, zorder=3)
        if df_run:
            ax.scatter(100 * df_run["cross_matrix"][pair]["mean"], y, s=34, marker="D",
                       color=INK, zorder=4)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([" → ".join(ARM_LABEL[a] for a in p.split("->")) for p in pairs[::-1]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("pontosság a célmódon (%)")
    ax.grid(axis="y", visible=False)
    handles = [Line2D([], [], ls="", marker="o", mfc="white", mec=MUTED, mew=1.6, ms=6,
                      label="8 jellemző, RF"),
               Line2D([], [], ls="", marker="o", color=MUTED, ms=6,
                      label="ugyanaz, módonként átskálázva")]
    if df_run:
        handles.append(Line2D([], [], ls="", marker="D", color=INK, ms=5, label="DF"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
              handletextpad=0.2, columnspacing=1.0)
    save(fig, collection, "traffic_transfer")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", required=True)
    p.add_argument("--length", type=int, default=10000, help="cells DF reads (bursts are counted in these)")
    p.add_argument("--min-visits", type=int, default=30)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--workers", type=int, default=0, help="0: one per core")
    args = p.parse_args()

    df = load_traces(args.collection, args.length, args.workers)
    if df.empty:
        sys.exit(f"no monitored traces under {os.path.join(TB_ROOT, args.collection, 'features')}")
    df, dropped = keep_sites(df, args.min_visits)
    print(f"{args.collection}: {len(df)} traces, {df['site'].nunique()} sites"
          f" (--min-visits {args.min_visits} drops {dropped})")

    result = {
        "config": {**vars(args), "cell_bytes": CELL_BYTES, "coarse_features": list(COARSE),
                   "sites": sorted(df["site"].unique()), "dropped_sites": dropped},
        "arms": summary(df),
        "shift_vs_baseline": site_shift(df),
        "coarse_transfer": coarse_transfer(df, [int(s) for s in args.seeds.split(",")],
                                           args.test_size),
    }

    s, t = result["arms"], result["coarse_transfer"]
    print(f"\n{'':22s}" + "".join(f"{a:>11s}" for a in ARMS))
    for k in ("cells_in_median", "cells_out_median", "out_share_median", "duration_median",
              "bursts_median", "in_burst_med_median", "cells_per_cell_bytes_out",
              "within_site_cv_cells_median"):
        print(f"{k:22.22s}" + "".join(f"{s[a][k]:>11.3f}" for a in ARMS))
    for arm, entry in result["shift_vs_baseline"].items():
        print(f"\n{arm} / baseline per site: " + ", ".join(
            f"{k} ×{v['median_ratio']:.2f}"
            + (f" (ρ {v['spearman']:.2f})" if v["spearman"] is not None else "")
            for k, v in entry.items()))
    print("\ncoarse-feature forest, accuracy % (raw / per-arm scaled), rows train, columns test")
    for a in ARMS:
        print(f"{a:>10s} " + "".join(
            f"{100 * t['raw'][f'{a}->{b}']['mean']:7.1f} /{100 * t['per_arm_scaled'][f'{a}->{b}']['mean']:5.1f}"
            for b in ARMS))

    path = os.path.join(FIGURES, f"traffic_{args.collection}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {os.path.relpath(path, REPO_ROOT)}")
    style()
    shift_figure(result["shift_vs_baseline"], args.collection)
    transfer_figure(result["coarse_transfer"], args.collection)


if __name__ == "__main__":
    main()
