"""Side-by-side table of the same scenarios on two or more collections.

Reads results files of evaluate_transport.py or evaluate_budget.py (one per
collection, same script) and prints every scenario they share as mean +- sd
over seeds: accuracy, or TPR / FPR for the open-world scenarios. With two
files a last column gives the difference of the means (second minus first), in
percentage points. The files should come from the same hardware: a table never
mixes CPU and GPU seeds.

    python src/compare_collections.py figures/metrics_transport_tb-main_cells_gpu.json \\
        figures/metrics_transport_s2_cells_gpu.json
    python src/compare_collections.py A.json B.json --latex > table.tex
"""

import argparse
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["config"].get("collection") or path, data["config"].get("device", "cpu"), data["scenarios"]


def metrics(entry):
    """[(name, mean, sd)] in percent: accuracy, or TPR and FPR for an open world."""
    if "accuracy_mean" in entry:
        return [("acc", 100 * entry["accuracy_mean"], 100 * entry["accuracy_sd"])]
    if "tpr_at_zero_threshold_mean" in entry:
        return [(m, 100 * entry[f"{m.lower()}_at_zero_threshold_mean"],
                 100 * entry[f"{m.lower()}_at_zero_threshold_sd"]) for m in ("TPR", "FPR")]
    return []


def cell(ms, latex):
    pm = r" $\pm$ " if latex else " +- "
    if len(ms) == 1:
        return f"{ms[0][1]:.1f}{pm}{ms[0][2]:.1f}"
    return " / ".join(f"{name} {m:.1f}{pm}{sd:.1f}" for name, m, sd in ms)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="results JSONs, one per collection")
    p.add_argument("--labels", default="", help="comma separated column names (default: collection)")
    p.add_argument("--latex", action="store_true", help="a tabular body instead of Markdown")
    args = p.parse_args()

    runs = [load(f) for f in args.files]
    devices = {d for _, d, _ in runs}
    if len(devices) > 1:
        sys.exit(f"files mix devices {sorted(devices)}; compare runs from the same hardware")
    labels = args.labels.split(",") if args.labels else [c for c, _, _ in runs]
    if len(labels) != len(runs):
        sys.exit("--labels needs one name per file")

    # Scenario order of the first file, keeping only those every file has.
    shared = [s for s in runs[0][2] if all(s in r[2] for r in runs[1:])]
    header = ["scenario", *labels] + (["diff"] if len(runs) == 2 else [])
    rows = []
    for s in shared:
        ms = [metrics(r[2][s]) for r in runs]
        if not all(ms):
            continue
        row = [s.replace("_", r"\_") if args.latex else s, *(cell(m, args.latex) for m in ms)]
        if len(runs) == 2:
            row.append(" / ".join(f"{b[1] - a[1]:+.1f}" for a, b in zip(ms[0], ms[1])))
        rows.append(row)

    if args.latex:
        print(r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}")
        print(r"\toprule")
        print(" & ".join(header) + r" \\")
        print(r"\midrule")
        for row in rows:
            print(" & ".join(row) + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
    else:
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for row in rows:
            print("| " + " | ".join(row) + " |")
    missing = [s for s in runs[0][2] if s not in shared]
    if missing:
        print(f"\n(not in every file: {', '.join(missing)})", file=sys.stderr)


if __name__ == "__main__":
    main()
