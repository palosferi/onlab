"""Post-round health check.

The series runs unattended for thirteen weeks. The failure mode that costs a
thesis is not a crash, which is obvious, but silent degradation: a round that
still produces files while quietly yielding too few usable captures per site to
analyse. This turns that into a number, writes it into the round metadata, and
exits non-zero when a round is not fit to use.

    python src/round_health.py --round latest
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts", "collection"))
import wf_config as cfg  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LONGITUDINAL_DIR = os.getenv(
    "TOR_WF_LONGITUDINAL_DIR", os.path.join(REPO_ROOT, "tor_dataset", "longitudinal")
)

# Below these a round cannot carry its own weight in the analysis.
MIN_USABLE_RATIO = float(os.getenv("TOR_WF_MIN_USABLE_RATIO", "0.55"))
MIN_SAMPLES_PER_SITE = int(os.getenv("TOR_WF_MIN_SAMPLES_PER_SITE", "3"))
MIN_SITES = int(os.getenv("TOR_WF_MIN_SITES", "20"))


def resolve_round(selector):
    if not os.path.isdir(LONGITUDINAL_DIR):
        sys.exit(f"No rounds under {LONGITUDINAL_DIR}")
    rounds = sorted(d for d in os.listdir(LONGITUDINAL_DIR)
                    if os.path.isdir(os.path.join(LONGITUDINAL_DIR, d)))
    if not rounds:
        sys.exit("No rounds collected yet")
    if selector == "latest":
        return rounds[-1]
    if selector not in rounds:
        sys.exit(f"Round {selector} not found. Have: {', '.join(rounds)}")
    return selector


def analyse(round_id):
    manifest = os.path.join(LONGITUDINAL_DIR, round_id, "manifest.csv")
    if not os.path.exists(manifest):
        sys.exit(f"No manifest for round {round_id}")
    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    targets = set(cfg.TARGET_SITES)
    report = {"round_id": round_id, "attempts": len(rows), "arms": {}}
    problems = []

    for arm in sorted({r["arm"] for r in rows}):
        arm_rows = [r for r in rows if r["arm"] == arm]
        usable = Counter(r["site"] for r in arm_rows if r["status"] == "ok")
        total = len(arm_rows)
        ok = sum(usable.values())
        ratio = ok / total if total else 0.0
        enough = {s for s, n in usable.items() if n >= MIN_SAMPLES_PER_SITE}
        zero = sorted(targets - set(usable))
        thin = sorted(s for s, n in usable.items() if n < MIN_SAMPLES_PER_SITE)

        report["arms"][arm] = {
            "attempts": total,
            "usable": ok,
            "usable_ratio": round(ratio, 4),
            "sites_with_enough": len(enough),
            "sites_zero_usable": zero,
            "sites_thin": thin,
            "failure_reasons": dict(Counter(r["status"] for r in arm_rows if r["status"] != "ok")),
        }

        if ratio < MIN_USABLE_RATIO:
            problems.append(f"{arm}: only {ratio:.0%} usable (floor {MIN_USABLE_RATIO:.0%})")
        if len(enough) < MIN_SITES:
            problems.append(
                f"{arm}: only {len(enough)} sites reached {MIN_SAMPLES_PER_SITE} usable "
                f"captures (floor {MIN_SITES})"
            )

    report["problems"] = problems
    report["healthy"] = not problems
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="latest")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    round_id = resolve_round(args.round)
    report = analyse(round_id)

    meta_path = os.path.join(LONGITUDINAL_DIR, round_id, "round_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    meta["health"] = report
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if not args.quiet:
        print(f"=== round {round_id} health ===")
        for arm, a in report["arms"].items():
            print(f"  {arm}: {a['usable']}/{a['attempts']} usable ({a['usable_ratio']:.0%}), "
                  f"{a['sites_with_enough']} sites with {MIN_SAMPLES_PER_SITE}+ captures")
            if a["sites_zero_usable"]:
                print(f"     no usable data: {', '.join(a['sites_zero_usable'])}")
            if a["failure_reasons"]:
                print(f"     failures: {a['failure_reasons']}")
        if report["healthy"]:
            print("  VERDICT: healthy")
        else:
            print("  VERDICT: PROBLEMS")
            for p in report["problems"]:
                print(f"     - {p}")

    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())
