"""Extract per-packet feature CSVs for one longitudinal round.

Deliberately reuses extract_features() from extract_all_features.py rather than
reimplementing it.  The whole drift measurement rests on the assumption that a
feature computed in September means the same thing as one computed in March, so
there must be exactly one implementation of it in the repository.

    python src/extract_round_features.py --round latest
    python src/extract_round_features.py --round 2026-W37
    python src/extract_round_features.py --round all
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_all_features import extract_features  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LONGITUDINAL_DIR = os.getenv(
    "TOR_WF_LONGITUDINAL_DIR", os.path.join(REPO_ROOT, "tor_dataset", "longitudinal")
)
ARMS = ("baseline", "obfs4")


def available_rounds():
    if not os.path.isdir(LONGITUDINAL_DIR):
        return []
    return sorted(
        d for d in os.listdir(LONGITUDINAL_DIR)
        if os.path.isdir(os.path.join(LONGITUDINAL_DIR, d)) and not d.startswith("_")
    )


def resolve_rounds(selector):
    rounds = available_rounds()
    if not rounds:
        sys.exit(f"No rounds found under {LONGITUDINAL_DIR}. Collect one first.")
    if selector == "latest":
        return rounds[-1:]
    if selector == "all":
        return rounds
    if selector not in rounds:
        sys.exit(f"Round '{selector}' not found. Available: {', '.join(rounds)}")
    return [selector]


def extract_arm(round_id, arm, overwrite=False):
    pcap_dir = os.path.join(LONGITUDINAL_DIR, round_id, arm)
    if not os.path.isdir(pcap_dir):
        return 0, 0
    out_dir = os.path.join(LONGITUDINAL_DIR, round_id, f"{arm}_features")
    os.makedirs(out_dir, exist_ok=True)

    # Only top-level PCAPs are used. Anything the collector quarantined under
    # _rejected/ stays out of the feature set by construction.
    pcaps = sorted(
        f for f in os.listdir(pcap_dir)
        if f.endswith(".pcap") and os.path.isfile(os.path.join(pcap_dir, f))
    )
    written, skipped = 0, 0
    for i, name in enumerate(pcaps, start=1):
        out_path = os.path.join(out_dir, name.replace(".pcap", ".csv"))
        if os.path.exists(out_path) and not overwrite:
            skipped += 1
            continue
        df = extract_features(os.path.join(pcap_dir, name))
        if df.empty:
            skipped += 1
            continue
        df.to_csv(out_path, index=False)
        written += 1
        if i % 50 == 0:
            print(f"    {round_id}/{arm}: {i}/{len(pcaps)}")
    return written, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="latest")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for round_id in resolve_rounds(args.round):
        print(f"[+] extracting round {round_id}")
        summary = {}
        for arm in ARMS:
            written, skipped = extract_arm(round_id, arm, overwrite=args.overwrite)
            if written or skipped:
                summary[arm] = {"written": written, "skipped": skipped}
                print(f"    {arm}: {written} written, {skipped} skipped")
        meta_path = os.path.join(LONGITUDINAL_DIR, round_id, "round_meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        meta["feature_extraction"] = summary
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()
