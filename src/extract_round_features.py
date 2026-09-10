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


def extract_arm(round_id, arm, overwrite=False, rejected=False):
    """Extract one arm. With rejected=True, process the quarantine instead.

    Quarantined captures are labelled failures: challenge pages, browser error
    pages, empty responses. Their features are worth having separately, because
    the spring t0 was collected without any validation and therefore probably
    contains challenge traffic labelled as real sites. These give a labelled
    set of what that traffic looks like, so the contamination can be estimated
    rather than only acknowledged.
    """
    base = os.path.join(LONGITUDINAL_DIR, round_id, arm)
    pcap_dir = os.path.join(base, "_rejected") if rejected else base
    if not os.path.isdir(pcap_dir):
        return 0, 0
    suffix = "_rejected_features" if rejected else "_features"
    out_dir = os.path.join(LONGITUDINAL_DIR, round_id, f"{arm}{suffix}")
    os.makedirs(out_dir, exist_ok=True)

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
    parser.add_argument("--include-rejected", action="store_true",
                        help="also extract the quarantined captures into "
                             "<arm>_rejected_features/")
    args = parser.parse_args()

    for round_id in resolve_rounds(args.round):
        print(f"[+] extracting round {round_id}")
        summary = {}
        for arm in ARMS:
            written, skipped = extract_arm(round_id, arm, overwrite=args.overwrite)
            if written or skipped:
                summary[arm] = {"written": written, "skipped": skipped}
                print(f"    {arm}: {written} written, {skipped} skipped")
            if args.include_rejected:
                rw, rs = extract_arm(round_id, arm, overwrite=args.overwrite, rejected=True)
                if rw or rs:
                    summary[f"{arm}_rejected"] = {"written": rw, "skipped": rs}
                    print(f"    {arm} rejected: {rw} written, {rs} skipped")
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
