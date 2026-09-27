"""Extract per-packet feature CSVs for a Tor Browser collection.

Mirrors tor_dataset/tb/<collection>/<arm>/<kind>/*.pcap into
tor_dataset/tb/<collection>/features/<arm>/<kind>/*.csv, reusing the one
extract_features() implementation so these traces mean the same thing as the
spring and longitudinal ones.

    python src/extract_tb_features.py --collection tb-2026-10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract_all_features import extract_features  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TB_ROOT = os.getenv("TOR_WF_TB_OUT", os.path.join(REPO_ROOT, "tor_dataset", "tb"))


def pcap_dirs(root):
    """(arm, kind, directory) for every accepted-capture folder."""
    for arm in sorted(os.listdir(root)):
        arm_dir = os.path.join(root, arm)
        if arm == "features" or not os.path.isdir(arm_dir):
            continue
        for kind in sorted(os.listdir(arm_dir)):
            kind_dir = os.path.join(arm_dir, kind)
            if os.path.isdir(kind_dir) and not kind.startswith("_"):
                yield arm, kind, kind_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection", required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    root = os.path.join(TB_ROOT, args.collection)
    if not os.path.isdir(root):
        sys.exit(f"no such collection: {root}")

    for arm, kind, src in pcap_dirs(root):
        out = os.path.join(root, "features", arm, kind)
        os.makedirs(out, exist_ok=True)
        pcaps = sorted(f for f in os.listdir(src) if f.endswith(".pcap"))
        written = skipped = 0
        for name in pcaps:
            dst = os.path.join(out, name[:-5] + ".csv")
            if os.path.exists(dst) and not args.overwrite:
                skipped += 1
                continue
            df = extract_features(os.path.join(src, name))
            if df.empty:
                skipped += 1
                continue
            df.to_csv(dst, index=False)
            written += 1
        print(f"{arm:10s} {kind:12s} {written:5d} written {skipped:5d} skipped")


if __name__ == "__main__":
    main()
