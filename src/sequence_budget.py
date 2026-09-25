"""How much of a trace a fixed-length packet window actually sees, per transport.

Deep Fingerprinting reads the first N packets of a trace and pads or truncates
to that length. N=5000 was fine while every arm was TCP, but the Snowflake
pilot showed its traces carry roughly four times as many packets as baseline
Tor for the same pages, because WebRTC wraps the same payload in far more,
far smaller datagrams.

That breaks the comparison in a way a single accuracy number would hide: with
the same N, the classifier sees most of a baseline page load and only part of a
Snowflake one, so a lower Snowflake score could mean "the transport hides more"
or merely "the window ended sooner". This reports, per arm, how much of a trace
a given N retains -- in packets and in bytes -- so N can be chosen from the data
and stated in the thesis rather than inherited from the paper.

Usage:
    python src/sequence_budget.py --spring
    python src/sequence_budget.py --round 2026-W39-pilot-snowflake
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPRING_DIR = os.path.join(REPO_ROOT, "tor_dataset", "extracted_features")
LONGITUDINAL_DIR = os.path.join(REPO_ROOT, "tor_dataset", "longitudinal")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")

DEFAULT_LENGTHS = (5000, 8000, 10000, 12000, 16000, 20000)


def trace_stats(path):
    """Per-trace shape: how many packets, how many bytes, how long."""
    try:
        df = pd.read_csv(path, usecols=["time_offset", "direction_size"])
    except Exception:
        return None
    if df.empty:
        return None
    sizes = df["direction_size"].to_numpy()
    magnitude = np.abs(sizes)
    return {
        "packets": int(len(sizes)),
        "bytes": int(magnitude.sum()),
        "median_packet_size": float(np.median(magnitude)),
        "duration_s": float(df["time_offset"].iloc[-1]),
        "outgoing_share": float((sizes > 0).mean()),
        # Cumulative bytes, so byte coverage at any N is one lookup.
        "cumulative_bytes": np.cumsum(magnitude),
    }


def coverage(traces, length):
    """What a window of `length` packets keeps, per trace, then summarised."""
    packet_frac, byte_frac, complete = [], [], 0
    for t in traces:
        n = t["packets"]
        kept = min(length, n)
        packet_frac.append(kept / n)
        byte_frac.append(float(t["cumulative_bytes"][kept - 1]) / max(t["bytes"], 1))
        complete += int(n <= length)
    return {
        "length": int(length),
        "traces_fully_covered": complete,
        "traces_fully_covered_pct": round(100 * complete / max(len(traces), 1), 1),
        "median_packets_kept_pct": round(100 * float(np.median(packet_frac)), 1),
        "median_bytes_kept_pct": round(100 * float(np.median(byte_frac)), 1),
    }


def describe(label, directory, lengths):
    if not os.path.isdir(directory):
        return None
    files = sorted(f for f in os.listdir(directory) if f.endswith(".csv"))
    traces = [s for s in (trace_stats(os.path.join(directory, f)) for f in files) if s]
    if not traces:
        return None

    packets = np.array([t["packets"] for t in traces])
    return {
        "arm": label,
        "directory": os.path.relpath(directory, REPO_ROOT),
        "n_traces": len(traces),
        "packets_median": int(np.median(packets)),
        "packets_p90": int(np.percentile(packets, 90)),
        "packets_max": int(packets.max()),
        "bytes_median": int(np.median([t["bytes"] for t in traces])),
        "median_packet_size": round(float(np.median([t["median_packet_size"] for t in traces])), 1),
        "duration_s_median": round(float(np.median([t["duration_s"] for t in traces])), 1),
        "outgoing_share_median": round(float(np.median([t["outgoing_share"] for t in traces])), 3),
        "coverage": [coverage(traces, n) for n in lengths],
    }


def print_report(arms, lengths):
    print(f"\n{'arm':28s} {'traces':>6s} {'pkts med':>9s} {'pkts p90':>9s} "
          f"{'bytes med':>11s} {'pkt size':>9s} {'dur s':>7s}")
    print("-" * 84)
    for a in arms:
        print(f"{a['arm']:28s} {a['n_traces']:>6d} {a['packets_median']:>9d} "
              f"{a['packets_p90']:>9d} {a['bytes_median']:>11d} "
              f"{a['median_packet_size']:>9.0f} {a['duration_s_median']:>7.1f}")

    print("\nmedian % of each trace's BYTES inside the first N packets")
    header = f"{'arm':28s}" + "".join(f"{n:>9d}" for n in lengths)
    print(header)
    print("-" * len(header))
    for a in arms:
        row = "".join(f"{c['median_bytes_kept_pct']:>9.1f}" for c in a["coverage"])
        print(f"{a['arm']:28s}{row}")

    print("\n% of traces that fit entirely inside the first N packets")
    print(header)
    print("-" * len(header))
    for a in arms:
        row = "".join(f"{c['traces_fully_covered_pct']:>9.1f}" for c in a["coverage"])
        print(f"{a['arm']:28s}{row}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spring", action="store_true", help="include the spring t0 arms")
    p.add_argument("--round", default="", help="a longitudinal round id, e.g. 2026-W39")
    p.add_argument(
        "--lengths",
        default=",".join(str(n) for n in DEFAULT_LENGTHS),
        help="candidate window lengths in packets",
    )
    p.add_argument("--out", default=os.path.join(FIGURES_DIR, "sequence_budget.json"))
    args = p.parse_args()

    if not args.spring and not args.round:
        args.spring = True
    lengths = [int(n) for n in args.lengths.split(",") if n.strip()]

    arms = []
    if args.spring:
        for name in ("baseline", "obfs4", "other"):
            got = describe(f"spring/{name}", os.path.join(SPRING_DIR, f"{name}_features"), lengths)
            if got:
                arms.append(got)
    if args.round:
        round_dir = os.path.join(LONGITUDINAL_DIR, args.round)
        if not os.path.isdir(round_dir):
            sys.exit(f"No such round: {round_dir}")
        for entry in sorted(os.listdir(round_dir)):
            if entry.endswith("_features") and not entry.endswith("_rejected_features"):
                arm = entry[: -len("_features")]
                got = describe(f"{args.round}/{arm}", os.path.join(round_dir, entry), lengths)
                if got:
                    arms.append(got)

    if not arms:
        sys.exit("No extracted features found to analyse.")

    print_report(arms, lengths)

    payload = {"lengths": lengths, "arms": arms}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
