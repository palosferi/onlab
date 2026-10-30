"""Trace statistics behind the traffic-characteristics analysis.

The analysis explains DF's cross-transport drop by what each transport does to
the cells DF reads, so its counts must be DF's: bytes rounded to 514-byte
cells, ACKs gone, bursts counted only in the window DF sees. The site filter
must keep the same sites evaluate_transport.py does, or the two disagree on
what "the 50 sites" are.

    python tests/test_transport_traffic.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from transport_traffic import keep_sites, trace_stats  # noqa: E402


def trace(*sizes):
    return pd.DataFrame({"time_offset": [0.1 * i for i in range(len(sizes))],
                         "direction_size": list(sizes)})


def check_cells_and_bursts():
    # out 1 cell, ACK (0 cells), in 2 cells, in 1 cell: runs +1 | -1 -1 -1
    s = trace(514, 66, -1028, -514)
    s = trace_stats(s)
    return (s["cells_out"], s["cells_in"], s["bursts"], s["in_burst_med"], s["out_burst_med"],
            s["in_burst_max"]) == (1, 3, 2, 3.0, 1.0, 3)


def check_out_share_in_window():
    # 4 incoming cells, then 4 outgoing: a 4-cell window sees only incoming
    s = trace_stats(trace(-2056, 2056), length=4)
    return s["out_share"] == 0.0 and s["bursts"] == 1 and s["cells_out"] == 4


def check_ack_only_is_none():
    return trace_stats(trace(66, -66, 54)) is None and trace_stats(trace()) is None


def check_size_histogram():
    s = trace_stats(trace(66, 200, 514, -1028, -1514, -4000))
    return s["hist_out"] == [1, 1, 1, 0, 0, 0] and s["hist_in"] == [0, 0, 0, 1, 1, 1]


def check_keep_sites_needs_every_arm():
    rows = ([{"site": "a", "arm": arm} for arm in ("baseline", "obfs4", "snowflake") for _ in range(3)]
            + [{"site": "b", "arm": arm} for arm in ("baseline", "obfs4") for _ in range(3)]
            + [{"site": "c", "arm": arm} for arm in ("baseline", "obfs4", "snowflake")
               for _ in range(3 if arm != "obfs4" else 2)])
    kept, dropped = keep_sites(pd.DataFrame(rows), 3)
    return set(kept["site"]) == {"a"} and dropped == ["b", "c"]


CASES = [
    ("cells and bursts are DF's", check_cells_and_bursts),
    ("outgoing share is taken in DF's window", check_out_share_in_window),
    ("ACK-only and empty traces are skipped", check_ack_only_is_none),
    ("packet sizes land in their cell bins", check_size_histogram),
    ("a site needs min visits on every arm", check_keep_sites_needs_every_arm),
]


def main():
    failures = 0
    for name, check in CASES:
        try:
            ok = bool(check())
        except Exception as exc:
            ok = False
            name = f"{name} ({exc})"
        failures += not ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {name}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
