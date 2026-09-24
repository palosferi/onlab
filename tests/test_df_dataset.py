"""Sequence loading and the chronological split.

The split is the part worth testing: a per-class time cut must never hand a
class zero training traces, must keep train strictly earlier than test within
a class, and must refuse to run at all when a timestamp is missing -- silently
treating an unparsed name as epoch zero would put it in train and look fine.

    python tests/test_df_dataset.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from df_dataset import _to_sequence, chronological_split, parse_trace_name  # noqa: E402


def ts(*values):
    return np.array([pd.Timestamp(v) for v in values], dtype="datetime64[ns]")


def check_name_parsing():
    site, stamp = parse_trace_name("/x/aliexpress_20260326_205400.csv")
    return site == "aliexpress" and stamp == pd.Timestamp("2026-03-26 20:54:00")


def check_underscored_site():
    # Site names containing underscores must survive: rsplit on the last two
    # fields, not split on the first.
    site, stamp = parse_trace_name("/x/index_hu_news_20260327_010537.csv")
    return site == "index_hu_news" and stamp == pd.Timestamp("2026-03-27 01:05:37")


def check_unparsable_name():
    site, stamp = parse_trace_name("/x/weird-name.csv")
    return site == "weird-name" and stamp is None


def check_padding():
    df = pd.DataFrame(
        {"direction_size": [536, -536, 536], "time_offset": [0.0, 0.1, 0.2],
         "inter_arrival_time": [0.0, 0.1, 0.1]}
    )
    seq = _to_sequence(df, "direction", 8)
    return len(seq) == 8 and list(seq[:3]) == [1, -1, 1] and seq[3:].sum() == 0


def check_truncation():
    n = 20
    df = pd.DataFrame(
        {"direction_size": [536] * n, "time_offset": np.arange(n) * 0.1,
         "inter_arrival_time": [0.1] * n}
    )
    return len(_to_sequence(df, "direction", 5)) == 5


def check_tiktok_signs_timing():
    df = pd.DataFrame(
        {"direction_size": [536, -536], "time_offset": [0.5, 1.5],
         "inter_arrival_time": [0.0, 1.0]}
    )
    seq = _to_sequence(df, "tiktok", 2)
    return np.allclose(seq, [0.5, -1.5])


def check_per_class_keeps_every_class():
    y = np.array(["a"] * 5 + ["b"] * 5)
    t = ts(*[f"2026-03-{d:02d}" for d in range(1, 6)] * 2)
    tr, te = chronological_split(y, t, test_size=0.2, per_class=True)
    return set(y[tr]) == {"a", "b"} and set(y[te]) == {"a", "b"}


def check_per_class_is_ordered():
    y = np.array(["a"] * 4 + ["b"] * 4)
    t = ts("2026-03-04", "2026-03-01", "2026-03-03", "2026-03-02",
           "2026-03-08", "2026-03-05", "2026-03-07", "2026-03-06")
    tr, te = chronological_split(y, t, test_size=0.25, per_class=True)
    for label in ("a", "b"):
        latest_train = t[tr][y[tr] == label].max()
        earliest_test = t[te][y[te] == label].min()
        if not latest_train < earliest_test:
            return False
    return True


def check_singleton_class_stays_in_train():
    y = np.array(["a", "a", "a", "a", "solo"])
    t = ts("2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05")
    tr, te = chronological_split(y, t, test_size=0.5, per_class=True)
    return "solo" in set(y[tr]) and "solo" not in set(y[te])


def check_global_split_is_one_cut():
    y = np.array(["a", "b", "a", "b"])
    t = ts("2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04")
    tr, te = chronological_split(y, t, test_size=0.5, per_class=False)
    return t[tr].max() < t[te].min()


def check_missing_timestamp_raises():
    y = np.array(["a", "a"])
    # load_sequences yields None for an unparsable name; numpy turns that into NaT.
    t = np.array([pd.Timestamp("2026-03-01"), None], dtype="datetime64[ns]")
    try:
        chronological_split(y, t)
    except ValueError:
        return True
    return False


CASES = [
    ("filename -> site and timestamp", check_name_parsing),
    ("site name containing underscores", check_underscored_site),
    ("unparsable name yields no timestamp", check_unparsable_name),
    ("short trace is zero padded", check_padding),
    ("long trace is truncated", check_truncation),
    ("tiktok mode signs the arrival time", check_tiktok_signs_timing),
    ("per-class split keeps every class", check_per_class_keeps_every_class),
    ("per-class split is time ordered", check_per_class_is_ordered),
    ("single-trace class stays in train", check_singleton_class_stays_in_train),
    ("global split is a single cut", check_global_split_is_one_cut),
    ("missing timestamp is rejected", check_missing_timestamp_raises),
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
