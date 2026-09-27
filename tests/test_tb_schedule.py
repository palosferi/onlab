"""The Tor Browser collection schedule.

Resuming a multi-day collection depends on the schedule being a pure function
of its inputs, and the interleaving claims in the thesis (no arm or sample kind
bunched in time) depend on how the batches are cut. Both are checked here.

    python tests/test_tb_schedule.py
"""

import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "collection"))

from collect_tb import (  # noqa: E402
    MANIFEST_FIELDS,
    build_schedule,
    capture_key,
    needs_capture,
    read_progress,
)

MON = {f"m{i}": f"https://m{i}.example" for i in range(5)}
UNMON = {f"u{i}.example": f"https://u{i}.example" for i in range(23)}
BG = {f"b{i}.example": f"https://b{i}.example" for i in range(4)}
ARMS = ["baseline", "obfs4", "snowflake"]


def schedule(**kw):
    args = dict(monitored=MON, unmonitored=UNMON, background=BG, arms=ARMS,
                repeats=8, bg_repeats=4, seed="t")
    args.update(kw)
    return build_schedule(**args)


def slots(batches):
    return [s for b in batches for s in b]


def check_deterministic():
    return schedule() == schedule()


def check_seed_changes_order():
    return schedule() != schedule(seed="other")


def check_batch_k_holds_repeat_k():
    for k, batch in enumerate(schedule()):
        reps = {s["repeat"] for s in batch if s["kind"] == "monitored"}
        if reps != {k + 1}:
            return False
    return True


def check_monitored_counts():
    got = {}
    for s in slots(schedule()):
        if s["kind"] == "monitored":
            for a in s["arms"]:
                got[(s["site"], a)] = got.get((s["site"], a), 0) + 1
    return len(got) == len(MON) * len(ARMS) and set(got.values()) == {8}


def check_unmonitored_once_per_arm_and_spread():
    batches = schedule()
    seen = [s for s in slots(batches) if s["kind"] == "unmonitored"]
    per_batch = [sum(s["kind"] == "unmonitored" for s in b) for b in batches]
    return (sorted(s["site"] for s in seen) == sorted(UNMON)
            and all(sorted(s["arms"]) == sorted(ARMS) for s in seen)
            and max(per_batch) - min(per_batch) <= 1)


def check_background_is_baseline_only():
    bg = [s for s in slots(schedule()) if s["kind"] == "background"]
    return (len(bg) == 4 * len(MON)
            and all(s["arms"] == ["baseline"] for s in bg)
            and all(s["bg_site"] in BG for s in bg)
            and all(0.5 <= s["bg_lead_s"] <= 3.0 for s in bg))


def check_background_spread_over_time():
    batches = schedule()
    used = [k for k, b in enumerate(batches) if any(s["kind"] == "background" for s in b)]
    return used == [0, 2, 4, 6]


def check_arm_order_varies():
    orders = {tuple(s["arms"]) for s in slots(schedule()) if s["kind"] == "monitored"}
    return len(orders) > 1


def check_no_background_without_baseline():
    bg = [s for s in slots(schedule(arms=["obfs4"])) if s["kind"] == "background"]
    return bg == []


def check_resume_from_manifest():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "manifest.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            for kind, arm, site, rep, status in [
                ("monitored", "obfs4", "m1", 1, "ok"),
                ("monitored", "baseline", "m2", 1, "load_timeout"),
                ("monitored", "baseline", "m3", 1, "blocked"),
                ("monitored", "baseline", "m3", 1, "blocked"),
            ]:
                row = {f: "" for f in MANIFEST_FIELDS}
                row.update(kind=kind, arm=arm, site=site, repeat=rep, status=status)
                w.writerow(row)
        done, attempts = read_progress(path)
    return (not needs_capture(capture_key("monitored", "obfs4", "m1", "1"), done, attempts, 2)
            and needs_capture(capture_key("monitored", "baseline", "m2", 1), done, attempts, 2)
            and not needs_capture(capture_key("monitored", "baseline", "m3", 1), done, attempts, 2)
            and needs_capture(capture_key("monitored", "baseline", "m4", 1), done, attempts, 2))


CASES = [
    ("same inputs give the same schedule", check_deterministic),
    ("a different seed reorders it", check_seed_changes_order),
    ("batch k holds the k-th monitored visit", check_batch_k_holds_repeat_k),
    ("every monitored site x arm visited `repeats` times", check_monitored_counts),
    ("unmonitored: once per arm, spread evenly", check_unmonitored_once_per_arm_and_spread),
    ("background visits are baseline only", check_background_is_baseline_only),
    ("background visits spread over batches", check_background_spread_over_time),
    ("arm order within a slot varies", check_arm_order_varies),
    ("no background visits without a baseline arm", check_no_background_without_baseline),
    ("resume: ok done, failures retried up to the cap", check_resume_from_manifest),
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
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
