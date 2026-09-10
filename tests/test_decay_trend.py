"""Decay trend statistics.

A curve drawn through three or four noisy points invites over-reading, so the
thesis needs the slope with a confidence interval and a p-value rather than an
eyeballed downward line. These cases check the fit reports significance only
when the data supports it.

    python tests/test_decay_trend.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from drift_eval import decay_trend  # noqa: E402


def rows(pairs):
    return [{"weeks_since_train": w, "accuracy": a} for w, a in pairs]


CASES = [
    ("too few points to fit",
     rows([(1, 70), (2, 65)]),
     lambda t: "note" in t),
    ("clean linear decay is significant and negative",
     rows([(1, 78), (2, 74), (3, 70), (4, 66), (5, 62), (6, 58)]),
     lambda t: t["significant_at_0.05"] and t["slope_accuracy_points_per_week"] < 0
               and t["r_squared"] > 0.99),
    ("flat noisy series is not significant",
     rows([(1, 70), (2, 68), (3, 71), (4, 69), (5, 70), (6, 69)]),
     lambda t: not t["significant_at_0.05"]),
    ("confidence interval brackets the slope",
     rows([(1, 80), (2, 76), (3, 73), (4, 68), (5, 65), (6, 61)]),
     lambda t: t["slope_95ci"][0] <= t["slope_accuracy_points_per_week"] <= t["slope_95ci"][1]),
]


def main():
    failures = 0
    for name, data, check in CASES:
        t = decay_trend(data)
        try:
            ok = bool(check(t))
        except Exception:
            ok = False
        failures += not ok
        detail = t.get("note") or (
            f"slope={t.get('slope_accuracy_points_per_week')} "
            f"ci={t.get('slope_95ci')} p={t.get('p_value')} r2={t.get('r_squared')}"
        )
        print(f"[{'ok  ' if ok else 'FAIL'}] {name:44s} {detail}")
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
