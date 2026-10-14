"""Trace selection and base-rate precision in evaluate_budget.py.

The budget and learning-curve results rest on "the n earliest training traces
per site": a test trace slipping in, or later traces taken instead of earlier
ones, would quietly inflate them. Precision at a base rate is plain arithmetic
on the sweep, checked here against a hand computation.

    python tests/test_budget.py
"""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import evaluate_budget as eb  # noqa: E402
from evaluate_transport import Split  # noqa: E402


def check(name, ok, detail=""):
    print(f"[{'ok  ' if ok else 'FAIL'}] {name:44s} {'' if ok else detail}")
    return ok


def main():
    results = []
    results.append(check("parse_counts", eb.parse_counts("0, 1,3,all") == [0, 1, 3, None]))

    # Two sites, ten traces each, timestamps out of order on purpose.
    rng = np.random.default_rng(0)
    y = np.array(["a"] * 10 + ["b"] * 10)
    minutes = np.concatenate([rng.permutation(10), rng.permutation(10) + 100])
    t = np.datetime64("2026-10-01T00:00") + minutes.astype("timedelta64[m]")
    X = np.zeros((20, 1, 4), dtype=np.float32)
    s = Split(X, y, t, test_size=0.2)

    picked = eb.earliest_per_site(s, 3)
    results.append(check("3 per site", sorted(np.unique(y[picked], return_counts=True)[1]) == [3, 3]))
    results.append(check("no test trace among them", not set(picked) & set(s.test)))
    for site in "ab":
        train_t = np.sort(t[s.train][y[s.train] == site])
        got = np.sort(t[picked][y[picked] == site])
        results.append(check(f"earliest training traces of {site}", (got == train_t[:3]).all(),
                             f"{got} vs {train_t[:3]}"))
    results.append(check("n=None is the whole training split",
                         set(eb.earliest_per_site(s, None)) == set(s.train)))
    results.append(check("n beyond the split takes what there is",
                         len(eb.earliest_per_site(s, 50)) == len(s.train)))

    # Precision: b*TPR / (b*(TPR+WPR) + (1-b)*FPR).
    sweep = [{"threshold": 0.0, "tpr": 0.9, "wpr": 0.05, "fpr": 0.3},
             {"threshold": 0.5, "tpr": 0.6, "wpr": 0.01, "fpr": 0.02},
             {"threshold": 0.9, "tpr": 0.2, "wpr": 0.0, "fpr": 0.0}]
    fake = {"scenarios": {"open_world_x": {"runs": [{"seed": 1, "sweep": sweep}]}}}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fake.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fake, f)
        eb.precision_table(path, rates=(0.01,))
        with open(os.path.join(d, "fake_precision.json"), encoding="utf-8") as f:
            table = json.load(f)["scenarios"]["open_world_x"]["base_rates"]["0.01"]
    zero = 0.01 * 0.9 / (0.01 * 0.95 + 0.99 * 0.3)
    best = 0.01 * 0.6 / (0.01 * 0.61 + 0.99 * 0.02)  # threshold 0.9 has TPR < 50%: not usable
    results.append(check("precision at threshold 0", abs(table["precision_at_zero_mean"] - zero) < 1e-12,
                         f"{table['precision_at_zero_mean']} vs {zero}"))
    results.append(check("best precision with TPR >= 50%", abs(table["best_precision_mean"] - best) < 1e-12,
                         f"{table['best_precision_mean']} vs {best}"))
    results.append(check("best threshold", table["best_thresholds"] == [0.5], str(table["best_thresholds"])))

    # A set_size sweep carries fpr_pool (sites never trained on); precision uses it.
    sweep = [{"threshold": 0.0, "tpr": 0.9, "fpr": 0.05, "fpr_pool": 0.2, "fpr_demoted": 0.01}]
    fake = {"scenarios": {"set_size_x": {"runs": [{"seed": 1, "sweep": sweep}]}}}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "fake.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fake, f)
        eb.precision_table(path, rates=(0.01,))
        with open(os.path.join(d, "fake_precision.json"), encoding="utf-8") as f:
            table = json.load(f)["scenarios"]["set_size_x"]["base_rates"]["0.01"]
    pool = 0.01 * 0.9 / (0.01 * 0.9 + 0.99 * 0.2)
    results.append(check("precision from the held-out pool's FPR",
                         abs(table["precision_at_zero_mean"] - pool) < 1e-12,
                         f"{table['precision_at_zero_mean']} vs {pool}"))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
