#!/usr/bin/env python3
"""CPU tests for expensive leftover-B helpers (no pickles, no GPU)."""
from __future__ import annotations

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from simulate_expensive_b import (  # noqa: E402
    bootstrap_delta_pct,
    cheap_order,
    estimated_work,
    expensive_order,
    hours_for_subset,
    pick_best_within_budget,
    prefix_until_hours,
    restrict_decoded,
    sequential_durations,
)


class WorkOrderTests(unittest.TestCase):
    def test_bigger_grid_is_more_work(self):
        small = {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[1]]}]}
        big = {
            "train": [{"input": [[1] * 10] * 10, "output": [[1] * 10] * 10}],
            "test": [{"input": [[1] * 10] * 10}],
        }
        self.assertGreater(estimated_work(big), estimated_work(small))

    def test_cheap_order_asc(self):
        work = {"cheap": 1.0, "mid": 5.0, "hard": 9.0}
        self.assertEqual(cheap_order(["cheap", "mid", "hard"], work), ["cheap", "mid", "hard"])

    def test_bootstrap_delta_sign(self):
        base = {f"t{i}": 0.0 for i in range(20)}
        new = {f"t{i}": 1.0 if i < 2 else 0.0 for i in range(20)}
        boot = bootstrap_delta_pct(base, new, iters=200, seed=0)
        self.assertGreater(boot["mean_pct"], 0.0)
        self.assertGreater(boot["p_gt_0"], 0.5)

    def test_prefix_until_hours_skips_tail(self):
        order = ["h", "m", "c"]
        dur = {"h": 3600, "m": 3600, "c": 3600}
        keep = prefix_until_hours(order, dur, 2.0)
        self.assertEqual(keep, ["h", "m"])
        self.assertAlmostEqual(hours_for_subset(keep, dur), 2.0)
        self.assertEqual(prefix_until_hours(order, dur, 0.0), [])

    def test_sequential_durations(self):
        done = {"a": 10.0, "b": 40.0, "c": 100.0}
        dur = sequential_durations(done, start_mtime=0.0)
        self.assertEqual(dur["a"], 10.0)
        self.assertEqual(dur["b"], 30.0)
        self.assertEqual(dur["c"], 60.0)

    def test_restrict_decoded(self):
        decoded = {"t1_0": {"x": 1}, "t2_0": {"y": 2}, "t1_1": {"z": 3}}
        out = restrict_decoded(decoded, {"t1"})
        self.assertEqual(set(out), {"t1_0", "t1_1"})

    def test_pick_best_within_budget(self):
        rows = [
            {"a_h": 8.0, "b_h": 7.0, "mixed_pct": 33.33, "total_h": 15.0},
            {"a_h": 8.0, "b_h": 5.5, "mixed_pct": 31.67, "total_h": 13.5},
            {"a_h": 8.0, "b_h": 3.5, "mixed_pct": 30.42, "total_h": 11.5},
            {"a_h": 8.0, "b_h": 2.0, "mixed_pct": 29.50, "total_h": 10.0},
        ]
        rec = pick_best_within_budget(rows, 13.58)
        self.assertEqual(rec["b_h"], 5.5)
        rec12 = pick_best_within_budget(rows, 11.67)
        self.assertEqual(rec12["b_h"], 3.5)


if __name__ == "__main__":
    unittest.main()
