#!/usr/bin/env python3
"""CPU tests for the B priority queue and checkpoint gating (no pickles, no GPU)."""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from b_priority_queue import priority_order, task_shares, top1_share  # noqa: E402


def _s(grid, beam=0.1):
    return {"solution": np.array(grid), "beam_score": beam, "score_aug": [beam]}


class PriorityQueueTests(unittest.TestCase):
    def test_top1_share(self):
        samples = {"a": _s([[1]]), "b": _s([[1]]), "c": _s([[2]]), "d": _s([[3]])}
        self.assertAlmostEqual(top1_share(samples), 0.5)
        self.assertEqual(top1_share({}), 0.0)

    def test_task_shares_min_over_outputs_and_missing(self):
        decoded = {
            "t1_0": {"a": _s([[1]]), "b": _s([[1]])},          # share 1.0
            "t1_1": {"a": _s([[1]]), "b": _s([[2]])},          # share 0.5
            "t2_0": {"a": _s([[1]]), "b": _s([[2]]), "c": _s([[3]])},  # share 1/3
        }
        sh = task_shares(decoded, ["t1", "t2", "t3"])
        self.assertAlmostEqual(sh["t1"], 0.5)
        self.assertAlmostEqual(sh["t2"], 1 / 3)
        self.assertEqual(sh["t3"], 0.0)

    def test_priority_uncertain_cheap_first_confident_last(self):
        tasks = ["confident_cheap", "uncertain_cheap", "uncertain_dear", "missing_dear"]
        work = {"confident_cheap": 10.0, "uncertain_cheap": 10.0, "uncertain_dear": 100.0, "missing_dear": 100.0}
        share = {"confident_cheap": 1.0, "uncertain_cheap": 0.2, "uncertain_dear": 0.2, "missing_dear": 0.0}
        order = priority_order(tasks, work, share)
        self.assertEqual(order[0], "uncertain_cheap")
        self.assertEqual(order[-1], "confident_cheap")
        self.assertLess(order.index("missing_dear"), order.index("uncertain_dear"))

    def test_skip_confident_drops_agreeing_tasks(self):
        tasks = ["a", "b", "c"]
        work = {t: 1.0 for t in tasks}
        share = {"a": 0.9, "b": 0.5, "c": 0.1}
        self.assertEqual(priority_order(tasks, work, share, skip_confident=0.5), ["c"])
        self.assertEqual(set(priority_order(tasks, work, share)), set(tasks))


class CheckpointGateTests(unittest.TestCase):
    def test_checkpoint_due_only_on_change_and_gap(self):
        try:
            import starter  # noqa: F401
        except Exception as e:  # torch missing on CPU test box
            self.skipTest(f"starter import unavailable: {e}")
        from starter import _CKPT_STATE, checkpoint_due
        with tempfile.TemporaryDirectory() as d:
            _CKPT_STATE.update(fp=None, t=0.0)
            now = time.time()
            self.assertTrue(checkpoint_due([d], now, 300, force=True))
            self.assertFalse(checkpoint_due([d], now + 10, 300))       # nothing changed
            open(os.path.join(d, "x_0.pkl"), "w").close()
            self.assertFalse(checkpoint_due([d], now + 10, 300))       # changed, gap too small
            self.assertTrue(checkpoint_due([d], now + 400, 300))       # changed, gap ok
            self.assertFalse(checkpoint_due([d], now + 800, 300))      # unchanged again


if __name__ == "__main__":
    unittest.main()
