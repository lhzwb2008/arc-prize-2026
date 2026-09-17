#!/usr/bin/env python3
"""CPU: leftover-B mixed mean_quality displaces gold; pass-pair does not."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from arc_decoder import (  # noqa: E402
    ArcDecoder,
    merge_keep_primary,
    merge_pass_pair,
    score_mean_quality,
)


GOLD = np.array([[1, 2], [3, 4]], dtype=int)
WRONG = np.array([[9, 9], [9, 9]], dtype=int)
ALT = np.array([[8, 8], [8, 8]], dtype=int)


def sample(grid, beam, aug):
    return {
        "solution": np.array(grid, dtype=int),
        "beam_score": float(beam),
        "score_aug": [float(aug)],
    }


def rank(decoded):
    dec = ArcDecoder(None, n_guesses=2)
    dec.decoded_results = decoded
    return dec.run_selection_algo(score_mean_quality)


def is_gold(g):
    return np.array_equal(g, GOLD)


def top2_has_gold(sel, bk="t_0"):
    return any(is_gold(g) for g in (sel.get(bk) or [])[:2])


class PoolMergeTests(unittest.TestCase):
    def test_a_only_keeps_top2(self):
        sel_a = rank({
            "t_0": {
                "a.out0": sample(GOLD, 0.4, 0.4),
                "a.out1": sample(ALT, 0.9, 0.9),
            }
        })
        out = merge_pass_pair(sel_a, {})
        self.assertTrue(is_gold(out["t_0"][0]))
        self.assertTrue(np.array_equal(out["t_0"][1], ALT))

    def test_mixed_quality_drops_gold_pass_pair_keeps_it(self):
        # Lower beam/aug => higher mean_quality. A ranks gold first.
        # B injects two even better unique wrongs. Mixed top-2 becomes
        # those wrongs; gold drops off. Pass-pair still keeps A's gold.
        decoded_a = {
            "t_0": {
                "a.geo0.out0": sample(GOLD, 0.40, 0.40),
                "a.geo1.out0": sample(ALT, 0.90, 0.90),
            }
        }
        decoded_b = {
            "t_0": {
                "b.geo0.out0": sample(WRONG, 0.05, 0.05),
                "b.geo1.out0": sample(np.array([[7, 7], [7, 7]]), 0.08, 0.08),
            }
        }
        sel_a = rank(decoded_a)
        sel_b = rank(decoded_b)
        self.assertTrue(is_gold(sel_a["t_0"][0]), "A alone must put gold first")

        mixed = {
            "t_0": {**decoded_a["t_0"], **decoded_b["t_0"]},
        }
        sel_mixed = rank(mixed)
        self.assertFalse(
            top2_has_gold(sel_mixed),
            f"mixed mean_quality should drop gold, got {[x.tolist() for x in sel_mixed['t_0'][:2]]}",
        )

        sel_pair = merge_pass_pair(sel_a, sel_b)
        self.assertTrue(is_gold(sel_pair["t_0"][0]))
        self.assertTrue(np.array_equal(sel_pair["t_0"][1], WRONG))
        self.assertTrue(top2_has_gold(sel_pair))

        sel_keep = merge_keep_primary(sel_a, sel_mixed)
        self.assertTrue(top2_has_gold(sel_keep), "keep-primary must recover A top-1 gold")
        # keep-primary still puts mixed #1 (wrong) as attempt_1; pair keeps gold as #1.

    def test_leftover_b_only_touches_some_tasks(self):
        decoded_a = {
            "cheap_0": {
                "a.out0": sample(GOLD, 0.40, 0.40),
                "a.out1": sample(ALT, 0.90, 0.90),
            },
            "hard_0": {
                "a.out0": sample(ALT, 0.30, 0.30),
                "a.out1": sample(GOLD, 0.90, 0.90),
            },
        }
        # Leftover B only finished the cheap task, with two high-q wrongs.
        decoded_b = {
            "cheap_0": {
                "b.out0": sample(WRONG, 0.05, 0.05),
                "b.out1": sample(np.array([[7, 7], [7, 7]]), 0.08, 0.08),
            }
        }
        sel_a = rank(decoded_a)
        sel_b = rank(decoded_b)
        mixed = {
            bk: {**(decoded_a.get(bk) or {}), **(decoded_b.get(bk) or {})}
            for bk in set(decoded_a) | set(decoded_b)
        }
        sel_mixed = rank(mixed)
        self.assertFalse(top2_has_gold(sel_mixed, "cheap_0"))
        # Hard task never got B, mixed == A (gold is attempt_2).
        self.assertTrue(top2_has_gold(sel_mixed, "hard_0"))

        sel_pair = merge_pass_pair(sel_a, sel_b)
        self.assertTrue(is_gold(sel_pair["cheap_0"][0]))
        self.assertTrue(top2_has_gold(sel_pair, "hard_0"))
        self.assertTrue(np.array_equal(sel_pair["hard_0"][0], ALT))
        self.assertTrue(is_gold(sel_pair["hard_0"][1]))

    def test_b_recovers_when_a_is_wrong(self):
        sel_a = rank({
            "t_0": {
                "a.out0": sample(WRONG, 0.2, 0.2),
                "a.out1": sample(ALT, 0.3, 0.3),
            }
        })
        sel_b = rank({
            "t_0": {
                "b.out0": sample(GOLD, 0.4, 0.4),
            }
        })
        out = merge_pass_pair(sel_a, sel_b)
        self.assertTrue(np.array_equal(out["t_0"][0], WRONG))
        self.assertTrue(is_gold(out["t_0"][1]))

    def test_same_top1_falls_back_to_a_second(self):
        sel_a = rank({
            "t_0": {
                "a.out0": sample(GOLD, 0.2, 0.2),
                "a.out1": sample(ALT, 0.4, 0.4),
            }
        })
        sel_b = rank({
            "t_0": {
                "b.out0": sample(GOLD, 0.3, 0.3),
            }
        })
        out = merge_pass_pair(sel_a, sel_b)
        self.assertTrue(is_gold(out["t_0"][0]))
        self.assertTrue(np.array_equal(out["t_0"][1], ALT))


if __name__ == "__main__":
    unittest.main()
