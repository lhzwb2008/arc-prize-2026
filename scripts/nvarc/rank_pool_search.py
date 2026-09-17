#!/usr/bin/env python3
"""CPU search over pooled ranking rules. No GPU.

Oracle = gold grid appears in any decoded candidate (ignore rank).
Each rule only reorders unique candidate grids; it cannot invent gold.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from arc_decoder import (
    ArcDecoder,
    hashable,
    merge_keep_primary,
    merge_pass_pair,
    score_full_probmul_3,
    score_kgmon,
    score_sum,
)
from arc_loader import ArcDataset


def _mean_aug(g):
    a = np.asarray(g["score_aug"], dtype=float)
    return float(a.mean()) if a.size else 0.0


def _beam(g):
    return float(g["beam_score"])


def make_score(getter):
    def fn(guesses):
        return score_sum(guesses, getter)
    fn.__name__ = getattr(getter, "__name__", "rule")
    return fn


def getter_votes(gs):
    return float(len(gs))


def getter_neg_mean_beam(gs):
    return -float(np.mean([_beam(g) for g in gs]))


def getter_neg_best_beam(gs):
    return -float(min(_beam(g) for g in gs))


def getter_neg_mean_aug(gs):
    return -float(np.mean([_mean_aug(g) for g in gs]))


def getter_neg_best_aug(gs):
    return -float(min(_mean_aug(g) for g in gs))


def getter_votes_then_aug(gs):
    return (len(gs), -float(np.mean([_mean_aug(g) for g in gs])))


def getter_votes_then_beam(gs):
    return (len(gs), -float(np.mean([_beam(g) for g in gs])))


def getter_sqrt_votes_minus_aug(gs):
    return float(np.sqrt(len(gs))) - float(np.mean([_mean_aug(g) for g in gs]))


def getter_votes2_minus_aug(gs):
    return 2.0 * len(gs) - float(np.mean([_mean_aug(g) for g in gs]))


def getter_sum_neg_beam(gs):
    return -float(np.sum([_beam(g) for g in gs]))


def getter_logvotes_minus_aug(gs):
    return float(np.log1p(len(gs))) - float(np.mean([_mean_aug(g) for g in gs]))


def getter_best_sample(gs):
    # one strongest sample: low beam, low aug
    return max((-_beam(g), -_mean_aug(g)) for g in gs)


def getter_mean_quality(gs):
    return float(np.mean([-_beam(g) - _mean_aug(g) for g in gs]))


def getter_vote_x_quality(gs):
    q = float(np.mean([-_beam(g) - _mean_aug(g) for g in gs]))
    return len(gs) * q


RULES = {
    "kgmon": score_kgmon,
    "probmul3": score_full_probmul_3,
    "votes": make_score(getter_votes),
    "neg_mean_beam": make_score(getter_neg_mean_beam),
    "neg_best_beam": make_score(getter_neg_best_beam),
    "neg_mean_aug": make_score(getter_neg_mean_aug),
    "neg_best_aug": make_score(getter_neg_best_aug),
    "votes_then_aug": make_score(getter_votes_then_aug),
    "votes_then_beam": make_score(getter_votes_then_beam),
    "sqrt_votes_minus_aug": make_score(getter_sqrt_votes_minus_aug),
    "votes2_minus_aug": make_score(getter_votes2_minus_aug),
    "sum_neg_beam": make_score(getter_sum_neg_beam),
    "logvotes_minus_aug": make_score(getter_logvotes_minus_aug),
    "best_sample": make_score(getter_best_sample),
    "mean_quality": make_score(getter_mean_quality),
    "vote_x_quality": make_score(getter_vote_x_quality),
}


def task_score(replies, selected, n_guesses=2):
    per = defaultdict(list)
    hit = present = 0
    n_out = 0
    for bk, golds in replies.items():
        n_out += 1
        gold = golds[0]
        guesses = list(selected.get(bk) or [])[:n_guesses]
        ok = any(np.array_equal(g, gold) for g in guesses)
        per[bk.split("_")[0]].append(bool(ok))
        if ok:
            hit += 1
    score = sum(sum(v) / len(v) for v in per.values())
    return score, hit, n_out


def oracle_score(replies, decoded):
    per = defaultdict(list)
    in_pool = 0
    n_out = 0
    for bk, golds in replies.items():
        n_out += 1
        hg = hashable(golds[0])
        present = False
        for s in (decoded.get(bk) or {}).values():
            try:
                if hashable(s["solution"]) == hg:
                    present = True
                    break
            except Exception:
                pass
        per[bk.split("_")[0]].append(present)
        if present:
            in_pool += 1
    return sum(sum(v) / len(v) for v in per.values()), in_pool, n_out


def pass_pair_top1(sel_a, sel_b):
    """Attempt 1 = A top-1, attempt 2 = B top-1 (or A's #2 if same)."""
    return merge_pass_pair(sel_a, sel_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/opt/data/kaggle/arc-agi_evaluation_challenges.json")
    ap.add_argument("--solutions", default="/opt/data/kaggle/arc-agi_evaluation_solutions.json")
    ap.add_argument("--outputs", action="append", required=True)
    ap.add_argument("--n-tasks", type=int, default=120)
    args = ap.parse_args()

    data = ArcDataset.from_file(args.data)
    data.load_replies(args.solutions)
    dm = data.split_multi_replies()
    replies = dm.replies

    dec = ArcDecoder(dm, n_guesses=2)
    for i, d in enumerate(args.outputs):
        dec.load_decoded_results(d, run_name="" if i == 0 else f".p{i}")

    oracle, in_pool, n_out = oracle_score(replies, dec.decoded_results)
    n = args.n_tasks
    print(f"outputs={n_out} gold_in_pool={in_pool} ({100*in_pool/n_out:.1f}%)")
    print(f"oracle {oracle:.3f}/{n} = {100*oracle/n:.2f}%")
    print()

    rows = []
    for name, fn in RULES.items():
        sel = dec.run_selection_algo(fn)
        sc, hit, _ = task_score(replies, sel)
        rows.append((sc, hit, name, "pool"))
        print(f"{sc:7.3f}  top2_out {hit:3d}/{n_out}  {name}")

    # keep-primary: first dir's kgmon top-1 forced into attempts
    if len(args.outputs) >= 2:
        dec_a = ArcDecoder(dm, n_guesses=2)
        dec_a.load_decoded_results(args.outputs[0])
        sel_a = dec_a.run_selection_algo(score_kgmon)
        for name, fn in (("kgmon", score_kgmon), ("votes_then_aug", RULES["votes_then_aug"]),
                         ("probmul3", score_full_probmul_3)):
            sel = merge_keep_primary(sel_a, dec.run_selection_algo(fn))
            sc, hit, _ = task_score(replies, sel)
            rows.append((sc, hit, f"keepA+{name}", "keep"))
            print(f"{sc:7.3f}  top2_out {hit:3d}/{n_out}  keepA+{name}")

        if len(args.outputs) == 2:
            dec_b = ArcDecoder(dm, n_guesses=2)
            dec_b.load_decoded_results(args.outputs[1])
            sel_b = dec_b.run_selection_algo(score_kgmon)
            sel = pass_pair_top1(sel_a, sel_b)
            sc, hit, _ = task_score(replies, sel)
            rows.append((sc, hit, "A_top1+B_top1", "pair"))
            print(f"{sc:7.3f}  top2_out {hit:3d}/{n_out}  A_top1+B_top1")

    rows.sort(reverse=True)
    print("\nbest:")
    for sc, hit, name, kind in rows[:8]:
        print(f"  {sc:7.3f} ({100*sc/n:.2f}%)  {name} [{kind}]")
    print(f"oracle ceiling {oracle:.3f} ({100*oracle/n:.2f}%)")


if __name__ == "__main__":
    main()
