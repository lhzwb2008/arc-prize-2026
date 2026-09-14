#!/usr/bin/env python3
"""CPU experiment: 6×6 first pass + 5×5 second pass vs 6×6 A+B.

Uses existing pickles. Tags pass-1 samples as primary and re-ranks with
6×6-weighted rules (keep-primary, weighted mean_quality, primary-only quality).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from arc_decoder import ArcDecoder, hashable, score_full_probmul_3, score_kgmon, score_sum
from arc_loader import ArcDataset
from finalize import merge_keep_primary
from rank_pool_search import (
    RULES,
    _beam,
    _mean_aug,
    make_score,
    oracle_score,
    pass_pair_top1,
    task_score,
)

W = "/opt/work/nvarc"
DATA = "/opt/data/kaggle/arc-agi_evaluation_challenges.json"
SOL = "/opt/data/kaggle/arc-agi_evaluation_solutions.json"

DIRS = {
    "6x6A": f"{W}/eval120_n6x6_a/outputs",
    "6x6B": f"{W}/eval120_n6x6_b/outputs",
    "5x5A": f"{W}/eval120_t3_a/outputs",
    "5x5B": f"{W}/eval120_t3_b/outputs",
}

POOLS = [
    ("6x6A+6x6B", "6x6A", "6x6B"),
    ("6x6A+5x5B", "6x6A", "5x5B"),
    ("6x6B+5x5A", "6x6B", "5x5A"),
    ("6x6A+5x5A", "6x6A", "5x5A"),
    ("6x6B+5x5B", "6x6B", "5x5B"),
    ("5x5A+5x5B", "5x5A", "5x5B"),
]


def tag_sources(dec):
    for samples in dec.decoded_results.values():
        for k, g in samples.items():
            g["_src"] = "extra" if ".p1" in k else "primary"


def merge_dec(dm, dec_a, dec_b):
    dec = ArcDecoder(dm, n_guesses=2)
    for bk, samples in dec_a.decoded_results.items():
        dest = dec.decoded_results.setdefault(bk, {})
        for k, v in samples.items():
            g = dict(v)
            g["_src"] = "primary"
            dest[k] = g
    for bk, samples in dec_b.decoded_results.items():
        dest = dec.decoded_results.setdefault(bk, {})
        for k, v in samples.items():
            g = dict(v)
            g["_src"] = "extra"
            dest[f"{k}.p1"] = g
    return dec


def getter_wmean(w):
    def getter(gs):
        num = den = 0.0
        for g in gs:
            q = -_beam(g) - _mean_aug(g)
            wt = w if g.get("_src") == "primary" else 1.0
            num += wt * q
            den += wt
        return num / den if den else 0.0

    getter.__name__ = f"w{w:g}_mean_q"
    return make_score(getter)


def getter_primary_only_q(gs):
    prim = [g for g in gs if g.get("_src") == "primary"]
    use = prim if prim else gs
    return float(np.mean([-_beam(g) - _mean_aug(g) for g in use]))


def getter_seen_in_primary(gs):
    q = float(np.mean([-_beam(g) - _mean_aug(g) for g in gs]))
    if any(g.get("_src") == "primary" for g in gs):
        q += 10.0
    return q


def getter_w_votes_q(w):
    def getter(gs):
        votes = 0.0
        qs = []
        for g in gs:
            votes += w if g.get("_src") == "primary" else 1.0
            qs.append(-_beam(g) - _mean_aug(g))
        return votes + float(np.mean(qs))

    getter.__name__ = f"w{w:g}_votes_q"
    return make_score(getter)


WEIGHTED = {
    "w2_mean_q": getter_wmean(2),
    "w3_mean_q": getter_wmean(3),
    "w5_mean_q": getter_wmean(5),
    "primary_only_q": make_score(getter_primary_only_q),
    "seen_in_primary": make_score(getter_seen_in_primary),
    "w2_votes_q": getter_w_votes_q(2),
    "w3_votes_q": getter_w_votes_q(3),
}

CORE_RULES = {
    "kgmon": RULES["kgmon"],
    "mean_quality": RULES["mean_quality"],
    "neg_mean_aug": RULES["neg_mean_aug"],
    "neg_best_aug": RULES["neg_best_aug"],
    "probmul3": RULES["probmul3"],
}


def load_one(dm, path):
    dec = ArcDecoder(dm, n_guesses=2)
    dec.load_decoded_results(path)
    return dec


def load_pool(dm, dir_a, dir_b):
    dec = ArcDecoder(dm, n_guesses=2)
    dec.load_decoded_results(dir_a)
    dec.load_decoded_results(dir_b, run_name=".p1")
    tag_sources(dec)
    return dec


def gold_split(replies, dec):
    both = only_p = only_e = neither = 0
    n_out = 0
    per = defaultdict(lambda: [False, False])
    for bk, golds in replies.items():
        n_out += 1
        hg = hashable(golds[0])
        in_p = in_e = False
        for k, s in (dec.decoded_results.get(bk) or {}).items():
            try:
                if hashable(s["solution"]) != hg:
                    continue
            except Exception:
                continue
            if s.get("_src") == "primary":
                in_p = True
            else:
                in_e = True
        task = bk.split("_")[0]
        per[task][0] = per[task][0] or in_p
        per[task][1] = per[task][1] or in_e
        if in_p and in_e:
            both += 1
        elif in_p:
            only_p += 1
        elif in_e:
            only_e += 1
        else:
            neither += 1
    task_only_p = task_only_e = task_both = task_none = 0
    for inp, ine in per.values():
        if inp and ine:
            task_both += 1
        elif inp:
            task_only_p += 1
        elif ine:
            task_only_e += 1
        else:
            task_none += 1
    n_unique = 0
    n_prim = n_extra = 0
    for samples in dec.decoded_results.values():
        grids = set()
        for k, s in samples.items():
            try:
                grids.add(hashable(s["solution"]))
            except Exception:
                pass
            if s.get("_src") == "primary":
                n_prim += 1
            else:
                n_extra += 1
        n_unique += len(grids)
    return {
        "out_both": both,
        "out_only_p": only_p,
        "out_only_e": only_e,
        "out_neither": neither,
        "n_out": n_out,
        "task_both": task_both,
        "task_only_p": task_only_p,
        "task_only_e": task_only_e,
        "task_none": task_none,
        "n_tasks": len(per),
        "n_samples_p": n_prim,
        "n_samples_e": n_extra,
        "n_unique_grids": n_unique,
    }


def report_score(path):
    if not os.path.exists(path):
        return None
    return float(json.load(open(path))["score"])


def main():
    data = ArcDataset.from_file(DATA)
    data.load_replies(SOL)
    dm = data.split_multi_replies()
    replies = dm.replies
    n = 120

    singles = {
        "6x6A": report_score(f"{W}/eval120_n6x6_a/report.json"),
        "6x6B": report_score(f"{W}/eval120_n6x6_b/report.json"),
        "5x5A": report_score(f"{W}/eval120_t3_a/report.json"),
        "5x5B": report_score(f"{W}/eval120_t3_b/report.json"),
    }
    print("single-pass (existing kgmon/default reports)")
    for k, v in singles.items():
        print(f"  {k:8} {v:.3f} ({100*v/n:.2f}%)")
    print()
    print("loading four pickle dirs once...")
    singles_dec = {name: load_one(dm, path) for name, path in DIRS.items()}
    print("loaded.")
    print()

    all_rows = []
    for pname, a, b in POOLS:
        print("=" * 72)
        print(f"{pname}   primary={a}  extra={b}")
        dec = merge_dec(dm, singles_dec[a], singles_dec[b])
        split = gold_split(replies, dec)
        oracle, in_pool, n_out = oracle_score(replies, dec.decoded_results)
        print(
            f"  samples P/E {split['n_samples_p']}/{split['n_samples_e']}  "
            f"unique_grids {split['n_unique_grids']}"
        )
        print(
            f"  gold outputs: both {split['out_both']}  onlyP {split['out_only_p']}  "
            f"onlyE {split['out_only_e']}  none {split['out_neither']}"
        )
        print(
            f"  gold tasks:   both {split['task_both']}  onlyP {split['task_only_p']}  "
            f"onlyE {split['task_only_e']}  none {split['task_none']}"
        )
        print(f"  oracle {oracle:.3f}/{n} = {100*oracle/n:.2f}%  gold_in_pool {in_pool}/{n_out}")

        sel_a = singles_dec[a].run_selection_algo(score_kgmon)
        sel_a_mq = singles_dec[a].run_selection_algo(CORE_RULES["mean_quality"])
        sel_b = singles_dec[b].run_selection_algo(score_kgmon)

        rows = []
        for name, fn in {**CORE_RULES, **WEIGHTED}.items():
            sel = dec.run_selection_algo(fn)
            sc, hit, _ = task_score(replies, sel)
            rows.append((sc, hit, name, "pool"))
            print(f"  {sc:7.3f}  {100*sc/n:5.2f}%  top2 {hit:3d}/{n_out}  {name}")

        for name, primary_sel, fn in (
            ("keepP+kgmon", sel_a, score_kgmon),
            ("keepP+mean_q", sel_a_mq, CORE_RULES["mean_quality"]),
            ("keepP+w3_mean_q", sel_a_mq, WEIGHTED["w3_mean_q"]),
            ("keepP+primary_only_q", sel_a_mq, WEIGHTED["primary_only_q"]),
        ):
            sel = merge_keep_primary(primary_sel, dec.run_selection_algo(fn))
            sc, hit, _ = task_score(replies, sel)
            rows.append((sc, hit, name, "keep"))
            print(f"  {sc:7.3f}  {100*sc/n:5.2f}%  top2 {hit:3d}/{n_out}  {name}")

        sel = pass_pair_top1(sel_a, sel_b)
        sc, hit, _ = task_score(replies, sel)
        rows.append((sc, hit, "P_top1+E_top1", "pair"))
        print(f"  {sc:7.3f}  {100*sc/n:5.2f}%  top2 {hit:3d}/{n_out}  P_top1+E_top1")

        rows.sort(reverse=True)
        best = rows[0]
        print(f"  BEST {best[2]} {best[0]:.3f} ({100*best[0]/n:.2f}%)  oracle {oracle:.3f}")
        all_rows.append((pname, oracle, split, rows))

    print()
    print("=" * 72)
    print("summary (oracle / mean_quality / w3_mean_q / keepP+mean_q / best)")
    print(f"{'pool':<14} {'oracle':>7} {'mean_q':>7} {'w3_mq':>7} {'keepP':>7} {'best':>7}  best_rule")
    for pname, oracle, split, rows in all_rows:
        by = {name: sc for sc, hit, name, kind in rows}
        best = max(rows)
        print(
            f"{pname:<14} {oracle:7.2f} {by.get('mean_quality', 0):7.2f} "
            f"{by.get('w3_mean_q', 0):7.2f} {by.get('keepP+mean_q', 0):7.2f} "
            f"{best[0]:7.2f}  {best[2]}"
        )
        print(
            f"{'':14}   onlyE_gold_tasks={split['task_only_e']}  "
            f"onlyP={split['task_only_p']}  both={split['task_both']}"
        )


if __name__ == "__main__":
    main()
