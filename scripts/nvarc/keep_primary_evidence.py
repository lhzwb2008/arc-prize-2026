#!/usr/bin/env python3
"""CPU: is keep-primary a real gain or 8×6 luck?

For every independent two-pass pair on the 120 public eval:
  pooled mean_quality  vs  keep-A  vs  keep-B (symmetry)  vs  keep-both
Counts flips (+/-), displacement risk, sign test across pairs, and a
task-level bootstrap of (keepA - pooled).
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter

import numpy as np

from arc_decoder import ArcDecoder, hashable, score_mean_quality
from arc_loader import ArcDataset
from rank_pool_search import oracle_score, task_score
from score_view_subset import keep_files

DATA = "/opt/data/kaggle/arc-agi_evaluation_challenges.json"
SOL = "/opt/data/kaggle/arc-agi_evaluation_solutions.json"
W = "/opt/work/nvarc"
OUT = f"{W}/eval120_search/keep_primary_evidence.json"
N = 120

# (name, dir, geos cut or None)
PAIRS = [
    ("8x6  half A/B", f"{W}/eval120_half_a/outputs", f"{W}/eval120_half_b/outputs", 6),
    ("8x8  half A/B", f"{W}/eval120_half_a/outputs", f"{W}/eval120_half_b/outputs", None),
    ("8x5  half A/B", f"{W}/eval120_half_a/outputs", f"{W}/eval120_half_b/outputs", 5),
    ("6x6  v11 A/B", f"{W}/eval120_n6x6_v11_a/outputs", f"{W}/eval120_n6x6_v11_b/outputs", None),
    ("6x6  old A/B", f"{W}/eval120_n6x6_a/outputs", f"{W}/eval120_n6x6_b/outputs", None),
    ("5x5  t3 A/B", f"{W}/eval120_t3_a/outputs", f"{W}/eval120_t3_b/outputs", None),
    ("16x16 + pass2", f"{W}/eval120/outputs", f"{W}/eval120_pass2/outputs", None),
]


def load(dm, path, geos, run_name=""):
    dec = ArcDecoder(dm, n_guesses=2)
    dec.load_decoded_results(path, run_name=run_name)
    if not geos:
        return dec.decoded_results
    kept = set(keep_files(path, geos, None))
    out = {}
    for bk, samples in dec.decoded_results.items():
        filt = {}
        for k, v in samples.items():
            fn = k.split(".out")[0]
            if run_name and fn.endswith(run_name):
                fn = fn[: -len(run_name)]
            if fn in kept:
                filt[k] = v
        if filt:
            out[bk] = filt
    return out


def sel_of(dm, decoded):
    d = ArcDecoder(dm, n_guesses=2)
    d.decoded_results = decoded
    return d.run_selection_algo(score_mean_quality)


def force_top1(sel_src, sel_pool):
    """Pooled top-2, but sel_src top-1 must be one of the attempts (replaces #2)."""
    out = {}
    forced = 0
    for bk in set(sel_src) | set(sel_pool):
        s1 = (sel_src.get(bk) or [None])[0]
        top = list(sel_pool.get(bk) or [])[:2]
        if s1 is not None and not any(np.array_equal(s1, g) for g in top):
            top = (top[:1] + [s1]) if top else [s1]
            forced += 1
        out[bk] = top
    return out, forced


def force_both(sel_a, sel_b, sel_pool):
    """attempt1 = pooled top-1; attempt2 = A top-1 if missing else B top-1 if missing else pooled #2."""
    out = {}
    for bk in set(sel_a) | set(sel_b) | set(sel_pool):
        pool = list(sel_pool.get(bk) or [])
        a1 = (sel_a.get(bk) or [None])[0]
        b1 = (sel_b.get(bk) or [None])[0]
        top = pool[:1]
        for cand in (a1, b1, *(pool[1:2])):
            if cand is None:
                continue
            if not any(np.array_equal(cand, g) for g in top):
                top.append(cand)
                break
        out[bk] = top[:2]
    return out


def per_output_ok(replies, sel):
    ok = {}
    for bk, golds in replies.items():
        gs = list(sel.get(bk) or [])[:2]
        ok[bk] = any(np.array_equal(g, golds[0]) for g in gs)
    return ok


def flips(ok_base, ok_new):
    plus = [bk for bk in ok_base if not ok_base[bk] and ok_new[bk]]
    minus = [bk for bk in ok_base if ok_base[bk] and not ok_new[bk]]
    return plus, minus


def displacement_risk(replies, sel_a, sel_pool):
    """Outputs where pooled #2 is gold and A top-1 is not among pooled top-2 (forcing would lose it)."""
    risk = []
    for bk, golds in replies.items():
        pool = list(sel_pool.get(bk) or [])[:2]
        if len(pool) < 2 or not np.array_equal(pool[1], golds[0]):
            continue
        a1 = (sel_a.get(bk) or [None])[0]
        if a1 is not None and not any(np.array_equal(a1, g) for g in pool):
            risk.append(bk)
    return risk


def task_points(replies, ok):
    per = {}
    for bk, v in ok.items():
        per.setdefault(bk.split("_")[0], []).append(v)
    return {t: sum(v) / len(v) for t, v in per.items()}


def bootstrap(pts_base, pts_new, iters=4000, seed=0):
    tasks = sorted(pts_base)
    d = np.array([pts_new[t] - pts_base[t] for t in tasks])
    rng = random.Random(seed)
    n = len(tasks)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(d[idx].sum())
    diffs = np.array(diffs)
    return {
        "mean": float(diffs.mean()),
        "p_lt_0": float((diffs < 0).mean()),
        "p_eq_0": float((diffs == 0).mean()),
        "p_gt_0": float((diffs > 0).mean()),
        "ci90": [float(np.percentile(diffs, 5)), float(np.percentile(diffs, 95))],
    }


def main():
    data = ArcDataset.from_file(DATA)
    data.load_replies(SOL)
    dm = data.split_multi_replies()
    replies = dm.replies

    rows = []
    tot_plus_a = tot_minus_a = tot_plus_b = tot_minus_b = 0
    for name, pa, pb, geos in PAIRS:
        if not (os.path.isdir(pa) and os.path.isdir(pb)):
            print(f"skip {name}")
            continue
        A = load(dm, pa, geos, "")
        B = load(dm, pb, geos, ".p1")
        pool = {bk: {**(A.get(bk) or {}), **(B.get(bk) or {})} for bk in set(A) | set(B)}
        sel_a, sel_b, sel_p = sel_of(dm, A), sel_of(dm, B), sel_of(dm, pool)

        keep_a, n_fa = force_top1(sel_a, sel_p)
        keep_b, n_fb = force_top1(sel_b, sel_p)
        keep_ab = force_both(sel_a, sel_b, sel_p)

        ok_a, ok_b = per_output_ok(replies, sel_a), per_output_ok(replies, sel_b)
        ok_p = per_output_ok(replies, sel_p)
        ok_ka, ok_kb, ok_kab = (per_output_ok(replies, s) for s in (keep_a, keep_b, keep_ab))

        sc = {
            "A": task_score(replies, sel_a)[0],
            "B": task_score(replies, sel_b)[0],
            "pool": task_score(replies, sel_p)[0],
            "keepA": task_score(replies, keep_a)[0],
            "keepB": task_score(replies, keep_b)[0],
            "keepAB": task_score(replies, keep_ab)[0],
        }
        ora = oracle_score(replies, pool)[0]
        pa_, ma_ = flips(ok_p, ok_ka)
        pb_, mb_ = flips(ok_p, ok_kb)
        risk_a = displacement_risk(replies, sel_a, sel_p)
        risk_b = displacement_risk(replies, sel_b, sel_p)
        # single-pass-correct outputs lost by pooling (what keep is meant to recover)
        a_right_pool_wrong = [bk for bk in ok_a if ok_a[bk] and not ok_p[bk]]
        b_right_pool_wrong = [bk for bk in ok_b if ok_b[bk] and not ok_p[bk]]
        a1_right_pool_wrong = [
            bk for bk in a_right_pool_wrong
            if (sel_a.get(bk) or [None])[0] is not None
            and np.array_equal((sel_a.get(bk) or [None])[0], replies[bk][0])
        ]

        boot = bootstrap(task_points(replies, ok_p), task_points(replies, ok_ka))
        tot_plus_a += len(pa_); tot_minus_a += len(ma_)
        tot_plus_b += len(pb_); tot_minus_b += len(mb_)

        row = {
            "pair": name,
            "oracle_pct": 100 * ora / N,
            **{k: 100 * v / N for k, v in sc.items()},
            "d_keepA": 100 * (sc["keepA"] - sc["pool"]) / N,
            "d_keepB": 100 * (sc["keepB"] - sc["pool"]) / N,
            "d_keepAB": 100 * (sc["keepAB"] - sc["pool"]) / N,
            "forcedA": n_fa, "forcedB": n_fb,
            "flipA_plus": pa_, "flipA_minus": ma_,
            "flipB_plus": pb_, "flipB_minus": mb_,
            "riskA": risk_a, "riskB": risk_b,
            "A_right_pool_wrong": a_right_pool_wrong,
            "A_top1_right_pool_wrong": a1_right_pool_wrong,
            "B_right_pool_wrong": b_right_pool_wrong,
            "boot_keepA_minus_pool": boot,
        }
        rows.append(row)
        print(f"\n=== {name}  oracle {row['oracle_pct']:.2f} ===")
        print(f"  A {row['A']:.2f}  B {row['B']:.2f}  pool {row['pool']:.2f}  "
              f"keepA {row['keepA']:.2f} ({row['d_keepA']:+.2f})  keepB {row['keepB']:.2f} ({row['d_keepB']:+.2f})  "
              f"keepAB {row['keepAB']:.2f} ({row['d_keepAB']:+.2f})")
        print(f"  keepA forced {n_fa}: +{len(pa_)} {pa_}  -{len(ma_)} {ma_}")
        print(f"  keepB forced {n_fb}: +{len(pb_)} {pb_}  -{len(mb_)} {mb_}")
        print(f"  displacement risk (pool#2 gold & src top1 outside): A {len(risk_a)} {risk_a}  B {len(risk_b)} {risk_b}")
        print(f"  A right but pool wrong: {len(a_right_pool_wrong)} (A top-1 right: {len(a1_right_pool_wrong)})  "
              f"B right but pool wrong: {len(b_right_pool_wrong)}")
        print(f"  bootstrap keepA-pool: mean {boot['mean']:+.2f} pts  P(<0) {boot['p_lt_0']:.3f}  "
              f"P(=0) {boot['p_eq_0']:.3f}  ci90 {boot['ci90'][0]:.2f}..{boot['ci90'][1]:.2f}")

    print("\n=== totals across pairs ===")
    print(f"  keepA flips: +{tot_plus_a} / -{tot_minus_a}")
    print(f"  keepB flips: +{tot_plus_b} / -{tot_minus_b}")
    # sign test: P(all-plus | fair coin) over independent-ish flips
    n_all = tot_plus_a + tot_minus_a
    if n_all:
        print(f"  sign test keepA: P(>= {tot_plus_a} of {n_all} plus | p=0.5) = {sum(__import__('math').comb(n_all,k) for k in range(tot_plus_a,n_all+1))/2**n_all:.4f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rows, open(OUT, "w"), indent=2, default=str)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
