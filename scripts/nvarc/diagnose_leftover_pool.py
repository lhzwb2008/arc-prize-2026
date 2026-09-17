#!/usr/bin/env python3
"""CPU: leftover-B vs full-B pooling. Needs local eval pickle dirs.

Compares A-only, mixed mean_quality, keep-primary, and pass-pair. Then
restricts B to a cheap-first slice (the Kaggle leftover-B regime) and
repeats, so we can see mixed ranking destroy A's floor.

  python diagnose_leftover_pool.py \
      --outputs /opt/work/nvarc/eval120_n8x6_a/outputs \
      --outputs-b /opt/work/nvarc/eval120_n8x6_b/outputs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from arc_decoder import (  # noqa: E402
    ArcDecoder,
    merge_keep_primary,
    merge_pass_pair,
    score_mean_quality,
)
from arc_loader import ArcDataset  # noqa: E402
from rank_pool_search import oracle_score, task_score  # noqa: E402


def estimated_work(task):
    def ntok(g):
        return len(g) * (len(g[0]) + 1) if g and g[0] else 0
    train_tokens = sum(ntok(p["input"]) + ntok(p["output"]) for p in task["train"])
    ratios = [ntok(p["output"]) / max(1, ntok(p["input"])) for p in task["train"]] or [1.0]
    ratios.sort()
    ratio = ratios[len(ratios) // 2]
    test_tokens = sum(ntok(t["input"]) * (1 + ratio) for t in task["test"])
    n_train = int(os.getenv("ARC_N_TRAIN_AUG", "8"))
    n_geos = int(os.getenv("ARC_N_EVAL_GEOS", "6"))
    return train_tokens * n_train + test_tokens * n_geos * len(task["test"])

N = 120


def pct(score: float, n: int = N) -> float:
    return 100.0 * float(score) / n


def load_sel(dm, path, run_name=""):
    dec = ArcDecoder(dm, n_guesses=2)
    dec.load_decoded_results(path, run_name=run_name)
    sel = dec.run_selection_algo(score_mean_quality) if dec.decoded_results else {}
    return dec, sel


def restrict_b(decoded, keep_tasks: set[str]) -> dict:
    out = {}
    for bk, samples in decoded.items():
        if bk.split("_")[0] in keep_tasks:
            out[bk] = samples
    return out


def merge_decoded(a, b):
    out = {}
    for bk in set(a) | set(b):
        out[bk] = {**(a.get(bk) or {}), **(b.get(bk) or {})}
    return out


def report(tag, replies, selected, n=N):
    sc, hit, n_out = task_score(replies, selected)
    print(f"  {tag:<22} {pct(sc):6.2f}  {sc:7.3f}/{n}  top2 {hit:3d}/{n_out}", flush=True)
    return sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/opt/data/kaggle/arc-agi_evaluation_challenges.json")
    ap.add_argument("--solutions", default="/opt/data/kaggle/arc-agi_evaluation_solutions.json")
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--outputs-b", required=True)
    ap.add_argument("--b-frac", type=float, default=0.35,
                    help="fraction of B tasks to keep (leftover-B)")
    ap.add_argument("--b-order", choices=["cheap", "expensive"], default="cheap",
                    help="which end of the work-sorted list leftover B keeps")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not (os.path.isdir(args.outputs) and os.path.isdir(args.outputs_b)):
        print("missing pickle dirs; skip")
        return 1

    data = ArcDataset.from_file(args.data)
    data.load_replies(args.solutions)
    dm = data.split_multi_replies()
    replies = dm.replies
    raw = json.loads(Path(args.data).read_text())

    dec_a, sel_a = load_sel(dm, args.outputs)
    dec_b, sel_b = load_sel(dm, args.outputs_b, run_name=".p1")
    pooled = merge_decoded(dec_a.decoded_results, dec_b.decoded_results)
    sel_mixed = ArcDecoder(dm, n_guesses=2)
    sel_mixed.decoded_results = pooled
    sel_p = sel_mixed.run_selection_algo(score_mean_quality)

    ora, in_pool, n_out = oracle_score(replies, pooled)
    print(f"full A+B oracle {pct(ora):.2f}  gold_in_pool {in_pool}/{n_out}", flush=True)
    print("=== full B ===", flush=True)
    scores = {
        "A": report("A-only", replies, sel_a),
        "B": report("B-only", replies, sel_b),
        "mixed": report("mixed mean_q", replies, sel_p),
        "keepP": report("keep-primary", replies, merge_keep_primary(sel_a, sel_p)),
        "pair": report("pass-pair", replies, merge_pass_pair(sel_a, sel_b)),
    }

    tasks = sorted(raw, key=lambda k: estimated_work(raw[k]))
    n_keep = max(1, int(round(len(tasks) * args.b_frac)))
    if args.b_order == "expensive":
        keep = set(tasks[-n_keep:])
        label = "expensive-first"
    else:
        keep = set(tasks[:n_keep])
        label = "cheap-first"
    decoded_b_left = restrict_b(dec_b.decoded_results, keep)
    sel_b_left = ArcDecoder(dm, n_guesses=2)
    sel_b_left.decoded_results = decoded_b_left
    sel_b2 = sel_b_left.run_selection_algo(score_mean_quality) if decoded_b_left else {}
    pooled_left = merge_decoded(dec_a.decoded_results, decoded_b_left)
    mix_left = ArcDecoder(dm, n_guesses=2)
    mix_left.decoded_results = pooled_left
    sel_p2 = mix_left.run_selection_algo(score_mean_quality)

    ora2, in_pool2, n_out2 = oracle_score(replies, pooled_left)
    print(f"\n=== leftover-B {label} {n_keep}/{len(tasks)} tasks ===", flush=True)
    print(f"leftover oracle {pct(ora2):.2f}  gold_in_pool {in_pool2}/{n_out2}", flush=True)
    left = {
        "A": report("A-only", replies, sel_a),
        "mixed": report("mixed mean_q", replies, sel_p2),
        "keepP": report("keep-primary", replies, merge_keep_primary(sel_a, sel_p2)),
        "pair": report("pass-pair", replies, merge_pass_pair(sel_a, sel_b2)),
    }
    print(
        f"leftover mixed-A {pct(left['mixed'] - left['A']):+.2f}  "
        f"pair-A {pct(left['pair'] - left['A']):+.2f}  "
        f"mixed-pair {pct(left['mixed'] - left['pair']):+.2f}",
        flush=True,
    )

    doc = {
        "full": {k: {"score": v, "pct": pct(v)} for k, v in scores.items()},
        "leftover": {k: {"score": v, "pct": pct(v)} for k, v in left.items()},
        "b_frac": args.b_frac,
        "b_order": args.b_order,
        "n_b_kept": n_keep,
        "oracle_full_pct": pct(ora),
        "oracle_leftover_pct": pct(ora2),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
