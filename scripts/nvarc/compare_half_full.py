#!/usr/bin/env python3
"""Score finished half-cost tasks against the existing full-cost eval120 run."""
from __future__ import annotations

import argparse
import bz2
import collections
import json
import os
import pickle
import sys

import numpy as np

from arc_decoder import ArcDecoder, hashable, score_kgmon
from arc_loader import ArcDataset


def pickle_task_ids(store: str) -> dict[str, int]:
    return collections.Counter(fn.split("_")[0] for fn in os.listdir(store) if "_" in fn)


def unique_grids(store: str, task: str) -> dict[str, set]:
    """base_key -> set of hashable grids from all pickles for that task."""
    out = collections.defaultdict(set)
    for fn in os.listdir(store):
        if not fn.startswith(task + "_"):
            continue
        bk = fn.split(".")[0]
        with bz2.BZ2File(os.path.join(store, fn)) as f:
            samples = pickle.load(f)
        for s in samples:
            out[bk].add(hashable(s["solution"]))
    return dict(out)


def per_task_from_selected(data: ArcDataset, selected: dict) -> dict[str, float]:
    per_task = {}
    for bk, correct in data.split_multi_replies().replies.items():
        task = bk.split("_")[0]
        guesses = selected.get(bk, [])
        ok = any(np.array_equal(g, correct[0]) for g in guesses[:2])
        per_task.setdefault(task, []).append(bool(ok))
    return {t: sum(v) / len(v) for t, v in per_task.items()}


def gold_in_pool(data: ArcDataset, grids: dict[str, set]) -> dict[str, float]:
    per_task = {}
    replies = data.split_multi_replies().replies
    for bk, correct in replies.items():
        task = bk.split("_")[0]
        h = hashable(correct[0])
        per_task.setdefault(task, []).append(h in grids.get(bk, set()))
    return {t: sum(v) / len(v) for t, v in per_task.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/opt/data/kaggle/arc-agi_evaluation_challenges.json")
    ap.add_argument("--solutions", default="/opt/data/kaggle/arc-agi_evaluation_solutions.json")
    ap.add_argument("--full-outputs", default="/opt/work/nvarc/eval120/outputs")
    ap.add_argument("--full-report", default="/opt/work/nvarc/eval120/report.json")
    ap.add_argument("--half-outputs", default="/opt/work/nvarc/eval120_half_a/outputs")
    args = ap.parse_args()

    half_n = pickle_task_ids(args.half_outputs)
    full_n = pickle_task_ids(args.full_outputs)
    keys = sorted(half_n)
    if not keys:
        print("no half-cost pickles yet")
        return 1

    full_rep = json.load(open(args.full_report))["per_task"]
    data = ArcDataset.from_file(args.data, keys=keys)
    data.load_replies(args.solutions)
    dm = data.split_multi_replies()

    dec_h = ArcDecoder(dm, n_guesses=2)
    dec_h.load_decoded_results(args.half_outputs)
    sel_h = dec_h.run_selection_algo(score_kgmon)
    score_h = per_task_from_selected(data, sel_h)

    # full selection restricted to the same keys (reload full pickles only for these tasks)
    dec_f = ArcDecoder(dm, n_guesses=2)
    for fn in os.listdir(args.full_outputs):
        if fn.split("_")[0] in half_n:
            path = os.path.join(args.full_outputs, fn)
            with bz2.BZ2File(path) as f:
                outputs = pickle.load(f)
            bk = fn.split(".")[0]
            dec_f.decoded_results[bk] = dec_f.decoded_results.get(bk, {})
            for i, sample in enumerate(outputs):
                dec_f.decoded_results[bk][f"{fn}.out{i}"] = sample
    sel_f = dec_f.run_selection_algo(score_kgmon) if dec_f.decoded_results else {}
    score_f = per_task_from_selected(data, sel_f) if sel_f else {k: full_rep.get(k, 0) for k in keys}

    print(f"half-A finished tasks with pickles: {len(keys)}")
    print(f"{'task':<12} {'full':>6} {'half':>6} {'d':>5} {'fpkl':>5} {'hpkl':>5} {'jacc':>6} {'goldF':>6} {'goldH':>6}")
    both_ok = half_only = full_only = both_miss = 0
    jaccs = []
    rows = []
    for t in keys:
        gf = unique_grids(args.full_outputs, t)
        gh = unique_grids(args.half_outputs, t)
        # mean jaccard over test inputs present in either
        js = []
        gold_f = gold_in_pool(data, gf).get(t, 0)
        gold_h = gold_in_pool(data, gh).get(t, 0)
        bks = set(gf) | set(gh)
        for bk in bks:
            a, b = gf.get(bk, set()), gh.get(bk, set())
            if a or b:
                js.append(len(a & b) / len(a | b))
        j = float(np.mean(js)) if js else 0.0
        jaccs.append(j)
        sf, sh = score_f.get(t, full_rep.get(t, 0)), score_h.get(t, 0)
        if sh > 0 and sf > 0:
            both_ok += 1
        elif sh > 0:
            half_only += 1
        elif sf > 0:
            full_only += 1
        else:
            both_miss += 1
        rows.append((t, sf, sh, sh - sf, full_n.get(t, 0), half_n[t], j, gold_f, gold_h))
        print(f"{t:<12} {sf:6.2f} {sh:6.2f} {sh-sf:+5.2f} {full_n.get(t,0):5d} {half_n[t]:5d} {j:6.2f} {gold_f:6.2f} {gold_h:6.2f}")

    n = len(keys)
    print()
    print(f"on these {n} tasks:")
    print(f"  full kgmon  {sum(r[1] for r in rows):.2f}/{n} = {100*sum(r[1] for r in rows)/n:.1f}%")
    print(f"  half kgmon  {sum(r[2] for r in rows):.2f}/{n} = {100*sum(r[2] for r in rows)/n:.1f}%")
    print(f"  both>0 {both_ok}  full-only {full_only}  half-only {half_only}  both-0 {both_miss}")
    print(f"  mean candidate Jaccard {float(np.mean(jaccs)) if jaccs else 0:.2f}")
    print(f"  gold in full pool {sum(r[7] for r in rows):.2f}  gold in half pool {sum(r[8] for r in rows):.2f}")
    print("  goldF/goldH = whether the gold grid appears in ANY decoded candidate (ranking-independent)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
