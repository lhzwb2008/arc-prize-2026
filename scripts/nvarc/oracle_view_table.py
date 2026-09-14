#!/usr/bin/env python3
"""CPU: oracle / mean_quality / kgmon for decode-view subsets of existing TTT pickles.

Loads each pickle dir once, then filters samples in memory. Does not need GPU.
Train-aug count is frozen in the pickle dir; only geos change.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from arc_decoder import ArcDecoder, score_kgmon, score_mean_quality
from arc_loader import ArcDataset
from rank_pool_search import oracle_score, task_score
from score_view_subset import keep_files

N = 120
# 3090 seconds / task from the 8×6 timing model; two-pass budget is 427s 3090-eq.
SEC_PER_TRAIN_AUG = 8.8
SEC_PER_VIEW = 23.9
TWO_PASS_BUDGET_S = 427.0


def n_train_from_tag(tag: str) -> int | None:
    # tags like 16x16, 8x8A, 6x8, 7x8, 5x8
    head = tag.split("x", 1)[0]
    try:
        return int("".join(c for c in head if c.isdigit()))
    except ValueError:
        return None


def two_pass_pct(n_train: int | None, geos: int) -> float | None:
    if not n_train:
        return None
    one = n_train * SEC_PER_TRAIN_AUG + geos * SEC_PER_VIEW
    return 100.0 * (2.0 * one) / TWO_PASS_BUDGET_S


def unique_tasks(outputs: str) -> int:
    keys = set()
    for fn in os.listdir(outputs):
        path = os.path.join(outputs, fn)
        if os.path.isfile(path) and "." in fn and not fn.startswith("."):
            keys.add(fn.split(".")[0])
    return len(keys)


def metrics_from_decoded(replies, decoded):
    ora, in_pool, n_out = oracle_score(replies, decoded)
    mq, _, _ = task_score(replies, {bk: score_mean_quality(v) for bk, v in decoded.items()})
    kg, _, _ = task_score(replies, {bk: score_kgmon(v) for bk, v in decoded.items()})
    return {
        "oracle": ora,
        "mean_quality": mq,
        "kgmon": kg,
        "gold_in_pool": in_pool,
        "n_out": n_out,
    }


def filter_decoded(decoded, outputs: str, geos: int):
    kept = set(keep_files(outputs, geos, None))
    out = {}
    for bk, samples in decoded.items():
        filt = {k: v for k, v in samples.items() if k.split(".out")[0] in kept}
        if filt:
            out[bk] = filt
    return out, len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], metavar="TAG=PATH",
                    help="repeatable, e.g. 8x8=/opt/work/nvarc/eval120_half_a/outputs")
    ap.add_argument("--geos", default="1,2,4,5,6,8")
    ap.add_argument("--min-tasks", type=int, default=100,
                    help="skip pickle dirs with fewer unique task ids")
    ap.add_argument("--data", default="/opt/data/kaggle/arc-agi_evaluation_challenges.json")
    ap.add_argument("--solutions", default="/opt/data/kaggle/arc-agi_evaluation_solutions.json")
    ap.add_argument("--out", default="/opt/work/nvarc/eval120_search/min_pool_views.json")
    args = ap.parse_args()
    geos = [int(x) for x in args.geos.split(",") if x.strip()]

    data = ArcDataset.from_file(args.data)
    data.load_replies(args.solutions)
    dm = data.split_multi_replies()
    replies = dm.replies

    runs = []
    print(f"{'run':<8} {'geos':>4} {'oracle':>8} {'mean_q':>8} {'kgmon':>8} "
          f"{'gold':>10} {'pkl':>6} {'2pass%':>7}")
    for spec in args.run:
        tag, path = spec.split("=", 1)
        if not os.path.isdir(path):
            print(f"skip {tag}: missing {path}")
            continue
        n_tasks = unique_tasks(path)
        if n_tasks < args.min_tasks:
            print(f"skip {tag}: {n_tasks} tasks (< {args.min_tasks}) at {path}")
            continue
        print(f"=== {tag}  {path}  tasks={n_tasks} ===")
        dec = ArcDecoder(dm, n_guesses=2)
        dec.load_decoded_results(path)
        n_train = n_train_from_tag(tag)
        rows = []
        for g in geos:
            subset, n_kept = filter_decoded(dec.decoded_results, path, g)
            row = metrics_from_decoded(replies, subset)
            row["geos"] = g
            row["n_pickles_kept"] = n_kept
            row["two_pass_budget_pct"] = two_pass_pct(n_train, g)
            rows.append(row)
            pct = row["two_pass_budget_pct"]
            pct_s = f"{pct:6.0f}%" if pct is not None else "     -"
            print(
                f"{tag:<8} {g:4d} {row['oracle']:8.2f} {row['mean_quality']:8.2f} "
                f"{row['kgmon']:8.2f} {row['gold_in_pool']:4d}/{row['n_out']:<4d} "
                f"{n_kept:6d} {pct_s}"
            )
        full = metrics_from_decoded(replies, dec.decoded_results)
        n_pkl = sum(
            1 for fn in os.listdir(path)
            if os.path.isfile(os.path.join(path, fn))
        )
        full["n_pickles"] = n_pkl
        full["n_tasks_pkl"] = n_tasks
        runs.append({"tag": tag, "path": path, "n_train": n_train, "full": full, "geos": rows})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"n_tasks": N, "runs": runs}, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
