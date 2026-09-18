#!/usr/bin/env python3
"""Order the leftover-B queue by pass-A uncertainty per unit cost.

Generic rule, no labels, no task ids: for each task read pass-A's own
candidates, take the top-1 vote share over its decoded samples (min across
test outputs; 0 when A produced nothing), and sort by
(1 - share) / estimated_work descending. Tasks where A already agrees with
itself go last; on the public n8x6 pickles B never lifted any task with
share >= 0.5.

    python b_priority_queue.py --data ... --outputs A_DIR --keys-out b_keys.json
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


def estimated_work(task, n_train: int | None = None, n_geos: int | None = None) -> float:
    n_train = n_train or int(os.getenv("ARC_N_TRAIN_AUG", "8"))
    n_geos = n_geos or int(os.getenv("ARC_N_EVAL_GEOS", "6"))

    def ntok(g):
        return len(g) * (len(g[0]) + 1) if g and g[0] else 0

    train_tokens = sum(ntok(p["input"]) + ntok(p["output"]) for p in task["train"])
    ratios = [ntok(p["output"]) / max(1, ntok(p["input"])) for p in task["train"]] or [1.0]
    ratios.sort()
    ratio = ratios[len(ratios) // 2]
    test_tokens = sum(ntok(t["input"]) * (1 + ratio) for t in task["test"])
    return float(train_tokens * n_train + test_tokens * n_geos * len(task["test"]))


def top1_share(samples: dict) -> float:
    """Vote share of the most common decoded grid among one output's samples."""
    n = len(samples)
    if n == 0:
        return 0.0
    counts: dict = {}
    for g in samples.values():
        k = tuple(map(tuple, np.asarray(g["solution"])))
        counts[k] = counts.get(k, 0) + 1
    return max(counts.values()) / n


def task_shares(decoded: dict, tasks: list[str]) -> dict[str, float]:
    per: dict[str, list[float]] = {t: [] for t in tasks}
    for bk, samples in decoded.items():
        t = bk.split("_")[0]
        if t in per:
            per[t].append(top1_share(samples))
    return {t: (min(v) if v else 0.0) for t, v in per.items()}


def priority_order(tasks: list[str], work: dict[str, float], share: dict[str, float],
                   skip_confident: float | None = None) -> list[str]:
    keep = [t for t in tasks if skip_confident is None or share.get(t, 0.0) < skip_confident]
    return sorted(keep, key=lambda t: (-((1.0 - share.get(t, 0.0)) / max(1.0, work[t])), work[t]))


def build_queue(data_path: str, a_dir: str, skip_done_dir: str = "",
                skip_confident: float | None = None) -> tuple[list[str], dict]:
    from arc_decoder import ArcDecoder

    raw = json.loads(Path(data_path).read_text())
    tasks = list(raw)
    work = {t: estimated_work(raw[t]) for t in tasks}
    dec = ArcDecoder(None, n_guesses=2)
    dec.load_decoded_results(a_dir)
    share = task_shares(dec.decoded_results, tasks)
    order = priority_order(tasks, work, share, skip_confident)
    if skip_done_dir and os.path.isdir(skip_done_dir):
        done = {fn.split("_", 1)[0] for fn in os.listdir(skip_done_dir) if "_" in fn}
        order = [t for t in order if t not in done]
    meta = {
        "n_tasks": len(tasks),
        "n_queue": len(order),
        "n_a_missing": sum(1 for t in tasks if share[t] == 0.0),
        "n_confident_ge_0.5": sum(1 for t in tasks if share[t] >= 0.5),
        "skip_confident": skip_confident,
        "head": [{"task": t, "share": round(share[t], 3), "work": work[t]} for t in order[:10]],
    }
    return order, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outputs", required=True, help="pass-A pickle dir")
    ap.add_argument("--skip-done", default="", help="B pickle dir; drop tasks already there")
    ap.add_argument("--skip-confident", type=float, default=None,
                    help="drop tasks whose A top-1 share >= this (off by default)")
    ap.add_argument("--keys-out", required=True)
    args = ap.parse_args()
    order, meta = build_queue(args.data, args.outputs, args.skip_done, args.skip_confident)
    Path(args.keys_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.keys_out).write_text(json.dumps(order) + "\n")
    print(json.dumps(meta), flush=True)
    print(f"wrote {len(order)} keys -> {args.keys_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
