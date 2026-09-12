"""Rescore an existing pickle dir with a subset of decode views.

Geo index selection matches arc_solver.py (pairs then optional transpose/rot).
This is a paired ablation: same TTT adapter, fewer DFS views in kgmon.

Example:
    python score_view_subset.py --outputs /opt/work/nvarc/eval120_half_a/outputs \\
        --geos 6 --report /tmp/half_g6.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile

from finalize import main as finalize_main


def geo_indices(n_eval_geos: int) -> list[int]:
    pairs = [(0, 2), (1, 3)]
    singles: list[int] = []
    if n_eval_geos >= 8:
        pairs += [(4, 6), (5, 7)]
    elif n_eval_geos >= 6:
        pairs += [(4, 6)]
    elif n_eval_geos >= 5:
        singles = [4]
    elif n_eval_geos < 4:
        pairs = [(0, 2)]
        if n_eval_geos < 2:
            pairs = []
            singles = [0]
    ids: list[int] = []
    for a, b in pairs:
        ids.extend((a, b))
    ids.extend(singles)
    return sorted(dict.fromkeys(ids))


def keep_files(src: str, geos: int, n_eval_aug: int | None) -> list[str]:
    groups: dict[str, list[str]] = {}
    for fn in os.listdir(src):
        path = os.path.join(src, fn)
        if not os.path.isfile(path) or fn.startswith("."):
            continue
        if "." not in fn:
            continue
        groups.setdefault(fn.split(".")[0], []).append(fn)

    want = set(geo_indices(geos))
    kept = []
    for files in groups.values():
        files = sorted(files)
        n = len(files)
        src_n_perm = 2 if n >= 15 else 1
        n_geos_present = n // src_n_perm
        take_perm = src_n_perm
        if n_eval_aug is not None:
            take_perm = min(src_n_perm, max(1, n_eval_aug))
        for gi in range(n_geos_present):
            if gi not in want:
                continue
            start = gi * src_n_perm
            kept.extend(files[start:start + take_perm])
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--geos", type=int, required=True)
    ap.add_argument("--n-eval-aug", type=int, default=None,
                    help="keep this many color perms per geo (default: all present)")
    ap.add_argument("--data", default=os.getenv(
        "NVARC_DATA", "/opt/data/kaggle/arc-agi_evaluation_challenges.json"))
    ap.add_argument("--solutions", default=os.getenv(
        "NVARC_SOL", "/opt/data/kaggle/arc-agi_evaluation_solutions.json"))
    ap.add_argument("--submission", default="")
    ap.add_argument("--report", required=True)
    ap.add_argument("--keys", default="")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    kept = keep_files(args.outputs, args.geos, args.n_eval_aug)
    tmp = tempfile.mkdtemp(prefix="viewsub_")
    try:
        for fn in kept:
            os.symlink(os.path.abspath(os.path.join(args.outputs, fn)), os.path.join(tmp, fn))
        sub = args.submission or os.path.join(os.path.dirname(args.report), "submission.json")
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        argv = [
            "finalize.py",
            "--data", args.data,
            "--solutions", args.solutions,
            "--outputs", tmp,
            "--submission", sub,
            "--report", args.report,
        ]
        if args.keys:
            argv += ["--keys", args.keys]
        sys.argv = argv
        finalize_main()
        meta = {
            "source": args.outputs,
            "geos": args.geos,
            "n_eval_aug": args.n_eval_aug,
            "n_pickles_kept": len(kept),
            "geo_indices": geo_indices(args.geos),
        }
        with open(args.report) as f:
            rep = json.load(f)
        rep["view_subset"] = meta
        with open(args.report, "w") as f:
            json.dump(rep, f, indent=1)
        print(f"view_subset geos={args.geos} kept={len(kept)} "
              f"score={rep.get('score')} / {rep.get('n_tasks')}", flush=True)
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
