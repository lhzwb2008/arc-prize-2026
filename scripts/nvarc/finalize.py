"""Local port of the NVARC notebook's last cell: pickles -> submission.json (+ score).

    python finalize.py --data .../arc-agi_evaluation_challenges.json \
        --solutions .../arc-agi_evaluation_solutions.json \
        --outputs /opt/work/nvarc/eval120/outputs --submission /opt/work/nvarc/eval120/submission.json

Prints per-task correctness so we can diff runs and compute the score over the
full task list (tasks with no pickle count as wrong), not only over decoded ones.
"""
import argparse
import json
import os

import numpy as np

from arc_loader import ArcDataset
from arc_decoder import (
    ArcDecoder,
    merge_keep_primary,
    merge_pass_pair,
    score_kgmon,
    score_mean_quality,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--solutions", default="")
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--outputs-extra", action="append", default=[],
                    help="extra pickle dirs (repeatable); default merge is A_top1+B_top1")
    ap.add_argument("--keep-primary", action="store_true",
                    help="legacy: mixed mean_quality then force --outputs top-1")
    ap.add_argument("--pool-mode", default="",
                    help="pair (default), mixed, or keep-primary")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--report", default="", help="optional json with per-task results")
    ap.add_argument("--keys", default="", help="comma list of task ids (default: all in --data)")
    args = ap.parse_args()

    keys = [k.strip() for k in args.keys.split(",") if k.strip()] or None
    data = ArcDataset.from_file(args.data, keys=keys)
    if args.solutions:
        data.load_replies(args.solutions)

    dm = data.split_multi_replies()
    decoder_a = ArcDecoder(dm, n_guesses=2)
    decoder_a.load_decoded_results(args.outputs)
    sel_a = decoder_a.run_selection_algo(score_mean_quality) if decoder_a.decoded_results else {}

    decoder_b = ArcDecoder(dm, n_guesses=2)
    for i, extra in enumerate(args.outputs_extra, 1):
        n_before = sum(len(v) for v in decoder_b.decoded_results.values())
        decoder_b.load_decoded_results(extra, run_name=f".p{i}")
        n_after = sum(len(v) for v in decoder_b.decoded_results.values())
        print(f"pass-B extra {extra}: +{n_after - n_before} samples")
    sel_b = decoder_b.run_selection_algo(score_mean_quality) if decoder_b.decoded_results else {}

    decoder = ArcDecoder(dm, n_guesses=2)
    decoder.decoded_results = {
        bk: {**(decoder_a.decoded_results.get(bk) or {}), **(decoder_b.decoded_results.get(bk) or {})}
        for bk in set(decoder_a.decoded_results) | set(decoder_b.decoded_results)
    }

    mode = "keep-primary" if args.keep_primary else (args.pool_mode or os.getenv("NVARC_POOL_MODE") or "pair")
    mode = mode.strip().lower().replace("_", "-")
    if mode in ("keep", "keep-primary"):
        selected = merge_keep_primary(sel_a, decoder.run_selection_algo(score_mean_quality))
    elif mode in ("mixed", "pool", "mean-quality"):
        selected = decoder.run_selection_algo(score_mean_quality)
        print("pool_mode=mixed (mean_quality over A+B samples)")
    else:
        selected = merge_pass_pair(sel_a, sel_b)
        mode = "pair"
    print(f"pool_mode={mode}")

    submission = data.get_submission(selected)
    with open(args.submission, "w") as f:
        json.dump(submission, f)

    n_tasks = len(data.keys)
    decoded_tasks = {bk.split("_")[0] for bk in decoder.decoded_results}
    print(f"tasks {n_tasks} decoded {len(decoded_tasks)} submission tasks {len(submission)}")

    if not args.solutions:
        return

    try:
        decoder.benchmark_selection_algos()
    except Exception as e:  # e.g. np.max on empty list when nothing is correct
        print(f"benchmark_selection_algos failed: {e!r}")

    with open(args.submission) as f:
        reload_submission = json.load(f)
    score = data.validate_submission(reload_submission)
    print(f"*** Reload score: {score}  ({score}/{n_tasks} = {100.0*score/n_tasks:.2f}%)")

    per_task = {}
    for bk, correct in data.split_multi_replies().replies.items():
        task = bk.split("_")[0]
        guesses = selected.get(bk, [])
        ok = any(np.array_equal(g, correct[0]) for g in guesses[:2])
        per_task.setdefault(task, []).append(bool(ok))
    report = {t: sum(v) / len(v) for t, v in per_task.items()}
    solved = sorted(t for t, s in report.items() if s > 0)
    print(f"solved tasks ({len(solved)}): {' '.join(solved)}")
    if args.report:
        with open(args.report, "w") as f:
            json.dump({"score": score, "n_tasks": n_tasks, "decoded_tasks": sorted(decoded_tasks),
                       "per_task": report}, f, indent=1)


if __name__ == "__main__":
    main()
