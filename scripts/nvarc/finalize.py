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
from arc_decoder import ArcDecoder, score_kgmon


def merge_keep_primary(sel_a, sel_p):
    """Pooled ranking, but pass-A top-1 is always one of the two attempts."""
    selected = {}
    n_forced = 0
    for bk in set(sel_a) | set(sel_p):
        a1 = (sel_a.get(bk) or [None])[0]
        top = list(sel_p.get(bk) or [])[:2]
        if a1 is not None and not any(np.array_equal(a1, g) for g in top):
            top = (top[:1] + [a1]) if top else [a1]
            n_forced += 1
        selected[bk] = top
    print(f"keep-primary: forced pass-A top-1 back on {n_forced} outputs")
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--solutions", default="")
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--outputs-extra", action="append", default=[],
                    help="extra pickle dirs to pool (repeatable); use with --keep-primary")
    ap.add_argument("--keep-primary", action="store_true",
                    help="force --outputs top-1 to remain among the two attempts")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--report", default="", help="optional json with per-task results")
    ap.add_argument("--keys", default="", help="comma list of task ids (default: all in --data)")
    args = ap.parse_args()

    keys = [k.strip() for k in args.keys.split(",") if k.strip()] or None
    data = ArcDataset.from_file(args.data, keys=keys)
    if args.solutions:
        data.load_replies(args.solutions)

    decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
    decoder.load_decoded_results(args.outputs)
    sel_primary = decoder.run_selection_algo(score_kgmon) if args.keep_primary else None
    for i, extra in enumerate(args.outputs_extra, 1):
        n_before = sum(len(v) for v in decoder.decoded_results.values())
        decoder.load_decoded_results(extra, run_name=f".p{i}")
        n_after = sum(len(v) for v in decoder.decoded_results.values())
        print(f"pooled extra {extra}: +{n_after - n_before} samples")

    selected = decoder.run_selection_algo(score_kgmon)
    if sel_primary is not None:
        selected = merge_keep_primary(sel_primary, selected)

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
