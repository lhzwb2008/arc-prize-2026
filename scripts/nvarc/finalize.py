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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--solutions", default="")
    ap.add_argument("--outputs", required=True)
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

    submission = data.get_submission(decoder.run_selection_algo())
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

    # per-task report (task-level score = mean over its test outputs, like the Kaggle metric)
    selected = decoder.run_selection_algo(score_kgmon)
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
