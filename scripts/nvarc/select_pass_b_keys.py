"""Choose pass-B tasks from pass-A pickle confidence (kgmon vote count).

A task is skipped when every test output already has a leading candidate
with at least --min-votes agreeing decode views. Empty or low-vote outputs
stay in pass B. TTT is per-task, so one weak test output pulls the whole
task back into B.

Optional --outputs-b + --solutions scores three policies on an existing
pair of pickle dirs (no GPU): A-only, uniform A+B, and confidence A+B
(B pickles used only on the selected keys).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from arc_decoder import ArcDecoder, hashable, score_kgmon
from arc_loader import ArcDataset


def top_votes(samples: dict) -> int:
    if not samples:
        return 0
    groups: dict[tuple, int] = {}
    for g in samples.values():
        h = hashable(g["solution"])
        groups[h] = groups.get(h, 0) + 1
    return max(groups.values()) if groups else 0


def task_confidence(data: ArcDataset, decoder: ArcDecoder) -> dict[str, dict]:
    rows = {}
    for task in data.keys:
        n_test = len(data.queries[task]["test"])
        votes = []
        for i in range(n_test):
            bk = f"{task}_{i}"
            votes.append(top_votes(decoder.decoded_results.get(bk, {})))
        rows[task] = {
            "n_test": n_test,
            "votes": votes,
            "min_votes": min(votes) if votes else 0,
            "mean_votes": float(np.mean(votes)) if votes else 0.0,
            "empty": sum(1 for v in votes if v == 0),
        }
    return rows


def select_keys(conf: dict[str, dict], min_votes: int) -> tuple[list[str], list[str]]:
    run, skip = [], []
    for task, row in sorted(conf.items()):
        if row["min_votes"] >= min_votes:
            skip.append(task)
        else:
            run.append(task)
    return run, skip


def kgmon_on(samples: dict):
    if not samples:
        return []
    return score_kgmon(samples)


def score_policy(data: ArcDataset, selected: dict) -> float:
    submission = data.get_submission(selected)
    return float(data.validate_submission(submission))


def samples_a_only(samples: dict) -> dict:
    return {k: v for k, v in samples.items() if ".b" not in k}


def build_selected(decoder: ArcDecoder, data: ArcDataset, run_tasks: set[str] | None):
    """If run_tasks is None, use every sample (uniform pool). Else B samples
    only count on tasks in run_tasks; skipped tasks use A samples only."""
    selected = {}
    dm = data.split_multi_replies()
    for task in data.keys:
        for i in range(len(data.queries[task]["test"])):
            bk = f"{task}_{i}"
            samples = decoder.decoded_results.get(bk, {})
            if run_tasks is not None and task not in run_tasks:
                samples = samples_a_only(samples)
            selected[bk] = kgmon_on(samples)
    return selected, dm


def task_scores(data: ArcDataset, selected: dict) -> dict[str, float]:
    per: dict[str, list[bool]] = {}
    replies = data.split_multi_replies().replies
    for bk, gold in replies.items():
        task = bk.split("_")[0]
        guesses = selected.get(bk, [])
        ok = any(np.array_equal(g, gold[0]) for g in guesses[:2])
        per.setdefault(task, []).append(bool(ok))
    return {t: sum(v) / len(v) for t, v in per.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", required=True, help="pass-A pickle dir")
    ap.add_argument("--data", required=True)
    ap.add_argument("--solutions", default="")
    ap.add_argument("--min-votes", type=int, default=8)
    ap.add_argument("--keys-file", default="", help="write pass-B task ids, one per line")
    ap.add_argument("--report", default="", help="json report path")
    ap.add_argument("--outputs-b", default="", help="existing pass-B pickles for offline scoring")
    args = ap.parse_args()

    data = ArcDataset.from_file(args.data)
    if args.solutions:
        data.load_replies(args.solutions)

    dec_a = ArcDecoder(data.split_multi_replies(), n_guesses=2)
    dec_a.load_decoded_results(args.outputs)
    conf = task_confidence(data, dec_a)
    run, skip = select_keys(conf, args.min_votes)

    gold = {}
    if args.solutions:
        sel_a, _ = build_selected(dec_a, data, run_tasks=set())
        gold = task_scores(data, sel_a)

    buckets = defaultdict(int)
    for row in conf.values():
        v = row["min_votes"]
        if v <= 0:
            buckets["0 empty"] += 1
        elif v <= 2:
            buckets["1-2"] += 1
        elif v <= 4:
            buckets["3-4"] += 1
        elif v <= 7:
            buckets["5-7"] += 1
        else:
            buckets["8+"] += 1

    print(f"min_votes>={args.min_votes}: run_B={len(run)} skip={len(skip)}")
    print("task min_votes buckets:", dict(buckets))
    if gold:
        skip_solved = sum(1 for t in skip if gold.get(t, 0) >= 1)
        skip_partial = sum(1 for t in skip if 0 < gold.get(t, 0) < 1)
        skip_zero = sum(1 for t in skip if gold.get(t, 0) == 0)
        print(f"skip set vs A gold: full={skip_solved} partial={skip_partial} zero={skip_zero}")

    compare = {}
    if args.outputs_b and args.solutions:
        pooled = ArcDecoder(data.split_multi_replies(), n_guesses=2)
        pooled.load_decoded_results(args.outputs)
        pooled.load_decoded_results(args.outputs_b, run_name=".b")
        sel_a, _ = build_selected(pooled, data, run_tasks=set())
        sel_u, _ = build_selected(pooled, data, run_tasks=None)
        sel_c, _ = build_selected(pooled, data, run_tasks=set(run))
        sa, su, sc = score_policy(data, sel_a), score_policy(data, sel_u), score_policy(data, sel_c)
        a_tasks = task_scores(data, sel_a)
        u_tasks = task_scores(data, sel_u)
        # B-unique wins among skipped tasks (would be lost by confidence B)
        lost = sorted(
            t for t in skip
            if u_tasks.get(t, 0) > a_tasks.get(t, 0)
        )
        gained_only_on_run = sorted(
            t for t in run
            if u_tasks.get(t, 0) > a_tasks.get(t, 0)
        )
        print(f"A-only {sa:.3f}/{len(data.keys)} = {100*sa/len(data.keys):.2f}%")
        print(f"uniform A+B {su:.3f}/{len(data.keys)} = {100*su/len(data.keys):.2f}%")
        print(f"confidence A+B {sc:.3f}/{len(data.keys)} = {100*sc/len(data.keys):.2f}%")
        print(f"delta confidence-vs-uniform {sc-su:+.3f}  B-unique-on-skip {len(lost)} {lost}")
        print(f"B-unique-on-runB {len(gained_only_on_run)}")
        compare = {
            "a_only": sa,
            "uniform": su,
            "confidence": sc,
            "lost_on_skip": lost,
            "b_unique_on_run": gained_only_on_run,
        }

    if args.keys_file:
        Path(args.keys_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.keys_file).write_text("".join(t + "\n" for t in run))
        print(f"wrote {len(run)} keys -> {args.keys_file}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps({
            "min_votes": args.min_votes,
            "n_run": len(run),
            "n_skip": len(skip),
            "run": run,
            "skip": skip,
            "buckets": dict(buckets),
            "confidence": conf,
            "a_gold": gold,
            "compare": compare,
        }, indent=2) + "\n")
        print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
