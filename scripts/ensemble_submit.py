#!/usr/bin/env python3
"""Kaggle-style submission: model grid as attempt_1, DSL solver as attempt_2.

Sources
  --dsl           run src/arc_solver (default)
  --model-json    a ttt_qwen.py result file (results[i].pred = list of grids)
  --model-sub     an existing submission.json whose attempt_1 to reuse

If the two attempts coincide, attempt_2 falls back to the DSL's second guess.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc_solver.solver import (  # noqa: E402
    build_submission,
    flatten_challenges,
    valid,
)


def load_challenges(path: Path) -> dict:
    if path.is_dir():
        tasks = {}
        for file in sorted(path.glob("*.json")):
            if file.name.startswith("._"):
                continue
            payload = json.loads(file.read_text())
            tasks.update(flatten_challenges(payload, stem=file.stem))
        return tasks
    return flatten_challenges(json.loads(path.read_text()), stem=path.stem)


def from_ttt_json(path: Path) -> dict[str, list]:
    """task_id -> list of grids (or None)."""
    data = json.loads(path.read_text())
    out = {}
    for rec in data.get("results") or []:
        out[rec["id"]] = rec.get("pred") or []
    return out


def from_submission(path: Path, key: str = "attempt_1") -> dict[str, list]:
    data = json.loads(path.read_text())
    return {tid: [p.get(key) for p in preds] for tid, preds in data.items()}


def copy_grid(g):
    return [row[:] for row in g]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ROOT / "data" / "full" / "data" / "evaluation")
    p.add_argument("--output", type=Path, default=ROOT / "demo" / "output" / "submission.json")
    p.add_argument("--dsl", action="store_true", default=True)
    p.add_argument("--no-dsl", action="store_false", dest="dsl")
    p.add_argument("--time-limit", type=float, default=20.0)
    p.add_argument("--model-json", type=Path, default=None, help="ttt_qwen.py results json")
    p.add_argument("--model-sub", type=Path, default=None, help="submission.json to take attempt_1 from")
    p.add_argument("--score", action="store_true", help="if gold outputs exist, print pass@1 / pass@2")
    args = p.parse_args()

    challenges = load_challenges(args.input)
    dsl = build_submission(challenges, time_limit=args.time_limit, log_every=20) if args.dsl else {}
    model = {}
    if args.model_json:
        model = from_ttt_json(args.model_json)
    elif args.model_sub:
        model = from_submission(args.model_sub)

    submission = {}
    for tid, task in challenges.items():
        n = len(task["test"])
        dsl_preds = dsl.get(tid) or [
            {"attempt_1": copy_grid(t["input"]), "attempt_2": copy_grid(t["input"])} for t in task["test"]
        ]
        model_preds = model.get(tid) or [None] * n
        merged = []
        for k, t in enumerate(task["test"]):
            fallback = copy_grid(t["input"])
            a1_dsl = dsl_preds[k]["attempt_1"] if k < len(dsl_preds) else fallback
            a2_dsl = dsl_preds[k]["attempt_2"] if k < len(dsl_preds) else fallback
            a1_m = model_preds[k] if k < len(model_preds) else None
            a1 = a1_m if valid(a1_m) else a1_dsl
            a2 = a2_dsl if a2_dsl != a1 else a1_dsl
            if a2 == a1:
                a2 = a2_dsl
            merged.append({"attempt_1": a1, "attempt_2": a2})
        submission[tid] = merged

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(submission, separators=(",", ":")))
    print(f"wrote {args.output} tasks={len(submission)}")

    if args.score:
        p1 = p2 = gold_n = 0
        for tid, task in challenges.items():
            for k, t in enumerate(task["test"]):
                gold = t.get("output")
                if gold is None:
                    continue
                gold_n += 1
                a1, a2 = submission[tid][k]["attempt_1"], submission[tid][k]["attempt_2"]
                p1 += int(a1 == gold)
                p2 += int(a1 == gold or a2 == gold)
        if gold_n:
            print(f"pass@1 {p1}/{gold_n} ({100*p1/gold_n:.1f}%)  pass@2 {p2}/{gold_n} ({100*p2/gold_n:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
