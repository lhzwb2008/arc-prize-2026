#!/usr/bin/env python3
"""Score the solver on the official ARC-AGI-2 training / evaluation sets (data/full).

Usage:
    python3 scripts/eval_full.py --split evaluation
    python3 scripts/eval_full.py --split training --limit 300 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc_solver.solver import score_task  # noqa: E402


def _run(item):
    task_id, task, time_limit = item
    t0 = time.time()
    try:
        res = score_task(task, time_limit=time_limit)
        hits = sum(1 for p in res["per_test"] if p["hit"])
        names = [p["names"] for p in res["per_test"]]
        return task_id, res["solved"], hits, len(res["per_test"]), time.time() - t0, names, None
    except Exception as exc:  # noqa: BLE001
        return task_id, False, 0, len(task["test"]), time.time() - t0, [], repr(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--show-solved", action="store_true")
    args = parser.parse_args()

    folder = ROOT / "data" / "full" / "data" / args.split
    files = sorted(folder.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    items = [(f.stem, json.loads(f.read_text()), args.time_limit) for f in files]

    t0 = time.time()
    solved = 0
    test_hits = 0
    test_total = 0
    slow = []
    errors = []
    solved_ids = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_run, it) for it in items]
        for n, fut in enumerate(as_completed(futures), start=1):
            task_id, ok, hits, total, secs, names, err = fut.result()
            solved += int(ok)
            test_hits += hits
            test_total += total
            if ok:
                solved_ids.append((task_id, names))
            if secs > args.time_limit:
                slow.append((task_id, secs))
            if err:
                errors.append((task_id, err))
            if n % 50 == 0:
                print(f"  {n}/{len(items)}  solved={solved}  elapsed={time.time() - t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\n{args.split}: tasks solved {solved}/{len(items)} ({100 * solved / max(len(items), 1):.1f}%)")
    print(f"test outputs hit {test_hits}/{test_total} ({100 * test_hits / max(test_total, 1):.1f}%)")
    print(f"wall {elapsed:.0f}s, avg {elapsed / max(len(items), 1):.2f}s/task with {args.workers} workers")
    if slow:
        print(f"over time limit: {len(slow)} e.g. {slow[:5]}")
    if errors:
        print(f"errors: {len(errors)} e.g. {errors[:3]}")
    if args.show_solved:
        for task_id, names in sorted(solved_ids):
            print(f"  {task_id}: {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
