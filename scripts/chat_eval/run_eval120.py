#!/usr/bin/env python3
"""ARC-AGI-2 public-120 eval via DashScope (百炼) chat completions.

Official-style: one test input per call, pass@2 = two independent samples,
parse last grid. No tools, no Cloud Agent.

  python3 scripts/chat_eval/run_eval120.py --status
  python3 scripts/chat_eval/run_eval120.py --limit 1 --ids 0520fde7 --data-dir data/examples
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts" / "cursor_cloud"))

from dashscope_client import (  # noqa: E402
    chat_complete,
    default_model,
    load_dashscope_env,
)
from run_eval120 import (  # noqa: E402
    EVAL_DIR,
    _now,
    _atomic_write,
    _read_json,
    acquire_lock,
    build_prompt,
    finalize_rec,
    gold_outputs,
    is_grid,
    load_record,
    load_tasks,
    on_stop,
    parse_official_grid,
    print_status,
    rebuild_outputs,
    save_record,
    score_task,
    task_path,
)

DEFAULT_RUN = ROOT / "runs" / "chat_eval120_dsv4pro"

STOP = False


def _on_stop(signum, frame):
    global STOP
    STOP = True
    on_stop(signum, frame)


def run_one(
    run_dir: Path,
    task_id: str,
    task: dict,
    *,
    model: str,
    effort: str,
    timeout_s: float,
    num_attempts: int,
    quiet: bool,
) -> None:
    rec = load_record(run_dir, task_id)
    if rec.get("status") == "done" and rec.get("attempts"):
        return

    n_test = len(task["test"])
    t0 = time.time()
    rec.update({
        "task_id": task_id,
        "n_test": n_test,
        "model": model,
        "effort": effort,
        "prompt_style": "official_v1",
        "provider": "dashscope",
        "error": None,
    })
    rec.setdefault("calls", [])
    rec.setdefault("tool_calls", [])
    save_record(run_dir, rec)

    attempt_keys = ("attempt_1", "attempt_2")[: max(1, min(2, num_attempts))]
    last_text = rec.get("raw_text") or ""
    last_status = rec.get("run_status") or "FINISHED"

    def has_call(ti: int, key: str) -> bool:
        return any(
            c.get("test_index") == ti and c.get("key") == key
            for c in rec.get("calls") or []
        )

    for ti in range(n_test):
        for key in attempt_keys:
            if STOP:
                rec["status"] = "running"
                save_record(run_dir, rec)
                return
            if has_call(ti, key):
                continue
            prompt = build_prompt(task, ti, cloud_agent=False)
            rec["current_slot"] = f"{ti}:{key}"
            rec["status"] = "running"
            rec["submitted_at"] = rec.get("submitted_at") or _now()
            save_record(run_dir, rec)
            print(f"  提交 {task_id} {rec['current_slot']}", flush=True)
            call_t0 = time.time()
            try:
                result = chat_complete(
                    prompt,
                    model=model,
                    reasoning_effort=effort,
                    timeout_s=timeout_s,
                    stream=False,
                    on_delta=None if quiet else (lambda s: sys.stdout.write(s) or sys.stdout.flush()),
                )
                last_text = result["text"]
                last_status = "FINISHED"
                usage = result.get("usage") or {}
            except Exception as exc:
                last_text = ""
                last_status = "ERROR"
                usage = {}
                rec["error"] = str(exc)
                print(f"  {task_id} {rec['current_slot']} API 失败: {exc}", flush=True)

            grid = parse_official_grid(last_text)
            call = {
                "test_index": ti,
                "key": key,
                "status": last_status,
                "elapsed_s": round(time.time() - call_t0, 2),
                "grid": grid,
                "raw_text": (last_text or "")[-20000:],
                "usage": usage,
                "tools": [],
            }
            rec["calls"] = [
                c for c in (rec.get("calls") or [])
                if not (c.get("test_index") == ti and c.get("key") == key)
            ] + [call]
            rec["raw_text"] = last_text
            save_record(run_dir, rec)
            print(
                f"  {task_id} {rec['current_slot']} {last_status} "
                f"grid={bool(grid)} {call['elapsed_s']}s",
                flush=True,
            )

    attempts = []
    any_grid = False
    for ti in range(n_test):
        a1 = next((c["grid"] for c in rec["calls"] if c.get("test_index") == ti and c.get("key") == "attempt_1" and is_grid(c.get("grid"))), None)
        a2 = next((c["grid"] for c in rec["calls"] if c.get("test_index") == ti and c.get("key") == "attempt_2" and is_grid(c.get("grid"))), None)
        if is_grid(a1) or is_grid(a2):
            any_grid = True
        a1 = a1 if is_grid(a1) else (a2 if is_grid(a2) else [[0]])
        a2 = a2 if is_grid(a2) else a1
        attempts.append({"attempt_1": a1, "attempt_2": a2})
    if not any_grid:
        attempts = None

    complete = attempts is not None and all(
        is_grid(item.get("attempt_1")) and item["attempt_1"] != [[0]]
        or is_grid(next((c.get("grid") for c in rec["calls"] if c.get("test_index") == i), None))
        for i, item in enumerate(attempts or [])
    )
    # Simpler complete check: every test has a real parsed grid.
    complete = True
    if attempts is None:
        complete = False
    else:
        for ti in range(n_test):
            if not any(
                c.get("test_index") == ti and is_grid(c.get("grid"))
                for c in rec["calls"]
            ):
                complete = False
                break

    finalize_rec(
        rec, task, last_text,
        "FINISHED" if complete else last_status,
        [], t0, attempts=attempts,
    )
    rec["elapsed_s"] = round(time.time() - t0, 2)
    save_record(run_dir, rec)
    print(
        f"  {task_id} {rec['status']} score={rec.get('task_score')} "
        f"elapsed={rec.get('elapsed_s')}s",
        flush=True,
    )


def main() -> int:
    load_dashscope_env()
    ap = argparse.ArgumentParser(description="DashScope chat public-120 eval")
    ap.add_argument("--data-dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--model", default=os.environ.get("DASHSCOPE_ARC_MODEL") or default_model())
    ap.add_argument("--effort", default="high", choices=["low", "high", "max"])
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    ap.add_argument("--num-attempts", type=int, default=2, choices=[1, 2])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--skip-ids", default="")
    ap.add_argument("--reset-ids", default="")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--retry-errors", action="store_true")
    args = ap.parse_args()

    tasks = load_tasks(args.data_dir)
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        tasks = [t for t in tasks if t[0] in want]
    if args.skip_ids:
        skip = {x.strip() for x in args.skip_ids.split(",") if x.strip()}
        tasks = [t for t in tasks if t[0] not in skip]
    if args.offset:
        tasks = tasks[args.offset:]
    if args.limit:
        tasks = tasks[: args.limit]

    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.reset_ids:
        for tid in args.reset_ids.split(","):
            tid = tid.strip()
            p = task_path(args.run_dir, tid)
            if p.is_file():
                p.unlink()
                print(f"reset {tid}", flush=True)

    report = rebuild_outputs(
        args.run_dir,
        load_tasks(args.data_dir) if args.data_dir == EVAL_DIR else tasks,
    )
    if args.status:
        print_status(report)
        return 0

    meta = {
        "started_at": _read_json(args.run_dir / "meta.json").get("started_at") or _now(),
        "updated_at": _now(),
        "provider": "dashscope",
        "model": args.model,
        "effort": args.effort,
        "num_attempts": args.num_attempts,
        "prompt_style": "official_v1",
        "n_selected": len(tasks),
    }
    _atomic_write(args.run_dir / "meta.json", meta)
    print(
        f"DashScope chat eval  model={args.model} effort={args.effort} "
        f"attempts={args.num_attempts} selected={len(tasks)} run_dir={args.run_dir}",
        flush=True,
    )
    print_status(report)

    lock = acquire_lock(args.run_dir)
    signal.signal(signal.SIGINT, _on_stop)
    signal.signal(signal.SIGTERM, _on_stop)
    try:
        for i, (task_id, task) in enumerate(tasks, start=1):
            if STOP:
                break
            rec = load_record(args.run_dir, task_id)
            if rec.get("status") == "done" and rec.get("attempts"):
                continue
            if rec.get("status") in ("error", "parse_fail") and not args.retry_errors:
                continue
            print(f"[{i}/{len(tasks)}] {task_id} n_test={len(task['test'])}", flush=True)
            run_one(
                args.run_dir, task_id, task,
                model=args.model,
                effort=args.effort,
                timeout_s=args.timeout_s,
                num_attempts=args.num_attempts,
                quiet=args.quiet,
            )
            report = rebuild_outputs(
                args.run_dir,
                load_tasks(args.data_dir) if args.data_dir == EVAL_DIR else tasks,
            )
            print_status(report)
    finally:
        rebuild_outputs(
            args.run_dir,
            load_tasks(args.data_dir) if args.data_dir == EVAL_DIR else tasks,
        )
        lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
