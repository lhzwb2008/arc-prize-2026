#!/usr/bin/env python3
"""Official-style ARC-AGI-2 public eval over any OpenAI-compatible chat API.

  python3 scripts/llm_bench/run_eval.py --status
  python3 scripts/llm_bench/run_eval.py --provider dashscope --ids 0520fde7 --limit 1 --num-attempts 1
  python3 scripts/llm_bench/run_eval.py --provider dashscope --effort high --workers 2

Default max_tokens is 131072 so thinking models can still emit the grid
(previous DashScope run died because 16384 tokens were all reasoning).
HTTP timeout defaults to unlimited (--timeout-s 0).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from harness import (  # noqa: E402
    EVAL_DIR,
    GRID_FOLLOWUP,
    _atomic_write,
    _now,
    _read_json,
    acquire_lock,
    build_prompt,
    finalize_rec,
    is_grid,
    load_record,
    load_tasks,
    parse_grid,
    print_status,
    rebuild_outputs,
    save_record,
    task_path,
    write_raw,
)
from provider import make_provider  # noqa: E402

DEFAULT_RUN = HERE.parents[1] / "runs" / "llm_bench_dsv4pro_high"
STOP = False
IO_LOCK = threading.Lock()


def _on_stop(signum, _frame):
    global STOP
    STOP = True
    print(f"\n收到信号 {signum}，当前题结束后退出；下次同一命令续跑", flush=True)


def _call_done(rec: dict, ti: int, key: str, retry_errors: bool) -> bool:
    for call in rec.get("calls") or []:
        if call.get("test_index") != ti or call.get("key") != key:
            continue
        if is_grid(call.get("grid")):
            return True
        if call.get("status") == "ERROR" and retry_errors:
            return False
        if call.get("status") == "FINISHED" and retry_errors and not is_grid(call.get("grid")):
            return False
        return True
    return False


def _replace_call(rec: dict, call: dict) -> None:
    rec["calls"] = [
        c
        for c in (rec.get("calls") or [])
        if not (c.get("test_index") == call["test_index"] and c.get("key") == call["key"])
    ] + [call]


def run_one(provider, run_dir: Path, task_id: str, task: dict, args) -> None:
    rec = load_record(run_dir, task_id)
    if rec.get("status") == "done" and rec.get("attempts") and not args.retry_errors:
        return
    n_test = len(task["test"])
    t0 = time.time()
    if rec.get("elapsed_s") and rec.get("status") in ("running", "error", "parse_fail"):
        rec["elapsed_s_offset"] = float(rec.get("elapsed_s") or 0)
        t0 = time.time()
    rec.update(
        {
            "task_id": task_id,
            "n_test": n_test,
            "model": args.model or provider.model,
            "effort": args.effort,
            "prompt_style": "official_v1",
            "provider": args.provider,
            "max_tokens": args.max_tokens,
            "error": None,
        }
    )
    rec.setdefault("calls", [])
    rec["status"] = "running"
    rec["submitted_at"] = rec.get("submitted_at") or _now()
    with IO_LOCK:
        save_record(run_dir, rec)

    attempt_keys = ("attempt_1", "attempt_2")[: max(1, min(2, args.num_attempts))]
    for ti in range(n_test):
        for key in attempt_keys:
            if STOP:
                rec["status"] = "running"
                with IO_LOCK:
                    save_record(run_dir, rec)
                return
            if _call_done(rec, ti, key, args.retry_errors):
                continue
            prompt = build_prompt(task, ti)
            rec["current_slot"] = f"{ti}:{key}"
            rec["status"] = "running"
            with IO_LOCK:
                save_record(run_dir, rec)
            print(f"  提交 {task_id} {rec['current_slot']}", flush=True)
            call_t0 = time.time()
            text = ""
            reasoning = ""
            usage: dict = {}
            finish = None
            truncated = False
            status = "FINISHED"
            followup = False
            try:
                result = provider.complete_prompt(
                    prompt,
                    max_tokens=args.max_tokens,
                    timeout_s=args.timeout_s,
                    stream=args.stream,
                    enable_thinking=not args.no_thinking,
                    reasoning_effort=args.effort,
                    on_delta=None if args.quiet else (lambda s: sys.stdout.write(s) or sys.stdout.flush()),
                )
                text = result.text or ""
                reasoning = result.reasoning or ""
                usage = result.usage or {}
                finish = result.finish_reason
                truncated = result.truncated
            except Exception as exc:
                status = "ERROR"
                rec["error"] = str(exc)
                print(f"  {task_id} {rec['current_slot']} API 失败: {exc}", flush=True)

            grid = parse_grid(text, reasoning)
            if (
                status == "FINISHED"
                and not is_grid(grid)
                and args.followup
            ):
                followup = True
                print(f"  {task_id} {rec['current_slot']} 无格子，关闭 thinking 追问一次", flush=True)
                try:
                    result2 = provider.complete(
                        [
                            {"role": "user", "content": prompt},
                            {
                                "role": "assistant",
                                "content": (text.strip() or "[reasoning only, truncated]")[:4000],
                            },
                            {"role": "user", "content": GRID_FOLLOWUP},
                        ],
                        max_tokens=min(8192, args.max_tokens),
                        timeout_s=args.timeout_s,
                        stream=False,
                        enable_thinking=False,
                        reasoning_effort=args.effort,
                    )
                    text = (text + "\n\n" + (result2.text or "")).strip()
                    reasoning = (reasoning + "\n\n" + (result2.reasoning or "")).strip()
                    usage = {
                        "first": usage,
                        "followup": result2.usage or {},
                    }
                    finish = result2.finish_reason
                    truncated = bool(truncated or result2.truncated)
                    grid = parse_grid(result2.text, result2.reasoning, text, reasoning)
                except Exception as exc:
                    rec["error"] = str(exc)
                    print(f"  {task_id} {rec['current_slot']} 追问失败: {exc}", flush=True)

            combined = "\n".join(p for p in (text, reasoning) if p)
            raw_rel = write_raw(run_dir, task_id, rec["current_slot"].replace(":", "_"), combined)
            call = {
                "test_index": ti,
                "key": key,
                "status": status,
                "elapsed_s": round(time.time() - call_t0, 2),
                "grid": grid,
                "raw_text": combined[-32000:],
                "raw_file": raw_rel,
                "usage": usage,
                "finish_reason": finish,
                "truncated": truncated,
                "followup": followup,
            }
            _replace_call(rec, call)
            rec["raw_text"] = combined[-32000:]
            with IO_LOCK:
                save_record(run_dir, rec)
            print(
                f"  {task_id} {rec['current_slot']} {status} "
                f"grid={bool(is_grid(grid))} followup={followup} "
                f"{call['elapsed_s']}s finish={finish}",
                flush=True,
            )

    finalize_rec(rec, task, t0)
    with IO_LOCK:
        save_record(run_dir, rec)
    print(
        f"  {task_id} {rec['status']} score={rec.get('task_score')} "
        f"elapsed={rec.get('elapsed_s')}s",
        flush=True,
    )


def _select_tasks(tasks, args):
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        tasks = [t for t in tasks if t[0] in want]
    if args.skip_ids:
        skip = {x.strip() for x in args.skip_ids.split(",") if x.strip()}
        tasks = [t for t in tasks if t[0] not in skip]
    if args.offset:
        tasks = tasks[args.offset :]
    if args.limit:
        tasks = tasks[: args.limit]
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenAI-compat official ARC-AGI-2 chat eval")
    ap.add_argument("--provider", default="dashscope", choices=["dashscope", "openai", "compat"])
    ap.add_argument("--data-dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--model", default="")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-tokens", type=int, default=131072)
    ap.add_argument(
        "--timeout-s",
        type=float,
        default=0.0,
        help="HTTP timeout seconds; 0 means wait until the model finishes",
    )
    ap.add_argument("--num-attempts", type=int, default=2, choices=[1, 2])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--skip-ids", default="")
    ap.add_argument("--reset-ids", default="")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--no-followup", action="store_true")
    ap.add_argument("--retry-errors", action="store_true", default=True)
    ap.add_argument("--no-retry-errors", action="store_false", dest="retry_errors")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--extra-body", default="")
    args = ap.parse_args()
    args.followup = not args.no_followup
    extra = json.loads(args.extra_body) if args.extra_body.strip() else None

    all_tasks = load_tasks(args.data_dir)
    tasks = _select_tasks(all_tasks, args)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.reset_ids:
        for tid in args.reset_ids.split(","):
            tid = tid.strip()
            p = task_path(args.run_dir, tid)
            if p.is_file():
                p.unlink()
                print(f"reset {tid}", flush=True)

    report_tasks = all_tasks if args.data_dir == EVAL_DIR and not args.ids else tasks
    report = rebuild_outputs(args.run_dir, report_tasks)
    if args.status:
        print_status(report)
        return 0

    provider = make_provider(
        args.provider,
        model=args.model or None,
        extra_body=extra,
        stream=True if args.stream else False,
        enable_thinking=not args.no_thinking,
        reasoning_effort=args.effort,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
    )
    args.model = provider.model

    meta = {
        "started_at": _read_json(args.run_dir / "meta.json").get("started_at") or _now(),
        "updated_at": _now(),
        "provider": args.provider,
        "model": provider.model,
        "base_url": provider.base_url,
        "effort": args.effort,
        "max_tokens": args.max_tokens,
        "timeout_s": args.timeout_s,
        "num_attempts": args.num_attempts,
        "prompt_style": "official_v1",
        "followup": args.followup,
        "workers": args.workers,
        "n_selected": len(tasks),
    }
    _atomic_write(args.run_dir / "meta.json", meta)
    print(
        f"llm_bench  provider={args.provider} model={provider.model} "
        f"effort={args.effort} max_tokens={args.max_tokens} "
        f"timeout={'none' if not args.timeout_s else str(args.timeout_s)+'s'} "
        f"attempts={args.num_attempts} workers={args.workers} "
        f"selected={len(tasks)} run_dir={args.run_dir}",
        flush=True,
    )
    print_status(report)

    lock = acquire_lock(args.run_dir)
    signal.signal(signal.SIGINT, _on_stop)
    signal.signal(signal.SIGTERM, _on_stop)

    def should_run(task_id: str) -> bool:
        rec = load_record(args.run_dir, task_id)
        if rec.get("status") == "done" and rec.get("attempts") and not args.retry_errors:
            return False
        if rec.get("status") == "done" and rec.get("attempts") and args.retry_errors:
            # Only rerun done tasks if explicitly reset.
            return False
        if rec.get("status") in ("error", "parse_fail") and not args.retry_errors:
            return False
        return True

    pending = [(tid, task) for tid, task in tasks if should_run(tid)]
    try:
        workers = max(1, args.workers)

        def _job(item):
            i, task_id, task = item
            if STOP:
                return
            print(f"[{i}/{len(pending)}] {task_id} n_test={len(task['test'])}", flush=True)
            run_one(provider, args.run_dir, task_id, task, args)
            with IO_LOCK:
                rep = rebuild_outputs(args.run_dir, report_tasks)
                print_status(rep)

        if workers == 1:
            for i, (task_id, task) in enumerate(pending, start=1):
                if STOP:
                    break
                _job((i, task_id, task))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [
                    pool.submit(_job, (i, task_id, task))
                    for i, (task_id, task) in enumerate(pending, start=1)
                ]
                for fut in as_completed(futs):
                    fut.result()
                    if STOP:
                        break
    finally:
        rebuild_outputs(args.run_dir, report_tasks)
        lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
