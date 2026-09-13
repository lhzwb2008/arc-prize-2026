#!/usr/bin/env python3
"""Resume-safe ARC-AGI-2 public-120 eval via Cursor Cloud Agents.

Prompt/scoring follow arcprize/arc-agi-benchmarking: official system_prompt.txt,
one test input per call, pass@2 as a second independent call, parse last grid.

Usage:
  python3 scripts/cursor_cloud/run_eval120.py              # start / continue
  python3 scripts/cursor_cloud/run_eval120.py --status     # print progress
  python3 scripts/cursor_cloud/run_eval120.py --limit 1    # smoke one task

Closing the laptop stops this process, not the cloud run. Re-run the same
command: finished tasks are skipped, in-flight runs are polled by saved ids.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from cursor_client import (  # noqa: E402
    cancel_run,
    create_agent,
    create_run,
    default_model_id,
    load_cursor_env,
    run_with_stream,
    sandbox_repo_url,
)

EVAL_DIR = ROOT / "data" / "full" / "data" / "evaluation"
DEFAULT_RUN = ROOT / "runs" / "cursor_cloud_eval120"

CONTAM_NEEDLES = (
    "arcprize.org",
    "arc-agi-2",
    "arc-agi_evaluation",
    "evaluation_solutions",
    "github.com/arcprize",
    "codeload.github.com/arcprize",
)

STOP = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def load_tasks(data_dir: Path) -> list[tuple[str, dict]]:
    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"没有题目：{data_dir}")
    out = []
    for path in files:
        task = json.loads(path.read_text(encoding="utf-8"))
        out.append((path.stem, task))
    return out


def fmt_grid(grid: list[list[int]]) -> str:
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


# Official ARC Prize harness (arc-agi-benchmarking system_prompt.txt).
# One test input per call; pass@2 is two independent calls, not two hypotheses
# in one JSON blob.
_OFFICIAL_TEMPLATE = """You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.

Respond in the format of the training output examples

--Training Examples--
{training_examples}
--End of Training Examples--

--Test Input--
{test_input}
--End of Test Input--

Your response:"""


def build_prompt(
    task: dict,
    test_index: int = 0,
    n_test: int | None = None,
    *,
    cloud_agent: bool = False,
) -> str:
    del n_test
    training = []
    for i, pair in enumerate(task["train"]):
        training.append(f"--Example {i}-- \n\n INPUT: \n\n")
        training.append(json.dumps(pair["input"]))
        training.append("\n\nOUTPUT: \n\n")
        training.append(json.dumps(pair["output"]))
        training.append("\n\n")
    test_input = json.dumps(task["test"][test_index]["input"])
    official = _OFFICIAL_TEMPLATE.format(
        training_examples="".join(training),
        test_input=test_input,
    )
    if cloud_agent:
        return (
            official
            + "\nDo not use tools, files, terminal, or search. Do not write Python. "
            "Think, then output only the grid as a JSON list of lists.\n"
        )
    return official


def is_grid(obj: Any) -> bool:
    if not isinstance(obj, list) or not obj:
        return False
    width = None
    for row in obj:
        if not isinstance(row, list) or not row:
            return False
        if width is None:
            width = len(row)
        if len(row) != width or not (1 <= len(row) <= 30):
            return False
        for cell in row:
            if not isinstance(cell, int) or isinstance(cell, bool) or not (0 <= cell <= 9):
                return False
    return 1 <= len(obj) <= 30


def grids_equal(a: Any, b: Any) -> bool:
    return is_grid(a) and is_grid(b) and a == b


def _extract_json_objects(text: str) -> list[Any]:
    blobs: list[str] = []
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.I):
        blobs.append(m.group(1))
    blobs.append(text)
    found: list[Any] = []
    for blob in blobs:
        start = blob.find("{")
        while start != -1:
            depth = 0
            for i, ch in enumerate(blob[start:], start=start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = blob[start : i + 1]
                        try:
                            found.append(json.loads(chunk))
                        except json.JSONDecodeError:
                            pass
                        break
            start = blob.find("{", start + 1)
    return found


def backscan_json_grid(text: str) -> Any:
    """Official arc-agi-benchmarking backscan: last JSON list-of-lists."""
    closing = None
    last = -1
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ("]", "}"):
            last = i
            closing = text[i]
            break
    if last < 0 or closing is None:
        return None
    opening = "[" if closing == "]" else "{"
    depth = 1
    start = -1
    for i in range(last - 1, -1, -1):
        ch = text[i]
        if ch == closing:
            depth += 1
        elif ch == opening:
            depth -= 1
            if depth == 0:
                start = i
                break
    if start < 0:
        return None
    try:
        parsed = json.loads(text[start : last + 1])
    except json.JSONDecodeError:
        return None
    if is_grid(parsed):
        return parsed
    if isinstance(parsed, dict):
        for key in ("attempt_1", "output", "grid"):
            if is_grid(parsed.get(key)):
                return parsed[key]
    return None


def parse_space_separated_grid(text: str) -> Any:
    best = None
    rows: list[list[int]] = []
    for line in text.splitlines():
        s = line.strip()
        if re.fullmatch(r"[0-9](?:[ \t]+[0-9])*", s):
            rows.append([int(x) for x in s.split()])
            continue
        if rows and is_grid(rows):
            best = rows
        rows = []
    if rows and is_grid(rows):
        best = rows
    return best


def parse_official_grid(text: str) -> Any:
    """Parse one output grid the way the official harness does."""
    if not text or not text.strip():
        return None
    boxed = re.search(r"\\boxed\{(.*?)\}", text, flags=re.DOTALL)
    if boxed:
        try:
            parsed = json.loads(boxed.group(1).strip())
            if is_grid(parsed):
                return parsed
        except json.JSONDecodeError:
            pass
    grid = backscan_json_grid(text)
    if grid is not None:
        return grid
    old = parse_attempts(text, 1)
    if old and is_grid(old[0].get("attempt_1")):
        return old[0]["attempt_1"]
    return parse_space_separated_grid(text)


def parse_attempts(text: str, n_test: int) -> list[dict] | None:
    if not text or not text.strip():
        return None
    for obj in reversed(_extract_json_objects(text)):
        if not isinstance(obj, dict):
            continue
        tests = obj.get("test")
        if tests is None and is_grid(obj.get("attempt_1")):
            tests = [obj]
        if tests is None and is_grid(obj.get("output")):
            tests = [{"attempt_1": obj["output"]}]
        if not isinstance(tests, list) or not tests:
            continue
        out: list[dict] = []
        ok = True
        for i in range(n_test):
            item = tests[i] if i < len(tests) and isinstance(tests[i], dict) else {}
            a1 = item.get("attempt_1") or item.get("output")
            a2 = item.get("attempt_2")
            if not is_grid(a1):
                ok = False
                break
            if not is_grid(a2):
                a2 = a1
            out.append({"attempt_1": a1, "attempt_2": a2})
        if ok and len(out) == n_test:
            return out
    return None


def gold_outputs(task: dict) -> list[list[list[int]]]:
    gold = []
    for item in task["test"]:
        if "output" not in item:
            raise RuntimeError("评测文件缺少 test.output，无法本地打分")
        gold.append(item["output"])
    return gold


def score_task(attempts: list[dict] | None, gold: list[list[list[int]]]) -> tuple[float, list[float]]:
    per: list[float] = []
    for i, g in enumerate(gold):
        item = attempts[i] if attempts and i < len(attempts) else {}
        hit = grids_equal(item.get("attempt_1"), g) or grids_equal(item.get("attempt_2"), g)
        per.append(1.0 if hit else 0.0)
    return (sum(per) / len(per) if per else 0.0), per


def tool_contaminated(tool_calls: list[dict]) -> bool:
    blob = json.dumps(tool_calls, ensure_ascii=False).lower()
    return any(n in blob for n in CONTAM_NEEDLES)


def task_path(run_dir: Path, task_id: str) -> Path:
    return run_dir / "tasks" / f"{task_id}.json"


def load_record(run_dir: Path, task_id: str) -> dict:
    rec = _read_json(task_path(run_dir, task_id))
    rec.setdefault("task_id", task_id)
    rec.setdefault("status", "pending")
    return rec


def save_record(run_dir: Path, rec: dict) -> None:
    _atomic_write(task_path(run_dir, rec["task_id"]), rec)


def placeholder_submission(tasks: list[tuple[str, dict]]) -> dict:
    sub: dict[str, list[dict]] = {}
    for task_id, task in tasks:
        n = len(task["test"])
        sub[task_id] = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in range(n)]
    return sub


def rebuild_outputs(run_dir: Path, tasks: list[tuple[str, dict]]) -> dict:
    submission = placeholder_submission(tasks)
    rows = []
    done = 0
    scored = 0.0
    honest = 0.0
    honest_n = 0
    contaminated_n = 0
    elapsed = 0.0
    for task_id, task in tasks:
        rec = load_record(run_dir, task_id)
        gold = gold_outputs(task)
        attempts = rec.get("attempts")
        if rec.get("status") in {"done", "parse_fail", "error", "timeout"} and attempts:
            submission[task_id] = attempts
        score, per = score_task(attempts if isinstance(attempts, list) else None, gold)
        rec_elapsed = float(rec.get("elapsed_s") or 0)
        elapsed += rec_elapsed
        if rec.get("status") == "done":
            done += 1
            scored += score
            if rec.get("contaminated"):
                contaminated_n += 1
            else:
                honest += score
                honest_n += 1
        rows.append(
            {
                "task_id": task_id,
                "status": rec.get("status"),
                "score": score if rec.get("status") == "done" else None,
                "per_test": per if rec.get("status") == "done" else None,
                "elapsed_s": rec_elapsed,
                "contaminated": bool(rec.get("contaminated")),
                "agent_id": rec.get("agent_id"),
                "run_id": rec.get("run_id"),
            }
        )
    n = len(tasks)
    report = {
        "updated_at": _now(),
        "n_tasks": n,
        "n_done": done,
        "n_pending": sum(1 for r in rows if r["status"] in (None, "pending")),
        "n_running": sum(1 for r in rows if r["status"] == "running"),
        "n_error": sum(1 for r in rows if r["status"] in ("error", "parse_fail", "timeout")),
        "n_contaminated": contaminated_n,
        "score_points": scored,
        "score_pct": 100.0 * scored / n,
        "done_pct": 100.0 * scored / done if done else 0.0,
        "honest_points": honest,
        "honest_pct": 100.0 * honest / n,
        "honest_done_pct": 100.0 * honest / honest_n if honest_n else 0.0,
        "elapsed_s_sum": elapsed,
        "elapsed_s_avg_done": elapsed / done if done else 0.0,
        "eta_s": (elapsed / done * (n - done)) if done else None,
        "tasks": rows,
    }
    _atomic_write(run_dir / "submission.json", submission)
    _atomic_write(run_dir / "report.json", report)
    return report


def print_status(report: dict) -> None:
    eta = report.get("eta_s")
    eta_h = f"{eta / 3600:.1f}h" if isinstance(eta, (int, float)) else "?"
    print(
        f"done {report['n_done']}/{report['n_tasks']}  "
        f"running {report['n_running']}  "
        f"err {report['n_error']}  "
        f"score {report['score_points']:.2f}/120 ({report['score_pct']:.2f}%)  "
        f"on-done {report['done_pct']:.1f}%  "
        f"honest {report['honest_pct']:.2f}%  "
        f"avg {report['elapsed_s_avg_done']:.0f}s/task  eta {eta_h}",
        flush=True,
    )


def acquire_lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / "runner.lock"
    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise SystemExit(f"已有评测进程在跑（锁 {lock_path}）。只看进度用 --status")
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid={os.getpid()} started={_now()}\n")
    fh.flush()
    return fh


def on_stop(signum, _frame):
    global STOP
    STOP = True
    print(f"\n收到信号 {signum}，当前题结束后退出；云端 run 会继续，下次同一命令续跑", flush=True)


PARSE_RETRY = (
    "Your previous reply was not a valid output grid. "
    "Respond with only the output grid as a JSON list of lists of integers, "
    "in the same format as the training OUTPUT examples."
)

NO_TOOLS_RETRY = (
    "Stop. Do not use tools, files, terminal, or search. Do not write Python. "
    "Output only the output grid as a JSON list of lists of integers."
)

_TOOL_NOISE = {"", "none", "tool-call-started", "tool-call-completed"}


def wait_run(
    rec: dict,
    timeout_ms: int,
    quiet: bool,
    stall_ms: int,
    max_task_ms: int,
    *,
    cancel_on_tool: bool = True,
) -> tuple[str, str, list[dict]]:
    tools: list[dict] = []
    seen: dict[str, int] = {}
    cancelled_tool = {"done": False}

    def on_tool(payload: dict) -> None:
        slim = {
            "name": payload.get("name") or payload.get("type"),
            "status": payload.get("status"),
            "args": payload.get("args"),
        }
        tools.append(slim)
        print(f"    tool {slim.get('name')} {slim.get('status')}", flush=True)
        name = str(slim.get("name") or "").strip().lower()
        if (
            cancel_on_tool
            and not cancelled_tool["done"]
            and name not in _TOOL_NOISE
            and rec.get("agent_id")
            and rec.get("run_id")
        ):
            cancelled_tool["done"] = True
            print("    检测到工具调用，取消本 run（逼它只出格子）", flush=True)
            try:
                cancel_run(str(rec["agent_id"]), str(rec["run_id"]))
            except Exception as exc:
                print(f"    cancel 失败: {exc}", flush=True)

    def on_text(delta: str) -> None:
        if not quiet:
            sys.stdout.write(delta)
            sys.stdout.flush()

    def on_thinking(delta: str) -> None:
        if not quiet:
            sys.stdout.write(delta)
            sys.stdout.flush()

    def on_event(name: str, payload) -> None:
        seen[name] = seen.get(name, 0) + 1
        if name in ("error", "sse_error") and seen[name] <= 3:
            print(f"    sse {name}: {json.dumps(payload, ensure_ascii=False)[:300]}", flush=True)
        elif name not in ("heartbeat", "assistant", "thinking", "interaction_update") and seen[name] <= 8:
            print(f"    sse {name} (x{seen[name]})", flush=True)

    text, status = run_with_stream(
        rec["agent_id"],
        rec["run_id"],
        timeout_ms=timeout_ms,
        stall_ms=stall_ms,
        max_task_ms=max_task_ms,
        on_assistant=on_text if not quiet else None,
        on_thinking=on_thinking if not quiet else None,
        on_tool_call=on_tool,
        on_event=on_event,
    )
    rec["sse_events"] = seen
    if cancelled_tool["done"] and status in ("CANCELLED", "FINISHED", "ERROR", "TIMEOUT"):
        # Treat a tool-abort as a failed sample so the caller can re-ask grid-only.
        if not parse_official_grid(text):
            status = "TOOLS"
    return text, status, tools


def ensure_agent_run(
    rec: dict,
    prompt: str,
    *,
    reuse_agent: str | None,
    model: str,
    effort: str,
    fast: bool,
    use_repo: bool,
) -> str | None:
    """Create or reuse. Returns agent_id. Caller must persist rec before wait."""
    if rec.get("agent_id") and rec.get("run_id") and rec.get("status") == "running":
        return rec["agent_id"]
    if reuse_agent:
        run_id = create_run(reuse_agent, prompt)
        rec["agent_id"] = reuse_agent
        rec["run_id"] = run_id
        rec["created_new_agent"] = False
    else:
        agent_id, run_id = create_agent(
            prompt,
            model=model,
            effort=effort,
            fast=fast,
            use_repo=use_repo,
            name="ARC-AGI-2 public eval",
        )
        rec["agent_id"] = agent_id
        rec["run_id"] = run_id
        rec["created_new_agent"] = True
    rec["status"] = "running"
    rec["submitted_at"] = rec.get("submitted_at") or _now()
    return rec["agent_id"]


def finalize_rec(
    rec: dict,
    task: dict,
    text: str,
    status: str,
    tools: list[dict],
    t0: float,
    attempts: list[dict] | None = None,
) -> None:
    rec["raw_text"] = text
    rec["run_status"] = status
    rec["tool_calls"] = (rec.get("tool_calls") or []) + tools
    rec["contaminated"] = tool_contaminated(rec["tool_calls"])
    rec["finished_at"] = _now()
    rec["elapsed_s"] = round(time.time() - t0 + float(rec.get("elapsed_s_offset") or 0), 2)
    n_test = len(task["test"])
    if attempts is None:
        attempts = parse_attempts(text, n_test)
        if attempts is None:
            grid = parse_official_grid(text)
            if is_grid(grid):
                attempts = [{"attempt_1": grid, "attempt_2": grid} for _ in range(n_test)]
    rec["attempts"] = attempts
    gold = gold_outputs(task)
    score, per = score_task(attempts, gold)
    rec["task_score"] = score
    rec["per_test"] = per
    if attempts is None or not all(is_grid((item or {}).get("attempt_1")) for item in attempts):
        rec["status"] = "parse_fail" if status == "FINISHED" else (
            "error" if status in ("SLOW", "STALL", "ERROR") else status.lower()
        )
        rec["error"] = rec.get("error") or f"unparsed run_status={status}"
    elif status == "FINISHED":
        rec["status"] = "done"
        rec["error"] = None
    else:
        rec["status"] = "error" if status in ("SLOW", "STALL", "ERROR") else status.lower()
        rec["error"] = rec.get("error") or f"run_status={status}"


def _remaining_ms(t0: float, max_task_ms: int, offset: float) -> int:
    if max_task_ms <= 0:
        return 10**9
    used = int((time.time() - t0 + offset) * 1000)
    return max(0, max_task_ms - used)


def _call_grid(rec: dict, test_index: int, key: str) -> Any:
    for call in rec.get("calls") or []:
        if call.get("test_index") == test_index and call.get("key") == key and is_grid(call.get("grid")):
            return call["grid"]
    return None


def _store_call(rec: dict, test_index: int, key: str, text: str, status: str, tools: list[dict]) -> dict:
    grid = parse_official_grid(text)
    call_elapsed = None
    try:
        started = datetime.fromisoformat(rec["call_started_at"])
        call_elapsed = round(time.time() - started.timestamp(), 1)
    except Exception:
        pass
    call = {
        "test_index": test_index,
        "key": key,
        "run_id": rec.get("run_id"),
        "status": status,
        "elapsed_s": call_elapsed,
        "grid": grid,
        "raw_text": (text or "")[-20000:],
        "tools": tools,
    }
    calls = [c for c in (rec.get("calls") or []) if not (c.get("test_index") == test_index and c.get("key") == key)]
    calls.append(call)
    rec["calls"] = calls
    rec["raw_text"] = text
    rec["tool_calls"] = (rec.get("tool_calls") or []) + tools
    return call


def _assemble_attempts(rec: dict, n_test: int) -> list[dict] | None:
    """Per-test pass@2 like the official scorer; missing tests get a placeholder."""
    out: list[dict] = []
    any_grid = False
    for ti in range(n_test):
        a1 = _call_grid(rec, ti, "attempt_1")
        a2 = _call_grid(rec, ti, "attempt_2")
        if is_grid(a1) or is_grid(a2):
            any_grid = True
        a1 = a1 if is_grid(a1) else (a2 if is_grid(a2) else [[0]])
        a2 = a2 if is_grid(a2) else a1
        out.append({"attempt_1": a1, "attempt_2": a2})
    return out if any_grid else None


def _start_prompt(
    rec: dict,
    prompt: str,
    *,
    reuse_agent: str | None,
    model: str,
    effort: str,
    fast: bool,
    use_repo: bool,
) -> None:
    rec["run_id"] = None
    rec["status"] = "pending"
    rec["call_started_at"] = _now()
    try:
        ensure_agent_run(
            rec, prompt,
            reuse_agent=reuse_agent,
            model=model, effort=effort, fast=fast, use_repo=use_repo,
        )
        return
    except Exception as exc:
        if not use_repo and not reuse_agent and sandbox_repo_url():
            print(f"  no-repo 失败，改挂沙箱仓库: {exc}", flush=True)
            ensure_agent_run(
                rec, prompt,
                reuse_agent=None,
                model=model, effort=effort, fast=fast, use_repo=True,
            )
            return
        raise


def run_one(
    run_dir: Path,
    task_id: str,
    task: dict,
    *,
    reuse_agent: str | None,
    model: str,
    effort: str,
    fast: bool,
    use_repo: bool,
    timeout_ms: int,
    stall_ms: int,
    max_task_ms: int,
    retries: int,
    quiet: bool,
    num_attempts: int = 1,
) -> str | None:
    rec = load_record(run_dir, task_id)
    if rec.get("status") == "done" and rec.get("attempts"):
        return rec.get("agent_id")

    n_test = len(task["test"])
    t0 = time.time()
    offset = 0.0
    if rec.get("status") == "running" and rec.get("submitted_at"):
        try:
            started = datetime.fromisoformat(rec["submitted_at"])
            offset = max(0.0, time.time() - started.timestamp())
        except Exception:
            offset = 0.0
    rec["elapsed_s_offset"] = offset
    rec["task_id"] = task_id
    rec["n_test"] = n_test
    rec["model"] = model
    rec["effort"] = effort
    rec["fast"] = fast
    rec["prompt_style"] = "official_v1"
    rec["error"] = None
    rec.setdefault("calls", [])
    rec.setdefault("tool_calls", rec.get("tool_calls") or [])
    agent = reuse_agent or rec.get("agent_id")
    last_text = rec.get("raw_text") or ""
    last_status = rec.get("run_status") or "FINISHED"
    last_tools: list[dict] = []

    attempt_keys = ("attempt_1", "attempt_2")[: max(1, min(2, num_attempts))]
    slots = [(ti, key) for ti in range(n_test) for key in attempt_keys]

    # max_task_ms caps each call (one test input, one attempt), like one
    # official API request. Output streaming gets extra grace in the client.
    slow_calls = 0
    if rec.get("status") == "running" and rec.get("agent_id") and rec.get("run_id"):
        slot = rec.get("current_slot") or "0:attempt_1"
        print(f"  续 poll {task_id} slot={slot} agent={rec.get('agent_id')} run={rec.get('run_id')}", flush=True)
        remain = _remaining_ms(time.time(), max_task_ms, offset)
        if max_task_ms > 0:
            remain = max(remain, 30_000)
        last_text, last_status, last_tools = wait_run(rec, timeout_ms, quiet, stall_ms, remain)
        try:
            ti_s, key = slot.split(":", 1)
            _store_call(rec, int(ti_s), key, last_text, last_status, last_tools)
        except ValueError:
            _store_call(rec, 0, "attempt_1", last_text, last_status, last_tools)
        agent = rec.get("agent_id")
        save_record(run_dir, rec)
        if last_status in ("SLOW", "STALL"):
            slow_calls += 1
            rec["error"] = "over_max_task_ms" if last_status == "SLOW" else "stalled_no_sse_activity"
        elif last_status == "TIMEOUT":
            rec["status"] = "running"
            rec["error"] = "local_timeout_will_repoll"
            save_record(run_dir, rec)
            print(f"  {task_id} 本地等待超时，记录仍为 running，下次会继续 poll", flush=True)
            return rec.get("agent_id")

    for ti, key in slots:
        if STOP:
            rec["status"] = "running"
            save_record(run_dir, rec)
            return rec.get("agent_id")
        if _call_grid(rec, ti, key):
            continue
        if key == "attempt_2" and not _call_grid(rec, ti, "attempt_1") and slow_calls:
            # attempt_1 already blew the per-call cap; a second sample would too.
            print(f"  skip {task_id} test{ti} {key}: attempt_1 over cap", flush=True)
            continue

        prompt = build_prompt(task, ti)
        rec["current_slot"] = f"{ti}:{key}"
        # Official pass@2 = two *independent* samples. A second call on the
        # same agent would just see and repeat attempt_1, so attempt_2 always
        # gets a fresh agent; attempt_1 may reuse one per --reuse-every.
        call_agent = agent if key == "attempt_1" else None
        try:
            _start_prompt(
                rec, prompt,
                reuse_agent=call_agent,
                model=model, effort=effort, fast=fast, use_repo=use_repo,
            )
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = str(exc)
            rec["finished_at"] = _now()
            save_record(run_dir, rec)
            print(f"  {task_id} create 失败: {exc}", flush=True)
            return reuse_agent
        save_record(run_dir, rec)
        print(
            f"  提交 {task_id} {rec['current_slot']} agent={rec['agent_id']} run={rec['run_id']}",
            flush=True,
        )
        last_text, last_status, last_tools = wait_run(rec, timeout_ms, quiet, stall_ms, max_task_ms)
        call = _store_call(rec, ti, key, last_text, last_status, last_tools)
        agent = rec.get("agent_id")
        save_record(run_dir, rec)

        if last_status in ("SLOW", "STALL"):
            slow_calls += 1
            rec["error"] = "over_max_task_ms" if last_status == "SLOW" else "stalled_no_sse_activity"
            print(f"  {task_id} {rec['current_slot']} {last_status}, 换下一个 test", flush=True)
            continue
        if last_status == "TIMEOUT":
            rec["status"] = "running"
            rec["error"] = "local_timeout_will_repoll"
            save_record(run_dir, rec)
            print(f"  {task_id} 本地等待超时，记录仍为 running，下次会继续 poll", flush=True)
            return rec.get("agent_id")

        need_retry = (
            retries > 0
            and rec.get("agent_id")
            and not is_grid(call.get("grid"))
            and last_status in ("FINISHED", "TOOLS", "CANCELLED")
        )
        if need_retry:
            why = "用了工具" if last_status == "TOOLS" else "解析失败"
            retry_prompt = NO_TOOLS_RETRY if last_status == "TOOLS" else PARSE_RETRY
            print(f"  {task_id} {rec['current_slot']} {why}，同 agent 再只要格子", flush=True)
            rec["current_slot"] = f"{ti}:{key}"
            try:
                _start_prompt(
                    rec, retry_prompt,
                    reuse_agent=rec["agent_id"],
                    model=model, effort=effort, fast=fast, use_repo=use_repo,
                )
            except Exception as exc:
                rec["error"] = str(exc)
                save_record(run_dir, rec)
                continue
            save_record(run_dir, rec)
            last_text, last_status, last_tools = wait_run(
                rec, timeout_ms, quiet, stall_ms, max_task_ms,
            )
            _store_call(rec, ti, key, last_text, last_status, last_tools)
            save_record(run_dir, rec)
            if last_status == "TIMEOUT":
                rec["status"] = "running"
                rec["error"] = "local_timeout_will_repoll"
                save_record(run_dir, rec)
                return rec.get("agent_id")

    attempts = _assemble_attempts(rec, n_test)
    # Task is "done" when every test has at least one grid; otherwise error.
    complete = attempts is not None and all(
        is_grid(_call_grid(rec, ti, "attempt_1")) or is_grid(_call_grid(rec, ti, "attempt_2"))
        for ti in range(n_test)
    )
    final_status = "FINISHED" if complete else ("SLOW" if slow_calls else last_status)
    finalize_rec(rec, task, last_text, final_status, last_tools, t0, attempts=attempts)
    if not complete and slow_calls:
        rec["error"] = "over_max_task_ms"
    rec["n_slow_calls"] = slow_calls
    save_record(run_dir, rec)
    print(
        f"  {task_id} {rec['status']} score={rec.get('task_score')} elapsed={rec.get('elapsed_s')}s "
        f"contam={rec.get('contaminated')}",
        flush=True,
    )
    return rec.get("agent_id")


def main() -> int:
    load_cursor_env()
    ap = argparse.ArgumentParser(description="Resume-safe public-120 Cursor Cloud eval")
    ap.add_argument("--data-dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--model", default=os.environ.get("CURSOR_MODEL_ID") or default_model_id())
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True,
                    help="model fast flag (default true; --no-fast to turn off)")
    ap.add_argument("--use-repo", action="store_true", help="attach CURSOR_SANDBOX_REPO_URL")
    ap.add_argument("--reuse-every", type=int, default=1,
                    help="reuse one cloud agent for attempt_1 across N tasks (1=fresh agent per call, "
                         "like stateless official API calls; >1 leaks earlier puzzles into context)")
    ap.add_argument("--timeout-ms", type=int, default=7_200_000,
                    help="local poll window per call (default 2h; does not cancel the cloud run)")
    ap.add_argument("--stall-ms", type=int, default=180_000,
                    help="cancel if no SSE activity for this long (0=disable)")
    ap.add_argument("--max-task-ms", type=int, default=0,
                    help="optional wall-clock cancel per call in ms (0=never cancel; default)")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--num-attempts", type=int, default=1, choices=[1, 2],
                    help="samples per test (1=pass@1, faster; 2=official pass@2)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--ids", default="", help="comma-separated task ids")
    ap.add_argument("--skip-ids", default="", help="comma-separated task ids to skip")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--reset-ids", default="", help="force-retry these ids")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-run tasks already marked error/parse_fail (default: keep their 0 and move on)")
    args = ap.parse_args()

    tasks = load_tasks(args.data_dir)
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

    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.reset_ids:
        for tid in args.reset_ids.split(","):
            tid = tid.strip()
            p = task_path(args.run_dir, tid)
            if p.is_file():
                p.unlink()
                print(f"reset {tid}", flush=True)

    report = rebuild_outputs(args.run_dir, load_tasks(args.data_dir) if args.data_dir == EVAL_DIR else tasks)
    if args.status:
        print_status(report)
        return 0

    use_repo = bool(args.use_repo and sandbox_repo_url())
    meta = {
        "started_at": _read_json(args.run_dir / "meta.json").get("started_at") or _now(),
        "updated_at": _now(),
        "model": args.model,
        "effort": args.effort,
        "fast": bool(args.fast),
        "use_repo": use_repo,
        "reuse_every": args.reuse_every,
        "data_dir": str(args.data_dir),
        "n_selected": len(tasks),
        "prompt_style": "official_v1",
    }
    _atomic_write(args.run_dir / "meta.json", meta)
    print(
        f"Cursor Cloud eval  model={args.model} effort={args.effort} "
        f"fast={bool(args.fast)} repo={use_repo} selected={len(tasks)} "
        f"run_dir={args.run_dir}",
        flush=True,
    )
    print_status(report)

    lock = acquire_lock(args.run_dir)
    signal.signal(signal.SIGINT, on_stop)
    signal.signal(signal.SIGTERM, on_stop)

    agent_id = None
    since_new = 0
    # Resume an in-flight agent from the first running record.
    for tid, _ in tasks:
        rec = load_record(args.run_dir, tid)
        if rec.get("status") == "running" and rec.get("agent_id"):
            agent_id = rec["agent_id"]
            break

    try:
        for i, (task_id, task) in enumerate(tasks, start=1):
            if STOP:
                break
            rec = load_record(args.run_dir, task_id)
            if rec.get("status") == "done" and rec.get("attempts"):
                continue
            if rec.get("status") in ("error", "parse_fail") and not args.retry_errors:
                continue
            reuse = None
            if args.reuse_every > 1 and agent_id and since_new < args.reuse_every:
                reuse = agent_id
            print(f"[{i}/{len(tasks)}] {task_id} n_test={len(task['test'])}", flush=True)
            agent_id = run_one(
                args.run_dir, task_id, task,
                reuse_agent=reuse,
                model=args.model,
                effort=args.effort,
                fast=bool(args.fast),
                use_repo=use_repo,
                timeout_ms=args.timeout_ms,
                stall_ms=args.stall_ms,
                max_task_ms=args.max_task_ms,
                retries=args.retries,
                quiet=args.quiet,
                num_attempts=args.num_attempts,
            )
            if reuse is None:
                since_new = 1
            else:
                since_new += 1
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
