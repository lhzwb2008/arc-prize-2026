"""Official ARC-AGI-2 prompt, grid parse, and resume-safe run IO."""

from __future__ import annotations

import ast
import fcntl
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "data" / "full" / "data" / "evaluation"

# Official ARC Prize harness (arc-agi-benchmarking system_prompt.txt).
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

GRID_FOLLOWUP = (
    "Your previous reply used the token budget on reasoning and did not emit "
    "a valid output grid. Respond with only the output grid as a JSON list of "
    "lists of integers 0-9, in the same format as the training OUTPUT examples."
)


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


def build_prompt(task: dict, test_index: int = 0) -> str:
    training = []
    for i, pair in enumerate(task["train"]):
        training.append(f"--Example {i}-- \n\n INPUT: \n\n")
        training.append(json.dumps(pair["input"]))
        training.append("\n\nOUTPUT: \n\n")
        training.append(json.dumps(pair["output"]))
        training.append("\n\n")
    test_input = json.dumps(task["test"][test_index]["input"])
    return _OFFICIAL_TEMPLATE.format(
        training_examples="".join(training),
        test_input=test_input,
    )


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


def parse_python_grid(text: str) -> Any:
    """Last nested int list, including Python literals with single quotes."""
    best = None
    for m in re.finditer(r"\[\s*\[(?:.|\n){0,8000}?\]\s*\]", text):
        chunk = m.group(0)
        parsed = None
        try:
            parsed = json.loads(chunk)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(chunk)
            except (ValueError, SyntaxError, MemoryError):
                parsed = None
        if is_grid(parsed):
            best = parsed
    return best


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


def parse_official_grid(text: str) -> Any:
    if not text or not text.strip():
        return None
    boxed = re.search(r"\\boxed\{(.*?)\}", text, flags=re.DOTALL)
    if boxed:
        inner = boxed.group(1).strip()
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(inner)
            except (json.JSONDecodeError, ValueError, SyntaxError, MemoryError):
                parsed = None
            if is_grid(parsed):
                return parsed
    grid = backscan_json_grid(text)
    if grid is not None:
        return grid
    old = parse_attempts(text, 1)
    if old and is_grid(old[0].get("attempt_1")):
        return old[0]["attempt_1"]
    py = parse_python_grid(text)
    if is_grid(py):
        return py
    return parse_space_separated_grid(text)


def parse_grid(*parts: str | None) -> Any:
    """Prefer answer content, then reasoning, then the concatenation."""
    seen: list[str] = []
    for part in parts:
        if not part or not str(part).strip():
            continue
        if part in seen:
            continue
        seen.append(part)
        grid = parse_official_grid(part)
        if is_grid(grid):
            return grid
    if len(seen) > 1:
        return parse_official_grid("\n".join(seen))
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


def task_path(run_dir: Path, task_id: str) -> Path:
    return run_dir / "tasks" / f"{task_id}.json"


def load_record(run_dir: Path, task_id: str) -> dict:
    rec = _read_json(task_path(run_dir, task_id))
    rec.setdefault("task_id", task_id)
    rec.setdefault("status", "pending")
    return rec


def save_record(run_dir: Path, rec: dict) -> None:
    _atomic_write(task_path(run_dir, rec["task_id"]), rec)


def write_raw(run_dir: Path, task_id: str, slot: str, text: str) -> str | None:
    if not text:
        return None
    path = run_dir / "raw" / f"{task_id}_{slot}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path.relative_to(run_dir))


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
        rows.append(
            {
                "task_id": task_id,
                "status": rec.get("status"),
                "score": score if rec.get("status") == "done" else None,
                "per_test": per if rec.get("status") == "done" else None,
                "elapsed_s": rec_elapsed,
                "model": rec.get("model"),
                "provider": rec.get("provider"),
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
        "score_points": scored,
        "score_pct": 100.0 * scored / n if n else 0.0,
        "done_pct": 100.0 * scored / done if done else 0.0,
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
        f"score {report['score_points']:.2f}/{report['n_tasks']} "
        f"({report['score_pct']:.2f}%)  "
        f"on-done {report['done_pct']:.1f}%  "
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


def assemble_attempts(rec: dict, n_test: int) -> list[dict] | None:
    attempts: list[dict] = []
    any_grid = False
    for ti in range(n_test):
        a1 = next(
            (
                c["grid"]
                for c in rec.get("calls") or []
                if c.get("test_index") == ti and c.get("key") == "attempt_1" and is_grid(c.get("grid"))
            ),
            None,
        )
        a2 = next(
            (
                c["grid"]
                for c in rec.get("calls") or []
                if c.get("test_index") == ti and c.get("key") == "attempt_2" and is_grid(c.get("grid"))
            ),
            None,
        )
        if is_grid(a1) or is_grid(a2):
            any_grid = True
        a1 = a1 if is_grid(a1) else (a2 if is_grid(a2) else [[0]])
        a2 = a2 if is_grid(a2) else a1
        attempts.append({"attempt_1": a1, "attempt_2": a2})
    return attempts if any_grid else None


def finalize_rec(rec: dict, task: dict, t0: float) -> None:
    n_test = len(task["test"])
    attempts = assemble_attempts(rec, n_test)
    rec["attempts"] = attempts
    rec["finished_at"] = _now()
    rec["elapsed_s"] = round(time.time() - t0 + float(rec.get("elapsed_s_offset") or 0), 2)
    gold = gold_outputs(task)
    score, per = score_task(attempts, gold)
    rec["task_score"] = score
    rec["per_test"] = per
    complete = True
    if attempts is None:
        complete = False
    else:
        for ti in range(n_test):
            if not any(
                c.get("test_index") == ti and is_grid(c.get("grid"))
                for c in rec.get("calls") or []
            ):
                complete = False
                break
    if complete:
        rec["status"] = "done"
        rec["error"] = None
    elif any(c.get("status") == "ERROR" for c in rec.get("calls") or []):
        rec["status"] = "error"
    else:
        rec["status"] = "parse_fail"
        rec["error"] = rec.get("error") or "unparsed grid"
