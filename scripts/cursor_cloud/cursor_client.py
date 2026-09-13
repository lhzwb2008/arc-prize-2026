"""Cursor Cloud Agents REST client.

Credential loading matches ../AIvideo: CURSOR_API_KEY from the environment
or AIvideo/.env. This module never writes the key to disk.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


_REPO_ROOT = Path(__file__).resolve().parents[2]
_AIVIDEO_ENV = _REPO_ROOT.parent / "AIvideo" / ".env"


def load_cursor_env() -> None:
    """Fill missing CURSOR_* from repo .env then AIvideo/.env."""
    for path in (_REPO_ROOT / ".env", _AIVIDEO_ENV):
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key.startswith("CURSOR_"):
                continue
            if key in os.environ and os.environ[key].strip():
                continue
            os.environ[key] = value.strip().strip('"').strip("'")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def auth_header() -> str:
    key = _env("CURSOR_API_KEY")
    if not key:
        raise RuntimeError(
            "缺少 CURSOR_API_KEY。可 export，或放在仓库 .env / ../AIvideo/.env"
        )
    token = base64.b64encode(f"{key}:".encode()).decode()
    return f"Basic {token}"


def base_url() -> str:
    return _env("CURSOR_BASE_URL", "https://api.cursor.com").rstrip("/")


def default_model_id() -> str:
    return _env("CURSOR_MODEL_ID", "grok-4.6")


def sandbox_repo_url() -> str:
    return _env("CURSOR_SANDBOX_REPO_URL")


_RETRY_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}


def _retry_after_s(raw: str, headers: Any = None, *, default: float = 60.0) -> float:
    if headers is not None:
        try:
            ra = headers.get("Retry-After")
            if ra is not None and str(ra).strip().isdigit():
                return max(1.0, float(str(ra).strip()))
        except Exception:
            pass
    try:
        data = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        details = data.get("details")
        if isinstance(details, list):
            for item in details:
                if not isinstance(item, dict):
                    continue
                debug = item.get("debug") if isinstance(item.get("debug"), dict) else {}
                inner = debug.get("details") if isinstance(debug.get("details"), dict) else {}
                info = inner.get("additionalInfo") if isinstance(inner.get("additionalInfo"), dict) else {}
                for blob in (info, inner, item, data):
                    if not isinstance(blob, dict):
                        continue
                    for key in ("retryAfter", "retry_after"):
                        val = blob.get(key)
                        if val is None:
                            continue
                        try:
                            return max(1.0, float(val))
                        except (TypeError, ValueError):
                            continue
        if "rate limit" in raw.lower() or "resource_exhausted" in raw.lower():
            return default
    return default


def _http(method: str, path: str, body: dict | None = None) -> tuple[int, Any, str]:
    url = base_url() + path
    data = None if body is None else json.dumps(body).encode()
    max_attempts = max(1, int(_env("CURSOR_HTTP_MAX_RETRIES", "6")))
    last_err: Exception | None = None
    last_status, last_parsed, last_raw = 0, None, ""
    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": auth_header(),
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode()
                try:
                    parsed = json.loads(raw) if raw else None
                except json.JSONDecodeError:
                    parsed = None
                return resp.status, parsed, raw
        except urllib.error.HTTPError as e:
            raw = e.read().decode() if e.fp else ""
            try:
                parsed = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                parsed = None
            last_status, last_parsed, last_raw = e.code, parsed, raw
            if e.code in _RETRY_HTTP_CODES and attempt < max_attempts:
                wait_s = _retry_after_s(
                    raw, getattr(e, "headers", None),
                    default=60.0 if e.code == 429 else 5.0,
                )
                if e.code != 429:
                    wait_s = min(120.0, max(wait_s, 2.0 * attempt))
                print(
                    f"[cursor] {method} {path} → {e.code}，{wait_s:.0f}s 后重试 "
                    f"({attempt}/{max_attempts - 1})",
                    flush=True,
                )
                time.sleep(wait_s)
                continue
            return e.code, parsed, raw
        except Exception as e:
            last_err = e
            if attempt < max_attempts:
                time.sleep(min(30.0, 0.5 * attempt))
                continue
            raise RuntimeError(f"HTTP {method} {path} 失败: {last_err}") from last_err
    if last_status:
        return last_status, last_parsed, last_raw
    raise RuntimeError(f"HTTP {method} {path} 失败: {last_err}")


def _ids(data: dict) -> tuple[str, str]:
    agent = data.get("agent") if isinstance(data.get("agent"), dict) else data
    run = data.get("run") if isinstance(data.get("run"), dict) else {}
    agent_id = agent.get("id") or data.get("id") or data.get("agentId")
    run_id = run.get("id") or data.get("runId") or agent.get("latestRunId")
    if not agent_id or not run_id:
        raise RuntimeError(f"响应缺少 agent/run id: {json.dumps(data)[:400]}")
    return str(agent_id), str(run_id)


def create_agent(
    prompt: str,
    *,
    model: str | None = None,
    effort: str | None = None,
    fast: bool | None = None,
    use_repo: bool = False,
    name: str = "ARC-AGI-2 public eval",
) -> tuple[str, str]:
    mid = model or default_model_id()
    model_obj: dict[str, Any] = {"id": mid}
    params: list[dict[str, str]] = []
    if effort:
        params.append({"id": "effort", "value": effort})
    if fast is not None:
        params.append({"id": "fast", "value": "true" if fast else "false"})
    if params:
        model_obj["params"] = params
    body: dict[str, Any] = {
        "prompt": {"text": prompt},
        "model": model_obj,
        "name": name[:100],
        "autoCreatePR": False,
    }
    repo = sandbox_repo_url() if use_repo else ""
    if repo:
        body["repos"] = [{"url": repo}]
    status, data, raw = _http("POST", "/v1/agents", body)
    if status not in (200, 201) or not isinstance(data, dict):
        raise RuntimeError(f"createAgent 失败 {status}: {raw[:800]}")
    return _ids(data)


def create_run(agent_id: str, prompt: str) -> str:
    for _ in range(30):
        status, data, raw = _http(
            "POST",
            f"/v1/agents/{agent_id}/runs",
            {"prompt": {"text": prompt}},
        )
        if status in (200, 201) and isinstance(data, dict):
            run = data.get("run") if isinstance(data.get("run"), dict) else data
            run_id = run.get("id")
            if not run_id:
                raise RuntimeError(f"createRun 无 run id: {raw[:400]}")
            return str(run_id)
        if status == 409:
            time.sleep(2)
            continue
        raise RuntimeError(f"createRun 失败 {status}: {raw[:800]}")
    raise RuntimeError(f"createRun: agent {agent_id} 一直 busy")


def get_run(agent_id: str, run_id: str) -> dict:
    status, data, raw = _http("GET", f"/v1/agents/{agent_id}/runs/{run_id}")
    if status != 200 or not isinstance(data, dict):
        raise RuntimeError(f"getRun 失败 {status}: {raw[:800]}")
    return data.get("run") if isinstance(data.get("run"), dict) else data


def cancel_run(agent_id: str, run_id: str) -> bool:
    status, _data, raw = _http("POST", f"/v1/agents/{agent_id}/runs/{run_id}/cancel", {})
    if status in (200, 201):
        return True
    if status == 409:
        return False
    raise RuntimeError(f"cancelRun 失败 {status}: {raw[:400]}")


def _dispatch_sse(
    event_name: str,
    payload: Any,
    on_assistant: Callable[[str], None] | None,
    on_tool_call: Callable[[dict], None] | None,
    on_thinking: Callable[[str], None] | None,
    on_event: Callable[[str, Any], None] | None,
    on_assistant_iu: Callable[[str], None] | None = None,
    on_thinking_iu: Callable[[str], None] | None = None,
) -> bool:
    """Return True if the stream should stop."""
    if on_event:
        on_event(event_name, payload)
    if not isinstance(payload, dict):
        return event_name in ("result", "done", "error")
    if event_name == "assistant" and on_assistant:
        text = payload.get("text")
        if isinstance(text, str) and text:
            on_assistant(text)
    elif event_name == "thinking" and on_thinking:
        text = payload.get("text")
        if isinstance(text, str) and text:
            on_thinking(text)
    elif event_name == "tool_call" and on_tool_call:
        on_tool_call(payload)
    elif event_name == "interaction_update":
        # The API emits the same delta both as `assistant`/`thinking` and as
        # `interaction_update`. Route the latter to the *_iu callbacks so the
        # caller can pick one channel instead of doubling every token.
        kind = str(payload.get("type") or "")
        if kind in ("text-delta", "assistant-delta") and on_assistant_iu:
            text = payload.get("text") or payload.get("delta") or ""
            if text:
                on_assistant_iu(str(text))
        elif kind in ("thinking-delta", "thought-delta") and on_thinking_iu:
            text = payload.get("text") or payload.get("delta") or ""
            if text:
                on_thinking_iu(str(text))
        elif kind.startswith("tool-call") and on_tool_call:
            on_tool_call(payload)
    return event_name in ("result", "done", "error")


def _consume_sse(
    agent_id: str,
    run_id: str,
    on_assistant: Callable[[str], None] | None = None,
    on_tool_call: Callable[[dict], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
    on_event: Callable[[str, Any], None] | None = None,
    timeout_s: float = 600,
    on_assistant_iu: Callable[[str], None] | None = None,
    on_thinking_iu: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Consume the run's SSE stream, reconnecting on transport errors.

    Without the stream we cannot see thinking or the answer being written,
    so a dropped connection must not silently blind the wall-clock cap.
    """
    url = f"{base_url()}/v1/agents/{agent_id}/runs/{run_id}/stream"
    deadline = time.time() + timeout_s
    attempt = 0
    while time.time() < deadline and not (should_stop and should_stop()):
        attempt += 1
        req = urllib.request.Request(
            url,
            headers={"Authorization": auth_header(), "Accept": "text/event-stream"},
        )
        try:
            with urllib.request.urlopen(req, timeout=max(5.0, deadline - time.time())) as resp:
                if on_event and attempt > 1:
                    on_event("sse_reconnected", {"attempt": attempt})
                event_name = "message"
                data_lines: list[str] = []
                while time.time() < deadline:
                    line = resp.readline()
                    if not line:
                        break
                    text = line.decode(errors="replace")
                    if text in ("\n", "\r\n"):
                        if data_lines:
                            try:
                                payload = json.loads("\n".join(data_lines))
                            except json.JSONDecodeError:
                                payload = {"raw": "\n".join(data_lines)}
                            if _dispatch_sse(
                                event_name, payload,
                                on_assistant, on_tool_call, on_thinking, on_event,
                                on_assistant_iu=on_assistant_iu,
                                on_thinking_iu=on_thinking_iu,
                            ):
                                return
                        event_name = "message"
                        data_lines = []
                        continue
                    if text.startswith("event:"):
                        event_name = text[6:].strip()
                    elif text.startswith("data:"):
                        data_lines.append(text[5:].strip())
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode(errors="replace")[:300]
            except Exception:
                pass
            if on_event:
                on_event("sse_error", {"message": f"HTTP {exc.code}", "body": body, "attempt": attempt})
            if exc.code in (400, 401, 403, 404):
                return
        except Exception as exc:
            if on_event:
                on_event("sse_error", {"message": str(exc), "attempt": attempt})
        # Stream ended without a terminal event (or failed): back off and retry.
        time.sleep(min(15.0, 2.0 * attempt))


def run_with_stream(
    agent_id: str,
    run_id: str,
    *,
    timeout_ms: int = 1_800_000,
    poll_interval_ms: int = 4000,
    stall_ms: int = 180_000,
    max_task_ms: int = 0,
    on_assistant: Callable[[str], None] | None = None,
    on_tool_call: Callable[[dict], None] | None = None,
    on_thinking: Callable[[str], None] | None = None,
    on_event: Callable[[str, Any], None] | None = None,
    output_grace_ms: int = 240_000,
) -> tuple[str, str]:
    """Wait for a run.

    max_task_ms caps *thinking* time. Once the model starts emitting the
    answer, cancelling would throw the solution away, so we allow up to
    output_grace_ms more for the output to finish.
    """
    assistant_buf: list[str] = []       # from `assistant` events
    assistant_iu_buf: list[str] = []    # from `interaction_update` text-delta
    last_activity = time.time()
    first_output_at: float | None = None

    def _touch() -> None:
        nonlocal last_activity
        last_activity = time.time()

    def _mark_output() -> None:
        nonlocal first_output_at
        if first_output_at is None:
            first_output_at = time.time()

    def _on_assistant(delta: str) -> None:
        _touch()
        _mark_output()
        assistant_buf.append(delta)
        if on_assistant:
            on_assistant(delta)

    def _on_assistant_iu(delta: str) -> None:
        _touch()
        _mark_output()
        assistant_iu_buf.append(delta)
        if on_assistant and not assistant_buf:
            on_assistant(delta)

    def _on_thinking(delta: str) -> None:
        _touch()
        if on_thinking:
            on_thinking(delta)

    def _on_thinking_iu(delta: str) -> None:
        _touch()

    def _on_tool(payload: dict) -> None:
        _touch()
        if on_tool_call:
            on_tool_call(payload)

    def _on_event(name: str, payload: Any) -> None:
        if name not in ("heartbeat", "status"):
            _touch()
        if on_event:
            on_event(name, payload)

    import threading

    stop_flag = {"stop": False}

    sse_thread = threading.Thread(
        target=_consume_sse,
        kwargs={
            "agent_id": agent_id,
            "run_id": run_id,
            "on_assistant": _on_assistant,
            "on_tool_call": _on_tool,
            "on_thinking": _on_thinking,
            "on_event": _on_event,
            "timeout_s": timeout_ms / 1000,
            "on_assistant_iu": _on_assistant_iu,
            "on_thinking_iu": _on_thinking_iu,
            "should_stop": lambda: stop_flag["stop"],
        },
        daemon=True,
    )
    sse_thread.start()

    started = time.time()
    deadline = started + timeout_ms / 1000
    final_status = "TIMEOUT"
    final_text = ""
    while time.time() < deadline:
        try:
            r = get_run(agent_id, run_id)
        except Exception:
            time.sleep(poll_interval_ms / 1000)
            continue
        status = str(r.get("status") or "")
        if status in ("FINISHED", "ERROR", "CANCELLED"):
            final_status = status
            final_text = r.get("result") or r.get("text") or ""
            if not isinstance(final_text, str):
                final_text = json.dumps(final_text)
            break
        if stall_ms > 0 and (time.time() - last_activity) * 1000 > stall_ms:
            try:
                cancel_run(agent_id, run_id)
            except Exception:
                pass
            final_status = "STALL"
            break
        if max_task_ms > 0 and (time.time() - started) * 1000 >= max_task_ms:
            if first_output_at is not None and (time.time() - first_output_at) * 1000 < output_grace_ms:
                # Answer is streaming; let it finish.
                if on_event:
                    on_event("output_grace", {"since_ms": int((time.time() - first_output_at) * 1000)})
                time.sleep(poll_interval_ms / 1000)
                continue
            try:
                cancel_run(agent_id, run_id)
            except Exception:
                pass
            final_status = "SLOW"
            break
        time.sleep(poll_interval_ms / 1000)

    stop_flag["stop"] = True
    sse_thread.join(timeout=2)
    streamed = "".join(assistant_buf) or "".join(assistant_iu_buf)
    text = final_text or streamed
    return text, final_status
