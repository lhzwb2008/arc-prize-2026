"""Alibaba DashScope (百炼) OpenAI-compatible chat client.

Loads DASHSCOPE_* from the environment, repo .env, then ../AIvideo/.env.
Never writes the key to disk.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


_REPO_ROOT = Path(__file__).resolve().parents[2]
_AIVIDEO_ENV = _REPO_ROOT.parent / "AIvideo" / ".env"

_RETRY_HTTP = {408, 425, 429, 500, 502, 503, 504}


def load_dashscope_env() -> None:
    for path in (_REPO_ROOT / ".env", _AIVIDEO_ENV):
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key.startswith("DASHSCOPE_"):
                continue
            if key in os.environ and os.environ[key].strip():
                continue
            os.environ[key] = value.strip().strip('"').strip("'")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def api_key() -> str:
    key = _env("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError(
            "缺少 DASHSCOPE_API_KEY。可 export，或放在仓库 .env / ../AIvideo/.env"
        )
    return key


def base_url() -> str:
    return _env(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/")


def default_model() -> str:
    return _env("DASHSCOPE_ARC_MODEL", "deepseek-v4-pro")


def _retry_after_s(headers: Any, default: float = 8.0) -> float:
    try:
        ra = headers.get("Retry-After") if headers is not None else None
        if ra is not None and str(ra).strip().replace(".", "", 1).isdigit():
            return max(1.0, float(str(ra).strip()))
    except Exception:
        pass
    return default


def chat_complete(
    prompt: str,
    *,
    model: str | None = None,
    reasoning_effort: str = "high",
    enable_thinking: bool = True,
    max_tokens: int = 16384,
    timeout_s: float = 1800.0,
    stream: bool = False,
    on_delta: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """One-shot chat completion. Returns {text, reasoning, usage, raw_status}.

    Default is non-stream: DashScope thinking streams often close after ~4 min
    with only reasoning_content and no final grid.
    """
    last_err: Exception | None = None
    modes = [False] if not stream else [True, False]
    for use_stream in modes:
        try:
            return _chat_once(
                prompt,
                model=model,
                reasoning_effort=reasoning_effort,
                enable_thinking=enable_thinking,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                stream=use_stream,
                on_delta=on_delta if use_stream else None,
            )
        except Exception as exc:
            last_err = exc
            msg = str(exc)
            if use_stream and ("为空" in msg or "content" in msg.lower()):
                continue
            raise
    raise last_err or RuntimeError("DashScope 调用失败")


def _chat_once(
    prompt: str,
    *,
    model: str | None,
    reasoning_effort: str,
    enable_thinking: bool,
    max_tokens: int,
    timeout_s: float,
    stream: bool,
    on_delta: Callable[[str], None] | None,
) -> dict[str, Any]:
    url = f"{base_url()}/chat/completions"
    body: dict[str, Any] = {
        "model": model or default_model(),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
        "enable_thinking": enable_thinking,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    if stream:
        body["stream_options"] = {"include_usage": True}

    data = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
    }
    last_err: Exception | None = None
    for attempt in range(1, 5):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                if stream:
                    return _read_stream(resp, on_delta=on_delta)
                payload = json.loads(resp.read().decode("utf-8"))
                return _from_payload(payload)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            last_err = RuntimeError(f"DashScope HTTP {exc.code}: {raw[:400]}")
            if exc.code in _RETRY_HTTP and attempt < 4:
                time.sleep(_retry_after_s(exc.headers, default=4.0 * attempt))
                continue
            raise last_err from exc
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            last_err = exc
            if attempt < 4:
                time.sleep(2.0 * attempt)
                continue
            raise
    raise last_err or RuntimeError("DashScope 调用失败")


def _message_text(msg: dict) -> tuple[str, str]:
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") for p in content if isinstance(p, dict)
        )
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if isinstance(reasoning, list):
        reasoning = "".join(
            p.get("text", "") for p in reasoning if isinstance(p, dict)
        )
    return str(content), str(reasoning)


def _from_payload(payload: dict) -> dict[str, Any]:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"DashScope 无 choices: {json.dumps(payload)[:400]}")
    text, reasoning = _message_text(choices[0].get("message") or {})
    if not str(text).strip() and str(reasoning).strip():
        text = reasoning
    if not str(text).strip():
        raise RuntimeError("DashScope content 为空")
    return {
        "text": text,
        "reasoning": reasoning,
        "usage": payload.get("usage") or {},
        "finish_reason": (choices[0] or {}).get("finish_reason"),
    }


def _read_stream(resp, on_delta: Callable[[str], None] | None = None) -> dict[str, Any]:
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict[str, Any] = {}
    finish = None
    buf = b""
    while True:
        chunk = resp.read(4096)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace").strip()
            if not text or not text.startswith("data:"):
                continue
            data = text[5:].strip()
            if data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if payload.get("usage"):
                usage = payload["usage"]
            for choice in payload.get("choices") or []:
                finish = choice.get("finish_reason") or finish
                delta = choice.get("delta") or choice.get("message") or {}
                piece = delta.get("content") or delta.get("text") or ""
                if isinstance(piece, list):
                    piece = "".join(
                        p.get("text", "") for p in piece if isinstance(p, dict)
                    )
                think = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if piece:
                    content.append(str(piece))
                    if on_delta:
                        on_delta(str(piece))
                if think:
                    reasoning.append(str(think))
    out = "".join(content).strip() or "".join(reasoning).strip()
    if not out:
        raise RuntimeError("DashScope 流式 content 为空")
    return {
        "text": out,
        "reasoning": "".join(reasoning),
        "usage": usage,
        "finish_reason": finish,
    }
