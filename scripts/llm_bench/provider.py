"""OpenAI-compatible chat provider. Swap DashScope / OpenAI / any compatible base URL."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


_REPO_ROOT = Path(__file__).resolve().parents[2]
_AIVIDEO_ENV = _REPO_ROOT.parent / "AIvideo" / ".env"
_RETRY_HTTP = {408, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    default_base: str
    default_model: str
    key_envs: tuple[str, ...]
    url_envs: tuple[str, ...]
    model_envs: tuple[str, ...]
    env_prefixes: tuple[str, ...]
    stream_default: bool = False
    thinking_default: bool = False
    send_enable_thinking: bool = False


PRESETS: dict[str, ProviderSpec] = {
    "dashscope": ProviderSpec(
        name="dashscope",
        default_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="deepseek-v4-pro",
        key_envs=("DASHSCOPE_API_KEY", "LLM_BENCH_API_KEY"),
        url_envs=("DASHSCOPE_BASE_URL", "LLM_BENCH_BASE_URL"),
        model_envs=("DASHSCOPE_ARC_MODEL", "LLM_BENCH_MODEL"),
        env_prefixes=("DASHSCOPE_", "LLM_BENCH_"),
        stream_default=False,
        thinking_default=True,
        send_enable_thinking=True,
    ),
    "openai": ProviderSpec(
        name="openai",
        default_base="https://api.openai.com/v1",
        default_model="gpt-5",
        key_envs=("OPENAI_API_KEY", "LLM_BENCH_API_KEY"),
        url_envs=("OPENAI_BASE_URL", "LLM_BENCH_BASE_URL"),
        model_envs=("OPENAI_MODEL", "LLM_BENCH_MODEL"),
        env_prefixes=("OPENAI_", "LLM_BENCH_"),
        stream_default=False,
        thinking_default=False,
        send_enable_thinking=False,
    ),
    "compat": ProviderSpec(
        name="compat",
        default_base="",
        default_model="",
        key_envs=("LLM_BENCH_API_KEY", "OPENAI_API_KEY"),
        url_envs=("LLM_BENCH_BASE_URL", "OPENAI_BASE_URL"),
        model_envs=("LLM_BENCH_MODEL", "OPENAI_MODEL"),
        env_prefixes=("LLM_BENCH_", "OPENAI_"),
        stream_default=False,
        thinking_default=False,
        send_enable_thinking=False,
    ),
}


@dataclass
class ChatResult:
    text: str
    reasoning: str
    usage: dict[str, Any]
    finish_reason: str | None
    elapsed_s: float
    stream: bool
    truncated: bool = False

    @property
    def combined(self) -> str:
        parts = [self.text or "", self.reasoning or ""]
        return "\n".join(p for p in parts if p.strip())


def load_env(prefixes: tuple[str, ...] = ("DASHSCOPE_", "OPENAI_", "LLM_BENCH_")) -> None:
    extra = os.environ.get("LLM_BENCH_ENV_FILE", "").strip()
    extra_path = Path(extra) if extra else None
    cwd_env = Path.cwd() / ".env"
    candidates = [cwd_env, _REPO_ROOT / ".env", _AIVIDEO_ENV]
    if extra_path is not None:
        candidates.insert(0, extra_path)
    seen: set[Path] = set()
    for path in candidates:
        try:
            path = path.resolve()
        except OSError:
            continue
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not any(key.startswith(p) for p in prefixes):
                continue
            if key in os.environ and os.environ[key].strip():
                continue
            os.environ[key] = value.strip().strip('"').strip("'")


def _env_first(names: tuple[str, ...], default: str = "") -> str:
    for name in names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return default


def _retry_after_s(headers: Any, default: float = 8.0) -> float:
    try:
        ra = headers.get("Retry-After") if headers is not None else None
        if ra is not None and str(ra).strip().replace(".", "", 1).isdigit():
            return max(1.0, float(str(ra).strip()))
    except Exception:
        pass
    return default


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        bits: list[str] = []
        for part in value:
            if isinstance(part, dict):
                bits.append(str(part.get("text") or ""))
            else:
                bits.append(str(part))
        return "".join(bits)
    return str(value)


class OpenAICompatProvider:
    def __init__(
        self,
        spec: ProviderSpec,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        extra_body: dict[str, Any] | None = None,
        stream: bool | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: str = "high",
        max_tokens: int = 131072,
        timeout_s: float = 0.0,
    ) -> None:
        load_env(spec.env_prefixes)
        self.spec = spec
        self.api_key = (api_key or _env_first(spec.key_envs)).strip()
        if not self.api_key:
            raise RuntimeError(
                f"缺少 API key（{' / '.join(spec.key_envs)}）。"
                "可 export，或放在仓库 .env / 运行目录 .env / ../AIvideo/.env"
            )
        self.base_url = (base_url or _env_first(spec.url_envs, spec.default_base)).rstrip("/")
        if not self.base_url:
            raise RuntimeError("缺少 base URL（LLM_BENCH_BASE_URL / OPENAI_BASE_URL）")
        self.model = (model or _env_first(spec.model_envs, spec.default_model)).strip()
        if not self.model:
            raise RuntimeError("缺少 model（--model / LLM_BENCH_MODEL）")
        self.extra_body = dict(extra_body or {})
        self.stream_default = spec.stream_default if stream is None else stream
        self.enable_thinking = (
            spec.thinking_default if enable_thinking is None else enable_thinking
        )
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
        stream: bool | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        extra_body: dict[str, Any] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> ChatResult:
        use_stream = self.stream_default if stream is None else stream
        think = self.enable_thinking if enable_thinking is None else enable_thinking
        effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort
        modes = [False] if not use_stream else [True, False]
        last_err: Exception | None = None
        for mode in modes:
            try:
                return self._complete_once(
                    messages,
                    max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                    timeout_s=timeout_s if timeout_s is not None else self.timeout_s,
                    stream=mode,
                    enable_thinking=think,
                    reasoning_effort=effort,
                    extra_body=extra_body,
                    on_delta=on_delta if mode else None,
                )
            except Exception as exc:
                last_err = exc
                msg = str(exc)
                if mode and ("为空" in msg or "empty" in msg.lower()):
                    continue
                raise
        raise last_err or RuntimeError("chat 调用失败")

    def complete_prompt(self, prompt: str, **kwargs: Any) -> ChatResult:
        return self.complete([{"role": "user", "content": prompt}], **kwargs)

    def _complete_once(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        timeout_s: float | None,
        stream: bool,
        enable_thinking: bool,
        reasoning_effort: str,
        extra_body: dict[str, Any] | None,
        on_delta: Callable[[str], None] | None,
    ) -> ChatResult:
        url = f"{self.base_url}/chat/completions"
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        if self.spec.send_enable_thinking:
            body["enable_thinking"] = bool(enable_thinking)
        for src in (self.extra_body, extra_body):
            if src:
                body.update(src)
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_err: Exception | None = None
        t0 = time.time()
        for attempt in range(1, 5):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                http_timeout = None if not timeout_s or timeout_s <= 0 else timeout_s
                with urllib.request.urlopen(req, timeout=http_timeout) as resp:
                    if stream:
                        parsed = _read_stream(resp, on_delta=on_delta)
                    else:
                        payload = json.loads(resp.read().decode("utf-8"))
                        parsed = _from_payload(payload)
                elapsed = round(time.time() - t0, 2)
                usage = parsed.get("usage") or {}
                finish = parsed.get("finish_reason")
                truncated = _is_truncated(finish, usage, parsed.get("text") or "")
                return ChatResult(
                    text=parsed.get("text") or "",
                    reasoning=parsed.get("reasoning") or "",
                    usage=usage,
                    finish_reason=finish,
                    elapsed_s=elapsed,
                    stream=stream,
                    truncated=truncated,
                )
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                last_err = RuntimeError(f"{self.spec.name} HTTP {exc.code}: {raw[:400]}")
                if exc.code in _RETRY_HTTP and attempt < 4:
                    time.sleep(_retry_after_s(exc.headers, default=4.0 * attempt))
                    continue
                raise last_err from exc
            except (TimeoutError, urllib.error.URLError, OSError) as exc:
                last_err = RuntimeError(f"{self.spec.name} 网络/超时: {exc}")
                if attempt < 4:
                    time.sleep(2.0 * attempt)
                    continue
                raise last_err from exc
        raise last_err or RuntimeError(f"{self.spec.name} 调用失败")


def _is_truncated(finish: str | None, usage: dict[str, Any], text: str) -> bool:
    if str(finish or "").lower() == "length":
        return True
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = int(details.get("reasoning_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if completion_tokens > 0 and reasoning_tokens >= completion_tokens and not str(text).strip():
        return True
    return False


def _message_text(msg: dict) -> tuple[str, str]:
    content = _as_text(msg.get("content"))
    reasoning = _as_text(msg.get("reasoning_content") or msg.get("reasoning"))
    return content, reasoning


def _from_payload(payload: dict) -> dict[str, Any]:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"无 choices: {json.dumps(payload)[:400]}")
    text, reasoning = _message_text(choices[0].get("message") or {})
    if not str(text).strip() and not str(reasoning).strip():
        raise RuntimeError("chat content 为空")
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
                piece = _as_text(delta.get("content") or delta.get("text"))
                think = _as_text(delta.get("reasoning_content") or delta.get("reasoning"))
                if piece:
                    content.append(piece)
                    if on_delta:
                        on_delta(piece)
                if think:
                    reasoning.append(think)
    text = "".join(content)
    think = "".join(reasoning)
    if not text.strip() and not think.strip():
        raise RuntimeError("流式 content 为空")
    return {
        "text": text,
        "reasoning": think,
        "usage": usage,
        "finish_reason": finish,
    }


def make_provider(name: str, **kwargs: Any) -> OpenAICompatProvider:
    key = (name or "dashscope").strip().lower()
    if key not in PRESETS:
        raise SystemExit(f"未知 --provider {name}，可选: {', '.join(PRESETS)}")
    return OpenAICompatProvider(PRESETS[key], **kwargs)
