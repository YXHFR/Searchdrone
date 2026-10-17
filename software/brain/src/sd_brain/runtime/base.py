"""The ``ModelRuntime`` interface and the OpenAI-compatible chat call they share."""

from __future__ import annotations

import time
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict

__all__ = [
    "ModelError",
    "ModelReply",
    "ModelRuntime",
    "ModelTimeoutError",
    "NotConfiguredError",
    "RawToolCall",
    "Sampling",
    "openai_chat",
]

Engine = Literal["cloud", "local"]


class ModelError(RuntimeError):
    """The model server failed (HTTP error, bad answer). ``kind`` goes to the turn log."""

    kind: Literal["http_error", "timeout", "parse_error"] = "http_error"


class ModelTimeoutError(ModelError):
    kind: Literal["http_error", "timeout", "parse_error"] = "timeout"


class NotConfiguredError(ModelError):
    """The runtime has no key or model set."""


class Sampling(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    temperature: float = 0.2
    top_p: float = 0.8
    max_tokens: int = 512
    seed: int | None = None


class RawToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    name: str
    arguments_json: str


class ModelReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str | None
    tool_calls: list[RawToolCall]
    finish_reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    engine: Engine | None = None
    model_id: str | None = None


class ModelRuntime(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def model_id(self) -> str: ...
    @property
    def supports_images(self) -> bool: ...

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
    ) -> ModelReply: ...


def parse_reply(data: Any, *, latency_ms: int) -> ModelReply:
    """An OpenAI ``chat/completions`` response body as a :class:`ModelReply`."""
    try:
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        calls = [
            RawToolCall(
                id=str(c.get("id") or f"call_{i}"),
                name=str(c["function"]["name"]),
                arguments_json=(
                    c["function"].get("arguments")
                    if isinstance(c["function"].get("arguments"), str)
                    else _dumps(c["function"].get("arguments") or {})
                ),
            )
            for i, c in enumerate(msg.get("tool_calls") or [])
        ]
        usage = data.get("usage") or {}
        return ModelReply(
            content=msg.get("content"),
            tool_calls=calls,
            finish_reason=str(choice.get("finish_reason") or "stop"),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
            model_id=data.get("model"),
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        err = ModelError(f"unexpected answer from the model server: {type(exc).__name__}")
        err.kind = "parse_error"
        raise err from None


def _dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


async def openai_chat(
    client: httpx.AsyncClient,
    url: str,
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    timeout_s: float,
    sampling: Sampling,
    extra_headers: dict[str, str] | None = None,
) -> ModelReply:
    """``POST {url}`` (``…/chat/completions``) with tools; raise on any failure."""
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "max_tokens": sampling.max_tokens,
    }
    if sampling.seed is not None:
        body["seed"] = sampling.seed
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    headers = {"Authorization": f"Bearer {api_key}", **(extra_headers or {})}
    start = time.monotonic()
    try:
        resp = await client.post(url, json=body, headers=headers, timeout=max(0.05, timeout_s))
    except httpx.TimeoutException:
        raise ModelTimeoutError("the model did not answer in time") from None
    except httpx.HTTPError as exc:
        raise ModelError(f"model server unreachable: {type(exc).__name__}") from None
    if resp.status_code != 200:
        raise ModelError(f"model server answered HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        err = ModelError("model server answered with non-JSON")
        err.kind = "parse_error"
        raise err from None
    return parse_reply(data, latency_ms=int((time.monotonic() - start) * 1000))
