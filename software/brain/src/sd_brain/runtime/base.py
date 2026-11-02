"""The ``ModelRuntime`` interface and the OpenAI-compatible chat call they share."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from sd_brain.config import LlmConfig

__all__ = [
    "ModelError",
    "ModelReply",
    "ModelRuntime",
    "ModelTimeoutError",
    "NotConfiguredError",
    "RawToolCall",
    "Sampling",
    "llm_sampling",
    "openai_chat",
]

Engine = Literal["cloud"]


class ModelError(RuntimeError):
    """The model server failed (HTTP error, bad answer). ``kind`` goes to the turn log."""

    kind: Literal["http_error", "timeout", "parse_error"] = "http_error"
    status: int | None = None  # the HTTP status when the server answered with an error


class ModelTimeoutError(ModelError):
    kind: Literal["http_error", "timeout", "parse_error"] = "timeout"


class NotConfiguredError(ModelError):
    """The runtime has no key or model set."""


class Sampling(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    temperature: float = 0.2
    top_p: float = 0.8
    max_tokens: int = 4096  # [llm]'s default (LlmConfig): what a flight sends
    seed: int | None = None


def llm_sampling(llm: LlmConfig) -> Sampling:
    """The sampling of ``[llm]``: what a flight sends, so a drill or evaluation sends it too."""
    return Sampling(
        temperature=llm.temperature, top_p=llm.top_p, max_tokens=llm.max_tokens, seed=llm.seed
    )


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
    reasoning_tokens: int = 0  # the share of completion_tokens the model spent reasoning
    latency_ms: int = 0
    engine: Engine | None = None
    model_id: str | None = None
    cost_usd: float | None = None  # the provider's own price for the call (OpenRouter usage)
    cost_sek: float | None = None  # what the ground's cloud budget booked for the call
    # False when the answer's usage was malformed: its counts read 0 and the cloud
    # budget books the call at its estimate, so a broken count never makes a call free.
    usage_valid: bool = True


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
        data = _mapping(data)
        choice = _mapping(data["choices"][0])
        msg = _mapping(choice.get("message"))
        calls = []
        for i, raw in enumerate(msg.get("tool_calls") or []):
            c = _mapping(raw)
            fn = _mapping(c["function"])
            args = fn.get("arguments")
            calls.append(
                RawToolCall(
                    id=str(c.get("id") or f"call_{i}"),
                    name=str(fn["name"]),
                    arguments_json=args if isinstance(args, str) else _dumps(args or {}),
                )
            )
        prompt_tokens, completion_tokens, reasoning_tokens, cost_usd, usage_valid = _usage(
            data.get("usage")
        )
        return ModelReply(
            content=msg.get("content"),
            tool_calls=calls,
            finish_reason=str(choice.get("finish_reason") or "stop"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_ms=latency_ms,
            model_id=data.get("model"),
            cost_usd=cost_usd,
            usage_valid=usage_valid,
        )
    # ArithmeticError: a backstop for a number no check above foresaw (an OverflowError).
    except (KeyError, IndexError, TypeError, ValueError, ArithmeticError) as exc:
        err = ModelError(f"unexpected answer from the model server: {type(exc).__name__}")
        err.kind = "parse_error"
        raise err from None


# The most tokens one call may state: what the ground's cloud budget books at most
# (``BudgetCommit``); a larger count is malformed usage.
MAX_TOKEN_COUNT = 10_000_000


def _usage(value: Any) -> tuple[int, int, int, float | None, bool]:
    """The token counts and price of a call: (prompt, completion, reasoning, cost, valid).

    Usage is metadata only. A malformed part (also a count that is no finite whole
    number from 0 to ``MAX_TOKEN_COUNT``) keeps the answer: its counts read 0 and
    ``valid`` is False, so the cloud budget books the call at its estimate. A broken
    count must neither make a good answer a parse error (a second, paid call to the
    fallback model) nor book the call as free. A left-out part is no error: it reads 0.
    """
    if value is None:
        return 0, 0, 0, None, True
    if not isinstance(value, Mapping):
        return 0, 0, 0, None, False
    details = value.get("completion_tokens_details")
    if details is not None and not isinstance(details, Mapping):
        return 0, 0, 0, None, False
    prompt = _count(value.get("prompt_tokens"))
    completion = _count(value.get("completion_tokens"))
    reasoning = _count((details or {}).get("reasoning_tokens"))
    if prompt is None or completion is None or reasoning is None:
        return 0, 0, 0, None, False
    return prompt, completion, reasoning, _price(value.get("cost")), True


# The highest price of one call the ground's budget books (``BudgetCommit.cost_usd``).
MAX_COST_USD = 1000


def _price(value: Any) -> float | None:
    """The provider's price; None (the ground prices the tokens) unless it is a number
    from 0 to ``MAX_COST_USD``. Compared before any float(), so a price too large for a
    float is dropped too, never an OverflowError."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not 0 <= value <= MAX_COST_USD:  # also NaN, which compares False
        return None
    price = float(value)
    return price if math.isfinite(price) else None


def _count(value: Any) -> int | None:
    """A token count: 0 when left out, None (malformed) when it is no finite whole number
    from 0 to ``MAX_TOKEN_COUNT``. Compared as it came, never through a float, so a count
    too large for one is malformed too, not an OverflowError."""
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if isinstance(value, float) and not value.is_integer():  # also NaN and infinity
        return None
    if not 0 <= value <= MAX_TOKEN_COUNT:
        return None
    return int(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    """An object of the answer; empty when left out (absent or null), a TypeError (a parse
    error) when it is anything else, also an empty string, list or 0."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("not an object")
    return value  # type: ignore[return-value]


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
    extra_body: dict[str, Any] | None = None,
) -> ModelReply:
    """``POST {url}`` (``…/chat/completions``) with tools; raise on any failure.

    No error message holds the key, the prompt or the answer: they go to logs.
    """
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
    body.update(extra_body or {})
    headers = {"Authorization": f"Bearer {api_key}", **(extra_headers or {})}
    start = time.monotonic()
    try:
        resp = await client.post(url, json=body, headers=headers, timeout=max(0.05, timeout_s))
    except httpx.TimeoutException:
        raise ModelTimeoutError("the model did not answer in time") from None
    except httpx.HTTPError as exc:
        raise ModelError(f"model server unreachable: {type(exc).__name__}") from None
    if resp.status_code != 200:
        err = ModelError(f"model server answered HTTP {resp.status_code}")
        err.status = resp.status_code
        raise err
    try:
        data = resp.json()
    except (ValueError, RecursionError):  # RecursionError: a body nested too deep
        err = ModelError("model server answered with non-JSON")
        err.kind = "parse_error"
        raise err from None
    return parse_reply(data, latency_ms=int((time.monotonic() - start) * 1000))
