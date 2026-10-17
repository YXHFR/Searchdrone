"""``ReplayRuntime``: returns written-down model answers in order (golden tests)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sd_brain.runtime.base import ModelReply, RawToolCall, Sampling

__all__ = ["ReplayExhaustedError", "ReplayNotConsumedError", "ReplayRuntime", "load_golden"]


class ReplayExhaustedError(AssertionError):
    """More model calls than written answers."""


class ReplayNotConsumedError(AssertionError):
    """Written answers left when the scenario ended."""


def _reply(raw: dict[str, Any]) -> ModelReply:
    calls = [
        RawToolCall(
            id=f"call_{i}",
            name=str(c["name"]),
            arguments_json=(
                c["arguments"]
                if isinstance(c.get("arguments"), str)
                else json.dumps(c.get("arguments", {}), ensure_ascii=False)
            ),
        )
        for i, c in enumerate(raw.get("tool_calls", []))
    ]
    return ModelReply(
        content=raw.get("content"),
        tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
        prompt_tokens=int(raw.get("prompt_tokens", 0)),
        completion_tokens=int(raw.get("completion_tokens", 0)),
        latency_ms=int(raw.get("latency_ms", 0)),
        engine="local",
        model_id="replay",
    )


def load_golden(path: Path) -> list[ModelReply]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows.sort(key=lambda r: (int(r["step"]), int(r["call"])))
    return [_reply(r["reply"]) for r in rows]


class ReplayRuntime:
    name = "replay"
    model_id = "replay"
    supports_images = False

    def __init__(self, replies: list[ModelReply]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
    ) -> ModelReply:
        self.calls.append({"messages": messages, "tools": tools})
        if not self._replies:
            raise ReplayExhaustedError(f"model call {len(self.calls)} has no written answer")
        return self._replies.pop(0)

    def assert_consumed(self) -> None:
        if self._replies:
            raise ReplayNotConsumedError(f"{len(self._replies)} written answers were not used")
