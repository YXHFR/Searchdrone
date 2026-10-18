"""``sd-brain perf-cloud``: how fast the cloud model answers one chat turn.

Each run sends the real system prompt, every exposable tool and one Swedish operator line,
the same shape as a turn, through :class:`CloudModelRuntime` (reasoning effort, provider
routing and the fallback model included). The report holds latencies, which model
answered and the error kinds; never the key, the prompt or the answer.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Any

import httpx

from sd_brain.ai_config import prompt_text
from sd_brain.config import CloudConfig
from sd_brain.constants import LLM_CALL_TIMEOUT_S
from sd_brain.runtime.base import ModelError, Sampling
from sd_brain.runtime.cloud import CloudModelRuntime
from sd_brain.tools import registry_tool_schemas
from sd_schemas.skills import SKILL_REGISTRY

__all__ = ["OPERATOR_LINE", "TARGET_P50_MS", "measure_cloud", "skip_line"]

OPERATOR_LINE = "Hur mycket batteri har vi kvar, och var är drönaren nu?"
# The aim for one turn; the hard limit is LLM_CALL_TIMEOUT_S, after which the command
# mode answers.
TARGET_P50_MS = 5000


def skip_line() -> str:
    return "hoppar över molnets svarstid (ingen SD_CLOUD_API_KEY eller [cloud] model)"


def _messages() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": prompt_text("system_v1.md")},
        {"role": "user", "content": OPERATOR_LINE},
    ]


async def measure_cloud(
    cfg: CloudConfig, key: str, client: httpx.AsyncClient, *, runs: int
) -> dict[str, Any]:
    runtime = CloudModelRuntime(cfg, key, client)
    tools = registry_tool_schemas(SKILL_REGISTRY)
    sampling = Sampling(temperature=0.2, top_p=0.8, max_tokens=512)
    latencies: list[int] = []
    models: dict[str, int] = {}
    errors: dict[str, int] = {}
    tool_calls = 0
    for _ in range(max(1, runs)):
        start = time.monotonic()
        try:
            reply = await runtime.chat(_messages(), tools, LLM_CALL_TIMEOUT_S, sampling)
        except ModelError as exc:
            errors[exc.kind] = errors.get(exc.kind, 0) + 1
            continue
        latencies.append(int((time.monotonic() - start) * 1000))
        name = reply.model_id or "?"
        models[name] = models.get(name, 0) + 1
        tool_calls += bool(reply.tool_calls)
    lat = sorted(latencies)
    p95 = lat[min(len(lat) - 1, math.ceil(0.95 * len(lat)) - 1)] if lat else None
    p50 = int(statistics.median(lat)) if lat else None
    return {
        "measurement": "cloud_turn_latency",
        "model": cfg.model,
        "fallback_model": cfg.fallback_model,
        "reasoning_effort": cfg.reasoning_effort,
        "runs": max(1, runs),
        "answered": len(lat),
        "with_tool_call": tool_calls,
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "latency_max_ms": lat[-1] if lat else None,
        "answered_by": models,
        "errors": errors,
        "target_p50_ms": TARGET_P50_MS,
        "timeout_s": LLM_CALL_TIMEOUT_S,
        "ok": p50 is not None and p50 <= TARGET_P50_MS and not errors,
    }
