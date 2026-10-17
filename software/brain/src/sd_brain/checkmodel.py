"""``sd-brain check-model``: is the local model fast, correct and within the VRAM budget?

Calls the model every ``interval_s`` for ``duration_s`` with a fixed prompt and only the
``goto`` tool. A call counts as good when it returns ``goto`` with arguments that
validate. The graphics memory is sampled outside this process (``just check-model``
runs ``nvidia-smi`` and passes the log), since the AI process starts no programs.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import statistics
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sd_brain.ai_config import model_sha256
from sd_brain.runtime.base import ModelError, Sampling
from sd_brain.runtime.local import LocalModelRuntime
from sd_brain.tools import clean_schema
from sd_schemas.skills import GotoAiParams

__all__ = ["VRAM_LIMIT_MB", "check_model", "vram_peak", "write_report"]

VRAM_LIMIT_MB = 7000
PROMPT = (
    "Flyg till P1.\n\nLÄGE: i luften, 40 m över hem, ram v1 installerad. "
    "Kartnålar: P1 200 m norr om hem."
)


def _goto_tool() -> dict[str, Any]:
    params = clean_schema(GotoAiParams.model_json_schema())
    params["additionalProperties"] = False
    return {
        "type": "function",
        "function": {"name": "goto", "description": "Fly to a map pin.", "parameters": params},
    }


def vram_peak(path: Path | None) -> int | None:
    """The highest value (MB) in an ``nvidia-smi --format=csv,noheader,nounits`` log."""
    if path is None or not path.is_file():
        return None
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        first = line.split(",")[0].strip()
        if first.isdigit():
            values.append(int(first))
    return max(values) if values else None


async def check_model(
    runtime: LocalModelRuntime,
    *,
    duration_s: float,
    interval_s: float,
    vram_log: Path | None,
    image: str,
) -> tuple[int, dict[str, Any]]:
    await runtime.check_key()
    tool = _goto_tool()
    latencies: list[int] = []
    calls = ok_calls = 0
    end = time.monotonic() + duration_s
    while True:
        start = time.monotonic()
        calls += 1
        try:
            reply = await runtime.chat(
                [{"role": "user", "content": PROMPT}], [tool], 15.0, Sampling(max_tokens=128)
            )
            latencies.append(int((time.monotonic() - start) * 1000))
            if reply.tool_calls and reply.tool_calls[0].name == "goto":
                GotoAiParams.model_validate(json.loads(reply.tool_calls[0].arguments_json))
                ok_calls += 1
        except (ModelError, ValidationError, json.JSONDecodeError):
            latencies.append(int((time.monotonic() - start) * 1000))
        if time.monotonic() + interval_s > end:
            break
        await asyncio.sleep(max(0.0, interval_s - (time.monotonic() - start)))
    peak = await asyncio.to_thread(vram_peak, vram_log)
    lat = sorted(latencies)
    report = {
        "calls": calls,
        "ok_calls": ok_calls,
        "p50_ms": statistics.median(lat) if lat else None,
        "p95_ms": lat[max(0, round(0.95 * len(lat)) - 1)] if lat else None,
        "vram_peak_mb": peak,
        "image": image,
        "model_sha256": model_sha256(),
    }
    good = ok_calls == calls and peak is not None and peak <= VRAM_LIMIT_MB
    return (0 if good else 1), report


def write_report(report_dir: Path, report: dict[str, Any]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    path = report_dir / f"check-model-{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path
