"""Shared test helpers for sd-brain: fixtures, a fake ground station and fake servers."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from sd_brain.backends import GrammarBackend
from sd_brain.backends.base import CommanderBackend
from sd_brain.config import BrainConfig
from sd_brain.decide.service import DecisionService
from sd_brain.evals.fake_ground import FakeGround
from sd_brain.evals.registry import foundation_registry
from sd_brain.evals.runner import EvalClock
from sd_brain.health import HealthState
from sd_brain.loop import CommanderLoop
from sd_schemas.commander import Situation

FIXTURES = Path(__file__).resolve().parents[1] / "src/sd_brain/evals/fixtures"
SCENARIOS = Path(__file__).resolve().parents[1] / "src/sd_brain/evals/scenarios"
GOLDEN = Path(__file__).resolve().parent / "golden"
REG = foundation_registry()


def fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def situation(name: str = "airborne_modeB.json", **patch: Any) -> Situation:
    data = copy.deepcopy(fixture(name)["situation"])
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    return Situation.model_validate(data)


def make_loop(
    backend: CommanderBackend,
    name: str = "airborne_modeB.json",
    *,
    cfg: BrainConfig | None = None,
    decide: DecisionService | None = None,
    chat_intents: bool = True,
) -> tuple[CommanderLoop, FakeGround, EvalClock]:
    fx = fixture(name)
    clock = EvalClock()
    fg = FakeGround(
        fx["situation"],
        next_charter=fx.get("next_charter"),
        reg=REG,
        now=clock.utc_now,
        chat_intents=chat_intents,
    )
    loop = CommanderLoop(
        fg,
        backend,
        GrammarBackend(),
        REG,
        cfg or BrainConfig(),
        "test-hash",
        health=HealthState(clock.monotonic, has_probe=False),
        decide=decide if decide is not None else DecisionService([]),
        clock=clock,  # type: ignore[arg-type]
    )
    loop.mission_id = fg.mission_id
    loop.ready_since = clock.utc_now()
    return loop, fg, clock


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def chat_reply(content: str | None = None, calls: list[tuple[str, dict[str, Any]]] = ()) -> dict:  # type: ignore[assignment]
    return {
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "content": content,
                    "tool_calls": [
                        {
                            "id": f"c{i}",
                            "type": "function",
                            "function": {"name": n, "arguments": json.dumps(a)},
                        }
                        for i, (n, a) in enumerate(calls)
                    ],
                },
                "finish_reason": "tool_calls" if calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
