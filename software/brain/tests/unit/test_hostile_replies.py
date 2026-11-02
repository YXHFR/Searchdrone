"""Hostile model answers through the whole turn: a number too long for ``int()`` (more
than 4300 digits) or arguments nested too deep for the JSON parser (about 10 000 levels)
get a repair round, never an exception that would end the brain."""

from __future__ import annotations

import json
from typing import Any

import pytest
from brain_helpers import REG, make_loop

from sd_brain.backends import LlmCommander
from sd_brain.config import BehaviourConfig
from sd_brain.context import ContextBuilder
from sd_brain.runtime.base import ModelReply, RawToolCall, Sampling
from sd_brain.runtime.replay import ReplayRuntime
from sd_brain.tools import PROFILE_TOOL
from sd_brain.turn import TurnDiagnostics, TurnInput, TurnOutput

DEPTH = 12_000
LONG_INT = "9" * 5000
DEEP_LIST = "[" * DEPTH + "]" * DEPTH
DEEP_OBJECT = '{"a":' * DEPTH + "1" + "}" * DEPTH
ANSWER = "Jag flyger inte någonstans just nu."


def _call(name: str, arguments_json: str) -> ModelReply:
    return ModelReply(
        content=None,
        tool_calls=[RawToolCall(id="c0", name=name, arguments_json=arguments_json)],
        finish_reason="tool_calls",
        engine="cloud",
    )


def _text(content: str) -> ModelReply:
    return ModelReply(content=content, tool_calls=[], finish_reason="stop", engine="cloud")


def _raw(arguments: str) -> str:
    return f'<tool_call>{{"name": "goto", "arguments": {arguments}}}</tool_call>'


CASES: dict[str, ModelReply] = {
    "goto with a long number": _call("goto", f'{{"pin": "P1", "alt_m": {LONG_INT}}}'),
    "goto nested too deep": _call("goto", DEEP_LIST),
    "goto with a deep object": _call("goto", DEEP_OBJECT),
    "search profile nested too deep": _call(PROFILE_TOOL, DEEP_LIST),
    "text call with a long number": _text(_raw(f'{{"pin": "P1", "alt_m": {LONG_INT}}}')),
    "text call nested too deep": _text(_raw(DEEP_OBJECT)),
}


@pytest.mark.parametrize("hostile", CASES.values(), ids=list(CASES))
async def test_a_hostile_answer_gets_a_repair_round(hostile: ModelReply) -> None:
    rt = ReplayRuntime([hostile, _text(ANSWER)])
    ctx = ContextBuilder(REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000)
    cfg = BehaviourConfig(max_repair_rounds=2)
    # Room for the payload: the repair round, not the context limit, is under test.
    cmd = LlmCommander(rt, ctx, REG, cfg, Sampling(), context_tokens=65536)
    loop, fg, _ = make_loop(cmd)
    loop.handle_event(fg.operator_message("Flyg till P1"))
    await loop.settle()
    assert len(rt.calls) == 2  # the repair round, then the model's own answer
    assert "not JSON" in json.dumps(rt.calls[1]["messages"][-1])
    assert not fg.commands and not fg.proposals and not fg.profiles
    assert fg.posted[-1].text == ANSWER


class _Crashing:
    """A backend whose decide fails in a way no except foresaw."""

    name = "cloud"

    def __init__(self) -> None:
        self.turns = 0

    async def decide(self, turn: TurnInput, tools: Any) -> TurnOutput:
        self.turns += 1
        if self.turns == 1:
            json.loads(DEEP_LIST)  # RecursionError
        return TurnOutput(diagnostics=TurnDiagnostics(backend="llm"))


async def test_no_single_answer_ends_the_brain(caplog: pytest.LogCaptureFixture) -> None:
    backend = _Crashing()
    loop, fg, _ = make_loop(backend)
    loop.handle_event(fg.operator_message("Flyg till P1"))
    await loop.settle()
    assert "backend_decide_failed" in caplog.text
    assert fg.posted[-1].refs.get("turn_failed") == "parse"
    assert fg.decisions[-1].error == "parse_error"
    loop.handle_event(fg.operator_message("Hallå?"))
    await loop.settle()
    assert backend.turns == 2  # the loop went on with the next turn
