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
from sd_brain.ground_client import GroundError
from sd_brain.runtime.base import ModelReply, RawToolCall, Sampling
from sd_brain.runtime.replay import ReplayExhaustedError, ReplayRuntime
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


def _commander(rt: ReplayRuntime) -> LlmCommander:
    ctx = ContextBuilder(REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000)
    cfg = BehaviourConfig(max_repair_rounds=2)
    return LlmCommander(rt, ctx, REG, cfg, Sampling(), context_tokens=65536)


async def _two_turns(rt: ReplayRuntime, fixture: str = "airborne_modeB.json") -> Any:
    """A hostile turn, then an ordinary one: the loop must still be running."""
    loop, fg, _ = make_loop(_commander(rt), fixture)
    loop.handle_event(fg.operator_message("Flyg till P1"))
    await loop.settle()
    loop.handle_event(fg.operator_message("Hallå?"))
    await loop.settle()
    assert fg.posted[-1].text == ANSWER  # the second turn ran
    return fg


async def test_a_huge_profile_key_is_capped_in_the_decision_log() -> None:
    """A 1 MB key name would make the decision log too large to build."""
    huge = _call(PROFILE_TOOL, json.dumps({"k" * 1_000_000: 1, "free_text": "x"}))
    fg = await _two_turns(ReplayRuntime([huge, _text(ANSWER), _text(ANSWER)]))
    logged = [c for d in fg.decisions for c in d.tool_calls if c["name"] == PROFILE_TOOL]
    assert logged
    keys = json.loads(logged[0]["arguments"])["redacted"]
    assert keys and all(len(k) <= 64 for k in keys)


async def test_a_lone_surrogate_in_a_draft_is_logged() -> None:
    args = '{"search_areas": [{"kind": "rect"}], "reason_sv": "\\ud800"}'
    draft = _call("propose_charter_change", args)
    rt = ReplayRuntime([draft] + [_text(ANSWER)] * 8)  # the refused draft has its retries
    fg = await _two_turns(rt, "draft_p1.json")
    assert any(d.proposal_draft and d.proposal_draft.get("reason_sv") == "?" for d in fg.decisions)


async def test_arguments_just_under_the_parser_limit_get_a_repair_round() -> None:
    """Nested so deep that they load but the repeat check cannot dump them again."""
    for depth in range(9850, 9998):  # the exact limit depends on the stack below
        deep = '{"a":' * depth + "1" + "}" * depth
        rt = ReplayRuntime([_call("goto", deep), _text(ANSWER)])
        loop, fg, _ = make_loop(_commander(rt))
        loop.handle_event(fg.operator_message("Flyg till P1"))
        await loop.settle()
        assert len(rt.calls) == 2, depth
        assert fg.posted[-1].text == ANSWER, depth


async def test_estimate_with_params_that_are_no_object_is_a_tool_error() -> None:
    est = _call("estimate", '{"skill": "goto", "params": [1, 2]}')
    rt = ReplayRuntime([est, _text(ANSWER)])
    loop, fg, _ = make_loop(_commander(rt))
    loop.handle_event(fg.operator_message("Hur långt är det till P1?"))
    await loop.settle()
    assert len(rt.calls) == 2
    assert "params must be an object" in json.dumps(rt.calls[1]["messages"][-1])
    assert fg.posted[-1].text == ANSWER


class _Raising:
    name = "cloud"

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.turns = 0

    async def decide(self, turn: TurnInput, tools: Any) -> TurnOutput:
        self.turns += 1
        if self.turns == 1:
            raise self.exc
        return TurnOutput(diagnostics=TurnDiagnostics(backend="llm"))


async def test_a_harness_out_of_answers_still_fails_loudly() -> None:
    loop, fg, _ = make_loop(_Raising(ReplayExhaustedError("replay ran out of answers")))
    loop.handle_event(fg.operator_message("Flyg till P1"))
    with pytest.raises(ReplayExhaustedError, match="out of answers"):
        await loop.settle()


async def test_an_assertion_from_a_dependency_does_not_end_the_loop() -> None:
    backend = _Raising(AssertionError("a library checked itself"))
    loop, fg, _ = make_loop(backend)
    loop.handle_event(fg.operator_message("Flyg till P1"))
    await loop.settle()
    assert fg.decisions[-1].error == "parse_error"
    loop.handle_event(fg.operator_message("Hallå?"))
    await loop.settle()
    assert backend.turns == 2


async def test_a_refused_token_in_decide_ends_the_loop() -> None:
    loop, fg, _ = make_loop(_Raising(GroundError("token refused", 401)))
    loop.handle_event(fg.operator_message("Flyg till P1"))
    with pytest.raises(GroundError):
        await loop.settle()


async def test_a_log_line_from_a_failed_turn_cannot_be_forged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    forged = ValueError("bad input\nERROR sd_brain.loop forged line")
    loop, fg, _ = make_loop(_Raising(forged))
    loop.handle_event(fg.operator_message("Flyg till P1"))
    await loop.settle()
    failed = [r.getMessage() for r in caplog.records if "backend_decide_failed" in r.getMessage()]
    assert failed and all("\n" not in m for m in failed)


async def test_a_turn_that_fails_after_the_answer_does_not_end_the_loop(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    rt = ReplayRuntime([_text(ANSWER), _text(ANSWER)])
    loop, fg, _ = make_loop(_commander(rt))
    calls = 0
    real = loop._log_decision  # pyright: ignore[reportPrivateUsage]

    async def broken(*args: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")
        await real(*args)

    monkeypatch.setattr(loop, "_log_decision", broken)
    loop.handle_event(fg.operator_message("Flyg till P1"))
    await loop.settle()
    assert "turn_failed" in caplog.text
    loop.handle_event(fg.operator_message("Hallå?"))
    await loop.settle()
    assert len(rt.calls) == 2 and fg.decisions  # the next turn ran and was logged
