"""A refused token (401) from the ground ends the turn loop, on every write the turn
makes, while other ground errors only cost the turn. The stop check runs beside the
loop: a 401 there is logged, and the next turn's read ends the loop. Plus the decision
log's last resorts: the reduced entry and a prompt value that is no JSON."""

from __future__ import annotations

import datetime as dt
import json
import logging
import uuid
from typing import Any

import pytest
from brain_helpers import REG, make_loop

import sd_brain.loop as loop_module
from sd_brain.backends import LlmCommander
from sd_brain.config import BehaviourConfig
from sd_brain.context import ContextBuilder
from sd_brain.ground_client import GroundError
from sd_brain.runtime.base import ModelReply, RawToolCall, Sampling
from sd_brain.runtime.replay import ReplayRuntime
from sd_brain.tools import PROPOSAL_TOOL
from sd_brain.turn import TurnDiagnostics, TurnInput, TurnOutput

ANSWER = "Jag flyger inte någonstans just nu."
RECT_P1 = {"kind": "rect", "center_pin_id": "P1", "width_m": 100, "height_m": 100}
DRAFT = {"search_areas": [RECT_P1], "default_alt_agl_m": 40, "reason_sv": "Sök vid P1."}


def _call(name: str, args: dict[str, Any]) -> ModelReply:
    return ModelReply(
        content=None,
        tool_calls=[RawToolCall(id="c0", name=name, arguments_json=json.dumps(args))],
        finish_reason="tool_calls",
        engine="cloud",
    )


def _text(content: str) -> ModelReply:
    return ModelReply(content=content, tool_calls=[], finish_reason="stop", engine="cloud")


def _commander(replies: list[ModelReply]) -> LlmCommander:
    ctx = ContextBuilder(REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000)
    return LlmCommander(ReplayRuntime(replies), ctx, REG, BehaviourConfig(), Sampling())


# Each write, with a turn that makes it: (ground method, fixture, operator line, answers).
WRITES: dict[str, tuple[str, str, str, list[ModelReply]]] = {
    "put_memory": ("put_memory", "airborne_modeB.json", "Hallå?", [_text(ANSWER)]),
    "submit": (
        "submit",
        "airborne_modeB.json",
        "Flyg till P1",
        [_call("goto", {"pin": "P1"}), _text("Flyger.")],
    ),
    "propose_charter": (
        "propose_charter",
        "draft_p1.json",
        "Sök en ruta vid P1",
        [_call(PROPOSAL_TOOL, DRAFT), _text(ANSWER)],
    ),
    "post_decision": ("post_decision", "airborne_modeB.json", "Hallå?", [_text(ANSWER)]),
}


def _failing(status: int) -> Any:
    async def fail(*args: Any) -> Any:
        raise GroundError("sd-ground (test)", status)

    return fail


@pytest.mark.parametrize("write", WRITES.values(), ids=list(WRITES))
async def test_a_refused_token_on_a_write_ends_the_loop(
    write: tuple[str, str, str, list[ModelReply]],
) -> None:
    method, fixture, line, replies = write
    loop, fg, _ = make_loop(_commander(replies), fixture)
    setattr(fg, method, _failing(401))
    loop.handle_event(fg.operator_message(line))
    with pytest.raises(GroundError) as info:
        await loop.settle()
    assert info.value.status == 401


@pytest.mark.parametrize("write", WRITES.values(), ids=list(WRITES))
async def test_another_ground_error_on_a_write_costs_only_the_turn(
    write: tuple[str, str, str, list[ModelReply]],
) -> None:
    method, fixture, line, replies = write
    loop, fg, _ = make_loop(_commander([*replies, *[_text(ANSWER)] * 8]), fixture)
    calls = 0
    real = getattr(fg, method)

    async def once(*args: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise GroundError("sd-ground svarar inte (test)", 503)
        return await real(*args)

    setattr(fg, method, once)
    loop.handle_event(fg.operator_message(line))
    await loop.settle()  # no exception: the loop goes on
    assert calls >= 1
    turns = len(loop.records)
    loop.handle_event(fg.operator_message("Hallå?"))
    await loop.settle()
    assert len(loop.records) > turns  # the next turn ran


class _Quiet:
    name = "cloud"

    async def decide(self, turn: TurnInput, tools: Any) -> TurnOutput:
        return TurnOutput(diagnostics=TurnDiagnostics(backend="llm"))


async def _raising(exc: BaseException) -> None:
    raise exc


async def test_a_refused_token_in_the_stop_check_is_logged_and_the_next_read_ends_the_loop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    loop, fg, _ = make_loop(_Quiet())
    await loop._side_task(_raising(GroundError("refused", 401)))  # pyright: ignore[reportPrivateUsage]
    assert any(
        r.levelno == logging.ERROR and "stoppkontrollen" in r.getMessage() for r in caplog.records
    )
    fg.situation = _failing(401)  # type: ignore[method-assign]
    loop.handle_event(fg.operator_message("Hallå?"))
    with pytest.raises(GroundError) as info:
        await loop.settle()
    assert info.value.status == 401


@pytest.mark.parametrize(
    "exc", [GroundError("sd-ground svarar inte", 503), GroundError("ingen status"), ValueError("x")]
)
async def test_other_errors_in_the_stop_check_are_raised(exc: BaseException) -> None:
    loop, _, _ = make_loop(_Quiet())
    with pytest.raises(type(exc)):
        await loop._side_task(_raising(exc))  # pyright: ignore[reportPrivateUsage]


async def test_an_entry_that_cannot_be_sent_is_sent_reduced(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(loop_module, "_sendable", lambda entry: False)
    replies = [_call("estimate", {"skill": "goto", "params": {"pin": "P1"}}), _text(ANSWER)]
    loop, fg, _ = make_loop(_commander(replies))
    loop.handle_event(fg.operator_message("Hur långt är det till P1?"))
    await loop.settle()
    assert "decision_log_reduced" in caplog.text
    entry = fg.decisions[-1]
    assert entry.prompt_json is None and entry.proposal_draft is None
    assert entry.proposal_errors == []
    assert entry.tool_calls == [{"name": "estimate"}]  # the name only, no arguments


async def test_a_prompt_value_that_is_no_json_is_logged_as_text() -> None:
    when = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    ref = uuid.UUID(int=1)

    class _Odd(_Quiet):
        async def decide(self, turn: TurnInput, tools: Any) -> TurnOutput:
            prompt = {"messages": [], "at": when, "ref": ref}
            return TurnOutput(diagnostics=TurnDiagnostics(backend="llm", prompt_json=prompt))

    loop, fg, _ = make_loop(_Odd())
    loop.handle_event(fg.operator_message("Hallå?"))
    await loop.settle()
    prompt = fg.decisions[-1].prompt_json
    assert prompt is not None
    assert prompt["at"] == str(when) and prompt["ref"] == str(ref)
