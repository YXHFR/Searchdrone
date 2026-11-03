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
from sd_brain.decide.service import DecisionService
from sd_brain.evals.fake_ground import FakeGround
from sd_brain.ground_client import GroundError
from sd_brain.loop import CommanderLoop
from sd_brain.runtime.base import ModelReply, RawToolCall, Sampling
from sd_brain.runtime.replay import ReplayRuntime
from sd_brain.tools import PROPOSAL_TOOL
from sd_brain.turn import TurnDiagnostics, TurnInput, TurnOutput
from sd_schemas.commander import DecisionLog

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


def _commander(replies: list[ModelReply], **cfg: Any) -> LlmCommander:
    ctx = ContextBuilder(REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000)
    return LlmCommander(ReplayRuntime(replies), ctx, REG, BehaviourConfig(**cfg), Sampling())


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


def _finished(method: str, loop: CommanderLoop, fg: FakeGround, log: str) -> bool:
    """Whether the turn whose write failed with a 503 still ran to its end."""
    if method == "submit":
        return any(m.text == "Ordern gick inte fram till markstationen." for m in fg.posted)
    if method == "propose_charter":
        outcome = loop.records[0].proposal
        return (
            outcome is not None
            and not outcome.ok
            and [e.code for e in outcome.errors] == ["GROUND_ERROR"]
        )
    if method == "post_decision":
        return "kunde inte spara beslutsloggen" in log
    return len(fg.decisions) == 1  # put_memory: the decision log after it was written


@pytest.mark.parametrize("write", WRITES.values(), ids=list(WRITES))
async def test_another_ground_error_on_a_write_costs_only_that_write(
    write: tuple[str, str, str, list[ModelReply]], caplog: pytest.LogCaptureFixture
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
    assert calls == 1
    # The turn handled the error itself and ran to its end; it was not cut short.
    assert "markstationen svarade inte" not in caplog.text
    assert _finished(method, loop, fg, caplog.text)
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
    # "Stopp!" starts the stop check beside the turn. The ground cannot take the report,
    # so the check falls back to its own chat line, and that write meets the 401.
    loop, fg, _ = make_loop(_Quiet(), decide=DecisionService([]), chat_intents="down")
    fg.post_message = _failing(401)  # type: ignore[method-assign]
    loop.handle_event(fg.operator_message("Stopp!"))
    await loop.settle()  # the side task logged it: nothing is raised here
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
    built: list[DecisionLog] = []

    def unsendable(entry: DecisionLog) -> bool:
        built.append(entry)
        return False

    monkeypatch.setattr(loop_module, "_sendable", unsendable)
    # A draft the brain refuses (a rect without its size): the entry has a draft and errors.
    refused = {**DRAFT, "search_areas": [{"kind": "rect", "center_pin_id": "P1"}]}
    replies = [_call(PROPOSAL_TOOL, refused), *[_text(ANSWER)] * 8]
    loop, fg, _ = make_loop(_commander(replies, max_repair_rounds=0), "draft_p1.json")
    loop.handle_event(fg.operator_message("Sök en ruta vid P1"))
    await loop.settle()
    full = next(e for e in built if e.proposal_draft is not None)
    assert full.proposal_errors and full.prompt_json is not None and full.tool_calls
    assert "decision_log_reduced" in caplog.text
    sent = next(d for d in fg.decisions if d.id == full.id)
    assert sent.prompt_json is None and sent.proposal_draft is None
    assert sent.proposal_errors == []
    assert sent.tool_calls == [{"name": PROPOSAL_TOOL}]  # the name only, no arguments


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


async def test_a_prompt_value_whose_text_holds_a_lone_surrogate_is_logged() -> None:
    class _Odd:
        def __str__(self) -> str:
            return "x\ud800"

    class _OddPrompt(_Quiet):
        async def decide(self, turn: TurnInput, tools: Any) -> TurnOutput:
            prompt = {"messages": [], "odd": _Odd()}
            return TurnOutput(diagnostics=TurnDiagnostics(backend="llm", prompt_json=prompt))

    loop, fg, _ = make_loop(_OddPrompt())
    loop.handle_event(fg.operator_message("Hallå?"))
    await loop.settle()
    assert fg.decisions  # the turn's audit entry was built and sent
    prompt = fg.decisions[-1].prompt_json
    assert prompt is not None and prompt["odd"] == "x?"
