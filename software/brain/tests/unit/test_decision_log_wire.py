"""The decision log must reach the ground whatever the model wrote. The ground client
sends JSON as httpx 0.28 does (UTF-8, ``ensure_ascii=False``), so a lone surrogate or
nesting too deep for the serialiser would make the request fail and the turn's audit
entry would be lost."""

from __future__ import annotations

import json

import httpx
import pytest
from brain_helpers import REG, fixture

from sd_brain.backends import GrammarBackend, LlmCommander
from sd_brain.config import BehaviourConfig, BrainConfig
from sd_brain.context import ContextBuilder
from sd_brain.decide.service import DecisionService
from sd_brain.evals.fake_ground import FakeGround
from sd_brain.evals.runner import EvalClock
from sd_brain.ground_client import GroundError
from sd_brain.health import HealthState
from sd_brain.loop import CommanderLoop
from sd_brain.runtime.base import ModelReply, RawToolCall, Sampling
from sd_brain.runtime.replay import ReplayRuntime
from sd_schemas.commander import BrainMessage, DecisionLog

ANSWER = "Jag flyger inte någonstans just nu."


class HttpxGround(FakeGround):
    """A fake ground that encodes each decision entry as the real client sends it."""

    async def post_decision(self, mission_id: str, entry: DecisionLog) -> None:
        httpx.Request("POST", "http://ground.test/decisions", json=entry.model_dump(mode="json"))
        await super().post_decision(mission_id, entry)


def _call(name: str, arguments_json: str, call_id: str = "c0") -> ModelReply:
    return ModelReply(
        content=None,
        tool_calls=[RawToolCall(id=call_id, name=name, arguments_json=arguments_json)],
        finish_reason="tool_calls",
        engine="cloud",
    )


def _text(content: str) -> ModelReply:
    return ModelReply(content=content, tool_calls=[], finish_reason="stop", engine="cloud")


async def _turn(replies: list[ModelReply], name: str = "airborne_modeB.json") -> HttpxGround:
    fx = fixture(name)
    clock = EvalClock()
    fg = HttpxGround(
        fx["situation"], next_charter=fx.get("next_charter"), reg=REG, now=clock.utc_now
    )
    rt = ReplayRuntime(replies)
    ctx = ContextBuilder(REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000)
    cmd = LlmCommander(
        rt, ctx, REG, BehaviourConfig(max_repair_rounds=2), Sampling(), context_tokens=65536
    )
    loop = CommanderLoop(
        fg,
        cmd,
        GrammarBackend(),
        REG,
        BrainConfig(),
        "test-hash",
        health=HealthState(clock.monotonic),
        decide=DecisionService([]),
        clock=clock,  # type: ignore[arg-type]
    )
    loop.mission_id = fg.mission_id
    loop.ready_since = clock.utc_now()
    loop.handle_event(fg.operator_message("Flyg till P1"))
    await loop.settle()
    return fg


def _all_logged(fg: HttpxGround, caplog: pytest.LogCaptureFixture) -> None:
    assert fg.decisions
    assert "kunde inte spara beslutsloggen" not in caplog.text
    assert "decision_log_unbuilt" not in caplog.text


async def test_a_lone_surrogate_in_estimate_params_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    est = _call("estimate", '{"skill": "goto", "params": {"pin": "P1\\ud800"}}')
    fg = await _turn([est, _text(ANSWER)])
    _all_logged(fg, caplog)
    assert fg.decisions[-1].tool_calls[0]["name"] == "estimate"


async def test_a_surrogate_tool_call_id_and_name_are_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fg = await _turn([_call("go\ud800to", '{"pin": "P1"}', call_id="c\udc00"), _text(ANSWER)])
    _all_logged(fg, caplog)
    assert fg.decisions[-1].tool_calls[0]["name"] == "go?to"


async def test_a_deep_charter_draft_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    deep = '{"a":' * 900 + "1" + "}" * 900
    args = f'{{"search_areas": [{{"kind": "rect", "x": {deep}}}], "reason_sv": "x"}}'
    replies = [_call("propose_charter_change", args)] + [_text(ANSWER)] * 8
    fg = await _turn(replies, "draft_p1.json")
    _all_logged(fg, caplog)
    drafts = [d.proposal_draft for d in fg.decisions if d.proposal_draft]
    assert drafts and "…" in json.dumps(drafts, ensure_ascii=False)


class _RefusingGround(HttpxGround):
    async def post_message(self, mission_id: str, msg: BrainMessage) -> str:
        raise GroundError("token refused", 401)


async def test_a_refused_token_on_a_write_ends_the_loop() -> None:
    fx = fixture("airborne_modeB.json")
    clock = EvalClock()
    fg = _RefusingGround(fx["situation"], reg=REG, now=clock.utc_now)
    rt = ReplayRuntime([_text(ANSWER)])
    ctx = ContextBuilder(REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000)
    cmd = LlmCommander(rt, ctx, REG, BehaviourConfig(), Sampling())
    loop = CommanderLoop(
        fg,
        cmd,
        GrammarBackend(),
        REG,
        BrainConfig(),
        "test-hash",
        health=HealthState(clock.monotonic),
        decide=DecisionService([]),
        clock=clock,  # type: ignore[arg-type]
    )
    loop.mission_id = fg.mission_id
    loop.handle_event(fg.operator_message("Hallå?"))
    with pytest.raises(GroundError) as info:
        await loop.settle()
    assert info.value.status == 401
