"""A charter request that the model gets wrong: refused in the turn, retried, given up once.

Replays the five drafts of the perf run where z-ai/glm-5.3-flash never produced a card
for "Flyg till P1 och sök av en ruta på 100 gånger 100 meter där, på 40 meters höjd.
Landa hemma sen." (P1 is a pin 200 m north of home): the brain now refuses each of them
in the turn, before any POST, and a correct draft becomes a proposal.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from brain_helpers import REG, make_loop, situation

from sd_brain.backends import LlmCommander
from sd_brain.charter_draft import CharterDraft, CharterError
from sd_brain.config import BehaviourConfig
from sd_brain.constants import MAX_CHARTER_ATTEMPTS
from sd_brain.context import ContextBuilder
from sd_brain.loop import CommanderLoop
from sd_brain.memory import empty_memory
from sd_brain.runtime.base import ModelReply, ModelTimeoutError, RawToolCall, Sampling
from sd_brain.runtime.replay import ReplayRuntime
from sd_brain.state_view import current_limits
from sd_brain.toolcall_parse import CallError, parse_tool_calls
from sd_brain.tools import PROPOSAL_TOOL, exposed_tools
from sd_brain.turn import TurnDiagnostics, TurnInput, TurnOutput

REQUEST = (
    "Flyg till P1 och sök av en ruta på 100 gånger 100 meter där, på 40 meters höjd. "
    "Landa hemma sen."
)
RECT_P1 = {"kind": "rect", "center_pin_id": "P1", "width_m": 100, "height_m": 100}
INVENTED = ["goto", "search", "rtl"]  # "search" and "rtl" are not registry names
REASON = "Sökning av en ruta på 100 x 100 m vid P1."

# The five recorded proposals, as far as the run's log kept them.
RECORDED: list[dict[str, Any]] = [
    # 1: the pin before the snapshot had it (and invented skill names)
    {"search_areas": [RECT_P1], "default_alt_agl_m": 40, "allowed_skills": INVENTED},
    # 2: a rect without its size
    {
        "search_areas": [{"kind": "rect", "center_pin_id": "P1"}],
        "default_alt_agl_m": 40,
        "allowed_skills": INVENTED,
    },
    # 3: a 200 x 200 m flight area around P1, without home
    {
        "operating_area": {"kind": "rect", "center_pin_id": "P1", "width_m": 200, "height_m": 200},
        "search_areas": [RECT_P1],
        "default_alt_agl_m": 40,
        "allowed_skills": INVENTED,
    },
    # 4: a flight area at lat 0, lon 0
    {
        "operating_area": {
            "kind": "rect",
            "center": {"lat": 0, "lon": 0},
            "width_m": 200,
            "height_m": 200,
        },
        "search_areas": [RECT_P1],
        "default_alt_agl_m": 40,
        "allowed_skills": INVENTED,
    },
    # 5: a rect without its size again
    {
        "search_areas": [{"kind": "rect", "center_pin_id": "P1"}],
        "default_alt_agl_m": 40,
        "allowed_skills": INVENTED,
    },
]
RECORDED = [{**d, "reason_sv": REASON} for d in RECORDED]
CORRECT = {"search_areas": [RECT_P1], "default_alt_agl_m": 40, "reason_sv": REASON}


def _reply(args: dict[str, Any], i: int = 0) -> ModelReply:
    return ModelReply(
        content=None,
        tool_calls=[RawToolCall(id=f"c{i}", name=PROPOSAL_TOOL, arguments_json=json.dumps(args))],
        finish_reason="tool_calls",
        engine="cloud",
    )


def _parse(args: dict[str, Any], **patch: Any) -> tuple[list[Any], CallError | None]:
    s = situation("draft_p1.json", **patch)
    now = dt.datetime(2026, 1, 1, 0, 1, tzinfo=dt.UTC)
    tools = exposed_tools(REG, s, empty_memory(s.mission_id), now)
    return parse_tool_calls(_reply(args), tools, REG, s, current_limits(s))


@pytest.mark.parametrize("i", range(5))
def test_every_recorded_draft_is_refused_in_the_turn(i: int) -> None:
    calls, err = _parse(RECORDED[i])
    assert calls == [] and err is not None
    assert err.kind == "proposal" and err.detail_sv.endswith(".")
    assert "propose_charter_change again" in err


def test_the_recorded_drafts_are_refused_for_their_own_fault() -> None:
    """Without the invented skill names each draft still fails, for the reason the run
    showed (a missing size, a flight area the AI may no longer draw)."""
    faults = []
    for d in RECORDED[1:]:
        rest = {k: v for k, v in d.items() if k != "allowed_skills"}
        _, err = _parse(rest)
        assert err is not None
        faults.append(err.field or "")
    assert faults[0] == "search_areas.0.width_m"  # the union's tag is not in the path
    assert faults[1].startswith("operating_area") and faults[2].startswith("operating_area")
    assert faults[3] == "search_areas.0.width_m"
    _, skills = _parse(RECORDED[0])
    assert skills is not None and skills.field == "allowed_skills"
    assert "search, rtl" in skills and "survey_sector" in skills
    # Draft 1 without the pin in the picture: the unknown pin is named.
    _, no_pin = _parse({k: v for k, v in RECORDED[0].items() if k != "allowed_skills"}, pins=[])
    assert no_pin is not None and "Okänd nål P1" in no_pin.detail_sv
    assert no_pin.field == "search_areas.0"


def test_draft_1_is_fine_once_the_pin_is_there() -> None:
    calls, err = _parse({k: v for k, v in RECORDED[0].items() if k != "allowed_skills"})
    assert err is None and calls[0].draft is not None


def test_the_correct_draft_becomes_a_proposal() -> None:
    calls, err = _parse(CORRECT)
    assert err is None and calls[0].kind == "proposal"
    with_skills = {**CORRECT, "allowed_skills": ["goto", "survey_sector", "return_home"]}
    assert _parse(with_skills)[1] is None


def test_mission_limits_are_checked_in_the_turn() -> None:
    _, err = _parse({**CORRECT, "alt_max_agl_m": 150})
    assert err is not None and err.field == "alt_max_agl_m" and "120" in err
    _, err = _parse({**CORRECT, "max_speed_ms": 30})
    assert err is not None and err.field == "max_speed_ms"


def _llm(replies: list[ModelReply], **cfg: Any) -> tuple[LlmCommander, ReplayRuntime]:
    rt = ReplayRuntime(replies)
    ctx = ContextBuilder(REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000)
    return LlmCommander(rt, ctx, REG, BehaviourConfig(**cfg), Sampling()), rt


async def test_bad_drafts_are_repaired_in_the_turn_without_a_post() -> None:
    replies = [_reply(d, i) for i, d in enumerate(RECORDED[1:4])] + [_reply(CORRECT, 9)]
    cmd, rt = _llm(replies)
    loop, fg, _ = make_loop(cmd, "draft_p1.json")
    loop.handle_event(fg.operator_message(REQUEST))
    await loop.settle()
    assert len(rt.calls) == 4  # the first try and three repairs, one turn
    assert len(fg.proposals) == 1  # only the correct draft was sent
    assert len(loop.records) == 1 and loop.records[0].proposal is not None
    assert loop.records[0].proposal.ok
    out = loop.records[0].output
    assert out is not None and out.diagnostics.repairs == 3 and out.diagnostics.error is None
    # The repair texts name the fault and how to fix it.
    tool_msgs = [m for m in rt.calls[-1]["messages"] if m["role"] == "tool"]
    assert any("width_m" in m["content"] for m in tool_msgs)
    assert any("Leave operating_area out" in m["content"] for m in tool_msgs)


async def test_after_the_last_attempt_one_charter_failed_line_and_no_more_calls() -> None:
    rounds = 1 + BehaviourConfig().max_proposal_repair_rounds
    replies = [_reply(RECORDED[i % 5], i) for i in range(rounds * MAX_CHARTER_ATTEMPTS)]
    cmd, rt = _llm(replies)
    loop, fg, _ = make_loop(cmd, "draft_p1.json")
    loop.handle_event(fg.operator_message(REQUEST))
    await loop.settle()
    assert len(rt.calls) == rounds * MAX_CHARTER_ATTEMPTS
    rt.assert_consumed()
    assert fg.proposals == []  # nothing ever reached the ground station
    failed = [m for m in fg.posted if m.refs.get("charter_failed")]
    assert len(failed) == 1 and fg.posted == failed
    # the last draft's own codes: a rect without its size, invented skill names
    assert failed[0].kind == "question"
    assert failed[0].refs["codes"] == ["DRAFT_INVALID", "SKILL_UNKNOWN"]
    assert "invalid" not in failed[0].text and "operating_area" not in failed[0].text
    assert failed[0].in_reply_to == fg.chat_rows[0].id
    assert [r.trigger for r in loop.records] == [
        "operator_message",
        "gate_reject_retry",
        "gate_reject_retry",
    ]
    # Refused drafts are not parse errors: the AI is not degraded by them.
    assert all(r.error is None for r in loop.records)
    assert loop.heartbeat()["status"] != "degraded"
    logs = fg.decisions
    assert [d.attempt for d in logs] == [1, 2, 3]
    assert all(d.proposal_ok is False and d.proposal_errors for d in logs)
    assert logs[0].proposal_draft is not None and "search_areas" in logs[0].proposal_draft
    await loop.settle()
    assert len(rt.calls) == rounds * MAX_CHARTER_ATTEMPTS  # nothing more


class _Scripted:
    """Prepared outputs, one per turn; ``during`` runs inside the first turn."""

    name = "cloud"

    def __init__(self, *outs: TurnOutput, during: Any = None) -> None:
        self.outs = list(outs)
        self.turns: list[TurnInput] = []
        self.during = during

    async def decide(self, turn: TurnInput, tools: Any) -> TurnOutput:
        self.turns.append(turn)
        if self.during is not None:
            during, self.during = self.during, None
            during()
        if self.outs:
            return self.outs.pop(0)
        return TurnOutput(diagnostics=TurnDiagnostics(backend="llm"))


def _proposal(args: dict[str, Any]) -> TurnOutput:
    return TurnOutput(
        charter_proposal=CharterDraft.model_validate(args),
        diagnostics=TurnDiagnostics(backend="llm"),
    )


REFUSED = {
    "ok": False,
    "errors": [
        {"code": "HOME_OUTSIDE_AREA", "detail_sv": "Hemplatsen ligger utanför flygområdet."}
    ],
}


async def test_an_operator_line_keeps_the_charter_errors() -> None:
    """R3: the retry an operator line overtakes is merged into that line's turn."""
    holder: dict[str, CommanderLoop] = {}

    def write() -> None:
        loop = holder["loop"]
        loop.handle_event(fg.operator_message("Använd P1."))

    backend = _Scripted(_proposal(CORRECT), _proposal(CORRECT), during=write)
    loop, fg, _ = make_loop(backend, "draft_p1.json")
    holder["loop"] = loop
    fg.responses = [{"match": {"skill": "propose_charter_change"}, "result": REFUSED, "times": 1}]
    loop.handle_event(fg.operator_message(REQUEST))
    await loop.settle()
    assert [t.trigger for t in backend.turns] == ["operator_message", "operator_message"]
    second = backend.turns[1]
    assert [e.code for e in second.charter_errors] == ["HOME_OUTSIDE_AREA"]
    assert second.attempt == 2 and second.retry_of is not None
    assert len(fg.proposals) == 2 and loop.records[1].proposal is not None
    assert loop.records[1].proposal.ok


async def test_attempts_start_over_when_the_pilot_decides() -> None:
    nothing = TurnOutput(diagnostics=TurnDiagnostics(backend="llm"))
    backend = _Scripted(
        _proposal(CORRECT), _proposal(CORRECT), nothing, nothing, _proposal(CORRECT)
    )
    loop, fg, _ = make_loop(backend, "draft_p1.json")
    fg.responses = [{"match": {"skill": "propose_charter_change"}, "result": REFUSED}]
    loop.handle_event(fg.operator_message(REQUEST))
    await loop.settle()
    # two refused attempts, then the AI says nothing more (it waits)
    assert [r.attempt for r in loop.records] == [1, 2, None]
    loop.handle_event({"type": "event.charter", "body": {"status": "rejected", "version": 1}})
    await loop.settle()
    loop.handle_event(fg.operator_message(REQUEST))
    await loop.settle()
    # a new request after the pilot's decision: the first attempt again
    asked = [r for r in loop.records if r.trigger == "operator_message"]
    assert len(asked) == 2 and asked[-1].attempt == 1
    assert not any(m.refs.get("charter_failed") for m in fg.posted)


async def test_the_situation_is_read_after_the_chat() -> None:
    backend = _Scripted()
    loop, fg, _ = make_loop(backend, "draft_p1.json")
    order: list[str] = []
    chat, sit = fg.chat, fg.situation

    async def chat_first(*a: Any) -> Any:
        order.append("chat")
        return await chat(*a)

    async def then_situation(*a: Any) -> Any:
        order.append("situation")
        return await sit(*a)

    fg.chat = chat_first  # type: ignore[method-assign]
    fg.situation = then_situation  # type: ignore[method-assign]
    loop.handle_event(fg.operator_message(REQUEST))
    await loop.settle()
    assert order[:2] == ["chat", "situation"]


async def test_mission_created_does_not_propose_when_the_operator_wrote() -> None:
    """The operator's line gets its own turn: the mission_created turn offers no
    charter tool then, so two proposals cannot race."""
    backend = _Scripted()
    loop, fg, _ = make_loop(backend, "draft_p1.json")
    fg.operator_message(REQUEST)  # in the chat, its event not yet here
    created = {"type": "event.mission", "mission_id": fg.mission_id, "body": {"created": True}}
    loop.handle_event(created)
    await loop.settle()
    assert backend.turns[0].trigger == "mission_created"
    assert PROPOSAL_TOOL not in {t.name for t in backend.turns[0].tools}
    loop2, fg2, _ = make_loop(_Scripted(), "draft_p1.json")
    backend2 = loop2.backend
    loop2.handle_event(
        {"type": "event.mission", "mission_id": fg2.mission_id, "body": {"created": True}}
    )
    await loop2.settle()
    assert PROPOSAL_TOOL in {t.name for t in backend2.turns[0].tools}  # type: ignore[attr-defined]


def test_mission_created_says_why_it_does_not_propose() -> None:
    s = situation("draft_p1.json")
    _, fg, _ = make_loop(_Scripted(), "draft_p1.json")
    fg.operator_message(REQUEST)
    tools = exposed_tools(REG, s, empty_memory(s.mission_id), _NOW)
    turn = TurnInput(
        turn_id=s.mission_id,
        mission_id=s.mission_id,
        trigger="mission_created",
        events=[],
        situation=s,
        chat_tail=list(fg.chat_rows),
        memory=empty_memory(s.mission_id),
        tools=[t for t in tools if t.name != PROPOSAL_TOOL],
        now=_NOW,
    )
    assert "egen tur" in _ctx().turn_block(turn)
    assert "egen tur" not in _ctx().turn_block(turn.model_copy(update={"tools": tools}))


# --- what the review asked for -----------------------------------------------------------

_NOW = dt.datetime(2026, 1, 1, 0, 1, tzinfo=dt.UTC)


def _ctx() -> ContextBuilder:
    return ContextBuilder(REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000)


def test_every_fault_of_a_draft_is_named_in_one_repair() -> None:
    """Draft 3 had a flight area the AI may not draw and invented skill names: both are
    in the one repair text, so a single round can fix them."""
    _, err = _parse(RECORDED[2])
    assert err is not None
    fields = [e.field or "" for e in err.errors]
    assert any(f.startswith("operating_area") for f in fields) and "allowed_skills" in fields
    assert "Leave operating_area out" in err and "search, rtl" in err
    assert {e.code for e in err.errors} == {"DRAFT_INVALID", "SKILL_UNKNOWN"}


@pytest.mark.parametrize(
    ("patch", "code", "field"),
    [
        # "på 40 meters höjd" as a band of one height: the ground's ALT_BAND_INVALID
        ({"alt_min_agl_m": 40, "alt_max_agl_m": 40}, "ALT_BAND_INVALID", "alt_min_agl_m"),
        ({"alt_max_agl_m": 30}, "ALT_BAND_INVALID", "alt_min_agl_m"),  # search height 40
        ({"max_range_from_pilot_m": 900}, "CHARTER_EXCEEDS_LIMITS", "max_range_from_pilot_m"),
        ({"battery_rtl_pct": 10}, "CHARTER_EXCEEDS_LIMITS", "battery_rtl_pct"),
        ({"base_version": 7}, "BASE_VERSION_STALE", "base_version"),
        ({"alt_max_agl_m": 150}, "CHARTER_EXCEEDS_LIMITS", "alt_max_agl_m"),
    ],
)
def test_ground_refusals_are_caught_in_the_turn(
    patch: dict[str, Any], code: str, field: str
) -> None:
    _, err = _parse({**CORRECT, **patch})
    assert err is not None and err.kind == "proposal"
    assert [(e.code, e.field) for e in err.errors] == [(code, field)]
    assert err.detail_sv.endswith(".")


def test_the_tool_schema_bounds_range_and_battery() -> None:
    s = situation("draft_p1.json")
    tools = exposed_tools(REG, s, empty_memory(s.mission_id), _NOW)
    tool = next(t for t in tools if t.name == PROPOSAL_TOOL)
    props = tool.openai["function"]["parameters"]["properties"]
    assert props["max_range_from_pilot_m"]["maximum"] == 500
    assert props["battery_rtl_pct"]["minimum"] == 15


def test_values_inside_the_limits_pass() -> None:
    assert _parse({**CORRECT, "alt_max_agl_m": 60})[1] is None
    assert _parse({**CORRECT, "battery_rtl_pct": 20, "max_range_from_pilot_m": 400})[1] is None


def test_a_brain_refusal_keeps_the_grounds_code() -> None:
    far = {"kind": "rect", "center_pin_id": "P1", "width_m": 2000, "height_m": 2000}
    _, err = _parse({**CORRECT, "search_areas": [far]})
    assert err is not None and [e.code for e in err.errors] == ["AREA_BEYOND_RANGE"]
    assert "närmare hemplatsen" in err.detail_sv  # plain Swedish for the operator
    assert "smaller search area" in err  # the hint is for the model only
    assert "Choose" not in err.detail_sv


def test_an_unreadable_draft_has_a_swedish_text() -> None:
    reply = ModelReply(
        content=None,
        tool_calls=[RawToolCall(id="c0", name=PROPOSAL_TOOL, arguments_json="{not json")],
        finish_reason="tool_calls",
        engine="cloud",
    )
    s = situation("draft_p1.json")
    tools = exposed_tools(REG, s, empty_memory(s.mission_id), _NOW)
    _, err = parse_tool_calls(reply, tools, REG, s, current_limits(s))
    assert err is not None and err.kind == "proposal"
    assert err.detail_sv == "Ramförslaget gick inte att läsa."


_EXAMPLES_FILE = Path(__file__).resolve().parents[2] / "src/sd_brain/prompts/examples_sv_v1.yaml"
_DRAFT_EXAMPLES: list[dict[str, Any]] = [
    ex["tool"]["arguments"]
    for ex in yaml.safe_load(_EXAMPLES_FILE.read_text(encoding="utf-8"))
    if (ex.get("tool") or {}).get("name") == PROPOSAL_TOOL
]


def test_the_prompt_has_charter_examples() -> None:
    assert len(_DRAFT_EXAMPLES) >= 2


@pytest.mark.parametrize("args", _DRAFT_EXAMPLES)
def test_the_prompts_charter_examples_parse(args: dict[str, Any]) -> None:
    """The examples the model is shown must pass the schema they teach."""
    p1 = situation("draft_p1.json").pins[0].model_dump(mode="json")
    p2 = {**p1, "id": "P2", "lat": p1["lat"] - 0.0009}
    calls, err = _parse(args, pins=[p1, p2])
    assert err is None, str(err)
    assert calls[0].draft is not None


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _SlowRuntime(ReplayRuntime):
    """Every model call takes ``per_call`` seconds of the commander's clock."""

    def __init__(self, replies: list[ModelReply], clock: _Clock, per_call: float) -> None:
        super().__init__(replies)
        self.clock = clock
        self.per_call = per_call
        self.timeouts: list[float] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
    ) -> ModelReply:
        self.timeouts.append(timeout_s)
        self.clock.t += self.per_call
        return await super().chat(messages, tools, timeout_s, sampling)


async def test_a_slow_model_hands_the_refused_draft_back_before_the_deadline() -> None:
    """6 s calls in a 20 s turn: after the third refusal (18 s) a fourth call would end
    past the turn, so each turn ends with its refused draft as an attempt instead of the
    loop's deadline text, and the last attempt still gives the one final line."""
    clock = _Clock()
    rt = _SlowRuntime([_reply(RECORDED[1], i) for i in range(9)], clock, per_call=6.0)
    cmd = LlmCommander(rt, _ctx(), REG, BehaviourConfig(), Sampling(), clock=clock)
    loop, fg, _ = make_loop(cmd, "draft_p1.json")
    loop.handle_event(fg.operator_message(REQUEST))
    await loop.settle()
    assert len(rt.calls) == 9  # three calls in each of the three turns
    assert rt.timeouts[:3] == [15.0, 13.0, 7.0]  # never past 19 s of the turn
    assert [r.attempt for r in loop.records] == [1, 2, 3]
    assert all(r.output and r.output.failed_proposal for r in loop.records)
    assert all(r.error is None for r in loop.records) and fg.proposals == []
    assert not any("hann inte" in m.text for m in fg.posted)
    assert [m.refs.get("charter_failed") for m in fg.posted] == [True]


async def test_a_model_timeout_after_a_refused_draft_is_still_an_attempt() -> None:
    class _Fails(ReplayRuntime):
        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            timeout_s: float,
            sampling: Sampling,
        ) -> ModelReply:
            if len(self.calls) == 1:
                self.calls.append({})
                raise ModelTimeoutError("slow")
            return await super().chat(messages, tools, timeout_s, sampling)

    rt = _Fails([_reply(RECORDED[1], 0), _reply(CORRECT, 1)])
    cmd = LlmCommander(rt, _ctx(), REG, BehaviourConfig(), Sampling())
    loop, fg, _ = make_loop(cmd, "draft_p1.json")
    loop.handle_event(fg.operator_message(REQUEST))
    await loop.settle()
    first = loop.records[0]
    assert first.output is not None and first.output.failed_proposal is not None
    assert first.output.diagnostics.engine != "command"  # no command-mode answer
    assert loop.health.turn_fails == 0
    # the refusal queued the retry, and the retry's correct draft became a card
    assert [r.trigger for r in loop.records] == ["operator_message", "gate_reject_retry"]
    assert len(fg.proposals) == 1 and not any("hann inte" in m.text for m in fg.posted)


async def test_the_proposal_repair_budget_is_configurable() -> None:
    replies = [_reply(RECORDED[1], i) for i in range(2 * MAX_CHARTER_ATTEMPTS)]
    cmd, rt = _llm(replies, max_proposal_repair_rounds=1)
    loop, fg, _ = make_loop(cmd, "draft_p1.json")
    loop.handle_event(fg.operator_message(REQUEST))
    await loop.settle()
    assert len(rt.calls) == 2 * MAX_CHARTER_ATTEMPTS  # the first try and one repair a turn
    for r in loop.records:
        assert r.output is not None and r.output.failed_proposal is not None
        assert r.output.diagnostics.repairs == 1


async def test_a_fault_the_model_cannot_fix_ends_at_once() -> None:
    """Home not known yet: no repair, no retry, one final line."""
    cmd, rt = _llm([_reply(CORRECT, 0)])
    loop, fg, _ = make_loop(cmd, "draft_p1.json")
    fg.sit["home"] = None
    loop.handle_event(fg.operator_message(REQUEST))
    await loop.settle()
    assert len(rt.calls) == 1 and len(loop.records) == 1
    failed = [m for m in fg.posted if m.refs.get("charter_failed")]
    assert len(failed) == 1 and failed[0].refs["attempts"] == 1
    assert "Hempositionen är okänd" in failed[0].text


async def test_an_override_drops_the_queued_retry_and_the_attempts() -> None:
    backend = _Scripted()
    loop, _, _ = make_loop(backend, "draft_p1.json")
    errors = [CharterError.model_validate(e) for e in REFUSED["errors"]]
    loop._charter_attempts = 2  # pyright: ignore[reportPrivateUsage]
    loop._followups.append(  # pyright: ignore[reportPrivateUsage]
        ("gate_reject_retry", {"retry_of": None, "charter_errors": errors, "attempt": 3})
    )
    loop.handle_event(
        {"type": "event.operator_override", "body": {"action": "stop", "active": True}}
    )
    await loop.settle()
    assert loop._charter_attempts == 0 and loop._carry is None  # pyright: ignore[reportPrivateUsage]
    assert [t.trigger for t in backend.turns] == ["operator_override"]
    assert backend.turns[0].charter_errors == []


async def test_a_new_mission_starts_the_attempts_over() -> None:
    backend = _Scripted(_proposal(CORRECT), _proposal(CORRECT))
    loop, fg, _ = make_loop(backend, "draft_p1.json")
    fg.responses = [{"match": {"skill": "propose_charter_change"}, "result": REFUSED}]
    loop.handle_event(fg.operator_message(REQUEST))
    await loop.settle()
    assert [r.attempt for r in loop.records][:2] == [1, 2]
    assert loop._charter_attempts == 2  # pyright: ignore[reportPrivateUsage]
    other = "0190f000-0000-7000-8000-00000000abcd"
    loop.handle_event({"type": "event.mission", "mission_id": other, "body": {"created": True}})
    assert loop._charter_attempts == 0 and loop._carry is None  # pyright: ignore[reportPrivateUsage]


def test_a_decided_card_drops_old_charter_errors() -> None:
    loop, _, _ = make_loop(_Scripted(), "draft_p1.json")
    loop._carry = {"charter_errors": [], "attempt": 2}  # pyright: ignore[reportPrivateUsage]
    loop._followups.append(("gate_reject_retry", {"charter_errors": [], "attempt": 2}))  # pyright: ignore[reportPrivateUsage]
    loop.handle_event({"type": "event.charter", "body": {"status": "approved", "version": 1}})
    assert loop._carry is None  # pyright: ignore[reportPrivateUsage]
    assert all("charter_errors" not in x for _, x in loop._followups)  # pyright: ignore[reportPrivateUsage]


async def test_an_observers_line_leaves_the_carry_for_the_pilot() -> None:
    backend = _Scripted()
    loop, fg, _ = make_loop(backend, "draft_p1.json")
    loop._carry = {"charter_errors": [], "attempt": 2}  # pyright: ignore[reportPrivateUsage]
    loop.handle_event(fg.operator_message("Hur går det?", author_role="system"))
    await loop.settle()
    assert backend.turns[0].attempt is None
    assert loop._carry is not None  # pyright: ignore[reportPrivateUsage]
