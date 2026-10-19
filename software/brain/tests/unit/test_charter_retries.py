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

import jsonschema
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
from sd_brain.sv import read
from sd_brain.toolcall_parse import CallError, parse_tool_calls
from sd_brain.tools import PROPOSAL_TOOL, exposed_tools, registry_tool_schemas
from sd_brain.turn import OutMessage, TurnDiagnostics, TurnInput, TurnOutput

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
    # Four different drafts a turn (drafts 2 and 5 are the same one): a repeat would end
    # the turn early. The last of each turn is a rect without its size.
    distinct = [RECORDED[0], RECORDED[2], RECORDED[3], RECORDED[1]]
    assert rounds == len(distinct)
    replies = [_reply(distinct[i % rounds], i) for i in range(rounds * MAX_CHARTER_ATTEMPTS)]
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
    drafts = [RECORDED[1 + i % 3] for i in range(9)]  # three different drafts a turn
    rt = _SlowRuntime([_reply(d, i) for i, d in enumerate(drafts)], clock, per_call=6.0)
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
    backend = _Scripted()
    loop, _, _ = make_loop(backend, "draft_p1.json")
    loop._charter_attempts = 2  # pyright: ignore[reportPrivateUsage]
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


# --- the perf run where the model named a drawn area that does not exist ------------------

# The operator_message turn of the run: P1 200 m north of home, no drawn area, and the
# model (z-ai/glm-5.3-flash) sent this draft three times, then gave up with a text.
DRAWN_AR1 = {
    "search_areas": [{"kind": "drawn", "area_id": "ar-1"}],
    "reason_sv": "Sökning av en ruta på 100 x 100 m runt P1 på 40 m.",
}
GAVE_UP = (
    "Inget ritat område finns på kartan, så ramförslaget refuserades tre gånger. Rita "
    "området på kartan (eller säg till), så lägger jag fram förslaget igen: ruta 100 x 100 "
    "m runt P1 på 40 m."
)
SUGGESTED = (
    'From the operator\'s line: "search_areas": [{"kind": "rect", "center_pin_id": "P1", '
    '"width_m": 100, "height_m": 100}], "default_alt_agl_m": 40.'
)


def _text(content: str) -> ModelReply:
    return ModelReply(content=content, tool_calls=[], finish_reason="stop", engine="cloud")


def _recorded_turn() -> list[ModelReply]:
    return [*(_reply(DRAWN_AR1, i) for i in range(3)), _text(GAVE_UP)]


class _PerTurn(ReplayRuntime):
    """Replays the same written turn in every turn: a call with only the system and the
    turn block starts the next script, so the answers a turn leaves unused stay unused."""

    def __init__(self, scripts: list[list[ModelReply]]) -> None:
        super().__init__([])
        self.scripts = [list(s) for s in scripts]
        self.used: list[int] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
    ) -> ModelReply:
        if len(messages) == 2:
            self._replies = self.scripts.pop(0)
            self.used.append(0)
        self.used[-1] += 1
        return await super().chat(messages, tools, timeout_s, sampling)


def _proposal_params(tools: list[dict[str, Any]]) -> dict[str, Any]:
    tool = next(t for t in tools if t["function"]["name"] == PROPOSAL_TOOL)
    return tool["function"]["parameters"]


def _no_area_turn() -> TurnInput:
    _, fg, _ = make_loop(_Scripted(), "draft_p1.json")
    fg.sit["areas"] = []
    fg.operator_message(REQUEST)
    s = fg.situation_model()
    return TurnInput(
        turn_id=s.mission_id,
        mission_id=s.mission_id,
        trigger="operator_message",
        events=[],
        situation=s,
        chat_tail=list(fg.chat_rows),
        memory=empty_memory(s.mission_id),
        tools=exposed_tools(REG, s, empty_memory(s.mission_id), _NOW),
        now=_NOW,
    )


async def test_the_drawn_ar1_turn_ends_as_a_failed_attempt_at_the_repeat() -> None:
    rt = ReplayRuntime(_recorded_turn())
    cmd = LlmCommander(rt, _ctx(), REG, BehaviourConfig(), Sampling())
    out = await cmd.decide(_no_area_turn(), None)  # type: ignore[arg-type]
    # F1: no drawn area in the picture, so the schema offers none.
    params = _proposal_params(rt.calls[0]["tools"])
    assert "operating_area" not in params["properties"]
    alts = params["properties"]["search_areas"]["items"]["anyOf"]
    assert [a["properties"]["kind"]["enum"] for a in alts] == [["rect"], ["circle"]]
    assert "drawn" not in json.dumps(params) and "ar-1" not in json.dumps(params)
    # F2: the repair names the pin, says there is no drawn area, shows a rect on P1 and
    # the area the operator's line describes.
    repair = next(m for m in rt.calls[1]["messages"] if m["role"] == "tool")["content"]
    assert "Okänt område ar-1" in repair and "No area is drawn on the map" in repair
    assert "Map pins: P1." in repair and "Drawn areas: none." in repair
    assert '{"kind": "rect", "center_pin_id": "P1", "width_m": <metres>' in repair
    assert SUGGESTED in repair
    # F3: the same draft again ends the turn; the third call and the text are not asked.
    assert len(rt.calls) == 2 and out.diagnostics.repairs == 1
    assert out.failed_proposal is not None and not out.failed_proposal.final
    assert [(e.code, e.field) for e in out.failed_proposal.errors] == [
        ("DRAFT_INVALID", "search_areas.0")
    ]
    with pytest.raises(AssertionError):
        rt.assert_consumed()


async def test_a_text_after_a_refused_draft_is_still_a_failed_attempt() -> None:
    """F4: the model's last reply is a text: the turn is still a refused attempt."""
    rt = ReplayRuntime([_reply(DRAWN_AR1, 0), _text(GAVE_UP)])
    cmd = LlmCommander(rt, _ctx(), REG, BehaviourConfig(), Sampling())
    out = await cmd.decide(_no_area_turn(), None)  # type: ignore[arg-type]
    rt.assert_consumed()
    assert out.failed_proposal is not None and out.charter_proposal is None
    assert [m.text for m in out.messages] == [GAVE_UP]  # kept for the record only


async def test_three_drawn_ar1_turns_end_in_one_charter_failed_question() -> None:
    """The recorded turn replayed in each attempt: every turn is counted, the AI's own
    text is never posted, and after the third exactly one charter_failed question."""
    rt = _PerTurn([_recorded_turn() for _ in range(MAX_CHARTER_ATTEMPTS)])
    cmd = LlmCommander(rt, _ctx(), REG, BehaviourConfig(), Sampling())
    loop, fg, _ = make_loop(cmd, "draft_p1.json")
    fg.sit["areas"] = []
    loop.handle_event(fg.operator_message(REQUEST))
    await loop.settle()
    assert rt.used == [2] * MAX_CHARTER_ATTEMPTS  # each turn stops at the repeat
    assert [r.trigger for r in loop.records] == [
        "operator_message",
        "gate_reject_retry",
        "gate_reject_retry",
    ]
    assert [r.attempt for r in loop.records] == [1, 2, 3]
    assert all(r.output and r.output.failed_proposal for r in loop.records)
    assert [d.attempt for d in fg.decisions] == [1, 2, 3]
    assert fg.proposals == []
    failed = [m for m in fg.posted if m.refs.get("charter_failed")]
    assert len(failed) == 1 and fg.posted == failed
    assert failed[0].kind == "question" and failed[0].refs["attempts"] == 3
    # The retries keep the operator's line, so their repairs suggest the rect on P1 too.
    last_repair = [m for m in rt.calls[-1]["messages"] if m["role"] == "tool"][-1]
    assert SUGGESTED in last_repair["content"]
    await loop.settle()
    assert len(fg.posted) == 1  # nothing more after the question


async def test_a_text_that_explains_a_mission_limit_is_the_answer() -> None:
    """A draft refused only for the mission's limits may be answered with a text."""
    limit = "Högsta tillåtna höjd är 120 m."
    cmd, rt = _llm([_reply({**CORRECT, "alt_max_agl_m": 150}, 0), _text(limit)])
    loop, fg, _ = make_loop(cmd, "draft_p1.json")
    loop.handle_event(fg.operator_message("Sök runt P1 upp till 150 meter"))
    await loop.settle()
    rt.assert_consumed()
    assert [m.text for m in fg.posted] == [limit]
    assert loop.records[0].output is not None
    assert loop.records[0].output.failed_proposal is None


# --- the proposal schema follows the situation --------------------------------------------


def _schema(**patch: Any) -> dict[str, Any]:
    s = situation("draft_p1.json", **patch)
    tools = exposed_tools(REG, s, empty_memory(s.mission_id), _NOW)
    return next(t for t in tools if t.name == PROPOSAL_TOOL).openai["function"]["parameters"]


def _shapes(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    alts = params["properties"]["search_areas"]["items"]["anyOf"]
    return {a["properties"]["kind"]["enum"][0]: a["properties"] for a in alts}


def _openai_clean(params: dict[str, Any]) -> None:
    text = json.dumps(params)
    assert "oneOf" not in text and "discriminator" not in text and '"const"' not in text
    assert "$ref" not in text and "$defs" not in text and '"enum": []' not in text


def test_the_schema_with_a_drawn_area_and_a_pin() -> None:
    params = _schema()
    shapes = _shapes(params)
    assert list(shapes) == ["rect", "circle", "drawn"]
    assert shapes["drawn"]["area_id"]["enum"] == ["ar-1"]
    assert params["properties"]["operating_area"]["properties"]["area_id"]["enum"] == ["ar-1"]
    assert shapes["rect"]["center_pin_id"]["enum"] == ["P1"]
    assert shapes["circle"]["center_pin_id"]["enum"] == ["P1"]
    _openai_clean(params)


def test_the_schema_without_a_drawn_area() -> None:
    params = _schema(areas=[])
    assert list(_shapes(params)) == ["rect", "circle"]
    assert "operating_area" not in params["properties"]
    _openai_clean(params)


def test_the_schema_without_pins() -> None:
    shapes = _shapes(_schema(pins=[]))
    assert "center_pin_id" not in shapes["rect"] and "center_pin_id" not in shapes["circle"]
    assert "center" in shapes["rect"] and list(shapes) == ["rect", "circle", "drawn"]
    bare = _schema(pins=[], areas=[])
    assert list(_shapes(bare)) == ["rect", "circle"]
    _openai_clean(bare)


def test_the_registry_schema_keeps_every_option() -> None:
    """The ai_config_hash schema has no situation: every kind, no fixed ids."""
    params = _proposal_params(registry_tool_schemas(REG))
    assert list(_shapes(params)) == ["rect", "circle", "drawn"]
    assert "operating_area" in params["properties"]
    assert "enum" not in _shapes(params)["drawn"]["area_id"]


def _area(**fields: Any) -> dict[str, Any]:
    return {"search_areas": [{"kind": "rect", "width_m": 100, "height_m": 100, **fields}]}


@pytest.mark.parametrize("patch", [{}, {"areas": []}, {"pins": []}])
def test_what_the_schema_accepts_charter_draft_accepts(patch: dict[str, Any]) -> None:
    """The offered schema follows CharterDraft: a draft it allows validates, and the
    drafts it leaves out are the ones the situation cannot fill. (With pins it does not
    say "exactly one of center_pin_id and center"; the turn's check does.)"""
    check = jsonschema.Draft202012Validator(_schema(**patch))
    no_pins = patch.get("pins") == []
    center = {"center": {"lat": -35.3614, "lon": 149.1652}} if no_pins else {"center_pin_id": "P1"}
    ok = {**_area(**center), "reason_sv": "x"}
    assert check.is_valid(ok)
    CharterDraft.model_validate(ok)
    assert check.is_valid(DRAWN_AR1) == (patch.get("areas") != [])
    assert not check.is_valid({**_area(center_pin_id="P7"), "reason_sv": "x"})
    # Without pins, a rect with no centre at all: CharterDraft refuses it, so does the schema.
    bare = {**_area(), "reason_sv": "x"}
    assert check.is_valid(bare) == (not no_pins)
    with pytest.raises(ValueError):
        CharterDraft.model_validate(bare)


def test_the_schema_has_no_example_ids_and_unique_enums() -> None:
    _, fg, _ = make_loop(_Scripted(), "draft_p1.json")
    fg.sit["areas"] = fg.sit["areas"] * 2  # the same label twice
    fg.sit["pins"] = fg.sit["pins"] * 2
    s = fg.situation_model()
    tools = exposed_tools(REG, s, empty_memory(s.mission_id), _NOW)
    params = next(t for t in tools if t.name == PROPOSAL_TOOL).openai["function"]["parameters"]
    shapes = _shapes(params)
    assert shapes["rect"]["center_pin_id"]["enum"] == ["P1"]
    assert shapes["drawn"]["area_id"]["enum"] == ["ar-1"]
    assert "e.g." not in json.dumps(params)
    no_pins = _shapes(_schema(pins=[]))
    assert "pin" not in no_pins["rect"]["kind"]["description"]


def test_the_prompt_shows_a_rect_first_and_an_area_id_only_with_its_context() -> None:
    examples = yaml.safe_load(_EXAMPLES_FILE.read_text(encoding="utf-8"))
    drafts = [ex for ex in examples if (ex.get("tool") or {}).get("name") == PROPOSAL_TOOL]
    assert drafts[0]["tool"]["arguments"]["search_areas"][0]["kind"] == "rect"
    for ex in drafts:
        for area in ex["tool"]["arguments"]["search_areas"]:
            if area["kind"] == "drawn":
                assert area["area_id"] in ex.get("context", "")


class _Reads:
    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}


async def test_a_text_from_before_the_refusal_is_not_the_limit_answer() -> None:
    """Only a text written after a draft refused for a limit answers the request."""

    class _Fails(ReplayRuntime):
        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            timeout_s: float,
            sampling: Sampling,
        ) -> ModelReply:
            if len(self.calls) == 2:
                self.calls.append({})
                raise ModelTimeoutError("slow")
            return await super().chat(messages, tools, timeout_s, sampling)

    note = RawToolCall(id="n0", name="brain_note", arguments_json='{"text": "P1"}')
    first = ModelReply(content="Jag kollar.", tool_calls=[note], finish_reason="tool_calls")
    rt = _Fails([first, _reply({**CORRECT, "alt_max_agl_m": 150}, 1)])
    cmd = LlmCommander(rt, _ctx(), REG, BehaviourConfig(), Sampling())
    out = await cmd.decide(_no_area_turn(), _Reads())  # type: ignore[arg-type]
    assert out.failed_proposal is not None
    assert [e.code for e in out.failed_proposal.errors] == ["CHARTER_EXCEEDS_LIMITS"]


def test_the_suggestion_leaves_out_what_the_mission_refuses() -> None:
    s = situation("draft_p1.json", areas=[])
    tools = exposed_tools(REG, s, empty_memory(s.mission_id), _NOW)
    lim = current_limits(s)
    high = read("Sök en ruta på 100 gånger 100 meter runt P1 på 150 meters höjd")
    draft = {**CORRECT, "default_alt_agl_m": 150}
    _, err = parse_tool_calls(_reply(draft), tools, REG, s, lim, high)
    assert err is not None and err.errors[0].code == "CHARTER_EXCEEDS_LIMITS"
    assert '"search_areas": [{"kind": "rect", "center_pin_id": "P1"' in err
    assert '"default_alt_agl_m": 150' not in err
    huge = read("Sök en ruta på 5000 gånger 5000 meter runt P1")
    _, err = parse_tool_calls(_reply(DRAWN_AR1), tools, REG, s, lim, huge)
    assert err is not None and "From the operator's line" not in err
    assert "Map pins: P1." in err


def test_the_retry_turn_shows_the_options_at_once() -> None:
    """The retry's first call already sees the pins, the areas and the suggestion."""
    turn = _no_area_turn().model_copy(
        update={
            "trigger": "gate_reject_retry",
            "charter_errors": [CharterError(code="DRAFT_INVALID", detail_sv="Okänt område.")],
            "attempt": 2,
        }
    )
    block = _ctx().turn_block(turn)
    assert "Map pins: P1." in block and "Drawn areas: none." in block
    assert SUGGESTED in block


async def test_a_retry_that_answers_with_text_is_still_a_failed_attempt() -> None:
    """A retry turn that proposes nothing is counted too: no silent end of the request,
    and after the third attempt the charter_failed question."""
    rt = _PerTurn([_recorded_turn(), [_text(GAVE_UP)], [_text(GAVE_UP)]])
    cmd = LlmCommander(rt, _ctx(), REG, BehaviourConfig(), Sampling())
    loop, fg, _ = make_loop(cmd, "draft_p1.json")
    fg.sit["areas"] = []
    loop.handle_event(fg.operator_message(REQUEST))
    await loop.settle()
    assert rt.used == [2, 1, 1]
    assert [r.attempt for r in loop.records] == [1, 2, 3]
    assert all(r.output and r.output.failed_proposal for r in loop.records)
    failed = [m for m in fg.posted if m.refs.get("charter_failed")]
    assert len(failed) == 1 and fg.posted == failed
    assert loop._charter_attempts == 0  # pyright: ignore[reportPrivateUsage]


async def test_a_retry_that_ends_without_a_proposal_ends_the_request() -> None:
    """A retry answered with a limit's explanation: posted, and the next request starts
    with all its attempts."""
    limit = "Högsta tillåtna höjd är 120 m."
    explains = TurnOutput(
        messages=[OutMessage(text=limit)], diagnostics=TurnDiagnostics(backend="llm")
    )
    backend = _Scripted(_proposal(CORRECT), explains)
    loop, fg, _ = make_loop(backend, "draft_p1.json")
    fg.responses = [{"match": {"skill": "propose_charter_change"}, "result": REFUSED, "times": 1}]
    loop.handle_event(fg.operator_message(REQUEST))
    await loop.settle()
    assert [t.trigger for t in backend.turns] == ["operator_message", "gate_reject_retry"]
    assert [m.text for m in fg.posted] == [limit]
    assert loop._charter_attempts == 0  # pyright: ignore[reportPrivateUsage]
