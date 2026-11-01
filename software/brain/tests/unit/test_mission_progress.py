"""The brain's fixes after perf run 20261019T092346Z (i0).

The operator's request kept as UPPDRAG, the charter line, the finish rule and backstop,
charter proposals only for an operator's line, the charter's approval reaching the AI
once installed, an honest situation text, the goto schema, a repeated refused order, a
whole-call model timeout, the slowest-call guard and the turn's diagnostics.
"""

from __future__ import annotations

import asyncio
import copy
import json
import time
from typing import Any

import pytest
from brain_helpers import REG, make_loop, situation
from pydantic import ValidationError

from sd_brain.backends import LlmCommander
from sd_brain.charter_draft import CharterError
from sd_brain.config import BehaviourConfig, BrainConfig
from sd_brain.constants import SURVEY_DEFAULT_SPACING_M
from sd_brain.context import ContextBuilder, min_survey_spacing_m, situation_text
from sd_brain.loop import DEADLINE_TEXT, FINISH_DONE_TEXT, FINISH_TEXT, PROPOSED_TEXT
from sd_brain.memory import (
    MissionRequest,
    empty_memory,
    is_acknowledgement,
    mission_request,
    to_update,
    with_changes,
)
from sd_brain.runtime.base import ModelReply, RawToolCall, Sampling
from sd_brain.runtime.replay import ReplayRuntime
from sd_brain.state_view import current_limits
from sd_brain.tools import GOTO_COORDS, PROPOSAL_TOOL, exposed_tools, skill_tool
from sd_brain.turn import (
    FailedProposal,
    OutMessage,
    TurnDiagnostics,
    TurnInput,
    TurnOutput,
)
from sd_schemas.base import new_id
from sd_schemas.probmap import SectorScore, situation_line
from sd_schemas.skills import SurveySectorAiParams

REQUEST = "Sök sektor A på 40 m. Landa hemma sen."
STOP_OVERRIDE = {
    "vehicle_id": "sim-01",
    "action": "stop",
    "underlying": None,
    "action_seq": 3,
    "set_by": "operator:demo",
    "set_at": "2026-01-01T00:00:00Z",
    "set_mono_ms": 1,
}


class Steps:
    """A backend that returns prepared outputs, one per turn, and records its inputs."""

    name = "cloud"

    def __init__(self, *outs: TurnOutput) -> None:
        self.outs = list(outs)
        self.turns: list[TurnInput] = []

    async def decide(self, turn: TurnInput, tools: Any) -> TurnOutput:
        self.turns.append(turn)
        if self.outs:
            return self.outs.pop(0)
        return TurnOutput(diagnostics=TurnDiagnostics(backend="llm"))


def _ctx() -> ContextBuilder:
    return ContextBuilder(REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000)


def _call(name: str, args: dict[str, Any], i: int = 0) -> RawToolCall:
    return RawToolCall(id=f"c{i}", name=name, arguments_json=json.dumps(args))


def _reply(*calls: RawToolCall, content: str | None = None) -> ModelReply:
    return ModelReply(
        content=content,
        tool_calls=list(calls),
        finish_reason="tool_calls" if calls else "stop",
        engine="cloud",
        model_id="test-model",
    )


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _SlowRuntime(ReplayRuntime):
    """Every call takes ``per_call`` seconds of the commander's clock."""

    def __init__(self, replies: list[ModelReply], clock: _Clock, per_call: float) -> None:
        super().__init__(replies)
        self.clock = clock
        self.per_call = per_call

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
    ) -> ModelReply:
        self.clock.t += self.per_call
        return await super().chat(messages, tools, timeout_s, sampling)


class _Hangs(ReplayRuntime):
    """A server that answers the headers and then trickles its body: the call itself
    ignores its timeout (as httpx's per-read timeout would)."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
    ) -> ModelReply:
        self.calls.append({"messages": messages, "tools": tools})
        await asyncio.sleep(10)
        raise AssertionError("not reached")


# --- O3: the operator's request (UPPDRAG) ---------------------------------------------


async def test_the_operators_request_survives_twenty_ai_lines() -> None:
    backend = Steps()
    loop, fg, clock = make_loop(backend)
    loop.handle_event(fg.operator_message(REQUEST))
    await loop.settle()
    for i in range(20):
        await loop._say(OutMessage(text=f"Sektor A pågår, rad {i}."))  # pyright: ignore[reportPrivateUsage]
    clock.t += 61
    assert loop.idle_tick()
    await loop.settle()
    turn = backend.turns[-1]
    assert turn.trigger == "idle_tick"
    assert all(m.role == "brain" for m in turn.chat_tail)  # the line left the tail
    block = _ctx().turn_block(turn)
    assert f'UPPDRAG (operatörens senaste begäran 00:00, citerad som data): "{REQUEST}"' in block
    req = mission_request(fg.mem)
    assert req is not None and req.text == REQUEST and req.flown  # asked in the air


async def test_the_request_ends_on_landing_and_a_new_one_replaces_it() -> None:
    backend = Steps()
    loop, fg, _ = make_loop(backend, "ready_charter_v1_modeB.json")
    loop.handle_event(fg.operator_message(REQUEST))
    await loop.settle()
    req = mission_request(fg.mem)
    assert req is not None and not req.flown
    # Still on the ground (before the start): kept.
    loop.handle_event({"type": "event.alert", "body": {"code": "X", "level": "warning"}})
    await loop.settle()
    assert mission_request(fg.mem) is not None
    # A newer request replaces it; a "fortsätt" does not.
    loop.handle_event(fg.operator_message("Sök sektor B i stället."))
    await loop.settle()
    loop.handle_event(fg.operator_message("Fortsätt"))
    await loop.settle()
    req = mission_request(fg.mem)
    assert req is not None and req.text == "Sök sektor B i stället."
    # Airborne, then landed and disarmed: the mission is over.
    fg.sit["vehicle"].update(airborne=True, landed=False, armed=True)
    fg.sit["mission_status"] = "ACTIVE"
    loop.handle_event({"type": "event.alert", "body": {"code": "Y", "level": "warning"}})
    await loop.settle()
    req = mission_request(fg.mem)
    assert req is not None and req.flown
    fg.sit["vehicle"].update(airborne=False, landed=True, armed=False)
    fg.sit["mission_status"] = "LANDED"
    loop.handle_event({"type": "event.alert", "body": {"code": "Z", "level": "warning"}})
    await loop.settle()
    assert mission_request(fg.mem) is None
    assert "UPPDRAG" not in _ctx().turn_block(backend.turns[-1])


def test_memory_keeps_the_start_and_the_mission_requests_apart() -> None:
    s = situation()
    mem = empty_memory(s.mission_id)
    req = MissionRequest(text="Sök A", message_id=new_id(), requested_at=_now())
    both = with_changes(mem, mission=req)
    assert mission_request(both) == req
    assert with_changes(both, mission=None).pending_operator_request is None
    # An older memory (the start request alone) still reads.
    old = mem.model_copy(
        update={
            "pending_operator_request": {
                "kind": "takeoff",
                "message_id": str(new_id()),
                "requested_at": "2026-01-01T00:00:00Z",
                "alt_m": 30,
            }
        }
    )
    from sd_brain.memory import pending_request

    start = pending_request(old)
    assert start is not None and start.alt_m == 30 and mission_request(old) is None


def _now() -> Any:
    import datetime as dt

    return dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


def test_the_charter_line_shows_the_sectors_and_the_notes() -> None:
    s = situation()
    assert s.charter is not None
    charter = s.charter.model_copy(update={"notes": 'Sök längs "bäcken" först.'})
    s = s.model_copy(update={"charter": charter})
    turn = _turn_input(s)
    block = _ctx().turn_block(turn)
    ram = block.splitlines()[0]
    assert ram.startswith("RAM: v1, sektorer A ca ")
    assert " m, mitt " in ram and "om hem; B ca " in ram
    assert "anteckningar (data): \"Sök längs 'bäcken' först.\"" in ram


def _turn_input(s: Any, trigger: Any = "idle_tick") -> TurnInput:
    return TurnInput(
        turn_id=new_id(),
        mission_id=s.mission_id,
        trigger=trigger,
        events=[],
        situation=s,
        chat_tail=[],
        memory=empty_memory(s.mission_id),
        tools=[],
        now=_now(),
    )


def test_the_prompt_has_the_finishing_rule() -> None:
    system = _ctx().system
    assert "FINISHING." in system and "call `return_home`" in system
    # A search without a find goes on while the battery allows, by the map's order.
    assert "search again instead of going home" in system and "Sektorer (POA" in system
    assert "WAITING CARDS." in system
    assert "så kör jag" not in system  # the example no longer teaches waiting


def test_the_search_again_numbers_come_from_the_code() -> None:
    """The default spacing is the executor's, the floor the survey tool's own minimum; the
    operator's words and a "come home" end the search again."""
    from sd_safety.constants import DEFAULT_SURVEY_SPACING_M

    assert SURVEY_DEFAULT_SPACING_M == DEFAULT_SURVEY_SPACING_M
    floor = min_survey_spacing_m()
    assert floor == 10
    with pytest.raises(ValidationError):
        SurveySectorAiParams(sector="A", spacing_m=floor - 0.1)
    system = _ctx().system
    assert "{" not in system.split("EXEMPEL")[0]  # every placeholder filled
    assert "one without `spacing_m` flew 30 m, so 20 m" in system
    assert "never below 10 m" in system
    assert "three times the minutes" in system and "till hemgräns" in system
    assert "the mission ends after the first round" in system
    assert "at once, also in the middle of a search" in system


def test_the_sector_line_shows_the_minutes_of_a_pass() -> None:
    score = SectorScore(
        sector_id="A",
        poa=0.4,
        covered_pct=100,
        mean_pod=0.3,
        pos_plan=0.1,
        est_min=6.3,
        psr=0.016,
        spacing_m=41.6,
    )
    line = situation_line([score])
    assert line == "Sektorer (POA kvar / täckt / medel-POD / PSR / min): A 0,40/100%/0,30/0,016/6,3"


async def test_a_plan_note_sent_with_the_action_is_kept() -> None:
    note = "Plan: 1 goto P1, 2 sök A, 3 hem."
    rt = ReplayRuntime(
        [
            _reply(
                _call("brain_note", {"text": note}, 0),
                _call("goto", {"pin": "P1", "rationale_sv": "Operatören bad om det."}, 1),
            )
        ]
    )
    cmd = LlmCommander(rt, _ctx(), REG, BehaviourConfig(), Sampling())
    loop, fg, _ = make_loop(cmd)
    loop.handle_event(fg.operator_message("Flyg till P1, sök A och landa hemma sen."))
    await loop.settle()
    assert [c.skill for c in fg.commands] == ["goto"]
    assert fg.mem.plan_note == note


# --- O4: the finish backstop ----------------------------------------------------------


def _submits(fg: Any) -> list[int]:
    """How many chat lines were posted when each order was submitted."""
    seen: list[int] = []
    orig = fg.submit

    async def submit(mid: str, req: Any) -> Any:
        seen.append(len(fg.posted))
        return await orig(mid, req)

    fg.submit = submit
    return seen


async def test_the_backstop_says_so_then_flies_home() -> None:
    backend = Steps()
    loop, fg, clock = make_loop(backend)
    seen = _submits(fg)
    clock.t += 170
    loop.idle_tick()
    await loop.settle()
    assert not any(c.skill == "return_home" for c in fg.commands)  # not yet
    clock.t += 11  # 181 s without input, an order or a skill's end
    assert loop.idle_tick()
    await loop.settle()
    assert [c.skill for c in fg.commands] == ["return_home"]
    order = fg.commands[0]
    assert order.backend == "rule" and order.params == {
        "reason": "Uppdraget är klart och inget har hänt på en stund."
    }
    assert [m.text for m in fg.posted[-2:]] == [FINISH_TEXT.format(minutes="3"), FINISH_DONE_TEXT]
    assert seen == [len(fg.posted) - 1]  # the line came first, the gate's ALLOW after
    assert all(t.trigger == "idle_tick" for t in backend.turns)
    assert len(backend.turns) == 1  # the 170 s idle tick; the backstop is no model turn
    # Once ordered, it waits another full period.
    clock.t += 10
    loop.idle_tick()
    await loop.settle()
    assert len(fg.commands) == 1


@pytest.mark.parametrize(
    "case",
    [
        "held",
        "paused",
        "pending_card",
        "active_skill",
        "operator",
        "ai_off",
        "landed",
        "stop_latch",
        "battery",
        "manual",
        "returning",
    ],
)
async def test_the_backstop_waits(case: str) -> None:
    fx = "pending_card_modeA.json" if case == "pending_card" else "airborne_modeB.json"
    backend = Steps()
    loop, fg, clock = make_loop(backend, fx)
    if case == "held":
        fg.sit["mission_status"] = "HELD"
    elif case == "paused":
        fg.sit["mission_status"] = "PAUSED"
        fg.sit["paused_by"] = "operator"
    elif case == "active_skill":
        fg.sit["active_command"] = copy.deepcopy(
            situation("airborne_survey.json").model_dump(mode="json")["active_command"]
        )
    elif case == "ai_off":
        fg.sit["ai_enabled"] = False
    elif case == "landed":
        fg.sit["vehicle"].update(airborne=False, landed=True, armed=False)
        fg.sit["mission_status"] = "LANDED"
    elif case in ("stop_latch", "battery"):
        fg.sit["override"] = copy.deepcopy(STOP_OVERRIDE)
        if case == "battery":
            fg.sit["override"]["action"] = "battery_rtl"
    elif case == "manual":
        fg.sit["vehicle"]["manual_control"] = True
    elif case == "returning":
        fg.sit["mission_status"] = "RETURNING"
        fg.sit["returning_cause"] = "operator"
    clock.t += 100
    if case == "operator":
        loop.handle_event(fg.operator_message("Hur går det?"))
        await loop.settle()
    clock.t += 100  # 200 s since the start, 100 s since the operator's line
    loop.idle_tick()
    await loop.settle()
    assert not any(c.skill == "return_home" for c in fg.commands)
    assert not any(m.text.startswith("Inget har hänt") for m in fg.posted)


def test_the_backstop_time_is_bounded() -> None:
    assert BehaviourConfig().finish_idle_s == 180
    for bad in (10, 4000):
        with pytest.raises(ValidationError):
            BehaviourConfig(finish_idle_s=bad)


async def test_the_backstop_follows_the_configured_time() -> None:
    cfg = BrainConfig.model_validate({"behaviour": {"finish_idle_s": 300}})
    loop, fg, clock = make_loop(Steps(), cfg=cfg)
    clock.t += 200
    loop.idle_tick()
    await loop.settle()
    assert fg.commands == []
    clock.t += 101
    loop.idle_tick()
    await loop.settle()
    assert [c.skill for c in fg.commands] == ["return_home"]
    assert fg.posted[-2].text == FINISH_TEXT.format(minutes="5")


# --- O2: charter proposals, the approval event, the situation text -------------------


async def test_a_charter_is_offered_only_for_an_operator_line_or_a_retry() -> None:
    refused = FailedProposal(errors=[CharterError(code="DRAFT_INVALID", detail_sv="Fel.")])
    backend = Steps(TurnOutput(failed_proposal=refused, diagnostics=TurnDiagnostics(backend="llm")))
    loop, fg, _ = make_loop(backend, "draft_p1.json")
    loop.handle_event(fg.operator_message("Sök runt P1, 100 x 100 m."))
    await loop.settle()
    loop.handle_event({"type": "event.charter", "body": {"status": "rejected", "version": 1}})
    loop.handle_event({"type": "event.alert", "body": {"code": "W", "level": "warning"}})
    await loop.settle()
    offered = {t.trigger: PROPOSAL_TOOL in {x.name for x in t.tools} for t in backend.turns}
    assert offered == {
        "operator_message": True,
        "gate_reject_retry": True,
        "charter_event": False,
    }


async def test_the_approval_wakes_the_ai_once_the_drone_has_the_charter() -> None:
    backend = Steps()
    loop, _, _ = make_loop(backend, "ready_charter_v1_modeB.json")
    for status in ("none", "pending"):
        body = {"status": "approved", "version": 1, "install_status": status}
        loop.handle_event({"type": "event.charter", "body": body})
        await loop.settle()
    assert backend.turns == []
    body = {"status": "approved", "version": 1, "install_status": "installed"}
    loop.handle_event({"type": "event.charter", "body": body})
    await loop.settle()
    assert [t.trigger for t in backend.turns] == ["charter_event"]


def test_the_situation_text_is_honest() -> None:
    s = situation()
    data = s.model_dump(mode="json")
    data["charter_installed"] = False
    card = copy.deepcopy(situation("pending_card_modeA.json").pending_approvals[0])
    card = card.model_copy(update={"kind": "charter", "subject_version": 2, "summary_sv": "Ram"})
    data["pending_approvals"] = [card.model_dump(mode="json")]
    charter = copy.deepcopy(data["charter"])
    charter["version"] = 2
    data["pending_proposal"] = {
        "id": charter["charter_id"],
        "mission_id": data["mission_id"],
        "version": 2,
        "status": "proposed",
        "body": charter,
        "proposed_by": "brain:ground",
        "approved_by": None,
        "approved_at": None,
        "install_status": "none",
        "fence_hash": None,
        "fence_verified_at": None,
        "approval_id": None,
        "created_at": "2026-01-01T00:00:00Z",
    }
    text = situation_text(type(s).model_validate(data), _now())
    assert "ram=v1 godkänd, installeras" in text
    assert "Väntar på klick: ramförslag v2\n" in text  # once, not as card and proposal
    assert "Fritt runt drönaren" not in text  # airborne
    landed = situation_text(situation("ready_charter_v1_modeB.json"), _now())
    assert "Fritt runt drönaren: ej bekräftad" in landed
    assert "ram=v1 (installerad, staket OK)" in landed


async def test_the_chat_says_a_charter_was_proposed() -> None:
    from sd_brain.charter_draft import CharterDraft

    draft = CharterDraft.model_validate(
        {
            "search_areas": [
                {"kind": "rect", "center_pin_id": "P1", "width_m": 100, "height_m": 100}
            ],
            "reason_sv": "Sökning runt P1.",
        }
    )
    backend = Steps(TurnOutput(charter_proposal=draft, diagnostics=TurnDiagnostics(backend="llm")))
    loop, fg, _ = make_loop(backend, "draft_p1.json")
    loop.handle_event(fg.operator_message("Sök 100 x 100 m runt P1."))
    await loop.settle()
    assert len(fg.proposals) == 1
    assert [m.text for m in fg.posted] == [
        "Jag har lagt fram ett ramförslag. Tryck Godkänn på kortet om det stämmer."
    ]


# --- RC4: the goto target, a repeated refused order -----------------------------------


def test_goto_requires_a_target_this_situation_has() -> None:
    goto = REG.get("goto")
    with_pins = situation("ready_charter_v1_modeB.json")
    params = skill_tool(goto, current_limits(with_pins), s=with_pins).openai["function"][
        "parameters"
    ]
    assert params["required"] == ["pin", "rationale_sv"]
    # A pin, or the sentinel for the coordinates the operator wrote.
    assert params["properties"]["pin"]["enum"] == ["P1", "P2", GOTO_COORDS]
    assert {"lat", "lon"} <= set(params["properties"])
    no_pins = situation(pins=[])
    params = skill_tool(goto, current_limits(no_pins), s=no_pins).openai["function"]["parameters"]
    assert params["required"] == ["lat", "lon", "rationale_sv"]
    assert "pin" not in params["properties"]
    assert params["additionalProperties"] is False and params["type"] == "object"
    # The registry's own schema (for the config hash) is unchanged: either target.
    plain = skill_tool(goto, None).openai["function"]["parameters"]
    assert plain["required"] == ["rationale_sv"] and {"pin", "lat", "lon"} <= set(
        plain["properties"]
    )


async def test_the_same_refused_order_twice_ends_the_turn() -> None:
    no_target = {"rationale_sv": "Flyger vidare."}
    rt = ReplayRuntime(
        [
            _reply(_call("goto", no_target, 0)),
            _reply(_call("goto", no_target, 1)),
            _reply(_call("goto", {"pin": "P1", "rationale_sv": "x"}, 2)),
        ]
    )
    behaviour = BehaviourConfig(max_repair_rounds=2)
    cmd = LlmCommander(rt, _ctx(), REG, behaviour, Sampling())
    loop, fg, _ = make_loop(cmd)
    loop.handle_event(fg.operator_message("Flyg vidare"))
    await loop.settle()
    assert len(rt.calls) == 2 and fg.commands == []
    assert fg.posted[-1].text.startswith("Jag försökte samma felaktiga order två gånger")
    assert fg.posted[-1].refs == {"turn_failed": "repeat"}
    assert _turn_log(fg).error == "parse_error"


# --- RC6: a whole-call timeout, the slowest-call guard ---------------------------------


async def test_a_trickling_model_call_times_out_as_a_whole() -> None:
    rt = _Hangs([])
    cmd = LlmCommander(rt, _ctx(), REG, BehaviourConfig(), Sampling())
    s = situation()
    mem = empty_memory(s.mission_id)
    turn = _turn_input(s, "operator_message").model_copy(
        update={"tools": exposed_tools(REG, s, mem, _now()), "budget_s": 1.6}
    )
    start = time.monotonic()
    out = await cmd.decide(turn, None)  # type: ignore[arg-type]
    assert time.monotonic() - start < 3
    assert out.diagnostics.error == "timeout"


async def test_no_round_starts_that_would_end_after_the_turn() -> None:
    clock = _Clock()
    read = _reply(_call("get_situation", {}))
    rt = _SlowRuntime([read, read, read, read], clock, per_call=6.0)
    cmd = LlmCommander(rt, _ctx(), REG, BehaviourConfig(), Sampling(), clock=clock)
    loop, fg, _ = make_loop(cmd)
    loop.handle_event(fg.operator_message("Hur ser det ut?"))
    await loop.settle()
    # 6, 12 and 18 s: a fourth 6 s call would end past the turn's 19 s.
    assert len(rt.calls) == 3
    assert _turn_log(fg).error == "deadline"
    assert fg.posted[-1].text == DEADLINE_TEXT


# --- RC7: what the decision log keeps -------------------------------------------------


async def test_the_prompt_in_the_log_has_the_offered_tools() -> None:
    rt = ReplayRuntime([_reply(content="Batteriet är på 70 %.")])
    cmd = LlmCommander(rt, _ctx(), REG, BehaviourConfig(), Sampling())
    loop, fg, _ = make_loop(cmd)
    loop.handle_event(fg.operator_message("Hur mycket batteri?"))
    await loop.settle()
    prompt = _turn_log(fg).prompt_json
    assert prompt is not None
    names = [t["function"]["name"] for t in prompt["tools"]]
    assert names == [t["function"]["name"] for t in rt.calls[0]["tools"]]
    assert "goto" in names and prompt["messages"][0]["role"] == "system"


async def test_a_turn_cut_at_the_deadline_keeps_its_diagnostics() -> None:
    rt = _Hangs([])
    cmd = LlmCommander(rt, _ctx(), REG, BehaviourConfig(), Sampling())
    loop, fg, _ = make_loop(cmd)
    loop.turn_deadline_s = 0.2
    loop.handle_event(fg.operator_message("Flyg till P1"))
    await loop.settle()
    entry = next(d for d in fg.decisions if d.trigger == "operator_message")
    assert entry.error == "deadline"
    assert entry.prompt_json is not None and entry.prompt_json["messages"]
    assert "goto" in [t["function"]["name"] for t in entry.prompt_json["tools"]]


def _turn_log(fg: Any) -> Any:
    """The decision log of the last turn (not a stop check's)."""
    return next(d for d in reversed(fg.decisions) if not d.trigger.startswith("stop"))


async def test_a_blocked_backstop_leaves_the_idle_ticks_alone() -> None:
    """A finish check that may not act is no model turn, and the AI's own idle tick
    still comes (a pilot's order card waits here, not the AI's own)."""
    backend = Steps()
    loop, fg, clock = make_loop(backend, "pending_card_modeA.json")
    clock.t += 181
    assert loop.idle_tick()  # the finish check
    await loop.settle()
    assert backend.turns == []
    assert loop.idle_tick()  # the idle tick, 181 s after the last turn
    await loop.settle()
    assert [t.trigger for t in backend.turns] == ["idle_tick"]
    assert fg.commands == []


# --- review fixes ---------------------------------------------------------------------


async def test_the_hand_back_after_flying_by_hand_restarts_the_backstop() -> None:
    loop, fg, clock = make_loop(Steps())
    fg.sit["vehicle"]["manual_control"] = True
    clock.t += 300  # flown by hand for 5 min
    loop.idle_tick()
    await loop.settle()
    fg.sit["vehicle"]["manual_control"] = False
    body = {"action": "hand_back", "via": "panel", "by": "operator:demo", "result": "ok"}
    loop.handle_event({"type": "event.vehicle_action", "body": body})
    await loop.settle()
    clock.t += 10
    loop.idle_tick()
    await loop.settle()
    assert not any(c.skill == "return_home" for c in fg.commands)
    assert not any(m.text.startswith("Inget har hänt") for m in fg.posted)


async def test_a_blocked_check_waits_a_whole_period_and_polls_seldom() -> None:
    loop, fg, clock = make_loop(Steps())
    fg.sit["mission_status"] = "HELD"
    calls = 0
    orig = fg.situation

    async def counted(mid: str) -> Any:
        nonlocal calls
        calls += 1
        return await orig(mid)

    fg.situation = counted  # type: ignore[method-assign]
    clock.t += 181
    loop.idle_tick()
    await loop.settle()
    first = calls
    for _ in range(10):  # 50 s of 5 s ticks: no new finish check
        clock.t += 5
        loop.idle_tick()
        await loop.settle()
    assert not loop._finish_due()  # pyright: ignore[reportPrivateUsage]
    assert calls - first <= 1  # at most the AI's own 60 s idle tick
    fg.sit["mission_status"] = "ACTIVE"  # free again: a whole period first
    clock.t += 100
    loop.idle_tick()
    await loop.settle()
    assert not any(c.skill == "return_home" for c in fg.commands)


async def test_the_backstop_line_comes_once_when_the_order_fails() -> None:
    from sd_brain.ground_client import GroundError

    loop, fg, clock = make_loop(Steps())

    async def down(mid: str, req: Any) -> Any:
        raise GroundError("503")

    fg.submit = down  # type: ignore[method-assign]
    clock.t += 181
    for _ in range(4):
        loop.idle_tick()
        await loop.settle()
        clock.t += 5
    assert sum(m.text.startswith("Inget har hänt") for m in fg.posted) == 1


async def test_a_find_restarts_the_backstop() -> None:
    loop, fg, clock = make_loop(Steps())
    clock.t += 150
    loop.handle_event({"type": "event.detection", "body": {"kind": "target", "label": "x"}})
    await loop.settle()
    clock.t += 60
    loop.idle_tick()
    await loop.settle()
    assert not any(c.skill == "return_home" for c in fg.commands)


async def test_an_answer_or_a_question_keeps_the_request() -> None:
    backend = Steps()
    loop, fg, _ = make_loop(backend, "ready_charter_v1_modeB.json")
    for line in (REQUEST, "Ja kör", "ok tack", "Hur mycket batteri?"):
        loop.handle_event(fg.operator_message(line))
        await loop.settle()
    req = mission_request(fg.mem)
    assert req is not None and req.text == "Hur mycket batteri?" and req.earlier == [REQUEST]
    block = _ctx().turn_block(backend.turns[-1])
    assert "Landa hemma sen." in block and "tidigare (äldst först)" in block
    assert is_acknowledgement("Ja, kör!") and not is_acknowledgement("Ja, landa här")


async def test_a_long_emoji_line_fits_the_memory() -> None:
    loop, fg, _ = make_loop(Steps(), "ready_charter_v1_modeB.json")
    for line in ("\U0001f681" * 700, "ö" * 700, "\U0001f50d" * 650):
        loop.handle_event(fg.operator_message(line))
        await loop.settle()
    req = mission_request(fg.mem)
    assert req is not None and len(req.earlier) == 2
    to_update(fg.mem)  # within the schema's 4 KiB
    size = len(json.dumps(fg.mem.pending_operator_request, separators=(",", ":")))
    assert size <= 4096


def test_a_failed_install_is_not_called_installing() -> None:
    data = situation().model_dump(mode="json")
    data["charter_installed"] = False
    data["alerts"] = [{"code": "CHARTER_INSTALL_FAILED", "level": "warning"}]
    s = type(situation()).model_validate(data)
    assert "ram=v1 godkänd, installationen misslyckades" in situation_text(s, _now())


async def test_a_repeated_bad_read_call_is_repaired_not_called_an_order() -> None:
    bad = _call("get_log", {"n": "tio"})
    rt = ReplayRuntime([_reply(bad), _reply(bad), _reply(content="Loggen är tom.")])
    cmd = LlmCommander(rt, _ctx(), REG, BehaviourConfig(max_repair_rounds=2), Sampling())
    loop, fg, _ = make_loop(cmd)
    loop.handle_event(fg.operator_message("Visa loggen"))
    await loop.settle()
    assert not any("samma felaktiga order" in m.text for m in fg.posted)


async def test_an_event_turn_without_the_proposal_tool_says_so_and_posts_no_parse_line() -> None:
    draft = {"search_areas": [{"kind": "rect", "center_pin_id": "P1", "width_m": 100}]}
    call = _call(PROPOSAL_TOOL, {**draft, "reason_sv": "Sök."})
    rt = ReplayRuntime([_reply(call), _reply(call)])
    cmd = LlmCommander(rt, _ctx(), REG, BehaviourConfig(), Sampling())
    loop, fg, _ = make_loop(cmd, "draft_p1.json")
    loop.handle_event({"type": "event.charter", "body": {"status": "rejected", "version": 1}})
    await loop.settle()
    prompt = rt.calls[0]["messages"][1]["content"]  # the turn block
    assert "OBS: ingen ram kan föreslås i den här turen" in prompt
    assert fg.proposals == []
    assert not any("turn_failed" in m.refs for m in fg.posted)
    assert PROPOSED_TEXT not in [m.text for m in fg.posted]


async def test_a_resent_install_wakes_the_ai_once() -> None:
    backend = Steps()
    loop, _, _ = make_loop(backend, "ready_charter_v1_modeB.json")
    body = {"charter_id": "c1", "status": "approved", "version": 1, "install_status": "installed"}
    for _ in range(3):
        loop.handle_event({"type": "event.charter", "body": dict(body)})
        await loop.settle()
    assert [t.trigger for t in backend.turns] == ["charter_event"]


async def test_the_charter_in_force_is_said_before_the_mission_is_ready() -> None:
    from sd_brain.loop import CHARTER_WITH_REQUEST

    loop, fg, _ = make_loop(Steps(), "ready_charter_v1_modeB.json")
    fg.sit["mission_status"] = "CHARTER_PROPOSED"  # READY comes just after
    loop.handle_event(fg.operator_message("Starta"))
    await loop.settle()
    body = {"charter_id": "c1", "status": "approved", "version": 1, "install_status": "installed"}
    loop.handle_event({"type": "event.charter", "body": body})
    await loop.settle()
    assert fg.posted[-1].text == CHARTER_WITH_REQUEST


async def test_goto_flies_to_coordinates_while_pins_exist() -> None:
    s = situation()
    pin = s.pins[0]
    args = {"pin": GOTO_COORDS, "lat": pin.lat, "lon": pin.lon, "rationale_sv": "Koordinaterna."}
    rt = ReplayRuntime([_reply(_call("goto", args))])
    cmd = LlmCommander(rt, _ctx(), REG, BehaviourConfig(), Sampling())
    loop, fg, _ = make_loop(cmd)
    loop.handle_event(fg.operator_message(f"Flyg till {pin.lat}, {pin.lon}"))
    await loop.settle()
    assert [c.skill for c in fg.commands] == ["goto"]
    assert fg.commands[0].params.get("pin") != GOTO_COORDS


async def test_the_time_guard_counts_as_a_healthy_turn() -> None:
    clock = _Clock()
    read = _reply(_call("get_situation", {}))
    rt = _SlowRuntime([read, read, read, read], clock, per_call=6.0)
    cmd = LlmCommander(rt, _ctx(), REG, BehaviourConfig(), Sampling(), clock=clock)
    loop, fg, _ = make_loop(cmd)
    loop.handle_event(fg.operator_message("Hur ser det ut?"))
    await loop.settle()
    assert _turn_log(fg).error == "deadline"
    assert loop.health.turn_fails == 0


async def test_a_trickling_cloud_starts_the_cloud_pause() -> None:
    from sd_brain.runtime.base import ModelError
    from sd_brain.runtime.fallback import BestAvailableRuntime

    class _Cloud:
        configured = True
        model_id = "m"

        async def chat(self, *args: Any) -> Any:
            await asyncio.sleep(10)

    rt = BestAvailableRuntime(_Cloud())  # type: ignore[arg-type]
    with pytest.raises(ModelError):
        await rt.chat([], [], 0.1, Sampling())
    assert rt.engine() == "command" and rt.last_engine == "command"
