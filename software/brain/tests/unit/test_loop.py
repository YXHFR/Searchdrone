"""The commander loop: retries, wake-ups, rules, deadline and the stop check (05 §5, §10)."""

from __future__ import annotations

import asyncio
import itertools
from typing import Any, Literal

import pytest
from brain_helpers import REG, make_loop

from sd_brain.backends.base import BackendCrashError
from sd_brain.config import BrainConfig
from sd_brain.constants import FRESH_SNAPSHOT_WAIT_S
from sd_brain.decide.engines import EngineAnswer
from sd_brain.decide.service import DecisionService
from sd_brain.loop import BACK_TEXT, DEADLINE_TEXT, RATE_TEXT, STOP_FALLBACK
from sd_brain.turn import AiAction, OutMessage, TurnDiagnostics, TurnInput, TurnOutput
from sd_schemas.base import new_id
from sd_schemas.decide import ChatIntentReport, ChoiceAnswer, DecideRequest, DecisionEngine

REJECT = {
    "decision": "REJECT",
    "reason_codes": ["OUTSIDE_CHARTER"],
    "message_sv": "Utanför ramen.",
    "status": "rejected_by_gate",
}


class Steps:
    """A backend that returns prepared outputs, one per turn, and records its inputs."""

    name = "cloud"

    def __init__(self, *outs: TurnOutput | Exception | float) -> None:
        self.outs: list[TurnOutput | Exception | float] = list(outs)
        self.turns: list[TurnInput] = []

    async def decide(self, turn: TurnInput, tools: Any) -> TurnOutput:
        self.turns.append(turn)
        out = (
            self.outs.pop(0)
            if self.outs
            else TurnOutput(diagnostics=TurnDiagnostics(backend="llm"))
        )
        if isinstance(out, Exception):
            raise out
        if isinstance(out, int | float):
            await asyncio.sleep(out)
            return TurnOutput(diagnostics=TurnDiagnostics(backend="llm"))
        return out


def act(skill: str, **params: Any) -> TurnOutput:
    return TurnOutput(
        action=AiAction(skill=skill, params=params, rationale_sv="test"),
        diagnostics=TurnDiagnostics(backend="llm"),
    )


def say(text: str) -> TurnOutput:
    return TurnOutput(messages=[OutMessage(text=text)], diagnostics=TurnDiagnostics(backend="llm"))


async def test_sync_reject_retries_once() -> None:
    backend = Steps(act("goto", pin="P1"), act("goto", pin="P1", alt_m=30))
    loop, fg, _ = make_loop(backend)
    fg.responses = [{"match": {"skill": "goto"}, "result": REJECT}]
    loop.handle_event(fg.operator_message("Flyg till P1"))
    await loop.settle()
    assert len(fg.commands) == 2
    assert [t.trigger for t in backend.turns] == ["operator_message", "gate_reject_retry"]
    assert backend.turns[1].reject is not None and backend.turns[1].reject.reason_codes == [
        "OUTSIDE_CHARTER"
    ]
    assert "Hur vill du gå vidare?" in fg.posted[-1].text


async def test_sync_reject_event_does_not_wake() -> None:
    backend = Steps(act("goto", pin="P1"), say("Ok."))
    loop, fg, _ = make_loop(backend)
    fg.responses = [{"match": {"skill": "goto"}, "result": REJECT, "times": 1}]
    loop.handle_event(fg.operator_message("Flyg till P1"))
    await loop.settle()
    turns = len(backend.turns)
    cid = str(fg.commands[0].command_id)
    loop.handle_event(
        {
            "type": "event.skill",
            "body": {"command_id": cid, "skill": "goto", "status": "rejected_by_gate"},
        }
    )
    await loop.settle()
    assert len(backend.turns) == turns


async def test_late_reject_wakes_without_retry() -> None:
    backend = Steps(act("goto", pin="P1"), say("Ordern avvisades. Vad vill du göra?"))
    loop, fg, _ = make_loop(backend)
    loop.handle_event(fg.operator_message("Flyg till P1"))
    await loop.settle()
    cid = str(fg.commands[0].command_id)
    loop.handle_event(
        {
            "type": "event.skill",
            "body": {"command_id": cid, "skill": "goto", "status": "rejected_onboard"},
        }
    )
    await loop.settle()
    assert [t.trigger for t in backend.turns] == ["operator_message", "skill_event"]
    assert len(fg.commands) == 1
    loop.handle_event(
        {"type": "event.skill", "body": {"command_id": cid, "skill": "goto", "status": "running"}}
    )
    await loop.settle()
    assert len(backend.turns) == 2  # running does not wake


async def test_charter_error_retries_up_to_three_attempts() -> None:
    draft = {
        "search_areas": [{"kind": "rect", "center_pin_id": "P1", "width_m": 100, "height_m": 100}],
        "alt_max_agl_m": 150,
        "reason_sv": "x",
    }
    from sd_brain.charter_draft import CharterDraft

    first = TurnOutput(
        charter_proposal=CharterDraft.model_validate(draft),
        diagnostics=TurnDiagnostics(backend="llm"),
    )
    second, third = (
        TurnOutput(
            charter_proposal=CharterDraft.model_validate(draft),
            diagnostics=TurnDiagnostics(backend="llm"),
        )
        for _ in range(2)
    )
    backend = Steps(first, second, third)
    loop, fg, _ = make_loop(backend)
    err = {
        "ok": False,
        "errors": [
            {
                "code": "CHARTER_EXCEEDS_LIMITS",
                "detail_sv": "Högsta höjd är 120 m.",
                "field": "alt_max_agl_m",
                "limit": 120,
            }
        ],
    }
    fg.responses = [{"match": {"skill": "propose_charter_change"}, "result": err}]
    loop.handle_event(fg.operator_message("Flyg på 150 m"))
    await loop.settle()
    assert len(fg.proposals) == 3 and len(backend.turns) == 3
    assert [t.trigger for t in backend.turns] == [
        "operator_message",
        "gate_reject_retry",
        "gate_reject_retry",
    ]
    assert [t.attempt for t in backend.turns] == [None, 2, 3]
    assert backend.turns[1].charter_errors[0].code == "CHARTER_EXCEEDS_LIMITS"
    asked = fg.chat_rows[0].id
    assert [t.in_reply_to for t in backend.turns] == [asked, asked, asked]
    assert len(fg.posted) == 1  # only the final line, nothing in between
    final = fg.posted[0]
    assert "Högsta höjd är 120 m." in final.text and "efter 3 försök" in final.text
    assert final.kind == "question" and final.in_reply_to == asked
    assert final.refs == {
        "charter_failed": True,
        "codes": ["CHARTER_EXCEEDS_LIMITS"],
        "attempts": 3,
    }
    logs = [d for d in fg.decisions if d.trigger != "stop_check"]
    assert [d.attempt for d in logs] == [1, 2, 3]
    assert logs[0].retry_of is None and str(logs[1].retry_of) == loop.records[0].turn_id
    assert logs[2].proposal_errors[0].field == "alt_max_agl_m"
    assert logs[0].proposal_draft is not None and logs[0].proposal_draft["alt_max_agl_m"] == 150


async def test_state_changed_retries_once() -> None:
    backend = Steps(act("goto", pin="P1"), act("goto", pin="P1"))
    loop, fg, _ = make_loop(backend)
    changed = {
        "decision": "REJECT",
        "reason_codes": ["STATE_CHANGED"],
        "message_sv": "Läget ändrades.",
        "status": "rejected_by_gate",
    }
    fg.responses = [{"match": {"skill": "goto"}, "result": changed, "times": 1}]
    loop.handle_event(fg.operator_message("Flyg till P1"))
    seq_before = fg.sit["snapshot_state_seq"]

    orig = fg.submit

    async def bump(mid: str, req: Any) -> Any:
        res = await orig(mid, req)
        fg.sit["snapshot_state_seq"] += 5
        return res

    fg.submit = bump  # type: ignore[method-assign]
    await loop.settle()
    assert len(fg.commands) == 2
    assert (
        fg.commands[1].based_on is not None and fg.commands[1].based_on.state_seq == seq_before + 5
    )


async def test_queued_messages_never_act() -> None:
    backend = Steps(act("goto", pin="P1"))
    loop, fg, _ = make_loop(backend)
    loop.handle_event(fg.operator_message("flyg till P1", delivery="queued"))
    await loop.settle()
    assert not fg.commands and not fg.proposals
    assert [t.trigger for t in backend.turns] == ["recovered"]
    assert {t.name for t in backend.turns[0].tools} == {
        "get_situation",
        "estimate",
        "get_log",
        "brain_note",
    }
    assert backend.turns[0].backlog[0].text == "flyg till P1"


async def test_reconnect_says_back_and_runs_recovered() -> None:
    backend = Steps(say("Medan jag var borta hände inget. Vad vill du göra?"))
    loop, fg, _ = make_loop(backend)
    await loop.connected(True)
    await loop.settle()
    assert fg.posted[0].text == BACK_TEXT and backend.turns[0].trigger == "recovered"


async def test_pending_approval_text() -> None:
    backend = Steps(
        TurnOutput(
            action=AiAction(skill="survey_sector", params={"sector": "B"}),
            messages=[OutMessage(text="Söker B.")],
            diagnostics=TurnDiagnostics(backend="llm"),
        )
    )
    loop, fg, _ = make_loop(backend, "airborne_modeA.json")
    fg.responses = [
        {"match": {"skill": "survey_sector"}, "result": {"decision": "PENDING_APPROVAL"}}
    ]
    loop.handle_event(fg.operator_message("Sök sektor B"))
    await loop.settle()
    assert [m.text for m in fg.posted] == ["Väntar på ditt klick: sök sektor B."]
    assert fg.commands[0].params["sector_id"] == "B" and fg.commands[0].backend == "llm"
    assert fg.decisions[-1].gate_decision == "PENDING_APPROVAL"


async def test_text_only_after_allow() -> None:
    out = TurnOutput(
        action=AiAction(skill="goto", params={"pin": "P1"}),
        messages=[OutMessage(text="Flyger till P1.")],
        diagnostics=TurnDiagnostics(backend="llm"),
    )
    loop, fg, _ = make_loop(Steps(out))
    loop.handle_event(fg.operator_message("P1"))
    await loop.settle()
    assert [m.text for m in fg.posted] == ["Flyger till P1."]
    assert fg.commands[0].ai_config_hash == "test-hash"


async def test_deadline_degrades_to_command_mode_after_three() -> None:
    backend = Steps(0.5, 0.5, 0.5)
    loop, fg, clock = make_loop(backend)
    loop.turn_deadline_s = 0.05
    for i in range(3):
        loop.handle_event(fg.operator_message(f"rad {i}"))
        await loop.settle()
    assert [m.text for m in fg.posted] == [DEADLINE_TEXT] * 3
    assert not fg.commands
    assert loop.health.degraded and loop.heartbeat()["status"] == "degraded"
    assert loop.heartbeat()["backend"] == "grammar"
    loop.handle_event(fg.operator_message("hjälp"))
    await loop.settle()
    assert fg.posted[-1].text.startswith("Kommandon:")
    clock.t += 61
    assert loop.health.may_recover()


async def test_backend_crash_counts_as_failure() -> None:
    loop, fg, _ = make_loop(Steps(BackendCrashError("x")))
    loop.handle_event(fg.operator_message("hej"))
    await loop.settle()
    assert fg.posted[0].text == DEADLINE_TEXT and loop.health.turn_fails == 1


async def test_rate_limit_three_times_degrades() -> None:
    backend = Steps(*[act("hold") for _ in range(6)])
    loop, fg, _ = make_loop(backend)
    fg.responses = [
        {
            "match": {"skill": "hold"},
            "result": {
                "decision": "REJECT",
                "reason_codes": ["RATE_LIMIT"],
                "message_sv": "För tätt.",
            },
        }
    ]
    for i in range(3):
        loop.handle_event(fg.operator_message(f"håll {i}"))
        await loop.settle()
    assert loop.health.degraded
    assert RATE_TEXT in [m.text for m in fg.posted]
    times = list(loop._action_times)
    assert all(b - a >= 3 for a, b in itertools.pairwise(times))


async def test_shutdown_mid_turn_stops_the_loop() -> None:
    """Cancelling the loop during a turn stops it (it is not an operator's interrupt)."""
    loop, fg, _ = make_loop(Steps(30.0))
    runner = asyncio.create_task(loop.run_forever())
    loop.handle_event(fg.operator_message("sök sektor A"))
    await asyncio.sleep(0.05)
    assert loop._turn_task is not None
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(runner, 2)


async def test_override_cancels_running_turn() -> None:
    backend = Steps(5.0, say("Stoppat. Drönaren står still."))
    loop, fg, _ = make_loop(backend)
    loop.handle_event(fg.operator_message("sök sektor A"))
    runner = asyncio.create_task(loop.settle())
    await asyncio.sleep(0.05)
    evt = {
        "type": "event.operator_override",
        "body": {"active": True, "action": "stop", "via": "button"},
    }
    fg.apply_event(evt)
    loop.handle_event(evt)
    await asyncio.wait_for(runner, 2)
    assert [t.trigger for t in backend.turns] == ["operator_message", "operator_override"]
    assert {t.name for t in backend.turns[1].tools} >= {"report", "mark_location"}
    assert "goto" not in {t.name for t in backend.turns[1].tools}


async def test_rule_resume_reply_under_stop() -> None:
    backend = Steps()
    loop, fg, _ = make_loop(backend, "stopped_modeB.json")
    loop.handle_event(fg.operator_message("fortsätt"))
    await loop.settle()
    assert not backend.turns and fg.posted[0].refs == {"proposed_action": "resume"}
    assert fg.sit["override"] is not None
    battery = {**fg.sit["override"], "action": "battery_rtl"}
    fg.sit["override"] = battery
    loop.handle_event(fg.operator_message("kör vidare"))
    await loop.settle()
    assert "batteriet" in fg.posted[-1].text


async def test_rule_takeoff_after_clearance() -> None:
    loop, fg, clock = make_loop(
        Steps(say("Bekräfta att det är fritt runt drönaren med knappen, så startar jag.")),
        "ready_charter_v1_modeB.json",
    )
    loop.handle_event(fg.operator_message("Lyft till 30 m"))
    await loop.settle()
    assert (
        fg.mem.pending_operator_request is not None
        and fg.mem.pending_operator_request["alt_m"] == 30
    )
    clock.t += 5
    evt = {
        "type": "event.takeoff_clearance",
        "body": {"status": "active", "valid_until": "2026-01-01T00:02:00Z"},
    }
    fg.apply_event(evt)
    loop.handle_event(evt)
    await loop.settle()
    assert fg.commands[-1].skill == "takeoff" and fg.commands[-1].params == {"alt_m": 30.0}
    assert fg.commands[-1].backend == "rule"
    assert fg.posted[-1].text == "Startar till 30 m."
    assert fg.mem.pending_operator_request is None


async def test_rule_takeoff_rejected_says_gate_text() -> None:
    loop, fg, _ = make_loop(Steps(say("Ok.")), "ready_charter_v1_modeB.json")
    fg.responses = [
        {
            "match": {"skill": "takeoff"},
            "result": {
                "decision": "REJECT",
                "reason_codes": ["CLEARANCE_REQUIRED"],
                "message_sv": "Bekräfta fritt runt.",
            },
        }
    ]
    loop.handle_event(fg.operator_message("Starta"))
    await loop.settle()
    evt = {
        "type": "event.takeoff_clearance",
        "body": {"status": "active", "valid_until": "2026-01-01T00:02:00Z"},
    }
    fg.apply_event(evt)
    loop.handle_event(evt)
    await loop.settle()
    assert fg.posted[-1].text == "Bekräfta fritt runt."
    assert len(fg.commands) == 1


async def test_idle_tick() -> None:
    backend = Steps(say("Ska jag flyga hem?"))
    loop, _, clock = make_loop(backend)
    assert not loop.idle_tick()  # too soon
    clock.t += 61
    assert loop.idle_tick()
    await loop.settle()
    assert backend.turns[0].trigger == "idle_tick"
    loop2, _, _ = make_loop(Steps(), "stopped_modeB.json")
    loop2.idle_tick(force=True)
    await loop2.settle()
    assert not loop2.records


@pytest.mark.parametrize(("kind", "waits"), [("charter", False), ("command", True)])
async def test_idle_tick_waits_only_for_its_own_action_card(kind: str, waits: bool) -> None:
    """After a failed turn in the air the idle tick tries again. The AI's own charter
    proposal waiting for a click does not stop it (the drone hovered 15 minutes); its own
    action card does, since that order needs the pilot's yes first."""
    from brain_helpers import situation

    from sd_schemas.charter import CharterProposal

    if kind == "charter":
        in_force = situation().charter
        assert in_force is not None
        draft = CharterProposal.model_validate({"area": in_force.area, "reason_sv": "Mindre."})
        first = TurnOutput(ready_proposal=draft, diagnostics=TurnDiagnostics(backend="llm"))
    else:
        first = act("goto", pin="P1")
    backend = Steps(first, say("Jag fortsätter mot P1."))
    loop, fg, _ = make_loop(backend)
    fg.responses = [{"match": {"skill": "goto"}, "result": {"decision": "PENDING_APPROVAL"}}]
    loop.idle_tick(force=True)  # the AI's own card: a charter proposal or an order card
    await loop.settle()
    rec = loop.records[-1]
    own = rec.proposal.approval_id if rec.proposal is not None else None
    if rec.result is not None:
        own = rec.result.approval_id
    assert own is not None
    card = {
        "id": str(own),
        "mission_id": fg.sit["mission_id"],
        "vehicle_id": fg.sit["vehicle_id"],
        "kind": kind,
        "subject_id": "01a150c5-0000-7000-8000-000000000002",
        "subject_version": 2 if kind == "charter" else 1,
        "status": "pending",
        "summary_sv": "Uppdragsram v2" if kind == "charter" else "Flyg till P1 på 40 m",
        "required_confirmations": [],
        "default_autonomy_mode": None,
        "gate": None,
        "requested_by": "brain",
        "requested_at": "2026-10-18T20:47:07Z",
        "expires_at": None,
        "decided_by": None,
        "decided_at": None,
        "reason": None,
    }
    fg.sit["pending_approvals"] = [card]
    loop.idle_tick(force=True)
    await loop.settle()
    assert len(backend.turns) == (1 if waits else 2)


class _StopEngine:
    name: DecisionEngine = "jev"
    location: Literal["cloud", "local"] = "cloud"
    configured: bool = True

    async def answer(self, req: DecideRequest, timeout_s: float) -> EngineAnswer:
        probs = {"none": 0.05, "stop": 0.9, "return_home": 0.05}
        return EngineAnswer(
            "jev", "cloud", {"intent": ChoiceAnswer(choice="stop", probabilities=probs)}, "jev-m", 3
        )


async def test_stop_check_reports_to_ground() -> None:
    backend = Steps(5.0)
    svc = DecisionService([_StopEngine()])
    loop, fg, _ = make_loop(backend, decide=svc)
    svc.on_answer = loop.log_decide
    loop.handle_event(fg.operator_message("vänta, stanna där!"))
    await asyncio.wait_for(loop.settle(), 3)
    assert fg.intents and fg.intents[0].intent == "stop" and fg.intents[0].engine == "jev"
    assert str(fg.intents[0].message_id) == str(fg.chat_rows[0].id)
    assert not fg.commands
    assert any(d.trigger == "stop_check" for d in fg.decisions)


async def test_stop_check_fallback_when_ground_unreachable() -> None:
    loop, fg, _ = make_loop(Steps(), decide=DecisionService([]), chat_intents="down")
    loop.handle_event(fg.operator_message("Stopp!"))
    await loop.settle()
    assert fg.posted[0].text == STOP_FALLBACK["stop"]
    assert [c.skill for c in fg.commands] == ["hold"] and fg.commands[0].backend == "rule"
    # Never a flight home from the AI's side: it stops and asks for the button.
    loop2, fg2, _ = make_loop(Steps(), decide=DecisionService([]), chat_intents="down")
    loop2.handle_event(fg2.operator_message("flyg hem"))
    await loop2.settle()
    assert fg2.posted[0].text == STOP_FALLBACK["return_home"]
    assert [c.skill for c in fg2.commands] == ["hold"]


async def test_stop_check_refused_by_ground_answers_the_line_as_usual() -> None:
    backend = Steps(say("Jag står kvar."))
    loop, fg, _ = make_loop(backend, decide=DecisionService([]), chat_intents="refused")
    loop.handle_event(fg.operator_message("Stopp!"))
    await loop.settle()
    assert fg.intents and not fg.commands  # no order of its own
    assert [t.trigger for t in backend.turns] == ["operator_message"]
    assert [m.text for m in fg.posted] == ["Jag står kvar."]


class _SlowStopEngine(_StopEngine):
    async def answer(self, req: DecideRequest, timeout_s: float) -> EngineAnswer:
        await asyncio.sleep(0.2)  # the turn answers the line first
        return await super().answer(req, timeout_s)


async def test_refused_line_already_answered_gets_no_second_answer() -> None:
    backend = Steps(say("Svar."))
    cfg = BrainConfig.model_validate({"decide": {"stop_check_timeout_s": 1.0}})
    svc = DecisionService([_SlowStopEngine()])
    loop, fg, _ = make_loop(backend, cfg=cfg, decide=svc, chat_intents="refused")
    loop.handle_event(fg.operator_message("vänta lite"))
    await loop.settle()
    await asyncio.sleep(0.4)
    await loop.settle()
    assert fg.intents  # the report went out and was refused
    assert [t.trigger for t in backend.turns] == ["operator_message"]


async def test_refused_line_is_not_answered_in_another_mission() -> None:
    backend = Steps(say("Svar."))
    loop, fg, _ = make_loop(backend, decide=DecisionService([]), chat_intents="refused")
    report = fg.report_chat_intent

    async def mission_changes(mission_id: str, r: ChatIntentReport) -> Any:
        loop.mission_id = str(new_id())  # a new mission while the report was in flight
        return await report(mission_id, r)

    fg.report_chat_intent = mission_changes  # type: ignore[method-assign]
    loop.handle_event(fg.operator_message("Stopp!"))
    await loop.settle()
    assert fg.intents and not backend.turns


async def test_stop_check_skips_lines_the_ground_handled() -> None:
    loop, fg, _ = make_loop(Steps(), decide=DecisionService([_StopEngine()]))
    evt = fg.operator_message("stopp")
    evt["body"]["handled_by"] = "ground"
    loop.handle_event(evt)
    await loop.settle()
    assert not fg.intents and not fg.commands


async def test_stop_check_disabled() -> None:
    cfg = BrainConfig.model_validate({"decide": {"stop_check": False}})
    loop, fg, _ = make_loop(Steps(), cfg=cfg, decide=DecisionService([_StopEngine()]))
    loop.handle_event(fg.operator_message("stopp"))
    await loop.settle()
    assert not fg.intents


async def test_not_pilot_cannot_order() -> None:
    backend = Steps(act("goto", pin="P1"))
    loop, fg, _ = make_loop(backend)
    evt = fg.operator_message("flyg till P1", author_role="system")
    loop.handle_event(evt)
    await loop.settle()
    assert not fg.commands
    assert {t.name for t in backend.turns[0].tools} == {
        "get_situation",
        "estimate",
        "get_log",
        "brain_note",
        "report",
    }


async def test_other_mission_events_are_ignored_and_note_saved() -> None:
    out = say("Plan klar.")
    out.note_update = "Plan: 1 A, 2 hem."
    loop, fg, _ = make_loop(Steps(out))
    loop.handle_event(
        {
            "type": "event.operator_message",
            "mission_id": "0192f49e-0b3a-7c55-8e2f-000000000000",
            "body": {},
        }
    )
    loop.handle_event(fg.operator_message("planera"))
    await loop.settle()
    assert fg.mem.plan_note == "Plan: 1 A, 2 hem."
    assert loop.records[0].trigger == "operator_message"


async def test_unknown_skill_from_backend_is_refused() -> None:
    loop, fg, _ = make_loop(Steps(act("set_param", name="x")))
    loop.handle_event(fg.operator_message("x"))
    await loop.settle()
    assert not fg.commands and "Jag fick inte ihop ordern" in fg.posted[0].text


async def test_registry_is_the_foundation_one() -> None:
    assert "goto" in REG and REG.get("goto").status == "implemented"


async def test_skill_end_waits_for_the_snapshot_after_it() -> None:
    """The end of the search arrives before the snapshot that shows the vehicle holding:
    the next order is based on the newer snapshot, not on the one the onboard computer
    would refuse as STATE_CHANGED."""
    backend = Steps(act("return_home"))
    loop, fg, _ = make_loop(backend, "airborne_survey.json")
    cid = str(fg.sit["active_command"]["id"])
    calls = 0
    situation = fg.situation

    async def lagging(mission_id: str) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:  # the newer snapshot: holding, nothing active
            fg.sit["active_command"] = None
            fg.sit["snapshot_state_seq"] = 1005
        return await situation(mission_id)

    fg.situation = lagging  # type: ignore[method-assign]
    loop.handle_event(
        {
            "type": "event.skill",
            "body": {"command_id": cid, "skill": "survey_sector", "status": "done"},
        }
    )
    await loop.settle()
    assert calls >= 3
    assert len(fg.commands) == 1
    assert fg.commands[0].based_on is not None and fg.commands[0].based_on.state_seq == 1005


async def test_skill_end_waits_for_a_newer_snapshot_too() -> None:
    """The ground has already marked the order done, but its onboard snapshot is still the
    one from before the onboard computer changed state: the turn waits for the next one."""
    backend = Steps(act("return_home"))
    loop, fg, _ = make_loop(backend, "airborne_survey.json")
    cid = str(fg.sit["active_command"]["id"])
    fg.sit["active_command"] = None
    calls = 0
    situation = fg.situation

    async def lagging(mission_id: str) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:
            fg.sit["snapshot_state_seq"] = 2000
        return await situation(mission_id)

    fg.situation = lagging  # type: ignore[method-assign]
    loop.handle_event(
        {
            "type": "event.skill",
            "body": {"command_id": cid, "skill": "survey_sector", "status": "done"},
        }
    )
    await loop.settle()
    assert calls >= 3
    assert len(fg.commands) == 1
    assert fg.commands[0].based_on is not None and fg.commands[0].based_on.state_seq == 2000


async def test_skill_end_waits_only_so_long() -> None:
    backend = Steps(act("return_home"))
    loop, fg, clock = make_loop(backend, "airborne_survey.json")
    cid = str(fg.sit["active_command"]["id"])
    start = clock.monotonic()
    loop.handle_event(
        {
            "type": "event.skill",
            "body": {"command_id": cid, "skill": "survey_sector", "status": "done"},
        }
    )
    await loop.settle()
    waited = clock.monotonic() - start
    # The stale snapshot is used once FRESH_SNAPSHOT_WAIT_S has passed, not earlier.
    assert FRESH_SNAPSHOT_WAIT_S <= waited <= FRESH_SNAPSHOT_WAIT_S + 0.5, waited
    assert len(backend.turns) == 1


async def test_the_wait_reads_the_skill_event_among_others() -> None:
    """A skill's end queued together with another event: the wait follows the skill's."""
    backend = Steps(act("return_home"))
    loop, fg, clock = make_loop(backend, "airborne_survey.json")
    cid = str(fg.sit["active_command"]["id"])
    start = clock.monotonic()
    loop.handle_event(
        {
            "type": "event.skill",
            "body": {"command_id": cid, "skill": "survey_sector", "status": "done"},
        }
    )
    loop.handle_event({"type": "event.alert", "body": {"level": "warning", "code": "X"}})
    await loop.settle()
    assert clock.monotonic() - start >= FRESH_SNAPSHOT_WAIT_S
    assert backend.turns and backend.turns[0].trigger == "skill_event"


@pytest.mark.parametrize(
    ("status", "paused_by"),
    [("PAUSED", "operator"), ("HELD", None)],  # HELD: STOP's BRAKE right after takeoff
)
async def test_resume_waits_until_the_release_is_confirmed(
    status: str, paused_by: str | None
) -> None:
    """The pilot's Resume reaches the AI before the onboard computer has confirmed it (the
    mission is still PAUSED by the operator, or HELD by the STOP): the turn waits for the
    confirmed picture, or the AI sees a lock and only asks the pilot to resume again."""
    backend = Steps(act("resume"))
    loop, fg, _ = make_loop(backend)
    fg.sit["mission_status"] = status
    fg.sit["paused_by"] = paused_by
    calls = 0
    situation = fg.situation

    async def confirming(mission_id: str) -> Any:
        nonlocal calls
        calls += 1
        if calls == 4:  # the onboard snapshot without the latch arrived
            fg.sit["mission_status"] = "ACTIVE"
            fg.sit["paused_by"] = None
        return await situation(mission_id)

    fg.situation = confirming  # type: ignore[method-assign]
    loop.handle_event(
        {"type": "event.operator_override", "body": {"active": False, "action": "resume"}}
    )
    await loop.settle()
    assert calls >= 4
    assert backend.turns and backend.turns[0].trigger == "operator_override"


async def test_ready_proposal_builds_on_the_charter_in_force() -> None:
    """A ready-made proposal (the scripted commander's) has no base version; the loop
    sets the version of the charter in force, so a change in the air is not stale."""
    from brain_helpers import situation

    from sd_schemas.charter import CharterProposal

    in_force = situation().charter
    assert in_force is not None
    proposal = CharterProposal.model_validate({"area": in_force.area, "reason_sv": "Mindre."})
    assert proposal.base_version is None
    backend = Steps(TurnOutput(ready_proposal=proposal, diagnostics=TurnDiagnostics(backend="llm")))
    loop, fg, _ = make_loop(backend)
    loop.handle_event(fg.operator_message("Krymp ramen"))
    await loop.settle()
    assert fg.proposals and fg.proposals[-1].base_version == in_force.version
