"""B2-10: the AI sees the search candidates and is woken by them, but never decides one."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any

import pytest
from brain_helpers import REG, make_loop, situation

from sd_brain.backends import ScriptedBackend
from sd_brain.backends.scripted import (
    ScriptError,
    _Fixture,  # pyright: ignore[reportPrivateUsage]
    _resolve,  # pyright: ignore[reportPrivateUsage]
)
from sd_brain.context import ContextBuilder, situation_text
from sd_brain.memory import empty_memory
from sd_brain.tools import exposed_tools, profile_tool, read_tools, registry_tool_schemas
from sd_brain.turn import TRIGGER_PRIORITY, TurnDiagnostics, TurnInput, TurnOutput
from sd_schemas.base import new_id

NOW = dt.datetime(2026, 1, 1, 0, 1, tzinfo=dt.UTC)
CID = "0192f49e-2000-7000-8000-000000000003"
DET = "0192f49e-3000-7000-8000-000000000009"


def candidate(**over: Any) -> dict[str, Any]:
    return {
        "candidate_id": CID,
        "seq": 3,
        "priority_class": "high",
        "priority": 0.72,
        "state": "ready",
        "lat": -35.3615,
        "lon": 149.1652,
        "sigma95_m": 4.1,
        "n_seen": 1,
        "n_in_view": 2,
        "engine": "fast",
        "detection_id": DET,
        "multi_frame": False,
        **over,
    }


def decided(decision: str = "confirm", after: str = "confirmed") -> dict[str, Any]:
    return {
        "candidate_id": CID,
        "seq": 3,
        "decision": decision,
        "after": after,
        "lat": -35.3615,
        "lon": 149.1652,
        "sigma95_m": 4.1,
    }


class Record:
    name = "cloud"

    def __init__(self) -> None:
        self.turns: list[TurnInput] = []

    async def decide(self, turn: TurnInput, tools: Any) -> TurnOutput:
        self.turns.append(turn)
        return TurnOutput(diagnostics=TurnDiagnostics(backend="llm"))


def test_the_candidate_triggers_rank_after_the_person_and_the_detection() -> None:
    p = list(TRIGGER_PRIORITY)
    assert p.index("candidate_decided") == p.index("alert_critical") + 1
    assert p.index("alert_critical") == p.index("operator_message") + 1
    assert p.index("candidate") == p.index("detection") + 1


async def test_a_high_candidate_wakes_the_ai_once_per_candidate_and_mission() -> None:
    backend = Record()
    loop, _, clock = make_loop(backend)
    mid = loop.mission_id
    clock.t += 600
    loop.handle_event({"type": "event.candidate", "mission_id": mid, "body": candidate()})
    await loop.settle()
    assert loop._activity_at == clock.monotonic()  # pyright: ignore[reportPrivateUsage]
    more = candidate(state="assessing", n_seen=2, multi_frame=True)
    loop.handle_event({"type": "event.candidate", "mission_id": mid, "body": more})
    await loop.settle()
    assert [t.trigger for t in backend.turns] == ["candidate"]
    # A new mission starts over.
    other = str(new_id())
    loop.handle_event({"type": "event.mission", "mission_id": other, "body": {"created": True}})
    await loop.settle()
    loop.handle_event({"type": "event.candidate", "mission_id": other, "body": candidate()})
    await loop.settle()
    assert [t.trigger for t in backend.turns][-1] == "candidate"
    assert [t.trigger for t in backend.turns].count("candidate") == 2


async def test_every_decision_wakes_the_ai() -> None:
    backend = Record()
    loop, _, _ = make_loop(backend)
    mid = loop.mission_id
    for body in (decided("reject", "rejected"), decided("reopen", "ready"), decided()):
        loop.handle_event({"type": "event.candidate_decided", "mission_id": mid, "body": body})
        await loop.settle()
    assert [t.trigger for t in backend.turns] == ["candidate_decided"] * 3
    summary = ContextBuilder(
        REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000
    ).event_summary(backend.turns[-1])
    assert summary.startswith("candidate_decided: candidate_decided K3 (decision=confirm")
    assert "after=confirmed" in summary


def test_the_situation_shows_the_candidate_lines() -> None:
    lines = {
        "probability_line": (
            "Sektorer (POA kvar / täckt / medel-POD / PSR / min@stråk m): "
            "A 0,40/0%/0,00/0,016/6,3@42"
        ),
        "candidates_line": "Kandidater: K3 HÖG (2/3 bilder; kod) 120 m NO | att granska 1, låg 0",
        "finds_line": "Fynd: inga bekräftade",
    }
    text = situation_text(situation(**lines), NOW).splitlines()
    sectors = next(i for i, x in enumerate(text) if x.startswith("Sektorer: "))
    assert text[sectors + 1 : sectors + 4] == list(lines.values())
    assert not any(
        x.startswith(("Kandidater", "Fynd")) for x in situation_text(situation(), NOW).splitlines()
    )


def _turn(trigger: Any, body: dict[str, Any]) -> TurnInput:
    s = situation()
    mem = empty_memory(s.mission_id)
    kind = "event.candidate" if trigger == "candidate" else "event.candidate_decided"
    return TurnInput(
        turn_id=new_id(),
        mission_id=s.mission_id,
        trigger=trigger,
        events=[{"type": kind, "body": body}],
        situation=s,
        chat_tail=[],
        memory=mem,
        tools=exposed_tools(REG, s, mem, NOW),
        now=NOW,
    )


class _NoTools:
    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {}


async def test_scripted_candidate_turns(tmp_path: Path) -> None:
    script = tmp_path / "cand.yaml"
    script.write_text(
        "scenario: SK-CAND\n"
        "turns:\n"
        "  - on: candidate\n"
        "    when: {priority_class: high, state: ready, engine: fast, multi_frame: false}\n"
        '    say: ["K{seq} hittad, granska bilden."]\n'
        "    action: {skill: inspect_detection, params: "
        '{detection_id: "event:detection_id", standoff_m: 20}}\n'
        "  - on: candidate\n"
        "    when: {multi_frame: true}\n"
        '    action: {skill: goto, params: {at: "event:candidate", alt_m: 30}}\n'
        "  - on: candidate_decided\n"
        "    when: {decision: confirm, after: confirmed}\n"
        '    action: {skill: orbit, params: {at: "event:candidate", radius_m: 45}}\n',
        encoding="utf-8",
    )
    b = ScriptedBackend.from_file(script)
    out = await b.decide(_turn("candidate", candidate()), _NoTools())
    assert [m.text for m in out.messages] == ["K3 hittad, granska bilden."]
    assert out.action is not None and out.action.skill == "inspect_detection"
    assert out.action.params == {"detection_id": DET, "standoff_m": 20}
    out = await b.decide(_turn("candidate", candidate(multi_frame=True)), _NoTools())
    assert out.action is not None
    assert out.action.params == {"lat": -35.3615, "lon": 149.1652, "alt_m": 30}
    assert (
        await b.decide(_turn("candidate_decided", decided("reject", "rejected")), _NoTools())
    ).action is None
    out = await b.decide(_turn("candidate_decided", decided()), _NoTools())
    assert out.action is not None and out.action.skill == "orbit"
    assert out.action.params == {"lat": -35.3615, "lon": 149.1652, "radius_m": 45}


def test_scripted_candidate_when_fields_are_checked(tmp_path: Path) -> None:
    script = tmp_path / "bad.yaml"
    script.write_text(
        "scenario: SK-CAND\nturns:\n  - on: candidate_decided\n    when: {by: operator:x}\n",
        encoding="utf-8",
    )
    with pytest.raises(ScriptError, match="by"):
        ScriptedBackend.from_file(script)


def test_event_fields_must_be_in_the_event() -> None:
    turn = _turn("candidate", candidate())
    assert _resolve({"x": "event:seq"}, _Fixture(None), turn, candidate()) == {"x": 3}
    with pytest.raises(ScriptError, match="event:missing"):
        _resolve({"detection_id": "event:missing"}, _Fixture(None), turn, candidate())
    with pytest.raises(ScriptError, match="event:candidate"):
        _resolve({"at": "event:candidate"}, _Fixture(None), turn, {"seq": 3})


def test_no_tool_decides_a_candidate() -> None:
    """Only a person confirms a find: no tool the AI could ever get confirms, rejects or
    decides anything, nor changes a priority."""
    names = {t["function"]["name"] for t in registry_tool_schemas(REG)}
    names |= {t.name for t in read_tools()} | {profile_tool().name}
    names |= {k.name for k in REG}  # also the skills the AI does not get yet
    pattern = re.compile(r"confirm|reject|decide|decision|approv|priority|verify|bekr|avvis")
    assert names and not {n for n in names if pattern.search(n)}
