"""Grammar, scripted and LLM backends (05 §6, §10.3; 07 script format)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest
from brain_helpers import REG, situation

from sd_brain.backends import GrammarBackend, LlmCommander, ScriptedBackend
from sd_brain.backends.base import BackendCrashError
from sd_brain.backends.scripted import ScriptError, load_script
from sd_brain.config import BehaviourConfig
from sd_brain.context import ContextBuilder
from sd_brain.memory import empty_memory
from sd_brain.runtime.base import ModelReply, ModelTimeoutError, RawToolCall, Sampling
from sd_brain.runtime.replay import ReplayRuntime
from sd_brain.tools import exposed_tools
from sd_brain.turn import Trigger, TurnInput
from sd_schemas.api import ChatMessage
from sd_schemas.base import new_id

NOW = dt.datetime(2026, 1, 1, 0, 1, tzinfo=dt.UTC)


def _msg(text: str) -> ChatMessage:
    return ChatMessage.model_validate(
        {
            "id": str(new_id()),
            "seq": 1,
            "mission_id": "0192f49e-0b3a-7c55-8e2f-1a2b3c4d5e6f",
            "role": "operator",
            "author_ref": "operator:demo",
            "kind": "text",
            "text": text,
            "in_reply_to": None,
            "refs": {},
            "author_role": "pilot",
            "parsed_intent": None,
            "handled_by": None,
            "delivery": "delivered",
            "created_at": "2026-01-01T00:00:00Z",
        }
    )


def _turn(
    text: str | None = None,
    fx: str = "airborne_modeB.json",
    trigger: Trigger = "operator_message",
    events: list[dict[str, Any]] | None = None,
) -> TurnInput:
    s = situation(fx)
    mem = empty_memory(s.mission_id)
    return TurnInput(
        turn_id=new_id(),
        mission_id=s.mission_id,
        trigger=trigger,
        events=events or [],  # type: ignore[arg-type]
        situation=s,
        chat_tail=[_msg(text)] if text else [],
        memory=mem,
        tools=exposed_tools(REG, s, mem, NOW),
        now=NOW,
    )


class _NoTools:
    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"situation": "LÄGE ..."}


async def test_scripted_when_matches_any_event_of_the_turn(tmp_path: Path) -> None:
    """A skill event and the alert it raised come in one turn, the alert last."""
    b = ScriptedBackend.from_file(_script(tmp_path))
    ev = [
        {"type": "event.skill", "body": {"skill": "goto", "status": "done"}},
        {"type": "event.alert", "body": {"code": "GATE_MISMATCH", "level": "warning"}},
    ]
    out = await b.decide(_turn(None, trigger="skill_event", events=ev), _NoTools())
    assert out.action is not None and out.action.skill == "survey_sector"


@pytest.mark.parametrize(
    ("text", "skill", "params"),
    [
        ("flyg till P1 på 50 m", "goto", {"pin": "P1", "alt_m": 50.0}),
        ("sök sektor b", "survey_sector", {"sector": "B"}),
        ("håll position", "hold", {}),
        ("fortsatt", "resume", {}),
    ],
)
async def test_grammar_orders(text: str, skill: str, params: dict[str, Any]) -> None:
    out = await GrammarBackend().decide(_turn(text), _NoTools())
    assert out.action is not None and out.action.skill == skill and out.action.params == params


async def test_grammar_texts() -> None:
    g = GrammarBackend()
    assert (await g.decide(_turn("hjälp"), _NoTools())).messages[0].text.startswith("Kommandon:")
    assert (await g.decide(_turn("status"), _NoTools())).messages[0].kind == "status"
    assert "Förstod inte" in (await g.decide(_turn("håll"), _NoTools())).messages[0].text
    assert not (await g.decide(_turn("stopp"), _NoTools())).messages  # the ground handled it
    stopped = await g.decide(_turn("fortsätt", "stopped_modeB.json"), _NoTools())
    assert stopped.action is None and stopped.messages[0].refs == {"proposed_action": "resume"}
    ground = await g.decide(_turn("starta", "ready_charter_v1_modeB.json"), _NoTools())
    assert ground.action is None and "inte just nu" in ground.messages[0].text
    assert not (await g.decide(_turn(None, trigger="idle_tick"), _NoTools())).messages
    assert not (await g.decide(_turn(None), _NoTools())).messages


SCRIPT = """
scenario: SC99
fixture: fixture.yaml
turns:
  - on: mission_created
    say: ["Hej."]
  - on: operator_message
    match: '^Använd ramen (?P<name>charter_\\w+)\\.$'
    say: ["Jag föreslår ramen {name}."]
    charter_proposal: {from_fixture: "{name}"}
  - on: operator_message
    match: '^Starta\\.$'
    action: {skill: takeoff, params: {alt_m: 40}}
  - on: skill_event
    when: {skill: takeoff, status: done}
    action: {skill: goto, params: {at: "fixture:P1", alt_m: 40}}
  - on: skill_event
    when: {skill: goto, status: [done]}
    action: {skill: survey_sector, params: {polygon: "fixture:survey_box", alt_m: 40}}
  - on: operator_message
    match: '^Elak\\.$'
    raw: {skill: set_param, params: {name: FENCE_ENABLE}}
  - on: operator_message
    match: '^Krascha\\.$'
    fail: crash
  - on: operator_message
    match: '.*'
    once: false
    say: ["Uppfattat."]
"""
FIXTURE = """
home: { lat: -35.363261, lon: 149.165230, alt_amsl_m: 584.0, heading_deg: 353 }
pins:
  - { id: "P1", east_m: 0.0, north_m: 200.0 }
survey_box: { center_pin: "P1", size_m: [100.0, 100.0], alt_m: 40.0 }
charters:
  charter_test:
    area_enu: [[-150,-100],[150,-100],[150,300],[-150,300]]
    geofence_buffer_m: 20
    sectors: [{ sector_id: "A", name: "survey_box", from: survey_box }]
"""


def _script(tmp_path: Path, text: str = SCRIPT) -> Path:
    (tmp_path / "fixture.yaml").write_text(FIXTURE)
    path = tmp_path / "SC99.yaml"
    path.write_text(text)
    return path


async def test_scripted_backend(tmp_path: Path) -> None:
    b = ScriptedBackend.from_file(_script(tmp_path))
    hello = await b.decide(_turn(None, trigger="mission_created"), _NoTools())
    assert hello.messages[0].text == "Hej."
    assert not (await b.decide(_turn(None, trigger="mission_created"), _NoTools())).messages  # once
    prop = await b.decide(_turn("Använd ramen charter_test."), _NoTools())
    assert (
        prop.ready_proposal is not None
        and prop.messages[0].text == "Jag föreslår ramen charter_test."
    )
    sectors = prop.ready_proposal.sectors  # the fixture's search box, not the whole area
    assert sectors is not None and [(s.sector_id, s.name) for s in sectors] == [("A", "survey_box")]
    lats = [p[1] for p in sectors[0].area["coordinates"][0]]
    assert max(lats) - min(lats) == pytest.approx(100 / 111_320, rel=0.02)
    take = await b.decide(_turn("Starta."), _NoTools())
    assert take.action is not None and take.action.adapted and take.action.params == {"alt_m": 40}
    ev = [{"type": "event.skill", "body": {"skill": "takeoff", "status": "done"}}]
    goto = await b.decide(_turn(None, trigger="skill_event", events=ev), _NoTools())
    assert goto.action is not None and goto.action.params["lat"] == pytest.approx(
        -35.3614624, abs=1e-5
    )
    ev = [{"type": "event.skill", "body": {"skill": "goto", "status": "done"}}]
    survey = await b.decide(_turn(None, trigger="skill_event", events=ev), _NoTools())
    assert survey.action is not None and survey.action.params["polygon"]["type"] == "Polygon"
    raw = await b.decide(_turn("Elak."), _NoTools())
    assert raw.action is not None and raw.action.raw
    with pytest.raises(BackendCrashError):
        await b.decide(_turn("Krascha."), _NoTools())
    assert (await b.decide(_turn("vad som helst"), _NoTools())).messages[0].text == "Uppfattat."
    assert (await b.decide(_turn("igen"), _NoTools())).messages[0].text == "Uppfattat."


async def test_scripted_ai_form_goes_through_the_adapter(tmp_path: Path) -> None:
    """``ai: true``: the script names the drawn path like a model would; the loop's
    adapter resolves it against the situation, never the script."""
    text = (
        "scenario: SC99\nturns:\n  - on: operator_message\n    match: '^Flyg vägen\\.$'\n"
        "    action: {skill: follow_path, ai: true, params: {path: pa-1, alt_m: 30}}\n"
    )
    b = ScriptedBackend.from_file(_script(tmp_path, text))
    out = await b.decide(_turn("Flyg vägen."), _NoTools())
    assert out.action is not None and not out.action.adapted
    assert out.action.params == {"path": "pa-1", "alt_m": 30}


@pytest.mark.parametrize(
    "bad",
    [
        "scenario: X\nturns:\n  - on: operator_message\n    unknown: 1\n",
        "scenario: X\nturns:\n  - on: skill_event\n    when: {colour: red}\n",
        "scenario: X\nturns:\n  - on: skill_event\n    match: 'x'\n",
        "scenario: X\nfixture: missing.yaml\nturns: []\n",
        "scenario: [\n",
    ],
)
def test_script_refuses(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(bad)
    with pytest.raises(ScriptError):
        load_script(path)


def _llm(
    *replies: ModelReply, behaviour: BehaviourConfig | None = None, context_tokens: int = 8192
) -> tuple[LlmCommander, ReplayRuntime]:
    rt = ReplayRuntime(list(replies))
    ctx = ContextBuilder(REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000)
    return LlmCommander(
        rt, ctx, REG, behaviour or BehaviourConfig(), Sampling(), context_tokens=context_tokens
    ), rt


def _r(content: str | None = None, *calls: tuple[str, str]) -> ModelReply:
    return ModelReply(
        content=content,
        tool_calls=[
            RawToolCall(id=f"c{i}", name=n, arguments_json=a) for i, (n, a) in enumerate(calls)
        ],
        finish_reason="stop",
        engine="cloud",
    )


async def test_llm_reads_then_acts() -> None:
    cmd, rt = _llm(
        _r(None, ("get_situation", "{}"), ("brain_note", '{"text": "Plan: P1"}')),
        _r(
            "Flyger.",
            ("goto", '{"pin": "P1", "rationale_sv": "x"}'),
            ("hold", '{"rationale_sv": "y"}'),
        ),
    )
    out = await cmd.decide(_turn("flyg till P1"), _NoTools())
    assert out.action is not None and out.action.skill == "goto"
    assert out.note_update == "Plan: P1" and out.diagnostics.rounds == 2
    assert out.messages[0].text == "Flyger."
    assert rt.calls[1]["messages"][-1]["role"] == "tool"


async def test_llm_repair_then_parse_error() -> None:
    cmd, _ = _llm(_r(None, ("goto", '{"pin": "P9"}')), _r(None, ("goto", '{"pin": "P8"}')))
    out = await cmd.decide(_turn("x"), _NoTools())
    assert (
        out.diagnostics.error == "parse_error"
        and out.diagnostics.repairs == 1
        and out.action is None
    )
    cmd, rt = _llm(_r('text but bad <tool_call>{"name": "nope"}</tool_call>'), _r("Okej."))
    out = await cmd.decide(_turn("x"), _NoTools())
    assert out.messages[0].text == "Okej." and rt.calls[1]["messages"][-1]["role"] == "user"


async def test_llm_proposal_and_limits() -> None:
    draft = (
        '{"search_areas": [{"kind": "rect", "center_pin_id": "P1", '
        '"width_m": 100, "height_m": 100}], "reason_sv": "x"}'
    )
    cmd, _ = _llm(_r(None, ("propose_charter_change", draft)))
    out = await cmd.decide(_turn("ruta"), _NoTools())
    assert out.charter_proposal is not None
    looping = [_r(None, ("get_log", '{"n": 1}')) for _ in range(5)]
    cmd, _ = _llm(*looping, behaviour=BehaviourConfig(max_read_tool_rounds=4))
    out = await cmd.decide(_turn("x"), _NoTools())
    assert out.diagnostics.error == "round_limit" and out.diagnostics.rounds == 5
    cmd, _ = _llm(context_tokens=1024)
    out = await cmd.decide(_turn("x"), _NoTools())
    assert out.diagnostics.error == "round_limit" and out.diagnostics.rounds == 0


class _Failing:
    name, model_id, supports_images = "x", "x", False

    async def chat(self, *a: Any, **k: Any) -> ModelReply:
        raise ModelTimeoutError("slow")


async def test_llm_runtime_error() -> None:
    ctx = ContextBuilder(REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000)
    cmd = LlmCommander(_Failing(), ctx, REG, BehaviourConfig(), Sampling())
    out = await cmd.decide(_turn("x"), _NoTools())
    assert out.diagnostics.error == "timeout"


async def test_scripted_flies_to_a_target_find_and_asks_the_pilot(tmp_path: Path) -> None:
    script = tmp_path / "find.yaml"
    script.write_text(
        "scenario: SK-RED\n"
        "turns:\n"
        "  - on: detection\n"
        "    when: {label: red_block}\n"
        "    once: false\n"
        '    say: ["{name_sv} hittat, bekräfta?"]\n'
        '    action: {skill: goto, params: {at: "event:find", alt_m: 25}}\n',
        encoding="utf-8",
    )
    b = ScriptedBackend.from_file(script)
    find = {
        "find_id": "f1",
        "label": "red_block",
        "name_sv": "Rött block",
        "kind": "target",
        "source": "color",
        "lat": -35.3615,
        "lon": 149.1652,
    }
    ev = [{"type": "event.detection", "body": find}]
    out = await b.decide(_turn(None, trigger="detection", events=ev), _NoTools())
    assert [m.text for m in out.messages] == ["Rött block hittat, bekräfta?"]
    assert out.action is not None and out.action.skill == "goto"
    assert out.action.params == {"lat": -35.3615, "lon": 149.1652, "alt_m": 25}
    other = [{"type": "event.detection", "body": {**find, "label": "person"}}]
    assert (
        await b.decide(_turn(None, trigger="detection", events=other), _NoTools())
    ).action is None


async def test_scripted_regex_groups_win_over_event_fields(tmp_path: Path) -> None:
    script = tmp_path / "clash.yaml"
    script.write_text(
        "scenario: SK-RED\n"
        "turns:\n"
        "  - on: operator_message\n"
        "    match: '^Vad är (?P<label>\\w+)\\?$'\n"
        "    once: false\n"
        '    say: ["{label}"]\n',
        encoding="utf-8",
    )
    b = ScriptedBackend.from_file(script)
    ev = [{"type": "event.detection", "body": {"label": "red_block", "kind": "target"}}]
    out = await b.decide(_turn("Vad är bil?", events=ev), _NoTools())
    assert [m.text for m in out.messages] == ["bil"]  # no TypeError over the shared key


def test_event_find_only_for_at(tmp_path: Path) -> None:
    from sd_brain.backends.scripted import _Fixture, _resolve  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(ScriptError, match="event:find"):
        _resolve({"polygon": "event:find"}, _Fixture(None), _turn(None), {"lat": 1, "lon": 2})
    with pytest.raises(ScriptError, match="event:find"):
        _resolve({"at": "event:find"}, _Fixture(None), _turn(None), {})


HERE_SCRIPT = SCRIPT.replace(
    "  - on: operator_message\n    match: '.*'",
    "  - on: operator_message\n    match: '^Upp\\.$'\n"
    "    action: {skill: goto, params: {at: here, alt_m: 200}}\n"
    "  - on: operator_message\n    match: '.*'",
)


async def test_scripted_at_here_is_the_drones_position(tmp_path: Path) -> None:
    b = ScriptedBackend.from_file(_script(tmp_path, HERE_SCRIPT))
    turn = _turn("Upp.")
    v = turn.situation.vehicle
    out = await b.decide(turn, _NoTools())
    assert out.action is not None
    assert out.action.params == {"lat": v.lat, "lon": v.lon, "alt_m": 200}


async def test_scripted_at_here_without_a_position_refuses(tmp_path: Path) -> None:
    b = ScriptedBackend.from_file(_script(tmp_path, HERE_SCRIPT))
    turn = _turn("Upp.")
    s = turn.situation
    vehicle = s.vehicle.model_copy(update={"lat": None})
    turn = turn.model_copy(update={"situation": s.model_copy(update={"vehicle": vehicle})})
    with pytest.raises(ScriptError):
        await b.decide(turn, _NoTools())
