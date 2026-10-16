"""Blocks, limits, the start rule and the tool filter (05 §4.5, §8.2, §8.4; AC-B14)."""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from brain_helpers import REG, situation

from sd_brain.memory import PendingRequest, empty_memory, pending_request, to_update, with_changes
from sd_brain.state_view import current_limits, fresh_request, motion_block, takeoff_ok
from sd_brain.tools import AdaptError, adapt, exposed_tools, read_tools, registry_tool_schemas
from sd_schemas.base import new_id
from sd_schemas.skills import SKILL_REGISTRY, SkillRegistry

NOW = dt.datetime(2026, 1, 1, 0, 1, tzinfo=dt.UTC)
READ = {t.name for t in read_tools()}
LATCH = READ | {"report", "mark_location"}
STOP = {
    "vehicle_id": "sim-01",
    "action": "stop",
    "underlying": None,
    "action_seq": 1,
    "set_by": "operator:demo",
    "set_at": "2026-01-01T00:00:00Z",
    "set_mono_ms": 1,
}


def _mem(request: bool, at: dt.datetime = NOW) -> Any:
    mem = empty_memory(new_id())
    if request:
        mem = with_changes(
            mem, pending=PendingRequest(kind="takeoff", message_id=new_id(), requested_at=at)
        )
    return mem


def _cleared() -> dict[str, Any]:
    return {
        "takeoff_clearance_valid_until": "2026-01-01T00:02:00Z",
        "brain_may_takeoff": True,
    }


def _names(
    s: Any,
    *,
    request: bool = False,
    policy: str = "request_then_clearance",
    reg: SkillRegistry = REG,
    **kw: Any,
) -> set[str]:
    return {t.name for t in exposed_tools(reg, s, _mem(request), NOW, policy=policy, **kw)}  # type: ignore[arg-type]


AIR = {"takeoff_clearance_valid_until": None}
CASES: list[tuple[str, str, dict[str, Any], dict[str, Any], set[str]]] = [
    (
        "no charter (DRAFT)",
        "draft_p1.json",
        {},
        {},
        READ | {"report", "mark_location", "propose_charter_change"},
    ),
    (
        "CHARTER_PROPOSED",
        "draft_p1.json",
        {"mission_status": "CHARTER_PROPOSED"},
        {},
        READ | {"report", "mark_location", "propose_charter_change"},
    ),
    (
        "READY, no clearance",
        "ready_charter_v1_modeB.json",
        {},
        {"request": True},
        READ | {"report", "mark_location", "propose_charter_change"},
    ),
    (
        "READY, clearance and request",
        "ready_charter_v1_modeB.json",
        _cleared(),
        {"request": True},
        READ | {"report", "mark_location", "propose_charter_change", "takeoff"},
    ),
    (
        "READY, clearance, no request",
        "ready_charter_v1_modeB.json",
        _cleared(),
        {},
        READ | {"report", "mark_location", "propose_charter_change"},
    ),
    (
        "READY, clearance, no request, clearance_only",
        "ready_charter_v1_modeB.json",
        _cleared(),
        {"policy": "clearance_only"},
        READ | {"report", "mark_location", "propose_charter_change", "takeoff"},
    ),
    (
        "ACTIVE airborne",
        "airborne_modeB.json",
        {},
        {},
        READ
        | {
            "report",
            "mark_location",
            "propose_charter_change",
            "goto",
            "survey_sector",
            "hold",
            "return_home",
            "resume",
        },
    ),
    ("STOP (PAUSED operator)", "stopped_modeB.json", {}, {}, LATCH),
    (
        "FLY HOME (RETURNING operator)",
        "airborne_modeB.json",
        {"mission_status": "RETURNING", "returning_cause": "operator"},
        {},
        LATCH,
    ),
    (
        "battery return",
        "airborne_modeB.json",
        {"mission_status": "RETURNING", "returning_cause": "battery"},
        {},
        LATCH,
    ),
    ("FAILSAFE", "airborne_modeB.json", {"mission_status": "FAILSAFE"}, {}, LATCH),
    ("HELD", "airborne_modeB.json", {"mission_status": "HELD"}, {}, LATCH),
    ("manual takeover", "airborne_modeB.json", {"vehicle": {"manual_control": True}}, {}, LATCH),
    (
        "PAUSED(brain): resume offered",
        "airborne_modeB.json",
        {"mission_status": "PAUSED", "paused_by": "brain"},
        {},
        READ
        | {
            "report",
            "mark_location",
            "propose_charter_change",
            "goto",
            "survey_sector",
            "hold",
            "return_home",
            "resume",
        },
    ),
    (
        "not in HOLD: no resume",
        "airborne_survey.json",
        {},
        {},
        READ
        | {
            "report",
            "mark_location",
            "propose_charter_change",
            "goto",
            "survey_sector",
            "hold",
            "return_home",
        },
    ),
    ("AI switch off", "airborne_modeB.json", {"ai_enabled": False}, {}, READ | {"report"}),
    (
        "mode A (gate asks for clicks)",
        "airborne_modeA.json",
        {},
        {},
        READ
        | {
            "report",
            "mark_location",
            "propose_charter_change",
            "goto",
            "survey_sector",
            "hold",
            "return_home",
            "resume",
        },
    ),
    (
        "LANDED",
        "airborne_modeB.json",
        {"mission_status": "LANDED", "vehicle": {"airborne": False, "landed": True}},
        {},
        READ | {"report", "mark_location", "propose_charter_change"},
    ),
    ("CLOSED", "airborne_modeB.json", {"mission_status": "CLOSED"}, {}, READ),
    ("recovered turn: read only", "airborne_modeB.json", {}, {"actions": False}, READ),
    ("not the pilot: report only", "airborne_modeB.json", {}, {"pilot": False}, READ | {"report"}),
]


@pytest.mark.parametrize(("name", "fx", "patch", "kw", "want"), CASES, ids=[c[0] for c in CASES])
def test_exposed_tools_table(
    name: str, fx: str, patch: dict[str, Any], kw: dict[str, Any], want: set[str]
) -> None:
    assert _names(situation(fx, **patch), **kw) == want


def test_stubs_and_stage2_never_exposed() -> None:
    s = situation()
    names = _names(s, reg=REG)
    assert not names & {"orbit", "land_at", "follow_me", "find_me", "inspect_detection"}
    # The real registry still has every skill as a stub: only read tools are offered.
    assert _names(s, reg=SKILL_REGISTRY) == READ


def test_stub_in_allowed_skills_is_not_exposed() -> None:
    s = situation()
    data = s.model_dump(mode="json")
    data["charter"]["allowed_skills"].append("orbit")
    assert "orbit" not in _names(type(s).model_validate(data))


def test_tool_bounds_follow_the_charter() -> None:
    tools = {
        t.name: t
        for t in exposed_tools(REG, situation("airborne_charter60.json"), _mem(False), NOW)
    }
    props = tools["goto"].openai["function"]["parameters"]["properties"]
    assert props["alt_m"]["maximum"] == 60
    assert props["alt_m"]["minimum"] == 15
    assert props["speed_ms"]["maximum"] == 15
    assert "rationale_sv" in tools["goto"].openai["function"]["parameters"]["required"]
    proposal = tools["propose_charter_change"].openai["function"]["parameters"]
    assert "autonomy_mode" not in proposal["properties"]
    assert proposal["properties"]["alt_max_agl_m"].get("maximum") == 1000


def test_registry_schemas_have_fixed_bounds() -> None:
    schemas = registry_tool_schemas(REG)
    goto = next(t for t in schemas if t["function"]["name"] == "goto")
    assert "maximum" not in goto["function"]["parameters"]["properties"]["alt_m"]
    assert len(schemas) <= 20


def test_motion_block_and_limits() -> None:
    assert motion_block(situation()) is None
    assert motion_block(situation(override=STOP)) == "stop"
    assert motion_block(situation(override={**STOP, "action": "battery_rtl"})) == "battery"
    assert motion_block(situation(override={**STOP, "action": "watchdog_rtl"})) == "return"
    assert (
        motion_block(situation(mission_status="RETURNING", returning_cause="failsafe"))
        == "failsafe"
    )
    assert motion_block(situation(mission_status="RETURNING", returning_cause="brain")) is None
    lim = current_limits(situation("airborne_charter60.json"))
    assert (lim.alt_max_agl_m, lim.mission_alt_max_agl_m) == (60, 120)
    lim = current_limits(situation("draft_p1.json"))
    assert (lim.alt_min_agl_m, lim.default_alt_agl_m) == (15, 40)


def test_takeoff_rule() -> None:
    s = situation("ready_charter_v1_modeB.json", **_cleared())
    ok = {"policy": "request_then_clearance", "ttl_s": 120}
    assert takeoff_ok(s, _mem(True), NOW, **ok)  # type: ignore[arg-type]
    assert not takeoff_ok(s, _mem(False), NOW, **ok)  # type: ignore[arg-type]
    old = _mem(True, NOW - dt.timedelta(seconds=300))
    assert fresh_request(s, old, NOW + dt.timedelta(seconds=200), 120) is None
    expired = situation(
        "ready_charter_v1_modeB.json",
        takeoff_clearance_valid_until="2026-01-01T00:00:30Z",
        brain_may_takeoff=True,
    )
    assert not takeoff_ok(expired, _mem(True), NOW, **ok)  # type: ignore[arg-type]
    armed = situation("ready_charter_v1_modeB.json", **_cleared(), vehicle={"armed": True})
    assert not takeoff_ok(armed, _mem(True), NOW, **ok)  # type: ignore[arg-type]
    # The request does not expire while the operator reads the charter card.
    assert s.charter is not None
    late_charter = s.model_copy(
        update={"charter": s.charter.model_copy(update={"approved_at": NOW})}
    )
    assert fresh_request(late_charter, _mem(True, NOW - dt.timedelta(seconds=200)), NOW, 120)


def test_memory_roundtrip() -> None:
    mem = _mem(True)
    assert pending_request(mem) is not None
    assert pending_request(with_changes(mem, pending=None)) is None
    assert to_update(with_changes(mem, plan_note="x" * 500)).plan_note == "x" * 400
    broken = mem.model_copy(update={"pending_operator_request": {"kind": "other"}})
    assert pending_request(broken) is None


def test_adapt() -> None:
    s = situation()
    goto = adapt("goto", {"pin": "P1"}, s)
    assert goto["alt_m"] == 40 and "speed_ms" not in goto
    assert adapt("goto", {"lat": -35.36, "lon": 149.16, "speed_ms": 5}, s)["speed_ms"] == 5
    survey = adapt("survey_sector", {"sector": "B", "spacing_m": 20}, s)
    assert survey == {"sector_id": "B", "pattern": "lawnmower", "alt_m": 40, "spacing_m": 20}
    mark = adapt("mark_location", {"label": "spår"}, s)
    assert mark["label"] == "spår" and mark["lat"] == s.vehicle.lat
    assert adapt("hold", {"reason": "x", "rationale_sv": "y"}, s) == {"reason": "x"}
    with pytest.raises(AdaptError, match="unknown sector C"):
        adapt("survey_sector", {"sector": "C"}, s)
    with pytest.raises(AdaptError, match="unknown pin P9"):
        adapt("goto", {"pin": "P9"}, s)
    nowhere = situation(vehicle={"lat": None, "lon": None})
    with pytest.raises(AdaptError):
        adapt("mark_location", {"label": "x"}, nowhere)
