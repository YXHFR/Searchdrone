"""Blocks, limits, the start rule and the tool filter (05 §4.5, §8.2, §8.4; AC-B14)."""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

import pytest
from brain_helpers import REG, situation, with_stub

from sd_brain.memory import PendingRequest, empty_memory, pending_request, to_update, with_changes
from sd_brain.state_view import current_limits, fresh_request, motion_block, takeoff_ok
from sd_brain.tools import (
    AdaptError,
    adapt,
    charter_skills,
    exposed_tools,
    read_tools,
    registry_tool_schemas,
)
from sd_schemas.base import new_id
from sd_schemas.skills import SkillRegistry

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
    reg: SkillRegistry = REG,
    **kw: Any,
) -> set[str]:
    return {t.name for t in exposed_tools(reg, s, _mem(request), NOW, **kw)}  # type: ignore[arg-type]


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
    assert "goto" in names
    # A skill that is a stub (in another build) never becomes a tool.
    assert "goto" not in _names(s, reg=with_stub("goto"))


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
    # A charter proposal is bounded by the mission's limits, not by the charter in force.
    assert proposal["properties"]["alt_max_agl_m"].get("maximum") == 120
    assert proposal["properties"]["default_alt_agl_m"].get("minimum") == 5
    assert proposal["properties"]["max_speed_ms"].get("maximum") == 15


def test_proposal_schema_shapes_and_skill_names() -> None:
    """R2/D1/D2: kind selects the shape, the flight area is only a drawn one, the skill
    names come from the registry, and the schema has no oneOf/discriminator/const."""
    s = situation("draft_p1.json")
    tools = {t.name: t for t in exposed_tools(REG, s, _mem(False), NOW)}
    params = tools["propose_charter_change"].openai["function"]["parameters"]
    props = params["properties"]
    assert props["allowed_skills"]["items"]["enum"] == charter_skills(REG)
    assert {"goto", "survey_sector", "return_home"} <= set(charter_skills(REG))
    assert "search" not in charter_skills(REG) and "rtl" not in charter_skills(REG)
    assert props["operating_area"]["properties"]["kind"]["enum"] == ["drawn"]
    shapes = {
        a["properties"]["kind"]["enum"][0]: set(a["required"])
        for a in props["search_areas"]["items"]["anyOf"]
    }
    assert shapes == {
        "drawn": {"kind", "area_id"},
        "rect": {"kind", "width_m", "height_m"},
        "circle": {"kind", "radius_m"},
    }
    text = json.dumps(params)
    assert "oneOf" not in text and "discriminator" not in text and '"const"' not in text
    assert "$ref" not in text and "$defs" not in text
    fields = [*props.items(), *props["operating_area"]["properties"].items()]
    for shape in props["search_areas"]["items"]["anyOf"]:
        fields += shape["properties"].items()
    assert [k for k, v in fields if "description" not in v] == []


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
    ok = {"ttl_s": 120}
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


def test_tool_tables_match_the_gate() -> None:
    """The filter mirrors the gate: a change there must show up here."""
    from sd_brain.tools import ALWAYS_ALLOWED, STATE_ALLOWS
    from sd_safety import rules
    from sd_safety.policy import ALLOWED_OUTSIDE_CHARTER_SKILLS

    assert ALWAYS_ALLOWED == ALLOWED_OUTSIDE_CHARTER_SKILLS
    for state in rules._GROUND_ONLY_STATES:  # pyright: ignore[reportPrivateUsage]
        assert STATE_ALLOWS[state.value] == rules._NO_FLIGHT_SKILLS  # pyright: ignore[reportPrivateUsage]
    assert STATE_ALLOWS["READY"] == rules._NO_FLIGHT_SKILLS | {"takeoff"}  # pyright: ignore[reportPrivateUsage]


def test_no_setting_drops_the_start_request() -> None:
    from sd_brain.config import ConfigError, load_config

    with pytest.raises(ConfigError):
        load_config(None, {"SD_BRAIN__BEHAVIOUR__TAKEOFF_POLICY": "clearance_only"})
    s = situation("ready_charter_v1_modeB.json", **_cleared())
    assert "takeoff" not in _names(s)


@pytest.mark.parametrize(
    ("state", "want"),
    [
        ("HOLD", {"goto", "survey_sector", "follow_path", "hold", "resume", "return_home"}),
        ("TRANSIT", {"goto", "survey_sector", "follow_path", "hold", "return_home"}),
        ("SURVEY", {"goto", "survey_sector", "follow_path", "hold", "return_home"}),
        ("TAKEOFF", {"return_home"}),
        ("RTL", {"return_home"}),
        ("LAND", set()),
        ("PREFLIGHT", set()),  # takeoff only with the start rule (other tests)
        (None, set()),  # the onboard state is unknown: nothing that moves the vehicle
    ],
)
def test_onboard_skills_only_from_their_executor_states(state: str | None, want: set[str]) -> None:
    s = situation(vehicle={"executor_state": state}, charter=_allowing("follow_path"))
    moving = {k.name for k in REG if k.allowed_from}
    assert _names(s) & moving == want
    # Mirrors the registry: exactly the onboard skills the executor may start from here.
    assert want == {k.name for k in REG if state in k.allowed_from and k.name != "takeoff"}


def _allowing(*extra: str) -> dict[str, Any]:
    """The fixture's charter, also allowing ``extra`` (its list predates them)."""
    charter = situation().charter
    assert charter is not None
    return {"allowed_skills": [*charter.allowed_skills, *extra]}


def _drawn_path(label: str = "pa-1") -> dict[str, Any]:
    s = situation()
    return {
        "id": str(new_id()),
        "mission_id": str(s.mission_id),
        "kind": "path",
        "label": label,
        "geometry": {
            "type": "LineString",
            "coordinates": [[149.1652, -35.3630], [149.1660, -35.3620], [149.1670, -35.3625]],
        },
        "created_by": "operator:demo",
        "created_at": "2026-10-18T12:00:00Z",
        "area_m2": None,
        "length_m": 230.0,
    }


def test_adapt_follow_path_takes_the_points_from_the_drawn_path() -> None:
    s = situation(paths=[_drawn_path()])
    out = adapt("follow_path", {"path": "pa-1", "rationale_sv": "längs bäcken"}, s)
    assert out == {
        "points": [
            {"lat": -35.3630, "lon": 149.1652},
            {"lat": -35.3620, "lon": 149.1660},
            {"lat": -35.3625, "lon": 149.1670},
        ],
        "alt_m": 40,
        "label": "pa-1",
    }
    assert adapt("follow_path", {"path": "pa-1", "alt_m": 25, "speed_ms": 4}, s)["speed_ms"] == 4
    with pytest.raises(AdaptError, match="unknown path pa-2; known: pa-1"):
        adapt("follow_path", {"path": "pa-2"}, s)


def test_follow_path_tool_names_a_path_and_never_takes_coordinates() -> None:
    s = situation(paths=[_drawn_path()], charter=_allowing("follow_path"))
    tool = next(
        t
        for t in exposed_tools(REG, s, empty_memory(s.mission_id), dt.datetime.now(dt.UTC))
        if t.name == "follow_path"
    )
    props = tool.openai["function"]["parameters"]["properties"]
    assert set(props) == {"path", "alt_m", "speed_ms", "rationale_sv"}
    assert tool.openai["function"]["parameters"]["additionalProperties"] is False


def test_the_situation_text_lists_drawn_paths() -> None:
    from sd_brain.context import situation_text

    text = situation_text(situation(paths=[_drawn_path()]), dt.datetime.now(dt.UTC))
    line = next(x for x in text.splitlines() if x.startswith("Ritade vägar"))
    assert '"pa-1" 3 punkter 0,2 km' in line and "follow_path" in line
