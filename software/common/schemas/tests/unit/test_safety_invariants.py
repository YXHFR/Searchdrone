"""Safety invariants the schemas enforce on their own, before any gate runs."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

import sd_schemas as s
from sd_schemas.base import PARAMS_MAX_BYTES, new_id
from sd_schemas.commander import PROMPT_JSON_MAX_BYTES
from sd_schemas.executor import GotoResumePoint, SurveyResumePoint
from sd_schemas.reason_codes import (
    APPROVAL_OUTCOME_CODES,
    CHARTER_VALIDATION_CODES,
    GATE_REASON_CODES,
    LIMITS_CODES,
    ONBOARD_REASON_CODES,
)
from sd_schemas.skills import SKILL_REGISTRY, UNRESOLVED, TargetContext
from sd_schemas_examples import AREA, OTHER, _charter, _decision_log, _lla

HERE = _lla()
CHARTER = _charter()
CTX = TargetContext(charter=CHARTER, position=HERE, home=HERE)
NO_POSITION = TargetContext(charter=CHARTER, position=None, home=HERE)
NO_CHARTER = TargetContext(charter=None, position=HERE, home=HERE)


# --- ids --------------------------------------------------------------------------------


def test_ids_must_be_uuid7() -> None:
    with pytest.raises(ValidationError, match="UUIDv7"):
        s.CommandCancel(command_id=uuid.uuid4())
    s.CommandCancel(command_id=new_id())


# --- limits -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"battery_rtl_pct": 0},
        {"battery_rtl_pct": 14.9},
        {"battery_rtl_pct": -5},
        {"max_speed_ms": float("inf")},
        {"max_alt_agl_m": float("nan")},
    ],
)
def test_safety_limits_floors_and_finite(bad: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        s.SafetyLimits(**bad)


@pytest.mark.parametrize(
    "bad",
    [{"max_alt_agl_m": float("inf")}, {"max_alt_agl_m": 100}, {"min_battery_rtl_pct": 10}],
)
def test_limit_caps_are_finite_and_keep_floors(bad: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        s.LimitCaps(**bad)


def test_owner_sets_the_cap_freely() -> None:
    caps = s.LimitCaps(max_alt_agl_m=500, max_speed_ms=30, max_range_from_pilot_m=5000)
    raised = s.SafetyLimits(max_alt_agl_m=450, max_speed_ms=28, max_range_from_pilot_m=4000)
    assert caps.allows(raised)
    assert not s.LimitCaps().allows(raised)


def test_caps_allow() -> None:
    caps = s.LimitCaps(max_alt_agl_m=200)
    assert caps.allows(s.SafetyLimits(max_alt_agl_m=200))
    assert not caps.allows(s.SafetyLimits(max_alt_agl_m=250))


def _charter_with(**kw: Any) -> s.MissionCharter:
    return CHARTER.model_copy(update=kw)


@pytest.mark.parametrize(
    "bad",
    [
        {"battery_rtl_pct": 0},
        {"alt_min_agl_m": -10},
        {"max_speed_ms": float("inf")},
        {"alt_min_agl_m": 50, "default_alt_agl_m": 40},
        {"alt_min_agl_m": 120, "alt_max_agl_m": 120, "default_alt_agl_m": 120},
    ],
)
def test_charter_bounds(bad: dict[str, Any]) -> None:
    data = CHARTER.model_dump(mode="json") | bad
    with pytest.raises(ValidationError):
        s.MissionCharter.model_validate(data)


def test_charter_autonomy_mode_is_never_implied() -> None:
    data = CHARTER.model_dump(mode="json")
    del data["autonomy_mode"]
    with pytest.raises(ValidationError, match="autonomy_mode"):
        s.MissionCharter.model_validate(data)


@pytest.mark.parametrize(
    "bad",
    [{"battery_rtl_pct": 0}, {"alt_min_agl_m": -10}, {"max_speed_ms": float("inf")}],
)
def test_proposal_bounds(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        s.CharterProposal(area=AREA, reason_sv="x", **bad)


# --- what the AI can send ----------------------------------------------------------------

FORBIDDEN_FIELDS = {
    "autonomy_mode",
    "limits",
    "approved_by",
    "approved_at",
    "approval_id",
    "home",
    "pilot_position",
    "fence_hash",
    "confirmations",
    "ai_enabled",
}
AI_MODELS: list[type[BaseModel]] = [
    s.SkillRequest,
    s.BrainMessage,
    s.CharterProposal,
    s.ProposeCharterChangeParams,
    s.MissionMemoryUpdate,
    s.DecisionLog,
]


@pytest.mark.parametrize("model", AI_MODELS, ids=lambda m: m.__name__)
def test_ai_models_have_no_control_fields(model: type[BaseModel]) -> None:
    assert set(model.model_fields) & FORBIDDEN_FIELDS == set()


def _skill_request(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "command_id": str(new_id()),
        "skill": "goto",
        "params": {"lat": -35.36, "lon": 149.16, "alt_m": 40},
        "rationale_sv": "x",
        "in_reply_to": None,
        "based_on": None,
        "charter_version": 1,
        "backend": "llm",
        "ai_config_hash": "abc",
    }
    return base | kw


@pytest.mark.parametrize(
    "bad",
    [
        {"skill": "Goto; DROP"},
        {"skill": "x" * 65},
        {"params": {"blob": "x" * (PARAMS_MAX_BYTES + 1)}},
        {"params": {str(i): i for i in range(33)}},
        {"ai_config_hash": "x" * 200},
    ],
)
def test_skill_request_is_bounded(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        s.SkillRequest.model_validate(_skill_request(**bad))


def test_decision_log_prompt_json_is_capped() -> None:
    big = {"x": "y" * PROMPT_JSON_MAX_BYTES}
    with pytest.raises(ValidationError, match="too large"):
        s.DecisionLog.model_validate(_decision_log() | {"prompt_json": big})


def test_memory_update_has_no_server_fields() -> None:
    assert "updated_at" not in s.MissionMemoryUpdate.model_fields


def test_chat_never_resumes_or_takes_over() -> None:
    base = s.ActionOutcome.model_validate(
        {
            "action_id": str(OTHER),
            "vehicle_id": "sim-01",
            "action": "stop",
            "action_seq": 1,
            "via": "chat",
            "by": "operator:demo",
            "result": "dispatched",
            "mode_requested": None,
            "mode_confirmed": None,
            "paths": {},
            "detail_sv": None,
            "t": "2026-10-15T12:00:00Z",
        }
    )
    for action in ("resume", "takeover", "handback"):
        with pytest.raises(ValidationError, match="chat"):
            s.ActionOutcome.model_validate(base.model_dump(mode="json") | {"action": action})


# --- clearance and manual control ------------------------------------------------------------


def test_clearance_validity_is_capped() -> None:
    data = {
        "clearance_id": str(OTHER),
        "mission_id": str(OTHER),
        "vehicle_id": "sim-01",
        "charter_id": str(OTHER),
        "charter_version": 1,
        "valid_for_ms": 10**9,
        "confirmed_by": "operator:demo",
        "origin": "local",
    }
    with pytest.raises(ValidationError):
        s.ClearanceConfirmation.model_validate(data)
    assert "origin" in s.ClearanceState.model_fields


def test_gate_context_knows_about_manual_control() -> None:
    from sd_schemas.gate import GateContext

    assert "manual_control" in GateContext.model_fields
    assert "MANUAL_CONTROL_ACTIVE" in GATE_REASON_CODES


def test_takeover_is_pilot_only_and_ordered() -> None:
    with pytest.raises(ValidationError):
        s.ManualTakeover(takeover_id=OTHER, active=True, action_seq=1, operator_id="brain:ground")
    assert "action_seq" in s.ManualTakeover.model_fields


# --- gate targets ------------------------------------------------------------------------------


def _params(name: str, data: dict[str, Any]) -> s.GateTargets:
    model = SKILL_REGISTRY.get(name).params_model
    return model.model_validate(data).gate_targets(CTX)


MOVING_EXAMPLES: dict[str, dict[str, Any]] = {
    "takeoff": {"alt_m": 40},
    "goto": {"lat": -35.3615, "lon": 149.1652, "alt_m": 40},
    "survey_sector": {"sector_id": "A"},
    "orbit": {"lat": -35.3615, "lon": 149.1652, "radius_m": 30, "alt_m": 40},
    "land_at": {"lat": -35.3615, "lon": 149.1652},
}


def test_every_moving_skill_is_covered_here() -> None:
    moving = {k.name for k in SKILL_REGISTRY if k.moves_vehicle}
    special = {"resume", "return_home"}  # resume needs a resume point; home is always allowed
    stage2 = {"inspect_detection", "follow_track", "follow_me", "find_me"}
    assert moving == set(MOVING_EXAMPLES) | special | stage2


@pytest.mark.parametrize("name", sorted(MOVING_EXAMPLES))
def test_moving_skill_gives_targets(name: str) -> None:
    t = _params(name, MOVING_EXAMPLES[name])
    assert t.resolved
    assert t.points or t.areas
    if name != "land_at":
        assert t.alts_m


@pytest.mark.parametrize("name", sorted(set(MOVING_EXAMPLES) - {"takeoff"}))
def test_moving_skill_without_position_is_unresolved(name: str) -> None:
    model = SKILL_REGISTRY.get(name).params_model
    assert model.model_validate(MOVING_EXAMPLES[name]).gate_targets(NO_POSITION) == UNRESOLVED


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("inspect_detection", {"detection_id": str(OTHER), "alt_m": 30, "standoff_m": 20}),
        (
            "follow_track",
            {"track_id": str(OTHER), "standoff_m": 20, "alt_m": 30, "max_speed_ms": 5},
        ),
        ("follow_me", {"standoff_m": 20, "alt_m": 30, "max_speed_ms": 5}),
        ("find_me", {"alt_m": 30, "hover_offset_m": 20}),
        ("resume", {}),
    ],
)
def test_targets_only_known_onboard_are_unresolved(name: str, data: dict[str, Any]) -> None:
    assert _params(name, data) == UNRESOLVED


def test_survey_unknown_sector_is_unresolved() -> None:
    assert _params("survey_sector", {"sector_id": "Z"}) == UNRESOLVED
    params = s.SurveySectorParams(sector_id="A")
    assert params.gate_targets(NO_CHARTER) == UNRESOLVED


def test_survey_sector_resolves_area_and_default_height() -> None:
    t = _params("survey_sector", {"sector_id": "A"})
    assert t.areas == [CHARTER.sectors[0].area]
    assert t.alts_m == [CHARTER.default_alt_agl_m]
    assert t.path[0].lat == HERE.lat


def test_survey_polygon_and_sector_are_exclusive() -> None:
    with pytest.raises(ValidationError):
        s.SurveySectorParams(polygon=AREA, sector_id="A")


def test_takeoff_falls_back_to_charter_height_and_position() -> None:
    t = s.TakeoffParams().gate_targets(TargetContext(charter=CHARTER, position=HERE, home=None))
    assert t.alts_m == [CHARTER.default_alt_agl_m]
    assert t.points[0].lat == HERE.lat
    assert s.TakeoffParams().gate_targets(NO_CHARTER) == UNRESOLVED


def test_land_at_in_place() -> None:
    t = s.LandAtParams().gate_targets(CTX)
    assert t.resolved
    assert t.points[0].lat == HERE.lat


def test_resume_rechecks_the_remaining_route() -> None:
    ll = s.LatLon(lat=-35.3620, lon=149.1650)
    goto = GotoResumePoint(command_id=OTHER, lat=ll.lat, lon=ll.lon, alt_m=40, speed_ms=8)
    t = s.ResumeParams().gate_targets(CTX.model_copy(update={"resume_point": goto}))
    assert t.points == [ll]
    assert t.alts_m == [40]
    with pytest.raises(ValidationError):
        SurveyResumePoint(
            command_id=OTHER,
            waypoints=[ll],
            alt_m=40,
            speed_ms=7,
            base_index=0,
            seq_offset=0,
            next_index=-1,
            interrupted_at=ll,
        )
    survey = SurveyResumePoint(
        command_id=OTHER,
        waypoints=[ll, ll],
        alt_m=40,
        speed_ms=7,
        base_index=0,
        seq_offset=0,
        next_index=1,
        interrupted_at=ll,
    )
    t = s.ResumeParams().gate_targets(CTX.model_copy(update={"resume_point": survey}))
    assert t.points == [ll]
    assert len(t.path) == 3


def test_return_home_is_never_blocked_by_targets() -> None:
    assert s.ReturnHomeParams().gate_targets(NO_POSITION).resolved


def test_mark_location_targets_its_point() -> None:
    t = s.MarkLocationParams(lat=-35.3615, lon=149.1652, label="M1").gate_targets(CTX)
    assert t.points[0].lon == 149.1652


def test_stop_follow_cannot_restart_movement() -> None:
    with pytest.raises(ValidationError):
        s.StopFollowParams.model_validate({"then": "resume"})


def test_moving_skill_cannot_use_no_targets() -> None:
    from sd_schemas.skills import HoldParams, SkillSpec

    with pytest.raises(ValidationError, match="gate targets"):
        SkillSpec(
            name="sneak",
            stage=2,
            status="stub",
            runs_on="onboard",
            params_model=HoldParams,
            moves_vehicle=True,
            allowed_when="airborne",
            rate_class="action",
            description_en="x",
            synonyms_sv=["x"],
            title_sv="x",
        )


def test_allowed_from_only_for_implemented_onboard() -> None:
    from sd_schemas.executor import ExecutorState
    from sd_schemas.skills import HoldParams, SkillSpec

    with pytest.raises(ValidationError, match="allowed_from"):
        SkillSpec(
            name="early",
            stage=1,
            status="stub",
            runs_on="onboard",
            params_model=HoldParams,
            moves_vehicle=False,
            allowed_when="airborne",
            allowed_from=frozenset({ExecutorState.HOLD}),
            rate_class="action",
            description_en="x",
            synonyms_sv=["x"],
            title_sv="x",
        )


# --- reason codes ------------------------------------------------------------------------------


def test_each_code_has_one_text() -> None:
    texts: dict[str, str] = {}
    for table in (
        GATE_REASON_CODES,
        APPROVAL_OUTCOME_CODES,
        CHARTER_VALIDATION_CODES,
        LIMITS_CODES,
        ONBOARD_REASON_CODES,
    ):
        for code, text in table.items():
            assert texts.setdefault(code, text) == text, code
