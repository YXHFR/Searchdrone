"""Every gate rule just inside (no code) and just outside (its code is in reason_codes).

Limits are read from the context, never hard-coded, except where the acceptance list
names the owner's standard values (120 m, 15 m/s, 500 m, 15 %).
"""

from __future__ import annotations

from typing import Any

import pytest
from gate_fixtures import (
    CHARTER,
    L_AREA,
    NOW,
    ON_GROUND,
    clearance,
    ctx,
    goto,
    ground_ctx,
    latch,
    latlon,
    onboard_ctx,
    point,
    polygon,
    registry_with,
    req,
    spec,
    vehicle,
)

from sd_safety import evaluate
from sd_safety.priority import REASON_PRIORITY
from sd_schemas.base import LatLon, LatLonAlt
from sd_schemas.charter import AutonomyMode
from sd_schemas.executor import ExecutorState
from sd_schemas.gate import GateContext, GateDecision, GateRequest, StateRef
from sd_schemas.mission import MissionState
from sd_schemas.safety import SafetyLimits


def codes(r: GateRequest, c: GateContext) -> list[str]:
    return evaluate(r, c).reason_codes


def decide(r: GateRequest, c: GateContext) -> GateDecision:
    return evaluate(r, c)


def test_baseline_allows() -> None:
    d = decide(req(), ctx())
    assert d.decision == "ALLOW"
    assert d.reason_codes == []
    assert d.rule_ids == []
    assert d.side == "ground"
    takeoff = decide(req("takeoff", {"alt_m": 40}), ground_ctx())
    assert takeoff.decision == "ALLOW", takeoff


# --- 1. height --------------------------------------------------------------------------------


def test_height_against_the_limit() -> None:
    limit = SafetyLimits().max_alt_agl_m
    assert limit == 120
    assert codes(req(params=goto(alt_m=limit)), ctx()) == []
    c = codes(req(params=goto(alt_m=limit + 0.1)), ctx())
    assert "ALT_ABOVE_LIMIT" in c


def test_height_against_the_charter() -> None:
    low = CHARTER.model_copy(update={"alt_max_agl_m": 60.0})
    c = ctx(charter=low)
    assert codes(req(params=goto(alt_m=60)), c) == []
    assert codes(req(params=goto(alt_m=60.1)), c) == ["ALT_ABOVE_CHARTER"]
    assert codes(req(params=goto(alt_m=CHARTER.alt_min_agl_m)), ctx()) == []
    assert codes(req(params=goto(alt_m=CHARTER.alt_min_agl_m - 0.1)), ctx()) == [
        "ALT_BELOW_CHARTER"
    ]


def test_goto_200_m_gives_both_height_codes_in_order() -> None:
    d = decide(req(params=goto(alt_m=200)), ctx())
    assert d.decision == "REJECT"
    assert d.reason_codes == ["ALT_ABOVE_LIMIT", "ALT_ABOVE_CHARTER"]
    assert d.rule_ids == ["R-ALT"]
    assert d.details["alt_m"] == 200
    assert d.message_sv.endswith("(och 1 till)")


def test_raised_limits_are_followed() -> None:
    raised = SafetyLimits(max_alt_agl_m=200)
    charter = CHARTER.model_copy(update={"alt_max_agl_m": 200.0})
    assert codes(req(params=goto(alt_m=150)), ctx(limits=raised, charter=charter)) == []


# --- 2. speed ---------------------------------------------------------------------------------


def test_speed_is_rejected_not_clamped() -> None:
    limit = SafetyLimits().max_speed_ms
    assert codes(req(params=goto(speed_ms=limit)), ctx()) == []
    d = decide(req(params=goto(speed_ms=limit + 0.1)), ctx())
    assert d.decision == "REJECT"
    assert "SPEED_ABOVE_LIMIT" in d.reason_codes
    assert d.details["speed_ms"] == pytest.approx(limit + 0.1)


def test_speed_against_the_charter() -> None:
    slow = ctx(charter=CHARTER.model_copy(update={"max_speed_ms": 8.0}))
    assert codes(req(params=goto(speed_ms=8)), slow) == []
    assert codes(req(params=goto(speed_ms=8.1)), slow) == ["SPEED_ABOVE_CHARTER"]
    assert codes(req(params=goto(speed_ms=8.1)), ctx(charter=None, mission_state=None)) != []


# --- 3. distance from the pilot ----------------------------------------------------------------


def _far_area() -> GateContext:
    """A charter area reaching 600 m south, with the limits and caps that allow it."""
    area = polygon((-100, -600), (100, -600), (100, 100), (-100, 100))
    limits = SafetyLimits(max_range_from_pilot_m=900)
    charter = CHARTER.model_copy(update={"area": area, "max_range_from_pilot_m": 500.0})
    return ctx(charter=charter, limits=limits)


def test_pilot_distance_from_the_charter() -> None:
    c = _far_area()
    assert codes(req(params=goto(0, -499.5)), c) == []
    d = decide(req(params=goto(0, -500.5)), c)
    assert d.reason_codes == ["PILOT_DISTANCE"]
    assert d.details["pilot_limit_m"] == 500
    assert d.details["pilot_distance_m"] == pytest.approx(500.5, abs=0.05)


def test_pilot_distance_from_the_limits() -> None:
    c = _far_area()
    charter = c.charter
    assert charter is not None
    c = ctx(
        charter=charter.model_copy(update={"max_range_from_pilot_m": 900.0}),
        limits=SafetyLimits(max_range_from_pilot_m=450),
    )
    assert "PILOT_DISTANCE" not in codes(req(params=goto(0, -449.5)), c)
    assert "PILOT_DISTANCE" in codes(req(params=goto(0, -450.5)), c)


def test_pilot_distance_uses_the_effective_pilot_position_too() -> None:
    v = vehicle(pilot_position=point(0, 300))  # the pilot walked north
    c = _far_area()
    assert "PILOT_DISTANCE" in codes(req(params=goto(0, -250)), ctx(vehicle=v, charter=c.charter))


def test_pilot_distance_without_a_charter_uses_the_vehicle() -> None:
    far = goto(0, 600)
    lat, lon = latlon(0, 600)
    c = ctx(charter=None, mission_state=MissionState.DRAFT)
    mark = {"lat": lat, "lon": lon, "label": "x"}
    assert "PILOT_DISTANCE" in codes(req("mark_location", mark, charter_version=None), c)
    no_vehicle = ctx(charter=None, vehicle=None, mission_state=MissionState.DRAFT)
    assert codes(req("mark_location", mark, charter_version=None), no_vehicle) == []
    assert far  # the goto form of the same point is covered by the moving-order tests


# --- 4. area, routes and concave charters -------------------------------------------------------


def test_target_outside_the_area() -> None:
    d = decide(req(params=goto(0, 310)), ctx())
    assert d.reason_codes == ["OUTSIDE_CHARTER"]
    assert d.details["outside_by_m"] == pytest.approx(10, abs=0.2)
    assert codes(req(params=goto(0, 299.5)), ctx()) == []


def test_route_through_the_notch_of_a_concave_charter() -> None:
    """Both ends inside the L, but the straight line crosses the notch."""
    here = vehicle(**dict(zip(("lat", "lon"), latlon(250, 50), strict=True)))
    c = ctx(vehicle=here)
    assert codes(req(params=goto(250, 0)), c) == []  # stays in the lower arm
    d = decide(req(params=goto(50, 250)), c)
    assert d.reason_codes == ["OUTSIDE_CHARTER"]
    assert d.details["outside_by_m"] > 10


def test_route_along_an_edge_and_through_the_reflex_corner() -> None:
    edge = vehicle(**dict(zip(("lat", "lon"), latlon(-300, -300), strict=True)))
    assert codes(req(params=goto(300, -300)), ctx(vehicle=edge)) == []  # along the south edge
    corner_line = vehicle(**dict(zip(("lat", "lon"), latlon(300, 100), strict=True)))
    assert codes(req(params=goto(100, 300)), ctx(vehicle=corner_line)) == ["OUTSIDE_CHARTER"]
    touching = vehicle(**dict(zip(("lat", "lon"), latlon(0, 0), strict=True)))
    assert codes(req(params=goto(200, 200)), ctx(vehicle=touching)) == ["OUTSIDE_CHARTER"]
    assert codes(req(params=goto(100, 100)), ctx(vehicle=touching)) == []  # ends on the corner


def test_onboard_planned_path_is_checked_leg_by_leg() -> None:
    ends_inside = [_lla(250, 50), _lla(50, 250)]  # crosses the notch
    c = onboard_ctx()
    planned = req(params=goto(50, 250), phase="onboard", planned_path=ends_inside)
    assert "OUTSIDE_CHARTER" in codes(planned, c)
    around = [_lla(250, 50), _lla(50, 50), _lla(50, 250)]
    r = req(params=goto(50, 250), phase="onboard", planned_path=around)
    here = vehicle(**dict(zip(("lat", "lon"), latlon(250, 50), strict=True)))
    assert "OUTSIDE_CHARTER" in codes(r, onboard_ctx(vehicle=here))  # its own target crosses
    resume = req("resume", {}, phase="onboard", planned_path=around)
    assert codes(resume, onboard_ctx(vehicle=here)) == []


def test_planned_path_heights_are_checked() -> None:
    high = [_lla(0, 50, alt=40), _lla(0, 100, alt=130)]
    r = req("resume", {}, phase="onboard", planned_path=high)
    assert codes(r, onboard_ctx()) == ["ALT_ABOVE_LIMIT", "ALT_ABOVE_CHARTER"]
    without_charter = ctx(side="onboard", charter=None, mission_state=None, snapshot_age_ms=0)
    r0 = req("resume", {}, phase="onboard", planned_path=high, charter_version=None)
    assert "ALT_ABOVE_LIMIT" not in codes(r0, without_charter)


def _lla(east: float, north: float, alt: float = 40.0) -> LatLonAlt:
    lat, lon = latlon(east, north)
    return LatLonAlt(lat=lat, lon=lon, alt_amsl_m=CHARTER.home.alt_amsl_m + alt)


def test_survey_area_must_lie_inside_the_charter() -> None:
    inside = polygon((-250, -250), (-50, -250), (-50, -50), (-250, -50))
    assert codes(req("survey_sector", {"polygon": inside}), ctx()) == []
    notch = polygon((50, 50), (250, 50), (250, 250), (50, 250))
    d = decide(req("survey_sector", {"polygon": notch}), ctx())
    assert d.reason_codes == ["OUTSIDE_CHARTER"]
    assert codes(req("survey_sector", {"sector_id": "A"}), ctx()) == []


def test_unknown_sector_and_self_crossing_areas_are_invalid() -> None:
    assert codes(req("survey_sector", {"sector_id": "Q"}), ctx()) == ["SCHEMA_INVALID"]
    bow = polygon((-200, -200), (-50, -50), (-50, -200), (-200, -50))
    assert "SCHEMA_INVALID" in codes(req("survey_sector", {"polygon": bow}), ctx())


def test_a_broken_charter_area_rejects_every_target() -> None:
    bow = polygon((-300, -300), (300, 300), (300, -300), (-300, 300))
    c = ctx(charter=CHARTER.model_copy(update={"area": bow}))
    d = decide(req(params=goto(0, 0)), c)
    assert "OUTSIDE_CHARTER" in d.reason_codes
    assert d.details["charter_area"] == "invalid"


def test_mark_location_must_be_inside_the_charter() -> None:
    lat, lon = latlon(0, 310)
    assert codes(req("mark_location", {"lat": lat, "lon": lon, "label": "M1"}), ctx()) == [
        "OUTSIDE_CHARTER"
    ]


def test_area_beyond_the_flight_controller_fence() -> None:
    near = polygon((-50, -510), (50, -510), (50, 200), (-50, 200))  # corner 512.4 m from home
    far = polygon((-50, -520), (50, -520), (50, 200), (-50, 200))  # corner 522.4 m
    limits = SafetyLimits(max_range_from_pilot_m=500)
    roomy = CHARTER.model_copy(update={"max_range_from_pilot_m": 500.0})
    ok = codes(req(), ctx(charter=roomy.model_copy(update={"area": near}), limits=limits))
    assert ok == []
    d = decide(req(), ctx(charter=roomy.model_copy(update={"area": far}), limits=limits))
    assert "AREA_BEYOND_FC_FENCE" in d.reason_codes
    assert d.details["fc_fence_limit_m"] == 515
    assert codes(req("return_home", {}), ctx(charter=roomy.model_copy(update={"area": far}))) == []
    report = req("report", {"text_sv": "hej"})
    assert codes(report, ctx(charter=roomy.model_copy(update={"area": far}))) == []


# --- 5. battery ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("battery", "expected"),
    [(15 + 5.1, []), (15 + 5.0, ["BATTERY_RESERVE"]), (None, ["BATTERY_UNKNOWN"])],
)
def test_brain_battery_margin(battery: float | None, expected: list[str]) -> None:
    assert codes(req(), ctx(vehicle=vehicle(battery_pct=battery))) == expected


@pytest.mark.parametrize("skill", ["return_home", "hold", "report", "mark_location"])
def test_battery_exempt_skills_pass_on_reserve(skill: str) -> None:
    params = {"return_home": {}, "hold": {}, "report": {"text_sv": "låg"}}.get(skill)
    if params is None:
        lat, lon = latlon(0, 0)
        params = {"lat": lat, "lon": lon, "label": "M1"}
    for battery in (10.0, None):
        assert codes(req(skill, params), ctx(vehicle=vehicle(battery_pct=battery))) == []


def test_battery_floor_for_every_issuer() -> None:
    operator = req(issuer_kind="operator", issuer_ref="operator:demo")
    assert codes(operator, ctx(vehicle=vehicle(battery_pct=15.1))) == []
    assert codes(operator, ctx(vehicle=vehicle(battery_pct=15.0))) == ["BATTERY_RESERVE"]
    assert codes(operator, ctx(vehicle=vehicle(battery_pct=None))) == []


def test_battery_level_comes_from_the_charter_but_never_below_the_limit() -> None:
    high = ctx(charter=CHARTER.model_copy(update={"battery_rtl_pct": 30.0}))
    assert codes(req(), high.model_copy(update={"vehicle": vehicle(battery_pct=25)})) == [
        "BATTERY_RESERVE"
    ]
    low_charter = CHARTER.model_copy(update={"battery_rtl_pct": 15.0})
    strict_limits = SafetyLimits(battery_rtl_pct=25)
    c = ctx(charter=low_charter, limits=strict_limits, vehicle=vehicle(battery_pct=28))
    assert codes(req(), c) == ["BATTERY_RESERVE"]
    no_charter = ctx(charter=None, vehicle=vehicle(battery_pct=18), mission_state=None)
    assert "BATTERY_RESERVE" in codes(req(charter_version=None), no_charter)


def test_returning_on_battery_rejects_ai_moves() -> None:
    c = ctx(mission_state=MissionState.RETURNING, returning_cause="battery")
    assert "FAILSAFE_ACTIVE" in codes(req(), c)
    assert codes(req("report", {"text_sv": "hem"}), c) == []


# --- 6. registry ----------------------------------------------------------------------------


def test_skill_outside_allowed_skills() -> None:
    narrow = ctx(charter=CHARTER.model_copy(update={"allowed_skills": ["goto"]}))
    assert codes(req("survey_sector", {"sector_id": "A"}), narrow) == ["SKILL_NOT_ALLOWED"]
    for always in ("hold", "return_home"):
        assert codes(req(always, {}), narrow) == []
    operator = req(
        "survey_sector", {"sector_id": "A"}, issuer_kind="operator", issuer_ref="operator:demo"
    )
    assert codes(operator, narrow) == []


def test_stub_is_not_implemented_even_after_approval() -> None:
    orbit = {"lat": latlon(0, 0)[0], "lon": latlon(0, 0)[1], "radius_m": 50, "alt_m": 40}
    assert "SKILL_NOT_IMPLEMENTED" in codes(req("orbit", orbit), ctx())
    approved = req("orbit", orbit, phase="approve")
    assert "SKILL_NOT_IMPLEMENTED" in codes(approved, ctx())


def _orbit(east: float, north: float, radius_m: float) -> dict[str, Any]:
    lat, lon = latlon(east, north)
    return {"lat": lat, "lon": lon, "radius_m": radius_m, "alt_m": 40}


def test_orbit_circle_is_checked_not_only_its_centre() -> None:
    # Once orbit is built, the whole circle must stay in the charter area: the centre at
    # 250 m east is inside (the area ends at 300 m), a 60 m circle around it is not.
    built = ctx(registry=registry_with(spec("orbit", status="implemented")))
    assert "OUTSIDE_CHARTER" not in codes(req("orbit", _orbit(250, 0, 40)), built)
    assert "OUTSIDE_CHARTER" in codes(req("orbit", _orbit(250, 0, 60)), built)
    # The circle also counts for the distance from the pilot (500 m): centre at 460 m.
    wide = CHARTER.model_copy(
        update={"area": polygon((-600, -600), (600, -600), (600, 600), (-600, 600))}
    )
    far = ctx(registry=registry_with(spec("orbit", status="implemented")), charter=wide)
    assert "PILOT_DISTANCE" not in codes(req("orbit", _orbit(0, 460, 30)), far)
    assert "PILOT_DISTANCE" in codes(req("orbit", _orbit(0, 460, 50)), far)


@pytest.mark.parametrize("name", ["set_param", "arm", "disable_fence", "no_such_skill"])
def test_unknown_and_forbidden_names(name: str) -> None:
    d = decide(req(name, {}), ctx())
    assert d.decision == "REJECT"
    assert "SKILL_UNKNOWN" in d.reason_codes
    assert d.reason_codes[0] == "SKILL_UNKNOWN"


def test_forbidden_name_is_unknown_even_if_a_registry_has_it() -> None:
    sneaky = spec("hold").model_copy(update={"name": "set_mode"})
    c = ctx(registry=registry_with(sneaky))
    assert "SKILL_UNKNOWN" in codes(req("set_mode", {}), c)
    renamed = {**dict(c.registry), "hold": sneaky}
    assert "SKILL_UNKNOWN" in codes(req("hold", {}), ctx(registry=renamed))


def test_bad_parameters_are_schema_invalid() -> None:
    assert codes(req(params={"lat": 1}), ctx()) == ["SCHEMA_INVALID"]
    assert codes(req(params=goto(alt_m=-5)), ctx()) == ["SCHEMA_INVALID"]


def test_ai_not_allowed_skill() -> None:
    hidden = spec("hold", ai_allowed=False)
    c = ctx(registry=registry_with(hidden))
    assert codes(req("hold", {}), c) == ["FORBIDDEN_ACTION"]
    op = req("hold", {}, issuer_kind="operator", issuer_ref="operator:demo")
    assert codes(op, c) == []


def test_autonomy_field_from_the_ai_is_forbidden() -> None:
    proposal = {"area": L_AREA, "reason_sv": "x", "autonomy_mode": "WITHIN_CHARTER"}
    c = codes(req("propose_charter_change", proposal), ctx())
    assert "AUTONOMY_CHANGE_FORBIDDEN" in c
    assert "SCHEMA_INVALID" in c


# --- 7. approvals -------------------------------------------------------------------------------


def test_mode_a_needs_approval_for_every_step_but_report_and_mark() -> None:
    a = CHARTER.model_copy(update={"autonomy_mode": AutonomyMode.APPROVE_EACH})
    c = ctx(charter=a, autonomy_mode=AutonomyMode.APPROVE_EACH)
    for skill, params in (("goto", None), ("hold", {}), ("return_home", {})):
        d = decide(req(skill, params), c)
        assert d.decision == "PENDING_APPROVAL", skill
        assert d.reason_codes == ["NEEDS_APPROVAL_MODE_A"]
    assert decide(req("report", {"text_sv": "ok"}), c).decision == "ALLOW"
    approved = decide(req(phase="approve"), c)
    assert approved.decision == "ALLOW"


def test_mode_a_if_either_the_context_or_the_charter_says_so() -> None:
    c = ctx(autonomy_mode=AutonomyMode.APPROVE_EACH)
    assert decide(req(), c).reason_codes == ["NEEDS_APPROVAL_MODE_A"]


def test_mode_b_approval_skills() -> None:
    proposal = {"area": L_AREA, "reason_sv": "större"}
    d = decide(req("propose_charter_change", proposal), ctx())
    assert d.decision == "PENDING_APPROVAL"
    assert d.reason_codes == ["NEEDS_APPROVAL_SKILL"]
    flagged = spec("goto", requires_approval_in_b=True)
    assert decide(req(), ctx(registry=registry_with(flagged))).reason_codes == [
        "NEEDS_APPROVAL_SKILL"
    ]


def test_approval_codes_never_become_a_card_onboard() -> None:
    a = CHARTER.model_copy(update={"autonomy_mode": AutonomyMode.APPROVE_EACH})
    d = decide(req(phase="onboard"), onboard_ctx(charter=a))
    assert d.decision == "ALLOW"
    assert d.side == "onboard"


def test_a_rejection_wins_over_an_approval_code() -> None:
    a = CHARTER.model_copy(update={"autonomy_mode": AutonomyMode.APPROVE_EACH})
    d = decide(req(params=goto(alt_m=200)), ctx(charter=a))
    assert d.decision == "REJECT"
    assert d.reason_codes[-1] == "NEEDS_APPROVAL_MODE_A"


# --- 8. takeoff --------------------------------------------------------------------------------


TAKEOFF = {"alt_m": 40}


def test_takeoff_needs_an_approved_charter() -> None:
    no_charter = ground_ctx(charter=None, autonomy_mode=None)
    assert "CHARTER_NOT_APPROVED" in codes(
        req("takeoff", TAKEOFF, charter_version=None), no_charter
    )
    unapproved = CHARTER.model_copy(update={"approved_at": None})
    assert "CHARTER_NOT_APPROVED" in codes(req("takeoff", TAKEOFF), ground_ctx(charter=unapproved))


def test_takeoff_needs_an_installed_charter_and_a_verified_fence() -> None:
    assert "CHARTER_NOT_INSTALLED" in codes(
        req("takeoff", TAKEOFF), ground_ctx(charter_installed=False)
    )
    assert "FENCE_NOT_VERIFIED" in codes(req("takeoff", TAKEOFF), ground_ctx(fence_verified=False))


@pytest.mark.parametrize(
    ("update", "code"),
    [
        ({"clearance": None}, "CLEARANCE_REQUIRED"),
        ({"clearance": clearance(consumed=True)}, "CLEARANCE_REQUIRED"),
        ({"clearance": clearance(origin="remote")}, "CLEARANCE_REQUIRED"),
        ({"clearance": clearance(charter_version=1)}, "CLEARANCE_REQUIRED"),
        ({"clearance": clearance(received_mono_ms=NOW + 5)}, "CLEARANCE_REQUIRED"),
        ({"clearance": clearance(received_mono_ms=NOW - 120_001)}, "CLEARANCE_EXPIRED"),
        ({"clearance": clearance(valid_for_ms=5_000)}, "CLEARANCE_EXPIRED"),
    ],
)
def test_takeoff_clearance(update: dict[str, Any], code: str) -> None:
    assert code in codes(req("takeoff", TAKEOFF), ground_ctx(**update))


def test_clearance_bound_to_mission_vehicle_and_charter() -> None:
    from sd_schemas.base import new_id

    for field, value in (
        ("mission_id", new_id()),
        ("vehicle_id", "sim-02"),
        ("charter_id", new_id()),
    ):
        c = ground_ctx(clearance=clearance(**{field: value}))
        assert "CLEARANCE_REQUIRED" in codes(req("takeoff", TAKEOFF), c), field
    assert (
        codes(
            req("takeoff", TAKEOFF), ground_ctx(clearance=clearance(received_mono_ms=NOW - 120_000))
        )
        == []
    )


def test_operator_and_mode_a_get_a_clearance_tick_box() -> None:
    op = req("takeoff", TAKEOFF, issuer_kind="operator", issuer_ref="operator:demo")
    d = decide(op, ground_ctx(clearance=None))
    assert d.decision == "PENDING_APPROVAL"
    assert d.reason_codes == ["NEEDS_CLEARANCE"]
    assert d.required_confirmations == ["clear_around_drone"]
    ticked = op.model_copy(
        update={"phase": "approve", "approval_confirmations": ["clear_around_drone"]}
    )
    assert decide(ticked, ground_ctx(clearance=None)).decision == "ALLOW"
    unticked = op.model_copy(update={"phase": "approve"})
    assert decide(unticked, ground_ctx(clearance=None)).reason_codes == ["NEEDS_CLEARANCE"]
    assert decide(op, ground_ctx()).decision == "ALLOW"
    a = CHARTER.model_copy(update={"autonomy_mode": AutonomyMode.APPROVE_EACH})
    brain_a = decide(req("takeoff", TAKEOFF), ground_ctx(clearance=None, charter=a))
    assert brain_a.reason_codes == ["NEEDS_CLEARANCE", "NEEDS_APPROVAL_MODE_A"]
    assert brain_a.required_confirmations == ["clear_around_drone"]


def test_rejected_card_has_no_tick_box() -> None:
    op = req("takeoff", TAKEOFF, issuer_kind="operator", issuer_ref="operator:demo")
    d = decide(op, ground_ctx(clearance=None, fence_verified=False))
    assert d.decision == "REJECT"
    assert d.required_confirmations == []


def test_clearance_onboard_is_always_required() -> None:
    c = onboard_ctx(
        vehicle=vehicle(**ON_GROUND), clearance=None, executor_state=ExecutorState.PREFLIGHT
    )
    op = req(
        "takeoff", TAKEOFF, issuer_kind="operator", issuer_ref="operator:demo", phase="onboard"
    )
    assert "CLEARANCE_REQUIRED" in codes(op, c)


def test_clearance_is_not_needed_in_the_air_or_for_exempt_skills() -> None:
    assert codes(req("hold", {}), ctx(clearance=None)) == []
    ground_report = req("report", {"text_sv": "redo"})
    assert codes(ground_report, ground_ctx(clearance=None)) == []


@pytest.mark.parametrize(
    ("vehicle_update", "ctx_update", "why"),
    [
        ({"armed": True}, {}, "armed"),
        ({"landed_state": "takeoff"}, {}, "not_on_ground"),
        ({"battery_pct": None}, {}, "battery_unknown"),
        ({"battery_pct": 34.9}, {}, "battery_low"),
        ({}, {"preflight_ok": None}, "preflight_not_ok"),
        ({}, {"preflight_ok": False}, "preflight_not_ok"),
    ],
)
def test_takeoff_preconditions(
    vehicle_update: dict[str, Any], ctx_update: dict[str, Any], why: str
) -> None:
    v = vehicle(**{**ON_GROUND, **vehicle_update})
    d = decide(req("takeoff", TAKEOFF), ground_ctx(vehicle=v, **ctx_update))
    assert "TAKEOFF_PRECONDITION" in d.reason_codes
    assert why in d.details["takeoff"]
    ok = vehicle(**{**ON_GROUND, "battery_pct": 35.0})
    assert codes(req("takeoff", TAKEOFF), ground_ctx(vehicle=ok)) == []


def test_takeoff_without_vehicle_state() -> None:
    d = decide(req("takeoff", TAKEOFF), ground_ctx(vehicle=None))
    assert {"STATE_STALE", "TAKEOFF_PRECONDITION"} <= set(d.reason_codes)
    assert "no_vehicle_state" in d.details["takeoff"]


def test_takeoff_uses_the_default_height_and_home() -> None:
    assert codes(req("takeoff", {}), ground_ctx()) == []
    low = CHARTER.model_copy(update={"default_alt_agl_m": 15.0, "alt_min_agl_m": 15.0})
    assert codes(req("takeoff", {"alt_m": 10}), ground_ctx(charter=low)) == ["ALT_BELOW_CHARTER"]


# --- 9. latches and people in charge ------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "code"),
    [
        ("stop", "RESUME_NEEDS_OPERATOR"),
        ("watchdog_stop", "RESUME_NEEDS_OPERATOR"),
        ("return_home", "OPERATOR_OVERRIDE_ACTIVE"),
        ("watchdog_rtl", "OPERATOR_OVERRIDE_ACTIVE"),
        ("battery_rtl", "FAILSAFE_ACTIVE"),
    ],
)
def test_latch_blocks_ai_moves(action: str, code: str) -> None:
    c = ctx(latch=latch(action), stop_latch=True)
    d = decide(req(), c)
    assert code in d.reason_codes
    assert d.details["latch"] == action
    assert code in codes(req("return_home", {}), c)  # even flying home
    assert codes(req("report", {"text_sv": "stoppad"}), c) == []
    lat, lon = latlon(0, 0)
    assert codes(req("mark_location", {"lat": lat, "lon": lon, "label": "M1"}), c) == []


def test_stop_latch_without_a_latch_record() -> None:
    assert "RESUME_NEEDS_OPERATOR" in codes(req(), ctx(stop_latch=True))
    assert "RESUME_NEEDS_OPERATOR" in codes(req(phase="onboard"), onboard_ctx(stop_latch=True))


def test_operator_must_release_before_new_orders() -> None:
    op = req(issuer_kind="operator", issuer_ref="operator:demo")
    c = ctx(latch=latch("stop"), stop_latch=True)
    assert codes(op, c) == ["RELEASE_REQUIRED"]
    assert codes(op.model_copy(update={"skill": "return_home", "params": {}}), c) == []


def test_paused_by_a_person() -> None:
    for by in ("operator", "system", None):
        c = ctx(mission_state=MissionState.PAUSED, paused_by=by)
        assert "RESUME_NEEDS_OPERATOR" in codes(req(), c), by
    mine = ctx(mission_state=MissionState.PAUSED, paused_by="brain")
    assert codes(req("resume", {}), mine) == []


def test_returning_by_a_person() -> None:
    for cause in ("operator", "system", None):
        c = ctx(mission_state=MissionState.RETURNING, returning_cause=cause)
        assert "OPERATOR_OVERRIDE_ACTIVE" in codes(req(), c), cause
    mine = ctx(mission_state=MissionState.RETURNING, returning_cause="brain")
    assert codes(req(), mine) == []


def test_failsafe_and_held() -> None:
    for c in (
        ctx(mission_state=MissionState.FAILSAFE),
        ctx(mission_state=MissionState.RETURNING, returning_cause="failsafe"),
        ctx(executor_state=ExecutorState.FAILSAFE),
    ):
        assert "FAILSAFE_ACTIVE" in codes(req(), c)
    assert "CONTROL_HELD" in codes(req(), ctx(mission_state=MissionState.HELD))
    assert "CONTROL_HELD" in codes(req(), ctx(executor_state=ExecutorState.HELD))
    held_onboard = onboard_ctx(executor_state=ExecutorState.HELD)
    assert "CONTROL_HELD" in codes(req(phase="onboard"), held_onboard)
    assert codes(req("report", {"text_sv": "x"}), ctx(mission_state=MissionState.HELD)) == []


def test_manual_control_pauses_the_ai() -> None:
    c = ctx(manual_control=True)
    assert "MANUAL_CONTROL_ACTIVE" in codes(req(), c)
    assert "MANUAL_CONTROL_ACTIVE" in codes(req("hold", {}), c)
    assert codes(req("report", {"text_sv": "x"}), c) == []
    op = req(issuer_kind="operator", issuer_ref="operator:demo")
    assert codes(op, c) == []


# --- the remaining rules -----------------------------------------------------------------


def test_ai_switched_off() -> None:
    c = ctx(ai_enabled=False)
    assert codes(req(), c) == ["AI_DISABLED"]
    assert codes(req("report", {"text_sv": "av"}), c) == []
    assert codes(req(phase="onboard"), onboard_ctx(ai_enabled=False)) == []  # ground rule


@pytest.mark.parametrize(
    ("state", "allowed", "denied"),
    [
        (MissionState.DRAFT, "report", "goto"),
        (MissionState.CHARTER_PROPOSED, "mark_location", "takeoff"),
        (MissionState.LANDED, "propose_charter_change", "takeoff"),
        (MissionState.READY, "takeoff", "goto"),
        (MissionState.CLOSED, None, "report"),
        (None, None, "report"),
    ],
)
def test_mission_state(state: MissionState | None, allowed: str | None, denied: str) -> None:
    def ok(skill: str) -> bool:
        return "MISSION_STATE_INVALID" not in codes(req(skill, {}), ctx(mission_state=state))

    if allowed is not None:
        assert ok(allowed)
    assert not ok(denied)


def test_charter_version() -> None:
    assert codes(req(charter_version=1), ctx()) == ["CHARTER_VERSION_MISMATCH"]
    assert codes(req(charter_version=None), ctx()) == ["CHARTER_VERSION_MISMATCH"]
    draft = ctx(charter=None, mission_state=MissionState.DRAFT)
    assert codes(req("report", {"text_sv": "x"}, charter_version=None), draft) == []
    assert codes(req("report", {"text_sv": "x"}, charter_version=3), draft) == [
        "CHARTER_VERSION_MISMATCH"
    ]
    op = req(issuer_kind="operator", issuer_ref="operator:demo", charter_version=1)
    assert codes(op, ctx()) == []  # the server fills in the operator's version


def test_link_down() -> None:
    assert codes(req(), ctx(sdlink_up=False)) == ["LINK_DOWN"]
    assert codes(req("report", {"text_sv": "x"}), ctx(sdlink_up=False)) == []


@pytest.mark.parametrize(
    ("update", "stale"),
    [
        ({"snapshot_age_ms": 2000}, False),
        ({"snapshot_age_ms": 2001}, True),
        ({"vehicle_age_ms": 2001}, True),
        ({"vehicle_age_ms": None}, True),
        ({"snapshot_age_ms": None}, True),
        ({"vehicle": None}, True),
    ],
)
def test_stale_state(update: dict[str, Any], stale: bool) -> None:
    assert ("STATE_STALE" in codes(req(), ctx(**update))) is stale


def test_order_based_on_an_old_state() -> None:
    assert codes(req(based_on=StateRef(state_epoch=7, state_seq=40)), ctx()) == []
    assert codes(req(based_on=StateRef(state_epoch=7, state_seq=39)), ctx()) == ["STATE_CHANGED"]
    assert codes(req(based_on=StateRef(state_epoch=6, state_seq=99)), ctx()) == ["STATE_CHANGED"]
    assert codes(req(based_on=None), ctx()) == ["STATE_CHANGED"]
    assert codes(req(), ctx(state_epoch=None)) == ["STATE_CHANGED"]
    assert codes(req(), ctx(last_transition_state_seq=None)) == ["STATE_CHANGED"]
    assert codes(req("report", {"text_sv": "x"}, based_on=None), ctx()) == []


def test_wrong_phase_of_flight() -> None:
    assert "INVALID_TRANSITION" in codes(req("takeoff", TAKEOFF), ctx())
    on_ground = ctx(vehicle=vehicle(**ON_GROUND))
    assert "INVALID_TRANSITION" in codes(req(), on_ground)
    low = ctx(vehicle=vehicle(alt_rel_home_m=2.0))
    assert "INVALID_TRANSITION" in codes(req(), low)
    assert "INVALID_TRANSITION" not in codes(req(), ctx(vehicle=vehicle(alt_rel_home_m=2.1)))


def test_operator_absent() -> None:
    assert codes(req(), ctx(operator_present=False)) == ["OPERATOR_ABSENT"]
    assert codes(req("hold", {}), ctx(operator_present=False)) == []


def test_pending_limit() -> None:
    assert codes(req(), ctx(brain_pending_approvals=1)) == ["PENDING_LIMIT"]


def test_rate_limit_counts_only_passed_orders() -> None:
    # The caller only records orders that got ALLOW or PENDING_APPROVAL; three rejected
    # orders are simply not in the list.
    assert codes(req(), ctx(recent_brain_actions_mono_ms=())) == []
    assert codes(req(), ctx(recent_brain_actions_mono_ms=(NOW - 2999,))) == ["RATE_LIMIT"]
    assert codes(req(), ctx(recent_brain_actions_mono_ms=(NOW - 3000,))) == []
    fourteen = tuple(NOW - 4000 - i * 1000 for i in range(14))
    assert codes(req(), ctx(recent_brain_actions_mono_ms=fourteen)) == []
    fifteen = tuple(NOW - 4000 - i * 1000 for i in range(15))
    assert codes(req(), ctx(recent_brain_actions_mono_ms=fifteen)) == ["RATE_LIMIT"]
    future = ctx(recent_brain_actions_mono_ms=(NOW + 10,))
    assert codes(req(), future) == ["RATE_LIMIT"]


def test_rate_limit_for_reports() -> None:
    report = req("report", {"text_sv": "x"})
    five = tuple(NOW - i * 5000 for i in range(5))
    assert codes(report, ctx(recent_brain_reports_mono_ms=five)) == []
    six = tuple(NOW - i * 5000 for i in range(6))
    assert codes(report, ctx(recent_brain_reports_mono_ms=six)) == ["RATE_LIMIT"]
    quiet = spec("report", rate_class="none")
    assert codes(report, ctx(registry=registry_with(quiet), recent_brain_reports_mono_ms=six)) == []


def test_unresolved_targets_are_rejected() -> None:
    c = ctx(vehicle=vehicle(alt_amsl_m=99_999.0))  # a position the schema refuses
    assert "SCHEMA_INVALID" in codes(req(), c)
    assert "SCHEMA_INVALID" in codes(req("resume", {}, phase="onboard"), onboard_ctx())


def test_resume_route_is_checked_onboard_not_on_the_ground() -> None:
    assert codes(req("resume", {}), ctx()) == []


def test_a_failing_parameter_model_rejects_instead_of_crashing() -> None:
    from sd_schemas.skills import GateTargets, SkillParams, TargetContext

    class Broken(SkillParams):
        def gate_targets(self, ctx: TargetContext) -> GateTargets:
            raise RuntimeError("bug")

    c = ctx(registry=registry_with(spec("goto", params_model=Broken)))
    d = decide(req(params={}), c)
    assert d.decision == "REJECT"
    assert d.details["targets"] == "error"


def test_system_orders_meet_the_hard_rules() -> None:
    system = req(issuer_kind="system", issuer_ref="system:ground", params=goto(alt_m=200))
    assert "ALT_ABOVE_LIMIT" in codes(system, ctx())


def test_reason_codes_are_sorted_by_priority() -> None:
    c = ctx(sdlink_up=False, operator_present=False, ai_enabled=False)
    out = codes(req(params=goto(0, 310, alt_m=200, speed_ms=30)), c)
    assert out == sorted(out, key=REASON_PRIORITY.index)
    assert len(out) == len(set(out))
    assert {"AI_DISABLED", "LINK_DOWN", "OUTSIDE_CHARTER", "OPERATOR_ABSENT"} <= set(out)


# --- findings from review -------------------------------------------------------------------


def test_empty_planned_path_is_not_a_checked_route() -> None:
    d = decide(req("resume", {}, phase="onboard", planned_path=[]), onboard_ctx())
    assert d.decision == "REJECT"
    assert d.details["targets"] == "unresolved"


def test_planned_path_length_is_capped() -> None:
    long = [_lla(0, 50)] * 401
    assert "SCHEMA_INVALID" in codes(
        req("resume", {}, phase="onboard", planned_path=long), onboard_ctx()
    )


def test_the_card_being_approved_does_not_count_against_itself() -> None:
    c = ctx(brain_pending_approvals=1, recent_brain_actions_mono_ms=(NOW - 1000,))
    assert codes(req(), c) == ["PENDING_LIMIT", "RATE_LIMIT"]
    assert codes(req(phase="approve"), c) == []


def test_missing_speed_is_checked_at_the_default_speed() -> None:
    slow = ctx(charter=CHARTER.model_copy(update={"max_speed_ms": 4.0}))
    assert codes(req(), slow) == ["SPEED_ABOVE_CHARTER"]
    assert codes(req(params=goto(speed_ms=4)), slow) == []
    assert codes(req("hold", {}), slow) == []
    lat, lon = latlon(0, 0)
    assert codes(req("mark_location", {"lat": lat, "lon": lon, "label": "M1"}), slow) == []
    slow_ground = ground_ctx(charter=slow.charter)
    assert codes(req("takeoff", {"alt_m": 40}), slow_ground) == []


def test_order_for_another_mission_or_vehicle() -> None:
    from sd_schemas.base import new_id

    assert "MISSION_STATE_INVALID" in codes(req(mission_id=new_id()), ctx())
    assert "MISSION_STATE_INVALID" in codes(
        req(vehicle_id="sim-02", phase="onboard"), onboard_ctx()
    )


def test_a_parameter_validator_that_raises_anything_rejects() -> None:
    from pydantic import model_validator

    from sd_schemas.skills import GotoParams

    class Odd(GotoParams):
        @model_validator(mode="after")
        def _boom(self) -> Odd:
            raise TypeError("not a ValueError")

    d = decide(req(), ctx(registry=registry_with(spec("goto", params_model=Odd))))
    assert d.reason_codes == ["SCHEMA_INVALID"]


def test_a_rule_that_raises_rejects(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    from sd_safety.rules import Facts, Finding, Rule

    module = importlib.import_module("sd_safety.evaluate")

    def broken(_f: Facts) -> list[Finding]:
        raise KeyError("bug")

    monkeypatch.setattr(
        module, "RULES", (*module.RULES, Rule("R-TEST", frozenset({"brain"}), False, broken))
    )
    d = decide(req(), ctx())
    assert d.decision == "REJECT"
    assert d.details["internal_error"] == "R-TEST"


def test_details_do_not_overwrite_each_other() -> None:
    far = polygon((-50, -520), (50, -520), (50, 600), (-50, 600))
    wide = CHARTER.model_copy(update={"area": far})
    d = decide(req(params=goto(0, 590, alt_m=130)), ctx(charter=wide))
    assert {"ALT_ABOVE_LIMIT", "AREA_BEYOND_FC_FENCE", "PILOT_DISTANCE"} <= set(d.reason_codes)
    assert d.details["alt_limit_m"] == 120
    assert d.details["pilot_limit_m"] == 500
    assert d.details["fc_fence_limit_m"] == 515


def test_infinite_distances_are_not_written_as_numbers() -> None:
    antipode = vehicle(pilot_position=LatLon(lat=-CHARTER.home.lat, lon=CHARTER.home.lon - 179.67))
    lat, lon = latlon(0, 300)
    mark = req("mark_location", {"lat": lat, "lon": lon, "label": "M1"})
    out = decide(mark, ctx(charter=None, mission_state=MissionState.DRAFT, vehicle=antipode))
    assert out.details["pilot_distance_m"] is None


def test_worst_case_geometry_stays_fast() -> None:
    import time

    teeth: list[tuple[float, float]] = [(-245, -250), (245, -250)]
    for i in range(49):
        x = 240 - i * 10
        teeth += [(x, 200), (x - 5, 200), (x - 5, -100), (x - 10, -100)]
    comb = CHARTER.model_copy(update={"area": polygon(*teeth[:198]), "sectors": []})
    zig = [(-240 + i * 4.8, -50 if i % 2 else 150) for i in range(100)]
    zig += [(zig[-1][0], -90), (-240, -90)]
    here = vehicle(**dict(zip(("lat", "lon"), latlon(0, -200), strict=True)))
    start = time.perf_counter()
    d = decide(req("survey_sector", {"polygon": polygon(*zig)}), ctx(charter=comb, vehicle=here))
    assert time.perf_counter() - start < 2.0
    assert "OUTSIDE_CHARTER" in d.reason_codes
