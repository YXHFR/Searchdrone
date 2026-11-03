"""The closer look's gate rules (spec "Titta närmare" §10): who may start one, one at a
time, an open candidate, the height floor, the ceiling and overflight.

Every new code has a test that sets it off and one that does not. The vehicle hovers
40 m above home, 50 m north of it (``gate_fixtures.vehicle``); the candidate is placed
in metres east/north of home.
"""

from __future__ import annotations

from typing import Any

from gate_fixtures import CHARTER, HOME_AMSL, IMPLEMENTED, ctx, latlon, onboard_ctx, req

from sd_safety import evaluate
from sd_schemas.base import LatLonAlt, new_id
from sd_schemas.gate import GateContext, GateFloor, GateRequest
from sd_schemas.keepout import Keepout

ALLOWED = [*IMPLEMENTED, "inspect_detection", "capture_stills"]
LIVE = CHARTER.model_copy(update={"allowed_skills": ALLOWED, "operation_mode": "live"})
EXERCISE = LIVE.model_copy(update={"operation_mode": "exercise", "overflight_allowed": True})
PILOT: dict[str, Any] = {"issuer_kind": "operator", "issuer_ref": "operator:demo"}
CANDIDATE = new_id()


def look(east: float = 0.0, north: float = 200.0, **extra: Any) -> dict[str, Any]:
    lat, lon = latlon(east, north)
    base = {
        "candidate_id": str(CANDIDATE),
        "lat": lat,
        "lon": lon,
        "sigma95_m": 5.0,
        "target_radius_m": 35.0,
    }
    return {**base, **extra}


def codes(r: GateRequest, c: GateContext) -> list[str]:
    return evaluate(r, c).reason_codes


def pilot_look(params: dict[str, Any] | None = None, **update: Any) -> GateRequest:
    return req("inspect_detection", look() if params is None else params, **{**PILOT, **update})


def at(east: float, north: float, alt_rel_home_m: float) -> LatLonAlt:
    lat, lon = latlon(east, north)
    return LatLonAlt(lat=lat, lon=lon, alt_amsl_m=HOME_AMSL + alt_rel_home_m)


def floor(above_home_m: float, reason: str = "map") -> GateFloor:
    return GateFloor(floor_amsl_m=HOME_AMSL + above_home_m, reason=reason)  # type: ignore[arg-type]


# --- R-INSPECT-SENDER ---------------------------------------------------------------------------


def test_the_pilot_starts_a_look_and_the_ai_only_proposes_one() -> None:
    c = ctx(charter=EXERCISE)
    assert codes(pilot_look(), c) == []
    assert codes(req("inspect_detection", look()), c) == ["NEEDS_APPROVAL_SKILL"]


def test_no_rule_and_no_deputy_starts_a_look() -> None:
    c = ctx(charter=EXERCISE)
    for update in (
        {"issuer_kind": "system", "issuer_ref": "rule:ground"},
        {"issuer_kind": "brain", "issuer_ref": "brain:deputy"},
        {"issuer_kind": "operator", "issuer_ref": "system:ground"},
    ):
        assert "INSPECT_NEEDS_PILOT" in codes(req("inspect_detection", look(), **update), c)
    on = onboard_ctx(charter=EXERCISE)
    deputy = req("inspect_detection", look(), issuer_ref="brain:deputy")
    assert "INSPECT_NEEDS_PILOT" in codes(deputy, on)
    # Other skills are not this rule's.
    assert "INSPECT_NEEDS_PILOT" not in codes(req("hold", {}, issuer_kind="system"), c)


# --- R-INSPECT-ONE ------------------------------------------------------------------------------


def test_one_look_at_a_time() -> None:
    busy = new_id()
    c = ctx(charter=EXERCISE, inspect_active=busy)
    out = evaluate(pilot_look(), c)
    assert out.reason_codes == ["INSPECT_BUSY"]
    assert out.details["inspect_command_id"] == str(busy)
    # Onboard the look's own order is the active one.
    own = pilot_look()
    on = onboard_ctx(charter=EXERCISE, inspect_active=own.command_id)
    assert "INSPECT_BUSY" not in codes(own, on)
    # Other orders are not refused for a look.
    assert codes(req(), ctx(inspect_active=busy)) == []


# --- R-INSPECT-TARGET ---------------------------------------------------------------------------


def test_the_candidate_must_be_open() -> None:
    c = ctx(charter=EXERCISE, inspectable_candidates=(CANDIDATE,))
    assert codes(pilot_look(), c) == []
    closed = ctx(charter=EXERCISE, inspectable_candidates=(new_id(),))
    out = evaluate(pilot_look(), closed)
    assert out.reason_codes == ["INSPECT_TARGET_CLOSED"]
    assert out.details["candidate_id"] == str(CANDIDATE)
    # A number the ground could not translate is closed too (and malformed).
    k99 = codes(pilot_look({"candidate_id": "K99"}), closed)
    assert {"INSPECT_TARGET_CLOSED", "SCHEMA_INVALID"} <= set(k99)
    # Upper-case ids are the same candidate.
    upper = look(candidate_id=str(CANDIDATE).upper())
    assert codes(pilot_look(upper), c) == []


def test_the_open_candidates_are_the_grounds() -> None:
    """Onboard the candidates are unknown: the rule is the ground's alone."""
    on = onboard_ctx(charter=EXERCISE, inspectable_candidates=(new_id(),))
    assert "INSPECT_TARGET_CLOSED" not in codes(pilot_look(), on)
    other = ctx(inspectable_candidates=(new_id(),))
    assert codes(req(), other) == []


# --- R-FLOOR and R-POWERLINE --------------------------------------------------------------------


def test_the_side_look_stays_above_the_floor() -> None:
    """In live the order has the side look's height: the floor must lie under it."""
    target = look()
    side = evaluate(pilot_look(target), ctx(charter=LIVE))
    assert side.decision == "ALLOW"
    assert codes(pilot_look(target), ctx(charter=LIVE, floor=floor(30.0))) == []
    out = evaluate(pilot_look(target), ctx(charter=LIVE, floor=floor(110.0)))
    assert out.reason_codes == ["FLOOR_VIOLATION"]
    assert out.details["floor_reason"] == "map"
    assert out.details["floor_amsl_m"] == HOME_AMSL + 110.0


def test_near_a_power_line_the_code_says_so() -> None:
    out = evaluate(pilot_look(), ctx(charter=LIVE, floor=floor(110.0, "powerline")))
    assert out.reason_codes == ["POWERLINE_ZONE"]


def test_onboard_every_step_of_the_path_is_checked() -> None:
    """The handler's path: its first point is where the drone is (climbing out from under
    the floor is never refused); every later point must be on or above the floor."""
    on = onboard_ctx(charter=EXERCISE, floor=floor(25.0), inspect_active=None)
    up = [at(0, 50, 20.0), at(0, 50, 26.0)]
    assert "FLOOR_VIOLATION" not in codes(pilot_look(planned_path=up), on)
    down = [at(0, 50, 40.0), at(0, 200, 30.0), at(0, 200, 24.0)]
    assert "FLOOR_VIOLATION" in codes(pilot_look(planned_path=down), on)
    exact = [at(0, 50, 25.0)]
    assert "FLOOR_VIOLATION" not in codes(pilot_look(planned_path=exact), on)
    below = [at(0, 50, 24.0)]
    assert "FLOOR_VIOLATION" in codes(pilot_look(planned_path=below), on)


def test_the_floor_is_not_checked_without_heights_or_a_move() -> None:
    """Over the candidate (sim, exercise) the order has no height: the drone keeps its own
    above the floor. A report does not move. Without a charter there are no targets."""
    c = ctx(charter=EXERCISE, floor=floor(200.0))
    assert "FLOOR_VIOLATION" not in codes(pilot_look(), c)
    report = req("report", {"text_sv": "hej"}, **PILOT)
    assert codes(report, ctx(floor=floor(500.0))) == []
    nothing = codes(pilot_look(), ctx(charter=None, floor=floor(500.0)))
    assert "FLOOR_VIOLATION" not in nothing
    assert "FLOOR_ABOVE_CEILING" not in nothing


def test_below_the_charters_band_is_the_floors_business() -> None:
    """A look may come down under the charter's lowest height (R-ALT leaves it to R-FLOOR);
    any other order may not."""
    on = onboard_ctx(charter=EXERCISE, floor=floor(5.0))
    low = [at(0, 50, 40.0), at(0, 200, 8.0)]
    assert "ALT_BELOW_CHARTER" not in codes(pilot_look(planned_path=low), on)
    assert "ALT_BELOW_CHARTER" in codes(req("goto", planned_path=low, **PILOT), on)


# --- R-FLOOR-CEIL -------------------------------------------------------------------------------


def test_a_floor_above_the_ceiling_refuses_the_look() -> None:
    top = min(EXERCISE.alt_max_agl_m, ctx().limits.max_alt_agl_m)
    under = ctx(charter=EXERCISE, floor=floor(top))
    assert codes(pilot_look(), under) == []
    over = evaluate(pilot_look(), ctx(charter=EXERCISE, floor=floor(top + 0.5)))
    assert over.reason_codes == ["FLOOR_ABOVE_CEILING"]
    assert over.details["ceiling_amsl_m"] == HOME_AMSL + top
    # The mission's limit is a ceiling too.
    lower_limit = ctx().limits.model_copy(update={"max_alt_agl_m": 60.0})
    capped = ctx(charter=EXERCISE, limits=lower_limit, floor=floor(61.0))
    assert "FLOOR_ABOVE_CEILING" in codes(pilot_look(), capped)
    # Only a move is checked.
    report = req("report", {"text_sv": "hej"}, **PILOT)
    assert codes(report, ctx(floor=floor(top + 50))) == []


# --- R-OVERFLIGHT and the suspended cylinder ----------------------------------------------------


def own_cylinder() -> Keepout:
    lat, lon = latlon(0, 200)
    return Keepout(
        keepout_id=new_id(),
        lat=lat,
        lon=lon,
        radius_m=35.0,
        source="candidate",
        source_id=CANDIDATE,
    )


def test_overflight_in_exercise_flies_over_its_own_suspended_cylinder() -> None:
    k = own_cylinder()
    blocked = ctx(charter=EXERCISE, keepouts=(k,))
    assert codes(pilot_look(), blocked) == ["KEEPOUT_VIOLATION"]
    suspended = ctx(charter=EXERCISE, keepouts=(k,), suspended_keepout_ids=(k.keepout_id,))
    assert codes(pilot_look(), suspended) == []


def test_live_without_the_permit_never_comes_nearer_than_the_cylinder() -> None:
    """The side look keeps out by itself; a path straight over (onboard) is refused."""
    on = onboard_ctx(charter=LIVE)
    side = [at(0, 50, 40.0), at(0, 160, 80.0)]
    assert "OVERFLIGHT_NOT_ALLOWED" not in codes(pilot_look(planned_path=side), on)
    over = [at(0, 50, 40.0), at(0, 200, 40.0)]
    out = evaluate(pilot_look(planned_path=over), on)
    assert "OVERFLIGHT_NOT_ALLOWED" in out.reason_codes
    assert out.details["candidate_id"] == str(CANDIDATE)
    # With the permit (overflight_allowed) the same path may go over.
    permit = onboard_ctx(charter=LIVE.model_copy(update={"overflight_allowed": True}))
    assert "OVERFLIGHT_NOT_ALLOWED" not in codes(pilot_look(planned_path=over), permit)


def test_overflight_needs_a_known_target() -> None:
    on = onboard_ctx(charter=LIVE)
    over = [at(0, 50, 40.0), at(0, 200, 40.0)]
    for params in (
        look(lat=None, lon=None),
        look(target_radius_m=None),
        {"candidate_id": "not-a-uuid"},
    ):
        found = codes(pilot_look(params, planned_path=over), on)
        assert "OVERFLIGHT_NOT_ALLOWED" not in found
    without_charter = codes(pilot_look(planned_path=over), onboard_ctx(charter=None))
    assert "OVERFLIGHT_NOT_ALLOWED" not in without_charter
    assert "OVERFLIGHT_NOT_ALLOWED" not in codes(req("goto", planned_path=over, **PILOT), on)
