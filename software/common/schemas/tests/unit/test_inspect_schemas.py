"""The closer look v2 (``inspect_detection``): its order, charter fields and messages."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from sd_schemas import inspect, link
from sd_schemas.charter import MissionCharter
from sd_schemas.executor import ExecutorState
from sd_schemas.geo import local_xy, plane_offset
from sd_schemas.keepout import INSPECT_STANDOFF_MARGIN_M
from sd_schemas.skills import SKILL_REGISTRY, InspectDetectionParams, TargetContext
from sd_schemas_examples import OTHER, _charter  # pyright: ignore[reportPrivateUsage]

SHA = "0" * 64


def test_the_order_names_a_candidate_and_no_height() -> None:
    order = InspectDetectionParams.model_validate({"candidate_id": str(OTHER)})
    assert order.lat is None and order.sigma95_m is None and order.target_radius_m is None
    for old in ({"detection_id": str(OTHER)}, {"alt_m": 30}, {"standoff_m": 20}):
        with pytest.raises(ValidationError):
            InspectDetectionParams.model_validate({"candidate_id": str(OTHER), **old})
    spec = SKILL_REGISTRY.get("inspect_detection")
    assert spec.requires_approval_in_b and spec.ttl_s == 15
    assert spec.allowed_from == {ExecutorState.HOLD, ExecutorState.TRANSIT, ExecutorState.SURVEY}


def test_the_side_look_keeps_off_the_cylinder_within_the_ceiling() -> None:
    charter = _charter()
    home = charter.home
    lat, lon = plane_offset(home.lat, home.lon, 0.0, 150.0)
    order = InspectDetectionParams(
        candidate_id=OTHER, lat=lat, lon=lon, sigma95_m=2.0, target_radius_m=32.0
    )
    ctx = TargetContext(charter=charter, position=home, home=home)
    found = order.geometry(ctx)
    assert found is not None
    point, alt, _, _ = found
    assert alt <= charter.alt_max_agl_m
    off = math.hypot(*local_xy(lat, lon, point.lat, point.lon))
    assert off == pytest.approx(32.0 + INSPECT_STANDOFF_MARGIN_M, abs=0.1)
    assert order.model_copy(update={"lat": None}).geometry(ctx) is None


def test_the_charter_carries_the_mode_and_the_floor_pack() -> None:
    plain = _charter()
    assert plain.operation_mode is None and not plain.overflight_allowed
    base = plain.model_dump()
    pack = {"floor_pack_id": "example-area", "floor_pack_sha256": SHA}
    good = MissionCharter.model_validate(
        base | {"operation_mode": "exercise", "overflight_allowed": True} | pack
    )
    assert good.operation_mode == "exercise" and good.floor_pack_id == "example-area"
    for bad in (
        {"operation_mode": "training"},
        {"overflight_allowed": True},  # never without a mode
        {"floor_pack_id": "example-area"},  # never without its hash
        {"floor_pack_sha256": SHA},
        {"floor_pack_id": "../etc", "floor_pack_sha256": SHA},
        {"floor_pack_id": "x", "floor_pack_sha256": "ABC"},
    ):
        with pytest.raises(ValidationError):
            MissionCharter.model_validate(base | bad)


def test_the_ground_answers_and_the_drone_reports() -> None:
    assert link.InspectOutcome(command_id=OTHER, outcome="confirmed").outcome == "confirmed"
    with pytest.raises(ValidationError):
        link.InspectOutcome.model_validate({"command_id": str(OTHER), "outcome": "maybe"})
    assert {"InspectOutcome", "InspectLower", "InspectStay"} <= link.RELIABLE_KINDS
    for kind in (inspect.EVENT_DONE, inspect.EVENT_LOWER_OFFER, "executor.inspect_done"):
        assert link.OnboardEvent(type=kind, severity="info").type == kind
    with pytest.raises(ValidationError):  # the ground's own event, never the drone's
        link.OnboardEvent(type=inspect.EVENT_PERMIT_CHANGED, severity="info")
    progress = inspect.InspectProgress(command_id=OTHER, candidate_id=OTHER, step=0, phase="await")
    assert progress.await_left_s is None and not progress.laser_confirmed


def test_the_match_radius_grows_with_the_uncertainty() -> None:
    assert inspect.match_radius_m(2.0) == inspect.MATCH_RADIUS_MIN_M
    assert inspect.match_radius_m(12.0) == 24.0


def test_with_overflight_the_order_goes_straight_over_without_a_height() -> None:
    charter = _charter().model_copy(
        update={"operation_mode": "exercise", "overflight_allowed": True}
    )
    home = charter.home
    lat, lon = plane_offset(home.lat, home.lon, 0.0, 150.0)
    order = InspectDetectionParams(
        candidate_id=OTHER, lat=lat, lon=lon, sigma95_m=2.0, target_radius_m=32.0
    )
    ctx = TargetContext(charter=charter, position=home, home=home)
    targets = order.gate_targets(ctx)
    assert targets.resolved and targets.alts_m == []
    assert targets.path[-1].lat == lat and targets.path[-1].lon == lon
    assert targets.speed_ms == min(inspect.INSPECT_SPEED_MS, charter.max_speed_ms)
    assert not order.model_copy(update={"lat": None}).gate_targets(ctx).resolved
    nowhere = TargetContext(charter=charter, position=None, home=home)
    assert not order.gate_targets(nowhere).resolved


def test_the_ai_names_the_card_by_its_number() -> None:
    spec = SKILL_REGISTRY.get("inspect_detection")
    model = spec.ai_params_model
    assert model is not None
    assert model.model_validate({"candidate_id": "K3"}).model_dump() == {"candidate_id": "K3"}
    for bad in ("K0", "3", "k3", str(OTHER), "K123456"):
        with pytest.raises(ValidationError):
            model.model_validate({"candidate_id": bad})


def test_the_permit_needs_its_reference() -> None:
    assert not inspect.Permit().allow_overflight_live
    inspect.Permit(specific_category_ref="SE-123", allow_overflight_live=True)
    with pytest.raises(ValidationError):
        inspect.Permit(specific_category_ref=" ", allow_overflight_live=True)


def test_the_looks_close_and_the_search_flown_on_after_it() -> None:
    """``inspect.closed``: how a look ended onboard and what the drone does next;
    ``inspect.resume_ended``: the end of the search it flew on after the look. Both stay
    out of the package's exports (and the generated schemas)."""
    from sd_schemas.base import new_id

    look, search = new_id(), new_id()
    c = inspect.InspectClosed(
        command_id=look,
        candidate_id=OTHER,
        cause="returned",
        next="resume",
        breakpoint="survey",
        resume_command_id=new_id(),
        resumed_command_id=search,
    )
    assert inspect.InspectClosed.model_validate(c.model_dump(mode="json")) == c
    assert inspect.EVENT_CLOSED == "inspect.closed"
    assert link.OnboardEvent(type=inspect.EVENT_CLOSED, severity="info").type == "inspect.closed"
    with pytest.raises(ValidationError):
        inspect.InspectClosed(command_id=look, candidate_id=OTHER, cause="returned", next="land")  # type: ignore[arg-type]
    e = inspect.InspectResumeEnded(command_id=look, resumed_command_id=search, status="done")
    assert e.reason_codes == []
    assert inspect.EVENT_RESUME_ENDED == "inspect.resume_ended"
    with pytest.raises(ValidationError):
        inspect.InspectResumeEnded(command_id=look, status="running")  # type: ignore[arg-type]
    import sd_schemas

    assert not {"InspectClosed", "InspectResumeEnded"} & set(sd_schemas.__all__)
