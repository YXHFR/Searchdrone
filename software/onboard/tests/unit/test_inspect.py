"""B2-09 onboard: orbit, capture_stills, inspect_detection and the keep-out reflex."""

from __future__ import annotations

import asyncio
import math
from itertools import pairwise
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from onboard_helpers import ONBOARD_SKILLS, PLANE, Rig, at, charter, rig, square

from sd_mav.mission import MISSION
from sd_mav.modes import HOME_MODES, CopterMode
from sd_mav.vehicle import SendOrigin
from sd_onboard.executor import executor as executor_module
from sd_onboard.executor.keepout_reflex import (
    KEEPOUT_CLIMB_MARGIN_M,
    LOCAL_KEEPOUTS_MAX,
    climb_alt,
    entry_m,
    leg_entry_m,
    merge_local,
    stop_m,
    way_out,
)
from sd_onboard.skills import inspect_detection, survey_sector
from sd_onboard.skills.base import PlanError
from sd_onboard.skills.goto import GotoHandler, goto_plan
from sd_onboard.skills.inspect_detection import EXITS_MAX
from sd_safety import evaluate
from sd_safety.keepout import circles, xy_route_violation
from sd_safety.targets import polygon_from_geojson
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.gate import GateContext, GateDecision, GateRequest
from sd_schemas.keepout import INSPECT_AIM_DEG, INSPECT_VIEW_DEG, Keepout
from sd_schemas.link import BBox, GeoDetectionMsg, KeepoutInstall, SkillCommand

SKILLS = [*ONBOARD_SKILLS, "orbit", "capture_stills", "inspect_detection"]


def keepout(east: float, north: float, r: float, source: str = "confirmed") -> Keepout:
    lat, lon = at(east, north)
    return Keepout(
        keepout_id=new_id(),
        lat=lat,
        lon=lon,
        radius_m=r,
        source=source,  # type: ignore[arg-type]
        source_id=new_id(),
    )


def install(r: Rig, *keepouts: Keepout) -> None:
    seq = r.ex.keepout_seq + 1
    r.ex.on_keepouts(KeepoutInstall(install_seq=seq, keepouts=list(keepouts)), new_id())


class Camera:
    """``Executor.camera``: one still id per call, the order's id kept."""

    def __init__(self, works: bool = True) -> None:
        self.works = works
        self.asked: list[tuple[UUID, float]] = []
        self.rig: Rig | None = None

    async def __call__(self, command_id: UUID) -> UUID | None:
        assert self.rig is not None
        self.asked.append((command_id, self.rig.clock.monotonic()))
        return new_id() if self.works else None


def camera(r: Rig, works: bool = True) -> Camera:
    cam = Camera(works)
    cam.rig = r
    r.ex.camera = cam
    return cam


def xy(r: Rig) -> tuple[float, float]:
    v = r.ex.vehicle_state()
    return PLANE.xy(v.lat, v.lon)


def track_xy(r: Rig, start: int) -> list[tuple[float, float]]:
    return [PLANE.xy(la, lo) for la, lo, _ in r.fc.track[start:]]


# --- capture_stills ---------------------------------------------------------------------


async def test_capture_stills_in_hold() -> None:
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        reason = r.ex.hold_reason
        cam = camera(r)
        cmd = await r.send(r.order("capture_stills", {"count": 3, "interval_s": 1.0}), 0.1)
        assert r.ex.state == S.HOLD and r.ex.hold_reason == reason
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 10)
        assert [c for c, _ in cam.asked] == [cmd.command_id] * 3
        gaps = [b - a for (_, a), (_, b) in pairwise(cam.asked)]
        assert all(abs(g - 1.0) < 0.1 for g in gaps)
        assert r.ex.state == S.HOLD and r.ex.hold_reason == reason
        assert "executor.stills_captured" in r.up.kinds()


async def test_capture_stills_without_a_camera_fails_and_stays() -> None:
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        cmd = await r.send(r.order("capture_stills", {"count": 2, "interval_s": 0.5}), 2)
        assert r.up.codes(cmd.command_id) == ("ONB_CAMERA_UNAVAILABLE",)
        assert r.ex.state == S.HOLD
        camera(r, works=False)
        again = await r.send(r.order("capture_stills", {}), 1)
        assert r.up.codes(again.command_id) == ("ONB_CAMERA_UNAVAILABLE",)


async def test_a_camera_that_hangs_does_not_hold_the_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor_module, "_CAPTURE_LIMIT_S", 0.05)

    async def stuck(_command_id: UUID) -> UUID | None:
        await asyncio.Event().wait()
        return None

    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        r.ex.camera = stuck
        cmd = await r.send(r.order("capture_stills", {"count": 1}), 0)
        await r.until(lambda: "failed" in r.up.stages(cmd.command_id), 60)
        assert "executor.capture_timeout" in r.up.kinds()
        assert r.up.codes(cmd.command_id) == ("ONB_CAMERA_UNAVAILABLE",)
        assert r.ex.state == S.HOLD


async def test_capture_stills_only_while_hovering() -> None:
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        lat, lon = at(150, 0)
        await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}), 0.5)
        assert r.ex.state == S.TRANSIT
        cmd = await r.send(r.order("capture_stills", {}))
        assert r.up.stages(cmd.command_id) == ["rejected"]
        assert r.ex.state == S.TRANSIT


# --- inspect_detection ------------------------------------------------------------------


def inspect(east: float, north: float, radius: float, **kw: Any) -> dict[str, Any]:
    lat, lon = at(east, north)
    params: dict[str, Any] = {
        "detection_id": str(new_id()),
        "standoff_m": 30,
        "lat": lat,
        "lon": lon,
        "target_radius_m": radius,
    }
    params.update(kw)
    return params


async def test_inspect_flies_to_the_edge_looks_down_and_takes_stills() -> None:
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        install(r, keepout(150, 0, 35))
        cam = camera(r)
        start = len(r.fc.track)
        cmd = await r.send(r.order("inspect_detection", inspect(150, 0, 35)), 0.5)
        assert r.ex.state == S.TRANSIT, r.up.acks[-2:]
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 120)
        assert r.ex.state == S.HOLD
        # On the drone's side (west), 35 + 10 m from the target, within 25 degrees.
        east, north = xy(r)
        assert math.hypot(east - 105, north) < 3
        alt = r.ex.vehicle_state().alt_rel_home_m
        view = math.degrees(math.atan2(math.hypot(east - 150, north), alt))
        assert INSPECT_AIM_DEG - 1.5 < view <= INSPECT_VIEW_DEG
        assert r.fc.target_yaw_deg is not None and abs(r.fc.target_yaw_deg) < 1
        assert [c for c, _ in cam.asked] == [cmd.command_id] * 5
        cyl = circles([keepout(150, 0, 35)], PLANE)
        assert xy_route_violation(track_xy(r, start), cyl) is None
        assert "executor.inspect_done" in r.up.kinds()


async def test_inspect_without_a_target_or_a_view_is_refused() -> None:
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS, alt_max_agl_m=60))
        await r.takeoff()
        no_target = inspect(150, 0, 35)
        del no_target["lat"]
        cmd = await r.send(r.order("inspect_detection", no_target))
        assert r.up.codes(cmd.command_id) == ("SCHEMA_INVALID",)
        # 100 m of cylinder at most 60 m up: never within 28 degrees of straight down.
        far = await r.send(r.order("inspect_detection", inspect(150, 0, 100)))
        assert r.up.codes(far.command_id) == ("ONB_INSPECT_GEOMETRY",)
        assert r.ex.state == S.HOLD


async def test_inspect_without_stills_fails_into_hold() -> None:
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        camera(r, works=False)
        cmd = await r.send(r.order("inspect_detection", inspect(80, 0, 31)), 0.5)
        await r.until(lambda: "failed" in r.up.stages(cmd.command_id), 120)
        assert r.up.codes(cmd.command_id) == ("ONB_CAMERA_UNAVAILABLE",)
        assert r.ex.state == S.HOLD


def _dists(track: list[tuple[float, float]], centre: tuple[float, float]) -> list[float]:
    return [math.dist(p, centre) for p in track]


async def test_inspect_inside_new_cylinders_leaves_them_first_then_plans_from_there() -> None:
    """The order comes with the drone inside two cylinders (the target's and an own
    detection's): it leaves both on a straight way out that goes deeper into neither, stops
    past them, and only then works out the point from where it stopped. The track never
    goes back into either."""
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        target = keepout(20, 0, 35)
        install(r, target)
        r.ex.on_detections([detection(-25, 0, 0.9, 3.0)])
        cyl = circles(r.ex.keepouts_in_force(), PLANE)
        assert len(cyl) == 2 and all(math.dist(xy(r), c) < rad for c, rad in cyl)
        cam = camera(r)
        start = len(r.fc.track)
        cmd = await r.send(r.order("inspect_detection", inspect(20, 0, 35)), 0.5)
        assert r.ex.state == S.TRANSIT, r.up.acks[-2:]
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 240)
        assert len(cam.asked) == 5 and "executor.keepout_hold" not in r.up.kinds()
        track = track_xy(r, start)
        inside = [any(math.dist(p, c) < rad for c, rad in cyl) for p in track]
        out = inside.index(False)
        assert not any(inside[out:])
        for c, _ in cyl:  # never deeper on the way out
            d = _dists(track[: out + 1], c)
            assert all(b >= a - 1e-6 for a, b in pairwise(d))
        east, north = xy(r)
        view = math.degrees(
            math.atan2(math.hypot(east - 20, north), r.ex.vehicle_state().alt_rel_home_m)
        )
        assert view <= INSPECT_VIEW_DEG


async def _inspect_with_a_cylinder_moved_onto_the_leg(
    r: Rig,
) -> tuple[Any, Keepout, int]:
    """A target 150 m east, the point 45 m short of it; a candidate's cylinder off to the
    north that the ground moves onto the leg (the same id) once the drone is on its way."""
    await r.install(charter(allowed_skills=SKILLS))
    await r.takeoff()
    target = keepout(150, 0, 35)
    off = keepout(60, 80, 15, source="candidate")
    install(r, target, off)
    camera(r)
    cmd = await r.send(r.order("inspect_detection", inspect(150, 0, 35)), 0.5)
    assert r.ex.state == S.TRANSIT, r.up.acks[-2:]
    await r.until(lambda: xy(r)[0] >= 15.0, 60)
    moved = off.model_copy(update=dict(zip(("lat", "lon"), at(60, -12), strict=True)))
    install(r, target, moved)
    return cmd, moved, len(r.fc.track)


async def test_a_cylinder_moved_onto_an_inspection_leg_plans_the_leg_again() -> None:
    async with rig() as r:
        cmd, moved, start = await _inspect_with_a_cylinder_moved_onto_the_leg(r)
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 240)
        assert "executor.keepout_replan" in r.up.kinds(), r.up.kinds()
        assert "executor.keepout_hold" not in r.up.kinds()
        assert xy_route_violation(track_xy(r, start), circles([moved], PLANE)) is None


async def test_a_cylinder_moved_onto_an_inspection_leg_with_no_way_hovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No new leg passes the gate (here as if it ate the return reserve): the order stops
    in a hover short of the cylinder (``KEEPOUT_VIOLATION``)."""
    async with rig() as r:

        def refused(req: GateRequest, ctx: GateContext) -> GateDecision:
            return evaluate(req, ctx).model_copy(
                update={"decision": "REJECT", "reason_codes": ["ROUTE_BATTERY"]}
            )

        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        target = keepout(150, 0, 35)
        off = keepout(60, 80, 15, source="candidate")
        install(r, target, off)
        camera(r)
        cmd = await r.send(r.order("inspect_detection", inspect(150, 0, 35)), 0.5)
        await r.until(lambda: xy(r)[0] >= 15.0, 60)
        monkeypatch.setattr(executor_module, "evaluate", refused)
        moved = off.model_copy(update=dict(zip(("lat", "lon"), at(60, -12), strict=True)))
        install(r, target, moved)
        start = len(r.fc.track)
        await r.until(lambda: r.ex.state == S.HOLD, 60)
        assert r.ex.hold_reason == "limit"
        assert "executor.keepout_replan_refused" in r.up.kinds(), r.up.kinds()
        assert ("KEEPOUT_VIOLATION",) in [c for ref, _, c in r.up.acks if ref == cmd.command_id]
        await r.run(10)
        assert xy_route_violation(track_xy(r, start), circles([moved], PLANE)) is None


async def test_a_cylinder_that_shrinks_under_an_inspection_changes_nothing() -> None:
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        target = keepout(150, 0, 35)
        near = keepout(60, -25, 20, source="candidate")
        install(r, target, near)
        camera(r)
        cmd = await r.send(r.order("inspect_detection", inspect(150, 0, 35)), 0.5)
        await r.until(lambda: xy(r)[0] >= 15.0, 60)
        install(r, target, near.model_copy(update={"radius_m": 12.0}))
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 240)
        kinds = r.up.kinds()
        assert "executor.keepout_replan" not in kinds and "executor.keepout_hold" not in kinds


async def test_cylinders_that_keep_coming_under_an_inspection_stop_it() -> None:
    """Each time the drone comes out, a new cylinder comes around it: after ``EXITS_MAX``
    ways out the order stops in a hover rather than chase its way out for ever."""
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        install(r, keepout(150, 0, 35))
        camera(r)
        r.ex.on_detections([detection(*xy(r), 0.9, 3.0)])
        cmd = await r.send(r.order("inspect_detection", inspect(150, 0, 35)), 0.5)
        assert r.ex.state == S.TRANSIT, r.up.acks[-2:]

        def ended() -> bool:
            return r.up.stages(cmd.command_id)[-1] in ("failed", "superseded")

        def out() -> bool:
            cyl = circles(r.ex.keepouts_in_force(), PLANE)
            return all(math.dist(xy(r), c) >= rad for c, rad in cyl)

        made = 0
        while not ended():
            await r.until(lambda: ended() or out(), 120)
            if not ended():
                r.ex.on_detections([detection(*xy(r), 0.9, 3.0)])
                made += 1
        assert made == EXITS_MAX
        assert r.ex.state == S.HOLD
        codes = [c for ref, _, c in r.up.acks if ref == cmd.command_id][-1]
        assert codes in (("ONB_INSPECT_GEOMETRY",), ("KEEPOUT_VIOLATION",))


# --- no way out over a person (B1), and the caps on planning again (B2) -----------------


def test_climb_alt_is_the_top_of_the_cylinders_plus_the_margin_within_the_charter() -> None:
    cyl = [((0.0, 0.0), 33.0), ((10.0, 0.0), 50.0), ((300.0, 0.0), 90.0)]
    top = 50.0 + KEEPOUT_CLIMB_MARGIN_M  # only the ones it is inside count
    assert climb_alt((5.0, 0.0), cyl, 40.0, 115.0) == top
    assert climb_alt((5.0, 0.0), cyl, 40.0, 50.0) == 50.0  # as high as the charter lets it
    assert climb_alt((5.0, 0.0), cyl, 70.0, 115.0) == 70.0  # above already: stays there
    assert climb_alt((5.0, 0.0), cyl, 40.0, 40.0) is None  # no higher allowed: no climb
    assert climb_alt((150.0, 0.0), cyl, 40.0, 115.0) is None  # inside none


def _refused(req: GateRequest, ctx: GateContext) -> GateDecision:
    return evaluate(req, ctx).model_copy(
        update={"decision": "REJECT", "reason_codes": ["ROUTE_BATTERY"]}
    )


async def _over_a_person_with_no_way_out(
    r: Rig, monkeypatch: pytest.MonkeyPatch
) -> tuple[SkillCommand, Keepout, tuple[float, float], int]:
    """On an inspection's leg a cylinder comes around the drone, its leg heading deeper,
    and the gate refuses every way out: the order, the cylinder, where the drone was and
    where the track was then."""
    await r.install(charter(allowed_skills=SKILLS))
    await r.takeoff()
    target = keepout(150, 0, 10)  # a small one: the inspection flies low
    install(r, target)
    camera(r)
    cmd = await r.send(r.order("inspect_detection", inspect(150, 0, 10)), 0.5)
    assert r.ex.state == S.TRANSIT, r.up.acks[-2:]
    await r.until(lambda: xy(r)[0] >= 15.0, 60)
    monkeypatch.setattr(executor_module, "evaluate", _refused)
    here = xy(r)
    # A wide one, so that its top and the margin lie above the drone and under the charter's.
    person = keepout(here[0] + 10.0, here[1], 90.0, source="candidate")
    assert r.ex.vehicle_state().alt_rel_home_m < person.radius_m + KEEPOUT_CLIMB_MARGIN_M - 1.5
    start = len(r.fc.track)
    install(r, target, person)
    return cmd, person, here, start


def _holds(r: Rig) -> list[dict[str, Any]]:
    return [d for t, _, d in r.up.events if t == "executor.keepout_hold"]


async def test_no_way_out_over_a_person_climbs_straight_up_then_hovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default for B1: straight up to the cylinder's top and the margin, inside the
    charter's height; there the order ends in the keep-out hover. The operator is told
    once, as the climb begins (``inside``, ``climb_to_m``)."""
    async with rig() as r:
        cmd, person, here, start = await _over_a_person_with_no_way_out(r, monkeypatch)
        await r.until(lambda: r.ex.state == S.HOLD, 120)
        want = person.radius_m + KEEPOUT_CLIMB_MARGIN_M
        holds = _holds(r)
        assert len(holds) == 1, holds
        assert holds[0]["inside"] is True and holds[0]["climb_to_m"] == pytest.approx(want)
        assert "executor.keepout_replan" not in r.up.kinds()
        assert r.up.codes(cmd.command_id) == ("KEEPOUT_VIOLATION",)
        assert r.ex.hold_reason == "limit" and r.fc.mode == CopterMode.LOITER
        assert abs(r.ex.vehicle_state().alt_rel_home_m - want) <= 1.5
        c = PLANE.xy(person.lat, person.lon)
        d0 = math.dist(here, c)
        track = track_xy(r, start)
        # Straight up: no deeper than braking from the leg's speed, and no farther off.
        assert min(_dists(track, c)) >= d0 - 8.0, (d0, min(_dists(track, c)))
        assert math.dist(track[-1], here) <= 10.0
        await r.run(10)
        assert len(_holds(r)) == 1  # hovering there: told once
        assert abs(r.ex.vehicle_state().alt_rel_home_m - want) <= 1.5


async def test_no_way_out_and_no_room_to_climb_hovers_where_it_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No higher allowed (here the charter's ceiling less the altitude limit's margin is
    below the drone): it hovers where it is, and the operator is told it cannot climb."""
    monkeypatch.setattr(executor_module, "_ALT_GOTO_BELOW_M", 200.0)
    async with rig() as r:
        cmd, _, _, _ = await _over_a_person_with_no_way_out(r, monkeypatch)
        alt = r.ex.vehicle_state().alt_rel_home_m
        await r.until(lambda: r.ex.state == S.HOLD, 60)
        holds = _holds(r)
        assert len(holds) == 1 and holds[0]["inside"] is True, holds
        assert "climb_to_m" not in holds[0]
        assert ("KEEPOUT_VIOLATION",) in [c for ref, _, c in r.up.acks if ref == cmd.command_id]
        assert r.ex.hold_reason == "limit"
        await r.run(5)
        assert r.ex.vehicle_state().alt_rel_home_m <= alt + 1.5


async def test_stop_during_the_climb_brakes_and_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    """STOP goes past the AI and the reflex as always, also while it climbs."""
    async with rig() as r:
        await _over_a_person_with_no_way_out(r, monkeypatch)
        await r.until(lambda: len(_holds(r)) == 1, 30)
        r.fc.set_mode_external(CopterMode.BRAKE)  # the ground's STOP, as MAVLink first
        await r.run(0.3)
        r.ex.on_override(r.notice("stop"))
        assert r.ex.state == S.HELD and r.ex.held_by == "operator_stop"
        alt = r.ex.vehicle_state().alt_rel_home_m
        await r.run(5)
        assert r.fc.mode == CopterMode.BRAKE  # the executor no longer steers
        assert r.ex.vehicle_state().alt_rel_home_m <= alt + 1.0


async def test_the_climb_stops_at_the_charter_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cylinder's top lies above what the charter allows: the drone climbs only to the
    charter's height less the altitude limit's margin, says so, and hovers there."""
    async with rig() as r:
        await _over_a_person_with_no_way_out(r, monkeypatch)
        alt = r.ex.vehicle_state().alt_rel_home_m
        cap = alt + 8.0
        assert cap < 90.0 + KEEPOUT_CLIMB_MARGIN_M - 1.5  # under the cylinder's top
        ceiling = r.ex.charter.alt_max_agl_m  # pyright: ignore[reportOptionalMemberAccess]
        monkeypatch.setattr(executor_module, "_ALT_GOTO_BELOW_M", ceiling - cap)
        highest, t0 = alt, r.clock.monotonic()
        while r.ex.state != S.HOLD:
            await r.run(0.5)
            highest = max(highest, r.ex.vehicle_state().alt_rel_home_m)
            assert r.clock.monotonic() - t0 < 120, "never hovered"
        holds = _holds(r)
        assert len(holds) == 1 and holds[0]["inside"] is True, holds
        assert holds[0]["climb_to_m"] == pytest.approx(round(cap, 1))
        await r.run(10)
        highest = max(highest, r.ex.vehicle_state().alt_rel_home_m)
        assert highest <= cap + 1.5, (highest, cap)
        assert abs(r.ex.vehicle_state().alt_rel_home_m - cap) <= 1.5


async def test_fly_home_during_the_climb_flies_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """FLY HOME goes past the AI and the reflex as always, also while it climbs."""
    async with rig() as r:
        await _over_a_person_with_no_way_out(r, monkeypatch)
        await r.until(lambda: len(_holds(r)) == 1, 30)
        assert r.ex.state == S.TRANSIT  # climbing
        r.ex.on_override(r.notice("return_home"))
        await r.ground.set_mode(CopterMode.SMART_RTL, origin=SendOrigin.GROUND)
        await r.run(1)
        assert r.ex.held_by == "operator_home"
        before = len(r.vehicle.sent)
        await r.run(10)
        assert r.fc.mode in HOME_MODES  # the climb does not take it back
        assert r.steering_sent(before) == []


async def test_the_battery_limit_during_the_climb_flies_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with rig() as r:
        await _over_a_person_with_no_way_out(r, monkeypatch)
        await r.until(lambda: len(_holds(r)) == 1, 30)
        assert r.ex.state == S.TRANSIT  # climbing
        r.fc.battery_pct = r.ex.charter.battery_rtl_pct - 1  # pyright: ignore[reportOptionalMemberAccess]
        await r.until(lambda: r.ex.state == S.RTL, 30)
        await r.run(5)
        assert r.ex.state == S.RTL and r.fc.mode in HOME_MODES
        assert "executor.limit_battery_rtl" in r.up.kinds()


async def test_a_way_out_planned_again_gets_a_new_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cylinder comes around the drone on an inspection's leg and the gate takes the way
    out: that leg gets the time it needs (the drone is over a person until it is out)."""
    renewed: list[bool] = []
    retarget = GotoHandler.retarget

    def spy(self: GotoHandler, *a: Any, **kw: Any) -> None:
        renewed.append(bool(kw.get("renew_deadline", False)))
        retarget(self, *a, **kw)

    monkeypatch.setattr(GotoHandler, "retarget", spy)
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        target = keepout(150, 0, 10)
        install(r, target)
        camera(r)
        await r.send(r.order("inspect_detection", inspect(150, 0, 10)), 0.5)
        await r.until(lambda: xy(r)[0] >= 15.0, 60)
        here = xy(r)
        install(r, target, keepout(here[0] + 10.0, here[1], 30.0, source="candidate"))
        await r.until(lambda: "executor.keepout_replan" in r.up.kinds(), 10)
        assert renewed and renewed[0] is True


async def test_a_new_end_while_goto_starts_keeps_its_deadline() -> None:
    """``retarget(renew_deadline=True)`` while ``start`` awaits the flight controller (the
    climb set as the leg begins): ``start`` does not put the old leg's deadline back."""

    class _StopError(Exception):
        pass

    now = [0.0]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        ctx = r.ex._plan_context(new_id())  # pyright: ignore[reportPrivateUsage]
        ctx = ctx.__class__(**{**_fields(ctx), "clock": clock})
        goto = GotoHandler()
        lat, lon = at(100, 0)
        goto.prepare(ctx, lat, lon, 40.0, 5.0)
        plan = goto_plan(ctx, lat, lon, 40.0, 5.0)
        seen = [0]

        def vehicle() -> Any:
            seen[0] += 1
            if seen[0] > 1:
                raise _StopError
            return SimpleNamespace(fc_mode=CopterMode.GUIDED.name)

        async def change_speed(_speed: float) -> None:
            now[0] = 100.0  # later, as the reflex sets the climb
            goto.retarget(ctx, lat, lon, 90.0, 5.0, renew_deadline=True)

        renewed = 100.0 + goto_plan(ctx, lat, lon, 90.0, 5.0).expected_duration_s * 2 + 30

        class _IO:
            clock: Any = None

            @property
            def vehicle(self) -> Any:
                return vehicle()

            async def change_speed(self, speed: float) -> None:
                await change_speed(speed)

        io = _IO()
        io.clock = clock
        with pytest.raises(_StopError):
            await goto.start(plan, io)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        assert goto._deadline == pytest.approx(renewed)  # pyright: ignore[reportPrivateUsage]


async def test_replans_of_one_inspection_have_a_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past ``REPLANS_MAX`` the reflex plans the leg no more: the order hovers short of the
    cylinder (the drone is outside it) instead."""
    monkeypatch.setattr(inspect_detection, "REPLANS_MAX", 0)
    async with rig() as r:
        cmd, moved, start = await _inspect_with_a_cylinder_moved_onto_the_leg(r)
        await r.until(lambda: r.ex.state == S.HOLD, 120)
        assert "executor.keepout_replan" not in r.up.kinds()
        holds = _holds(r)
        assert len(holds) == 1 and holds[0]["inside"] is False, holds
        assert ("KEEPOUT_VIOLATION",) in [c for ref, _, c in r.up.acks if ref == cmd.command_id]
        assert xy_route_violation(track_xy(r, start), circles([moved], PLANE)) is None


async def test_a_goto_planned_again_keeps_its_first_deadline() -> None:
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        ctx = r.ex._plan_context(new_id())  # pyright: ignore[reportPrivateUsage]
        goto = GotoHandler()
        lat, lon = at(100, 0)
        goto.prepare(ctx, lat, lon, 40.0, 5.0)
        goto.retarget(ctx, lat, lon, 40.0, 5.0)  # not started yet: the deadline is set
        first = goto._deadline  # pyright: ignore[reportPrivateUsage]
        assert math.isfinite(first)
        await r.run(5)
        far = at(400, 0)
        goto.retarget(ctx.__class__(**{**_fields(ctx), "clock": r.clock}), *far, 40.0, 5.0)
        assert goto._deadline == first  # pyright: ignore[reportPrivateUsage]
        goto.retarget(ctx, *far, 40.0, 5.0, renew_deadline=True)
        assert goto._deadline > first  # pyright: ignore[reportPrivateUsage]


def _fields(ctx: Any) -> dict[str, Any]:
    return {f: getattr(ctx, f) for f in ctx.__slots__}


def test_a_leg_to_the_point_starts_the_ways_out_count_again() -> None:
    h = inspect_detection.InspectHandler()
    h._exits = EXITS_MAX - 1  # pyright: ignore[reportPrivateUsage]
    leg = inspect_detection._Leg(0.0, 0.0, 40.0, 5.0, None, exit=False)  # pyright: ignore[reportPrivateUsage]

    class _Goto:
        def prepare(self, *a: Any, **k: Any) -> None: ...

    h.goto = _Goto()  # type: ignore[assignment]
    h._plan_of = lambda ctx, leg: None  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]
    h._fly(None, leg, retarget=False)  # type: ignore[arg-type]  # pyright: ignore[reportPrivateUsage]
    assert h._exits == 0  # pyright: ignore[reportPrivateUsage]


# --- orbit ------------------------------------------------------------------------------


async def test_orbit_flies_the_ring_and_hovers() -> None:
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        lat, lon = at(0, 100)
        cmd = await r.send(
            r.order("orbit", {"lat": lat, "lon": lon, "radius_m": 40, "alt_m": 40}), 0.5
        )
        assert r.ex.state == S.TRANSIT, r.up.acks[-2:]
        items = r.fc.missions[MISSION]
        assert len(items) == 4 + 25  # home, speed, 24 corners and back, loiter, RTL
        await r.until(lambda: r.fc.mission_current >= 3, 60)
        start = len(r.fc.track)
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 400)
        assert r.ex.state == S.HOLD
        dists = [math.dist(p, (0, 100)) for p in track_xy(r, start)]
        assert min(dists) > 39 and max(dists) < 45


async def test_orbit_too_close_to_a_cylinder_is_refused() -> None:
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        install(r, keepout(0, 100, 35))
        lat, lon = at(0, 100)
        near = await r.send(r.order("orbit", {"lat": lat, "lon": lon, "radius_m": 40, "alt_m": 40}))
        assert r.up.codes(near.command_id) == ("KEEPOUT_VIOLATION",)
        ok = await r.send(
            r.order("orbit", {"lat": lat, "lon": lon, "radius_m": 46, "alt_m": 40}), 0.5
        )
        assert r.ex.state == S.TRANSIT, r.up.codes(ok.command_id)


# --- the reflex -------------------------------------------------------------------------


async def test_a_new_cylinder_reroutes_the_search() -> None:
    box = square(60, east=0, north=120)
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        cmd = await r.send(
            r.order("survey_sector", {"polygon": box, "spacing_m": 30, "speed_ms": 8}), 1
        )
        assert r.ex.state == S.SURVEY
        first = list(r.fc.missions[MISSION])
        k = keepout(0, 120, 20, source="candidate")
        install(r, k)
        await r.run(0.5)
        assert "executor.keepout_reroute" in r.up.kinds(), r.up.kinds()
        assert r.ex.state == S.SURVEY and r.fc.missions[MISSION] != first
        start = len(r.fc.track)
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 600)
        assert xy_route_violation(track_xy(r, start), circles([k], PLANE)) is None
        assert "superseded" not in r.up.stages(cmd.command_id)


BRAKE_MS2 = 2.5  # ArduCopter's LOITER braking (LOIT_BRK_ACCEL)
SOFT_BRAKE_MS2 = 1.0  # a brake too soft to stop short of ``_close_cylinder_ahead``'s cylinder


async def _close_cylinder_ahead(
    r: Rig, brake_ms2: float = BRAKE_MS2
) -> tuple[SkillCommand, Keepout, int]:
    """A search at 8 m/s on a long leg, then a cylinder whose edge is 17 m ahead on it: too
    close to go around at this speed, inside the reflex's stop distance, but farther than
    braking at ``brake_ms2`` takes (the rig's flight controller brakes that hard), unless
    that brake is ``SOFT_BRAKE_MS2``. The order, the cylinder and where the track was when
    it came."""
    box = square(60, east=0, north=120)
    await r.install(charter(allowed_skills=SKILLS))
    await r.takeoff()
    cmd = await r.send(
        r.order("survey_sector", {"polygon": box, "spacing_m": 30, "speed_ms": 8}), 1
    )

    def on_a_long_leg() -> bool:  # at full speed along it, 60 m or more to go
        ahead = r.fc.missions[MISSION][r.fc.mission_current]
        to, here = PLANE.xy(ahead.x / 1e7, ahead.y / 1e7), xy(r)
        d, speed = math.dist(here, to), math.hypot(*r.fc.vel)
        along = (r.fc.vel[0] * (to[0] - here[0]) + r.fc.vel[1] * (to[1] - here[1])) / d
        return r.fc.mission_current >= 3 and d > 60 and speed > 7.9 and along > 0.999 * speed

    await r.until(on_a_long_leg, 300)
    here, (ve, vn) = xy(r), r.fc.vel
    speed = math.hypot(ve, vn)
    braking = speed**2 / (2 * brake_ms2) + speed * 0.5
    assert stop_m(speed) > 17
    assert (braking < 17) is (brake_ms2 != SOFT_BRAKE_MS2), (braking, brake_ms2)
    k = keepout(here[0] + 37 * ve / speed, here[1] + 37 * vn / speed, 20, source="detection")
    start = len(r.fc.track)
    install(r, k)
    await r.run(0.5)
    reroutes = [d for t, _, d in r.up.events if t == "executor.keepout_reroute"]
    assert reroutes and reroutes[-1]["stop_first"] is True
    return cmd, k, start


async def test_a_cylinder_closer_than_the_turn_stops_the_search_first() -> None:
    """It stops in LOITER first and goes around from where it stopped. ArduCopter starts a
    new AUTO leg from its stopping point, so a way around planned from where the drone was
    when the cylinder came would be flown from a point past that."""
    async with rig(fc_options={"accel_ms2": BRAKE_MS2}) as r:
        cmd, k, start = await _close_cylinder_ahead(r)
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 600)
        assert "executor.keepout_reroute_stopped" in r.up.kinds()
        assert "superseded" not in r.up.stages(cmd.command_id)
        assert xy_route_violation(track_xy(r, start), circles([k], PLANE)) is None


@pytest.mark.parametrize("refused_by", ["gate", "planner"])
async def test_no_way_around_from_where_the_search_stopped_is_a_keepout_hold(
    monkeypatch: pytest.MonkeyPatch, refused_by: str
) -> None:
    async with rig(fc_options={"accel_ms2": BRAKE_MS2}) as r:
        cmd, k, start = await _close_cylinder_ahead(r)
        if refused_by == "gate":
            r.ex.path_refusal = lambda active, path: ("KEEPOUT_VIOLATION",)  # type: ignore[method-assign]
        else:

            def no_way(self: survey_sector.SurveyRun, ctx: Any) -> Any:
                raise PlanError("KEEPOUT_VIOLATION", "ingen väg")

            monkeypatch.setattr(survey_sector.SurveyRun, "plan", no_way)
        await r.until(lambda: r.ex.state == S.HOLD, 60)
        assert r.up.codes(cmd.command_id) == ("KEEPOUT_VIOLATION",)
        assert [h["inside"] for h in _holds(r)] == [False]  # short of the cylinder: orange
        assert "executor.keepout_reroute_stopped" not in r.up.kinds()
        assert r.ex.hold_reason == "limit" and r.fc.mode == CopterMode.LOITER
        assert xy_route_violation(track_xy(r, start), circles([k], PLANE)) is None


async def test_no_way_around_from_a_stop_inside_the_cylinder_says_inside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A soft brake stops the search inside the cylinder it stopped for, and no way on is
    planned from there: the hold says ``inside`` (the red alarm), from where the drone is
    when the order fails."""

    def no_way(self: survey_sector.SurveyRun, ctx: Any) -> Any:
        raise PlanError("KEEPOUT_VIOLATION", "ingen väg")

    async with rig(fc_options={"accel_ms2": SOFT_BRAKE_MS2}) as r:
        cmd, k, _ = await _close_cylinder_ahead(r, SOFT_BRAKE_MS2)
        monkeypatch.setattr(survey_sector.SurveyRun, "plan", no_way)
        await r.until(lambda: r.ex.state == S.HOLD, 60)
        assert r.up.codes(cmd.command_id) == ("KEEPOUT_VIOLATION",)
        ((c, rad),) = circles([k], PLANE)
        assert math.dist(xy(r), c) < rad  # it stopped inside
        holds = _holds(r)
        assert [h["inside"] for h in holds] == [True]
        # Already above the cylinder's top and the margin: it "climbs" to where it is.
        alt = r.ex.vehicle_state().alt_rel_home_m
        assert alt >= rad + KEEPOUT_CLIMB_MARGIN_M - 1.5
        assert holds[0]["climb_to_m"] == pytest.approx(alt, abs=1.5)


async def test_a_way_around_the_gate_refuses_for_another_reason_says_that_reason() -> None:
    """The way around from the stop clears the cylinder but the gate says no for the
    battery: the order fails with the gate's reason, not as a keep-out hold."""
    async with rig(fc_options={"accel_ms2": BRAKE_MS2}) as r:
        cmd, k, start = await _close_cylinder_ahead(r)
        r.ex.path_refusal = lambda active, path: ("BATTERY_RESERVE",)  # type: ignore[method-assign]
        await r.until(lambda: r.ex.state == S.HOLD, 60)
        assert r.up.codes(cmd.command_id) == ("BATTERY_RESERVE",)
        assert "executor.keepout_hold" not in r.up.kinds()
        assert r.ex.hold_reason == "skill_done" and r.fc.mode == CopterMode.LOITER
        assert xy_route_violation(track_xy(r, start), circles([k], PLANE)) is None


async def test_a_drone_that_never_stands_still_is_planned_from_where_it_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flight controller never reports the drone still: the rest is planned from where
    it is once ``STOP_WAIT_S`` has passed, through the same gate."""
    monkeypatch.setattr(survey_sector, "still", lambda vehicle: False)
    async with rig(fc_options={"accel_ms2": BRAKE_MS2}) as r:
        cmd, k, start = await _close_cylinder_ahead(r)
        came = r.clock.monotonic()
        await r.until(lambda: "executor.keepout_reroute_stopped" in r.up.kinds(), 60)
        assert r.clock.monotonic() - came >= survey_sector.STOP_WAIT_S - 1.0
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 600)
        assert xy_route_violation(track_xy(r, start), circles([k], PLANE)) is None


async def test_a_new_cylinder_on_a_goto_hovers() -> None:
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        lat, lon = at(0, 180)
        cmd = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}), 1)
        assert r.ex.state == S.TRANSIT
        k = keepout(0, 120, 30, source="candidate")
        install(r, k)
        await r.run(0.5)
        assert r.ex.state == S.TRANSIT  # far off yet: it flies on
        start = len(r.fc.track)
        await r.until(lambda: r.ex.state == S.HOLD, 120)
        assert r.ex.hold_reason == "limit"
        assert ("KEEPOUT_VIOLATION",) in [c for ref, _, c in r.up.acks if ref == cmd.command_id]
        assert [h["inside"] for h in _holds(r)] == [False]  # short of it: orange
        await r.run(10)
        assert xy_route_violation(track_xy(r, start), circles([k], PLANE)) is None
        assert xy(r)[1] < 90.0


async def test_a_cylinder_off_the_route_changes_nothing() -> None:
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        lat, lon = at(0, 180)
        cmd = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}), 1)
        install(r, keepout(120, 120, 30))
        await r.run(0.5)
        assert r.ex.state == S.TRANSIT
        assert "executor.keepout_hold" not in r.up.kinds()
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 120)


async def test_return_home_never_stops_for_a_cylinder() -> None:
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        lat, lon = at(0, 150)
        await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}), 1)
        await r.until(lambda: r.ex.state == S.HOLD, 120)
        await r.send(r.order("return_home", {}), 0.5)
        assert r.ex.state == S.RTL
        install(r, keepout(0, 75, 30, source="candidate"))
        await r.run(0.5)
        assert r.ex.state == S.RTL and "executor.keepout_hold" not in r.up.kinds()


# --- the drone's own cylinders ----------------------------------------------------------


def detection(east: float, north: float, score: float, sigma: float = 5.0) -> GeoDetectionMsg:
    lat, lon = at(east, north)
    return GeoDetectionMsg(
        detection_id=new_id(),
        bbox=BBox(x=10, y=10, w=20, h=40),
        score=score,
        lat=lat,
        lon=lon,
        sigma95_m=sigma,
        px_size=40,
        crop_inline=False,
        label="person",
    )


async def test_own_detections_make_cylinders_until_the_ground_takes_over() -> None:
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        sure, unsure = detection(100, 100, 0.8), detection(-100, 100, 0.3)
        r.ex.on_detections([sure, unsure])
        (k,) = r.ex.keepouts_in_force()
        assert k.source == "detection" and k.source_id == sure.detection_id
        assert abs(k.radius_m - 35.0) < 1e-9
        assert r.up.kinds().count("executor.keepout_created") == 1
        # Again, or inside the first one: nothing new.
        r.ex.on_detections([sure, detection(101, 100, 0.9, sigma=1.0)])
        assert len(r.ex.keepouts_in_force()) == 1
        # The gate counts it: a goto through it is refused.
        await r.takeoff()
        lat, lon = at(150, 150)
        cmd = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}))
        assert r.up.codes(cmd.command_id) == ("KEEPOUT_VIOLATION",)
        # A ground install soon after keeps it, as does one a minute later without it; one
        # a minute later that covers it replaces it.
        install(r, keepout(100, 100, 40, source="candidate"))
        assert len(r.ex.keepouts_in_force()) == 2
        await r.run(61)
        install(r)
        assert r.ex.keepouts_in_force() == (k,)
        ground = keepout(100, 100, 40, source="candidate")
        install(r, ground)
        assert r.ex.keepouts_in_force() == (ground,)


def _covers(big: Keepout, small: Keepout) -> bool:
    c = PLANE.xy(big.lat, big.lon)
    return math.dist(c, PLANE.xy(small.lat, small.lon)) + small.radius_m <= big.radius_m


async def test_repeated_detections_of_one_person_merge_into_one_cylinder() -> None:
    """One person seen again and again, the position a few metres off each time: one
    local cylinder that grows to cover every detection's, never smaller. A person farther
    off gets a cylinder of their own."""
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        first = detection(100, 100, 0.9, sigma=5.0)
        r.ex.on_detections([first])
        ((k0, _),) = r.ex.local_keepouts.values()
        seen = [k0]
        for i in range(60):
            dx, dy = 8.0 * math.cos(i), 8.0 * math.sin(i)
            r.ex.on_detections([detection(100 + dx, 100 + dy, 0.9, sigma=5.0)])
            ((k, _),) = r.ex.local_keepouts.values()
            seen.append(k)
        assert r.up.kinds().count("executor.keepout_created") == 1
        assert "executor.keepout_merged" in r.up.kinds()
        final = seen[-1]
        # Each step covers the one before (so every detection's circle), with a new id.
        for a, b in pairwise(seen):
            if a != b:
                assert _covers(b, a) and a.keepout_id != b.keepout_id
        assert final.source == "detection" and final.radius_m < 50.0
        merged = [d for t, _, d in r.up.events if t == "executor.keepout_merged"]
        assert merged[-1]["keepout_id"] == str(final.keepout_id)
        r.ex.on_detections([detection(160, 100, 0.9, sigma=5.0)])
        assert len(r.ex.local_keepouts) == 2


async def test_a_merged_cylinder_holds_a_goto_short_of_the_person() -> None:
    """A goto past the drone's own cylinder, clear of it. The same person seen again,
    less sure: the grown cylinder now holds the drone, which came outside the old one, so
    it is not let through. The drone never goes deeper into the grown one: it climbs
    straight up over it as an inspection does, then hovers there."""
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        r.ex.on_detections([detection(50, 120, 0.9, sigma=10.0)])
        ((old, _),) = r.ex.local_keepouts.values()
        lat, lon = at(0, 180)
        await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 5}), 1)
        await r.until(lambda: xy(r)[1] >= 90.0, 120)
        assert r.ex.state == S.TRANSIT
        r.ex.on_detections([detection(35, 108, 0.9, sigma=25.0)])
        ((grown, _),) = r.ex.local_keepouts.values()
        assert grown.keepout_id != old.keepout_id
        centre = PLANE.xy(grown.lat, grown.lon)
        assert math.dist(xy(r), centre) < grown.radius_m
        assert math.dist(xy(r), PLANE.xy(old.lat, old.lon)) > old.radius_m
        start, alt = len(r.fc.track), r.ex.vehicle_state().alt_rel_home_m
        want = grown.radius_m + KEEPOUT_CLIMB_MARGIN_M
        assert alt < want - 1.5  # under the grown cylinder's top: it climbs
        await r.until(lambda: r.ex.state == S.HOLD, 60)
        assert r.ex.hold_reason == "limit" and r.fc.mode == CopterMode.LOITER
        assert math.dist(xy(r), centre) < grown.radius_m
        holds = _holds(r)
        assert [h["inside"] for h in holds] == [True]  # over the person: red
        assert holds[0]["climb_to_m"] == pytest.approx(round(want, 1))
        assert abs(r.ex.vehicle_state().alt_rel_home_m - want) <= 1.5
        await r.run(5)
        dists = [math.dist(p, centre) for p in track_xy(r, start)]
        assert min(dists) >= dists[0] - 3.0  # the brake, no deeper
        assert len(_holds(r)) == 1


async def test_one_person_twice_in_one_batch_holds_a_goto_outside() -> None:
    """Two detections of one person in one batch, the goto heading across them: the
    first cylinder is not yet seen by the reflex when the second grows it. The drone came
    outside the first, so the grown one does not let it through: it hovers outside the
    first and never comes nearer the person."""
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        lat, lon = at(0, 180)
        await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 5}), 1)
        await r.until(lambda: xy(r)[1] >= 60.0, 120)
        assert r.ex.state == S.TRANSIT
        r.ex.on_detections([detection(0, 120, 0.9, sigma=10.0), detection(0, 105, 0.9, sigma=25.0)])
        ((grown, _),) = r.ex.local_keepouts.values()
        assert "executor.keepout_merged" in r.up.kinds()
        assert math.dist(xy(r), PLANE.xy(grown.lat, grown.lon)) < grown.radius_m
        start = len(r.fc.track)
        await r.until(lambda: r.ex.state == S.HOLD, 20)
        assert r.ex.hold_reason == "limit"
        await r.run(5)
        dists = [math.dist(p, (0.0, 120.0)) for p in track_xy(r, start)]
        assert min(dists) >= 40.0  # outside the first cylinder
        assert min(dists) >= dists[0] - 3.0  # the brake, no nearer


async def test_one_person_twice_in_one_batch_over_the_drone_lets_it_leave() -> None:
    """Two detections of one person in one batch, the first cylinder already round the
    drone: as with one such cylinder, the drone leaves along its leg instead of hovering
    over the person."""
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        lat, lon = at(0, 180)
        await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 5}), 1)
        await r.until(lambda: xy(r)[1] >= 60.0, 120)
        assert r.ex.state == S.TRANSIT
        r.ex.on_detections([detection(0, 70, 0.9, sigma=10.0), detection(0, 80, 0.9, sigma=25.0)])
        ((grown, _),) = r.ex.local_keepouts.values()
        assert "executor.keepout_merged" in r.up.kinds()
        centre = PLANE.xy(grown.lat, grown.lon)
        assert math.dist(xy(r), centre) < grown.radius_m
        start = len(r.fc.track)
        await r.until(lambda: math.dist(xy(r), centre) > grown.radius_m, 60)
        assert "executor.keepout_hold" not in r.up.kinds()
        assert r.ex.state == S.TRANSIT
        assert all(p[1] >= 59.0 for p in track_xy(r, start))  # on along its leg


def test_merge_local_geometry() -> None:
    old = keepout(0, 0, 31.0, source="detection")
    # Too far apart, or one would grow past the largest radius: no merge.
    assert merge_local(keepout(16, 0, 31.0, source="detection"), [old], PLANE) is None
    huge = keepout(0, 0, 480.0, source="detection")
    assert merge_local(keepout(100, 0, 480.0, source="detection"), [huge], PLANE) is None
    # The new one holds the old one: it is the new one, a little larger.
    wide = keepout(5, 0, 50.0, source="detection")
    found = merge_local(wide, [old], PLANE)
    assert found is not None and found[0] is old
    assert _covers(found[1], wide) and _covers(found[1], old)
    assert found[1].radius_m == pytest.approx(50.1)
    # The old one holds the new one: the old one, a little larger.
    smaller = keepout(2, 0, 20.0, source="detection")
    found = merge_local(smaller, [old], PLANE)
    assert found is not None and found[1].radius_m == pytest.approx(31.1)
    # More than twice the new one's radius: no merge (one that small inside is already
    # kept out, the executor drops it before merging).
    assert merge_local(keepout(2, 0, 10.0, source="detection"), [old], PLANE) is None
    # A person who moves on gets a new cylinder instead of dragging one along.
    grown = old
    for step in range(1, 20):
        found = merge_local(keepout(15 * step, 0, 31.0, source="detection"), [grown], PLANE)
        if found is None:
            break
        grown = found[1]
    assert found is None and grown.radius_m <= 62.0
    # The nearest of several.
    near = keepout(10, 0, 31.0, source="detection")
    found = merge_local(keepout(12, 0, 31.0, source="detection"), [old, near], PLANE)
    assert found is not None and found[0] is near


def test_way_out_of_cylinders() -> None:
    area = polygon_from_geojson(square(200.0), PLANE)
    cyl = [((0.0, 0.0), 30.0), ((0.0, 50.0), 15.0)]
    assert way_out((40.0, 0.0), cyl, area) is None  # in none
    out = way_out((0.0, 10.0), cyl, area)
    assert out is not None
    # Out of the first, deeper into neither, into no other.
    assert math.dist(out, cyl[0][0]) > cyl[0][1]
    assert xy_route_violation([(0.0, 10.0), out], cyl) is None
    # The charter's area leaves no way out.
    tight = polygon_from_geojson(square(20.0), PLANE)
    assert way_out((0.0, 0.0), [((0.0, 0.0), 30.0)], tight) is None


@pytest.mark.parametrize("over", [False, True])
async def test_a_full_list_of_own_cylinders_stops_a_search(
    monkeypatch: pytest.MonkeyPatch, over: bool
) -> None:
    """The hover tells the ground whether the drone is over a person (in a cylinder in
    force) when the list fills, as every keep-out hover does."""
    box = square(60, east=0, north=120)
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        cmd = await r.send(
            r.order("survey_sector", {"polygon": box, "spacing_m": 30, "speed_ms": 8}), 1
        )
        assert r.ex.state == S.SURVEY
        # Behind the drone, 16 m apart: none holds another, none is close enough to merge
        # with another, none is on the route.
        side = math.isqrt(LOCAL_KEEPOUTS_MAX - 1) + 1
        r.ex.on_detections(
            [
                detection(-80 + 16 * (i % side), -90 - 16 * (i // side), 0.9, sigma=1.0)
                for i in range(LOCAL_KEEPOUTS_MAX)
            ]
        )
        assert len(r.ex.local_keepouts) == LOCAL_KEEPOUTS_MAX
        assert r.ex.state == S.SURVEY
        monkeypatch.setattr(r.ex, "_over_a_keepout", lambda: over)
        r.ex.on_detections([detection(140, -90, 0.9)])
        assert "executor.keepouts_full" in r.up.kinds()
        assert r.ex.state == S.HOLD and r.ex.hold_reason == "limit"
        holds = [d for t, _, d in r.up.events if t == "executor.keepout_hold"]
        assert [h["inside"] for h in holds] == [over], holds
        assert r.up.stages(cmd.command_id)[-1] == "superseded"


async def test_inside_a_new_cylinder_the_search_leaves_it_first() -> None:
    box = square(60, east=0, north=120)
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        cmd = await r.send(
            r.order("survey_sector", {"polygon": box, "spacing_m": 30, "speed_ms": 8}), 1
        )
        await r.until(lambda: r.fc.mission_current >= 4, 120)  # on a search leg
        here = xy(r)
        ahead = r.fc.missions[MISSION][r.fc.mission_current]
        to = PLANE.xy(ahead.x / 1e7, ahead.y / 1e7)
        d = math.dist(here, to)
        centre = (here[0] + 10 * (to[0] - here[0]) / d, here[1] + 10 * (to[1] - here[1]) / d)
        k = keepout(*centre, 25, source="detection")
        install(r, k)
        start = len(r.fc.track)
        await r.run(0.5)
        assert "executor.keepout_reroute" in r.up.kinds()
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 600)
        # Never closer to the person than when the cylinder came, until out of it.
        dists = [math.dist(p, centre) for p in track_xy(r, start)]
        out = next(i for i, x in enumerate(dists) if x >= 25)
        assert all(b >= a - 0.5 for a, b in pairwise(dists[: out + 1]))
        later = track_xy(r, start + out)
        assert xy_route_violation(later, circles([k], PLANE)) is None


BRAKE_M = 2.0  # how far the drone goes on while the reflex stops it


async def test_two_own_cylinders_under_the_drone_the_search_leaves_both() -> None:
    """Two stills of a person right below, too far apart to merge: the centres either side
    of the drone. The way out goes deeper into neither (straight out from the first would
    go into the second), and the search goes on."""
    box = square(60, east=0, north=120)
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        cmd = await r.send(
            r.order("survey_sector", {"polygon": box, "spacing_m": 30, "speed_ms": 8}), 1
        )
        await r.until(lambda: r.fc.mission_current >= 4, 120)  # on a search leg
        here = xy(r)
        ahead = r.fc.missions[MISSION][r.fc.mission_current]
        to = PLANE.xy(ahead.x / 1e7, ahead.y / 1e7)
        d = math.dist(here, to)
        u = ((to[0] - here[0]) / d, (to[1] - here[1]) / d)
        # One centre just behind, one ahead: out from the first is into the second.
        dets = [
            detection(here[0] - u[0], here[1] - u[1], 0.9, 3.0),
            detection(here[0] + 18 * u[0], here[1] + 18 * u[1], 0.9, 3.0),
        ]
        start = len(r.fc.track)
        r.ex.on_detections(dets)
        await r.run(0.5)
        assert "executor.keepout_reroute" in r.up.kinds()
        assert "executor.keepout_reroute_refused" not in r.up.kinds()
        assert "executor.keepout_hold" not in r.up.kinds()
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 600)
        cyl = circles([k for k, _ in r.ex.local_keepouts.values()], PLANE)
        assert len(cyl) == 2
        track = track_xy(r, start)
        # No closer to either centre than braking takes, then out and never back in.
        for c, _ in cyl:
            assert min(math.dist(p, c) for p in track) >= math.dist(track[0], c) - BRAKE_M
        inside = [any(math.dist(p, c) < rad for c, rad in cyl) for p in track]
        out = inside.index(False)
        assert not any(inside[out:])


async def test_no_way_around_inside_the_charter_hovers(monkeypatch: pytest.MonkeyPatch) -> None:
    box = square(60, east=0, north=120)
    monkeypatch.setattr(survey_sector, "reroute", lambda *_a, **_k: None)
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        cmd = await r.send(
            r.order("survey_sector", {"polygon": box, "spacing_m": 30, "speed_ms": 8}), 1
        )
        install(r, keepout(0, 120, 30, source="candidate"))
        await r.run(0.5)
        assert r.ex.state == S.HOLD and [h["inside"] for h in _holds(r)] == [False]
        assert r.up.stages(cmd.command_id)[-1] == "superseded"
        assert r.ex.resume_point is not None  # the search can go on later


async def _search_over_a_person(
    r: Rig, monkeypatch: pytest.MonkeyPatch, radius: float = 50.0
) -> tuple[SkillCommand, Keepout, tuple[float, float], int]:
    """A cylinder comes around the drone on a search leg and no way out is planned (the
    search is no ``Replans``): the order, the cylinder, where the drone was and where the
    track was then."""
    box = square(60, east=0, north=120)
    monkeypatch.setattr(survey_sector, "reroute", lambda *_a, **_k: None)
    await r.install(charter(allowed_skills=SKILLS))
    await r.takeoff()
    cmd = await r.send(
        r.order("survey_sector", {"polygon": box, "spacing_m": 30, "speed_ms": 8}), 1
    )
    await r.until(lambda: r.fc.mission_current >= 4, 120)  # on a search leg
    here = xy(r)
    person = keepout(here[0], here[1], radius, source="candidate")
    start = len(r.fc.track)
    install(r, person)
    return cmd, person, here, start


async def test_no_way_out_of_a_cylinder_around_a_search_climbs_then_hovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over a person with no way out, a search climbs straight up as an inspection does:
    to the cylinder's top and the margin, in an internal goto that replaces the order;
    there it hovers. The operator is told once, as the climb begins (the red alarm)."""
    async with rig() as r:
        cmd, person, here, start = await _search_over_a_person(r, monkeypatch)
        want = person.radius_m + KEEPOUT_CLIMB_MARGIN_M
        assert r.ex.vehicle_state().alt_rel_home_m < want - 1.5
        await r.run(0.5)
        assert r.ex.state == S.TRANSIT  # climbing
        assert r.up.stages(cmd.command_id)[-1] == "superseded"
        assert ("KEEPOUT_VIOLATION",) in [c for ref, _, c in r.up.acks if ref == cmd.command_id]
        assert r.ex.resume_point is not None  # the search can go on later
        await r.until(lambda: r.ex.state == S.HOLD, 120)
        holds = _holds(r)
        assert len(holds) == 1, holds
        assert holds[0]["inside"] is True and holds[0]["climb_to_m"] == pytest.approx(want)
        assert holds[0]["command_id"] == str(cmd.command_id)
        assert r.ex.hold_reason == "limit" and r.fc.mode == CopterMode.LOITER
        assert abs(r.ex.vehicle_state().alt_rel_home_m - want) <= 1.5
        # Straight up: no farther off than braking from the search's speed.
        assert math.dist(track_xy(r, start)[-1], here) <= 10.0
        await r.run(10)
        assert len(_holds(r)) == 1  # hovering there: told once
        assert abs(r.ex.vehicle_state().alt_rel_home_m - want) <= 1.5


async def test_no_way_out_of_a_search_and_no_room_to_climb_hovers_where_it_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No higher allowed: the search hovers where it is, and the operator is told it
    cannot climb (``inside`` and no ``climb_to_m``)."""
    monkeypatch.setattr(executor_module, "_ALT_GOTO_BELOW_M", 200.0)
    async with rig() as r:
        cmd, _, _, _ = await _search_over_a_person(r, monkeypatch)
        alt = r.ex.vehicle_state().alt_rel_home_m
        await r.run(0.5)
        assert r.ex.state == S.HOLD
        holds = _holds(r)
        assert len(holds) == 1 and holds[0]["inside"] is True, holds
        assert "climb_to_m" not in holds[0]
        assert r.up.stages(cmd.command_id)[-1] == "superseded"
        await r.run(5)
        assert r.ex.vehicle_state().alt_rel_home_m <= alt + 1.5


async def test_a_search_climbs_only_to_the_charter_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cylinder's top lies above what the charter allows: the search's climb stops at
    the charter's height less the altitude limit's margin, and never goes above it."""
    async with rig() as r:
        await _search_over_a_person(r, monkeypatch, radius=100.0)
        alt = r.ex.vehicle_state().alt_rel_home_m
        cap = alt + 8.0
        ceiling = r.ex.charter.alt_max_agl_m  # pyright: ignore[reportOptionalMemberAccess]
        monkeypatch.setattr(executor_module, "_ALT_GOTO_BELOW_M", ceiling - cap)
        highest, t0 = alt, r.clock.monotonic()
        while r.ex.state != S.HOLD:
            await r.run(0.5)
            highest = max(highest, r.ex.vehicle_state().alt_rel_home_m)
            assert r.clock.monotonic() - t0 < 120, "never hovered"
        holds = _holds(r)
        assert len(holds) == 1 and holds[0]["climb_to_m"] == pytest.approx(round(cap, 1))
        await r.run(10)
        highest = max(highest, r.ex.vehicle_state().alt_rel_home_m)
        assert highest <= cap + 1.5, (highest, cap)


async def test_stop_during_a_search_climb_brakes_and_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STOP goes past the AI and the reflex as always, also during another order's climb."""
    async with rig() as r:
        await _search_over_a_person(r, monkeypatch)
        await r.until(lambda: len(_holds(r)) == 1, 30)
        await r.run(2)
        assert r.ex.state == S.TRANSIT  # climbing
        r.fc.set_mode_external(CopterMode.BRAKE)  # the ground's STOP, as MAVLink first
        await r.run(0.3)
        r.ex.on_override(r.notice("stop"))
        assert r.ex.state == S.HELD and r.ex.held_by == "operator_stop"
        assert r.ex.active is None  # the climb is dropped
        before = len(r.vehicle.sent)
        await r.run(5)
        # The executor no longer steers (the fake keeps its last target on a mode change
        # it was not asked for, so its altitude here says nothing about BRAKE).
        assert r.fc.mode == CopterMode.BRAKE
        assert r.steering_sent(before) == []


async def test_fly_home_during_a_search_climb_flies_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FLY HOME goes past the AI and the reflex as always, also during another order's
    climb."""
    async with rig() as r:
        await _search_over_a_person(r, monkeypatch)
        await r.until(lambda: len(_holds(r)) == 1, 30)
        await r.run(2)
        assert r.ex.state == S.TRANSIT  # climbing
        r.ex.on_override(r.notice("return_home"))
        await r.ground.set_mode(CopterMode.SMART_RTL, origin=SendOrigin.GROUND)
        await r.run(1)
        assert r.ex.held_by == "operator_home"
        before = len(r.vehicle.sent)
        await r.run(10)
        assert r.fc.mode in HOME_MODES  # the climb does not take it back
        assert r.steering_sent(before) == []


async def test_a_detour_the_gate_refuses_hovers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The new route goes through the whole gate: refused (here as if the detour ate the
    return reserve), the search stops in a hover instead."""
    box = square(60, east=0, north=120)
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        cmd = await r.send(
            r.order("survey_sector", {"polygon": box, "spacing_m": 30, "speed_ms": 8}), 1
        )
        assert r.ex.state == S.SURVEY

        def refused(req: GateRequest, ctx: GateContext) -> GateDecision:
            return evaluate(req, ctx).model_copy(
                update={"decision": "REJECT", "reason_codes": ["ROUTE_BATTERY"]}
            )

        monkeypatch.setattr(executor_module, "evaluate", refused)
        install(r, keepout(0, 120, 20, source="candidate"))
        await r.run(0.5)
        assert "executor.keepout_reroute_refused" in r.up.kinds(), r.up.kinds()
        assert "executor.keepout_reroute" not in r.up.kinds()
        assert r.ex.state == S.HOLD and r.ex.hold_reason == "limit"
        assert r.up.stages(cmd.command_id)[-1] == "superseded"


def test_entry_and_stop_distances() -> None:
    cyl = [((0.0, 100.0), 30.0)]
    assert entry_m([(0.0, 0.0), (0.0, 200.0)], cyl) == pytest.approx(70.0)
    assert entry_m([(0.0, 0.0), (50.0, 0.0), (50.0, 200.0)], cyl) is None
    assert entry_m([(0.0, 0.0), (0.0, 50.0), (0.0, 200.0)], cyl) == pytest.approx(70.0)
    assert entry_m([(0.0, 90.0), (0.0, 200.0)], cyl) == 0.0  # inside and deeper
    assert entry_m([(0.0, 90.0), (0.0, 0.0)], cyl) is None  # inside, leaving
    # Through a cylinder it was inside when it came: the first leg only.
    assert entry_m([(0.0, 90.0), (0.0, 200.0)], cyl, through=frozenset({0})) is None
    back = [(0.0, 90.0), (0.0, 200.0), (0.0, 0.0)]
    assert entry_m(back, cyl, through=frozenset({0})) == pytest.approx(110.0 + 70.0)
    assert stop_m(0.0) < stop_m(4.0) < stop_m(15.0)
    assert stop_m(-1.0) == stop_m(0.0)


def test_leg_entry_from_where_the_drone_is() -> None:
    cyl = [((0.0, 100.0), 30.0)]
    assert leg_entry_m((0.0, 0.0), (0.0, 200.0), cyl) == pytest.approx(70.0)
    assert leg_entry_m((0.0, 0.0), (50.0, 0.0), cyl) is None
    assert leg_entry_m((0.0, 0.0), (0.0, 0.0), cyl) is None
    # Inside: leaving is clear, ending inside or going deeper is an entry at once.
    assert leg_entry_m((0.0, 90.0), (0.0, 50.0), cyl) is None
    assert leg_entry_m((0.0, 90.0), (0.0, 80.0), cyl) == 0.0
    assert leg_entry_m((0.0, 80.0), (0.0, 140.0), cyl) == 0.0
    # A way out seen from a little further on comes a few cm closer: still leaving.
    assert leg_entry_m((0.5, 90.0), (-40.0, 89.0), cyl) is None
    assert leg_entry_m((5.0, 90.0), (-40.0, 89.0), cyl) == 0.0


async def test_a_goto_over_a_new_own_detection_flies_on_out() -> None:
    """A person seen under the drone: the cylinder comes with the drone inside it, and the
    goto leaves it along its leg rather than hover over the person."""
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        lat, lon = at(0, 180)
        cmd = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}), 1)
        await r.until(lambda: xy(r)[1] >= 60.0, 120)
        e, n = xy(r)
        r.ex.on_detections([detection(e, n + 5.0, 0.9)])
        await r.run(1)
        assert r.ex.state == S.TRANSIT and "executor.keepout_hold" not in r.up.kinds()
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 300)
