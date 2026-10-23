"""B2-09 onboard: orbit, capture_stills, inspect_detection and the keep-out reflex."""

from __future__ import annotations

import asyncio
import math
from itertools import pairwise
from typing import Any
from uuid import UUID

import pytest
from onboard_helpers import ONBOARD_SKILLS, PLANE, Rig, at, charter, rig, square

from sd_mav.mission import MISSION
from sd_onboard.executor import executor as executor_module
from sd_onboard.executor.keepout_reflex import LOCAL_KEEPOUTS_MAX, entry_m, stop_m
from sd_onboard.skills import survey_sector
from sd_safety import evaluate
from sd_safety.keepout import circles, xy_route_violation
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.gate import GateContext, GateDecision, GateRequest
from sd_schemas.keepout import INSPECT_AIM_DEG, INSPECT_VIEW_DEG, Keepout
from sd_schemas.link import BBox, GeoDetectionMsg, KeepoutInstall

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
        assert "executor.keepout_hold" in r.up.kinds()
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


async def test_a_full_list_of_own_cylinders_stops_a_search() -> None:
    box = square(60, east=0, north=120)
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS))
        await r.takeoff()
        cmd = await r.send(
            r.order("survey_sector", {"polygon": box, "spacing_m": 30, "speed_ms": 8}), 1
        )
        assert r.ex.state == S.SURVEY
        # Behind the drone, a few metres apart: none holds another, none is on the route.
        r.ex.on_detections(
            [detection(-140 + 2.8 * i, -90, 0.9, sigma=1.0) for i in range(LOCAL_KEEPOUTS_MAX)]
        )
        assert r.ex.state == S.SURVEY
        r.ex.on_detections([detection(140, -90, 0.9)])
        assert "executor.keepouts_full" in r.up.kinds()
        assert r.ex.state == S.HOLD and r.ex.hold_reason == "limit"
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
    """Two stills of a person right below: the centres either side of the drone. The way
    out goes deeper into neither (straight out from the first would go into the second),
    and the search goes on."""
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
        # One centre just behind, one just ahead: out from the first is into the second.
        dets = [
            detection(here[0] - u[0], here[1] - u[1], 0.9, 3.0),
            detection(here[0] + 2 * u[0], here[1] + 2 * u[1], 0.9, 3.0),
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
        assert r.ex.state == S.HOLD and "executor.keepout_hold" in r.up.kinds()
        assert r.up.stages(cmd.command_id)[-1] == "superseded"
        assert r.ex.resume_point is not None  # the search can go on later


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
