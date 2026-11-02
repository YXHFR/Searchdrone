"""A search around keep-out cylinders that come and go: the rest is planned from the
search pattern, so a way around a cylinder that is gone is not flown on (SITL SK03, where
every rejected candidate's way around stayed in the search and it ran out of time)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from onboard_helpers import ONBOARD_SKILLS, PLANE, Rig, at, charter, rig, square

from sd_mav.mission import MISSION
from sd_onboard.executor import executor as executor_module
from sd_onboard.executor.keepout_reflex import stop_m, survey_run
from sd_onboard.skills.survey_sector import SurveyRun
from sd_safety import evaluate
from sd_safety.keepout import circles, xy_route_violation
from sd_safety.targets import Xy
from sd_schemas.base import LatLon, new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.gate import GateContext, GateDecision, GateRequest
from sd_schemas.keepout import Keepout
from sd_schemas.link import KeepoutInstall, SkillCommand

BOX = square(60, east=0, north=120)
# Each change waits for the reroute before it to fly: a new mission cancelled while its
# switch to AUTO is still on the way is a separate race (it ends in HELD, "external").
_SETTLE_S = 2.0


def keepout(east: float, north: float, r: float) -> Keepout:
    lat, lon = at(east, north)
    return Keepout(
        keepout_id=new_id(),
        lat=lat,
        lon=lon,
        radius_m=r,
        source="candidate",
        source_id=new_id(),
    )


def install(r: Rig, *keepouts: Keepout) -> None:
    seq = r.ex.keepout_seq + 1
    r.ex.on_keepouts(KeepoutInstall(install_seq=seq, keepouts=list(keepouts)), new_id())


def run_of(r: Rig) -> SurveyRun:
    active = r.ex.active
    assert active is not None
    run = survey_run(active.handler)
    assert run is not None
    return run


def corners(run: SurveyRun) -> list[tuple[float, float]]:
    """The corners of the ways around still to fly."""
    return [
        PLANE.xy(p.lat, p.lon)
        for p, o in zip(run.waypoints[run.next_index :], run.origin[run.next_index :], strict=True)
        if o is None
    ]


def centre(k: Keepout) -> tuple[float, float]:
    return PLANE.xy(k.lat, k.lon)


async def _search_around(r: Rig, k: Keepout) -> tuple[SkillCommand, list[Xy]]:
    """A search, and a cylinder on it that it goes around: the order and its pattern."""
    await r.install(charter(allowed_skills=ONBOARD_SKILLS))
    await r.takeoff()
    cmd = await r.send(
        r.order("survey_sector", {"polygon": BOX, "spacing_m": 30, "speed_ms": 8}), 1
    )
    assert r.ex.state == S.SURVEY
    pattern = [PLANE.xy(p.lat, p.lon) for p in run_of(r).pattern]
    install(r, k)
    await r.run(_SETTLE_S)
    assert r.up.kinds().count("executor.keepout_reroute") == 1, r.up.kinds()
    assert run_of(r).detours() and corners(run_of(r))
    return cmd, pattern


async def test_a_lifted_cylinder_drops_its_way_around() -> None:
    """The candidate is rejected (its cylinder goes): the search flies the pattern again,
    the points the way around left out included, and ends."""
    async with rig() as r:
        cmd, _ = await _search_around(r, keepout(0, 120, 40))
        install(r)
        await r.run(0.5)
        assert "executor.keepout_straighten" in r.up.kinds(), r.up.kinds()
        run = run_of(r)
        assert not run.detours() and not corners(run)
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 600)


async def test_a_cylinder_handed_to_the_ground_keeps_the_way_around() -> None:
    """The same cylinder under another id (the drone's own handed to the ground): nothing
    shorter, so the search flies on as it is, no stop."""
    async with rig() as r:
        k = keepout(0, 120, 40)
        await _search_around(r, k)
        before = list(run_of(r).waypoints)
        install(r, k.model_copy(update={"keepout_id": new_id()}))
        await r.run(0.5)
        kinds = r.up.kinds()
        assert "executor.keepout_straighten" not in kinds
        assert kinds.count("executor.keepout_reroute") == 1, kinds
        assert run_of(r).waypoints == before


async def test_a_reroute_goes_around_the_cylinders_in_force_only() -> None:
    """One cylinder goes as another comes: the new way around is planned from the
    pattern, so every point left is the pattern's or a corner around the second."""
    async with rig() as r:
        first = keepout(0, 120, 40)
        _, pattern = await _search_around(r, first)
        second = keepout(45, 150, 10)
        install(r, second)
        await r.run(0.5)
        assert r.up.kinds().count("executor.keepout_reroute") == 2, r.up.kinds()
        a, b = centre(first), centre(second)
        run = run_of(r)
        left = [PLANE.xy(p.lat, p.lon) for p in run.waypoints[run.next_index :]]
        around = [p for p in left if min(math.dist(p, q) for q in pattern) > 0.5]
        assert around, "a way around the second"
        assert all(math.dist(p, b) < math.dist(p, a) for p in around), around


def test_the_pattern_rest_follows_the_pattern_points_reached() -> None:
    """Pattern 0..3, a way around (two corners) that left point 1 out (inside the
    cylinder): before point 2 is reached the rest is points 1..3 (the one left out back
    in), after it point 3 only."""
    pts = [LatLon(lat=float(i), lon=0.0) for i in range(4)]
    run = SurveyRun(new_id(), pts, 40.0, 5.0)
    corner = LatLon(lat=1.5, lon=1.0)
    run.waypoints = [pts[0], corner, corner, pts[2], pts[3]]
    run.origin = [0, None, None, 2, 3]
    run.next_index = 2  # on the way around
    assert run.pattern_rest() == pts[1:] and run.detours()
    run.next_index = 4
    assert run.pattern_rest() == pts[3:] and not run.detours()


async def test_a_straighten_keeps_the_way_around_a_cylinder_still_in_force() -> None:
    """Two cylinders on the search, one lifted: the rest goes around the other still."""
    async with rig() as r:
        lifted = keepout(0, 120, 40)
        cmd, _ = await _search_around(r, lifted)
        kept = keepout(-45, 80, 8)  # on the last leg
        install(r, lifted, kept)
        await r.run(_SETTLE_S)
        assert any(math.dist(p, centre(kept)) < 25 for p in corners(run_of(r))), "around both"
        install(r, kept)
        await r.run(0.5)
        assert "executor.keepout_straighten" in r.up.kinds(), r.up.kinds()
        run = run_of(r)
        a, b = centre(lifted), centre(kept)
        assert corners(run), "still around the one in force"
        assert all(math.dist(p, b) < math.dist(p, a) for p in corners(run)), corners(run)
        start = len(r.fc.track)
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 600)
        track = [PLANE.xy(lat, lon) for lat, lon, *_ in r.fc.track[start:]]
        assert xy_route_violation(track, circles([kept], PLANE)) is None


async def test_a_straighten_the_gate_refuses_is_not_flown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shorter rest after a lift goes through the whole gate: refused (here for the
    battery), the search flies on as it is and says why."""
    async with rig() as r:
        cmd, _ = await _search_around(r, keepout(0, 120, 40))
        before = list(run_of(r).waypoints)

        def refused(req: GateRequest, ctx: GateContext) -> GateDecision:
            return evaluate(req, ctx).model_copy(
                update={"decision": "REJECT", "reason_codes": ["ROUTE_BATTERY"]}
            )

        monkeypatch.setattr(executor_module, "evaluate", refused)
        install(r)
        await r.run(0.5)
        kinds = r.up.kinds()
        assert "executor.keepout_straighten" not in kinds, kinds
        refusals = [d for t, _, d in r.up.events if t == "executor.keepout_straighten_refused"]
        assert [d["codes"] for d in refusals] == [["ROUTE_BATTERY"]], refusals
        assert run_of(r).waypoints == before
        assert r.ex.state == S.SURVEY and "superseded" not in r.up.stages(cmd.command_id)


async def test_a_lift_with_a_new_cylinder_on_the_pattern_rest_goes_around_the_new_one() -> None:
    """One cylinder goes as another comes where only the pattern runs (inside the first,
    off its way around): the shorter rest is planned around the new one."""
    async with rig() as r:
        first = keepout(0, 120, 40)
        cmd, _ = await _search_around(r, first)
        # On the leg at east 15, inside the first and off its way around (east 38 to 45).
        second = keepout(15, 120, 4)
        assert math.dist(centre(second), centre(first)) < 40
        assert all(math.dist(p, centre(second)) > 4 for p in corners(run_of(r)))
        install(r, second)
        await r.run(0.5)
        kinds = r.up.kinds()
        assert "executor.keepout_straighten" in kinds, kinds
        run = run_of(r)
        here = PLANE.xy(r.fc.lat, r.fc.lon)
        left = [here, *(PLANE.xy(p.lat, p.lon) for p in run.waypoints[run.next_index :])]
        assert xy_route_violation(left, circles([second], PLANE)) is None
        start = len(r.fc.track)
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 600)
        track = [PLANE.xy(lat, lon) for lat, lon, *_ in r.fc.track[start:]]
        assert xy_route_violation(track, circles([second], PLANE)) is None


def test_a_resume_on_a_way_around_goes_on_from_the_pattern() -> None:
    """Held on a way around: the resume point is the pattern's rest (the point the way
    around left out back in), so a resume plans its own way around the cylinders then."""
    pts = [LatLon(lat=float(i), lon=0.0) for i in range(4)]
    run = SurveyRun(new_id(), pts, 40.0, 5.0)
    corner = LatLon(lat=1.5, lon=1.0)
    run.waypoints = [pts[0], corner, corner, pts[2], pts[3]]
    run.origin = [0, None, None, 2, 3]
    run.next_index = 2
    vehicle = SimpleNamespace(lat=1.2, lon=0.8)
    rp = run.resume_point(vehicle)  # type: ignore[arg-type]
    assert rp is not None
    assert rp.waypoints == pts and rp.base_index == rp.next_index == 1
    run.next_index = 4  # past the way around: as before
    rp = run.resume_point(vehicle)  # type: ignore[arg-type]
    assert rp is not None and rp.waypoints == run.waypoints and rp.next_index == 4


BRAKE_MS2 = 2.5  # ArduCopter's LOITER braking (LOIT_BRK_ACCEL)


def _xy(r: Rig) -> Xy:
    return PLANE.xy(r.fc.lat, r.fc.lon)


def _track(r: Rig, start: int) -> list[Xy]:
    return [PLANE.xy(lat, lon) for lat, lon, *_ in r.fc.track[start:]]


@pytest.mark.parametrize("lift_after_s", [0.1, 0.3, 0.8])
async def test_a_cylinder_lifted_while_the_search_stops_first_does_not_cut_the_stop(
    lift_after_s: float,
) -> None:
    """#161 review: the search goes around a cylinder further on; then one comes 17 m
    ahead at 8 m/s, too close to go around (it stops first), and the one further on is
    lifted while it brakes. The lift must not straighten the route from where the drone
    was: that cancels the stop and flies into the close one. The plan from the stop takes
    the lift into account."""
    # On a later leg, between pattern points: lifted, the rest is about 80 m shorter.
    far = keepout(-15, 120, 36)
    async with rig(fc_options={"accel_ms2": BRAKE_MS2}) as r:
        await r.install(charter(allowed_skills=ONBOARD_SKILLS))
        await r.takeoff()
        cmd = await r.send(
            r.order("survey_sector", {"polygon": BOX, "spacing_m": 30, "speed_ms": 8}), 1
        )
        install(r, far)
        await r.run(_SETTLE_S)
        assert run_of(r).detours()

        def on_a_long_leg() -> bool:  # at full speed along it, 60 m or more to go
            ahead = r.fc.missions[MISSION][r.fc.mission_current]
            to, here = PLANE.xy(ahead.x / 1e7, ahead.y / 1e7), _xy(r)
            d, speed = math.dist(here, to), math.hypot(*r.fc.vel)
            along = (r.fc.vel[0] * (to[0] - here[0]) + r.fc.vel[1] * (to[1] - here[1])) / d
            return r.fc.mission_current >= 3 and d > 60 and speed > 7.9 and along > 0.999 * speed

        await r.until(on_a_long_leg, 300)
        here, (ve, vn) = _xy(r), r.fc.vel
        speed = math.hypot(ve, vn)
        assert stop_m(speed) > 17 > speed**2 / (2 * BRAKE_MS2) + speed * 0.5
        close = keepout(here[0] + 37 * ve / speed, here[1] + 37 * vn / speed, 20)
        start = len(r.fc.track)
        install(r, close, far)
        await r.run(lift_after_s)
        reroutes = [d for t, _, d in r.up.events if t == "executor.keepout_reroute"]
        assert reroutes[-1]["stop_first"] is True
        assert run_of(r).detours()
        install(r, close)  # the far one goes while it brakes
        ends = ("executor.keepout_reroute_stopped", "executor.keepout_straighten")
        await r.until(lambda: any(e in r.up.kinds() for e in ends), 60)
        kinds = r.up.kinds()
        assert "executor.keepout_straighten" not in kinds, kinds
        assert "executor.keepout_reroute_stopped" in kinds, kinds
        assert xy_route_violation(_track(r, start), circles([close], PLANE)) is None
        # Planned from the stop around the close one only: nothing goes around the far one.
        assert all(
            math.dist(p, centre(close)) < math.dist(p, centre(far)) for p in corners(run_of(r))
        ), corners(run_of(r))
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 600)
        assert xy_route_violation(_track(r, start), circles([close], PLANE)) is None
