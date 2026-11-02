"""A search around keep-out cylinders that come and go: the rest is planned from the
search pattern, so a way around a cylinder that is gone is not flown on (SITL SK03, where
every rejected candidate's way around stayed in the search and it ran out of time)."""

from __future__ import annotations

import math

from onboard_helpers import ONBOARD_SKILLS, PLANE, Rig, at, charter, rig, square

from sd_onboard.executor.keepout_reflex import survey_run
from sd_onboard.skills.survey_sector import SurveyRun
from sd_safety.targets import Xy
from sd_schemas.base import LatLon, new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.keepout import Keepout
from sd_schemas.link import KeepoutInstall, SkillCommand

BOX = square(60, east=0, north=120)


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
    await r.run(0.5)
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
