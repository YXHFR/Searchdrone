"""A search around keep-out cylinders that come and go: the rest is planned from the
search pattern, so a way around a cylinder that is gone is not flown on (SITL SK03, where
every rejected candidate's way around stayed in the search and it ran out of time)."""

from __future__ import annotations

import math
from collections.abc import Callable
from types import SimpleNamespace

import pytest
from onboard_helpers import ONBOARD_SKILLS, PLANE, Rig, at, charter, rig, square

from sd_mav.mission import MISSION
from sd_mav.modes import CopterMode
from sd_onboard.executor import executor as executor_module
from sd_onboard.executor.keepout_reflex import path_clear, stop_m, survey_run
from sd_onboard.skills.survey_sector import BLOCKED_MIN_M, SurveyRun
from sd_safety import evaluate
from sd_safety.keepout import circles, xy_route_violation
from sd_safety.targets import Xy
from sd_schemas.base import LatLon, new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.gate import GateContext, GateDecision, GateRequest
from sd_schemas.keepout import Keepout
from sd_schemas.link import BBox, GeoDetectionMsg, KeepoutInstall, SkillCommand

BOX = square(60, east=0, north=120)
# Between one change and the next: short, so a change often comes while the reroute
# before it is still switching to AUTO (the executor's own mode, never a takeover).
_GAP_S = 0.3
_SETTLE_S = 2.0  # long enough for a reroute to fly


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
    await r.run(_GAP_S)
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


# Cylinders by the side of the search: the way around cuts the corners of the legs in
# them, and is shorter than the pattern (SITL SK10, #234).
_CORNER_CUTS = [(55, 120, 30), (45, 150, 25), (55, 95, 35)]


@pytest.mark.parametrize(("east", "north", "radius"), _CORNER_CUTS)
async def test_a_lifted_cylinder_the_way_around_cut_a_corner_of_is_searched_again(
    east: float, north: float, radius: float
) -> None:
    """The candidate is rejected (its cylinder goes) after the way around left out the
    legs' ends in it: the search flies them after all, though that is the longer way, and
    ends. Before, only a rest 30 m shorter replaced the way around, so the area by a
    rejected decoy (in SK10 the person's) was never searched."""
    async with rig() as r:
        k = keepout(east, north, radius)
        cmd, _ = await _search_around(r, k)
        assert sum(m for _, m in run_of(r).blocked) >= BLOCKED_MIN_M
        start = len(r.fc.track)
        install(r)
        await r.run(0.5)
        assert "executor.keepout_straighten" in r.up.kinds(), r.up.kinds()
        assert not run_of(r).detours() and not corners(run_of(r))
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 600)
        track = [PLANE.xy(lat, lon) for lat, lon, *_ in r.fc.track[start:]]
        assert any(math.dist(p, centre(k)) < radius - BLOCKED_MIN_M for p in track)


@pytest.mark.parametrize(("east", "north", "radius"), [(0, 120, 40), *_CORNER_CUTS])
async def test_a_cylinder_handed_to_the_ground_keeps_the_way_around(
    east: float, north: float, radius: float
) -> None:
    """The same cylinder under another id (the drone's own handed to the ground): nothing
    shorter and nothing searched again, so the search flies on as it is, no stop."""
    async with rig() as r:
        k = keepout(east, north, radius)
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
        await r.run(_GAP_S)
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


@pytest.mark.parametrize("why", ["gate", "path"])
async def test_a_refused_straighten_is_tried_again_and_told_once(
    monkeypatch: pytest.MonkeyPatch, why: str
) -> None:
    """A straighten refused by the gate or by ``path_clear`` is not given up on: tried
    again while the way around is flown, told once for the same reason, and flown once
    allowed, so the lifted area is still searched."""
    async with rig() as r:
        cmd, _ = await _search_around(r, keepout(0, 120, 40))
        before = list(run_of(r).waypoints)
        refuse = True
        code = "ROUTE_BATTERY" if why == "gate" else "KEEPOUT_VIOLATION"
        real_evaluate, real_clear = evaluate, path_clear

        def gate(req: GateRequest, ctx: GateContext) -> GateDecision:
            d = real_evaluate(req, ctx)
            if refuse and why == "gate":
                return d.model_copy(update={"decision": "REJECT", "reason_codes": [code]})
            return d

        def clear(*a: object, **kw: object) -> bool:
            return False if refuse and why == "path" else real_clear(*a, **kw)  # type: ignore[arg-type]

        monkeypatch.setattr(executor_module, "evaluate", gate)
        monkeypatch.setattr(executor_module, "path_clear", clear)
        install(r)
        await r.run(executor_module._STRAIGHTEN_RETRY_MS / 1000 * 2.5)  # several tries
        refusals = [d for t, _, d in r.up.events if t == "executor.keepout_straighten_refused"]
        assert [d["codes"] for d in refusals] == [[code]], refusals
        assert "executor.keepout_straighten" not in r.up.kinds()
        assert run_of(r).waypoints == before and r.ex.state == S.SURVEY
        refuse = False
        await r.run(executor_module._STRAIGHTEN_RETRY_MS / 1000 + 0.5)
        assert r.up.kinds().count("executor.keepout_straighten") == 1, r.up.kinds()
        assert not run_of(r).detours()
        assert len([t for t, _, _ in r.up.events if t.endswith("straighten_refused")]) == 1
        assert "superseded" not in r.up.stages(cmd.command_id)


async def test_two_refusal_reasons_that_take_turns_are_told_once_each() -> None:
    """A straighten refused by turns for a cylinder on the way and by the gate: each
    reason is told once, not again on every retry."""
    async with rig() as r:
        await _search_around(r, keepout(0, 120, 40))
        turns = [("KEEPOUT_VIOLATION",), ("ROUTE_BATTERY",)]
        tries: list[tuple[str, ...]] = []

        def refusal(*_: object, **__: object) -> tuple[str, ...]:
            tries.append(turns[len(tries) % 2])
            return tries[-1]

        r.ex._route_refusal = refusal  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]
        install(r)
        await r.run(executor_module._STRAIGHTEN_RETRY_MS / 1000 * 4.5)
        assert len(tries) >= 4, tries
        refusals = [d for t, _, d in r.up.events if t == "executor.keepout_straighten_refused"]
        assert [d["codes"] for d in refusals] == [list(c) for c in turns], refusals


async def test_a_refused_straighten_does_not_carry_over_to_the_next_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A search whose straighten was refused is replaced by a new order: the refusal's
    retry time and the reasons told belong to the old order. The new order's own refused
    straighten is tried at once and told, though for the same reason."""
    async with rig() as r:
        first, _ = await _search_around(r, keepout(0, 120, 40))

        def refused(req: GateRequest, ctx: GateContext) -> GateDecision:
            return evaluate(req, ctx).model_copy(
                update={"decision": "REJECT", "reason_codes": ["ROUTE_BATTERY"]}
            )

        monkeypatch.setattr(executor_module, "evaluate", refused)
        install(r)
        await r.run(0.5)
        assert r.up.kinds().count("executor.keepout_straighten_refused") == 1, r.up.kinds()
        monkeypatch.setattr(executor_module, "evaluate", evaluate)
        second = await r.send(
            r.order("survey_sector", {"polygon": BOX, "spacing_m": 30, "speed_ms": 8}), 0.0
        )
        # Gone as the new order starts, not only once the reflex next runs for it.
        assert r.ex.active is not None and r.ex.active.command_id == second.command_id
        assert r.ex._lift_retry_ms is None  # pyright: ignore[reportPrivateUsage]
        assert not r.ex._lift_refused  # pyright: ignore[reportPrivateUsage]
        assert "superseded" in r.up.stages(first.command_id)
        install(r, keepout(0, 120, 40))
        await r.run(_GAP_S)
        assert r.up.kinds().count("executor.keepout_reroute") == 2, r.up.kinds()
        monkeypatch.setattr(executor_module, "evaluate", refused)
        install(r)
        await r.run(0.5)
        refusals = [d for t, _, d in r.up.events if t == "executor.keepout_straighten_refused"]
        assert [d["command_id"] for d in refusals] == [
            str(first.command_id),
            str(second.command_id),
        ], refusals


def _run_blocked_by(*ks: Keepout) -> SurveyRun:
    ends = [LatLon(lat=lat, lon=lon) for lat, lon in (at(-100, 0), at(100, 0))]
    run = SurveyRun(new_id(), ends, 40.0, 8.0, skill="survey_sector")
    run.blocked = [(k, 0.0) for k in ks]
    return run


def test_regained_m_counts_only_lifted_cylinders_and_never_below_zero() -> None:
    """``regained_m``: a cylinder still in force regains nothing; a route that searches
    less of a lifted one than before counts zero, not less."""
    lifted, kept = keepout(-40, 0, 10), keepout(40, 0, 10)
    run = _run_blocked_by(lifted, kept)
    around: list[Xy] = [(-100, 0), (-60, 0), (-40, 20), (-20, 0), (20, 0), (40, 20), (60, 0)]
    straight: list[Xy] = [(-100, 0), (100, 0)]
    assert run.regained_m([kept], PLANE, around, straight) == pytest.approx(20.0, abs=0.1)
    assert run.regained_m([lifted, kept], PLANE, around, straight) == 0.0
    assert run.regained_m([kept], PLANE, straight, around) == 0.0


def test_regained_m_counts_a_stretch_inside_two_lifted_cylinders_once() -> None:
    """Two lifted cylinders that overlap: the search inside both counts once."""
    run = _run_blocked_by(keepout(-5, 0, 10), keepout(5, 0, 10))
    around: list[Xy] = [(-100, 0), (-30, 0), (-20, 30), (20, 30), (30, 0), (100, 0)]
    straight: list[Xy] = [(-100, 0), (100, 0)]
    assert run.regained_m([], PLANE, around, straight) == pytest.approx(30.0, abs=0.1)


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


def _straightens(monkeypatch: pytest.MonkeyPatch) -> list[Xy]:
    """Where the executor tries a shorter rest after a lift (``_reroute`` with ``here``)."""
    straightens: list[Xy] = []
    reroute = executor_module.Executor._reroute

    def spy(self: executor_module.Executor, *a: object, **kw: object) -> bool:
        if kw.get("here") is not None:
            straightens.append(kw["here"])  # type: ignore[arg-type]
        return reroute(self, *a, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(executor_module.Executor, "_reroute", spy)
    return straightens


@pytest.mark.parametrize("lift_after_s", [0.1, 0.3, 0.8])
async def test_a_cylinder_lifted_while_the_search_stops_first_does_not_cut_the_stop(
    lift_after_s: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#161 review: the search goes around a cylinder further on; then one comes 17 m
    ahead at 8 m/s, too close to go around (it stops first), and the one further on is
    lifted while it brakes. The lift must not straighten the route from where the drone
    was: that cancels the stop and flies into the close one. The plan from the stop takes
    the lift into account. The lift is not dropped: tried once the stop has planned."""
    straightens = _straightens(monkeypatch)
    # On a later leg, between pattern points: lifted, the rest is about 80 m shorter.
    far = keepout(-15, 120, 36)
    async with rig(fc_options={"accel_ms2": BRAKE_MS2}) as r:
        await r.install(charter(allowed_skills=ONBOARD_SKILLS))
        await r.takeoff()
        cmd = await r.send(
            r.order("survey_sector", {"polygon": BOX, "spacing_m": 30, "speed_ms": 8}), 1
        )
        install(r, far)
        await r.run(_GAP_S)
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
        assert not straightens
        await r.run(_GAP_S)
        assert straightens  # held back while it stopped, tried after
        assert xy_route_violation(_track(r, start), circles([close], PLANE)) is None
        # Planned from the stop around the close one only: nothing goes around the far one.
        assert all(
            math.dist(p, centre(close)) < math.dist(p, centre(far)) for p in corners(run_of(r))
        ), corners(run_of(r))
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 600)
        assert xy_route_violation(_track(r, start), circles([close], PLANE)) is None


WIDE = square(120, east=0, north=60)  # legs of 240 m
_ARDUCOPTER = {"accel_ms2": BRAKE_MS2}


def _ahead(r: Rig, m: float, radius: float) -> Keepout:
    """A cylinder ``m`` ahead of the drone along its velocity."""
    here, (ve, vn) = _xy(r), r.fc.vel
    speed = math.hypot(ve, vn)
    return keepout(here[0] + m * ve / speed, here[1] + m * vn / speed, radius)


def _on_route(r: Rig, radius: float) -> Keepout:
    """A cylinder on the search's route, halfway along the second leg still to fly."""
    run = run_of(r)
    a, b = (PLANE.xy(p.lat, p.lon) for p in run.waypoints[run.next_index : run.next_index + 2])
    return keepout((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, radius)


async def _searching(r: Rig) -> SkillCommand:
    """A search on a long leg at full speed, 150 m or more to go."""
    await r.install(charter(allowed_skills=ONBOARD_SKILLS))
    await r.takeoff()
    cmd = await r.send(
        r.order("survey_sector", {"polygon": WIDE, "spacing_m": 60, "speed_ms": 8}), 1
    )

    def on_a_long_leg() -> bool:
        ahead = r.fc.missions[MISSION][r.fc.mission_current]
        to = PLANE.xy(ahead.x / 1e7, ahead.y / 1e7)
        speed = math.hypot(*r.fc.vel)
        return r.fc.mission_current >= 1 and math.dist(_xy(r), to) > 150 and speed > 7.9

    await r.until(on_a_long_leg, 300)
    return cmd


async def _cancel_auto_on_the_way(r: Rig, change: Callable[[], None], kind: str) -> None:
    """``change`` as the HEARTBEAT showing the executor's switch to AUTO comes, before
    the switch has seen it: the reroute ``change`` starts (``kind``) cancels the switch."""
    await r.until(
        lambda: r.ex.auth.pending_mode == CopterMode.AUTO and r.fc.mode == CopterMode.AUTO, 10
    )
    r.fc.heartbeat_held = True
    assert r.ex.vehicle.telemetry.fc_custom_mode != int(CopterMode.AUTO)
    n = r.up.kinds().count(kind)
    change()
    r.fc.release_heartbeat()
    await r.until(lambda: r.up.kinds().count(kind) > n, 1)


def _never_taken(r: Rig) -> None:
    """Every mode change here was the executor's own: it is never HELD by one, and it
    steers the search on."""
    kinds = r.up.kinds()
    assert "executor.control_taken" not in kinds, kinds
    assert r.ex.state == S.SURVEY, (r.ex.state, r.ex.held_by, r.fc.mode)


async def test_reroutes_in_quick_succession_keep_steering() -> None:
    """Three cylinders on the route, each installed while the reroute for the one before
    is switching to AUTO: every reroute cancels the one before after its AUTO went out.
    That late AUTO is the executor's own, not a takeover; the search steers on and keeps
    out of a cylinder that comes after."""
    async with rig(fc_options=_ARDUCOPTER) as r:
        cmd = await _searching(r)
        start = len(r.fc.track)
        put = [_ahead(r, 50, 6)]
        install(r, *put)
        for _ in range(2):
            await r.until(lambda: r.ex.active is not None and run_of(r).detours(), 5)
            put.append(_on_route(r, 6))
            await _cancel_auto_on_the_way(r, lambda: install(r, *put), "executor.keepout_reroute")
        await r.run(_SETTLE_S)
        assert r.up.kinds().count("executor.keepout_reroute") == 3, r.up.kinds()
        _never_taken(r)
        later = _ahead(r, 60, 8)
        install(r, *put, later)
        await r.run(_SETTLE_S)
        _never_taken(r)
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 600)
        assert "executor.control_taken" not in r.up.kinds(), r.up.kinds()
        assert xy_route_violation(_track(r, start), circles([*put, later], PLANE)) is None


async def test_a_lift_while_a_reroute_switches_to_auto_keeps_steering() -> None:
    """#161 round 2: a cylinder comes on the route (a way around it, no stop) and is
    lifted while the reroute's switch to AUTO is on the way. The straighten cancels that
    reroute; the late AUTO is the executor's own, so the search steers on and goes
    around a cylinder that comes after."""
    async with rig(fc_options=_ARDUCOPTER) as r:
        cmd = await _searching(r)
        start = len(r.fc.track)
        install(r, _ahead(r, 115, 60))
        await _cancel_auto_on_the_way(r, lambda: install(r), "executor.keepout_straighten")
        await r.run(_SETTLE_S)
        _never_taken(r)
        later = _ahead(r, 60, 8)
        install(r, later)
        await r.run(_SETTLE_S)
        _never_taken(r)
        await r.until(lambda: "done" in r.up.stages(cmd.command_id), 600)
        assert "executor.control_taken" not in r.up.kinds(), r.up.kinds()
        assert xy_route_violation(_track(r, start), circles([later], PLANE)) is None


def _detection(east: float, north: float) -> GeoDetectionMsg:
    lat, lon = at(east, north)
    return GeoDetectionMsg(
        detection_id=new_id(),
        bbox=BBox(x=10, y=10, w=20, h=40),
        score=0.9,
        lat=lat,
        lon=lon,
        sigma95_m=5.0,
        px_size=40,
        crop_inline=False,
        label="person",
    )


async def test_a_merged_cylinder_of_the_drones_own_is_not_a_lift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same person seen again: the drone's own cylinder grows into one with a new id.
    The old id is not a cylinder lifted, so the search tries no shorter rest for it."""
    straightens = _straightens(monkeypatch)
    async with rig(fc_options=_ARDUCOPTER) as r:
        await _searching(r)
        k = _ahead(r, 80, 1)
        east, north = centre(k)
        r.ex.on_detections([_detection(east, north)])
        await r.run(_SETTLE_S)
        assert r.up.kinds().count("executor.keepout_reroute") == 1, r.up.kinds()
        assert run_of(r).detours()
        r.ex.on_detections([_detection(east + 6, north)])
        await r.run(0.5)
        assert "executor.keepout_merged" in r.up.kinds(), r.up.kinds()
        assert straightens == [], straightens
        _never_taken(r)
