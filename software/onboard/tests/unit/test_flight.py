"""A whole flight against the fake flight controller: takeoff, goto, survey, home."""

from __future__ import annotations

from onboard_helpers import PLANE, at, rig, square

from sd_geo import coverage_fraction
from sd_schemas.executor import ExecutorState as S


async def test_demo_flight_on_the_fake() -> None:
    async with rig() as r:
        takeoff = await r.takeoff(40)
        assert r.up.stages(takeoff.command_id)[-1] == "done"
        assert abs(r.fc.alt_rel - 40) <= 2
        lat, lon = at(100, 50)
        goto = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 8}))
        assert r.ex.state == S.TRANSIT
        await r.until(lambda: r.ex.state == S.HOLD, 120)
        assert r.up.stages(goto.command_id)[-1] == "done"
        assert r.fc.mode.name == "LOITER"
        box = square(40, east=100, north=50)
        start = len(r.fc.track)
        survey = await r.send(
            r.order("survey_sector", {"polygon": box, "spacing_m": 30, "speed_ms": 7}), 1.0
        )
        assert r.ex.state == S.SURVEY, r.up.acks
        await r.until(lambda: r.ex.state == S.HOLD, 300, step=0.05)
        assert r.up.stages(survey.command_id)[-1] == "done"
        corners = [PLANE.xy(lat, lon) for lon, lat in box["coordinates"][0][:-1]]
        track = [(*PLANE.xy(la, lo), alt) for la, lo, alt in r.fc.track[start:]]
        assert coverage_fraction(corners, track, 40, 30) >= 0.95
        home = await r.send(r.order("return_home"))
        assert r.ex.state == S.RTL
        await r.until(lambda: r.ex.state == S.LANDED, 300, step=0.05)
        assert r.up.stages(home.command_id)[-1] == "done"
        assert r.fc.distance_from_home_m() <= 3
        arms = [
            m for m in r.vehicle.sent if m.get_type() == "COMMAND_LONG" and int(m.command) == 400
        ]
        assert len(arms) == 1
        await r.run(6)
        assert r.ex.state == S.PREFLIGHT
