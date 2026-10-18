"""SC-PATH: the drone flies a path the pilot drew, on ArduCopter SITL.

* SC_PATH: the pilot draws pa-1 (5 points) and writes "Flyg vägen pa-1."; the AI
  proposes follow_path by the path's name, the pilot approves the card, every point is
  passed within 3 m and the drone then hovers at the last one.
* SC_PATH_stop: STOP halfway along the path: BRAKE within 1 s, the speed at least halved
  after 2 s, standing still within 5 s and holding there for 20 s (as SC02); after the
  pilot's Resume the AI flies the rest of the path and the drone hovers at its end.

The distance to every point and the brake times are measured values in the test's
``runs/<run>/result.json``.
"""

from __future__ import annotations

import asyncio
import math
import sys
import time
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_rig import (  # pyright: ignore[reportMissingImports]
    H,
    Scenario,
    fresh_db,
    migrated_db,
    passwords,
    scenario,
    sim,
)

from sd_safety.targets import Plane, distance_m
from sd_sim.report import measure

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.brain]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

# Along a "trail" north of home, inside charter_test and its fence margin (east/north m).
PATH_ENU = [(0.0, 60.0), (60.0, 110.0), (100.0, 170.0), (40.0, 220.0), (-60.0, 180.0)]
ALT_M = 30.0
POINT_LIMIT_M = 3.0
BRAKE_LIMIT_S = 1.0  # the button -> BRAKE on the flight controller
STILL_LIMIT_S = 5.0  # standing still, the same limit as SC02
HOVER_S = 10.0


async def draw_and_fly(sc: Scenario) -> dict[str, Any]:
    """The pilot draws pa-1 and asks for it; the AI's card is approved. The order."""
    line = [[sc.at(e, n)[1], sc.at(e, n)[0]] for e, n in PATH_ENU]
    res = await sc.r.panel.post(
        f"/api/v1/missions/{sc.mid}/annotations",
        json={
            "kind": "path",
            "label": "pa-1",
            "geometry": {"type": "LineString", "coordinates": line},
        },
        headers=H,
    )
    assert res.status_code == 201, res.text
    await sc.say("Flyg vägen pa-1.")
    cmd = await sc.wait_command("follow_path", "pending_approval", 30, issuer_kind="brain")
    # The AI named the path; the points it asks to fly are the drawn ones (to the
    # centimetre: the database stores them with nine decimals).
    asked = [(p["lon"], p["lat"]) for p in cmd["params"]["points"]]
    assert len(asked) == len(line)
    for (lon, lat), (dlon, dlat) in zip(asked, line, strict=True):
        assert distance_m(lat, lon, dlat, dlon) < 0.01
    res = await sc.r.approve(str(cmd["approval_id"]))
    assert res.status_code == 200, res.text
    return cmd


def closest_m(sc: Scenario, since: float) -> list[float]:
    """For each point of the path, how close the flown track came (armed, since ``since``)."""
    plane = Plane(*sc.r.home)
    xy = [plane.xy(s.lat, s.lon) for s in sc.track if s.t >= since and s.armed]
    return [min(math.dist(p, q) for p in xy) for q in PATH_ENU]


@pytest.mark.budget_s(540)
async def test_sc_path_points_within_3_m_then_hover(
    sim: Any, migrated_db: Any, tmp_path: Path
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne_with_ai("SC_PATH")
        t0 = time.monotonic()
        cmd = await draw_and_fly(sc)
        await sc.wait_command("follow_path", "running", 30, issuer_kind="brain")
        await sc.wait_command("follow_path", "done", 240, issuer_kind="brain")
        errors = closest_m(sc, t0)
        for i, err in enumerate(errors, start=1):
            measure(f"vägpunkt {i} av {len(errors)} nådd", err, POINT_LIMIT_M, unit="m")
        measure("sämsta vägpunkten", max(errors), POINT_LIMIT_M, unit="m")
        assert max(errors) <= POINT_LIMIT_M, errors
        # It hovers at the last point: LOITER, still, at the path's height.
        end = sc.at(*PATH_ENU[-1])
        await sc.until(lambda: sc.tel.groundspeed_ms < 0.5, 15, "still at the end")
        await sc.hold_for(
            HOVER_S,
            lambda: (
                sc.mode() == "LOITER"
                and distance_m(*end, sc.tel.lat, sc.tel.lon) <= POINT_LIMIT_M
                and abs(sc.tel.alt_rel_home_m - ALT_M) <= 2.0
            ),
            "svävar vid vägens sista punkt",
        )
        # The ground's gate asked for the click, then let the approved order go.
        gates = await sc.gate(cmd["id"])
        assert sorted(g["decision"] for g in gates if g["side"] == "ground") == [
            "ALLOW",
            "PENDING_APPROVAL",
        ], gates
        await sc.check_invariants()


@pytest.mark.budget_s(600)
async def test_sc_path_stop_halfway_brakes_at_once(
    sim: Any, migrated_db: Any, tmp_path: Path
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne_with_ai("SC_PATH")
        await draw_and_fly(sc)
        await sc.wait_command("follow_path", "running", 30, issuer_kind="brain")
        # Halfway: near the third point, flying at speed.
        plane = Plane(*sc.r.home)
        await sc.until(
            lambda: (
                math.dist(plane.xy(sc.tel.lat, sc.tel.lon), PATH_ENU[2]) < 20
                and sc.tel.groundspeed_ms > 3.0
            ),
            180,
            "halfway along the path at speed",
        )
        speed = sc.tel.groundspeed_ms
        pressed = time.monotonic()
        res = await sc.press("stop")
        assert res.status_code == 202, res.text
        await sc.until(lambda: sc.tel.groundspeed_ms < 0.5, 10, "still after STOP")
        await asyncio.sleep(1.0)
        # The test rig's own channel to the flight controller: mode and speed per sample.
        after = [x for x in sc.track if x.t >= pressed]
        brake_s = next(x.t for x in after if x.mode == "BRAKE") - pressed
        at_2_s = next(x.groundspeed_ms for x in after if x.t - pressed >= 2.0)
        still_s = next(x.t for x in after if x.groundspeed_ms < 0.5) - pressed
        measure("STOPP på vägen: BRAKE", brake_s, BRAKE_LIMIT_S)
        measure("STOPP på vägen: fart efter 2 s", at_2_s, speed / 2, unit="m/s")
        measure("STOPP på vägen: stillastående", still_s, STILL_LIMIT_S)
        assert sc.mode() == "BRAKE", sc.modes()
        assert brake_s <= BRAKE_LIMIT_S and at_2_s <= speed / 2, (brake_s, at_2_s, speed)
        assert still_s <= STILL_LIMIT_S, still_s
        # Holds where it stopped, with SC02's limits: 20 s within 3 m and 1 m in height.
        here = (sc.tel.lat, sc.tel.lon, sc.tel.alt_rel_home_m)
        await sc.hold_for(
            20.0,
            lambda: (
                distance_m(here[0], here[1], sc.tel.lat, sc.tel.lon) <= 3.0
                and abs(sc.tel.alt_rel_home_m - here[2]) <= 1.0
            ),
            "står kvar där vägen stoppades",
        )

        # The pilot's Resume: the AI's resume flies the rest of the path, as a path
        # (TRANSIT), and the drone hovers at its last point.
        before = [c["id"] for c in await sc.commands()]
        res = await sc.press("resume")
        assert res.status_code == 202, res.text
        await sc.wait_command("resume", "running", 30, issuer_kind="brain", besides=before)

        async def flies_on_as_a_path() -> bool:
            return (await sc.onboard_state())["executor_state"] == "TRANSIT"

        await sc.until(flies_on_as_a_path, 5, "Fortsätt: vägen flygs vidare i TRANSIT")
        await sc.wait_command("resume", "done", 180, issuer_kind="brain", besides=before)
        end = sc.at(*PATH_ENU[-1])
        rest = closest_m(sc, pressed)[3:]  # the points still left when it stopped
        measure(
            "Fortsätt efter STOPP: sämsta återstående vägpunkt", max(rest), POINT_LIMIT_M, unit="m"
        )
        assert max(rest) <= POINT_LIMIT_M, rest
        await sc.until(lambda: sc.tel.groundspeed_ms < 0.5, 15, "still at the end after Resume")
        assert sc.mode() == "LOITER", sc.modes()
        assert distance_m(*end, sc.tel.lat, sc.tel.lon) <= POINT_LIMIT_M

        res = await sc.press("return_home")
        assert res.status_code == 202
        await sc.until(lambda: not sc.tel.armed, 240, "landed at home")
        await sc.check_invariants()
