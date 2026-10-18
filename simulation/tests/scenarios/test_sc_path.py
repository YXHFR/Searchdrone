"""SC-PATH: the drone flies a path the pilot drew, on ArduCopter SITL.

* SC_PATH: the pilot draws pa-1 (5 points) and writes "Flyg vägen pa-1."; the AI
  proposes follow_path by the path's name, the pilot approves the card, every point is
  passed within 3 m and the drone then hovers at the last one.
* SC_PATH_stop: STOP halfway along the path: standing still within 2 s, and the path is
  not taken up again without the pilot.

The distance to every point is written to ``runs/<run>/result.json`` (one criterion per
point) and ``path_points.json``.
"""

from __future__ import annotations

import json
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
from sd_sim.report import Report
from sd_sim.util import new_run_dir

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.brain]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

# Along a "trail" north of home, inside charter_test and its fence margin (east/north m).
PATH_ENU = [(0.0, 60.0), (60.0, 110.0), (100.0, 170.0), (40.0, 220.0), (-60.0, 180.0)]
ALT_M = 30.0
POINT_LIMIT_M = 3.0
STILL_LIMIT_S = 2.0
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


def report(sc: Scenario, name: str, instance: int) -> Report:
    run_dir = new_run_dir(instance, name)
    return Report(name, run_dir, instance, "builtin", "scripted")


@pytest.mark.budget_s(540)
async def test_sc_path_points_within_3_m_then_hover(
    sim: Any, migrated_db: Any, tmp_path: Path, sim_instance: int
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        rep = report(sc, "SC_PATH", sim_instance)
        report_dir = rep.run_dir
        try:
            await sc.airborne_with_ai("SC_PATH")
            t0 = time.monotonic()
            cmd = await draw_and_fly(sc)
            await sc.wait_command("follow_path", "running", 30, issuer_kind="brain")
            await sc.wait_command("follow_path", "done", 240, issuer_kind="brain")
            errors = closest_m(sc, t0)
            for i, err in enumerate(errors, start=1):
                rep.check(
                    f"SC_PATH.{i}",
                    f"Punkt {i} av {len(errors)} nås inom {POINT_LIMIT_M:g} m",
                    err <= POINT_LIMIT_M,
                    measured=round(err, 2),
                    limit=POINT_LIMIT_M,
                    unit="m",
                )
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
                "hovers at the last point",
            )
            rep.check(
                "SC_PATH.hover",
                f"Svävar vid sista punkten i {HOVER_S:g} s (LOITER)",
                True,
                measured=round(distance_m(*end, sc.tel.lat, sc.tel.lon), 2),
                limit=POINT_LIMIT_M,
                unit="m",
            )
            gates = await sc.gate(cmd["id"])
            sides = [(g["side"], g["decision"]) for g in gates]
            assert ("ground", "PENDING_APPROVAL") in sides and ("onboard", "ALLOW") in sides
            (report_dir / "path_points.json").write_text(
                json.dumps(
                    {
                        "points_enu_m": PATH_ENU,
                        "alt_m": ALT_M,
                        "closest_m": [round(e, 2) for e in errors],
                        "limit_m": POINT_LIMIT_M,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            assert max(errors) <= POINT_LIMIT_M, errors
            await sc.check_invariants()
        except Exception as exc:
            rep.error = f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            rep.write()


@pytest.mark.budget_s(480)
async def test_sc_path_stop_halfway_stands_still_within_2_s(
    sim: Any, migrated_db: Any, tmp_path: Path, sim_instance: int
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        rep = report(sc, "SC_PATH_stop", sim_instance)
        try:
            await sc.airborne_with_ai("SC_PATH")
            await draw_and_fly(sc)
            await sc.wait_command("follow_path", "running", 30, issuer_kind="brain")
            # Halfway: past the second point, flying at speed.
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
            res = await sc.press("stop")
            assert res.status_code == 202, res.text
            took = await sc.until(lambda: sc.tel.groundspeed_ms < 0.5, 10, "still after STOP")
            rep.check(
                "SC_PATH_stop.1",
                f"STOPP mitt på vägen: stillastående inom {STILL_LIMIT_S:g} s",
                took <= STILL_LIMIT_S,
                measured=round(took, 2),
                limit=STILL_LIMIT_S,
                unit="s",
                clock="host",
                note=f"fart vid STOPP {speed:.1f} m/s",
            )
            assert sc.mode() == "BRAKE", sc.modes()
            here = (sc.tel.lat, sc.tel.lon)
            await sc.hold_for(
                10.0,
                lambda: distance_m(*here, sc.tel.lat, sc.tel.lon) <= 3.0,
                "stands where it stopped",
            )
            rep.check("SC_PATH_stop.2", "Står kvar där den stoppades i 10 s", True)
            assert took <= STILL_LIMIT_S, took
            res = await sc.press("return_home")
            assert res.status_code == 202
            await sc.until(lambda: not sc.tel.armed, 240, "landed at home")
            await sc.check_invariants()
        except Exception as exc:
            rep.error = f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            rep.write()
