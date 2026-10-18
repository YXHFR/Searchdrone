"""SC07: the whole simulated radio is down (MAVLink to the ground, SD-link, video).

* a, 10 s, holding after a ``goto``: nothing changes in the flight (same mode, within
  3 m); the AI's orders sent every 3 s during the outage (``goto(OUT)`` and ``hold`` on
  the commander API) never run onboard; the ground saw SD-link degraded and MAVLink lost,
  and SD-link is back within 10 s.
* b, 90 s: the flight controller's GCS failsafe flies home within ``FS_GCS_TIMEOUT`` +
  L-FS of the last heartbeat from 250, the onboard computer does not fight it, and it
  lands at home with the radio still down (down for 90 s, longer until it has landed);
  afterwards the vehicle is connected again
  within 10 s and the mission does not resume by itself.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from rig_ground import (  # pyright: ignore[reportMissingImports]
    Watched,
    first_choice_or_proof,
    landed_home,
    scenario_x,
    wait_mode,
)
from scenario_rig import AI_RATE_S, RAN  # pyright: ignore[reportMissingImports]

from sd_mav import CopterMode
from sd_safety.targets import distance_m
from sd_schemas.executor import ExecutorState as S
from sd_sim import expect

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.script("SC07")]

ORDER_EVERY_S = 3.0
OUTAGE_S = 90.0


async def _at_p1(sc: Watched) -> None:
    await sc.arrange_airborne()
    await asyncio.sleep(AI_RATE_S)
    await sc.say("Flyg till P1.")
    await sc.wait_done("goto", 90, issuer_kind="brain")
    await sc.until(lambda: sc.onboard.ex.state == S.HOLD, 10, "HOLD efter goto")
    await sc.until(lambda: sc.tel.groundspeed_ms < expect.STILL_MS, 10, "står still")


async def _a(sc: Watched) -> None:
    await _at_p1(sc)
    lat, lon, alt = sc.tel.lat, sc.tel.lon, sc.tel.alt_rel_home_m
    mode = sc.tel.fc_custom_mode
    out = sc.pin("OUT")
    down = time.monotonic()
    await sc.radio(up=False)
    sent: list[str] = []
    moved = 0.0
    outage = 10.0
    while time.monotonic() - down < outage:
        for skill, params in (("goto", {"lat": out[0], "lon": out[1], "alt_m": 40}), ("hold", {})):
            res = await sc.commander_order(skill, **params)
            if res.status_code == 201:
                sent.append(str(res.json()["command_id"]))
        end = time.monotonic() + ORDER_EVERY_S
        while time.monotonic() < end:
            moved = max(moved, distance_m(lat, lon, sc.tel.lat, sc.tel.lon))
            assert sc.tel.fc_custom_mode == mode, f"läget ändrades till {sc.tel.fc_mode}"
            await asyncio.sleep(0.1)
    await sc.radio(up=True)
    up = time.monotonic()
    sc.criterion(
        "SC07.a.1",
        moved <= expect.L_HOLD_H_M and abs(sc.tel.alt_rel_home_m - alt) <= expect.L_HOLD_V_M,
        "ingen märkbar förändring i flygningen",
        measured=round(moved, 2),
        limit=expect.L_HOLD_H_M,
        unit="m",
    )
    statuses = [(await sc.command(c))["status"] for c in sent]
    sc.criterion(
        "SC07.a.2",
        bool(sent) and not any(s in RAN for s in statuses),
        "ingen AI-order körs under avbrottet",
        orders=len(sent),
        statuses=sorted(set(statuses)),
    )
    await sc.until(lambda: sc.r.link.up("sim-01"), expect.L_LINK_LOST_S, "SD-link uppe")
    sc.criterion(
        "SC07.a.3",
        sc.link_change(down, "sdlink", "degraded") is not None
        and sc.link_change(down, "mavlink", "down") is not None,
        "marken såg SD-link degraded och MAVLink borta; SD-link uppe igen",
        measured=round(time.monotonic() - up, 2),
        limit=expect.L_LINK_LOST_S,
    )
    await sc.finish()


async def _b(sc: Watched) -> None:
    await _at_p1(sc)
    down = time.monotonic()  # the last heartbeat from 250 reached the FC before this
    await sc.radio(up=False)
    timeout = expect.value("FS_GCS_TIMEOUT")
    chain = expect.fs_chain("FS_GCS_ENABLE")
    t, mode = await wait_mode(sc, chain, timeout + expect.L_FS_S + 10, "GCS-failsafe")
    # One heartbeat period (1 s) for when the last heartbeat went out before the cut.
    limit = timeout + expect.L_FS_S + 1.0
    sc.criterion(
        "SC07.b.1",
        t - down <= limit and first_choice_or_proof(sc, mode, chain, down),
        "flygkontrollens GCS-failsafe",
        measured=[round(t - down, 2), CopterMode(mode).name],
        limit=limit,
        unit="s",
    )
    await landed_home(sc, "SC07.b.2", t)
    # The radio stays down at least 90 s and until the vehicle has landed: from P1
    # (200 m out) the failsafe's way home and the landing take about 100 s.
    landed = time.monotonic() - down
    sc.criterion("SC07.b.2b", True, "landad med radion nere", measured=round(landed, 1))
    await asyncio.sleep(max(0.0, OUTAGE_S - landed))
    first = len(sc.tl.sent)
    await sc.radio(up=True)
    await sc.until(
        lambda: sc.r.g.vehicle_link.state() == "ok" and sc.r.link.up("sim-01"),
        expect.L_LINK_LOST_S,
        "fordonet anslutet igen",
    )
    await asyncio.sleep(5)
    status = await sc.r.status(sc.mission_id)
    steering = sc.steering_since(first, ("onboard", "ground"))
    sc.criterion(
        "SC07.b.3",
        status != "ACTIVE" and not steering and not sc.tel.armed,
        "uppdraget återupptas inte av sig självt",
        status=status,
    )
    await sc.finish()


VARIANTS = {"a": _a, "b": _b}


@pytest.mark.budget_s(300)
@pytest.mark.parametrize("variant", sorted(VARIANTS))
async def test_sc07(variant: str, sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario_x(f"test_sc07[{variant}]", sim, migrated_db, tmp_path, "SC07") as sc:
        await VARIANTS[variant](sc)
