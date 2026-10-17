"""The real ``sd-brain`` flies ArduCopter SITL through the ground station (builtin physics).

Everything is real but the language model: ``sd-ground``'s mission flow with its
database, the gate, the emergency path, SD-link and the onboard executor against SITL
(as ``test_mission_flow_sitl``), and ``sd-brain`` on the commander API over real HTTP and
WebSocket in the fixed command mode (``brain_rig``). The pilot is the panel API.

* the AI takes off on the pilot's "Starta" (approved charter, "clear around the drone",
  the pilot's request) and flies to the pilot's pin P1 on "flyg till P1", every order
  through the gate;
* the pilot's STOP stops the vehicle past the AI, and while the stop holds the AI sends
  nothing that moves it, even when asked;
* the pilot's FLY HOME brings it home past the AI, and it lands.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "onboard"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "software/ground/tests/helpers"))

from brain_rig import free_port, run_brain, serve_commander  # pyright: ignore[reportMissingImports]
from flow_rig import VID, H, Rig, WallClock, ground_stack  # pyright: ignore[reportMissingImports]
from test_mission_flow_sitl import (  # pyright: ignore[reportMissingImports]
    P1,
    SitlOnboard,
    _ready,
    fresh_db,
    migrated_db,
    passwords,
)
from test_onboard_sitl import Sim, _vehicle, sim  # pyright: ignore[reportMissingImports]

from sd_mav import Role
from sd_obs.clock import MonotonicClock
from sd_safety.targets import distance_m
from sd_schemas.base import new_id

pytestmark = [pytest.mark.sitl, pytest.mark.slow]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]


@pytest.fixture
async def flight(sim: Sim, migrated_db: Any, tmp_path: Path) -> AsyncIterator[tuple[Rig, int]]:
    ground = _vehicle(f"udpin:127.0.0.1:{sim.ports['ground_mavlink']}", Role.GROUND)
    port = free_port()

    def onboard(_clock: WallClock, url: str, token: str) -> SitlOnboard:
        return SitlOnboard(sim, url, token)

    async with (
        ground_stack(
            migrated_db.name,
            migrated_db.app_password,
            tmp_path,
            vehicle=ground,
            mav_clock=MonotonicClock(),
            onboard=onboard,
            http={"commander_bind": f"127.0.0.1:{port}"},
        ) as r,
        serve_commander(r, port),
    ):
        yield r, port


async def _say(r: Rig, mid: str, text: str) -> None:
    res = await r.panel.post(
        f"/api/v1/missions/{mid}/chat",
        json={"client_msg_id": str(new_id()), "text": text},
        headers=H,
    )
    assert res.status_code == 201, res.text


async def _press(r: Rig, action: str) -> None:
    res = await r.panel.post(
        f"/api/v1/vehicles/{VID}/actions/{action}",
        json={"client_action_id": str(new_id())},
        headers=H,
    )
    assert res.status_code == 202, res.text


async def _brain_orders(r: Rig, mid: str) -> list[dict[str, Any]]:
    cmds = (await r.panel.get(f"/api/v1/missions/{mid}/commands")).json()
    return [c for c in cmds if c["issuer_kind"] == "brain"]


async def _until_brain_done(r: Rig, mid: str, skill: str, limit_s: float) -> None:
    start = time.monotonic()
    while True:
        cmd = next((c for c in await _brain_orders(r, mid) if c["skill"] == skill), None)
        if cmd is not None and cmd["status"] == "done":
            return
        if cmd is not None and cmd["status"] in (
            "failed",
            "rejected_by_onboard",
            "rejected_by_gate",
            "cancelled",
        ):
            raise AssertionError(f"{skill}: {cmd['status']} {cmd['status_reason']}")
        if time.monotonic() - start > limit_s:
            state = None if cmd is None else cmd["status"]
            raise AssertionError(f"the AI's {skill}: not done within {limit_s} s ({state})")
        await asyncio.sleep(0.2)


def _mode(r: Rig) -> str:
    return str(r.core.vehicle.telemetry.fc_mode)


@pytest.mark.budget_s(420)
async def test_ai_flies_and_the_pilot_stops_and_brings_it_home_on_sitl(
    flight: tuple[Rig, int], sim: Sim
) -> None:
    r, port = flight
    mid = await _ready(r)
    lat, lon = r.at(*P1)
    res = await r.panel.post(
        f"/api/v1/missions/{mid}/annotations",
        json={
            "kind": "pin",
            "label": "P1",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
        },
        headers=H,
    )
    assert res.status_code == 201, res.text

    async with run_brain(r, port):
        res = await r.clearance(mid)
        assert res.status_code == 200, res.text
        await _say(r, mid, "Starta")
        await _until_brain_done(r, mid, "takeoff", 90)
        sim.rc.throttle = 1500  # in the air: the pilot centres the throttle stick
        await r.wait_status(mid, "ACTIVE")

        await asyncio.sleep(3.1)  # the gate's rate limit: one AI action per 3 s
        await _say(r, mid, "Flyg till P1")
        await _until_brain_done(r, mid, "goto", 90)
        tel = r.core.vehicle.telemetry
        assert distance_m(lat, lon, tel.lat, tel.lon) < 10.0

        # The pilot's STOP: past the AI, straight to the flight controller.
        await _press(r, "stop")
        await r.until(lambda: _mode(r) in ("BRAKE", "LOITER"), 5, "stopped")
        before = len(await _brain_orders(r, mid))
        await asyncio.sleep(3.1)
        await _say(r, mid, "Flyg till P1")  # the AI gets no tool that moves the vehicle
        await asyncio.sleep(5.0)
        assert len(await _brain_orders(r, mid)) == before
        assert _mode(r) in ("BRAKE", "LOITER")

        # The pilot's FLY HOME: past the AI; it comes home and lands.
        await _press(r, "return_home")
        await r.until(lambda: _mode(r) in ("SMART_RTL", "RTL", "LAND"), 5, "flying home")
        sim.rc.throttle = 1000
        await r.wait_status(mid, "LANDED", 180)
        tel = r.core.vehicle.telemetry
        assert distance_m(r.home[0], r.home[1], tel.lat, tel.lon) < 5.0
        assert [c["skill"] for c in await _brain_orders(r, mid)] == ["takeoff", "goto"]
