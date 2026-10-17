"""sd_mav against ArduCopter SITL (builtin physics, headless).

One simulator stack per module. The onboard computer's channel goes through mavtap,
the ground station's through linksim, exactly like the full system:

* both sides connect, see a HEARTBEAT within 5 s and publish telemetry at >= 4 Hz;
* the flight controller's HEARTBEAT comes at 5 Hz once asked for;
* a mode change on the allowlist is accepted, a refused one never leaves;
* three missions and a fence go up and come back with the same hash;
* a cut ground link is noticed within 3 s, and streams come back after it;
* the flight helpers fly (arm, take off, goto, speed, LOITER, LAND), the ground station
  disarms only on the ground, and the pilot's sticks work from the ground station (the
  onboard computer may not send them: ArduCopter takes sticks only from its GCS id).
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav import CopterMode, ForbiddenMavlinkError, MavResult, Role, SendOrigin, Vehicle, connect
from sd_mav.mission import (
    FENCE,
    MISSION,
    download,
    fence_hash,
    fence_items,
    mission_hash,
    survey_mission,
    upload,
)
from sd_obs.clock import MonotonicClock
from sd_schemas.link import StickAxes
from sd_sim import config, paths
from sd_sim.harness import Harness
from sd_sim.launch import Stack
from sd_sim.pytest_plugin import worker_instance
from sd_sim.sitl import CMAC_HOME
from sd_sim.util import LOOPBACK_OPENER, new_run_dir

pytestmark = [
    pytest.mark.sitl,
    pytest.mark.slow,
    pytest.mark.skipif(not paths.arducopter_bin().is_file(), reason="ArduCopter SITL is not built"),
]

READY_S = 90.0
HOME_LAT, HOME_LON = (float(v) for v in CMAC_HOME.split(",")[:2])


@dataclass
class Sim:
    stack: Stack
    ports: dict[str, int]

    def onboard_url(self) -> str:
        return f"udpin:127.0.0.1:{self.ports['onboard_mavlink']}"

    def ground_url(self) -> str:
        return f"udpin:127.0.0.1:{self.ports['ground_mavlink']}"

    def set_ground_link(self, up: bool) -> None:
        body = json.dumps({"state": "up" if up else "down"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.ports['linksim_control']}/flows/mavlink",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.stack.linksim_token}",
                "Content-Type": "application/json",
            },
        )
        with LOOPBACK_OPENER.open(req, timeout=2) as resp:
            resp.read()


@pytest.fixture(scope="module")
def sim() -> Iterator[Sim]:
    instance = worker_instance()
    cfg = config.load(
        {"physics": "builtin", "gui": "false", "video": "off", "instance": str(instance)}
    )
    cfg.home, cfg.personal_home = CMAC_HOME, False
    stack = Stack(cfg, new_run_dir(instance, "sd_mav"), observer=False)
    harness = Harness(stack.ports["harness_mavlink"])
    try:
        stack.up()
        harness.wait(lambda s: s.heartbeat, 60, "HEARTBEAT från flygkontrollen")
        harness.wait(lambda s: s.ready, READY_S, "position från EKF och GPS")
        harness.close()
        yield Sim(stack, stack.ports)
    finally:
        harness.close()
        stack.down()


class Rate:
    """Counts telemetry updates and FC heartbeats seen by one vehicle."""

    def __init__(self, vehicle: Vehicle) -> None:
        self.updates = 0
        vehicle.subscribe(self._tick)

    def _tick(self, _t: Any) -> None:
        self.updates += 1


def _vehicle(url: str, role: Role, events: list[tuple[str, dict[str, Any]]]) -> Vehicle:
    ids = (1, 191) if role is Role.ONBOARD else (250, 190)
    conn = connect(url, role=role, source_system=ids[0], source_component=ids[1])
    return Vehicle(
        conn, role=role, clock=MonotonicClock(), on_event=lambda k, d: events.append((k, d))
    )


async def _fc_heartbeats(vehicle: Vehicle, seconds: float) -> int:
    seen = 0

    def count(msg: Any) -> bool:
        nonlocal seen
        if msg.get_type() == "HEARTBEAT":
            seen += 1
        return False

    await vehicle.wait_message(count, seconds)
    return seen


@pytest.mark.budget_s(150)
def test_connect_heartbeat_and_telemetry_rate(sim: Sim) -> None:
    async def run() -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        onboard = _vehicle(sim.onboard_url(), Role.ONBOARD, events)
        ground = _vehicle(sim.ground_url(), Role.GROUND, events)
        t0 = time.monotonic()
        await onboard.start()
        await ground.start()
        try:
            assert await onboard.wait_connected(5.0), "no HEARTBEAT on the onboard channel"
            assert await ground.wait_connected(5.0), "no HEARTBEAT on the ground channel"
            assert time.monotonic() - t0 <= 5.0
            await asyncio.sleep(3.0)  # streams requested and running
            failed = [d for k, d in events if k == "mav.streams_requested" and d["failed"]]
            assert not failed, failed
            rates = [Rate(onboard), Rate(ground)]
            hb_onboard, hb_ground = await asyncio.gather(
                _fc_heartbeats(onboard, 10.0), _fc_heartbeats(ground, 10.0)
            )
            for rate in rates:
                assert rate.updates / 10.0 >= 4.0, rate.updates
            assert hb_onboard / 10.0 >= 4.0 and hb_ground / 10.0 >= 4.0, (hb_onboard, hb_ground)
            t = onboard.telemetry
            assert t.home is not None and abs(t.home.lat - HOME_LAT) < 1e-4
            assert t.landed_state == "on_ground" and not t.armed and t.gps_fix_type >= 3
            assert t.fc_firmware is not None and t.fc_firmware.startswith("4.7")
            assert {(k, d.get("msg_type")) for k, d in events if "forbidden" in k} == set()
        finally:
            await onboard.close()
            await ground.close()

    asyncio.run(run())


@pytest.mark.budget_s(120)
def test_allowlisted_mode_change_and_refusal(sim: Sim) -> None:
    async def run() -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        onboard = _vehicle(sim.onboard_url(), Role.ONBOARD, events)
        await onboard.start()
        try:
            assert await onboard.wait_connected(5.0)
            assert (
                await onboard.set_mode(CopterMode.GUIDED, origin=SendOrigin.SKILL)
                is MavResult.ACCEPTED
            )
            await asyncio.sleep(0.6)
            assert onboard.telemetry.fc_mode == "GUIDED"
            sent = onboard.sent_count
            with pytest.raises(ForbiddenMavlinkError):
                await onboard.set_mode(CopterMode.STABILIZE, origin=SendOrigin.SKILL)
            with pytest.raises(ForbiddenMavlinkError):
                await onboard.send(
                    mavlink.MAVLink_param_set_message(1, 1, b"FENCE_ENABLE", 0, 2),
                    origin=SendOrigin.SKILL,
                )
            assert onboard.sent_count == sent
            await asyncio.sleep(0.6)
            assert onboard.telemetry.fc_mode == "GUIDED"
            # Reading is allowed (the safety parameter file comes with the onboard ticket).
            assert await onboard.read_param("FENCE_ENABLE", origin=SendOrigin.STARTUP) in (0.0, 1.0)
            assert (
                await onboard.set_mode(CopterMode.LOITER, origin=SendOrigin.SKILL)
                is MavResult.ACCEPTED
            )
        finally:
            await onboard.close()

    asyncio.run(run())


@pytest.mark.budget_s(150)
def test_mission_roundtrip_hash_stable(sim: Sim) -> None:
    async def run() -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        onboard = _vehicle(sim.onboard_url(), Role.ONBOARD, events)
        ground = _vehicle(sim.ground_url(), Role.GROUND, events)
        await onboard.start()
        await ground.start()
        try:
            assert await onboard.wait_connected(5.0) and await ground.wait_connected(5.0)
            for _ in range(50):  # HOME_POSITION arrives with the streams
                if onboard.telemetry.home is not None and ground.telemetry.home is not None:
                    break
                await asyncio.sleep(0.2)
            home = onboard.telemetry.home
            assert home is not None
            d = 0.001
            missions = [
                survey_mission(home, [(HOME_LAT + d, HOME_LON)], 40.0, 7.0),
                survey_mission(
                    home, [(HOME_LAT + d, HOME_LON), (HOME_LAT + d, HOME_LON + d)], 35.5, 5.0
                ),
                survey_mission(
                    home,
                    [(HOME_LAT + i * 0.0003, HOME_LON + (i % 2) * 0.001) for i in range(12)],
                    120.0,
                    15.0,
                ),
            ]
            for items in missions:
                hashes = set()
                for _ in range(2):
                    await upload(onboard, items, origin=SendOrigin.SKILL)
                    back = await download(onboard, MISSION, origin=SendOrigin.STARTUP)
                    assert len(back) == len(items)
                    hashes.add(mission_hash(back))
                assert hashes == {mission_hash(items)}
            corners = [
                (HOME_LAT - d, HOME_LON - d),
                (HOME_LAT - d, HOME_LON + d),
                (HOME_LAT + d, HOME_LON + d),
                (HOME_LAT + d, HOME_LON - d),
            ]
            fence = fence_items(corners)
            fence_hashes = set()
            for _ in range(2):
                await upload(ground, fence, origin=SendOrigin.GROUND)
                fence_hashes.add(
                    fence_hash(await download(ground, FENCE, origin=SendOrigin.GROUND))
                )
            fence_hashes.add(fence_hash(await download(onboard, FENCE, origin=SendOrigin.STARTUP)))
            assert fence_hashes == {fence_hash(fence)}
            with pytest.raises(ForbiddenMavlinkError):
                await upload(onboard, fence, origin=SendOrigin.SKILL)
        finally:
            await onboard.close()
            await ground.close()

    asyncio.run(run())


@pytest.mark.budget_s(120)
def test_lost_link_detected_within_3s(sim: Sim) -> None:
    async def run() -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        ground = _vehicle(sim.ground_url(), Role.GROUND, events)
        await ground.start()
        lost: list[float] = []
        try:
            assert await ground.wait_connected(5.0)
            await asyncio.sleep(2.0)
            # Only from now on: the first updates before the link was up say "not connected".
            ground.subscribe(
                lambda t: None if t.fc_connected or lost else lost.append(time.monotonic())
            )
            cut = time.monotonic()
            sim.set_ground_link(False)
            for _ in range(60):
                if lost:
                    break
                await asyncio.sleep(0.1)
            assert lost, "the cut link was not noticed"
            assert "mav.fc_lost" in [k for k, _ in events]
            assert lost[0] - cut <= 3.0, lost[0] - cut
            requested_before = sum(1 for k, _ in events if k == "mav.streams_requested")
            sim.set_ground_link(True)
            assert await ground.wait_connected(5.0)
            for _ in range(100):
                if sum(1 for k, _ in events if k == "mav.streams_requested") > requested_before:
                    break
                await asyncio.sleep(0.2)
            assert sum(1 for k, _ in events if k == "mav.streams_requested") > requested_before, [
                k for k, _ in events
            ]
            assert [k for k, _ in events].count("mav.fc_connected") == 2
        finally:
            sim.set_ground_link(True)
            await ground.close()

    asyncio.run(run())


def _metres(t: Any, lat: float, lon: float) -> float:
    from sd_safety.targets import distance_m

    return distance_m(lat, lon, t.lat, t.lon)


@pytest.mark.budget_s(180)
def test_flight_helpers_and_ground_sticks(sim: Sim) -> None:
    """arm, take off, goto, speed, LOITER and LAND through the allowlist; the ground may
    disarm only on the ground. The pilot's sticks go as MANUAL_CONTROL from the ground
    station (250/190) and move the vehicle; when they stop it stands still in LOITER.
    The onboard computer's allowlist refuses sticks for every origin."""

    async def run() -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        onboard = _vehicle(sim.onboard_url(), Role.ONBOARD, events)
        ground = _vehicle(sim.ground_url(), Role.GROUND, events)
        await onboard.start()
        await ground.start()
        try:
            assert await onboard.wait_connected(5.0) and await ground.wait_connected(5.0)
            await asyncio.sleep(3.0)
            assert await onboard.set_mode(CopterMode.GUIDED, origin=SendOrigin.SKILL) is (
                MavResult.ACCEPTED
            )
            await asyncio.sleep(0.6)
            for _ in range(15):  # pre-arm checks can need a few seconds after start
                if await onboard.arm(origin=SendOrigin.SKILL) is MavResult.ACCEPTED:
                    break
                await asyncio.sleep(2.0)
            await asyncio.sleep(0.6)
            # Standing armed on the ground: the ground station's STOP may disarm.
            assert await ground.disarm(origin=SendOrigin.GROUND) is MavResult.ACCEPTED
            await asyncio.sleep(0.6)
            assert not onboard.telemetry.armed
            assert await onboard.arm(origin=SendOrigin.SKILL) is MavResult.ACCEPTED
            assert await onboard.takeoff(15.0, origin=SendOrigin.SKILL) is MavResult.ACCEPTED
            for _ in range(40):
                if onboard.telemetry.alt_rel_home_m > 14.0:
                    break
                await asyncio.sleep(0.5)
            assert onboard.telemetry.alt_rel_home_m > 14.0
            assert onboard.telemetry.landed_state in ("in_air", "takeoff")
            with pytest.raises(ForbiddenMavlinkError):  # never in the air
                await ground.disarm(origin=SendOrigin.GROUND)
            assert await onboard.change_speed(5.0, origin=SendOrigin.SKILL) is MavResult.ACCEPTED
            target = (HOME_LAT + 0.0003, HOME_LON)  # about 33 m north
            await onboard.goto(*target, 15.0, origin=SendOrigin.SKILL)
            for _ in range(40):
                if _metres(onboard.telemetry, *target) < 2.0:
                    break
                await asyncio.sleep(0.5)
            assert _metres(onboard.telemetry, *target) < 2.0
            assert await ground.set_mode(CopterMode.LOITER, origin=SendOrigin.GROUND) is (
                MavResult.ACCEPTED
            )
            await asyncio.sleep(0.6)
            axes = StickAxes.model_validate(
                {
                    "takeover_id": "0192f4a3-0000-7000-8000-0000000000c0",
                    "input_seq": 1,
                    "pitch": 0.6,
                    "roll": 0.0,
                    "throttle": 0.5,
                    "yaw": 0.0,
                }
            )
            for origin in SendOrigin:  # never from the onboard computer
                with pytest.raises(ForbiddenMavlinkError):
                    await onboard.manual_control(axes, origin=origin)
            t0 = onboard.telemetry
            for _ in range(30):  # 3 s of forward stick at 10 Hz
                await ground.manual_control(axes, origin=SendOrigin.MANUAL)
                await asyncio.sleep(0.1)
            moved = _metres(onboard.telemetry, t0.lat, t0.lon)
            assert moved > 3.0, f"sticks from 250/190 moved the vehicle only {moved:.1f} m"
            assert onboard.telemetry.fc_mode == "LOITER"
            # No more sticks: LOITER brakes and holds (RC_OVERRIDE_TIME ends the override).
            await asyncio.sleep(4.0)
            t1 = onboard.telemetry
            await asyncio.sleep(3.0)
            drift = _metres(onboard.telemetry, t1.lat, t1.lon)
            assert drift < 2.0, f"drifted {drift:.1f} m after the sticks stopped"
            events.clear()  # the refused onboard sticks above are expected
            assert await onboard.set_mode(CopterMode.GUIDED, origin=SendOrigin.SKILL) is (
                MavResult.ACCEPTED
            )
            assert await ground.set_mode(CopterMode.LAND, origin=SendOrigin.GROUND) is (
                MavResult.ACCEPTED
            )
            for _ in range(60):
                if not onboard.telemetry.armed:
                    break
                await asyncio.sleep(0.5)
            assert not onboard.telemetry.armed
            assert not [d for k, d in events if "forbidden" in k and d["origin"] != "ground"]
        finally:
            await onboard.close()
            await ground.close()

    asyncio.run(run())
