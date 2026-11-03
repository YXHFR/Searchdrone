"""The simulated laser through ArduCopter SITL (builtin physics, headless).

``rangefinder = "grf250"``: the laser process answers SITL's position with
``DISTANCE_SENSOR`` on serial6, the flight controller reads it as its rangefinder and
reports it on the onboard computer's channel, and TN-4's ``LaserReader`` confirms the
ground under the hovering drone. A dropout switched on at runtime breaks the
confirmation, switching it off brings it back, and the laser's own messages never reach
the onboard computer's channel directly.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from typing import Any

import pytest

from sd_mav import CopterMode, MavResult, Role, SendOrigin, Vehicle, connect
from sd_obs.clock import MonotonicClock
from sd_onboard.config import RangefinderConfig
from sd_onboard.rangefinder import LaserReader
from sd_sim import config, paths, rangefinder
from sd_sim.harness import Harness
from sd_sim.launch import Stack
from sd_sim.pytest_plugin import worker_instance
from sd_sim.sitl import CMAC_HOME
from sd_sim.util import new_run_dir

pytestmark = [
    pytest.mark.sitl,
    pytest.mark.slow,
    pytest.mark.skipif(not paths.arducopter_bin().is_file(), reason="ArduCopter SITL is not built"),
]

READY_S = 90.0
HOVER_M = 15.0  # above the GRF-250's 10 m minimum


@pytest.fixture(scope="module")
def stack() -> Iterator[Stack]:
    instance = worker_instance()
    cfg = config.load(
        {
            "physics": "builtin",
            "gui": "false",
            "video": "off",
            "instance": str(instance),
            "rangefinder": "grf250",
        }
    )
    cfg.home, cfg.personal_home = CMAC_HOME, False
    st = Stack(cfg, new_run_dir(instance, "rangefinder"), observer=False)
    harness = Harness(st.ports["harness_mavlink"])
    try:
        st.up()
        harness.wait(lambda s: s.heartbeat, 60, "HEARTBEAT från flygkontrollen")
        harness.wait(lambda s: s.ready, READY_S, "position från EKF och GPS")
        harness.close()
        yield st
    finally:
        harness.close()
        st.down()


async def _until(check: Any, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if check():
            return True
        await asyncio.sleep(0.2)
    return bool(check())


@pytest.mark.budget_s(180)
def test_the_flight_controller_reports_the_simulated_laser(stack: Stack) -> None:
    profile = rangefinder.PROFILES["grf250"]
    min_m, max_m = profile.tested_m
    reader = LaserReader(RangefinderConfig(model="other", min_tested_m=min_m, max_tested_m=max_m))

    async def run() -> None:
        events: list[tuple[str, dict[str, Any]]] = []
        url = f"udpin:127.0.0.1:{stack.ports['onboard_mavlink']}"
        conn = connect(url, role=Role.ONBOARD, source_system=1, source_component=191)
        onboard = Vehicle(
            conn,
            role=Role.ONBOARD,
            clock=MonotonicClock(),
            on_event=lambda k, d: events.append((k, d)),
        )
        onboard.subscribe(reader.observe)
        await onboard.start()

        def state() -> Any:
            return reader.state(onboard.telemetry.rangefinder_mono_ms or 0)

        def now_state() -> Any:
            return reader.state(time.monotonic_ns() // 1_000_000)  # the Vehicle's clock

        try:
            assert await onboard.wait_connected(5.0)
            await asyncio.sleep(3.0)
            # On the ground the laser sees the ground closer than its 10 m minimum.
            assert onboard.telemetry.rangefinder_mono_ms is not None, "no DISTANCE_SENSOR"
            assert onboard.telemetry.rangefinder_m < min_m and not state().confirmed
            assert await onboard.set_mode(CopterMode.GUIDED, origin=SendOrigin.SKILL) is (
                MavResult.ACCEPTED
            )
            for _ in range(15):  # pre-arm checks can need a few seconds after start
                if await onboard.arm(origin=SendOrigin.SKILL) is MavResult.ACCEPTED:
                    break
                await asyncio.sleep(2.0)
            assert await onboard.takeoff(HOVER_M, origin=SendOrigin.SKILL) is MavResult.ACCEPTED
            assert await _until(lambda: onboard.telemetry.alt_rel_home_m > HOVER_M - 1, 30)
            assert await _until(lambda: now_state().confirmed, 10), "the laser never confirmed"
            t = onboard.telemetry
            surface = now_state().surface_amsl_m
            ground = t.alt_amsl_m - t.alt_rel_home_m  # builtin physics: flat ground at home
            # The reader keeps the highest surface of 2 s of readings with the GRF-250's
            # noise (±0.5 m), and the EKF's height is not the simulator's truth.
            assert surface == pytest.approx(ground, abs=1.5)

            stack.set_rangefinder_fault("dropout")
            assert await _until(lambda: not now_state().confirmed, 3), "a dropout went unseen"
            stack.set_rangefinder_fault(None)
            assert await _until(lambda: now_state().confirmed, 10), "no confirmation after it"

            leaked = [d for k, d in events if "forbidden" in k or "unexpected" in k]
            assert not leaked, leaked
        finally:
            await onboard.close()  # the stack goes down with the module

    asyncio.run(run())
