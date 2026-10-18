"""SC15: STOP and FLY HOME work without the helpers.

The AI flies to P1. At 60 m from home a helper fails, and 2 s later the pilot presses:

* a: the AI dead, STOP: the flight controller brakes and holds; the panel shows the AI
  as down.
* b: the AI dead, FLY HOME: home and landed.
* c: the onboard computer dead, STOP: the ground's own DO_SET_MODE brakes it; the alarm
  for a silent onboard computer comes all the same.
* d: the onboard computer dead, FLY HOME: home and landed.
* f: the ground's MAVLink cut, STOP: the ground cannot reach the flight controller, so
  the onboard computer brakes on the ground's notice over SD-link.

A paused database (e) needs the instance's own database container; it runs only on
the owner's computer and is not part of this module.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from rig_emergency import (  # pyright: ignore[reportMissingImports]
    DO_SET_MODE,
    ack_log,
    lands_home,
    still_and_hold,
    stop_ack_and_mode,
    system_status,
)
from scenario_rig import AI_RATE_S, Scenario, scenario  # pyright: ignore[reportMissingImports]

from sd_mav.modes import HOME_STEPS, STOP_STEPS
from sd_safety.constants import COMPANION_WATCHDOG_S, OVERRIDE_ONBOARD_FALLBACK_S
from sd_sim import expect

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.script("SC15")]

FAULT_AT_M = 60.0
PRESS_AFTER_S = 2.0
FALLBACK_ACK_S = OVERRIDE_ONBOARD_FALLBACK_S + 1.0


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param("a", marks=pytest.mark.budget_s(150)),
        pytest.param("b", marks=pytest.mark.budget_s(200)),
        pytest.param("c", marks=pytest.mark.budget_s(130)),
        pytest.param("d", marks=pytest.mark.budget_s(200)),
        pytest.param("f", marks=pytest.mark.budget_s(130)),
    ],
)
async def test_sc15(sim: Any, migrated_db: Any, tmp_path: Path, variant: str) -> None:
    async with scenario(f"test_sc15[{variant}]", sim, migrated_db, tmp_path, "SC15") as sc:
        await _run(sc, variant)


async def _run(sc: Scenario, variant: str) -> None:
    ground_acks = ack_log(sc, sc.r.core.vehicle)
    onboard_acks = ack_log(sc, sc.onboard.vehicle)
    await sc.arrange_airborne()
    await asyncio.sleep(AI_RATE_S)
    await sc.say("Flyg till P1.")
    await sc.wait_command("goto", "running", 30, issuer_kind="brain")
    await sc.until(lambda: sc.from_home_m() >= FAULT_AT_M, 60, f"{FAULT_AT_M:.0f} m från hem")

    t_fault = time.monotonic()
    if variant in ("a", "b"):
        await sc.kill_brain()
    elif variant in ("c", "d"):
        await sc.kill_onboard()
    else:
        await sc.flow("mavlink", False)
    await asyncio.sleep(max(0.0, t_fault + PRESS_AFTER_S - time.monotonic()))
    stop = variant in ("a", "c", "f")
    t0 = await sc.press("stop" if stop else "return_home")

    if variant == "f":
        await _onboard_fallback(sc, onboard_acks, t0)
        return
    if stop:
        await stop_ack_and_mode(sc, ground_acks, t0, expect.STOP_MODE, STOP_STEPS, "SC15.ack")
        await still_and_hold(sc, t0, "SC15", hold=variant == "a")
    else:
        await stop_ack_and_mode(sc, ground_acks, t0, expect.HOME_MODE, HOME_STEPS, "SC15.ack")
        await lands_home(sc, t0, "SC15")

    if variant == "a":

        async def brain_down() -> bool:
            parts = (await system_status(sc))["parts"]
            return parts.get("brain", {}).get("state") == "down"

        await sc.until(brain_down, expect.L_BRAIN_DOWN_S, "AI:n visas som nere")
        sc.criterion("SC15.brain", True, "panelen visar brain = down")
    if variant == "c":
        silent = [t for t, kind, _ in sc._bus if kind == "alarm.ONBOARD_SILENT"]  # pyright: ignore[reportPrivateUsage]
        took = None if not silent else round(silent[0] - t_fault, 2)
        sc.criterion(
            "SC15.alarm",
            took is not None and took <= COMPANION_WATCHDOG_S + 1.0,
            "larm för tyst ombordsdator",
            measured=took,
            limit=COMPANION_WATCHDOG_S + 1.0,
            unit="s",
        )
    await sc.finish()


async def _onboard_fallback(sc: Scenario, acks: Any, t0: float) -> None:
    """The ground cannot reach the flight controller: the onboard computer brakes."""
    await sc.until(
        lambda: acks.first(DO_SET_MODE, t0) is not None,
        FALLBACK_ACK_S + 1.0,
        "COMMAND_ACK(DO_SET_MODE) på ombordskanalen",
    )
    ack = acks.first(DO_SET_MODE, t0)
    took = round(ack.t - t0, 3)
    tel = sc.onboard.vehicle.telemetry
    sc.criterion(
        "SC15.f.ack",
        took <= FALLBACK_ACK_S,
        "ombordsdatorn bromsar efter markens meddelande",
        measured=took,
        limit=FALLBACK_ACK_S,
        unit="s",
        clock="host",
    )
    await sc.until(
        lambda: sc.onboard.vehicle.telemetry.fc_custom_mode in {int(m) for m in STOP_STEPS},
        expect.L_STOP_HB_S,
        "STOPP-läge på ombordskanalen",
    )
    await sc.until(
        lambda: sc.onboard.vehicle.telemetry.groundspeed_ms < expect.STILL_MS,
        expect.L_STOP_STILL_S + 1.0,
        "står still",
    )
    # Counted from the moment the onboard computer could brake (its fallback waits for
    # the ground first), so the fallback delay is not taken from the braking time.
    still = time.monotonic() - ack.t
    sc.criterion(
        "SC15.f.still",
        still <= expect.L_STOP_STILL_S and tel.fc_custom_mode == int(expect.STOP_MODE),
        "står still (L-STOP-STILL, från ombordsdatorns broms)",
        measured=round(still, 2),
        from_call=round(time.monotonic() - t0, 2),
        limit=expect.L_STOP_STILL_S,
    )
    await sc.flow("mavlink", True)
    await sc.finish()
