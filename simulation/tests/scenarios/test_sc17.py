"""SC17: the onboard computer dies without anyone pressing a button.

* a: killed during a ``goto`` (GUIDED): the ground's companion watchdog stops the
  vehicle (STOP's mode) and then flies it home (FLY HOME's mode), with the events
  ``vehicle.onboard_silent`` and ``vehicle.onboard_silent_rtl``; it lands at home.
* c: killed during ``survey_sector`` (AUTO): the flight controller flies the uploaded
  search on until the watchdog's home time, then home.
* d: the onboard computer and the ground station both killed during a ``goto``: the
  flight controller's own GCS failsafe flies home.
* e: the pilot's STOP first, then the onboard computer dies: the watchdog never acts
  during an active STOP (the alarm is still raised); FLY HOME afterwards lands it.
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
from scenario_rig import AI_RATE_S  # pyright: ignore[reportMissingImports]

from sd_mav import CopterMode
from sd_safety.constants import COMPANION_WATCHDOG_RTL_S
from sd_sim import expect

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.script("SC17")]

AIRBORNE = (CopterMode.GUIDED, CopterMode.AUTO, CopterMode.LOITER)


async def _goto(sc: Watched) -> float:
    """``goto(P1)`` under way in GUIDED, about 10 s out; returns the host time."""
    await asyncio.sleep(AI_RATE_S)
    await sc.say("Flyg till P1.")
    await sc.wait_command("goto", "running", 10, issuer_kind="brain")
    await sc.until(lambda: sc.tel.fc_custom_mode == CopterMode.GUIDED, 10, "GUIDED")
    await asyncio.sleep(8)
    assert sc.tel.groundspeed_ms > 1.0, "drönaren är på väg"
    return time.monotonic()


def _events(sc: Watched, kind: str, since: float) -> list[float]:
    return [t for t, k, _ in sc._bus if k == kind and t >= since]  # pyright: ignore[reportPrivateUsage]


async def _watchdog_home(sc: Watched, killed: float) -> float:
    """The watchdog flies home within its limit; returns when the mode changed."""
    t, mode = await wait_mode(
        sc, expect.fs_chain("FS_GCS_ENABLE"), expect.L_WATCHDOG_RTL_S + 5, "hemflygning"
    )
    sc.criterion(
        "SC17.home",
        t - killed <= expect.L_WATCHDOG_RTL_S
        and first_choice_or_proof(sc, mode, (expect.HOME_MODE, CopterMode.RTL), killed),
        "vakthunden flyger hem",
        measured=[round(t - killed, 2), CopterMode(mode).name],
        limit=expect.L_WATCHDOG_RTL_S,
        unit="s",
    )
    return t


async def _a(sc: Watched) -> None:
    await sc.arrange_airborne()
    killed = await _goto(sc)
    await sc.kill_onboard()
    t, mode = await wait_mode(
        sc, (expect.STOP_MODE, CopterMode.LOITER), expect.L_WATCHDOG_STOP_S + 5, "STOPP-läget"
    )
    sc.criterion(
        "SC17.stop",
        t - killed <= expect.L_WATCHDOG_STOP_S and mode == expect.STOP_MODE,
        "vakthunden stoppar",
        measured=[round(t - killed, 2), CopterMode(mode).name],
        limit=expect.L_WATCHDOG_STOP_S,
        unit="s",
    )
    home = await _watchdog_home(sc, killed)
    await asyncio.sleep(0.5)
    sc.criterion(
        "SC17.events",
        bool(_events(sc, "vehicle.onboard_silent", killed))
        and bool(_events(sc, "vehicle.onboard_silent_rtl", killed)),
        "händelserna för tyst ombordsdator",
    )
    await landed_home(sc, "SC17.land", home)
    await sc.finish()


async def _c(sc: Watched) -> None:
    await sc.arrange_airborne()
    await asyncio.sleep(AI_RATE_S)
    await sc.say("Sök av rutan.")
    await sc.wait_command("survey_sector", "running", 30, issuer_kind="brain")
    await sc.until(lambda: sc.tel.fc_custom_mode == CopterMode.AUTO, 30, "AUTO")
    await asyncio.sleep(5)
    killed = time.monotonic()
    await sc.kill_onboard()
    # The search flies on in AUTO until the watchdog's home time.
    await sc.still(
        lambda: sc.tel.fc_custom_mode == CopterMode.AUTO,
        COMPANION_WATCHDOG_RTL_S - 2,
        "sökningen fortsätter i AUTO",
    )
    sc.criterion("SC17.continues", True, "sökningen fortsätter till vakthundens hemtid")
    home = await _watchdog_home(sc, killed)
    await landed_home(sc, "SC17.land", home)
    await sc.finish()


async def _d(sc: Watched) -> None:
    await sc.arrange_airborne()
    await _goto(sc)
    await sc.kill_onboard()
    await sc.snapshot_now()
    killed = await sc.kill_ground()  # the last heartbeat from 250 is no later than this
    timeout = expect.value("FS_GCS_TIMEOUT")
    chain = expect.fs_chain("FS_GCS_ENABLE")
    t, mode = await wait_mode(sc, chain, timeout + expect.L_FS_S + 10, "GCS-failsafe")
    sc.criterion(
        "SC17.fs",
        t - killed <= timeout + expect.L_FS_S + 1.0
        and first_choice_or_proof(sc, mode, chain, killed),
        "flygkontrollens GCS-failsafe flyger hem",
        measured=[round(t - killed, 2), CopterMode(mode).name],
        limit=timeout + expect.L_FS_S,
        unit="s",
    )
    await landed_home(sc, "SC17.land", t)
    sc.finish_offline(["INV9"])


async def _e(sc: Watched) -> None:
    await sc.arrange_airborne()
    await _goto(sc)
    pressed = await sc.press("stop")
    await sc.until(
        lambda: sc.tel.groundspeed_ms < expect.STILL_MS, expect.L_STOP_STILL_S, "står still"
    )
    killed = time.monotonic()
    first = len(sc.tl.sent)
    await sc.kill_onboard()
    hold = COMPANION_WATCHDOG_RTL_S + 5
    await sc.still(lambda: sc.tel.fc_custom_mode == expect.STOP_MODE, hold, "STOPP står kvar")
    other = [
        m
        for m in sc.tl.sent[first:]
        if m.sender == "ground" and m.mode is not None and m.mode != expect.STOP_MODE
    ]
    sc.criterion("SC17.nowatchdog", not other, "vakthunden agerar inte under STOPP", n=len(other))
    sc.criterion(
        "SC17.alarm",
        bool(_events(sc, "alarm.ONBOARD_SILENT", killed)),
        "larmet för tyst ombordsdator finns ändå",
    )
    assert pressed < killed
    home = await sc.press("return_home")
    await wait_mode(sc, (expect.HOME_MODE, CopterMode.RTL), expect.L_STOP_HB_S + 2, "FLYG HEM")
    await landed_home(sc, "SC17.land", home)
    await sc.finish()


VARIANTS = {"a": _a, "c": _c, "d": _d, "e": _e}


@pytest.mark.budget_s(240)
@pytest.mark.parametrize("variant", sorted(VARIANTS))
async def test_sc17(variant: str, sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario_x(f"test_sc17[{variant}]", sim, migrated_db, tmp_path, "SC17") as sc:
        await VARIANTS[variant](sc)
