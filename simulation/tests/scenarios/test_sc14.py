"""SC14: only SD-link is cut (linksim's ``link`` flow); MAVLink stays up.

* a: 10 s during the search: the ground sees SD-link degraded after about 3 s and lost
  after about 10 s; the search flies on unchanged; an order sent meanwhile is refused on
  the ground with ``LINK_DOWN`` and never reaches the vehicle; afterwards the link comes
  back (a new Hello) within 10 s.
* c: 60 s: the onboard computer holds at ``LINK_LOSS_DETECT_S + SDLINK_GRACE_S`` and
  flies home at ``+ SDLINK_HOLD_S`` (each +- 1 s), and lands at home.
* d: SD-link down, then the pilot's STOP: it arrives over MAVLink alone, and the vehicle
  stands still for 30 s with nothing steering from the onboard computer.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from rig_ground import (  # pyright: ignore[reportMissingImports]
    Watched,
    landed_home,
    scenario_x,
    wait_mode,
)
from scenario_rig import AI_RATE_S  # pyright: ignore[reportMissingImports]

from sd_mav import CopterMode
from sd_safety.constants import LINK_LOSS_DETECT_S, SDLINK_GRACE_S, SDLINK_HOLD_S
from sd_schemas.executor import ExecutorState as S
from sd_sim import expect

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.script("SC14")]

TOLERANCE_S = 1.0


async def _searching(sc: Watched) -> None:
    await sc.arrange_airborne()
    await asyncio.sleep(AI_RATE_S)
    await sc.say("Sök av rutan.")
    await sc.wait_command("survey_sector", "running", 30, issuer_kind="brain")
    await sc.until(lambda: sc.tel.fc_custom_mode == CopterMode.AUTO, 30, "AUTO")


def _near(value: float | None, want: float, tol: float = TOLERANCE_S) -> bool:
    return value is not None and abs(value - want) <= tol


async def _installed_again(sc: Watched) -> None:
    """After the link is back the ground installs the charter again; the invariants read
    the charter's install status at the end, so wait until that is done."""
    await sc.until(lambda: sc.r.link.up("sim-01"), expect.L_LINK_LOST_S, "SD-link uppe igen")
    await asyncio.sleep(1.0)  # the new session's Hello and the install it triggers

    async def installed() -> bool:
        res = await sc.r.panel.get(f"/api/v1/missions/{sc.mission_id}/charters")
        return all(c["install_status"] == "installed" for c in res.json() if c["approved_at"])

    await sc.until(installed, expect.L_LINK_LOST_S, "ramen installerad igen")


async def _a(sc: Watched) -> None:
    await _searching(sc)
    down = time.monotonic()
    await sc.flow("link", up=False)
    # Once the ground has noticed (degraded) and the AI's rate limit has passed.
    await sc.until(
        lambda: sc.link_change(down, "sdlink", "degraded") is not None,
        expect.L_LINK_DEGRADED_S + 2,
        "SD-link degraded",
    )
    await asyncio.sleep(max(0.0, AI_RATE_S - (time.monotonic() - down)))
    res = await sc.commander_order("hold")
    body = res.json()
    sc.criterion(
        "SC14.a.order",
        res.status_code == 201
        and body["decision"] == "REJECT"
        and "LINK_DOWN" in body["reason_codes"],
        "en order under avbrottet får LINK_DOWN på marken",
        codes=body.get("reason_codes"),
    )
    held_id = str(body["command_id"])
    await sc.still(
        lambda: sc.tel.fc_custom_mode == CopterMode.AUTO and sc.onboard.ex.state == S.SURVEY,
        expect.L_LINK_LOST_S + 1.5 - (time.monotonic() - down),
        "sökningen fortsätter oförändrad",
    )
    await sc.flow("link", up=True)
    up = time.monotonic()
    degraded = sc.link_change(down, "sdlink", "degraded")
    lost = sc.link_change(down, "sdlink", "lost")
    sc.criterion(
        "SC14.a.link",
        _near(None if degraded is None else degraded - down, expect.L_LINK_DEGRADED_S)
        and _near(None if lost is None else lost - down, expect.L_LINK_LOST_S),
        "L-LINK: degraded och lost i tid",
        measured=[
            None if degraded is None else round(degraded - down, 2),
            None if lost is None else round(lost - down, 2),
        ],
        limit=[expect.L_LINK_DEGRADED_S, expect.L_LINK_LOST_S],
        unit="s",
    )
    await sc.until(lambda: sc.r.link.up("sim-01"), expect.L_LINK_LOST_S, "SD-link uppe igen")
    sc.criterion(
        "SC14.a.up",
        True,
        "Hello igen och länken uppe",
        measured=round(time.monotonic() - up, 2),
        limit=expect.L_LINK_LOST_S,
    )
    sc.criterion(
        "SC14.a.never",
        (await sc.command(held_id))["status"] == "rejected_by_gate",
        "ordern stoppas på marken och når aldrig ombord",
    )
    await sc.until(lambda: sc.onboard.ex.state == S.SURVEY, 5, "sökningen pågår")
    await _installed_again(sc)
    await sc.finish()


async def _c(sc: Watched) -> None:
    await _searching(sc)
    down = time.monotonic()
    await sc.flow("link", up=False)
    hold_at = LINK_LOSS_DETECT_S + SDLINK_GRACE_S
    home_at = hold_at + SDLINK_HOLD_S
    await sc.until(lambda: sc.onboard.ex.state == S.HOLD, hold_at + 5, "HOLD ombord")
    held = time.monotonic() - down
    sc.criterion(
        "SC14.c.hold",
        _near(held, hold_at) and sc.onboard.ex.hold_reason == "link_loss",
        "hold vid länkbortfall",
        measured=round(held, 2),
        limit=hold_at,
        unit="s",
    )
    t, mode = await wait_mode(
        sc, (expect.SKILL_RTL_MODE, CopterMode.RTL), home_at - held + 5, "hem"
    )
    sc.criterion(
        "SC14.c.home",
        _near(t - down, home_at) and mode == expect.SKILL_RTL_MODE,
        "hemflygning efter grace + hold",
        measured=[round(t - down, 2), CopterMode(mode).name],
        limit=home_at,
        unit="s",
    )
    await landed_home(sc, "SC14.c.land", t)
    await sc.flow("link", up=True)
    await _installed_again(sc)
    await sc.finish()


async def _d(sc: Watched) -> None:
    await sc.arrange_airborne()
    await asyncio.sleep(AI_RATE_S)
    await sc.say("Flyg till P1.")
    await sc.wait_command("goto", "running", 10, issuer_kind="brain")
    await asyncio.sleep(5)
    await sc.flow("link", up=False)
    await asyncio.sleep(1.0)
    pressed = await sc.press("stop")
    t, _ = await wait_mode(sc, (expect.STOP_MODE,), expect.L_STOP_HB_S, "STOPP via MAVLink")
    sc.criterion(
        "SC14.d.stop",
        t - pressed <= expect.L_STOP_HB_S,
        "STOPP når flygkontrollern via MAVLink",
        measured=round(t - pressed, 2),
        limit=expect.L_STOP_HB_S,
        unit="s",
    )
    await sc.until(
        lambda: sc.tel.groundspeed_ms < expect.STILL_MS, expect.L_STOP_STILL_S, "står still"
    )
    first = len(sc.tl.sent)
    h, v = await sc.drift(30.0)
    steering = sc.steering_since(first, ("onboard",))
    sc.criterion(
        "SC14.d.still",
        h <= expect.L_HOLD_H_M and v <= expect.L_HOLD_V_M and not steering,
        "står still 30 s utan styrning från 1/191",
        measured=[round(h, 2), round(v, 2), len(steering)],
    )
    await sc.flow("link", up=True)
    home = await sc.press("return_home")
    await wait_mode(sc, (expect.HOME_MODE, CopterMode.RTL), expect.L_STOP_HB_S + 2, "FLYG HEM")
    await landed_home(sc, "SC14.d.land", home)
    await _installed_again(sc)
    await sc.finish()


VARIANTS = {"a": _a, "c": _c, "d": _d}


@pytest.mark.budget_s(260)
@pytest.mark.parametrize("variant", sorted(VARIANTS))
async def test_sc14(variant: str, sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario_x(f"test_sc14[{variant}]", sim, migrated_db, tmp_path, "SC14") as sc:
        await VARIANTS[variant](sc)
