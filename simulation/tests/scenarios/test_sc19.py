"""SC19: the operator is gone (no pilot WebSocket) during an active search.

After ``operator_absent_s`` without a pilot connection the ground raises
``OPERATOR_ABSENT`` and holds the vehicle on STOP's path (latch ``watchdog_stop``); the
AI's orders are refused with ``OPERATOR_ABSENT``; it never flies home by itself. When the
pilot is back the vehicle still stands until Resume; FLY HOME then lands it.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from rig_faults import brain_started, presence, tap  # pyright: ignore[reportMissingImports]
from scenario_rig import AI_RATE_S, Scenario, scenario  # pyright: ignore[reportMissingImports]

from sd_safety.constants import OPERATOR_ABSENT_S
from sd_sim import expect

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.script("SC19")]

HOME_MODES = {int(m) for m in (*expect.fs_chain("FS_GCS_ENABLE"), expect.HOME_MODE)}
NO_HOME_S = 60.0
ORDER_AFTER_ALARM_S = 10.0


@pytest.mark.budget_s(310)
async def test_sc19(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with (
        scenario("test_sc19", sim, migrated_db, tmp_path, "SC19") as sc,
        presence(sc),
        tap(sc) as bus,
    ):
        await _run(sc, bus)


async def _run(sc: Scenario, bus: Any) -> None:
    await sc.arrange_airborne()
    await asyncio.sleep(AI_RATE_S)
    await sc.say("Sök av rutan.")
    await brain_started(sc, "survey_sector")
    await sc.r.wait_status(sc.mission_id, "ACTIVE", 10)

    # The pilot's panel closes: no pilot connection is left.
    sc.r.presence[0] = False
    closed = time.monotonic()
    await sc.until(
        lambda: bus.first("alarm.OPERATOR_ABSENT", closed), expect.L_ABSENT_S + 5, "larm"
    )
    alarm = bus.first("alarm.OPERATOR_ABSENT", closed)
    assert alarm is not None
    events = [t for t, kind, _ in sc._bus if kind == "vehicle.operator_absent"]  # pyright: ignore[reportPrivateUsage]
    sc.criterion(
        "SC19.1",
        alarm - closed <= expect.L_ABSENT_S and bool(events),
        "larm OPERATOR_ABSENT och vehicle.operator_absent (L-ABSENT)",
        measured=round(alarm - closed, 1),
        limit=expect.L_ABSENT_S,
        clock="host",
    )
    assert OPERATOR_ABSENT_S + 2 == expect.L_ABSENT_S

    # The hold: STOP's mode on the ground's path, with the latch watchdog_stop.
    took = await sc.until(lambda: sc.mode() == expect.STOP_MODE, expect.L_STOP_HB_S, "STOPP-läget")
    sent = [
        m.t - alarm
        for m in sc.tl.sent
        if m.sender == "ground" and m.mode == expect.STOP_MODE and m.t >= alarm - 0.5
    ]
    latch = sc.r.actions.latches.get("sim-01")
    sc.criterion(
        "SC19.2",
        bool(sent)
        and sent[0] <= expect.L_STOP_ACK_S
        and latch is not None
        and latch.action == "watchdog_stop",
        "STOPP-läget via markens väg (L-STOP-ACK)",
        measured=None if not sent else round(sent[0], 2),
        limit=expect.L_STOP_ACK_S,
        hb_s=round(took, 2),
        latch=None if latch is None else latch.action,
    )

    # 10 s after the alarm the AI tries to move the vehicle: refused.
    await asyncio.sleep(max(0.0, alarm + ORDER_AFTER_ALARM_S - time.monotonic()))
    lat, lon = sc.pin("P1")
    t = time.monotonic()
    res = await sc.commander_order("goto", lat=lat, lon=lon, alt_m=40)
    assert res.status_code == 201, res.text
    body = res.json()
    sc.criterion(
        "SC19.3",
        body["decision"] == "REJECT" and "OPERATOR_ABSENT" in body["reason_codes"],
        "AI:ns goto avvisas med OPERATOR_ABSENT",
        codes=body["reason_codes"],
        measured=round(time.monotonic() - t, 2),
        limit=expect.L_GATE_S,
    )

    # Never home by itself.
    await sc.still(
        lambda: sc.mode() not in HOME_MODES and sc.tel.armed,
        max(0.0, alarm + NO_HOME_S - time.monotonic()),
        "ingen automatisk hemflygning",
    )
    sc.criterion("SC19.4", True, "ingen automatisk hemflygning på 60 s", limit=NO_HOME_S)

    # The pilot is back: the vehicle stands until Resume.
    sc.r.presence[0] = True
    await sc.still(lambda: sc.mode() == expect.STOP_MODE, 5, "står kvar tills Fortsätt")
    await sc.press("resume")
    await sc.until(lambda: sc.mode() != expect.STOP_MODE, 10, "lämnar STOPP-läget efter Fortsätt")
    await asyncio.sleep(1)
    started = await sc.press("return_home")
    await sc.until(lambda: not sc.tel.armed, expect.L_LAND_S, "landad")
    sc.criterion(
        "SC19.5",
        sc.from_home_m() <= expect.L_LAND_M,
        "Fortsätt, sedan FLYG HEM: landar hemma (L-LAND)",
        measured=round(sc.from_home_m(), 1),
        limit=expect.L_LAND_M,
        land_s=round(time.monotonic() - started, 1),
    )
    await sc.finish()
