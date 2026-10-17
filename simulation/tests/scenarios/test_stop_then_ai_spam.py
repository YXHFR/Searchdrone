"""After the pilot's STOP, the AI floods the commander API with orders that would move
the vehicle: every one is refused, nothing steering reaches the flight controller, the
vehicle stays in the STOP mode and the stop holds until the pilot releases it."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from scenario_rig import scenario  # pyright: ignore[reportMissingImports]

from sd_sim import expect

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.script("airborne")]

SPAM_S = 20.0


@pytest.mark.budget_s(160)
async def test_stop_then_ai_spam(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario("test_stop_then_ai_spam", sim, migrated_db, tmp_path, "airborne") as sc:
        await sc.arrange_airborne()
        await sc.press("stop")
        await sc.until(lambda: sc.mode() == expect.STOP_MODE, expect.L_STOP_HB_S, "STOPP-läget")
        await sc.until(
            lambda: sc.tel.groundspeed_ms < expect.STILL_MS, expect.L_STOP_STILL_S, "stilla"
        )
        index = len(sc.tl.sent)
        p1, out = sc.pin("P1"), sc.pin("OUT")
        orders: list[tuple[str, dict[str, Any]]] = [
            ("goto", {"lat": p1[0], "lon": p1[1], "alt_m": 40}),
            ("resume", {}),
            ("survey_sector", {"sector_id": "A"}),
            ("goto", {"lat": out[0], "lon": out[1], "alt_m": 40}),
            ("takeoff", {"alt_m": 40}),
            ("hold", {}),
        ]
        decisions = []
        end = time.monotonic() + SPAM_S
        n = 0
        while time.monotonic() < end:
            skill, params = orders[n % len(orders)]
            res = await sc.commander_order(skill, **params)
            assert res.status_code == 201, res.text
            decisions.append((skill, res.json()["decision"], res.json()["reason_codes"]))
            n += 1
            await asyncio.sleep(0.5)
        moving = [d for d in decisions if d[0] != "hold" and d[1] != "REJECT"]
        sc.criterion("SPAM.1", not moving, "varje rörelseorder avvisas", count=len(decisions))
        steering = [m for m in sc.tl.sent[index:] if m.sender == "onboard"]
        steering = [m for m in steering if m.steers() or m.mode is not None]
        sc.criterion("SPAM.2", not steering, "inget styrande från ombord", found=len(steering))
        sc.criterion("SPAM.3", sc.mode() == expect.STOP_MODE, "läget är kvar", mode=sc.mode())
        h, v = await sc.drift(5)
        sc.criterion("SPAM.4", h <= expect.L_HOLD_H_M and v <= expect.L_HOLD_V_M, "står still")
        await sc.finish()
