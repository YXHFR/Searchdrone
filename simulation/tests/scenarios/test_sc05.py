"""SC05: the height limits.

In the air at 40 m the vehicle is ordered up to 200 m above its position (on the AI's
own path), then the AI is asked to fly to HIGH at 150 m. Both are refused with
``ALT_ABOVE_CHARTER`` (and ``ALT_ABOVE_LIMIT``, both are over the hard limit) and the
vehicle keeps its height. The flight controller's own height fence is the one in the
safety parameter file, at most 5 m over the hard limit.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from scenario_rig import AI_RATE_S, Scenario, scenario  # pyright: ignore[reportMissingImports]

from sd_safety.constants import FENCE_ALT_MARGIN_M
from sd_safety.limits import ABSOLUTE_MAX_ALT_AGL_M
from sd_sim import expect

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.script("SC05")]

UP_TO_M = 200.0


@pytest.mark.budget_s(150)
async def test_sc05(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario("test_sc05", sim, migrated_db, tmp_path, "SC05") as sc:
        await _run(sc)


def _refused(body: dict[str, Any]) -> bool:
    codes = body["reason_codes"]
    return body["decision"] == "REJECT" and {"ALT_ABOVE_CHARTER", "ALT_ABOVE_LIMIT"} <= set(codes)


async def _run(sc: Scenario) -> None:
    await sc.arrange_airborne()
    first_sent = len(sc.tl.sent)

    # 1. Straight up to 200 m above the current position (the AI's own path and token).
    t = time.monotonic()
    res = await sc.commander_order("goto", lat=sc.tel.lat, lon=sc.tel.lon, alt_m=UP_TO_M)
    assert res.status_code == 201, res.text
    body = res.json()
    sc.criterion(
        "SC05.1a",
        _refused(body),
        "200 m avvisas",
        codes=body["reason_codes"],
        measured=round(time.monotonic() - t, 2),
        limit=expect.L_GATE_S,
    )
    await sc.assert_drift("SC05.1a-drift")

    # 2. The AI, asked in the chat, to HIGH at 150 m.
    await asyncio.sleep(AI_RATE_S)
    t0 = time.monotonic()
    await sc.say("Flyg till HIGH.")
    cmd = await sc.wait_command("goto", "rejected_by_gate", 10, issuer_kind="brain", after=1)
    codes = await sc.codes(cmd["id"])
    ok = {"ALT_ABOVE_CHARTER", "ALT_ABOVE_LIMIT"} <= set(codes)
    sc.criterion("SC05.1b", ok, "HIGH på 150 m avvisas", codes=codes)
    await sc.wait_chat(
        lambda m: m.get("role") == "brain" and "höjden" in m.get("text", ""), expect.L_CHAT_S
    )
    sc.criterion("SC05.2", True, "befälhavaren förklarar", measured=time.monotonic() - t0)
    await sc.assert_drift("SC05.1b-drift")
    alt = sc.tel.alt_rel_home_m
    sc.criterion("SC05.1c", abs(alt - 40) <= 2, "höjden 40 +- 2 m", measured=round(alt, 1))

    # 3. The flight controller's height fence (read, never written).
    fence = await asyncio.to_thread(sc.sim.harness.read_param, "FENCE_ALT_MAX")
    want = expect.value("FENCE_ALT_MAX")
    sc.criterion(
        "SC05.3",
        abs(fence - want) < 1e-3 and fence <= ABSOLUTE_MAX_ALT_AGL_M + FENCE_ALT_MARGIN_M,
        "FENCE_ALT_MAX som i parameterfilen och inom taket + 5 m",
        measured=fence,
        limit=ABSOLUTE_MAX_ALT_AGL_M + FENCE_ALT_MARGIN_M,
    )
    steering = sc.steering_since(first_sent, ("onboard", "ground"))
    sc.criterion("SC05.4", not steering, "inget styrande till FC", found=len(steering))
    await sc.finish()
