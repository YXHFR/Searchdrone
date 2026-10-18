"""SC10: the ground station (``sd-ground``) restarts in the middle of a search.

* a, a short outage (killed and started again at once): its heartbeat from 250 is back
  well within ``FS_GCS_TIMEOUT`` - 2 s, so the GCS failsafe never acts; the search flies
  on (the onboard computer has the order); within 15 s the new ground station is ready
  and shows the mission, the approved charter, the chat and the running order with the
  same ids; the search route is not uploaded again.
* b, a long outage (started again only after ``FS_GCS_TIMEOUT`` + 10 s; the harness's
  own heartbeat, 253, keeps going): the GCS failsafe acts on the missing 250 heartbeat
  anyway; the onboard computer goes to FAILSAFE and steers nothing; after the restart
  the ground shows the flight home and does not resume the search; it lands at home.
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
from sd_schemas.executor import ExecutorState as S
from sd_sim import expect

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.script("SC10")]

READY_AFTER_RESTART_S = 15.0
HEARTBEAT_MARGIN_S = 2.0
FAILSAFE_MODES = frozenset({CopterMode.RTL, CopterMode.SMART_RTL, CopterMode.LAND})


async def _searching(sc: Watched) -> dict[str, Any]:
    await sc.arrange_airborne()
    await asyncio.sleep(AI_RATE_S)
    await sc.say("Sök av rutan.")
    cmd = await sc.wait_command("survey_sector", "running", 30, issuer_kind="brain")
    await sc.until(lambda: sc.tel.fc_custom_mode == CopterMode.AUTO, 30, "AUTO")
    await asyncio.sleep(5)
    return cmd


def _heartbeats_after(sc: Watched, t: float) -> list[float]:
    return [m.t for m in sc.tl.sent if m.sender == "ground" and m.type == "HEARTBEAT" and m.t > t]


async def _a(sc: Watched) -> None:
    survey = await _searching(sc)
    before = await sc.r.mission(sc.mission_id)
    chat_before = [m["id"] for m in await sc.chat()]
    killed = await sc.kill_ground()
    restarted = await sc.start_ground(brain=True)
    timeout = expect.value("FS_GCS_TIMEOUT")
    beats = _heartbeats_after(sc, killed)
    first = beats[0] - killed if beats else None
    sc.criterion(
        "SC10.a.1",
        first is not None and first <= timeout - HEARTBEAT_MARGIN_S,
        "heartbeat från 250 i tid efter omstarten",
        measured=None if first is None else round(first, 2),
        limit=timeout - HEARTBEAT_MARGIN_S,
        unit="s",
    )

    async def healthy() -> bool:
        res = await sc.r.panel.get("/api/v1/health")
        return res.status_code == 200 and res.json()["ready"] is True

    await sc.until(healthy, READY_AFTER_RESTART_S, "/health redo")
    after = await sc.r.mission(sc.mission_id)
    chat_after = [m["id"] for m in await sc.chat()]
    now = await sc.command(survey["id"])
    sc.criterion(
        "SC10.a.4",
        after["mission"]["id"] == before["mission"]["id"]
        and after["mission"]["status"] == "ACTIVE"
        and after["charter"]["id"] == before["charter"]["id"]
        and set(chat_before) <= set(chat_after)
        and now["status"] == "running",
        "samma uppdrag, ram, chatt och pågående order efter omstarten",
        measured=round(time.monotonic() - restarted, 2),
        limit=READY_AFTER_RESTART_S,
    )
    await sc.still(
        lambda: sc.tel.fc_custom_mode == CopterMode.AUTO and sc.onboard.ex.state == S.SURVEY,
        10,
        "sökningen fortsätter",
    )
    modes = {m for t, m in sc.observer.modes if t >= killed} if sc.observer else set()
    sc.criterion(
        "SC10.a.2",
        not (modes & {int(m) for m in FAILSAFE_MODES}),
        "ingen GCS-failsafe",
        modes=sorted(modes),
    )
    sc.criterion("SC10.a.3", True, "sökningen fortsätter efter omstarten")
    later = [m for m in sc.tl.sent if m.type == "MISSION_COUNT" and m.t > killed]
    sc.criterion("SC10.a.5", not later, "sökrutten laddas inte upp igen", n=len(later))
    await sc.finish(["INV9"])


async def _b(sc: Watched) -> None:
    await _searching(sc)
    killed = await sc.kill_ground()  # the last heartbeat from 250 is no later than this
    timeout = expect.value("FS_GCS_TIMEOUT")
    chain = expect.fs_chain("FS_GCS_ENABLE")
    first = len(sc.tl.sent)
    t, mode = await wait_mode(sc, chain, timeout + expect.L_FS_S + 5, "GCS-failsafe")
    sc.criterion(
        "SC10.b.1",
        t - killed <= timeout + expect.L_FS_S and first_choice_or_proof(sc, mode, chain, killed),
        "GCS-failsafen trots harnessens heartbeat",
        measured=[round(t - killed, 2), CopterMode(mode).name],
        limit=timeout + expect.L_FS_S,
        unit="s",
    )
    await sc.until(lambda: sc.onboard.ex.state == S.FAILSAFE, 5, "FAILSAFE ombord")
    await asyncio.sleep(max(0.0, timeout + 10 - (time.monotonic() - killed)))
    steering = sc.steering_since(first, ("onboard",))
    sc.criterion("SC10.b.2", not steering, "ombord FAILSAFE och inget styrande", n=len(steering))
    await sc.start_ground(brain=True)
    await landed_home(sc, "SC10.b.3", t)
    status = await sc.r.status(sc.mission_id)
    cmds = [c for c in await sc.commands() if c["skill"] == "survey_sector"]
    sc.criterion(
        "SC10.b.3b",
        status != "ACTIVE" and len(cmds) == 1,
        "markstationen återupptar inte sökningen",
        status=status,
        surveys=len(cmds),
    )
    await sc.finish(["INV9"])


VARIANTS = {"a": _a, "b": _b}


@pytest.mark.budget_s(230)
@pytest.mark.parametrize("variant", sorted(VARIANTS))
async def test_sc10(variant: str, sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario_x(f"test_sc10[{variant}]", sim, migrated_db, tmp_path, "SC10") as sc:
        await VARIANTS[variant](sc)
