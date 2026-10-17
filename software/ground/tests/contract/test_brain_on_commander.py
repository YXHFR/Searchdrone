"""The real ``sd-brain`` against the real ``/commander/v1`` (B1-10 against B1-12).

Everything on the ground is real (database, mission flow, gate, emergency path, SD-link,
both APIs; ``flow_rig``); only the flight controller and the onboard computer are fakes.
The commander API is served over real HTTP and WebSocket (``brain_rig``), and ``sd-brain``
runs its own startup, event stream and turn loop on it in the fixed command mode. The
pilot is the panel API.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from brain_rig import (  # pyright: ignore[reportMissingImports]
    SaysStop,
    free_port,
    run_brain,
    serve_commander,
)
from flow_rig import VID, H, Rig, rig  # pyright: ignore[reportMissingImports]

from sd_brain.decide.service import DecisionService
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]


@pytest.fixture
async def stack(migrated_db: Db, tmp_path: Path) -> AsyncIterator[tuple[Rig, int]]:
    port = free_port()
    async with (
        rig(
            migrated_db.name,
            migrated_db.app_password,
            tmp_path,
            http={"commander_bind": f"127.0.0.1:{port}"},
        ) as r,
        serve_commander(r, port),
    ):
        yield r, port


async def _say(r: Rig, mid: str, text: str) -> dict[str, Any]:
    res = await r.panel.post(
        f"/api/v1/missions/{mid}/chat",
        json={"client_msg_id": str(new_id()), "text": text},
        headers=H,
    )
    assert res.status_code == 201, res.text
    return res.json()


async def _orders(r: Rig, mid: str) -> list[dict[str, Any]]:
    res = await r.panel.get(f"/api/v1/missions/{mid}/commands")
    assert res.status_code == 200, res.text
    return res.json()


async def _intents(r: Rig, mid: str) -> list[dict[str, Any]]:
    """The ground's ``chat.intent`` events: one per stop-check report it acted on."""
    await r.g.flush_events()
    res = await r.panel.get(f"/api/v1/missions/{mid}/events", params={"type": "chat.intent"})
    assert res.status_code == 200, res.text
    return res.json()


async def _chat(r: Rig, mid: str) -> list[dict[str, Any]]:
    res = await r.panel.get(f"/api/v1/missions/{mid}/chat")
    assert res.status_code == 200, res.text
    return res.json()["items"]


async def test_brain_takes_off_through_the_gate_on_the_real_commander_api(
    stack: tuple[Rig, int],
) -> None:
    r, port = stack
    mid = await r.ready_mission()
    async with run_brain(r, port) as b:
        assert b.loop.mission_id == mid  # found the active mission at startup
        res = await r.clearance(mid)
        assert res.status_code == 200, res.text
        await r.until(lambda: r.onboard.clearances, 5, "clearance onboard")
        await _say(r, mid, "Starta")

        async def took_off() -> bool:
            return any(
                o["skill"] == "takeoff" and o["status"] == "done" for o in await _orders(r, mid)
            )

        await r.until(took_off, 15, "the AI's takeoff done through the gate")
        takeoff = next(o for o in await _orders(r, mid) if o["skill"] == "takeoff")
        assert takeoff["issuer_kind"] == "brain"
        assert [o.skill for o in r.onboard.orders] == ["takeoff"]

        async def answered() -> bool:
            return any(m["role"] == "brain" for m in await _chat(r, mid))

        await r.until(answered, 10, "the AI's chat line after the gate's answer")


async def test_stop_check_stops_through_the_real_chat_intent_route(
    stack: tuple[Rig, int],
) -> None:
    r, port = stack
    mid = await r.ready_mission()
    r.fly()
    r.onboard.set_state(ExecutorState.HOLD)
    async with run_brain(r, port) as b:
        b.loop.decide = DecisionService([SaysStop()], on_answer=b.loop.log_decide)
        line = await _say(r, mid, "vänta, stanna där!")
        assert line["handled_by"] is None  # not the exact words: the AI's check decides

        await r.until(lambda: r.actions.latches.get(VID) is not None, 10, "STOP from the AI")
        latch = r.actions.latches.get(VID)
        assert latch is not None and latch.action == "stop"
        await r.until(lambda: r.onboard.stop_latch, 5, "the notice on the onboard side")

        async def reported() -> bool:
            return bool(await _intents(r, mid))

        await r.until(reported, 5, "the ground's chat.intent event")
        payload = (await _intents(r, mid))[0]["payload"]
        assert payload["engine"] == "jev" and payload["acted"] == "stop"
        assert payload["message_id"] == line["id"]

        async def explained() -> bool:
            return any(m["role"] == "system" for m in await _chat(r, mid))

        await r.until(explained, 5, "the ground says why the drone stopped")
        assert not [o for o in await _orders(r, mid) if o["issuer_kind"] == "brain"]


async def test_exact_stop_word_is_the_grounds_own(stack: tuple[Rig, int]) -> None:
    r, port = stack
    mid = await r.ready_mission()
    r.fly()
    r.onboard.set_state(ExecutorState.HOLD)
    async with run_brain(r, port) as b:
        b.loop.decide = DecisionService([SaysStop()], on_answer=b.loop.log_decide)
        line = await _say(r, mid, "Stopp!")
        assert line["handled_by"] == "ground" and line["parsed_intent"] == "stop"
        latch = r.actions.latches.get(VID)
        assert latch is not None and latch.action == "stop"
        await asyncio.sleep(1.0)  # time for the AI to (not) report the same line
        assert not await _intents(r, mid)  # the ground acted itself; the AI did not report
