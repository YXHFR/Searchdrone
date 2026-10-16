"""WebSocket ``/ws/v1/live`` for the panel.

Authenticated with the session cookie (same origin), or for scripts with
``Authorization: Bearer <CLI token>`` and no ``Origin`` (``PanelGuard`` refuses the
combination). The server says ``hello``; the client subscribes and may resume with
``server_epoch`` + ``since_seq``; otherwise it gets ``resync_required`` and reloads over
REST. ``ping`` every 10 s; 30 s without one closes the socket. A pilot WebSocket counts as
an operator who is present.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field

from sd_ground.api.deps import ground, principal_from
from sd_ground.api.tasks import until_first_done
from sd_ground.bus import BusEvent

__all__ = ["PING_TIMEOUT_S", "router"]

PING_TIMEOUT_S = 30.0
SUBSCRIBE_TIMEOUT_S = 10.0
TOPICS = frozenset(
    {
        "system",
        "telemetry",
        "vehicle",
        "mission",
        "commands",
        "approvals",
        "gate",
        "chat",
        "alerts",
        "detections",
    }
)

router = APIRouter()


class _Subscribe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: str
    topics: list[str] = Field(default_factory=list[str], max_length=len(TOPICS))
    vehicle_id: str | None = None
    mission_id: str | None = None
    server_epoch: str | None = None
    since_seq: int | None = None


def _for_client(event: BusEvent, sub: _Subscribe) -> bool:
    if sub.vehicle_id is not None and event.vehicle_id not in (None, sub.vehicle_id):
        return False
    return not (sub.mission_id is not None and event.mission_id not in (None, sub.mission_id))


@router.websocket("/ws/v1/live")
async def live(ws: WebSocket) -> None:
    g = ground(ws)
    who = principal_from(ws)
    await ws.accept()
    if who is None:
        await ws.close(4401)
        return
    await ws.send_json(
        {
            "topic": "system",
            "type": "hello",
            "seq": None,
            "data": {
                "server_epoch": g.bus.server_epoch,
                "version": g.version,
                "profile": g.config.profile,
            },
        }
    )
    try:
        raw = await asyncio.wait_for(ws.receive_json(), SUBSCRIBE_TIMEOUT_S)
        sub = _Subscribe.model_validate(raw)
        if sub.op != "subscribe" or not set(sub.topics) <= TOPICS:
            raise ValueError("expected subscribe")
    except (TimeoutError, ValueError, WebSocketDisconnect):
        with contextlib.suppress(Exception):
            await ws.close(4400)
        return
    bus_sub = g.bus.subscribe(sub.topics)
    pilot = who.role == "pilot"
    if pilot:
        g.pilot_ws += 1
        g.operator_seen_mono = g.clock.monotonic()
    last_ping = [g.clock.monotonic()]
    try:
        replay = g.bus.replay(sub.server_epoch, sub.since_seq)
        if replay is None:
            await ws.send_json(
                {"type": "resync_required", "data": {"server_epoch": g.bus.server_epoch}}
            )
        else:
            for event in replay:
                if bus_sub.wants(event) and _for_client(event, sub):
                    await ws.send_json(event.envelope())

        async def reader() -> None:
            while True:
                msg: dict[str, Any] = await ws.receive_json()
                if msg.get("op") == "ping":
                    last_ping[0] = g.clock.monotonic()
                    if pilot:
                        g.operator_seen_mono = last_ping[0]
                    pong: dict[str, Any] = {"type": "pong"}
                    if isinstance(msg.get("client_ns"), int):
                        pong["server_ns"] = time.clock_gettime_ns(time.CLOCK_REALTIME)
                    await ws.send_json(pong)

        async def writer() -> None:
            while True:
                event = await bus_sub.queue.get()
                if event is None:  # fell behind: the client resynchronises
                    await ws.close(1001)
                    return
                if _for_client(event, sub):
                    await ws.send_json(event.envelope())

        async def watchdog() -> None:
            while g.clock.monotonic() - last_ping[0] < PING_TIMEOUT_S:
                await g.clock.sleep(1.0)
            await ws.close(1001)

        await until_first_done(reader, writer, watchdog)
    finally:
        g.bus.unsubscribe(bus_sub)
        if pilot:
            g.pilot_ws -= 1
