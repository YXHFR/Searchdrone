"""SD-link on the ground: one session per vehicle with the onboard computer.

``GroundLink`` runs ``sd_link.LinkServer`` on ``[sdlink] bind`` (loopback) and is the
ground station's ``SdLink`` (link state, the latest ``StateSnapshot``) and the emergency
path's ``NoticeSink`` (STOP / FLY HOME notices, releases, takeovers).

* **Hello:** the action numbers move above the onboard computer's last one; a latch the
  ground has goes back in ``HelloAck``; a latch only the onboard side has is adopted (the
  stricter side wins) and reported, so the mission flow can follow it.
* **Inbound:** snapshots are kept with the time they arrived; everything else
  (``Ack``, ``CharterAck``, ``OnboardEvent``) goes to the ``InboundHandler`` the mission
  flow installs. Connect and disconnect are reported too, so the flow can install the
  charter again after a reconnect.
* **Commander status** goes to the onboard side at 1 Hz (``CommanderStatus``), so it
  knows whether the AI is there.

The emergency path never waits here: ``VehicleActions`` sends its notices from the main
loop's pump, and STOP / FLY HOME reach the flight controller over MAVLink regardless.
Only a take-over notice waits for its ``Ack`` (``send_acked``), in a task of its own, so
that its LOITER reaches the flight controller after the onboard side knows about it.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Protocol

from sd_ground.logging import get_logger
from sd_ground.services.latches import ADOPTED_BY, Latches
from sd_ground.sources import PartState
from sd_link.server import LinkServer
from sd_link.session import Closed, Expired, Inbound, LinkBusyError, LinkClosedError, LinkSession
from sd_link.session import LinkState as SessionState
from sd_link.tokens import TokenStore
from sd_obs.clock import Clock
from sd_schemas.api import LinkStatus
from sd_schemas.base import SdModel, new_id
from sd_schemas.link import (
    SCHEMA_MAJOR,
    Ack,
    CommanderStatus,
    Hello,
    HelloAck,
    LatchRef,
    StateSnapshot,
)
from sd_schemas.mission import OperatorLatch

__all__ = ["COMMANDER_STATUS_S", "GroundLink", "InboundHandler"]

COMMANDER_STATUS_S: Final = 1.0
ACK_MEMORY: Final = 64  # executor Acks kept per vehicle for a send_acked that waits late
# The executor's own answers. "received" is the link's receipt before the executor has it.
_EXECUTOR_ACKS: Final = frozenset({"done", "rejected", "failed", "expired", "superseded"})
log = get_logger("sd_ground.link")


class InboundHandler(Protocol):
    async def on_inbound(self, vehicle_id: str, inbound: Inbound) -> None: ...

    async def on_connected(self, vehicle_id: str, hello: Hello) -> None: ...

    async def on_disconnected(self, vehicle_id: str) -> None: ...

    async def on_expired(self, vehicle_id: str, expired: Expired) -> None: ...


@dataclass(frozen=True, slots=True)
class SnapshotSeen:
    snapshot: StateSnapshot
    received_mono_ms: int


class GroundLink:
    def __init__(
        self,
        *,
        clock: Clock,
        latches: Latches,
        tokens: TokenStore,
        vehicle_ids: list[str],
        bind: str = "127.0.0.1:7600",
        commander_state: Callable[[], str] = lambda: "down",
        on_event: Callable[[str, str, str, dict[str, Any]], None] | None = None,
    ) -> None:
        host, _, port = bind.rpartition(":")
        self.clock = clock
        self.latches = latches
        self.vehicle_ids = list(vehicle_ids)
        self.handler: InboundHandler | None = None
        self.server_epoch = new_id()
        self._commander_state = commander_state
        self._on_event = on_event
        self._snapshots: dict[str, SnapshotSeen] = {}
        self._hellos: dict[str, Hello] = {}
        self._ack_waiters: dict[tuple[str, uuid.UUID], asyncio.Future[str]] = {}
        self._acked: dict[str, OrderedDict[uuid.UUID, str]] = {}  # vehicle -> msg_id -> stage
        self.server = LinkServer(
            tokens=tokens,
            is_known_vehicle=lambda v: v in self.vehicle_ids,
            on_hello=self._on_hello,
            on_session=self._on_session,
            clock=clock,
            host=host,
            port=int(port),
            on_event=lambda name, vid: self._event(name, "info", vid, {}),
        )

    @classmethod
    def from_file(cls, path: Path, **kw: Any) -> GroundLink:
        return cls(tokens=TokenStore.load(path), **kw)

    async def start(self) -> None:
        await self.server.start()

    async def stop(self) -> None:
        await self.server.stop()

    @property
    def port(self) -> int:
        return self.server.port

    def _event(self, type_: str, severity: str, vehicle_id: str, payload: dict[str, Any]) -> None:
        if self._on_event is not None:
            self._on_event(type_, severity, vehicle_id, payload)

    # --- sources.SdLink ---------------------------------------------------------------

    def _session(self, vehicle_id: str) -> LinkSession | None:
        return self.server.session_for(vehicle_id)

    def state(self) -> PartState:
        states = [self._state_of(v) for v in self.vehicle_ids]
        if any(s == "up" for s in states):
            return "ok"
        return "degraded" if any(s == "degraded" for s in states) else "down"

    def _state_of(self, vehicle_id: str) -> str:
        session = self._session(vehicle_id)
        if session is None:
            return "lost"
        if session.state is SessionState.UP:
            return "up"
        return "degraded" if session.state is SessionState.DEGRADED else "lost"

    def link_status(self, vehicle_id: str) -> LinkStatus:
        state = self._state_of(vehicle_id)
        session = self._session(vehicle_id)
        age = None
        if session is not None:
            age = max(0, round((self.clock.monotonic() - session.last_inbound) * 1000))
        return LinkStatus.model_validate(
            {
                "sdlink": state,
                "mavlink": "lost",  # filled in by the caller from the MAVLink thread
                "onboard_heartbeat_age_ms": age,
                "rtt_ms": None,
            }
        )

    def snapshot(self, vehicle_id: str) -> StateSnapshot | None:
        seen = self._snapshots.get(vehicle_id)
        return None if seen is None else seen.snapshot

    def snapshot_seen(self, vehicle_id: str) -> SnapshotSeen | None:
        return self._snapshots.get(vehicle_id)

    def hello(self, vehicle_id: str) -> Hello | None:
        return self._hellos.get(vehicle_id)

    # --- NoticeSink and the mission flow's sends --------------------------------------

    def up(self, vehicle_id: str) -> bool:
        return self._state_of(vehicle_id) == "up"

    async def send(self, vehicle_id: str, message: SdModel) -> bool:
        return await self.send_message(vehicle_id, message) is not None

    async def send_acked(
        self, vehicle_id: str, message: SdModel, timeout_s: float
    ) -> Literal["acked", "refused", "timeout", "down"]:
        """Send one message and wait for the onboard executor's ``Ack`` that names it, the
        send and the wait together within ``timeout_s`` (a take-over notice before its
        LOITER). ``done`` is ``acked``; ``rejected``, ``expired`` and the like ``refused``.

        The timeout runs on the event loop's time and only bounds this helper; the
        take-over's own deadline is kept by the MAVLink thread on its clock."""
        try:
            async with asyncio.timeout(timeout_s):
                msg_id = await self.send_message(vehicle_id, message)
                if msg_id is None:
                    return "down"
                key = (vehicle_id, msg_id)
                stage = self._acked.get(vehicle_id, {}).get(msg_id)  # answered while queued
                if stage is None:
                    waiter: asyncio.Future[str] = asyncio.get_running_loop().create_future()
                    self._ack_waiters[key] = waiter
                    try:
                        stage = await waiter
                    finally:
                        self._ack_waiters.pop(key, None)
        except TimeoutError:
            return "timeout"
        return "acked" if stage == "done" else "refused"

    def _on_ack(self, vehicle_id: str, ack: Ack) -> None:
        if ack.stage not in _EXECUTOR_ACKS:
            return
        msg_id = uuid.UUID(str(ack.ref_id))
        acked = self._acked.setdefault(vehicle_id, OrderedDict())
        acked[msg_id] = ack.stage
        while len(acked) > ACK_MEMORY:
            acked.popitem(last=False)
        key = (vehicle_id, msg_id)
        waiter = self._ack_waiters.get(key)
        if waiter is not None and not waiter.done():
            waiter.set_result(ack.stage)

    async def send_message(
        self,
        vehicle_id: str,
        message: SdModel,
        *,
        mission_id: str | None = None,
        ttl_ms: int | None = None,
    ) -> uuid.UUID | None:
        """Send one message; the envelope's ``msg_id``, or ``None`` when the link is not
        up or the message could not be queued."""
        session = self._session(vehicle_id)
        if session is None or session.state is not SessionState.UP:
            return None
        try:
            envelope = await session.send(
                message,
                mission_id=None if mission_id is None else uuid.UUID(str(mission_id)),
                ttl_ms=ttl_ms,
            )
        except (LinkBusyError, LinkClosedError, ValueError) as exc:
            log.warning("sdlink.send_failed", kind=type(message).__name__, error=type(exc).__name__)
            return None
        return envelope.msg_id

    # --- the server's callbacks -------------------------------------------------------

    async def _on_hello(self, vehicle_id: str, hello: Hello) -> HelloAck:
        self._hellos[vehicle_id] = hello
        self.latches.raise_seq(vehicle_id, hello.last_action_seq)
        ours = self.latches.get(vehicle_id)
        theirs = hello.operator_latch
        if ours is None and theirs is not None:
            adopted = OperatorLatch(
                vehicle_id=vehicle_id,
                action=theirs.action,
                action_seq=theirs.action_seq,
                set_by=ADOPTED_BY,
                set_at=self.clock.utc_now(),
                set_mono_ms=self.clock.monotonic_ms(),
            )
            self.latches.adopt(adopted)
            ours = adopted
            self._event(
                "alarm.LATCH_ADOPTED",
                "warning",
                vehicle_id,
                {"action": theirs.action, "action_seq": theirs.action_seq},
            )
        ref = None if ours is None else LatchRef(action=ours.action, action_seq=ours.action_seq)
        return HelloAck(
            schema_major=SCHEMA_MAJOR, server_epoch=self.server_epoch, operator_latch=ref
        )

    async def _on_session(self, session: LinkSession, hello: Hello) -> None:
        vid = session.vehicle_id
        self._event("vehicle.sdlink_up", "info", vid, {"boot_id": hello.boot_id})
        status = asyncio.ensure_future(self._commander_status(session))
        if self.handler is not None:
            await _quietly(self.handler.on_connected(vid, hello))
        try:
            while True:
                event = await session.next_event()
                if isinstance(event, Closed):
                    break
                if isinstance(event, Inbound):
                    if isinstance(event.body, StateSnapshot):
                        self._snapshots[vid] = SnapshotSeen(event.body, self.clock.monotonic_ms())
                    elif isinstance(event.body, Ack):
                        self._on_ack(vid, event.body)
                    if self.handler is not None:
                        await _quietly(self.handler.on_inbound(vid, event))
                elif isinstance(event, Expired) and self.handler is not None:
                    await _quietly(self.handler.on_expired(vid, event))
        finally:
            status.cancel()
            with contextlib.suppress(BaseException):
                await status
            self._event("alarm.SDLINK_LOST", "warning", vid, {})
            if self.handler is not None:
                await _quietly(self.handler.on_disconnected(vid))

    async def _commander_status(self, session: LinkSession) -> None:
        while True:
            state = self._commander_state()
            body = CommanderStatus.model_validate(
                {
                    "connected": state != "down",
                    "state": state if state in ("ready", "busy", "degraded") else "down",
                }
            )
            with contextlib.suppress(LinkBusyError, LinkClosedError, ValueError):
                await session.send(body)
            await self.clock.sleep(COMMANDER_STATUS_S)


async def _quietly(work: Awaitable[None]) -> None:
    """A failing handler must not end the session."""
    try:
        await work
    except Exception as exc:
        log.error("sdlink.handler_failed", error=type(exc).__name__, detail=str(exc)[:200])
