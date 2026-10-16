"""One SD-link connection after the handshake, the same on both sides.

The session numbers outgoing messages, tracks reliable ones until they are
acknowledged, resends and expires them, answers duplicates with the same ``Ack``,
sends ``Heartbeat`` at 1 Hz and follows the link state from inbound traffic:
``up`` -> ``degraded`` after ``LINK_LOSS_DETECT_S`` of silence -> ``lost`` after
``LINK_LOST_S``, when the connection is closed with 1001.

Safety rules the session keeps:

* **Expiry is final on the wire.** When a reliable message expires (no Ack in time),
  the connection is aborted at once, with everything still queued in it, so a stale
  order, release or clearance can never arrive later. The application gets
  ``Expired`` for it and ``Closed.unacked`` for the rest, and resynchronises on the
  next connection. Nothing is ever resent on a new connection.
* **Liveness never waits for writes.** Link state and expiry are evaluated by their
  own task. A write blocked longer than ``WRITE_TIMEOUT_S`` aborts the connection.
* **Stick values are never replayed.** ``ManualControl`` is sent best-effort with
  :meth:`LinkSession.offer` (dropped unless the link is up and the writer free). On
  receipt only the newest value is kept for the application, and values arriving
  within ``MANUAL_QUARANTINE_S`` after a stall (no frame for ``MANUAL_STALL_GAP_S``)
  are dropped, because they are an old burst.

Delivery order holds when the application waits for an Ack before sending the next
order (the ground sends one movement order at a time); a resend can overtake.

The application reads ``LinkEvent`` values with :meth:`LinkSession.next_event` until
``Closed``. Events not read within ``MAX_EVENT_BACKLOG`` close the connection.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from sd_link.clock import Clock, first_or_timeout
from sd_link.close_codes import CloseCode
from sd_link.codes import LINK_NO_ACK, SCHEMA_INVALID
from sd_link.constants import (
    CLOSE_TIMEOUT_S,
    HEARTBEAT_INTERVAL_S,
    LINK_LOSS_DETECT_S,
    LINK_LOST_S,
    MANUAL_QUARANTINE_S,
    MANUAL_STALL_GAP_S,
    MAX_EVENT_BACKLOG,
    MAX_INVALID_FRAMES,
    MAX_PENDING_RELIABLE,
    TICK_S,
    WRITE_TIMEOUT_S,
)
from sd_link.envelope import (
    GROUND_SENDER,
    DecodeError,
    Side,
    build_envelope,
    decode,
    encode,
    kind_of,
    may_send,
    peer_sender,
    vehicle_sender,
)
from sd_link.reliability import DedupCache, ResendQueue, ack_keys, inbound_deadline
from sd_link.transport import ABORT_CODE, Transport, TransportClosedError
from sd_schemas.base import SdModel
from sd_schemas.gate import GateDecision
from sd_schemas.link import Ack, AckStage, Envelope, Heartbeat, ManualControl

__all__ = [
    "Closed",
    "Expired",
    "Inbound",
    "LinkBusyError",
    "LinkClosedError",
    "LinkEvent",
    "LinkSession",
    "LinkState",
    "StateChanged",
]

log = logging.getLogger("sd_link")

_HANDSHAKE_KINDS: Final = frozenset({"Hello", "HelloAck"})
_BEST_EFFORT_ONLY: Final = frozenset({"ManualControl"})


class LinkState(StrEnum):
    UP = "up"
    DEGRADED = "degraded"
    LOST = "lost"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class Inbound:
    """A new message from the peer (duplicates and heartbeats are not delivered)."""

    envelope: Envelope
    body: SdModel
    received_at: float  # receiver's monotonic clock
    deadline: float | None  # received_at + ttl; None without ttl

    def expired(self, now: float) -> bool:
        return self.deadline is not None and now >= self.deadline


@dataclass(frozen=True, slots=True)
class Expired:
    """A reliable message we sent got no Ack in time (``reason`` = ``LINK_NO_ACK``).
    The connection is aborted right after, so the message cannot arrive later."""

    envelope: Envelope
    reason: str


@dataclass(frozen=True, slots=True)
class StateChanged:
    state: LinkState


@dataclass(frozen=True, slots=True)
class Closed:
    """The connection ended. ``unacked``: reliable messages that never got an Ack and
    were not reported as ``Expired``."""

    code: int | None
    reason: str
    unacked: Sequence[Envelope] = field(default_factory=tuple[Envelope, ...])


LinkEvent = Inbound | Expired | StateChanged | Closed


class LinkClosedError(Exception):
    pass


class LinkBusyError(Exception):
    """Too many unacknowledged reliable messages on this connection."""


class _ManualSlot:
    """Marker in the event queue: the newest stick value is in ``_latest_manual``."""


_MANUAL_SLOT: Final = _ManualSlot()


class LinkSession:
    def __init__(
        self,
        transport: Transport,
        *,
        side: Side,
        vehicle_id: str,
        clock: Clock,
        seq_start: int = 1,
        dedup: DedupCache | None = None,
        degraded_after_s: float = LINK_LOSS_DETECT_S,
        lost_after_s: float = LINK_LOST_S,
        heartbeat_s: float = HEARTBEAT_INTERVAL_S,
        tick_s: float = TICK_S,
    ) -> None:
        self.side: Side = side
        self.vehicle_id = vehicle_id
        self._transport = transport
        self._clock = clock
        self._sender = GROUND_SENDER if side == "ground" else vehicle_sender(vehicle_id)
        self._peer = peer_sender(side, vehicle_id)
        self._peer_side: Side = "vehicle" if side == "ground" else "ground"
        self._seq = seq_start
        self._degraded_after = degraded_after_s
        self._lost_after = lost_after_s
        self._heartbeat_s = heartbeat_s
        self._tick_s = tick_s
        self._pending = ResendQueue()
        self._seen = dedup if dedup is not None else DedupCache()
        self._events: asyncio.Queue[LinkEvent | _ManualSlot] = asyncio.Queue()
        self._latest_manual: Inbound | None = None
        self._backlog = 0  # Inbound/Expired events not yet read
        self._invalid = 0
        self._state = LinkState.UP
        now = clock.now()
        self._last_inbound = now
        self._manual_quarantine_until = now
        self._last_heartbeat = now - heartbeat_s  # first heartbeat on the first tick
        self._close_code: int | None = None
        self._close_reason = ""
        self._closing = False
        self._send_lock = asyncio.Lock()

    # --- state -------------------------------------------------------------------

    @property
    def state(self) -> LinkState:
        return self._state

    @property
    def last_inbound(self) -> float:
        """Monotonic time of the last frame from the peer (any frame, heartbeat too)."""
        return self._last_inbound

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def next_event(self) -> LinkEvent:
        """The next event. After ``Closed`` no more events come."""
        while True:
            item = await self._events.get()
            event = self._take(item)
            if event is not None:
                return event

    def events_nowait(self) -> list[LinkEvent]:
        """Every event already waiting, without blocking."""
        events: list[LinkEvent] = []
        while not self._events.empty():
            event = self._take(self._events.get_nowait())
            if event is not None:
                events.append(event)
        return events

    def _take(self, item: LinkEvent | _ManualSlot) -> LinkEvent | None:
        if isinstance(item, _ManualSlot):
            event, self._latest_manual = self._latest_manual, None
            return event
        if isinstance(item, Inbound | Expired):
            self._backlog -= 1
        return item

    # --- sending -----------------------------------------------------------------

    async def send(
        self,
        body: SdModel,
        *,
        mission_id: uuid.UUID | None = None,
        ttl_ms: int | None = None,
        corr_id: uuid.UUID | None = None,
    ) -> Envelope:
        """Send one message. Reliable kinds are tracked until acknowledged.

        Raises ValueError for a kind this side may not send (or ``ManualControl``,
        which goes through :meth:`offer`) or a bad ttl, LinkBusyError when too much
        is unacknowledged and LinkClosedError when closed.
        """
        kind = kind_of(body)
        if kind in _BEST_EFFORT_ONLY:
            raise ValueError(f"{kind} is best-effort: use offer()")
        return await self._send(body, mission_id, ttl_ms, corr_id, budget=True)

    async def offer(
        self, body: SdModel, *, ttl_ms: int | None, mission_id: uuid.UUID | None = None
    ) -> bool:
        """Send an unreliable value best-effort (stick values). Dropped (False) unless
        the link is up and nothing else is being written; never queued or resent."""
        kind = kind_of(body)
        if kind not in _BEST_EFFORT_ONLY:
            raise ValueError(f"{kind} is not a best-effort kind")
        if self._state is not LinkState.UP or self._closing or self._send_lock.locked():
            return False
        try:
            await self._send(body, mission_id, ttl_ms, None, budget=False)
        except LinkClosedError:
            return False
        return True

    async def ack(
        self,
        ref_id: uuid.UUID,
        stage: AckStage,
        *,
        reason_codes: Sequence[str] = (),
        detail_sv: str | None = None,
        gate: GateDecision | None = None,
        mission_id: uuid.UUID | None = None,
    ) -> Envelope:
        """Acknowledge a message (``ref_id`` = its msg_id, or the command_id of a
        SkillCommand). Terminal stages are reliable. The Ack frame is remembered so
        that a duplicate of the message gets exactly this answer again."""
        body = Ack(
            ref_id=ref_id,
            stage=stage,
            reason_codes=list(reason_codes),
            detail_sv=detail_sv,
            gate=gate,
        )
        return await self._send(body, mission_id, None, ref_id, budget=True)

    async def _send(
        self,
        body: SdModel,
        mission_id: uuid.UUID | None,
        ttl_ms: int | None,
        corr_id: uuid.UUID | None,
        *,
        budget: bool,
    ) -> Envelope:
        kind = kind_of(body)
        if not may_send(self.side, kind) or kind in _HANDSHAKE_KINDS:
            raise ValueError(f"{self.side} may not send {kind} on a session")
        async with self._send_lock:
            if self._closing:
                raise LinkClosedError("the link is closed")
            # Validate and encode before the seq is used, so refusals leave no gap.
            envelope = build_envelope(
                body,
                sender=self._sender,
                vehicle_id=self.vehicle_id,
                seq=self._seq,
                sent_utc=self._clock.utc_now(),
                mission_id=mission_id,
                ttl_ms=ttl_ms,
                corr_id=corr_id,
            )
            text = encode(envelope)
            if envelope.reliable:
                if budget and len(self._pending) >= MAX_PENDING_RELIABLE:
                    raise LinkBusyError(f"{len(self._pending)} messages without Ack")
                self._pending.track(envelope, self._clock.now())
            self._seq += 1
            if isinstance(body, Ack):
                self._seen.record_ack(body, text)
            await self._write(text)
        return envelope

    async def close(self, code: int = CloseCode.GOING_AWAY, reason: str = "") -> None:
        """Graceful close (queued frames go first), at most ``CLOSE_TIMEOUT_S``, then
        abort."""
        if self._closing:
            return
        self._mark_closing(code, reason)
        done, _ = await first_or_timeout(
            self._clock, lambda: self._close_transport(code, reason), CLOSE_TIMEOUT_S
        )
        if not done:
            self._transport.abort()

    def abort(self, code: int = ABORT_CODE, reason: str = "aborted") -> None:
        """Drop the connection now, with everything still queued in it."""
        if not self._closing:
            self._mark_closing(code, reason)
        self._transport.abort()

    async def _close_transport(self, code: int, reason: str) -> None:
        with contextlib.suppress(TransportClosedError):
            await self._transport.close(code, reason)

    def _mark_closing(self, code: int, reason: str) -> None:
        self._closing = True
        self._close_code, self._close_reason = code, reason

    # --- running -----------------------------------------------------------------

    async def run(self) -> None:
        """Read, watch and tick until the connection ends; then emit ``Closed``."""
        helpers = [
            asyncio.ensure_future(self._loop("watch", self.watch)),
            asyncio.ensure_future(self._loop("tick", self.tick)),
        ]
        try:
            await self._read_loop()
        finally:
            for task in helpers:
                task.cancel()
            await asyncio.gather(*helpers, return_exceptions=True)
            self._finish()

    async def _read_loop(self) -> None:
        while True:
            try:
                frame = await self._transport.recv()
            except TransportClosedError as exc:
                if self._close_code is None:
                    self._close_code, self._close_reason = exc.code, exc.reason
                return
            await self._on_frame(frame)

    async def _loop(self, name: str, step: Callable[[], Awaitable[None]]) -> None:
        try:
            while not self._closing:
                await self._clock.sleep(self._tick_s)
                await step()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.critical(
                "sdlink %s loop failed; aborting: vehicle=%s", name, self.vehicle_id, exc_info=True
            )
            self.abort(ABORT_CODE, f"{name} failed")

    async def watch(self) -> None:
        """Link state and expiry. Never writes, so a blocked writer cannot stall it."""
        if self._closing:
            return
        now = self._clock.now()
        expired = self._pending.take_expired(now)
        if expired:
            for envelope in expired:
                log.warning("sdlink no ack: kind=%s msg_id=%s", envelope.kind, envelope.msg_id)
                self._put(Expired(envelope, LINK_NO_ACK))
            # The frames may still sit in a buffer: drop them with the connection.
            self.abort(ABORT_CODE, "no ack")
            return
        silence = now - self._last_inbound
        if silence >= self._lost_after:
            self._set_state(LinkState.LOST)
            log.warning("sdlink lost: vehicle=%s silence=%.1fs", self.vehicle_id, silence)
            await self.close(CloseCode.GOING_AWAY, "link lost")
            return
        if silence >= self._degraded_after and self._state is LinkState.UP:
            self._set_state(LinkState.DEGRADED)

    async def tick(self) -> None:
        """Heartbeat and resends."""
        if self._closing:
            return
        now = self._clock.now()
        if now - self._last_heartbeat >= self._heartbeat_s:
            self._last_heartbeat = now
            await self._send_quietly(Heartbeat())
        async with self._send_lock:
            if self._closing:
                return
            for envelope in self._pending.take_resends(self._clock.now(), self._next_seq):
                await self._write(encode(envelope))

    # --- receiving ---------------------------------------------------------------

    async def _on_frame(self, frame: str | bytes) -> None:
        now = self._clock.now()
        if now - self._last_inbound >= MANUAL_STALL_GAP_S:
            self._manual_quarantine_until = now + MANUAL_QUARANTINE_S
        self._last_inbound = now
        if self._state is LinkState.DEGRADED:
            self._set_state(LinkState.UP)
        try:
            envelope, body = decode(frame)
        except DecodeError as exc:
            log.warning("sdlink bad frame: vehicle=%s %s", self.vehicle_id, exc)
            if await self._count_invalid():
                return
            if exc.reliable and exc.msg_id is not None:
                await self._reject(exc.msg_id, exc.reason)
            return
        if envelope.sender != self._peer or envelope.vehicle_id != self.vehicle_id:
            log.error("sdlink sender mismatch: expected %s/%s", self._peer, self.vehicle_id)
            await self.close(CloseCode.FORBIDDEN, "sender mismatch")
            return
        if not may_send(self._peer_side, envelope.kind) or envelope.kind in _HANDSHAKE_KINDS:
            log.warning("sdlink unexpected kind from %s: %s", self._peer, envelope.kind)
            if await self._count_invalid():
                return
            if envelope.reliable:
                await self._reject(envelope.msg_id, SCHEMA_INVALID)
            return
        if isinstance(body, Heartbeat):
            return
        if isinstance(body, Ack):
            self._pending.ack(body.ref_id)
        if envelope.reliable:
            if self._seen.seen(envelope):
                await self._repeat_ack(envelope)
                return
            self._seen.remember(envelope)
            await self._ack_received(envelope)
        inbound = Inbound(envelope, body, now, inbound_deadline(now, envelope.ttl_ms))
        if isinstance(body, ManualControl):
            self._deliver_manual(inbound, now)
            return
        if self._backlog >= MAX_EVENT_BACKLOG:
            log.critical("sdlink events not consumed; closing: vehicle=%s", self.vehicle_id)
            await self.close(CloseCode.POLICY_VIOLATION, "event backlog")
            return
        self._put(inbound)

    def _deliver_manual(self, inbound: Inbound, now: float) -> None:
        if now < self._manual_quarantine_until:
            log.info("sdlink stick value after a stall dropped: vehicle=%s", self.vehicle_id)
            return
        if self._latest_manual is None:
            self._events.put_nowait(_MANUAL_SLOT)
        self._latest_manual = inbound  # only the newest value waits for the application

    async def _count_invalid(self) -> bool:
        self._invalid += 1
        if self._invalid > MAX_INVALID_FRAMES:
            log.critical("sdlink too many invalid frames; closing: vehicle=%s", self.vehicle_id)
            await self.close(CloseCode.POLICY_VIOLATION, "invalid frames")
            return True
        return False

    async def _ack_received(self, envelope: Envelope) -> None:
        ref = ack_keys(envelope)[-1]  # command_id for a SkillCommand, else msg_id
        await self._send_quietly(
            Ack(ref_id=ref, stage="received"), corr_id=ref, mission_id=envelope.mission_id
        )

    async def _repeat_ack(self, envelope: Envelope) -> None:
        frame = self._seen.last_ack_frame(envelope)
        if frame is None:
            await self._ack_received(envelope)
            return
        async with self._send_lock:  # the same frame again, not tracked again
            if not self._closing:
                await self._write(frame)

    async def _reject(self, msg_id: uuid.UUID, reason: str) -> None:
        await self._send_quietly(
            Ack(ref_id=msg_id, stage="rejected", reason_codes=[reason]), corr_id=msg_id
        )

    # --- helpers -----------------------------------------------------------------

    async def _send_quietly(
        self,
        body: SdModel,
        *,
        corr_id: uuid.UUID | None = None,
        mission_id: uuid.UUID | None = None,
    ) -> None:
        """Link-generated messages: not counted against the application's budget."""
        try:
            await self._send(body, mission_id, None, corr_id, budget=False)
        except (LinkClosedError, TransportClosedError, ValueError) as exc:
            log.debug("sdlink could not send %s: %s", type(body).__name__, exc)

    async def _write(self, text: str) -> None:
        """Write one frame. Called with the send lock held. A write that blocks longer
        than WRITE_TIMEOUT_S (a dead link with TCP still open) aborts the connection."""
        try:
            done, _ = await first_or_timeout(
                self._clock, lambda: self._transport.send(text), WRITE_TIMEOUT_S
            )
        except TransportClosedError:
            # The read loop sees the close and ends the session; tracked messages are
            # handed back in Closed.unacked.
            log.debug("sdlink write after close: vehicle=%s", self.vehicle_id)
            return
        if not done:
            log.critical("sdlink write blocked; aborting: vehicle=%s", self.vehicle_id)
            self.abort(ABORT_CODE, "write blocked")

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    def _put(self, event: Inbound | Expired) -> None:
        self._backlog += 1
        self._events.put_nowait(event)

    def _set_state(self, state: LinkState) -> None:
        if state is not self._state:
            self._state = state
            self._events.put_nowait(StateChanged(state))

    def _finish(self) -> None:
        if self._state is LinkState.CLOSED:
            return
        self._closing = True
        unacked = self._pending.drain()
        self._set_state(LinkState.CLOSED)
        self._events.put_nowait(Closed(self._close_code, self._close_reason, tuple(unacked)))
