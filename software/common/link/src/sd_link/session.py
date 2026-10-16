"""One SD-link connection after the handshake, the same on both sides.

The session numbers outgoing messages, tracks reliable ones until they are
acknowledged, resends and expires them, answers duplicates with the same ``Ack``,
sends ``Heartbeat`` at 1 Hz and follows the link state from inbound traffic:
``up`` -> ``degraded`` after ``LINK_LOSS_DETECT_S`` of silence -> ``lost`` after
``LINK_LOST_S``, when the connection is closed with 1001.

The application reads ``LinkEvent`` values from :meth:`LinkSession.next_event`.
Nothing is ever resent on a new connection: when a session ends, its unacknowledged
reliable messages are handed back in ``Closed.unacked`` and the application decides
(the ground marks old orders ``expired``/``LINK_RECONNECT``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from sd_link.clock import Clock
from sd_link.close_codes import CloseCode
from sd_link.codes import LINK_NO_ACK, SCHEMA_INVALID
from sd_link.constants import (
    HEARTBEAT_INTERVAL_S,
    LINK_LOSS_DETECT_S,
    LINK_LOST_S,
    MAX_PENDING_RELIABLE,
    TICK_S,
)
from sd_link.envelope import GROUND_SENDER as _GROUND
from sd_link.envelope import (
    DecodeError,
    Side,
    build_envelope,
    decode,
    encode,
    may_send,
    peer_sender,
)
from sd_link.envelope import vehicle_sender as _vehicle_sender
from sd_link.reliability import DedupCache, ResendQueue, ack_keys, inbound_deadline
from sd_link.transport import Transport, TransportClosedError
from sd_schemas.base import SdModel
from sd_schemas.gate import GateDecision
from sd_schemas.link import Ack, AckStage, Envelope, Heartbeat

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
    """A reliable message we sent got no Ack in time (``reason`` = ``LINK_NO_ACK``)."""

    envelope: Envelope
    reason: str


@dataclass(frozen=True, slots=True)
class StateChanged:
    state: LinkState


@dataclass(frozen=True, slots=True)
class Closed:
    """The connection ended. ``unacked``: reliable messages that never got an Ack."""

    code: int | None
    reason: str
    unacked: Sequence[Envelope] = field(default_factory=tuple[Envelope, ...])


LinkEvent = Inbound | Expired | StateChanged | Closed


class LinkClosedError(Exception):
    pass


class LinkBusyError(Exception):
    """Too many unacknowledged reliable messages on this connection."""


class LinkSession:
    def __init__(
        self,
        transport: Transport,
        *,
        side: Side,
        vehicle_id: str,
        clock: Clock,
        seq_start: int = 1,
        degraded_after_s: float = LINK_LOSS_DETECT_S,
        lost_after_s: float = LINK_LOST_S,
        heartbeat_s: float = HEARTBEAT_INTERVAL_S,
        tick_s: float = TICK_S,
    ) -> None:
        self.side: Side = side
        self.vehicle_id = vehicle_id
        self._transport = transport
        self._clock = clock
        self._sender = _GROUND if side == "ground" else _vehicle_sender(vehicle_id)
        self._peer = peer_sender(side, vehicle_id)
        self._peer_side: Side = "vehicle" if side == "ground" else "ground"
        self._seq = seq_start
        self._degraded_after = degraded_after_s
        self._lost_after = lost_after_s
        self._heartbeat_s = heartbeat_s
        self._tick_s = tick_s
        self._pending = ResendQueue()
        self._seen = DedupCache()
        self._events: asyncio.Queue[LinkEvent] = asyncio.Queue()
        self._state = LinkState.UP
        now = clock.now()
        self._last_inbound = now
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
        return await self._events.get()

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

        Raises ValueError for a kind this side may not send or a bad ttl,
        LinkBusyError when too much is unacknowledged and LinkClosedError when closed.
        """
        kind = type(body).__name__
        if not may_send(self.side, kind) or kind in ("Hello", "HelloAck"):
            raise ValueError(f"{self.side} may not send {kind} on a session")
        if self._closing or self._state is LinkState.CLOSED:
            raise LinkClosedError("the link is closed")
        async with self._send_lock:
            envelope = build_envelope(
                body,
                sender=self._sender,
                vehicle_id=self.vehicle_id,
                seq=self._next_seq(),
                sent_utc=self._clock.utc_now(),
                mission_id=mission_id,
                ttl_ms=ttl_ms,
                corr_id=corr_id,
            )
            text = encode(envelope)
            if envelope.reliable:
                if len(self._pending) >= MAX_PENDING_RELIABLE:
                    raise LinkBusyError(f"{len(self._pending)} messages without Ack")
                self._pending.track(envelope, self._clock.now())
            await self._write(text)
        return envelope

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
        SkillCommand). Terminal stages are reliable. The Ack is remembered so that a
        duplicate of the message gets this answer again."""
        body = Ack(
            ref_id=ref_id,
            stage=stage,
            reason_codes=list(reason_codes),
            detail_sv=detail_sv,
            gate=gate,
        )
        self._seen.record_ack(body)
        return await self.send(body, mission_id=mission_id, corr_id=ref_id)

    async def close(self, code: int = CloseCode.GOING_AWAY, reason: str = "") -> None:
        if self._closing:
            return
        self._closing = True
        self._close_code, self._close_reason = code, reason
        with contextlib.suppress(TransportClosedError):
            await self._transport.close(code, reason)

    # --- running -----------------------------------------------------------------

    async def run(self) -> None:
        """Read and tick until the connection ends; then emit ``Closed``."""
        ticker = asyncio.ensure_future(self._tick_loop())
        try:
            await self._read_loop()
        finally:
            ticker.cancel()
            await asyncio.gather(ticker, return_exceptions=True)
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

    async def _tick_loop(self) -> None:
        while not self._closing:
            await self._clock.sleep(self._tick_s)
            await self.tick()

    async def tick(self) -> None:
        """Heartbeat, resends, expiry and the link state. Called every ``tick_s``."""
        if self._closing:
            return
        now = self._clock.now()
        silence = now - self._last_inbound
        if silence >= self._lost_after:
            self._set_state(LinkState.LOST)
            log.warning("sdlink lost: vehicle=%s silence=%.1fs", self.vehicle_id, silence)
            await self.close(CloseCode.GOING_AWAY, "link lost")
            return
        if silence >= self._degraded_after and self._state is LinkState.UP:
            self._set_state(LinkState.DEGRADED)
        if now - self._last_heartbeat >= self._heartbeat_s:
            self._last_heartbeat = now
            await self._send_quietly(Heartbeat())
        async with self._send_lock:
            resend, expired = self._pending.due(now, self._next_seq)
            for envelope in resend:
                await self._write(encode(envelope))
        for envelope in expired:
            log.warning("sdlink no ack: kind=%s msg_id=%s", envelope.kind, envelope.msg_id)
            self._events.put_nowait(Expired(envelope, LINK_NO_ACK))

    # --- receiving ---------------------------------------------------------------

    async def _on_frame(self, frame: str | bytes) -> None:
        now = self._clock.now()
        self._last_inbound = now
        if self._state is LinkState.DEGRADED:
            self._set_state(LinkState.UP)
        try:
            envelope, body = decode(frame)
        except DecodeError as exc:
            log.warning("sdlink bad frame: vehicle=%s %s", self.vehicle_id, exc)
            if exc.reliable and exc.msg_id is not None:
                await self._reject(exc.msg_id, exc.reason)
            return
        if envelope.sender != self._peer or envelope.vehicle_id != self.vehicle_id:
            log.error("sdlink sender mismatch: expected %s/%s", self._peer, self.vehicle_id)
            await self.close(CloseCode.FORBIDDEN, "sender mismatch")
            return
        if not may_send(self._peer_side, envelope.kind) or envelope.kind in ("Hello", "HelloAck"):
            log.warning("sdlink unexpected kind from %s: %s", self._peer, envelope.kind)
            if envelope.reliable:
                await self._reject(envelope.msg_id, SCHEMA_INVALID)
            return
        if isinstance(body, Heartbeat):
            return
        if isinstance(body, Ack):
            self._pending.ack(body.ref_id)
        if envelope.reliable:
            if envelope.msg_id in self._seen:
                await self._repeat_ack(envelope)
                return
            self._seen.remember(envelope)
            await self._ack_received(envelope)
        self._events.put_nowait(
            Inbound(envelope, body, now, inbound_deadline(now, envelope.ttl_ms))
        )

    async def _ack_received(self, envelope: Envelope) -> None:
        ref = ack_keys(envelope)[-1]  # command_id for a SkillCommand, else msg_id
        await self._send_quietly(
            Ack(ref_id=ref, stage="received"), corr_id=ref, mission_id=envelope.mission_id
        )
        self._seen.record_ack(Ack(ref_id=ref, stage="received"))

    async def _repeat_ack(self, envelope: Envelope) -> None:
        last = self._seen.last_ack(envelope.msg_id)
        if last is None:
            last = Ack(ref_id=ack_keys(envelope)[-1], stage="received")
        await self._send_quietly(last, corr_id=last.ref_id, mission_id=envelope.mission_id)

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
        try:
            await self.send(body, corr_id=corr_id, mission_id=mission_id)
        except (LinkClosedError, LinkBusyError, TransportClosedError) as exc:
            log.debug("sdlink could not send %s: %s", type(body).__name__, exc)

    async def _write(self, text: str) -> None:
        try:
            await self._transport.send(text)
        except TransportClosedError:
            # The read loop sees the close and ends the session; tracked messages are
            # handed back in Closed.unacked.
            log.debug("sdlink write after close: vehicle=%s", self.vehicle_id)

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

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
