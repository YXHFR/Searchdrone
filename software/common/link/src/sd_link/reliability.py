"""Acknowledgement, resend and idempotency, shared by both sides.

Pure logic: the caller passes ``now`` (monotonic seconds from its ``Clock``).

* A reliable message without an ``Ack`` is resent after ``SDLINK_RESEND_S``, at most
  ``SDLINK_MAX_RESENDS`` times. A resend keeps ``msg_id``, gets a new ``seq`` and its
  ``ttl_ms`` reduced by the time since the first send. A message whose ttl has run out
  is not sent again. After the last attempt it expires (``LINK_NO_ACK``).
* The receiver remembers the last ``DEDUP_WINDOW`` msg_ids. A duplicate gets the same
  ``Ack`` again and is not handed to the application.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Final

from sd_link.constants import DEDUP_WINDOW, SDLINK_MAX_RESENDS, SDLINK_RESEND_S
from sd_schemas.link import Ack, AckStage, Envelope

__all__ = [
    "ACK_STAGE_RANK",
    "TERMINAL_STAGES",
    "DedupCache",
    "Pending",
    "ResendQueue",
    "ack_keys",
    "advance_stage",
    "inbound_deadline",
]

TERMINAL_STAGES: Final[frozenset[AckStage]] = frozenset(
    {"rejected", "done", "failed", "superseded", "expired"}
)
# A later Ack never moves a status backwards; terminal stages are final.
ACK_STAGE_RANK: Final[dict[AckStage, int]] = {
    "received": 1,
    "accepted": 2,
    "running": 3,
    "rejected": 4,
    "done": 4,
    "failed": 4,
    "superseded": 4,
    "expired": 4,
}


def advance_stage(current: AckStage | None, new: AckStage) -> AckStage | None:
    """The stage after an Ack arrives. Returns ``current`` when ``new`` would go
    backwards or ``current`` is already terminal."""
    if current is None:
        return new
    if current in TERMINAL_STAGES:
        return current
    return new if ACK_STAGE_RANK[new] > ACK_STAGE_RANK[current] else current


def ack_keys(envelope: Envelope) -> tuple[uuid.UUID, ...]:
    """The ids an Ack may refer to: the msg_id, and the command_id of a SkillCommand."""
    if envelope.kind == "SkillCommand":
        command_id = uuid.UUID(str(envelope.body["command_id"]))
        return (envelope.msg_id, command_id)
    return (envelope.msg_id,)


def inbound_deadline(received_at: float, ttl_ms: int | None) -> float | None:
    """ttl is counted from receipt with the receiver's monotonic clock."""
    return None if ttl_ms is None else received_at + ttl_ms / 1000.0


@dataclass(slots=True)
class Pending:
    envelope: Envelope  # as last sent
    first_sent: float
    last_sent: float
    first_ttl_ms: int | None
    resends: int = 0
    keys: tuple[uuid.UUID, ...] = ()


@dataclass(slots=True)
class ResendQueue:
    """Reliable messages sent on one connection and not yet acknowledged."""

    resend_s: float = SDLINK_RESEND_S
    max_resends: int = SDLINK_MAX_RESENDS
    _by_msg: dict[uuid.UUID, Pending] = field(default_factory=dict[uuid.UUID, Pending])
    _by_key: dict[uuid.UUID, uuid.UUID] = field(default_factory=dict[uuid.UUID, uuid.UUID])

    def __len__(self) -> int:
        return len(self._by_msg)

    def __iter__(self) -> Iterator[Pending]:
        return iter(list(self._by_msg.values()))

    def track(self, envelope: Envelope, now: float) -> None:
        if not envelope.reliable:
            raise ValueError("only reliable messages are tracked")
        keys = ack_keys(envelope)
        self._by_msg[envelope.msg_id] = Pending(envelope, now, now, envelope.ttl_ms, 0, keys)
        for key in keys:
            self._by_key[key] = envelope.msg_id

    def ack(self, ref_id: uuid.UUID) -> Envelope | None:
        """Stop resending what ``ref_id`` refers to. Any Ack stage stops the resend."""
        msg_id = self._by_key.get(ref_id)
        if msg_id is None:
            return None
        return self._remove(msg_id).envelope

    def due(self, now: float, next_seq: Callable[[], int]) -> tuple[list[Envelope], list[Envelope]]:
        """Messages to send again now, and messages that have expired (LINK_NO_ACK)."""
        resend: list[Envelope] = []
        expired: list[Envelope] = []
        for pending in list(self._by_msg.values()):
            # Anchored to the first send so that tick granularity does not add up.
            if now - pending.first_sent < (pending.resends + 1) * self.resend_s - 1e-9:
                continue
            ttl = self._remaining_ttl(pending, now)
            if pending.resends >= self.max_resends or (ttl is not None and ttl <= 0):
                expired.append(self._remove(pending.envelope.msg_id).envelope)
                continue
            again = pending.envelope.model_copy(update={"seq": next_seq(), "ttl_ms": ttl})
            pending.envelope = again
            pending.last_sent = now
            pending.resends += 1
            resend.append(again)
        return resend, expired

    def drain(self) -> list[Envelope]:
        """Everything still unacknowledged, oldest first; the queue is emptied.
        Used when the connection ends: nothing is resent on a new connection."""
        pending = sorted(self._by_msg.values(), key=lambda p: p.first_sent)
        self._by_msg.clear()
        self._by_key.clear()
        return [p.envelope for p in pending]

    def _remaining_ttl(self, pending: Pending, now: float) -> int | None:
        if pending.first_ttl_ms is None:
            return None
        return pending.first_ttl_ms - round((now - pending.first_sent) * 1000)

    def _remove(self, msg_id: uuid.UUID) -> Pending:
        pending = self._by_msg.pop(msg_id)
        for key in pending.keys:
            self._by_key.pop(key, None)
        return pending


@dataclass(slots=True)
class _Seen:
    keys: tuple[uuid.UUID, ...]
    ack: Ack | None = None


@dataclass(slots=True)
class DedupCache:
    """The last ``size`` reliable msg_ids from one sender, each with the last Ack sent
    for it, so that a duplicate gets the same answer and no new effect."""

    size: int = DEDUP_WINDOW
    _seen: OrderedDict[uuid.UUID, _Seen] = field(default_factory=OrderedDict[uuid.UUID, _Seen])
    _by_key: dict[uuid.UUID, uuid.UUID] = field(default_factory=dict[uuid.UUID, uuid.UUID])

    def __contains__(self, msg_id: uuid.UUID) -> bool:
        return msg_id in self._seen

    def __len__(self) -> int:
        return len(self._seen)

    def remember(self, envelope: Envelope) -> None:
        keys = ack_keys(envelope)
        self._seen[envelope.msg_id] = _Seen(keys)
        for key in keys:
            self._by_key[key] = envelope.msg_id
        while len(self._seen) > self.size:
            old_id, old = self._seen.popitem(last=False)
            for key in old.keys:
                if self._by_key.get(key) == old_id:
                    del self._by_key[key]

    def record_ack(self, ack: Ack) -> None:
        """Remember the latest Ack sent for a message (matched by msg_id or command_id)."""
        msg_id = self._by_key.get(ack.ref_id)
        if msg_id is not None:
            self._seen[msg_id].ack = ack

    def last_ack(self, msg_id: uuid.UUID) -> Ack | None:
        seen = self._seen.get(msg_id)
        return None if seen is None else seen.ack
