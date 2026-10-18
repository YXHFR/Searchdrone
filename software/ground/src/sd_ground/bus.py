"""The internal event bus: one place where live events get their ``seq``.

Every event except ``telemetry`` gets the next ``seq`` of this server start (shared by all
topics) and is kept in a ring of the latest 2 000, so a WebSocket client that reconnects
with the same ``server_epoch`` and a ``since_seq`` still in the ring gets exactly what it
missed. Subscribers are bounded queues; a subscriber that falls too far behind is dropped
(its WebSocket closes and the client resynchronises), so one slow client never blocks the
server.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from sd_schemas.base import new_id

__all__ = ["PERIODIC", "RING_SIZE", "BusEvent", "EventBus", "Subscription"]

RING_SIZE = 2000
# The periodic streams (telemetry 5 Hz, vehicle.state 2 Hz) carry no seq and stay out of
# the replay ring: the next sample replaces a missed one.
PERIODIC = frozenset({"telemetry", "vehicle.state"})
QUEUE_SIZE = 1000


@dataclass(frozen=True)
class BusEvent:
    seq: int | None
    ts: dt.datetime
    topic: str
    type: str
    vehicle_id: str | None
    mission_id: str | None
    data: dict[str, Any]

    def envelope(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts.isoformat().replace("+00:00", "Z"),
            "topic": self.topic,
            "type": self.type,
            "vehicle_id": self.vehicle_id,
            "mission_id": self.mission_id,
            "data": self.data,
        }


@dataclass
class Subscription:
    topics: frozenset[str] | None  # None = every topic
    queue: asyncio.Queue[BusEvent | None] = field(
        default_factory=lambda: asyncio.Queue[BusEvent | None](QUEUE_SIZE)
    )
    dropped: bool = False

    def wants(self, event: BusEvent) -> bool:
        return self.topics is None or event.topic in self.topics


class EventBus:
    """Lives in the main event loop. ``publish`` never blocks."""

    def __init__(self) -> None:
        self.server_epoch = str(new_id())
        self._seq = 0
        self._ring: deque[BusEvent] = deque(maxlen=RING_SIZE)
        self._subs: list[Subscription] = []

    @property
    def last_seq(self) -> int:
        return self._seq

    def publish(
        self,
        topic: str,
        type_: str,
        data: dict[str, Any],
        *,
        ts: dt.datetime,
        vehicle_id: str | None = None,
        mission_id: str | None = None,
    ) -> BusEvent:
        seq: int | None = None
        if type_ not in PERIODIC:
            self._seq += 1
            seq = self._seq
        event = BusEvent(seq, ts, topic, type_, vehicle_id, mission_id, data)
        if seq is not None:
            self._ring.append(event)
        for sub in list(self._subs):
            if not sub.wants(event):
                continue
            try:
                sub.queue.put_nowait(event)
            except asyncio.QueueFull:
                self._drop(sub)
        return event

    def _drop(self, sub: Subscription) -> None:
        sub.dropped = True
        if sub in self._subs:
            self._subs.remove(sub)
        while not sub.queue.empty():
            sub.queue.get_nowait()
        sub.queue.put_nowait(None)

    def subscribe(self, topics: Iterable[str] | None = None) -> Subscription:
        sub = Subscription(None if topics is None else frozenset(topics))
        self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        if sub in self._subs:
            self._subs.remove(sub)

    def replay(self, server_epoch: str | None, since_seq: int | None) -> list[BusEvent] | None:
        """The events after ``since_seq``, or ``None`` when the client must resynchronise."""
        if server_epoch != self.server_epoch or since_seq is None:
            return None
        if since_seq > self._seq:
            return None
        if since_seq == self._seq:
            return []
        if not self._ring or self._ring[0].seq is None or self._ring[0].seq > since_seq + 1:
            return None
        return [e for e in self._ring if e.seq is not None and e.seq > since_seq]
