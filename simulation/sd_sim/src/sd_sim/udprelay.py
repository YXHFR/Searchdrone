"""A UDP relay between the simulated vehicle and a process on the ground side.

The vehicle side sends to ``listen_port``; the relay remembers its address and
forwards every datagram to ``target_port`` from its own socket. Replies that come back
to that socket from the target go to the vehicle's last address. Everything binds
127.0.0.1. ``mavtap`` uses it without faults; ``linksim`` adds delay, jitter and loss.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

from sd_sim.tlog import TlogWriter

LOOPBACK = "127.0.0.1"
FROM_VEHICLE = "from_vehicle"
TO_VEHICLE = "to_vehicle"

# Decides what happens to one datagram: None = drop, otherwise the delay in seconds.
Policy = Callable[[str], float | None]


def _pass(_direction: str) -> float | None:
    return 0.0


@dataclass
class Counters:
    packets_in: int = 0
    packets_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    dropped: int = 0
    rtp_frames: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class UdpRelay:
    name: str
    listen_port: int
    target_port: int
    one_way: bool = False
    policy: Policy = _pass
    tap: TlogWriter | None = None
    counters: dict[str, Counters] = field(
        default_factory=lambda: {FROM_VEHICLE: Counters(), TO_VEHICLE: Counters()}
    )
    _vehicle_addr: tuple[str, int] | None = None
    _vehicle_side: asyncio.DatagramTransport | None = None
    _ground_side: asyncio.DatagramTransport | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        relay = self

        class VehicleSide(asyncio.DatagramProtocol):
            def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
                relay._vehicle_addr = addr
                relay._handle(data, FROM_VEHICLE)

        class GroundSide(asyncio.DatagramProtocol):
            def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
                if relay.one_way or addr[1] != relay.target_port:
                    return
                relay._handle(data, TO_VEHICLE)

        vehicle_side, _ = await loop.create_datagram_endpoint(
            VehicleSide, local_addr=(LOOPBACK, self.listen_port)
        )
        ground_side, _ = await loop.create_datagram_endpoint(GroundSide, local_addr=(LOOPBACK, 0))
        self._vehicle_side = vehicle_side
        self._ground_side = ground_side

    def _handle(self, data: bytes, direction: str) -> None:
        stats = self.counters[direction]
        stats.packets_in += 1
        stats.bytes_in += len(data)
        if direction == FROM_VEHICLE and len(data) >= 2 and data[1] & 0x80:
            stats.rtp_frames += 1  # RTP marker bit: last packet of a video frame
        delay = self.policy(direction)
        if self.tap is not None:
            self.tap.write(data, direction, dropped=delay is None)
        if delay is None:
            stats.dropped += 1
            return
        if delay <= 0:
            self._send(data, direction)
        else:
            asyncio.get_running_loop().call_later(delay, self._send, data, direction)

    def _send(self, data: bytes, direction: str) -> None:
        if direction == FROM_VEHICLE:
            if self._ground_side is None:
                return
            self._ground_side.sendto(data, (LOOPBACK, self.target_port))
        else:
            if self._vehicle_side is None or self._vehicle_addr is None:
                return
            self._vehicle_side.sendto(data, self._vehicle_addr)
        stats = self.counters[direction]
        stats.packets_out += 1
        stats.bytes_out += len(data)

    def close(self) -> None:
        for transport in (self._vehicle_side, self._ground_side):
            if transport is not None:
                transport.close()
        if self.tap is not None:
            self.tap.close()

    def state(self) -> dict[str, dict[str, int]]:
        return {direction: c.as_dict() for direction, c in self.counters.items()}
