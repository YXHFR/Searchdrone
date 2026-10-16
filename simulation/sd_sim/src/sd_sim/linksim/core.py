"""The simulated radio link: MAVLink (UDP), SD-link (TCP) and video (UDP) in one place.

Everything that goes over the radio in reality goes through here, so "radio down"
breaks all three flows at the same time.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from sd_sim.tlog import TlogWriter
from sd_sim.udprelay import UdpRelay

PROFILES_YAML = Path(__file__).with_name("profiles.yaml")
FLOWS = ("mavlink", "link", "video")
# A TCP outage longer than this resets the old connections when the link comes back,
# so both ends notice and sd-onboard reconnects with a fresh Hello.
TCP_RESET_AFTER_S = 3.0
_READ_CHUNK = 65536


@dataclass(frozen=True)
class Profile:
    name: str
    delay_ms: float
    jitter_ms: float
    loss_pct: float


def load_profiles(path: Path = PROFILES_YAML) -> dict[str, Profile]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        name: Profile(
            name=name,
            delay_ms=float(row["delay_ms"]),
            jitter_ms=float(row["jitter_ms"]),
            loss_pct=float(row["loss_pct"]),
        )
        for name, row in raw.items()
    }


class LinkState:
    """Profile, random source and the up/down switch of each flow."""

    def __init__(self, profile: Profile, seed: int = 1) -> None:
        self.profile = profile
        self.rng = random.Random(seed)  # noqa: S311 - fault injection, not security
        self.up: dict[str, bool] = dict.fromkeys(FLOWS, True)

    def set_profile(self, profile: Profile, seed: int) -> None:
        self.profile = profile
        self.rng = random.Random(seed)  # noqa: S311 - fault injection, not security

    def udp_policy(self, flow: str):
        def decide(_direction: str) -> float | None:
            if not self.up[flow]:
                return None
            p = self.profile
            if p.loss_pct >= 100 or (p.loss_pct > 0 and self.rng.random() * 100 < p.loss_pct):
                return None
            jitter = self.rng.uniform(-p.jitter_ms, p.jitter_ms) if p.jitter_ms else 0.0
            return max(0.0, p.delay_ms + jitter) / 1000.0

        return decide

    def tcp_delay_s(self) -> float:
        return max(0.0, self.profile.delay_ms) / 1000.0

    def tcp_dead(self) -> bool:
        return self.profile.loss_pct >= 100

    def udp_up(self, flow: str):
        return lambda: self.up[flow] and self.profile.loss_pct < 100


def _reset(writer: asyncio.StreamWriter) -> None:
    """Close with RST instead of FIN (SO_LINGER 0)."""
    sock = writer.get_extra_info("socket")
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    writer.transport.abort()


class _Pair:
    def __init__(
        self,
        flow: TcpFlow,
        vehicle: tuple[asyncio.StreamReader, asyncio.StreamWriter],
        ground: tuple[asyncio.StreamReader, asyncio.StreamWriter],
    ) -> None:
        self.flow = flow
        self.vehicle = vehicle
        self.ground = ground
        self.tasks: list[asyncio.Task[None]] = []

    def start(self) -> None:
        self.tasks = [
            asyncio.create_task(self._pump(self.vehicle[0], self.ground[1])),
            asyncio.create_task(self._pump(self.ground[0], self.vehicle[1])),
        ]
        done = asyncio.gather(*self.tasks, return_exceptions=True)
        done.add_done_callback(lambda _f: self._finished())

    def _finished(self) -> None:
        """Both directions ended (FIN both ways): close the sockets and forget the pair."""
        for _, writer in (self.vehicle, self.ground):
            writer.close()
        self.flow.forget(self)

    async def _pump(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        queue: asyncio.Queue[tuple[float, bytes]] = asyncio.Queue()
        sender = asyncio.create_task(self._send(queue, writer))
        try:
            while True:
                await self.flow.wait_up()
                data = await reader.read(_READ_CHUNK)
                # Bytes (or the FIN) read just before the link went down are held back.
                await self.flow.wait_up()
                due = time.monotonic() + self.flow.state.tcp_delay_s()
                await queue.put((due, data))
                if not data:
                    break
        except (ConnectionError, OSError):
            pass
        except asyncio.CancelledError:
            sender.cancel()
            raise
        await queue.join()
        sender.cancel()

    async def _send(
        self, queue: asyncio.Queue[tuple[float, bytes]], writer: asyncio.StreamWriter
    ) -> None:
        while True:
            due, data = await queue.get()
            try:
                wait = due - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                await self.flow.wait_up()
                if data:
                    writer.write(data)
                    await writer.drain()
                    self.flow.bytes += len(data)
                elif writer.can_write_eof():
                    writer.write_eof()
            except (ConnectionError, OSError):
                pass
            finally:
                queue.task_done()

    def reset(self) -> None:
        self.flow.forget(self)
        for task in self.tasks:
            task.cancel()
        _reset(self.vehicle[1])
        _reset(self.ground[1])


class TcpFlow:
    """TCP proxy for SD-link. Loss and jitter never apply; delay keeps the order."""

    def __init__(self, listen_port: int, target_port: int, state: LinkState) -> None:
        self.listen_port = listen_port
        self.target_port = target_port
        self.state = state
        self.connections = 0
        self.bytes = 0
        self._pairs: list[_Pair] = []
        self._pending: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []
        self._up = asyncio.Event()
        self._up.set()
        self._down_since: float | None = None
        self._server: asyncio.Server | None = None

    def is_up(self) -> bool:
        return self.state.up["link"] and not self.state.tcp_dead()

    async def wait_up(self) -> None:
        await self._up.wait()

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._accept, host="127.0.0.1", port=self.listen_port
        )
        self.sync()  # e.g. started with the dead profile

    def forget(self, pair: _Pair) -> None:
        if pair in self._pairs:
            self._pairs.remove(pair)

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        if not self.is_up():
            self._pending.append((reader, writer))
            return
        await self._connect((reader, writer))

    async def _connect(self, vehicle: tuple[asyncio.StreamReader, asyncio.StreamWriter]) -> None:
        try:
            ground = await asyncio.open_connection("127.0.0.1", self.target_port)
        except OSError:
            _reset(vehicle[1])
            return
        pair = _Pair(self, vehicle, ground)
        self._pairs.append(pair)
        pair.start()

    def sync(self) -> None:
        """Apply a change of the up/down switch or the profile."""
        if self.is_up() and not self._up.is_set():
            down_for = time.monotonic() - (self._down_since or time.monotonic())
            if down_for > TCP_RESET_AFTER_S:
                for pair in list(self._pairs):
                    pair.reset()
            self._down_since = None
            self._up.set()
            pending, self._pending = self._pending, []
            for vehicle in pending:
                asyncio.create_task(self._connect(vehicle))  # noqa: RUF006 - kept in _pairs
        elif not self.is_up() and self._up.is_set():
            self._down_since = time.monotonic()
            self._up.clear()

    def close(self) -> None:
        if self._server is not None:
            self._server.close()
        for pair in list(self._pairs):
            pair.reset()
        for _, writer in self._pending:
            writer.transport.abort()


class LinkSim:
    def __init__(
        self,
        ports: dict[str, int],
        profile: Profile,
        *,
        seed: int = 1,
        radio_tlog: Path | None = None,
    ) -> None:
        self.state = LinkState(profile, seed)
        self.mavlink = UdpRelay(
            name="mavlink",
            listen_port=ports["radio_mavlink_in"],
            target_port=ports["ground_mavlink"],
            policy=self.state.udp_policy("mavlink"),
            is_up=self.state.udp_up("mavlink"),
            tap=TlogWriter(radio_tlog) if radio_tlog is not None else None,
        )
        self.video = UdpRelay(
            name="video",
            listen_port=ports["video_raw_udp"],
            target_port=ports["video_out_udp"],
            one_way=True,
            policy=self.state.udp_policy("video"),
            is_up=self.state.udp_up("video"),
        )
        self.link = TcpFlow(ports["linksim_sdlink"], ports["ground_sdlink"], self.state)

    async def start(self) -> None:
        await self.mavlink.start()
        await self.video.start()
        await self.link.start()

    def set_flow(self, flow: str, up: bool) -> None:
        self.state.up[flow] = up
        self.link.sync()

    def set_radio(self, up: bool) -> None:
        for flow in FLOWS:
            self.state.up[flow] = up
        self.link.sync()

    def set_profile(self, profile: Profile, seed: int) -> None:
        self.state.set_profile(profile, seed)
        self.link.sync()

    def snapshot(self) -> dict[str, object]:
        return {
            "profile": self.state.profile.name,
            "flows": {
                "mavlink": {"up": self.state.up["mavlink"], **self.mavlink.state()},
                "video": {"up": self.state.up["video"], **self.video.state()},
                "link": {
                    "up": self.state.up["link"],
                    "connections": self.link.connections,
                    "bytes": self.link.bytes,
                },
            },
        }

    def flush(self) -> None:
        if self.mavlink.tap is not None:
            self.mavlink.tap.flush()

    def close(self) -> None:
        self.mavlink.close()
        self.video.close()
        self.link.close()
