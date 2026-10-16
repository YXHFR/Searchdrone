"""SD-link (TCP) through linksim: a short outage keeps the connection, a long one resets it.

The real reset threshold is 3 s; the test scales the times down so it runs in a second.
"""

import asyncio
import socket

import pytest

from sd_sim.linksim import core

PROFILES = core.load_profiles()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Ground:
    """A line server standing in for sd-ground: answers HELLO with WELCOME, echoes the rest."""

    def __init__(self) -> None:
        self.accepted = 0
        self.closed = 0

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.accepted += 1
        try:
            while line := await reader.readline():
                writer.write(b"WELCOME\n" if line == b"HELLO\n" else b"ECHO " + line)
                await writer.drain()
        except ConnectionError:
            pass
        finally:
            self.closed += 1
            writer.close()


async def start(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[core.LinkSim, Ground, int, asyncio.Server]:
    monkeypatch.setattr(core, "TCP_RESET_AFTER_S", 0.3)
    ground = Ground()
    gport, lport = free_port(), free_port()
    server = await asyncio.start_server(ground.serve, "127.0.0.1", gport)
    udp = dict.fromkeys(("radio_mavlink_in", "ground_mavlink", "video_raw_udp", "video_out_udp"), 0)
    sim = core.LinkSim(
        {**udp, "linksim_sdlink": lport, "ground_sdlink": gport}, PROFILES["perfect"]
    )
    await sim.start()
    return sim, ground, lport, server


async def hello(port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"HELLO\n")
    await writer.drain()
    assert await asyncio.wait_for(reader.readline(), 2) == b"WELCOME\n"
    return reader, writer


async def test_short_outage_keeps_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    sim, ground, port, server = await start(monkeypatch)
    try:
        reader, writer = await hello(port)
        sim.set_flow("link", False)
        writer.write(b"during\n")
        await writer.drain()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(reader.readline(), 0.1)  # held back while down
        sim.set_flow("link", True)
        assert await asyncio.wait_for(reader.readline(), 2) == b"ECHO during\n"
        assert ground.accepted == 1 and ground.closed == 0
        writer.close()
    finally:
        sim.close()
        server.close()


async def test_long_outage_resets_and_reconnects(monkeypatch: pytest.MonkeyPatch) -> None:
    sim, ground, port, server = await start(monkeypatch)
    try:
        reader, writer = await hello(port)
        sim.set_radio(False)
        await asyncio.sleep(0.5)
        # A new connection while down is accepted but not passed on yet.
        pending = asyncio.create_task(hello(port))
        await asyncio.sleep(0.1)
        assert ground.accepted == 1
        sim.set_radio(True)
        with pytest.raises((ConnectionError, asyncio.IncompleteReadError)):
            data = await asyncio.wait_for(reader.read(100), 2)
            if not data:
                raise ConnectionResetError
        new_reader, new_writer = await asyncio.wait_for(pending, 5)
        assert ground.accepted == 2
        new_writer.write(b"after\n")
        await new_writer.drain()
        assert await asyncio.wait_for(new_reader.readline(), 2) == b"ECHO after\n"
        new_writer.close()
        writer.close()
    finally:
        sim.close()
        server.close()


async def test_delay_keeps_order(monkeypatch: pytest.MonkeyPatch) -> None:
    sim, _ground, port, server = await start(monkeypatch)
    try:
        sim.set_profile(PROFILES["field_typical"], seed=3)  # 40 ms, no loss or jitter for TCP
        reader, writer = await hello(port)
        for i in range(20):
            writer.write(f"m{i}\n".encode())
        await writer.drain()
        got = [await asyncio.wait_for(reader.readline(), 2) for _ in range(20)]
        assert got == [f"ECHO m{i}\n".encode() for i in range(20)]
        writer.close()
    finally:
        sim.close()
        server.close()
