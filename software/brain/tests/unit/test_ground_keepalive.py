"""A pooled connection the ground's server closes as it is reused (``sd_brain.ground_http``).

The ground's server closes a connection idle for 5 s; httpx kept one as long, so a turn's
request could go out on a connection the server was closing and fail with no answer
(``RemoteProtocolError``): the turn was lost (SK09 seed 3 in CI). A small server here does
the same with a short delay: a request on a connection idle for at least ``CLOSE_AFTER_S``
is met by the connection closing without an answer, or by a reset (``ReadError``).
"""

from __future__ import annotations

import asyncio
import socket
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import pytest

from sd_brain.ground_client import GroundClient, GroundError
from sd_brain.ground_http import GROUND_KEEPALIVE_S, ground_http
from sd_schemas.commander import GROUND_HTTP_KEEPALIVE_S

CLOSE_AFTER_S = 0.3
IDLE_S = 3 * CLOSE_AFTER_S  # a wait well past the server's idle limit, on a slow runner too


@dataclass
class Server:
    port: int = 0
    connections: int = 0
    answered: int = 0
    stale: int = 0  # requests met by a close: the server was closing the idle connection
    methods: list[str] | None = None
    reset: bool = False  # a stale request is met by a reset (RST), not a plain close

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


async def _read_request(reader: asyncio.StreamReader) -> str | None:
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, ConnectionError):
        return None
    lines = head.decode("latin-1").split("\r\n")
    length = 0
    for line in lines[1:]:
        name, _, value = line.partition(":")
        if name.strip().lower() == "content-length":
            length = int(value.strip())
    if length:
        await reader.readexactly(length)
    return lines[0].split(" ")[0]


@pytest.fixture
async def server() -> AsyncIterator[Server]:
    state = Server(methods=[])
    loop = asyncio.get_running_loop()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        state.connections += 1
        last: float | None = None
        try:
            while True:
                method = await _read_request(reader)
                if method is None:
                    return
                if last is not None and loop.time() - last >= CLOSE_AFTER_S:
                    state.stale += 1  # closed as it was reused: no answer at all
                    if state.reset:
                        sock = writer.get_extra_info("socket")
                        sock.setsockopt(
                            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                        )
                    return
                assert state.methods is not None
                state.methods.append(method)
                state.answered += 1
                body = b"[]"
                writer.write(
                    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                    + f"content-length: {len(body)}\r\n\r\n".encode()
                    + body
                )
                await writer.drain()
                last = loop.time()
        finally:
            writer.close()

    srv = await asyncio.start_server(handle, "127.0.0.1", 0)
    state.port = srv.sockets[0].getsockname()[1]
    async with srv:
        yield state


async def test_a_reused_connection_the_server_is_closing_gets_no_answer(server: Server) -> None:
    """The failure itself: httpx's own idle limit (5 s) reuses the connection."""
    async with httpx.AsyncClient(trust_env=False) as client:
        assert (await client.get(server.url + "/a")).status_code == 200
        await asyncio.sleep(IDLE_S)
        with pytest.raises(httpx.RemoteProtocolError):
            await client.get(server.url + "/a")
    assert server.stale == 1


async def test_a_get_on_a_stale_connection_is_asked_again(server: Server) -> None:
    """A GET changes nothing on the ground: asked once more on a new connection."""
    g = GroundClient(server.url, "t", httpx.AsyncClient(trust_env=False))
    try:
        assert await g.skills() == []
        await asyncio.sleep(IDLE_S)
        assert await g.skills() == []
    finally:
        await g.aclose()
    assert server.stale == 1 and server.connections == 2 and server.answered == 2


async def test_a_post_on_a_stale_connection_is_not_sent_twice(server: Server) -> None:
    """An order or a post may have reached the ground: its caller hears of the failure."""
    g = GroundClient(server.url, "t", httpx.AsyncClient(trust_env=False))
    try:
        assert await g.skills() == []
        await asyncio.sleep(IDLE_S)
        with pytest.raises(GroundError):
            await g._request("POST", "/missions/m/decisions", json={})  # pyright: ignore[reportPrivateUsage]
    finally:
        await g.aclose()
    assert server.stale == 1 and server.methods == ["GET"]


async def test_idle_connections_are_dropped_before_the_server_drops_them(server: Server) -> None:
    """The ground's client keeps an idle connection shorter than the server: the next
    request opens a new one and nothing is lost (here with the delays scaled down)."""
    assert GROUND_KEEPALIVE_S <= GROUND_HTTP_KEEPALIVE_S / 2
    async with ground_http(keepalive_s=CLOSE_AFTER_S / 2, trust_env=False) as client:
        assert (await client.get(server.url + "/a")).status_code == 200
        await asyncio.sleep(IDLE_S)
        assert (await client.get(server.url + "/a")).status_code == 200
    assert server.stale == 0 and server.connections == 2


async def test_a_get_on_a_reset_connection_is_asked_again(server: Server) -> None:
    """A connection reset under a GET (``ReadError``) is as stale as one closed."""
    server.reset = True
    g = GroundClient(server.url, "t", httpx.AsyncClient(trust_env=False))
    try:
        assert await g.skills() == []
        await asyncio.sleep(IDLE_S)
        assert await g.skills() == []
    finally:
        await g.aclose()
    assert server.stale == 1 and server.connections == 2 and server.answered == 2


async def test_a_post_on_a_reset_connection_is_not_sent_twice(server: Server) -> None:
    server.reset = True
    g = GroundClient(server.url, "t", httpx.AsyncClient(trust_env=False))
    try:
        assert await g.skills() == []
        await asyncio.sleep(IDLE_S)
        with pytest.raises(GroundError):
            await g._request("POST", "/missions/m/decisions", json={})  # pyright: ignore[reportPrivateUsage]
    finally:
        await g.aclose()
    assert server.stale == 1 and server.methods == ["GET"]


async def test_a_get_named_in_lower_case_is_asked_again(server: Server) -> None:
    g = GroundClient(server.url, "t", httpx.AsyncClient(trust_env=False))
    try:
        await g._request("get", "/skills")  # pyright: ignore[reportPrivateUsage]
        await asyncio.sleep(IDLE_S)
        assert (await g._request("get", "/skills")).status_code == 200  # pyright: ignore[reportPrivateUsage]
    finally:
        await g.aclose()
    assert server.stale == 1 and server.answered == 2


async def test_a_stale_get_asked_again_is_logged_with_its_method_and_path(
    server: Server, caplog: pytest.LogCaptureFixture
) -> None:
    g = GroundClient(server.url, "t", httpx.AsyncClient(trust_env=False))
    try:
        await g.skills()
        await asyncio.sleep(IDLE_S)
        with caplog.at_level("INFO", logger="sd_brain.ground"):
            await g.skills()
    finally:
        await g.aclose()
    assert "GET /skills: RemoteProtocolError, frågar igen" in caplog.text


async def test_the_ground_client_never_goes_through_a_proxy_from_the_environment() -> None:
    async with ground_http() as client:
        assert client.trust_env is False
