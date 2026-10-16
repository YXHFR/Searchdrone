"""``LossyProxy``: a WebSocket proxy between a real client and a real server that
drops, delays or breaks frames, for tests over real sockets. Binds only 127.0.0.1."""

from __future__ import annotations

import asyncio
import random
from typing import Self

from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from sd_link.constants import MAX_FRAME_BYTES

__all__ = ["LossyProxy"]

_PASS_HEADERS = ("Authorization", "Origin")


class LossyProxy:
    def __init__(
        self,
        upstream_url: str,
        *,
        loss: float = 0.0,
        delay_s: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.upstream_url = upstream_url
        self.loss = loss
        self.delay_s = delay_s
        self.broken = False  # drop every frame both ways while set
        self.dropped = 0
        self._rng = random.Random(seed)  # noqa: S311  # test noise, not crypto
        self._server: Server | None = None
        self._pairs: set[tuple[ServerConnection, ClientConnection]] = set()
        self.port = 0

    @property
    def url(self) -> str:
        path = self.upstream_url.split("/", 3)[3]
        return f"ws://127.0.0.1:{self.port}/{path}"

    async def start(self) -> None:
        self._server = await serve(self._handle, "127.0.0.1", 0, max_size=MAX_FRAME_BYTES)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    async def cut(self, code: int = 1006) -> None:
        """Close every proxied connection (the client sees the link drop)."""
        for down, up in list(self._pairs):
            await up.close(1001)
            await down.close(code if code != 1006 else 1001)

    async def _handle(self, down: ServerConnection) -> None:
        headers = {}
        if down.request is not None:
            for name in _PASS_HEADERS:
                value = down.request.headers.get(name)
                if value is not None:
                    headers[name] = value
        try:
            async with connect(
                self.upstream_url,
                additional_headers=headers,
                max_size=MAX_FRAME_BYTES,
                origin=None,
            ) as up:
                pair = (down, up)
                self._pairs.add(pair)
                try:
                    await asyncio.gather(self._pipe(down, up), self._pipe(up, down))
                finally:
                    self._pairs.discard(pair)
                    if up.close_code is not None and down.close_code is None:
                        await down.close(up.close_code, up.close_reason or "")
        except (OSError, ConnectionClosed):
            await down.close(1011)

    async def _pipe(
        self, src: ServerConnection | ClientConnection, dst: ServerConnection | ClientConnection
    ) -> None:
        try:
            async for frame in src:
                if self.broken or self._rng.random() < self.loss:
                    self.dropped += 1
                    continue
                if self.delay_s:
                    await asyncio.sleep(self.delay_s)
                await dst.send(frame)
        except ConnectionClosed:
            pass
        code = src.close_code
        if dst.close_code is None:
            await dst.close(code if code and code >= 1000 and code != 1006 else 1001)
