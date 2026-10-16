"""linksim's control API on 127.0.0.1 (simulator only).

With mirrored WSL networking any web page in the Windows browser can send a simple
POST to localhost, so every request needs (1) ``Authorization: Bearer <linksim_token>``,
(2) a ``Host`` of ``127.0.0.1:<port>`` or ``localhost:<port>`` and (3) for PUT/POST
``Content-Type: application/json``. Otherwise 401, 421 or 415, with no effect.
"""

from __future__ import annotations

import asyncio
import hmac
import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

from sd_sim.linksim.core import FLOWS, LinkSim, Profile

MAX_HEADER_BYTES = 8192
MAX_BODY_BYTES = 4096
MAX_CONNECTIONS = 16
IO_TIMEOUT_S = 5.0


@dataclass
class Request:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes


class BadRequestError(Exception):
    pass


async def read_request(reader: asyncio.StreamReader) -> Request:
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError) as exc:
        raise BadRequestError from exc
    if len(head) > MAX_HEADER_BYTES:
        raise BadRequestError
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
        raise BadRequestError
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        name, sep, value = line.partition(":")
        if not sep:
            raise BadRequestError
        headers[name.strip().lower()] = value.strip()
    try:
        length = int(headers.get("content-length", "0"))
    except ValueError as exc:
        raise BadRequestError from exc
    if length < 0 or length > MAX_BODY_BYTES:
        raise BadRequestError
    body = await reader.readexactly(length) if length else b""
    return Request(parts[0], parts[1], headers, body)


class ControlApi:
    def __init__(self, sim: LinkSim, profiles: dict[str, Profile], token: str, port: int) -> None:
        if not token:
            raise ValueError("linksim behöver en token")
        self.sim = sim
        self.profiles = profiles
        self._token = token.encode()
        self.port = port
        self._server: asyncio.Server | None = None
        self._slots = asyncio.Semaphore(MAX_CONNECTIONS)

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._serve, host="127.0.0.1", port=self.port)

    def close(self) -> None:
        if self._server is not None:
            self._server.close()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._slots.locked():
            writer.transport.abort()  # too many open connections
            return
        async with self._slots:
            await self._serve_one(reader, writer)

    async def _serve_one(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            try:
                request = await asyncio.wait_for(read_request(reader), timeout=IO_TIMEOUT_S)
            except (BadRequestError, TimeoutError):
                status, payload = HTTPStatus.BAD_REQUEST, {"error": "bad request"}
            else:
                status, payload = self.handle(request)
            body = json.dumps(payload).encode()
            writer.write(
                f"HTTP/1.1 {status.value} {status.phrase}\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Connection: close\r\n\r\n".encode()
                + body
            )
            await asyncio.wait_for(writer.drain(), timeout=IO_TIMEOUT_S)
        except (ConnectionError, OSError, TimeoutError):
            pass
        finally:
            writer.close()

    def _authorized(self, headers: dict[str, str]) -> bool:
        scheme, _, given = headers.get("authorization", "").partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(given.encode(), self._token)

    def handle(self, req: Request) -> tuple[HTTPStatus, dict[str, Any]]:
        if not self._authorized(req.headers):
            return HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"}
        if req.headers.get("host") not in (f"127.0.0.1:{self.port}", f"localhost:{self.port}"):
            return HTTPStatus.MISDIRECTED_REQUEST, {"error": "wrong host"}
        if req.method in ("PUT", "POST"):
            ctype = req.headers.get("content-type", "").split(";")[0].strip().lower()
            if ctype != "application/json":
                return HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "json only"}
            try:
                body = json.loads(req.body or b"{}")
            except ValueError:
                return HTTPStatus.BAD_REQUEST, {"error": "invalid json"}
            if not isinstance(body, dict):
                return HTTPStatus.BAD_REQUEST, {"error": "expected an object"}
        else:
            body = {}
        return self._route(req.method, req.path, body)

    def _route(
        self, method: str, path: str, body: dict[str, Any]
    ) -> tuple[HTTPStatus, dict[str, Any]]:
        if method == "GET" and path == "/health":
            return HTTPStatus.OK, {"ok": True}
        if method == "GET" and path == "/state":
            return HTTPStatus.OK, self.sim.snapshot()
        if method == "PUT" and path == "/profile":
            name = body.get("name")
            seed = body.get("seed", 1)
            if name not in self.profiles or not isinstance(seed, int):
                return HTTPStatus.BAD_REQUEST, {"error": "unknown profile"}
            self.sim.set_profile(self.profiles[name], seed)
            return HTTPStatus.OK, self.sim.snapshot()
        if method == "POST" and (path == "/radio" or path.startswith("/flows/")):
            state = body.get("state")
            if state not in ("up", "down"):
                return HTTPStatus.BAD_REQUEST, {"error": "state must be up or down"}
            if path == "/radio":
                self.sim.set_radio(state == "up")
            else:
                flow = path.removeprefix("/flows/")
                if flow not in FLOWS:
                    return HTTPStatus.NOT_FOUND, {"error": "unknown flow"}
                self.sim.set_flow(flow, state == "up")
            return HTTPStatus.OK, self.sim.snapshot()
        return HTTPStatus.NOT_FOUND, {"error": "not found"}
