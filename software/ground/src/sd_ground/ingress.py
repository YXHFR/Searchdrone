"""The ingress thread: the panel listeners run in their own thread and event loop.

STOP and RETURN HOME are handled completely here, using only thread-safe memory (the
session store) and the emergency path, so they work while the main event loop, the
database or the AI hang. Every other request is passed to the main loop over a bridge.

This module may not import the database, the services, the API routes or the shared
context (import-linter contract ``ingress-isolation``).
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any, Literal

from pydantic import ValidationError
from starlette.requests import cookie_parser

from sd_ground import emergency
from sd_ground.auth import (
    SESSION_COOKIE,
    Principal,
    SessionStore,
    principal_for,
    session_for,
)
from sd_ground.emergency import EmergencyRefusedError
from sd_ground.sources import EmergencyPath, EmergencyUnavailableError
from sd_ground.web import error_body
from sd_schemas.api import ActionRequest

__all__ = ["IngressApp", "IngressThread", "MainLoopBridge"]

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_ONE = re.compile(
    r"^/api/v1/vehicles/(?P<vid>[a-z0-9][a-z0-9-]{2,31})/actions/(?P<action>stop|return_home)$"
)
_ALL = re.compile(r"^/api/v1/actions/(?P<action>stop|return_home)_all$")
_MAX_BODY = 4096


class MainLoopBridge:
    """Runs an ASGI app in the main event loop on behalf of the ingress loop."""

    def __init__(self, app: ASGIApp, main_loop: asyncio.AbstractEventLoop) -> None:
        self.app = app
        self.main_loop = main_loop

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        here = asyncio.get_running_loop()
        if here is self.main_loop:
            await self.app(scope, receive, send)
            return

        async def bridged_receive() -> Message:
            return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(receive(), here))  # type: ignore[arg-type]

        async def bridged_send(message: Message) -> None:
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(send(message), here))  # type: ignore[arg-type]

        future = asyncio.run_coroutine_threadsafe(
            self.app(scope, bridged_receive, bridged_send),  # type: ignore[arg-type]
            self.main_loop,
        )
        try:
            await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            future.cancel()
            raise


async def _respond(send: Send, status: int, payload: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
                (b"cache-control", b"no-store"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})


class IngressApp:
    """STOP / RETURN HOME here; everything else to ``forward`` (the bridge).

    Any logged-in user (pilot or observer) may press STOP; RETURN HOME needs the pilot.
    """

    def __init__(
        self, *, sessions: SessionStore, emergency: EmergencyPath, forward: ASGIApp
    ) -> None:
        self.sessions = sessions
        self.emergency = emergency
        self.forward = forward

    def _principal(self, scope: Scope) -> Principal | None:
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])
        }
        # The same lenient parser as every other route: a broken cookie from another app on
        # localhost must never hide the pilot's session from STOP / RETURN HOME.
        cookie = cookie_parser(headers.get("cookie", "")).get(SESSION_COOKIE)
        rec = session_for(
            self.sessions, authorization=headers.get("authorization", ""), cookie=cookie
        )
        return None if rec is None else principal_for(rec, scope.get("sd_listener"))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.forward(scope, receive, send)
            return
        path = scope.get("path", "")
        one, every = _ONE.match(path), _ALL.match(path)
        if one is None and every is None:
            await self.forward(scope, receive, send)
            return
        body = b""
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return
            body += message.get("body", b"")
            if len(body) > _MAX_BODY:
                await _respond(
                    send, 413, error_body(413, "BODY_TOO_LARGE", "Anropet är för stort.")
                )
                return
            if not message.get("more_body"):
                break
        who = self._principal(scope)
        if who is None:
            await _respond(send, 401, error_body(401, "UNAUTHENTICATED", "Logga in först."))
            return
        try:
            req = ActionRequest.model_validate(json.loads(body or b"{}"))
        except (ValidationError, ValueError):  # ValueError: bad JSON or not UTF-8
            await _respond(send, 422, error_body(422, "VALIDATION", "Fel i indata."))
            return
        match = one or every
        if match is None:  # pragma: no cover - one of the two matched above
            return
        action: Literal["stop", "return_home"] = (
            "stop" if match["action"] == "stop" else "return_home"
        )
        try:
            if one is not None:
                result = emergency.dispatch(
                    self.emergency, action, one["vid"], str(req.client_action_id), who
                )
                payload = result.model_dump_json().encode()
            else:
                payload = (
                    emergency.dispatch_all(self.emergency, action, str(req.client_action_id), who)
                    .model_dump_json()
                    .encode()
                )
        except (EmergencyRefusedError, EmergencyUnavailableError) as exc:
            await _respond(send, exc.status, error_body(exc.status, exc.code, exc.detail_sv))
            return
        await _respond(send, 202, payload)


class IngressThread(threading.Thread):
    """Runs the panel servers (uvicorn) in their own thread with its own event loop."""

    def __init__(self, servers: list[Any], beat: Callable[[], None]) -> None:
        super().__init__(name="sd-ground-ingress", daemon=True)
        self.servers = servers
        self.beat = beat
        self.loop: asyncio.AbstractEventLoop | None = None
        self.started_event = threading.Event()

    def run(self) -> None:
        asyncio.run(self._main())

    async def _main(self) -> None:
        self.loop = asyncio.get_running_loop()

        async def heartbeat() -> None:
            while True:
                self.beat()
                await asyncio.sleep(1.0)  # its own sign of life, in real time

        beat_task = asyncio.create_task(heartbeat())
        serving = [asyncio.create_task(s.serve()) for s in self.servers]
        while not all(getattr(s, "started", False) for s in self.servers):
            if any(t.done() for t in serving):
                break
            await asyncio.sleep(0.05)
        self.started_event.set()
        await asyncio.gather(*serving, return_exceptions=True)
        beat_task.cancel()

    def stop(self) -> None:
        for server in self.servers:
            server.should_exit = True
