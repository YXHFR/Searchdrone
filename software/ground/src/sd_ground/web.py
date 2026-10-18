"""Browser protection for every listener, as plain ASGI middleware.

``PanelGuard`` wraps the panel listeners (:8080, and :8443 in LAN mode): ``Host``
allow-list (421, against DNS rebinding), ``X-SD-Client: panel`` plus JSON on every
changing request (403 ``CSRF``), same-origin ``Origin`` (403, WebSocket 4403), no CORS,
a body limit, the security headers on every response and no commander token at all.

``LoopbackGuard`` wraps the commander API (:7700): loopback ``Host`` only, and any
``Origin`` header (only browsers send one) is refused.

These run in front of the routes, so the emergency routes in the ingress thread get the
same protection as everything else.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from typing import Any

from sd_ground.auth import bearer_of

__all__ = [
    "COMMANDER_BODY_LIMIT",
    "PANEL_BODY_LIMIT",
    "SECURITY_HEADERS",
    "LoopbackGuard",
    "PanelGuard",
    "error_body",
]

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

PANEL_BODY_LIMIT = 256 * 1024
COMMANDER_BODY_LIMIT = 2 * 1024 * 1024
# Same values as software/common/schemas/security_headers.json (a test compares them).
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; worker-src 'self' blob:; "
        "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; "
        "style-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
        "form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "geolocation=(), camera=(), microphone=(), payment=(), usb=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}
_CHANGING = {"POST", "PUT", "PATCH", "DELETE"}
_WHEP_TYPES = {"application/sdp", "application/trickle-ice-sdpfrag"}

_TITLES = {
    401: "Unauthorized",
    403: "Forbidden",
    413: "Payload Too Large",
    421: "Misdirected Request",
    422: "Unprocessable Entity",
    503: "Service Unavailable",
}


def error_body(status: int, code: str, detail_sv: str) -> bytes:
    return json.dumps(
        {
            "type": "about:blank",
            "title": _TITLES.get(status, "Error"),
            "status": status,
            "code": code,
            "detail_sv": detail_sv,
            "errors": [],
        }
    ).encode()


def _headers(scope: Scope) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_name, raw_value in scope.get("headers", []):
        name = raw_name.decode("latin-1").lower()
        out.setdefault(name, raw_value.decode("latin-1"))
    return out


def _bearer(headers: dict[str, str]) -> str | None:
    return bearer_of(headers.get("authorization", ""))


async def _reject_http(send: Send, status: int, code: str, detail_sv: str) -> None:
    body = error_body(status, code, detail_sv)
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
        (b"cache-control", b"no-store"),
    ]
    headers += [(k.lower().encode(), v.encode()) for k, v in SECURITY_HEADERS.items()]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _reject_ws(receive: Receive, send: Send, code: int) -> None:
    """Accept, then close with the code, so the client sees the code and not HTTP 403."""
    message = await receive()
    if message["type"] != "websocket.connect":
        return
    await send({"type": "websocket.accept"})
    await send({"type": "websocket.close", "code": code})


class _BodyLimit:
    """Counts streamed bytes; past the limit the body ends with 413 instead."""

    def __init__(self, receive: Receive, limit: int) -> None:
        self._receive, self._limit, self.seen, self.exceeded = receive, limit, 0, False

    async def __call__(self) -> Message:
        message = await self._receive()
        if message["type"] == "http.request":
            self.seen += len(message.get("body", b""))
            if self.seen > self._limit:
                self.exceeded = True
                return {"type": "http.disconnect"}
        return message


async def _limited(app: ASGIApp, scope: Scope, receive: Receive, send: Send, limit: int) -> None:
    headers = _headers(scope)
    length = headers.get("content-length", "")
    if length.isdigit() and int(length) > limit:
        await _reject_http(send, 413, "BODY_TOO_LARGE", "Anropet är för stort.")
        return
    counter = _BodyLimit(receive, limit)
    started = False

    async def guarded_send(message: Message) -> None:
        nonlocal started
        if counter.exceeded and not started:
            return
        if message["type"] == "http.response.start":
            started = True
        await send(message)

    await app(scope, counter, guarded_send)
    if counter.exceeded and not started:
        await _reject_http(send, 413, "BODY_TOO_LARGE", "Anropet är för stort.")


class PanelGuard:
    def __init__(
        self,
        app: ASGIApp,
        *,
        allowed_hosts: Callable[[], Iterable[str]],
        is_commander_token: Callable[[str], bool],
        profile: str,
        is_search_token: Callable[[str], bool] = lambda _: False,
    ) -> None:
        self.app = app
        self._allowed_hosts = allowed_hosts
        self._is_commander = is_commander_token
        self._is_search = is_search_token
        self._profile = profile

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = _headers(scope)
        host = headers.get("host", "").lower()
        is_ws = scope["type"] == "websocket"
        if host not in {h.lower() for h in self._allowed_hosts()}:
            if is_ws:
                await _reject_ws(receive, send, 4403)
            else:
                await _reject_http(send, 421, "HOST_NOT_ALLOWED", "Okänd värd.")
            return
        origin = headers.get("origin")
        scheme = "https" if scope.get("scheme") in ("https", "wss") else "http"
        if origin is not None and origin.lower() != f"{scheme}://{host}":
            if is_ws:
                await _reject_ws(receive, send, 4403)
            else:
                await _reject_http(
                    send, 403, "ORIGIN_NOT_ALLOWED", "Anropet kommer från en annan sida."
                )
            return
        token = _bearer(headers)
        if token is not None and self._is_commander(token):
            if is_ws:
                await _reject_ws(receive, send, 4403)
            else:
                await _reject_http(
                    send, 403, "COMMANDER_FORBIDDEN", "AI:ns nyckel gäller inte i panelen."
                )
            return
        if token is not None and self._is_search(token):
            if is_ws:
                await _reject_ws(receive, send, 4403)
            else:
                await _reject_http(
                    send, 403, "SEARCH_FORBIDDEN", "Söktjänstens nyckel gäller inte i panelen."
                )
            return
        if is_ws:
            if token is not None and origin is not None:
                await _reject_ws(receive, send, 4403)
                return
            await self.app(scope, receive, send)
            return
        if "x-sd-dev-proxy" in headers and self._profile != "sim":
            await _reject_http(send, 403, "DEV_PROXY_FORBIDDEN", "Utvecklingsproxyn är avstängd.")
            return
        method = scope.get("method", "GET").upper()
        path: str = scope.get("path", "")
        if method in _CHANGING:
            ctype = headers.get("content-type", "").split(";", 1)[0].strip().lower()
            whep = path.startswith("/api/v1/video/") and "/whep" in path
            ok_type = ctype == "application/json" or (whep and ctype in _WHEP_TYPES)
            if headers.get("x-sd-client") != "panel" or not ok_type:
                await _reject_http(send, 403, "CSRF", "Anropet saknar panelens huvud.")
                return

        async def add_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                kept = [
                    (k, v)
                    for k, v in message.get("headers", [])
                    if not k.lower().startswith(b"access-control-")
                    and k.lower().decode("latin-1") not in {h.lower() for h in SECURITY_HEADERS}
                ]
                kept += [(k.lower().encode(), v.encode()) for k, v in SECURITY_HEADERS.items()]
                if path.startswith("/api/") and not any(
                    k.lower() == b"cache-control" for k, _ in kept
                ):
                    kept.append((b"cache-control", b"no-store"))
                message = {**message, "headers": kept}
            await send(message)

        await _limited(self.app, scope, receive, add_headers, PANEL_BODY_LIMIT)


class LoopbackGuard:
    """For :7700 (commander) and :7600 (SD-link): loopback ``Host`` and no ``Origin``."""

    def __init__(self, app: ASGIApp, *, port: int, body_limit: int = COMMANDER_BODY_LIMIT) -> None:
        self.app = app
        self._hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        self._limit = body_limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = _headers(scope)
        is_ws = scope["type"] == "websocket"
        if headers.get("host", "").lower() not in self._hosts:
            if is_ws:
                await _reject_ws(receive, send, 4403)
            else:
                await _reject_http(send, 421, "HOST_NOT_ALLOWED", "Okänd värd.")
            return
        if "origin" in headers:
            if is_ws:
                await _reject_ws(receive, send, 4403)
            else:
                await _reject_http(
                    send, 403, "ORIGIN_NOT_ALLOWED", "Webbläsare får inte anropa här."
                )
            return
        if is_ws:
            await self.app(scope, receive, send)
        else:
            await _limited(self.app, scope, receive, send, self._limit)
