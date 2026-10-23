"""Errors, the caller's identity and role checks shared by the panel routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import HTTPConnection

from sd_ground.auth import (
    SESSION_COOKIE,
    SESSION_HEADROOM,
    Principal,
    SessionRecord,
    principal_for,
    session_for,
)
from sd_ground.context import Ground
from sd_ground.db.repo import Repo

__all__ = [
    "COOKIE",
    "ApiStatusError",
    "ground",
    "install_error_handlers",
    "not_built",
    "principal_from",
    "repo_of",
    "require",
    "require_headroom",
    "require_interactive",
]

COOKIE = SESSION_COOKIE
_TITLES = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    413: "Payload Too Large",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    501: "Not Implemented",
    503: "Service Unavailable",
}


class ApiStatusError(Exception):
    def __init__(
        self, status: int, code: str, detail_sv: str, errors: list[dict[str, Any]] | None = None
    ) -> None:
        super().__init__(code)
        self.status, self.code, self.detail_sv, self.errors = status, code, detail_sv, errors or []


def problem(
    status: int, code: str, detail_sv: str, errors: list[dict[str, Any]] | None = None
) -> JSONResponse:
    return JSONResponse(
        {
            "type": "about:blank",
            "title": _TITLES.get(status, "Error"),
            "status": status,
            "code": code,
            "detail_sv": detail_sv,
            "errors": errors or [],
        },
        status_code=status,
    )


def install_error_handlers(app: FastAPI) -> None:
    async def api_err(_: Request, exc: Exception) -> JSONResponse:
        if not isinstance(exc, ApiStatusError):
            raise exc
        return problem(exc.status, exc.code, exc.detail_sv, exc.errors)

    async def validation(_: Request, exc: Exception) -> JSONResponse:
        if not isinstance(exc, RequestValidationError):
            raise exc
        errors = [
            {
                "loc": [str(p) if not isinstance(p, int) else p for p in err.get("loc", ())],
                "code": str(err.get("type", "invalid")),
                "detail_sv": str(err.get("msg", "Ogiltigt värde.")),
            }
            for err in exc.errors()
        ]
        return problem(422, "VALIDATION", "Fel i indata.", errors)

    async def http_exc(_: Request, exc: Exception) -> JSONResponse:
        if not isinstance(exc, StarletteHTTPException):
            raise exc
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        return problem(
            exc.status_code, code, "Finns inte." if exc.status_code == 404 else str(exc.detail)
        )

    app.add_exception_handler(ApiStatusError, api_err)
    app.add_exception_handler(RequestValidationError, validation)
    app.add_exception_handler(StarletteHTTPException, http_exc)


def ground(conn: HTTPConnection) -> Ground:
    g = conn.app.state.ground
    if not isinstance(g, Ground):
        raise TypeError("app.state.ground is not set")
    return g


# Path parameters every router takes: a UUID and a vehicle id.
_UUID = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
IdPath = Annotated[str, Path(pattern=_UUID)]
VehiclePath = Annotated[str, Path(pattern=r"^[a-z0-9][a-z0-9-]{2,31}$")]


def repo_of(g: Ground) -> Repo:
    if g.repo is None or not g.db_ok:
        raise ApiStatusError(503, "DB_DOWN", "Databasen svarar inte just nu.")
    return g.repo


def not_built(ticket: str) -> ApiStatusError:
    return ApiStatusError(501, "NOT_BUILT", f"Den här delen är inte byggd än ({ticket}).")


def session_of(conn: HTTPConnection) -> SessionRecord | None:
    return session_for(
        ground(conn).sessions,
        authorization=conn.headers.get("authorization", ""),
        cookie=conn.cookies.get(COOKIE),
    )


def principal_from(conn: HTTPConnection) -> Principal | None:
    rec = session_of(conn)
    return None if rec is None else principal_for(rec, conn.scope.get("sd_listener"))


def require(role: Literal["pilot", "observer"]) -> Callable[[Request], Principal]:
    def dependency(request: Request) -> Principal:
        who = principal_from(request)
        if who is None:
            raise ApiStatusError(401, "UNAUTHENTICATED", "Logga in först.")
        if role == "pilot" and who.role != "pilot":
            raise ApiStatusError(403, "FORBIDDEN_ROLE", "Bara piloten får göra det här.")
        return who

    return dependency


def require_interactive(request: Request) -> Principal:
    """Approvals and "Fritt runt drönaren" need a person in the panel, not a script."""
    who = require("pilot")(request)
    if not who.interactive and not ground(request).test_rig:
        raise ApiStatusError(
            403,
            "INTERACTIVE_SESSION_REQUIRED",
            "Det här måste göras av en människa i panelen.",
        )
    return who


def require_headroom(request: Request) -> None:
    """Takeoff and charter approval need at least 2 h left on the pilot's session."""
    rec = session_of(request)
    if rec is None:
        raise ApiStatusError(401, "UNAUTHENTICATED", "Logga in först.")
    if ground(request).sessions.headroom(rec) < SESSION_HEADROOM:
        raise ApiStatusError(409, "SESSION_EXPIRING", "Logga in igen innan start.")
