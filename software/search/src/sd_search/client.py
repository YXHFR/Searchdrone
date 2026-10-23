"""``GroundClient``: the search API (``/search/v1``) on the ground station's loopback
listener, with the search token in the ``Authorization`` header only.

Every path this client can reach starts with ``/search/v1/``: it has no way to give an
order, approve, press STOP or decide on a candidate (E7). ``SearchApi`` is what the
service needs; tests implement it in memory.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Protocol

import httpx
from pydantic import TypeAdapter

from sd_schemas.search import (
    CandidateBatch,
    CandidateWritten,
    SearchDetection,
    SearchHeartbeat,
    SearchProfile,
    SearchTrack,
)

__all__ = ["PREFIX", "ConflictError", "GroundClient", "GroundError", "SearchApi"]

log = logging.getLogger("sd_search.client")
PREFIX = "/search/v1"
_DETECTIONS = TypeAdapter(list[SearchDetection])
_TRACKS = TypeAdapter(list[SearchTrack])
_WRITTEN = TypeAdapter(list[CandidateWritten])


class GroundError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ConflictError(GroundError):
    """409: the ground's candidates moved on (a person decided, a stale rev)."""


class SearchApi(Protocol):
    async def heartbeat(self, beat: SearchHeartbeat) -> None: ...
    async def missions(self) -> list[str]: ...
    async def detections(
        self, mission_id: str, after_seq: int, limit: int
    ) -> list[SearchDetection]: ...
    async def crop(self, detection_id: str) -> bytes | None: ...
    async def tracks(self, mission_id: str) -> list[SearchTrack]: ...
    async def write(self, mission_id: str, batch: CandidateBatch) -> list[CandidateWritten]: ...
    async def profile(self, mission_id: str) -> SearchProfile | None: ...


class GroundClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout_s: float = 5.0,
        transport: httpx.AsyncBaseTransport | None = None,  # tests: the app in process
    ) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_s,
            trust_env=False,  # loopback only: never through a proxy
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            res = await self._http.request(method, f"{PREFIX}{path}", **kw)
        except httpx.HTTPError as exc:
            raise GroundError(f"{method} {PREFIX}{path}: {type(exc).__name__}") from exc
        if res.status_code == 409:
            raise ConflictError(f"{method} {PREFIX}{path}: 409", 409)
        return res

    @staticmethod
    def _ok(res: httpx.Response) -> httpx.Response:
        if res.status_code >= 400:
            code = ""
            with contextlib.suppress(ValueError):
                code = str(res.json().get("code", ""))
            raise GroundError(f"{res.request.url.path}: {res.status_code} {code}", res.status_code)
        return res

    async def heartbeat(self, beat: SearchHeartbeat) -> None:
        self._ok(await self._request("POST", "/heartbeat", json=beat.model_dump(mode="json")))

    async def missions(self) -> list[str]:
        res = self._ok(await self._request("GET", "/missions"))
        return [str(m["mission_id"]) for m in res.json()]

    async def detections(
        self, mission_id: str, after_seq: int, limit: int
    ) -> list[SearchDetection]:
        res = self._ok(
            await self._request(
                "GET",
                f"/missions/{mission_id}/detections",
                params={"after_seq": after_seq, "limit": limit},
            )
        )
        return _DETECTIONS.validate_python(res.json())

    async def crop(self, detection_id: str) -> bytes | None:
        res = await self._request("GET", f"/detections/{detection_id}/crop")
        if res.status_code == 404:
            return None
        return self._ok(res).content

    async def tracks(self, mission_id: str) -> list[SearchTrack]:
        res = self._ok(await self._request("GET", f"/missions/{mission_id}/candidates"))
        return _TRACKS.validate_python(res.json())

    async def write(self, mission_id: str, batch: CandidateBatch) -> list[CandidateWritten]:
        res = self._ok(
            await self._request(
                "POST", f"/missions/{mission_id}/candidates", json=batch.model_dump(mode="json")
            )
        )
        return _WRITTEN.validate_python(res.json())

    async def profile(self, mission_id: str) -> SearchProfile | None:
        """The approved profile in force, or ``None`` (no profile yet, or the ground has
        no profile route): matching then only asks "is it a person"."""
        res = await self._request("GET", f"/missions/{mission_id}/search-profile")
        if res.status_code in (204, 404):
            return None
        body = self._ok(res).json()
        return None if body is None else SearchProfile.model_validate(body)
