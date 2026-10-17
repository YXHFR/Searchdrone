"""Video for the panel through the ground station: WHEP to MediaMTX on loopback.

``POST /api/v1/video/{vehicle_id}/main/whep`` (SDP offer) is passed on to
``<mediamtx_whep>/drone/{vehicle_id}/main/whep`` and the answer comes back with its
``Location`` rewritten to ``/api/v1/video/{vehicle_id}/main/whep/{session}``; ``PATCH``
(trickle ICE) and ``DELETE`` (close) on that address go to the same session in MediaMTX.
Every route needs a session (observer or pilot; 401 without), the panel guard's
``X-SD-Client`` and content-type rules apply, and only the headers WHEP needs are passed
on in either direction (never the cookie). MediaMTX's own ports stay on 127.0.0.1, so
the panel can reach the video only here; the media itself (UDP 8189) goes directly.

The HLS fallback on the server is only built if WebRTC turns out not to work on the
phone (it answers 501 until then); the panel's HLS logic is in place.
"""

from __future__ import annotations

import re
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Path, Request, Response

from sd_ground.api.deps import ApiStatusError, ground, not_built, require
from sd_ground.auth import Principal
from sd_ground.context import Ground
from sd_ground.logging import get_logger
from sd_ground.services import video as video_service

__all__ = ["rewrite_location", "router"]

log = get_logger("sd_ground.video_proxy")
router = APIRouter(prefix="/api/v1")

VehiclePath = Annotated[str, Path(pattern=r"^[a-z0-9][a-z0-9-]{2,31}$")]
SessionPath = Annotated[str, Path(pattern=r"^[A-Za-z0-9-]{1,64}$")]
Observer = Annotated[Principal, Depends(require("observer"))]
G = Annotated[Ground, Depends(ground)]

MAX_SDP = 64 * 1024
_UPSTREAM_REQUEST = ("content-type", "if-match")
_UPSTREAM_RESPONSE = ("content-type", "etag", "accept-patch")
_SESSION = re.compile(r"/whep/([A-Za-z0-9-]{1,64})/?$")


def rewrite_location(location: str | None, vehicle_id: str) -> str | None:
    """MediaMTX's session address as the panel's: ``/api/v1/video/{vid}/main/whep/{id}``."""
    if not location:
        return None
    m = _SESSION.search(location.split("?", 1)[0])
    return None if m is None else f"/api/v1/video/{vehicle_id}/main/whep/{m.group(1)}"


def _upstream(g: Ground, vehicle_id: str, session: str | None = None) -> str:
    base = f"{g.config.video.mediamtx_whep}/{video_service.path_name(vehicle_id)}/whep"
    return base if session is None else f"{base}/{session}"


async def _forward(
    g: Ground, request: Request, method: str, vehicle_id: str, session: str | None
) -> Response:
    if not g.config.video.enabled:
        raise ApiStatusError(503, "VIDEO_DISABLED", "Video är avstängd i den här instansen.")
    body = await request.body()
    if len(body) > MAX_SDP:
        raise ApiStatusError(413, "PAYLOAD_TOO_LARGE", "För stort WHEP-meddelande.")
    headers = {k: v for k in _UPSTREAM_REQUEST if (v := request.headers.get(k))}
    if method == "DELETE":
        headers.pop("content-type", None)
        body = b""
    try:
        async with video_service.client() as http:
            res = await http.request(
                method, _upstream(g, vehicle_id, session), content=body, headers=headers
            )
    except httpx.HTTPError as exc:
        log.warning("video.proxy_failed", method=method, error=type(exc).__name__)
        raise ApiStatusError(
            503, "VIDEO_UNAVAILABLE", "Videoservern svarar inte. Kör just video-up."
        ) from None
    if res.status_code == 404:
        raise ApiStatusError(
            404, "VIDEO_NOT_FOUND", "Ingen video från drönaren just nu (eller okänd session)."
        )
    if res.status_code >= 400:
        log.warning("video.upstream_error", method=method, status=res.status_code)
        raise ApiStatusError(502, "VIDEO_UPSTREAM", "Videoservern avvisade begäran.")
    out = Response(content=res.content, status_code=res.status_code)
    for key in _UPSTREAM_RESPONSE:
        if value := res.headers.get(key):
            out.headers[key] = value
    if method == "POST":
        location = rewrite_location(res.headers.get("location"), vehicle_id)
        if location is None:
            raise ApiStatusError(502, "VIDEO_UPSTREAM", "Videoservern gav ingen session.")
        out.headers["Location"] = location
    return out


@router.post("/video/{vehicle_id}/main/whep", status_code=201)
async def whep_offer(vehicle_id: VehiclePath, request: Request, g: G, _: Observer) -> Response:
    return await _forward(g, request, "POST", vehicle_id, None)


@router.patch("/video/{vehicle_id}/main/whep/{session}")
async def whep_patch(
    vehicle_id: VehiclePath, session: SessionPath, request: Request, g: G, _: Observer
) -> Response:
    return await _forward(g, request, "PATCH", vehicle_id, session)


@router.delete("/video/{vehicle_id}/main/whep/{session}")
async def whep_delete(
    vehicle_id: VehiclePath, session: SessionPath, request: Request, g: G, _: Observer
) -> Response:
    return await _forward(g, request, "DELETE", vehicle_id, session)


@router.get("/video/{vehicle_id}/main/hls/{path:path}")
async def hls(vehicle_id: VehiclePath, path: str, _: Observer) -> Response:
    raise not_built("B1-09: HLS-reservvägen byggs bara om WebRTC inte fungerar i mobilen")
