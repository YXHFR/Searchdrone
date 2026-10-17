"""The panel's only way to the live video: a WHEP proxy in front of MediaMTX.

``POST /api/v1/video/{vehicle_id}/main/whep`` (an SDP offer) is passed to MediaMTX's
WebRTC listener on loopback (``[video] mediamtx_whep``, ``/drone/{vehicle_id}/main/whep``)
and the SDP answer comes back. MediaMTX's ``Location`` for the session is rewritten to
``/api/v1/video/{vehicle_id}/main/whep/{session}``, where ``PATCH`` (trickle ICE) and
``DELETE`` (end the session) are passed on in the same way. So the browser talks only
to the panel's own origin, with the panel's session and CSRF checks (``web.PanelGuard``
lets ``application/sdp`` and ``application/trickle-ice-sdpfrag`` through on these paths
only), and MediaMTX's signalling never has to listen outside 127.0.0.1. Only the media
itself (UDP 8189) goes directly between the browser and MediaMTX.

The HLS fallback route answers 501 ``NOT_BUILT``: it is only built if WebRTC turns out
not to work on the phone (``PanelConfig.video.hls_url`` stays ``null`` until then).
"""

from __future__ import annotations

import re
from typing import Annotated
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Path, Request, Response

from sd_ground.api.deps import ApiStatusError
from sd_ground.api.panel import G, Observer, VehiclePath
from sd_ground.logging import get_logger
from sd_ground.services.video import stream_path

__all__ = ["SESSION_RE", "TIMEOUT_S", "panel_location", "router"]

log = get_logger("sd_ground.video_proxy")
router = APIRouter(prefix="/api/v1/video")

TIMEOUT_S = 5.0
SDP = "application/sdp"
SDP_FRAG = "application/trickle-ice-sdpfrag"
SESSION_RE = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
SessionPath = Annotated[str, Path(pattern=SESSION_RE)]
# Headers of MediaMTX's answer that the browser needs; nothing else is passed on.
_PASS = ("content-type", "etag", "accept-patch")
# Requests to MediaMTX carry only the body and these headers: never the panel's cookie.
_transport: httpx.AsyncBaseTransport | None = None


def set_transport(transport: httpx.AsyncBaseTransport | None) -> None:
    """Tests: answer the upstream calls with a fake MediaMTX."""
    global _transport
    _transport = transport


def panel_location(vehicle_id: str, upstream: str | None) -> str | None:
    """MediaMTX's session ``Location`` as the panel's path, or ``None`` if unexpected."""
    if not upstream:
        return None
    path = urlsplit(upstream).path
    prefix = f"/{stream_path(vehicle_id)}/whep/"
    if not path.startswith(prefix):
        return None
    session = path[len(prefix) :]
    if not re.match(SESSION_RE, session):
        return None
    return f"/api/v1/video/{vehicle_id}/main/whep/{session}"


def _ctype(request: Request) -> str:
    return request.headers.get("content-type", "").split(";", 1)[0].strip().lower()


async def _forward(
    g: G, method: str, upstream_path: str, *, body: bytes, headers: dict[str, str]
) -> httpx.Response:
    if not g.config.video.enabled:
        raise ApiStatusError(503, "VIDEO_DISABLED", "Videon är avstängd på markstationen.")
    base = g.config.video.mediamtx_whep.rstrip("/")
    try:
        async with httpx.AsyncClient(
            base_url=base, timeout=TIMEOUT_S, trust_env=False, transport=_transport
        ) as client:
            return await client.request(method, upstream_path, content=body, headers=headers)
    except httpx.HTTPError as exc:
        log.warning("video.proxy_unreachable", error=type(exc).__name__)
        raise ApiStatusError(
            503, "VIDEO_UNAVAILABLE", "Videoservern svarar inte. Kör just video-up."
        ) from exc


def _answer(res: httpx.Response, *, location: str | None = None) -> Response:
    if res.status_code == 404:
        raise ApiStatusError(404, "VIDEO_NOT_FOUND", "Ingen video från drönaren just nu.")
    if res.status_code >= 500:
        raise ApiStatusError(502, "VIDEO_UPSTREAM", "Videoservern gav ett fel.")
    if res.status_code >= 400:
        status = 400 if res.status_code == 400 else 502
        raise ApiStatusError(status, "VIDEO_REJECTED", "Videoservern tog inte emot anropet.")
    headers = {k: v for k, v in res.headers.items() if k.lower() in _PASS}
    if location is not None:
        headers["Location"] = location
    return Response(content=res.content, status_code=res.status_code, headers=headers)


@router.post("/{vehicle_id}/main/whep")
async def whep_offer(vehicle_id: VehiclePath, request: Request, g: G, _: Observer) -> Response:
    if _ctype(request) != SDP:
        raise ApiStatusError(415, "UNSUPPORTED_MEDIA_TYPE", "Väntade application/sdp.")
    res = await _forward(
        g,
        "POST",
        f"/{stream_path(vehicle_id)}/whep",
        body=await request.body(),
        headers={"Content-Type": SDP},
    )
    if res.status_code != 201:
        return _answer(res)
    location = panel_location(vehicle_id, res.headers.get("location"))
    if location is None:
        log.error("video.proxy_bad_location")
        raise ApiStatusError(502, "VIDEO_UPSTREAM", "Videoservern gav ett oväntat svar.")
    return _answer(res, location=location)


@router.patch("/{vehicle_id}/main/whep/{session}")
async def whep_patch(
    vehicle_id: VehiclePath, session: SessionPath, request: Request, g: G, _: Observer
) -> Response:
    if _ctype(request) != SDP_FRAG:
        raise ApiStatusError(415, "UNSUPPORTED_MEDIA_TYPE", f"Väntade {SDP_FRAG}.")
    headers = {"Content-Type": SDP_FRAG}
    if "if-match" in request.headers:
        headers["If-Match"] = request.headers["if-match"]
    res = await _forward(
        g,
        "PATCH",
        f"/{stream_path(vehicle_id)}/whep/{session}",
        body=await request.body(),
        headers=headers,
    )
    return _answer(res)


@router.delete("/{vehicle_id}/main/whep/{session}")
async def whep_delete(vehicle_id: VehiclePath, session: SessionPath, g: G, _: Observer) -> Response:
    res = await _forward(
        g, "DELETE", f"/{stream_path(vehicle_id)}/whep/{session}", body=b"", headers={}
    )
    return _answer(res)


@router.get("/{vehicle_id}/main/hls/{path:path}")
async def hls(vehicle_id: VehiclePath, path: str, _: Observer) -> Response:
    raise ApiStatusError(501, "NOT_BUILT", "HLS-reserven är inte byggd; videon går via WebRTC.")
