"""The search API (``/search/v1``), only for ``sd-search`` with the search token.

It lives on the commander's loopback-only listener but has its own token and role. The
search service reads the open missions, their approved search profile and their target
detections and crops, writes candidates with the code's assessment, reads its candidates
back after a restart and sends a heartbeat. It has no route for orders, approvals, STOP /
RETURN HOME, chat as a person or a decision on a candidate (it can never set ``confirmed``
or ``rejected``), and its token is refused on the commander API and on the panel listeners
(403).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse

from sd_ground.api.deps import ApiStatusError, IdPath, ground, repo_of
from sd_ground.context import Ground
from sd_ground.services import candidates as candidates_service
from sd_ground.services import files as files_service
from sd_ground.services import search_profile as profile_service
from sd_ground.services.missions import MissionFlow
from sd_schemas.search import (
    CandidateBatch,
    CandidateWritten,
    SearchDetection,
    SearchHeartbeat,
    SearchProfile,
    SearchTrack,
)

__all__ = ["require_search", "router"]

router = APIRouter(prefix="/search/v1")


def require_search(request: Request) -> None:
    g = ground(request)
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not g.search_token.matches(token):
        if scheme.lower() == "bearer" and g.brain_token.matches(token):
            raise ApiStatusError(403, "COMMANDER_FORBIDDEN", "AI:ns nyckel gäller inte här.")
        raise ApiStatusError(401, "UNAUTHENTICATED", "Söktjänstens nyckel saknas eller är fel.")


Search = Annotated[None, Depends(require_search)]
G = Annotated[Ground, Depends(ground)]


def _flow(g: Ground) -> MissionFlow:
    repo_of(g)
    if g.flow is None:
        raise ApiStatusError(503, "DB_DOWN", "Uppdragsflödet är inte igång just nu.")
    return g.flow


@router.get("/health")
async def health(_: Search, g: G) -> dict[str, Any]:
    """The search service's view of the ground station: up, and whether the database is."""
    return {"ok": True, "role": "search", "version": g.version, "db_ok": g.db_ok}


@router.post("/heartbeat", status_code=204)
async def heartbeat(body: SearchHeartbeat, _: Search, g: G) -> None:
    """The search service is alive (``parts.search`` in the health)."""
    g.search_seen_mono = g.clock.monotonic()
    g.search_cloud = body.cloud


@router.get("/missions")
async def missions(_: Search, g: G) -> list[dict[str, Any]]:
    """The open missions (the ones that may get detections)."""
    return [
        {"mission_id": str(m.id), "vehicle_id": m.vehicle_id, "status": m.status.value}
        for m in await repo_of(g).open_missions()
    ]


@router.get("/missions/{mission_id}/detections", response_model=list[SearchDetection])
async def detections(
    mission_id: IdPath,
    _: Search,
    g: G,
    after_seq: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=candidates_service.FEED_MAX)] = 200,
) -> list[SearchDetection]:
    """The mission's target detections after ``after_seq``, oldest first."""
    return await candidates_service.feed(_flow(g), mission_id, after_seq, limit)


@router.get("/detections/{detection_id}/crop")
async def crop(detection_id: IdPath, _: Search, g: G) -> FileResponse:
    """A target detection's crop (JPEG), only from inside its mission's folder."""
    flow = _flow(g)
    found = await candidates_service.repo(flow).crop_path(detection_id)
    path = None
    if found is not None:
        await flow.open_mission_of(found[0])  # search reads open missions only (409)
        path = files_service.resolve_file(flow.mission_dir(found[0]), found[1])
    if path is None or not path.is_file():
        raise ApiStatusError(404, "CROP_NOT_FOUND", "Bilden finns inte.")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/missions/{mission_id}/search-profile", response_model=SearchProfile | None)
async def get_search_profile(mission_id: IdPath, _: Search, g: G) -> SearchProfile | None:
    """The approved version in force, or ``null``: a proposal from the AI is never
    returned here, so the matching cannot use one (E10)."""
    flow = _flow(g)
    await flow.open_mission_of(mission_id)  # search reads open missions only (409)
    return await profile_service.approved(flow, mission_id)


@router.get("/missions/{mission_id}/candidates", response_model=list[SearchTrack])
async def candidates(mission_id: IdPath, _: Search, g: G) -> list[SearchTrack]:
    """Every candidate in force with the detections it observed (to resume after a
    restart)."""
    return await candidates_service.tracks(_flow(g), mission_id)


@router.post("/missions/{mission_id}/candidates", response_model=list[CandidateWritten])
async def write_candidates(
    mission_id: IdPath, body: CandidateBatch, _: Search, g: G
) -> list[CandidateWritten]:
    """New candidates and new revisions, all or none. 409 for a stale ``base_rev``, an
    observation already written, or a candidate a person or the pilot holds."""
    return await candidates_service.write(_flow(g), mission_id, body)
