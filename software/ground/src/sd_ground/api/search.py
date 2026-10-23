"""The search API (``/search/v1``), only for ``sd-search`` with the search token.

It lives on the commander's loopback-only listener but has its own token and role. The
search service reads missions, the approved profile and perception, and writes candidates,
assessments and probability maps (later tickets add those routes). It has no route for
orders, approvals, STOP / RETURN HOME, chat as a person or a decision on a candidate, and
its token is refused on the commander API and on the panel listeners (403).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Request

from sd_ground.api.deps import ApiStatusError, ground, repo_of
from sd_ground.context import Ground
from sd_ground.services import search_profile as profile_service
from sd_schemas.search import SearchProfile

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
_UUID = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
IdPath = Annotated[str, Path(pattern=_UUID)]


@router.get("/health")
async def health(_: Search, g: G) -> dict[str, Any]:
    """The search service's view of the ground station: up, and whether the database is."""
    return {"ok": True, "role": "search", "version": g.version, "db_ok": g.db_ok}


@router.get("/missions/{mission_id}/search-profile", response_model=SearchProfile | None)
async def get_search_profile(mission_id: IdPath, _: Search, g: G) -> SearchProfile | None:
    """The approved version in force, or ``null``: a proposal from the AI is never
    returned here, so the matching cannot use one (E10)."""
    repo_of(g)
    if g.flow is None:
        raise ApiStatusError(503, "DB_DOWN", "Uppdragsflödet är inte igång just nu.")
    return await profile_service.approved(g.flow, mission_id)
