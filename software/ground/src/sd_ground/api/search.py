"""The search API (``/search/v1``), only for ``sd-search`` with the search token.

It lives on the commander's loopback-only listener but has its own token and role. The
search service reads missions, the approved profile and perception, and writes candidates,
assessments and probability maps (later tickets add those routes). It has no route for
orders, approvals, STOP / RETURN HOME, chat as a person or a decision on a candidate, and
its token is refused on the commander API and on the panel listeners (403).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from sd_ground.api.deps import ApiStatusError, ground
from sd_ground.context import Ground

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


@router.get("/health")
async def health(_: Search, g: G) -> dict[str, Any]:
    """The search service's view of the ground station: up, and whether the database is."""
    return {"ok": True, "role": "search", "version": g.version, "db_ok": g.db_ok}
