"""The cloud budget's routes.

``/internal/v1/cloud-budget/{reserve,commit,release}`` are on the commander's loopback
listener for the two processes that call cloud models, each with its own token: the
process booked is the one whose token was shown, never one named in the body, and a
process can only book or release its own reservations. ``GET /api/v1/cloud-budget`` shows
the panel the spend and the caps; ``PUT /api/v1/missions/{id}/cloud-budget`` lets the
pilot change a mission's cap, never above the outer cap.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from sd_ground.api.deps import ApiStatusError, IdPath, ground, repo_of, require
from sd_ground.auth import Principal
from sd_ground.context import Ground
from sd_ground.services.cloud_budget import CloudBudget, Process
from sd_schemas.cloud_budget import (
    BudgetCommit,
    BudgetCommitted,
    BudgetGrant,
    BudgetRelease,
    BudgetReserve,
    CloudBudgetStatus,
    MissionBudgetCap,
)

__all__ = ["internal_router", "panel_router", "require_cloud_caller"]

internal_router = APIRouter(prefix="/internal/v1/cloud-budget")
panel_router = APIRouter(prefix="/api/v1")


def require_cloud_caller(request: Request) -> Process:
    g = ground(request)
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    token = token.strip()
    if scheme.lower() == "bearer" and token:
        if g.brain_token.matches(token):
            return "commander"
        if g.search_token.matches(token):
            return "search"
    raise ApiStatusError(401, "UNAUTHENTICATED", "Nyckel saknas eller är fel.")


Caller = Annotated[Process, Depends(require_cloud_caller)]
G = Annotated[Ground, Depends(ground)]
Pilot = Annotated[Principal, Depends(require("pilot"))]
Observer = Annotated[Principal, Depends(require("observer"))]
MissionQuery = Annotated[
    str | None, Query(pattern=r"^[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")
]


def budget_of(g: Ground) -> CloudBudget:
    if g.cloud_budget is None:
        raise ApiStatusError(503, "NO_LEDGER", "Molnbudgeten är inte igång.")
    return g.cloud_budget


@internal_router.post("/reserve", response_model=BudgetGrant)
async def reserve(body: BudgetReserve, process: Caller, g: G) -> BudgetGrant:
    """A reservation for one call, or a refusal (``granted`` false, with the reason).
    Without the books (no database) every call is refused."""
    if g.cloud_budget is None:
        return BudgetGrant(granted=False, reason="NO_LEDGER", remaining_sek=0.0)
    return await g.cloud_budget.reserve(process, body)


@internal_router.post("/commit", response_model=BudgetCommitted)
async def commit(body: BudgetCommit, process: Caller, g: G) -> BudgetCommitted:
    """The real use of a granted call. 404: not the caller's reservation; 409: already
    booked, released or expired."""
    result = await budget_of(g).commit(process, body)
    if result == "unknown":
        raise ApiStatusError(404, "RESERVATION_NOT_FOUND", "Reservationen finns inte.")
    if result == "closed":
        raise ApiStatusError(409, "RESERVATION_CLOSED", "Reservationen är redan avslutad.")
    return result


@internal_router.post("/release", status_code=204)
async def release(body: BudgetRelease, process: Caller, g: G) -> None:
    """Nothing was sent: the reservation no longer counts."""
    result = await budget_of(g).release(process, str(body.reservation_id))
    if result == "unknown":
        raise ApiStatusError(404, "RESERVATION_NOT_FOUND", "Reservationen finns inte.")
    if result == "closed":
        raise ApiStatusError(409, "RESERVATION_CLOSED", "Reservationen är redan avslutad.")


@panel_router.get("/cloud-budget", response_model=CloudBudgetStatus)
async def get_budget(_: Observer, g: G, mission_id: MissionQuery = None) -> CloudBudgetStatus:
    """Spent and caps today and for the mission (the latest one with a call, unless one
    is named)."""
    budget = budget_of(g)
    await budget.refresh()
    return budget.status(mission_id)


@panel_router.put("/missions/{mission_id}/cloud-budget", response_model=CloudBudgetStatus)
async def put_mission_cap(
    mission_id: IdPath, body: MissionBudgetCap, who: Pilot, g: G
) -> CloudBudgetStatus:
    """The pilot sets the mission's cap (at most the outer cap). Logged."""
    mission = await repo_of(g).mission(mission_id)
    if mission is None:
        raise ApiStatusError(404, "MISSION_NOT_FOUND", "Uppdraget finns inte.")
    if mission.status == "CLOSED":
        raise ApiStatusError(409, "MISSION_CLOSED", "Uppdraget är stängt.")
    budget = budget_of(g)
    await budget.refresh()
    return await budget.set_mission_cap(mission_id, body.cap_sek, who.issuer_ref)
