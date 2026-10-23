"""The cloud budget: what a cloud call may cost, asked of the ground station first.

The ground station keeps the books for every cloud call (the commander's chat and the
search service's image matching share one cap). A process asks for a reservation with
its estimate before the call; without a granted reservation no call is made. After the
call it commits the real use (or releases the reservation if nothing was sent). The
process is taken from the caller's token, never from the body.
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import Field

from sd_schemas.base import Id, SdModel

__all__ = [
    "CLOUD_BUDGET_CEILING_SEK_PER_DAY",
    "BudgetCommit",
    "BudgetCommitted",
    "BudgetDenial",
    "BudgetGrant",
    "BudgetRelease",
    "BudgetReserve",
    "BudgetState",
    "CloudBudgetStatus",
    "MissionBudgetCap",
]

# The outer cap in code: no configuration and no pilot can go above it.
CLOUD_BUDGET_CEILING_SEK_PER_DAY: Final = 1000.0

BudgetDenial = Literal["MISSION_CAP", "DAY_CAP", "UNKNOWN_PRICE", "NO_LEDGER"]
BudgetState = Literal["ok", "warning", "exhausted"]


class BudgetReserve(SdModel):
    """A cloud call about to be made: what it is for and how large it may get."""

    purpose: Literal["chat", "match"]
    mission_id: Id | None = None
    model_id: str = Field(min_length=1, max_length=120)
    est_prompt_tokens: int = Field(ge=0, le=1_000_000)
    est_completion_tokens: int = Field(ge=0, le=100_000)
    image_count: int = Field(default=0, ge=0, le=3)


class BudgetGrant(SdModel):
    """``granted`` false: no call. ``reservation_id`` is set exactly when granted."""

    granted: bool
    reservation_id: Id | None = None
    reason: BudgetDenial | None = None
    est_sek: float = Field(default=0.0, ge=0)
    remaining_sek: float = Field(ge=0)


class BudgetCommit(SdModel):
    """The real use of a granted call. ``cost_usd`` is the provider's own price, when it
    states one; otherwise the ground prices the tokens."""

    reservation_id: Id
    prompt_tokens: int = Field(ge=0, le=10_000_000)
    completion_tokens: int = Field(ge=0, le=10_000_000)
    cost_usd: float | None = Field(default=None, ge=0, le=1000)


class BudgetCommitted(SdModel):
    actual_sek: float = Field(ge=0)


class BudgetRelease(SdModel):
    """Nothing was sent (the call failed before reaching the provider)."""

    reservation_id: Id


class CloudBudgetStatus(SdModel):
    """What the panel shows: spent and caps for the mission (when one is given) and the
    day (Swedish date). ``exhausted``: no cloud calls; ``warning`` from 80 %."""

    mission_id: Id | None = None
    mission_sek: float = Field(ge=0)
    mission_cap_sek: float = Field(ge=0)
    day_sek: float = Field(ge=0)
    day_cap_sek: float = Field(ge=0)
    state: BudgetState


class MissionBudgetCap(SdModel):
    """The pilot raises (or lowers) the mission's cap, never above the outer cap."""

    cap_sek: float = Field(ge=0, le=CLOUD_BUDGET_CEILING_SEK_PER_DAY)
