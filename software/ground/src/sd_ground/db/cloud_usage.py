"""SQL for the cloud budget's books (``cloud_usage``, append-only).

Every step of a reservation is its own row: ``reserved`` when it is granted, then exactly
one of ``committed`` (with the real use), ``released`` (nothing was sent) or ``expired``
(not booked in time, counted at its estimate). ``UNIQUE (reservation_id, status)`` keeps a
step from being written twice.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

__all__ = ["CloudUsageRepo", "UsageRow", "UsageStatus"]

UsageStatus = Literal["reserved", "committed", "released", "expired"]


@dataclass(frozen=True)
class UsageRow:
    reservation_id: str
    mission_id: str | None
    process: Literal["commander", "search"]
    purpose: str
    model_id: str
    est_sek: float
    status: UsageStatus
    created_at: dt.datetime
    actual_sek: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class CloudUsageRepo:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def add(self, row: UsageRow) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO cloud_usage (reservation_id, mission_id, process, purpose, "
                    "model_id, est_sek, actual_sek, prompt_tokens, completion_tokens, status, "
                    "created_at) VALUES (:reservation_id, :mission_id, :process, :purpose, "
                    ":model_id, :est_sek, :actual_sek, :prompt_tokens, :completion_tokens, "
                    ":status, :created_at)"
                ),
                {
                    "reservation_id": row.reservation_id,
                    "mission_id": row.mission_id,
                    "process": row.process,
                    "purpose": row.purpose,
                    "model_id": row.model_id,
                    "est_sek": row.est_sek,
                    "actual_sek": row.actual_sek,
                    "prompt_tokens": row.prompt_tokens,
                    "completion_tokens": row.completion_tokens,
                    "status": row.status,
                    "created_at": row.created_at,
                },
            )

    async def since(self, t: dt.datetime) -> Sequence[UsageRow]:
        """Every row of the reservations made from ``t`` on, plus every row of the open
        missions (their totals count whenever they started), oldest first."""
        async with self.engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text(
                            "SELECT u.reservation_id::text AS reservation_id, "
                            "u.mission_id::text AS mission_id, u.process, u.purpose, "
                            "u.model_id, u.est_sek, u.actual_sek, u.prompt_tokens, "
                            "u.completion_tokens, u.status, u.created_at FROM cloud_usage u "
                            "WHERE u.reservation_id IN (SELECT reservation_id FROM cloud_usage "
                            "WHERE status = 'reserved' AND (created_at >= :t OR mission_id IN "
                            "(SELECT id FROM missions WHERE status <> 'CLOSED'))) "
                            "ORDER BY u.created_at, u.id"
                        ),
                        {"t": t},
                    )
                )
                .mappings()
                .all()
            )
        return [UsageRow(**dict(r)) for r in rows]

    async def mission_caps(self) -> dict[str, float]:
        """The pilot's latest cap per open mission (``cloud_budget.mission_cap`` events)."""
        async with self.engine.connect() as conn:
            rows = (
                (
                    await conn.execute(
                        text(
                            "SELECT DISTINCT ON (e.mission_id) e.mission_id::text AS mission_id, "
                            "(e.payload->>'cap_sek')::float AS cap_sek FROM events e "
                            "JOIN missions m ON m.id = e.mission_id "
                            "WHERE e.type = 'cloud_budget.mission_cap' AND m.status <> 'CLOSED' "
                            "ORDER BY e.mission_id, e.t DESC, e.id DESC"
                        )
                    )
                )
                .mappings()
                .all()
            )
        return {str(r["mission_id"]): float(r["cap_sek"]) for r in rows}
