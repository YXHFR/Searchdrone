"""SQL for the probability map: the stills it takes in and its versions.

Part of the database layer. ``probability_maps`` is append-only: every version is a new
row, numbered from 1 per mission (``UNIQUE (mission_id, version)``), written under a
per-mission advisory lock so the check against the newest version and the insert are one
step. A version after a frozen one is refused here, next to the write, while a find is
still confirmed; once the pilot has reopened it the map takes versions again.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from sqlalchemy import RowMapping, text
from sqlalchemy.ext.asyncio import AsyncEngine

__all__ = ["MapWriteError", "ProbabilityMapRepo"]

_LOCK = "SELECT pg_advisory_xact_lock(hashtext('probability_maps:' || :m))"
_HEAD = (
    'SELECT version, reason, grid, "row", frozen, event_seq, created_at '
    "FROM probability_maps WHERE mission_id = :m ORDER BY version DESC LIMIT 1"
)
_CONFIRMED = (
    "SELECT EXISTS (SELECT 1 FROM (SELECT DISTINCT ON (candidate_id) state FROM candidates "
    "WHERE mission_id = :m ORDER BY candidate_id, rev DESC) newest WHERE state = 'confirmed')"
)


class MapWriteError(Exception):
    """A version the ground does not take (the API answers 409)."""

    def __init__(self, code: str, detail_sv: str) -> None:
        super().__init__(code)
        self.code, self.detail_sv = code, detail_sv


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


class ProbabilityMapRepo:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def _all(self, sql: str, **params: Any) -> Sequence[RowMapping]:
        async with self.engine.connect() as conn:
            return (await conn.execute(text(sql), params)).mappings().all()

    async def stills(self, mission_id: str, after_seq: int, limit: int) -> list[dict[str, Any]]:
        """The mission's stills after ``after_seq``, oldest first, with their footprints."""
        rows = await self._all(
            "SELECT still_id::text AS still_id, seq, command_id::text AS command_id, "
            "captured_utc, ST_AsGeoJSON(footprint) AS footprint, px_per_m, "
            "COALESCE((detector->>'ok')::boolean, false) AS detector_ok "
            "FROM perception_stills WHERE mission_id = :m AND seq > :after "
            "ORDER BY seq LIMIT :n",
            m=mission_id,
            after=after_seq,
            n=limit,
        )
        out: list[dict[str, Any]] = []
        for r in rows:
            ring = _json(r["footprint"])["coordinates"][0]
            d = dict(r)
            d["footprint"] = [{"lat": float(p[1]), "lon": float(p[0])} for p in ring[:4]]
            out.append(d)
        return out

    async def head(self, mission_id: str) -> dict[str, Any] | None:
        """The newest version without its arrays."""
        rows = await self._all(_HEAD, m=mission_id)
        if not rows:
            return None
        d = dict(rows[0])
        d["grid"] = _json(d["grid"])
        return d

    async def latest(self, mission_id: str) -> dict[str, Any] | None:
        """The newest version with its arrays (zlib-compressed float32)."""
        rows = await self._all(
            'SELECT version, reason, grid, "row", frozen, event_seq, created_at, p, pod_cum '
            "FROM probability_maps WHERE mission_id = :m ORDER BY version DESC LIMIT 1",
            m=mission_id,
        )
        if not rows:
            return None
        d = dict(rows[0])
        d["grid"] = _json(d["grid"])
        return d

    async def insert(
        self,
        mission_id: str,
        *,
        version: int,
        reason: str,
        grid: dict[str, Any],
        p: bytes,
        pod_cum: bytes,
        row: float,
        frozen: bool,
        event_seq: int,
    ) -> bool:
        """Store the next version: ``version`` must be the newest + 1, and nothing comes
        after a frozen version while a candidate is still confirmed. True when this version
        follows a frozen one (the find was reopened and the map is unlocked)."""
        async with self.engine.begin() as conn:
            await conn.execute(text(_LOCK), {"m": mission_id})
            head = (await conn.execute(text(_HEAD), {"m": mission_id})).mappings().first()
            newest = 0 if head is None else int(head["version"])
            after_frozen = head is not None and bool(head["frozen"])
            if after_frozen and (await conn.execute(text(_CONFIRMED), {"m": mission_id})).scalar():
                raise MapWriteError(
                    "MAP_FROZEN", "Kartan är låst: en kandidat är bekräftad som funnen."
                )
            if version != newest + 1:
                raise MapWriteError(
                    "MAP_VERSION_STALE",
                    f"Kartan har version {newest}; nästa måste vara {newest + 1}.",
                )
            await conn.execute(
                text(
                    "INSERT INTO probability_maps "
                    '(mission_id, version, reason, grid, p, pod_cum, "row", frozen, event_seq) '
                    "VALUES (:m, :v, :reason, CAST(:grid AS jsonb), :p, :pod, :row, :frozen, :seq)"
                ),
                {
                    "m": mission_id,
                    "v": version,
                    "reason": reason,
                    "grid": json.dumps(grid),
                    "p": p,
                    "pod": pod_cum,
                    "row": row,
                    "frozen": frozen,
                    "seq": event_seq,
                },
            )
        return after_frozen
