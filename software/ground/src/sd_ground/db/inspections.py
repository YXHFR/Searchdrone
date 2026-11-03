"""SQL for the closer look ("Titta närmare"): the ``inspections`` table and the candidate's
moves into and out of ``inspecting``.

Part of the database layer. Both moves are a new candidate revision written under the
mission's candidate lock (the same as ``db.candidates``), together with the look's start
or end row, so a look and its candidate's state never disagree.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import RowMapping, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

__all__ = ["InspectionRepo", "LookRow"]

_OPENABLE = ("ready", "assessing")
_LOCK = "SELECT pg_advisory_xact_lock(hashtext('sd_candidates:' || :m))"
_STATE = (
    "SELECT state, rev, seq FROM candidates WHERE candidate_id = :c AND mission_id = :m "
    "ORDER BY rev DESC LIMIT 1"
)
_NEXT_REV = (
    "INSERT INTO candidates (candidate_id, rev, mission_id, seq, state, geom, sigma95_m, "
    "n_seen, n_in_view, best_detector_score, refined_person_score, priority, merged_into, "
    "latest_assessment_id, first_seen_utc, last_seen_utc, created_at) SELECT candidate_id, "
    ":rev, mission_id, seq, :state, geom, sigma95_m, n_seen, n_in_view, best_detector_score, "
    "refined_person_score, priority, merged_into, latest_assessment_id, first_seen_utc, "
    "last_seen_utc, :now FROM candidates WHERE candidate_id = :c AND rev = :prev"
)
_LOOKS = (
    "SELECT s.command_id::text AS command_id, s.candidate_id::text AS candidate_id, "
    "s.mission_id::text AS mission_id, s.started_at, s.started_by, s.operation_mode, "
    "s.overflight, e.ended_at, e.outcome, e.detail, (SELECT seq FROM candidates k "
    "WHERE k.candidate_id = s.candidate_id LIMIT 1) AS seq FROM inspections s "
    "LEFT JOIN inspections e ON e.command_id = s.command_id AND e.row_kind = 'end' "
    "WHERE s.row_kind = 'start'"
)


@dataclass(frozen=True)
class LookRow:
    command_id: str
    candidate_id: str
    mission_id: str
    seq: int
    started_at: dt.datetime
    started_by: str
    operation_mode: str
    overflight: bool
    ended_at: dt.datetime | None
    outcome: str | None
    detail: str | None


def _look(row: RowMapping) -> LookRow:
    return LookRow(**{k: row[k] for k in LookRow.__dataclass_fields__})


class InspectionRepo:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def _all(self, sql: str, **params: Any) -> Sequence[RowMapping]:
        async with self.engine.connect() as conn:
            return (await conn.execute(text(sql), params)).mappings().all()

    @staticmethod
    async def _move(
        conn: AsyncConnection, candidate_id: str, mission_id: str, now: dt.datetime, state: str
    ) -> None:
        row = (await conn.execute(text(_STATE), {"c": candidate_id, "m": mission_id})).one()
        await conn.execute(
            text(_NEXT_REV),
            {"c": candidate_id, "rev": row.rev + 1, "state": state, "now": now, "prev": row.rev},
        )

    async def start(
        self,
        *,
        command_id: str,
        candidate_id: str,
        mission_id: str,
        started_by: str,
        operation_mode: str,
        overflight: bool,
        now: dt.datetime,
    ) -> bool:
        """The candidate goes ``inspecting`` and the look's start row is written, when the
        candidate is ``ready`` or ``assessing`` and the mission has no open look; else
        nothing is written (``False``)."""
        async with self.engine.begin() as conn:
            await conn.execute(text(_LOCK), {"m": mission_id})
            row = (await conn.execute(text(_STATE), {"c": candidate_id, "m": mission_id})).first()
            if row is None or row.state not in _OPENABLE or await self._open(conn, mission_id):
                return False
            await self._move(conn, candidate_id, mission_id, now, "inspecting")
            await conn.execute(
                text(
                    "INSERT INTO inspections (command_id, candidate_id, mission_id, row_kind, "
                    "started_at, started_by, operation_mode, overflight) VALUES (:cmd, :c, :m, "
                    "'start', :now, :by, :mode, :over)"
                ),
                {
                    "cmd": command_id,
                    "c": candidate_id,
                    "m": mission_id,
                    "now": now,
                    "by": started_by,
                    "mode": operation_mode,
                    "over": overflight,
                },
            )
            return True

    async def end(
        self,
        look: LookRow,
        *,
        outcome: str,
        detail: str | None,
        now: dt.datetime,
        to_ready: bool,
        steps: int | None = None,
        lowest_above_ground_m: float | None = None,
        clear: bool | None = None,
    ) -> bool:
        """The look's end row (once: ``False`` when it already ended) and, with
        ``to_ready``, the candidate back to ``ready`` if it is still ``inspecting``."""
        async with self.engine.begin() as conn:
            await conn.execute(text(_LOCK), {"m": look.mission_id})
            res = await conn.execute(
                text(
                    "INSERT INTO inspections (command_id, candidate_id, mission_id, row_kind, "
                    "ended_at, outcome, detail, steps, lowest_above_ground_m, clear) VALUES "
                    "(:cmd, :c, :m, 'end', :now, :outcome, :detail, :steps, :low, :clear) "
                    "ON CONFLICT (command_id, row_kind) DO NOTHING"
                ),
                {
                    "cmd": look.command_id,
                    "c": look.candidate_id,
                    "m": look.mission_id,
                    "now": now,
                    "outcome": outcome,
                    "detail": detail,
                    "steps": steps,
                    "low": lowest_above_ground_m,
                    "clear": clear,
                },
            )
            if res.rowcount == 0:
                return False
            state = (
                await conn.execute(text(_STATE), {"c": look.candidate_id, "m": look.mission_id})
            ).one()
            if to_ready and state.state == "inspecting":
                await self._move(conn, look.candidate_id, look.mission_id, now, "ready")
            return True

    @staticmethod
    async def _open(conn: AsyncConnection, mission_id: str) -> bool:
        res = await conn.execute(
            text(
                "SELECT 1 FROM inspections s WHERE s.mission_id = :m AND s.row_kind = 'start' "
                "AND NOT EXISTS (SELECT 1 FROM inspections e WHERE e.command_id = s.command_id "
                "AND e.row_kind = 'end') LIMIT 1"
            ),
            {"m": mission_id},
        )
        return res.first() is not None

    async def looks(self, mission_id: str) -> list[LookRow]:
        """Every look of the mission, oldest first."""
        rows = await self._all(
            f"{_LOOKS} AND s.mission_id = :m ORDER BY s.started_at, s.command_id", m=mission_id
        )
        return [_look(r) for r in rows]

    async def look(self, command_id: str) -> LookRow | None:
        rows = await self._all(f"{_LOOKS} AND s.command_id = :c", c=command_id)
        return _look(rows[0]) if rows else None

    async def open_looks(self) -> list[LookRow]:
        """The looks without an end row, in every mission."""
        rows = await self._all(f"{_LOOKS} AND e.command_id IS NULL ORDER BY s.started_at")
        return [_look(r) for r in rows]

    async def inspectable(self, mission_id: str) -> list[str]:
        """The candidates a look may start on: ``ready`` or ``assessing`` now."""
        rows = await self._all(
            "SELECT candidate_id::text AS c FROM (SELECT DISTINCT ON (candidate_id) "
            "candidate_id, state FROM candidates WHERE mission_id = :m "
            "ORDER BY candidate_id, rev DESC) newest WHERE state IN ('ready', 'assessing')",
            m=mission_id,
        )
        return [str(r["c"]) for r in rows]

    async def by_seq(self, mission_id: str, seq: int) -> str | None:
        """The candidate with display number ``seq`` (K3 is 3)."""
        rows = await self._all(
            "SELECT candidate_id::text AS c FROM candidates WHERE mission_id = :m AND seq = :s "
            "LIMIT 1",
            m=mission_id,
            s=seq,
        )
        return str(rows[0]["c"]) if rows else None
