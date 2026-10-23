"""SQL for candidates: the search service's feed and writes, and the panel's queue.

Part of the database layer. Every table here is append-only: a change to a candidate is
a new row with the next ``rev``, written under a per-mission advisory lock so two writers
cannot give out the same display number or the same revision. The search service may
never write ``confirmed`` or ``rejected``, never change the state of a candidate a person
or the pilot holds (``inspecting``; new sightings and scores still come in, the state
stays) and never merge one; the checks are here, next to the write, not only in the API.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import RowMapping, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from sd_schemas.search import CandidateRevision, CandidateState

__all__ = ["CandidateRepo", "CandidateWriteError", "Written"]

# A candidate in these states takes no revision from the search service.
CLOSED = frozenset({"confirmed", "rejected", "merged"})

_LATEST = (
    "SELECT DISTINCT ON (candidate_id) candidate_id::text AS candidate_id, rev, "
    "mission_id::text AS mission_id, seq, state, ST_Y(geom) AS lat, ST_X(geom) AS lon, "
    "sigma95_m, n_seen, n_in_view, best_detector_score, refined_person_score, priority, "
    "merged_into::text AS merged_into, latest_assessment_id::text AS latest_assessment_id, "
    "first_seen_utc, last_seen_utc FROM candidates"
)
_DETECTION = (
    "d.detection_id::text AS detection_id, d.seq, d.still_id::text AS still_id, d.label, "
    "d.score, ST_Y(d.geom) AS lat, ST_X(d.geom) AS lon, d.sigma95_m, d.px_size, d.bbox, "
    "d.crop_box, d.crop_path IS NOT NULL AS has_crop, s.captured_utc"
)


class CandidateWriteError(Exception):
    """A revision the search service may not write (the API answers 409)."""

    def __init__(self, code: str, detail_sv: str) -> None:
        super().__init__(code)
        self.code, self.detail_sv = code, detail_sv


@dataclass(frozen=True)
class Written:
    candidate_id: str
    rev: int
    seq: int
    state: str
    created: bool
    assessment_id: str | None


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


# ``real`` columns read back as float4 noise (0.05 -> 0.0500000007): round them.
_ROUND = {
    "priority": 4,
    "best_detector_score": 4,
    "refined_person_score": 4,
    "score": 4,
    "sigma95_m": 2,
    "px_size": 2,
    "distance_m": 2,
}


def _rounded(row: RowMapping) -> dict[str, Any]:
    d = dict(row)
    for k, n in _ROUND.items():
        if d.get(k) is not None:
            d[k] = round(float(d[k]), n)
    return d


def _detection(row: RowMapping) -> dict[str, Any]:
    d = _rounded(row)
    d["bbox"] = _json(d["bbox"])
    d["crop_box"] = _json(d["crop_box"])
    return d


class CandidateRepo:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def _all(self, sql: str, **params: Any) -> Sequence[RowMapping]:
        async with self.engine.connect() as conn:
            return (await conn.execute(text(sql), params)).mappings().all()

    # --- the search service's reads ---------------------------------------------------

    async def feed(self, mission_id: str, after_seq: int, limit: int) -> list[dict[str, Any]]:
        """The mission's target detections after ``after_seq``, oldest first."""
        rows = await self._all(
            f"SELECT {_DETECTION} FROM detections d "  # noqa: S608 - constants only
            "JOIN perception_stills s ON s.still_id = d.still_id "
            "WHERE d.mission_id = :m AND d.kind = 'target' AND d.seq > :after "
            "ORDER BY d.seq LIMIT :n",
            m=mission_id,
            after=after_seq,
            n=limit,
        )
        return [_detection(r) for r in rows]

    async def crop_path(self, detection_id: str) -> tuple[str, str] | None:
        """(mission id, crop path) of a target detection's crop, if it has one."""
        rows = await self._all(
            "SELECT mission_id::text AS m, crop_path FROM detections "
            "WHERE detection_id = :d AND kind = 'target' AND crop_path IS NOT NULL",
            d=detection_id,
        )
        return None if not rows else (str(rows[0]["m"]), str(rows[0]["crop_path"]))

    async def latest(self, mission_id: str) -> list[dict[str, Any]]:
        """Every candidate of the mission in its newest revision, by display number."""
        rows = await self._all(
            f"SELECT * FROM ({_LATEST} WHERE mission_id = :m "  # noqa: S608 - constants only
            "ORDER BY candidate_id, rev DESC) newest ORDER BY seq, candidate_id",
            m=mission_id,
        )
        return [_rounded(r) for r in rows]

    async def sightings(self, mission_id: str) -> dict[str, list[dict[str, Any]]]:
        """The detections each candidate of the mission observed, oldest first."""
        rows = await self._all(
            f"SELECT o.candidate_id::text AS candidate_id, {_DETECTION} "  # noqa: S608
            "FROM candidate_observations o JOIN detections d ON d.detection_id = o.detection_id "
            "JOIN perception_stills s ON s.still_id = d.still_id "
            "WHERE d.mission_id = :m ORDER BY d.seq",
            m=mission_id,
        )
        out: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            d = _detection(r)
            out.setdefault(d.pop("candidate_id"), []).append(d)
        return out

    async def decided(self, candidate_ids: list[str]) -> set[str]:
        if not candidate_ids:
            return set()
        rows = await self._all(
            "SELECT DISTINCT candidate_id::text AS c FROM candidate_decisions "
            "WHERE candidate_id = ANY(CAST(:ids AS uuid[]))",
            ids=candidate_ids,
        )
        return {str(r["c"]) for r in rows}

    # --- the search service's writes -------------------------------------------------

    async def write(
        self, mission_id: str, revisions: list[CandidateRevision], now: dt.datetime
    ) -> list[Written]:
        """All revisions in one transaction, or none (``CandidateWriteError``)."""
        try:
            async with self.engine.begin() as conn:
                await conn.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext('sd_candidates:' || :m))"),
                    {"m": mission_id},
                )
                out: list[Written] = []
                for rev in revisions:
                    out.append(await self._write_one(conn, mission_id, rev, now))
                return out
        except IntegrityError as exc:
            raise CandidateWriteError(
                "CANDIDATE_CONFLICT", "Kandidaten eller observationen finns redan."
            ) from exc

    async def _latest_one(self, conn: AsyncConnection, candidate_id: str) -> RowMapping | None:
        res = await conn.execute(
            text(f"{_LATEST} WHERE candidate_id = :c ORDER BY candidate_id, rev DESC LIMIT 1"),
            {"c": candidate_id},
        )
        return res.mappings().first()

    async def _write_one(
        self, conn: AsyncConnection, mission_id: str, r: CandidateRevision, now: dt.datetime
    ) -> Written:
        cid = str(r.candidate_id)
        prev = await self._latest_one(conn, cid)
        if r.base_rev == 0:
            if prev is not None:
                raise CandidateWriteError("CANDIDATE_EXISTS", "Kandidaten finns redan.")
            seq_row = await conn.execute(
                text("SELECT COALESCE(MAX(seq), 0) + 1 FROM candidates WHERE mission_id = :m"),
                {"m": mission_id},
            )
            seq = int(seq_row.scalar_one())
            state = str(r.state)
            assessment_id: str | None = None
        else:
            if prev is None or prev["mission_id"] != mission_id:
                raise CandidateWriteError("CANDIDATE_NOT_FOUND", "Kandidaten finns inte.")
            if int(prev["rev"]) != r.base_rev:
                raise CandidateWriteError(
                    "REV_CONFLICT", "Kandidaten har ändrats sedan den lästes."
                )
            if prev["state"] in CLOSED:
                raise CandidateWriteError(
                    "CANDIDATE_CLOSED", "Kandidaten är avgjord eller sammanslagen."
                )
            held = prev["state"] == CandidateState.INSPECTING.value
            if r.state is not None and held:
                raise CandidateWriteError(
                    "CANDIDATE_HELD", "En person eller piloten har kandidaten just nu."
                )
            seq = int(prev["seq"])
            state = str(prev["state"]) if r.state is None else str(r.state)
            assessment_id = prev["latest_assessment_id"]
        if r.merged_into is not None:
            target = await self._latest_one(conn, str(r.merged_into))
            if target is None or target["mission_id"] != mission_id or str(r.merged_into) == cid:
                raise CandidateWriteError(
                    "MERGE_TARGET_INVALID", "Kandidaten att slå ihop med finns inte."
                )
            if target["state"] in CLOSED or target["state"] == CandidateState.INSPECTING.value:
                raise CandidateWriteError(
                    "CANDIDATE_HELD", "En person eller piloten har kandidaten just nu."
                )
            if await self._decided(conn, cid) or await self._decided(conn, str(r.merged_into)):
                raise CandidateWriteError(
                    "CANDIDATE_HELD", "En person har redan bedömt någon av kandidaterna."
                )
        for o in r.observations:
            ok = await conn.execute(
                text(
                    "SELECT 1 FROM detections WHERE detection_id = :d AND mission_id = :m "
                    "AND kind = 'target'"
                ),
                {"d": str(o.detection_id), "m": mission_id},
            )
            if ok.first() is None:
                raise CandidateWriteError(
                    "DETECTION_NOT_FOUND", "Detektionen finns inte i uppdraget."
                )
            await conn.execute(
                text(
                    "INSERT INTO candidate_observations (candidate_id, detection_id, distance_m) "
                    "VALUES (:c, :d, :dist)"
                ),
                {"c": cid, "d": str(o.detection_id), "dist": o.distance_m},
            )
        if r.assessment is not None:
            a = r.assessment
            res = await conn.execute(
                text(
                    "INSERT INTO candidate_assessments (candidate_id, profile_version, engine, "
                    "model_id, body, latency_ms, cost_sek) VALUES (:c, :pv, :e, :mid, "
                    "CAST(:body AS jsonb), :lat, :cost) RETURNING id::text"
                ),
                {
                    "c": cid,
                    "pv": a.profile_version,
                    "e": a.engine,
                    "mid": a.model_id,
                    "body": a.model_dump_json(),
                    "lat": a.latency_ms,
                    "cost": a.cost_sek,
                },
            )
            assessment_id = str(res.scalar_one())
        in_view = await conn.execute(
            text(
                "SELECT COUNT(*) FROM perception_stills WHERE mission_id = :m "
                "AND ST_Covers(footprint, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326))"
            ),
            {"m": mission_id, "lat": r.lat, "lon": r.lon},
        )
        n_in_view = max(r.n_seen, int(in_view.scalar_one()))
        new_rev = r.base_rev + 1
        await conn.execute(
            text(
                "INSERT INTO candidates (candidate_id, rev, mission_id, seq, state, geom, "
                "sigma95_m, n_seen, n_in_view, best_detector_score, refined_person_score, "
                "priority, merged_into, latest_assessment_id, first_seen_utc, last_seen_utc, "
                "created_at) VALUES (:c, :rev, :m, :seq, :state, "
                "ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), :sigma, :n_seen, :n_in_view, "
                ":best, :refined, :priority, :merged_into, :aid, :first, :last, :now)"
            ),
            {
                "c": cid,
                "rev": new_rev,
                "m": mission_id,
                "seq": seq,
                "state": state,
                "lat": r.lat,
                "lon": r.lon,
                "sigma": r.sigma95_m,
                "n_seen": r.n_seen,
                "n_in_view": n_in_view,
                "best": r.best_detector_score,
                "refined": r.refined_person_score,
                "priority": r.priority,
                "merged_into": None if r.merged_into is None else str(r.merged_into),
                "aid": assessment_id,
                "first": r.first_seen_utc,
                "last": r.last_seen_utc,
                "now": now,
            },
        )
        return Written(
            cid, new_rev, seq, state, r.base_rev == 0, assessment_id if r.assessment else None
        )

    @staticmethod
    async def _decided(conn: AsyncConnection, candidate_id: str) -> bool:
        res = await conn.execute(
            text("SELECT 1 FROM candidate_decisions WHERE candidate_id = :c LIMIT 1"),
            {"c": candidate_id},
        )
        return res.first() is not None

    # --- the panel's reads -----------------------------------------------------------

    async def assessments(self, candidate_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Every assessment of the candidates, oldest first."""
        if not candidate_ids:
            return {}
        rows = await self._all(
            "SELECT id::text AS id, candidate_id::text AS candidate_id, body, created_at "
            "FROM candidate_assessments WHERE candidate_id = ANY(CAST(:ids AS uuid[])) "
            "ORDER BY created_at, id",
            ids=candidate_ids,
        )
        out: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            out.setdefault(str(r["candidate_id"]), []).append(
                {"id": r["id"], "created_at": r["created_at"], "body": _json(r["body"])}
            )
        return out

    async def thumbs(self, mission_id: str) -> dict[str, str]:
        """Per candidate: the observed detection with a crop and the largest px_size."""
        rows = await self._all(
            "SELECT DISTINCT ON (o.candidate_id) o.candidate_id::text AS c, "
            "d.detection_id::text AS d FROM candidate_observations o "
            "JOIN detections d ON d.detection_id = o.detection_id "
            "WHERE d.mission_id = :m AND d.crop_path IS NOT NULL "
            "ORDER BY o.candidate_id, d.px_size DESC, d.seq",
            m=mission_id,
        )
        return {str(r["c"]): str(r["d"]) for r in rows}

    async def decisions(self, candidate_id: str) -> list[dict[str, Any]]:
        rows = await self._all(
            "SELECT id::text AS id, decision, note, subject_rev, operator_ref, created_at "
            "FROM candidate_decisions WHERE candidate_id = :c ORDER BY created_at, id",
            c=candidate_id,
        )
        return [dict(r) for r in rows]

    async def observations(self, candidate_id: str) -> list[dict[str, Any]]:
        rows = await self._all(
            "SELECT d.detection_id::text AS detection_id, o.distance_m, d.score, d.label, "
            "d.px_size, ST_Y(d.geom) AS lat, ST_X(d.geom) AS lon, d.sigma95_m, "
            "d.crop_path IS NOT NULL AS has_crop, s.captured_utc FROM candidate_observations o "
            "JOIN detections d ON d.detection_id = o.detection_id "
            "JOIN perception_stills s ON s.still_id = d.still_id "
            "WHERE o.candidate_id = :c ORDER BY d.seq",
            c=candidate_id,
        )
        return [_rounded(r) for r in rows]

    async def candidate_mission(self, candidate_id: str) -> str | None:
        rows = await self._all(
            "SELECT mission_id::text AS m FROM candidates WHERE candidate_id = :c LIMIT 1",
            c=candidate_id,
        )
        return None if not rows else str(rows[0]["m"])

    async def approved_profile(self, mission_id: str) -> dict[str, Any] | None:
        """The approved profile in force (the highest approved version), as the table holds
        it: the body with the row's mission, version and approval. Never a proposal."""
        rows = await self._all(
            "SELECT body, mission_id::text AS mission_id, profile_version, approved_by, "
            "approved_at FROM search_profiles WHERE mission_id = :m "
            "AND approved_by IS NOT NULL AND approved_at IS NOT NULL "
            "ORDER BY profile_version DESC LIMIT 1",
            m=mission_id,
        )
        if not rows:
            return None
        row = rows[0]
        body = row["body"] if isinstance(row["body"], dict) else json.loads(row["body"])
        return {
            **body,
            "mission_id": row["mission_id"],
            "profile_version": int(row["profile_version"]),
            "approved_by": row["approved_by"],
            "approved_at": row["approved_at"],
        }

    async def profile_version(self, mission_id: str) -> int | None:
        """The approved profile version in force, if any."""
        rows = await self._all(
            "SELECT MAX(profile_version) AS v FROM search_profiles "
            "WHERE mission_id = :m AND approved_by IS NOT NULL AND approved_at IS NOT NULL",
            m=mission_id,
        )
        v = rows[0]["v"] if rows else None
        return None if v is None else int(v)
