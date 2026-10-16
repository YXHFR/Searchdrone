"""All SQL of the running server. Nothing else in ``sd_ground`` talks SQL.

Runs as the role ``sd_app``: it can add rows but never change or delete the history
tables (``events``, ``gate_decisions``, ``chat_messages``, ``brain_decisions``).
Geometries are converted to and from GeoJSON here.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from typing import Any, cast

from sqlalchemy import RowMapping, text
from sqlalchemy.ext.asyncio import AsyncEngine

from sd_ground.auth import SessionRecord
from sd_schemas.api import Annotation, ChatMessage, Event, Mission, MissionFile, MissionSummary
from sd_schemas.api import Vehicle as VehicleModel
from sd_schemas.base import new_id
from sd_schemas.commander import MissionMemory
from sd_schemas.mission import OperatorLatch
from sd_schemas.safety import SafetyLimits

__all__ = ["EventRow", "Repo", "TelemetryRow", "VehicleRow"]

EventRow = dict[str, Any]
TelemetryRow = dict[str, Any]
VehicleRow = dict[str, Any]

_MISSION_COLS = """m.id, m.vehicle_id, m.title, m.status, m.paused_by, m.returning_cause,
  m.state_seq, m.limits, m.active_charter_id, c.autonomy_mode, m.ai_enabled, m.created_by,
  m.created_at, m.activated_at, m.landed_at, m.closed_at, m.closed_reason, m.notes"""
_MISSION_FROM = "missions m LEFT JOIN charters c ON c.id = m.active_charter_id"


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _mission(row: RowMapping) -> Mission:
    data = dict(row)
    data["limits"] = SafetyLimits.model_validate(_json(data["limits"]))
    return Mission.model_validate(data)


class Repo:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def close(self) -> None:
        await self.engine.dispose()

    async def _all(self, sql: str, **params: Any) -> Sequence[RowMapping]:
        async with self.engine.connect() as conn:
            return (await conn.execute(text(sql), params)).mappings().all()

    async def _one(self, sql: str, **params: Any) -> RowMapping | None:
        async with self.engine.connect() as conn:
            return (await conn.execute(text(sql), params)).mappings().first()

    async def _exec(self, sql: str, **params: Any) -> int:
        async with self.engine.begin() as conn:
            return (await conn.execute(text(sql), params)).rowcount

    # --- health ----------------------------------------------------------------------

    async def ping(self) -> None:
        await self._one("SELECT 1 AS ok")

    async def schema_revision(self) -> str | None:
        row = await self._one("SELECT version_num FROM alembic_version")
        return None if row is None else str(row["version_num"])

    # --- vehicles --------------------------------------------------------------------

    async def vehicles(self) -> list[VehicleRow]:
        rows = await self._all("SELECT id, name, kind, mav_sysid FROM vehicles ORDER BY id")
        return [dict(r) for r in rows]

    async def vehicle(self, vehicle_id: str) -> VehicleRow | None:
        row = await self._one(
            "SELECT id, name, kind, mav_sysid FROM vehicles WHERE id = :id", id=vehicle_id
        )
        return None if row is None else dict(row)

    async def add_vehicle(self, vehicle_id: str, name: str, kind: str, sysid: int) -> bool:
        """True when the vehicle was added; False when it already existed."""
        count = await self._exec(
            "INSERT INTO vehicles (id, name, kind, mav_sysid) VALUES (:id, :name, :kind, :sysid) "
            "ON CONFLICT (id) DO NOTHING",
            id=vehicle_id,
            name=name,
            kind=kind,
            sysid=sysid,
        )
        return count == 1

    @staticmethod
    def vehicle_model(row: VehicleRow, *, connected: bool) -> VehicleModel:
        return VehicleModel(
            id=row["id"],
            name=row["name"],
            kind=row["kind"],
            mav_sysid=row["mav_sysid"],
            connected=connected,
        )

    # --- missions --------------------------------------------------------------------

    async def open_mission(self, vehicle_id: str) -> Mission | None:
        row = await self._one(
            f"SELECT {_MISSION_COLS} FROM {_MISSION_FROM} "  # noqa: S608 - constants only
            "WHERE m.vehicle_id = :v AND m.status <> 'CLOSED'",
            v=vehicle_id,
        )
        return None if row is None else _mission(row)

    async def open_missions(self) -> list[Mission]:
        rows = await self._all(
            f"SELECT {_MISSION_COLS} FROM {_MISSION_FROM} WHERE m.status <> 'CLOSED'"  # noqa: S608
        )
        return [_mission(r) for r in rows]

    async def mission(self, mission_id: str) -> Mission | None:
        row = await self._one(
            f"SELECT {_MISSION_COLS} FROM {_MISSION_FROM} WHERE m.id = :id",  # noqa: S608
            id=mission_id,
        )
        return None if row is None else _mission(row)

    async def missions(self, status: str | None, limit: int) -> list[MissionSummary]:
        rows = await self._all(
            "SELECT id, vehicle_id, title, status, created_at FROM missions "
            "WHERE (CAST(:status AS text) IS NULL OR status = :status) "
            "ORDER BY created_at DESC LIMIT :limit",
            status=status,
            limit=limit,
        )
        return [MissionSummary.model_validate(dict(r)) for r in rows]

    async def patch_mission(self, mission_id: str, title: str | None, notes: str | None) -> bool:
        count = await self._exec(
            "UPDATE missions SET title = COALESCE(:title, title), "
            "notes = COALESCE(:notes, notes) WHERE id = :id",
            id=mission_id,
            title=title,
            notes=notes,
        )
        return count == 1

    async def set_ai_enabled(self, mission_id: str, enabled: bool) -> bool:
        count = await self._exec(
            "UPDATE missions SET ai_enabled = :enabled WHERE id = :id AND status <> 'CLOSED'",
            id=mission_id,
            enabled=enabled,
        )
        return count == 1

    async def latch(self, vehicle_id: str) -> OperatorLatch | None:
        row = await self._one(
            "SELECT vehicle_id, action, underlying, action_seq, set_by, set_at "
            "FROM operator_latches WHERE vehicle_id = :v AND action IS NOT NULL",
            v=vehicle_id,
        )
        if row is None:
            return None
        return OperatorLatch.model_validate({**dict(row), "set_mono_ms": 0})

    # --- events ----------------------------------------------------------------------

    async def add_events(self, rows: Sequence[EventRow]) -> None:
        if not rows:
            return
        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO events (id, mission_id, vehicle_id, t, source, type, severity, "
                    "payload) VALUES (:id, :mission_id, :vehicle_id, :t, :source, :type, "
                    ":severity, CAST(:payload AS jsonb))"
                ),
                [{**r, "payload": json.dumps(r.get("payload", {}))} for r in rows],
            )

    async def events(
        self, mission_id: str, after: dt.datetime | None, type_: str | None, limit: int
    ) -> list[Event]:
        rows = await self._all(
            "SELECT id, mission_id, vehicle_id, t, source, type, severity, payload FROM events "
            "WHERE mission_id = :m AND (CAST(:after AS timestamptz) IS NULL OR t > :after) "
            "AND (CAST(:type AS text) IS NULL OR type = :type) ORDER BY t, id LIMIT :limit",
            m=mission_id,
            after=after,
            type=type_,
            limit=limit,
        )
        return [Event.model_validate({**dict(r), "payload": _json(r["payload"])}) for r in rows]

    # --- sessions and CLI tokens -----------------------------------------------------

    async def save_session(self, rec: SessionRecord) -> None:
        await self._exec(
            "INSERT INTO auth_sessions (id, kind, name, role, issuer_ref, auth_mode, "
            "token_sha256, created_at, last_used_at, expires_at) VALUES (uuidv7(), :kind, :name, "
            ":role, :issuer_ref, :auth_mode, :sha, :created_at, :last_used_at, :expires_at) "
            "ON CONFLICT (token_sha256) DO UPDATE SET last_used_at = EXCLUDED.last_used_at",
            kind=rec.kind,
            name=rec.name,
            role=rec.role,
            issuer_ref=rec.issuer_ref,
            auth_mode=rec.auth_mode,
            sha=rec.token_sha256,
            created_at=rec.created_at,
            last_used_at=rec.last_used_at,
            expires_at=rec.expires_at,
        )

    async def touch_sessions(self, records: Sequence[SessionRecord]) -> None:
        if not records:
            return
        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    "UPDATE auth_sessions SET last_used_at = :t "
                    "WHERE token_sha256 = :sha AND (last_used_at IS NULL OR last_used_at < :t)"
                ),
                [{"t": r.last_used_at, "sha": r.token_sha256} for r in records],
            )

    async def sessions(self) -> list[SessionRecord]:
        """Every session and token not revoked; the caller decides which are still valid."""
        rows = await self._all(
            "SELECT id, kind, name, role, issuer_ref, auth_mode, token_sha256, created_at, "
            "COALESCE(last_used_at, created_at) AS last_used_at, expires_at FROM auth_sessions "
            "WHERE revoked_at IS NULL AND (expires_at IS NULL OR expires_at > now())"
        )
        return [
            SessionRecord(
                id=str(r["id"]),
                kind=r["kind"],
                role=r["role"],
                issuer_ref=r["issuer_ref"],
                auth_mode=r["auth_mode"],
                token_sha256=r["token_sha256"],
                created_at=r["created_at"],
                last_used_at=r["last_used_at"],
                expires_at=r["expires_at"],
                name=r["name"],
            )
            for r in rows
        ]

    async def token_exists(self, name: str) -> bool:
        row = await self._one(
            "SELECT 1 AS x FROM auth_sessions WHERE kind = 'token' AND name = :n", n=name
        )
        return row is not None

    async def revoke_token(self, name: str) -> bool:
        count = await self._exec(
            "UPDATE auth_sessions SET revoked_at = now(), name = name || '#revoked-' || id "
            "WHERE kind = 'token' AND name = :n AND revoked_at IS NULL",
            n=name,
        )
        return count == 1

    async def revoke_session(self, token_sha256: str) -> None:
        await self._exec(
            "UPDATE auth_sessions SET revoked_at = now() WHERE token_sha256 = :sha "
            "AND revoked_at IS NULL",
            sha=token_sha256,
        )

    # --- telemetry -------------------------------------------------------------------

    async def add_telemetry(self, rows: Sequence[TelemetryRow]) -> None:
        if not rows:
            return
        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO telemetry (vehicle_id, mission_id, t, pos, alt_rel_home_m, "
                    "heading_deg, groundspeed_ms, climb_ms, battery_pct, battery_v, flight_mode, "
                    "armed, landed_state, gps_fix, sats, executor_state) VALUES (:vehicle_id, "
                    ":mission_id, :t, ST_SetSRID(ST_MakePoint(:lon, :lat, :alt_amsl_m), 4326), "
                    ":alt_rel_home_m, :heading_deg, :groundspeed_ms, :climb_ms, :battery_pct, "
                    ":battery_v, :flight_mode, :armed, :landed_state, :gps_fix, :sats, "
                    ":executor_state)"
                ),
                list(rows),
            )

    async def telemetry_count(self, mission_id: str) -> int:
        row = await self._one(
            "SELECT count(*) AS n FROM telemetry WHERE mission_id = :m", m=mission_id
        )
        return 0 if row is None else int(row["n"])

    # --- chat, annotations, files ----------------------------------------------------

    async def chat(
        self, mission_id: str, after_seq: int | None, before_seq: int | None, limit: int
    ) -> list[ChatMessage]:
        rows = await self._all(
            "SELECT * FROM (SELECT id, seq, mission_id, role, author_ref, kind, text, "
            "in_reply_to, refs, parsed_intent, handled_by, created_at FROM chat_messages "
            "WHERE mission_id = :m AND (CAST(:after AS bigint) IS NULL OR seq > :after) "
            "AND (CAST(:before AS bigint) IS NULL OR seq < :before) "
            "ORDER BY seq DESC LIMIT :limit) x ORDER BY seq",
            m=mission_id,
            after=after_seq,
            before=before_seq,
            limit=limit,
        )
        return [self._chat(r) for r in rows]

    @staticmethod
    def _chat(r: RowMapping) -> ChatMessage:
        data = dict(r)
        data["refs"] = _json(data["refs"])
        data["author_role"] = "pilot" if data["role"] == "operator" else None
        data["delivery"] = "n/a"
        return ChatMessage.model_validate(data)

    async def add_chat(
        self,
        *,
        mission_id: str,
        vehicle_id: str,
        msg_id: str,
        role: str,
        author_ref: str,
        kind: str,
        text_: str,
        in_reply_to: str | None,
        refs: dict[str, Any],
    ) -> tuple[ChatMessage, bool]:
        """Add a chat line. A repeated ``msg_id`` returns the stored line (``False``)."""
        async with self.engine.begin() as conn:
            row = (
                (
                    await conn.execute(
                        text(
                            "INSERT INTO chat_messages (id, mission_id, vehicle_id, role, "
                            "author_ref, kind, text, in_reply_to, refs) VALUES (:id, :m, :v, "
                            ":role, :author, :kind, :text, :reply, CAST(:refs AS jsonb)) "
                            "ON CONFLICT (id) DO NOTHING RETURNING id, seq, mission_id, role, "
                            "author_ref, kind, text, in_reply_to, refs, parsed_intent, "
                            "handled_by, created_at"
                        ),
                        {
                            "id": msg_id,
                            "m": mission_id,
                            "v": vehicle_id,
                            "role": role,
                            "author": author_ref,
                            "kind": kind,
                            "text": text_,
                            "reply": in_reply_to,
                            "refs": json.dumps(refs),
                        },
                    )
                )
                .mappings()
                .first()
            )
            created = row is not None
            if row is None:
                row = (
                    (
                        await conn.execute(
                            text(
                                "SELECT id, seq, mission_id, role, author_ref, kind, text, "
                                "in_reply_to, refs, parsed_intent, handled_by, created_at "
                                "FROM chat_messages WHERE id = :id AND mission_id = :m"
                            ),
                            {"id": msg_id, "m": mission_id},
                        )
                    )
                    .mappings()
                    .first()
                )
                if row is None:
                    raise LookupError("message id already used in another mission")
        return self._chat(row), created

    async def chat_message_exists(self, mission_id: str, msg_id: str) -> bool:
        row = await self._one(
            "SELECT 1 AS x FROM chat_messages WHERE id = :id AND mission_id = :m",
            id=msg_id,
            m=mission_id,
        )
        return row is not None

    async def annotations(self, mission_id: str) -> list[Annotation]:
        rows = await self._all(
            "SELECT id, mission_id, kind, label, ST_AsGeoJSON(geom) AS geometry, created_by, "
            "created_at, CASE WHEN kind = 'area' THEN ST_Area(geom::geography) END AS area_m2 "
            "FROM map_annotations WHERE mission_id = :m AND deleted_at IS NULL ORDER BY created_at",
            m=mission_id,
        )
        return [
            Annotation.model_validate({**dict(r), "geometry": _json(r["geometry"])}) for r in rows
        ]

    async def files(self, mission_id: str) -> list[MissionFile]:
        rows = await self._all(
            "SELECT id, kind, rel_path, bytes, sha256, started_at, ended_at FROM mission_files "
            "WHERE mission_id = :m ORDER BY created_at",
            m=mission_id,
        )
        return [MissionFile.model_validate(dict(r)) for r in rows]

    # --- the commander's memory and decision log -------------------------------------

    async def memory(self, mission_id: str) -> MissionMemory | None:
        row = await self._one(
            "SELECT body, updated_at FROM brain_memory WHERE mission_id = :m", m=mission_id
        )
        if row is None:
            return None
        return MissionMemory.model_validate({**_json(row["body"]), "updated_at": row["updated_at"]})

    async def put_memory(self, mission_id: str, body: dict[str, Any], now: dt.datetime) -> None:
        await self._exec(
            "INSERT INTO brain_memory (mission_id, body, updated_at) VALUES (:m, "
            "CAST(:body AS jsonb), :t) ON CONFLICT (mission_id) DO UPDATE SET "
            "body = EXCLUDED.body, updated_at = EXCLUDED.updated_at",
            m=mission_id,
            body=json.dumps(body),
            t=now,
        )

    async def add_decision(
        self, *, decision_id: str, mission_id: str, vehicle_id: str, turn_id: str, body: Any
    ) -> bool:
        count = await self._exec(
            "INSERT INTO brain_decisions (id, mission_id, vehicle_id, turn_id, body) VALUES "
            "(:id, :m, :v, :turn, CAST(:body AS jsonb)) ON CONFLICT (id) DO NOTHING",
            id=decision_id,
            m=mission_id,
            v=vehicle_id,
            turn=turn_id,
            body=json.dumps(body),
        )
        return count == 1


def new_event(
    type_: str,
    *,
    t: dt.datetime,
    source: str = "ground",
    severity: str = "info",
    mission_id: str | None = None,
    vehicle_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> EventRow:
    return {
        "id": str(new_id()),
        "mission_id": mission_id,
        "vehicle_id": vehicle_id,
        "t": t,
        "source": source,
        "type": type_,
        "severity": severity,
        "payload": cast(dict[str, Any], payload or {}),
    }
