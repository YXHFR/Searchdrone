"""SQL for the mission flow: missions, charters, sectors, approvals, orders, the gate's
decisions, clearances, latches, annotations and files.

Part of the database layer (``sd_ground.db``): nothing outside it talks SQL. Runs as
``sd_app``, which can add rows everywhere it needs but never change or delete the history
tables. Geometries go in and out as GeoJSON.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Sequence
from typing import Any

from sqlalchemy import RowMapping, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from sd_schemas.api import Approval, Charter, Clearance, Command, Mission
from sd_schemas.base import new_id
from sd_schemas.charter import MissionCharter, Sector
from sd_schemas.gate import GateDecision
from sd_schemas.mission import OperatorLatch
from sd_schemas.safety import SafetyLimits

__all__ = ["FlowRepo", "MissionExistsError"]

_MISSION_COLS = """m.id, m.vehicle_id, m.title, m.status, m.paused_by, m.returning_cause,
  m.state_seq, m.limits, m.active_charter_id, c.autonomy_mode, m.ai_enabled, m.created_by,
  m.created_at, m.activated_at, m.landed_at, m.closed_at, m.closed_reason, m.notes"""
_CHARTER_COLS = """c.id, c.mission_id, c.vehicle_id, c.version, c.status, c.body, c.proposed_by,
  c.autonomy_mode, c.approved_by, c.approved_at, c.install_status, c.fence_hash,
  c.fence_verified_at, c.created_at,
  (SELECT a.id FROM approvals a WHERE a.kind = 'charter' AND a.subject_id = c.id
   ORDER BY a.requested_at DESC LIMIT 1) AS approval_id"""
_APPROVAL_COLS = """id, mission_id, vehicle_id, kind, subject_id, subject_version, status,
  summary_sv, required_confirmations, confirmations, default_autonomy_mode, gate,
  requested_by, requested_at, expires_at, decided_by, decided_at, reason"""
_COMMAND_COLS = """id, mission_id, vehicle_id, skill, params, issuer_kind, issuer_ref, backend,
  rationale_sv, in_reply_to, charter_version, based_on_state_epoch, based_on_state_seq,
  status, status_reason, approval_id, expires_at, created_at, updated_at, command_hash,
  charter_id, ai_config_hash"""
OPEN_STATUSES = ("pending_approval", "dispatched", "received", "accepted", "running")


class MissionExistsError(Exception):
    """The vehicle already has an open mission."""


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def mission_of(row: RowMapping) -> Mission:
    data = dict(row)
    data["limits"] = SafetyLimits.model_validate(_json(data["limits"]))
    return Mission.model_validate(data)


def charter_body(row: RowMapping) -> MissionCharter:
    """The stored charter with the values set at approval (columns) laid over it."""
    body = dict(_json(row["body"]))
    if row["autonomy_mode"] is not None:
        body["autonomy_mode"] = row["autonomy_mode"]
    body["approved_by"] = row["approved_by"]
    body["approved_at"] = row["approved_at"]
    body["fence_hash"] = row["fence_hash"]
    return MissionCharter.model_validate(body)


def charter_of(row: RowMapping) -> Charter:
    return Charter.model_validate(
        {
            "id": row["id"],
            "mission_id": row["mission_id"],
            "version": row["version"],
            "status": row["status"],
            "body": charter_body(row),
            "proposed_by": row["proposed_by"],
            "approved_by": row["approved_by"],
            "approved_at": row["approved_at"],
            "install_status": row["install_status"],
            "fence_hash": row["fence_hash"],
            "fence_verified_at": row["fence_verified_at"],
            "approval_id": row["approval_id"],
            "created_at": row["created_at"],
        }
    )


def approval_of(row: RowMapping) -> Approval:
    data = dict(row)
    data["required_confirmations"] = _json(data["required_confirmations"]) or []
    gate = _json(data.get("gate"))
    data["gate"] = None if gate is None else GateDecision.model_validate(gate)
    data.pop("confirmations", None)
    return Approval.model_validate(data)


def command_of(row: RowMapping) -> Command:
    data = dict(row)
    data["params"] = _json(data["params"])
    epoch, seq = data.pop("based_on_state_epoch"), data.pop("based_on_state_seq")
    data["based_on"] = None if epoch is None else {"state_epoch": epoch, "state_seq": seq}
    for extra in ("command_hash", "charter_id", "ai_config_hash"):
        data.pop(extra, None)
    return Command.model_validate(data)


def _point(lat: float, lon: float) -> str:
    return json.dumps({"type": "Point", "coordinates": [lon, lat]})


class FlowRepo:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def _all(self, sql: str, **params: Any) -> Sequence[RowMapping]:
        async with self.engine.connect() as conn:
            return (await conn.execute(text(sql), params)).mappings().all()

    async def _one(self, sql: str, **params: Any) -> RowMapping | None:
        async with self.engine.connect() as conn:
            return (await conn.execute(text(sql), params)).mappings().first()

    async def _exec(self, sql: str, **params: Any) -> int:
        async with self.engine.begin() as conn:
            return (await conn.execute(text(sql), params)).rowcount

    @staticmethod
    async def _x(conn: AsyncConnection, sql: str, **params: Any) -> Any:
        return await conn.execute(text(sql), params)

    @staticmethod
    async def _event(
        conn: AsyncConnection,
        type_: str,
        *,
        t: dt.datetime,
        mission_id: str | None,
        vehicle_id: str | None,
        payload: dict[str, Any],
        severity: str = "info",
    ) -> None:
        await conn.execute(
            text(
                "INSERT INTO events (id, mission_id, vehicle_id, t, source, type, severity, "
                "payload) VALUES (:id, :m, :v, :t, 'ground', :type, :sev, CAST(:p AS jsonb))"
            ),
            {
                "id": str(new_id()),
                "m": mission_id,
                "v": vehicle_id,
                "t": t,
                "type": type_,
                "sev": severity,
                "p": json.dumps(payload, default=str),
            },
        )

    # --- vehicles and missions ---------------------------------------------------------

    async def ensure_vehicle(self, vehicle_id: str, kind: str, sysid: int) -> None:
        await self._exec(
            "INSERT INTO vehicles (id, name, kind, mav_sysid) VALUES (:id, :id, :kind, :sysid) "
            "ON CONFLICT (id) DO NOTHING",
            id=vehicle_id,
            kind=kind,
            sysid=sysid,
        )

    async def mission(self, mission_id: str) -> Mission | None:
        row = await self._one(
            f"SELECT {_MISSION_COLS} FROM missions m LEFT JOIN charters c "  # noqa: S608
            "ON c.id = m.active_charter_id WHERE m.id = :id",
            id=mission_id,
        )
        return None if row is None else mission_of(row)

    async def open_mission(self, vehicle_id: str) -> Mission | None:
        row = await self._one(
            f"SELECT {_MISSION_COLS} FROM missions m LEFT JOIN charters c "  # noqa: S608
            "ON c.id = m.active_charter_id WHERE m.vehicle_id = :v AND m.status <> 'CLOSED'",
            v=vehicle_id,
        )
        return None if row is None else mission_of(row)

    async def create_mission(
        self,
        *,
        mission_id: str,
        vehicle_id: str,
        title: str,
        created_by: str,
        limits: SafetyLimits,
        now: dt.datetime,
    ) -> None:
        """A new mission in DRAFT; ``MissionExistsError`` when one is already open."""
        try:
            async with self.engine.begin() as conn:
                await self._x(
                    conn,
                    "INSERT INTO missions (id, vehicle_id, title, status, state_seq, limits, "
                    "created_by, created_at, data_dir) VALUES (:id, :v, :title, 'DRAFT', 1, "
                    "CAST(:limits AS jsonb), :by, :t, :dir)",
                    id=mission_id,
                    v=vehicle_id,
                    title=title,
                    limits=limits.model_dump_json(),
                    by=created_by,
                    t=now,
                    dir=f"missions/{mission_id}",
                )
                await self._event(
                    conn,
                    "mission.state_changed",
                    t=now,
                    mission_id=mission_id,
                    vehicle_id=vehicle_id,
                    payload={
                        "from": None,
                        "to": "DRAFT",
                        "trigger": "MISSION_OPENED",
                        "state_seq": 1,
                        "created_by": created_by,
                    },
                )
        except IntegrityError:
            raise MissionExistsError(vehicle_id) from None

    async def set_state(
        self,
        mission_id: str,
        *,
        expect_seq: int,
        status: str,
        paused_by: str | None,
        returning_cause: str | None,
        trigger: str,
        previous: str,
        now: dt.datetime,
        closed_reason: str | None = None,
    ) -> int | None:
        """Move the mission to ``status`` if nobody moved it since ``expect_seq``; the new
        ``state_seq``, or ``None`` when it had moved (the caller reads it again)."""
        stamps = {
            "ACTIVE": "activated_at = COALESCE(activated_at, :t),",
            "LANDED": "landed_at = :t, disarmed_at = :t,",
            "CLOSED": "closed_at = :t, closed_reason = :reason,",
        }.get(status, "")
        async with self.engine.begin() as conn:
            row = (
                (
                    await self._x(
                        conn,
                        f"UPDATE missions SET {stamps} status = :s, paused_by = :p, "  # noqa: S608
                        "returning_cause = :r, state_seq = state_seq + 1 "
                        "WHERE id = :id AND state_seq = :seq RETURNING state_seq, vehicle_id",
                        s=status,
                        p=paused_by,
                        r=returning_cause,
                        id=mission_id,
                        seq=expect_seq,
                        t=now,
                        reason=closed_reason,
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            await self._event(
                conn,
                "mission.state_changed",
                t=now,
                mission_id=mission_id,
                vehicle_id=row["vehicle_id"],
                payload={
                    "from": previous,
                    "to": status,
                    "paused_by": paused_by,
                    "returning_cause": returning_cause,
                    "trigger": trigger,
                    "state_seq": int(row["state_seq"]),
                },
            )
            return int(row["state_seq"])

    async def set_limits(self, mission_id: str, limits: SafetyLimits) -> bool:
        return (
            await self._exec(
                "UPDATE missions SET limits = CAST(:l AS jsonb) WHERE id = :id "
                "AND status <> 'CLOSED'",
                l=limits.model_dump_json(),
                id=mission_id,
            )
            == 1
        )

    async def data_dir(self, mission_id: str) -> str | None:
        row = await self._one("SELECT data_dir FROM missions WHERE id = :id", id=mission_id)
        return None if row is None else str(row["data_dir"])

    # --- charters ----------------------------------------------------------------------

    async def charters(self, mission_id: str) -> list[Charter]:
        rows = await self._all(
            f"SELECT {_CHARTER_COLS} FROM charters c WHERE c.mission_id = :m "  # noqa: S608
            "ORDER BY c.version",
            m=mission_id,
        )
        return [charter_of(r) for r in rows]

    async def charter(self, charter_id: str) -> Charter | None:
        row = await self._one(
            f"SELECT {_CHARTER_COLS} FROM charters c WHERE c.id = :id",  # noqa: S608
            id=charter_id,
        )
        return None if row is None else charter_of(row)

    async def next_version(self, mission_id: str) -> int:
        row = await self._one(
            "SELECT COALESCE(max(version), 0) + 1 AS v FROM charters WHERE mission_id = :m",
            m=mission_id,
        )
        return 1 if row is None else int(row["v"])

    async def add_proposal(
        self,
        charter: MissionCharter,
        *,
        proposed_by: str,
        approval: dict[str, Any],
        now: dt.datetime,
    ) -> list[str]:
        """Store a proposed charter and its card; older waiting proposals (and their
        cards) become superseded. Returns the superseded charter ids."""
        body = charter.model_dump(mode="json")
        home = charter.home
        async with self.engine.begin() as conn:
            old = (
                (
                    await self._x(
                        conn,
                        "UPDATE charters SET status = 'superseded' WHERE mission_id = :m AND "
                        "status = 'proposed' RETURNING id",
                        m=str(charter.mission_id),
                    )
                )
                .scalars()
                .all()
            )
            if old:
                await self._x(
                    conn,
                    "UPDATE approvals SET status = 'superseded', reason = 'NEWER_PROPOSAL' "
                    "WHERE kind = 'charter' AND status = 'pending' AND subject_id = ANY(:ids)",
                    ids=list(old),
                )
            await self._x(
                conn,
                "INSERT INTO charters (id, mission_id, vehicle_id, version, status, body, area, "
                "home, pilot_position, proposed_by, created_at) VALUES (:id, :m, :v, :ver, "
                "'proposed', CAST(:body AS jsonb), ST_SetSRID(ST_GeomFromGeoJSON(:area), 4326), "
                "ST_SetSRID(ST_MakePoint(:hlon, :hlat, :halt), 4326), "
                "ST_SetSRID(ST_MakePoint(:plon, :plat), 4326), :by, :t)",
                id=str(charter.charter_id),
                m=str(charter.mission_id),
                v=charter.vehicle_id,
                ver=charter.version,
                body=json.dumps(body),
                area=json.dumps(charter.area),
                hlon=home.lon,
                hlat=home.lat,
                halt=home.alt_amsl_m,
                plon=charter.pilot_position.lon,
                plat=charter.pilot_position.lat,
                by=proposed_by,
                t=now,
            )
            await self._insert_approval(conn, approval)
            await self._event(
                conn,
                "charter.proposed",
                t=now,
                mission_id=str(charter.mission_id),
                vehicle_id=charter.vehicle_id,
                payload={
                    "charter_id": str(charter.charter_id),
                    "version": charter.version,
                    "by": proposed_by,
                },
            )
        return [str(x) for x in old]

    async def approve_charter(
        self,
        *,
        charter: MissionCharter,
        approval_id: str,
        by: str,
        now: dt.datetime,
        confirmations: list[str],
        sectors: list[Sector],
        keep_fence_from: str | None,
    ) -> None:
        """Approve in one transaction: the older approved charter becomes superseded, this
        one approved with its mode, the mission points at it, the card is decided, the
        sectors follow (unknown ids get ``skipped``). ``keep_fence_from``: an approved
        charter whose verified fence carries over (a smaller area in the air)."""
        cid, mid, vid = str(charter.charter_id), str(charter.mission_id), charter.vehicle_id
        async with self.engine.begin() as conn:
            await self._x(
                conn,
                "UPDATE charters SET status = 'superseded' WHERE mission_id = :m AND "
                "status = 'approved'",
                m=mid,
            )
            await self._x(
                conn,
                "UPDATE charters c SET status = 'approved', autonomy_mode = :mode, "
                "approved_by = :by, approved_at = :t, install_status = 'none', "
                "fence_hash = o.fence_hash, fence_verified_at = o.fence_verified_at "
                "FROM (SELECT CAST(:keep AS uuid) AS id) k "
                "LEFT JOIN charters o ON o.id = k.id WHERE c.id = :id",
                mode=str(charter.autonomy_mode),
                by=by,
                t=now,
                keep=keep_fence_from,
                id=cid,
            )
            await self._x(
                conn, "UPDATE missions SET active_charter_id = :c WHERE id = :m", c=cid, m=mid
            )
            await self._x(
                conn,
                "UPDATE approvals SET status = 'approved', decided_by = :by, decided_at = :t, "
                "confirmations = CAST(:conf AS jsonb) WHERE id = :id",
                by=by,
                t=now,
                conf=json.dumps(confirmations),
                id=approval_id,
            )
            await self._sectors(conn, mid, cid, vid, sectors, now)
            await self._event(
                conn,
                "charter.approved",
                t=now,
                mission_id=mid,
                vehicle_id=vid,
                payload={
                    "charter_id": cid,
                    "version": charter.version,
                    "autonomy_mode": str(charter.autonomy_mode),
                    "by": by,
                },
            )

    async def _sectors(
        self,
        conn: AsyncConnection,
        mission_id: str,
        charter_id: str,
        vehicle_id: str,
        sectors: list[Sector],
        now: dt.datetime,
    ) -> None:
        ids = [s.sector_id for s in sectors]
        for s in sectors:
            await self._x(
                conn,
                "INSERT INTO mission_sectors (mission_id, sector_id, charter_id, vehicle_id, name, "
                "order_idx, area, updated_at) VALUES (:m, :sid, :c, :v, :name, :idx, "
                "ST_SetSRID(ST_GeomFromGeoJSON(:area), 4326), :t) ON CONFLICT (mission_id, "
                "sector_id) DO UPDATE SET charter_id = EXCLUDED.charter_id, name = EXCLUDED.name, "
                "order_idx = EXCLUDED.order_idx, area = EXCLUDED.area, "
                "updated_at = EXCLUDED.updated_at, "
                "status = CASE WHEN mission_sectors.status = 'skipped' THEN 'pending' "
                "ELSE mission_sectors.status END",
                m=mission_id,
                sid=s.sector_id,
                c=charter_id,
                v=vehicle_id,
                name=s.name,
                idx=s.order_idx,
                area=json.dumps(s.area),
                t=now,
            )
        await self._x(
            conn,
            "UPDATE mission_sectors SET status = 'skipped', updated_at = :t WHERE mission_id = :m "
            "AND NOT (sector_id = ANY(:ids))",
            m=mission_id,
            ids=ids,
            t=now,
        )

    async def reject_charter(self, charter_id: str) -> None:
        await self._exec(
            "UPDATE charters SET status = 'rejected' WHERE id = :id AND status = 'proposed'",
            id=charter_id,
        )

    async def waiting_proposals(self, mission_id: str) -> int:
        row = await self._one(
            "SELECT count(*) AS n FROM charters WHERE mission_id = :m AND status = 'proposed'",
            m=mission_id,
        )
        return 0 if row is None else int(row["n"])

    async def set_install_status(self, charter_id: str, status: str) -> None:
        await self._exec(
            "UPDATE charters SET install_status = :s WHERE id = :id", s=status, id=charter_id
        )

    async def set_fence(
        self, charter_id: str, fence_hash: str | None, verified_at: dt.datetime | None
    ) -> None:
        await self._exec(
            "UPDATE charters SET fence_hash = :h, fence_verified_at = :t WHERE id = :id",
            h=fence_hash,
            t=verified_at,
            id=charter_id,
        )

    # --- sectors -----------------------------------------------------------------------

    async def sectors(self, mission_id: str) -> list[dict[str, Any]]:
        rows = await self._all(
            "SELECT sector_id, name, order_idx, status, progress, ST_AsGeoJSON(area) AS area, "
            "ST_Area(area::geography) AS area_m2, ST_AsGeoJSON(covered) AS covered "
            "FROM mission_sectors WHERE mission_id = :m ORDER BY order_idx",
            m=mission_id,
        )
        return [
            {
                **dict(r),
                "area": _json(r["area"]),
                "covered": _json(r["covered"]),
                "area_m2": float(r["area_m2"]),
            }
            for r in rows
        ]

    async def set_sector_progress(
        self, mission_id: str, sector_id: str, status: str, progress: float, now: dt.datetime
    ) -> None:
        await self._exec(
            "UPDATE mission_sectors SET status = :s, progress = :p, updated_at = :t "
            "WHERE mission_id = :m AND sector_id = :sid AND status <> 'skipped'",
            s=status,
            p=max(0.0, min(1.0, progress)),
            t=now,
            m=mission_id,
            sid=sector_id,
        )

    async def set_sector_covered(
        self, mission_id: str, sector_id: str, covered: dict[str, Any] | None, now: dt.datetime
    ) -> None:
        await self._exec(
            "UPDATE mission_sectors SET covered = CASE WHEN CAST(:g AS text) IS NULL THEN NULL "
            "ELSE ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(:g), 4326)) END, updated_at = :t "
            "WHERE mission_id = :m AND sector_id = :sid",
            g=None if covered is None else json.dumps(covered),
            t=now,
            m=mission_id,
            sid=sector_id,
        )

    # --- approvals ---------------------------------------------------------------------

    @staticmethod
    async def _insert_approval(conn: AsyncConnection, a: dict[str, Any]) -> None:
        await conn.execute(
            text(
                "INSERT INTO approvals (id, mission_id, vehicle_id, kind, subject_id, "
                "subject_version, status, summary_sv, required_confirmations, "
                "default_autonomy_mode, gate, requested_by, requested_at, expires_at, "
                "decided_by, decided_at, reason) VALUES (:id, :mission_id, :vehicle_id, :kind, "
                ":subject_id, :subject_version, :status, :summary_sv, CAST(:req AS jsonb), "
                ":default_autonomy_mode, CAST(:gate AS jsonb), :requested_by, :requested_at, "
                ":expires_at, :decided_by, :decided_at, :reason)"
            ),
            {
                "id": a["id"],
                "mission_id": a["mission_id"],
                "vehicle_id": a["vehicle_id"],
                "kind": a["kind"],
                "subject_id": a["subject_id"],
                "subject_version": a.get("subject_version", 1),
                "status": a.get("status", "pending"),
                "summary_sv": a["summary_sv"],
                "req": json.dumps(a.get("required_confirmations", [])),
                "default_autonomy_mode": a.get("default_autonomy_mode"),
                "gate": None if a.get("gate") is None else json.dumps(a["gate"]),
                "requested_by": a["requested_by"],
                "requested_at": a["requested_at"],
                "expires_at": a.get("expires_at"),
                "decided_by": a.get("decided_by"),
                "decided_at": a.get("decided_at"),
                "reason": a.get("reason"),
            },
        )

    async def add_approval(self, a: dict[str, Any]) -> None:
        async with self.engine.begin() as conn:
            await self._insert_approval(conn, a)

    async def approval(self, approval_id: str) -> Approval | None:
        row = await self._one(
            f"SELECT {_APPROVAL_COLS} FROM approvals WHERE id = :id",  # noqa: S608
            id=approval_id,
        )
        return None if row is None else approval_of(row)

    async def approvals(self, mission_id: str, status: str | None) -> list[Approval]:
        rows = await self._all(
            f"SELECT {_APPROVAL_COLS} FROM approvals WHERE mission_id = :m "  # noqa: S608
            "AND (CAST(:s AS text) IS NULL OR status = :s) ORDER BY requested_at",
            m=mission_id,
            s=status,
        )
        return [approval_of(r) for r in rows]

    async def decide_approval(
        self,
        approval_id: str,
        *,
        status: str,
        by: str | None,
        now: dt.datetime,
        reason: str | None,
        confirmations: list[str] | None = None,
    ) -> bool:
        """Close a pending card; ``False`` when it was no longer pending."""
        return (
            await self._exec(
                "UPDATE approvals SET status = :s, decided_by = :by, decided_at = :t, "
                "reason = :r, confirmations = COALESCE(CAST(:conf AS jsonb), confirmations) "
                "WHERE id = :id AND status = 'pending'",
                s=status,
                by=by if status == "approved" or by is not None else None,
                t=now,
                r=reason,
                conf=None if confirmations is None else json.dumps(confirmations),
                id=approval_id,
            )
            == 1
        )

    async def expired_approvals(self, now: dt.datetime) -> list[Approval]:
        rows = await self._all(
            f"SELECT {_APPROVAL_COLS} FROM approvals WHERE status = 'pending' "  # noqa: S608
            "AND expires_at IS NOT NULL AND expires_at <= :t",
            t=now,
        )
        return [approval_of(r) for r in rows]

    async def pending_command_approvals(self, vehicle_id: str) -> list[Approval]:
        rows = await self._all(
            f"SELECT {_APPROVAL_COLS} FROM approvals WHERE vehicle_id = :v "  # noqa: S608
            "AND kind = 'command' AND status = 'pending'",
            v=vehicle_id,
        )
        return [approval_of(r) for r in rows]

    async def brain_pending_approvals(self, vehicle_id: str) -> int:
        row = await self._one(
            "SELECT count(*) AS n FROM approvals a JOIN commands c ON c.id = a.subject_id "
            "WHERE a.vehicle_id = :v AND a.kind = 'command' AND a.status = 'pending' "
            "AND c.issuer_kind = 'brain'",
            v=vehicle_id,
        )
        return 0 if row is None else int(row["n"])

    async def expire_all_pending(self, now: dt.datetime) -> int:
        """At start: every waiting order card is void (``SERVER_RESTART``)."""
        return await self._exec(
            "UPDATE approvals SET status = 'expired', decided_at = :t, reason = 'SERVER_RESTART' "
            "WHERE status = 'pending' AND kind = 'command'",
            t=now,
        )

    # --- orders and the gate's decisions -----------------------------------------------

    async def command(self, command_id: str) -> Command | None:
        row = await self._one(
            f"SELECT {_COMMAND_COLS} FROM commands WHERE id = :id",  # noqa: S608
            id=command_id,
        )
        return None if row is None else command_of(row)

    async def command_row(self, command_id: str) -> dict[str, Any] | None:
        row = await self._one(
            f"SELECT {_COMMAND_COLS} FROM commands WHERE id = :id",  # noqa: S608
            id=command_id,
        )
        return None if row is None else {**dict(row), "params": _json(row["params"])}

    async def commands(self, mission_id: str, status: str | None) -> list[Command]:
        rows = await self._all(
            f"SELECT {_COMMAND_COLS} FROM commands WHERE mission_id = :m "  # noqa: S608
            "AND (CAST(:s AS text) IS NULL OR status = :s) ORDER BY created_at",
            m=mission_id,
            s=status,
        )
        return [command_of(r) for r in rows]

    async def open_commands(self, vehicle_id: str) -> list[Command]:
        rows = await self._all(
            f"SELECT {_COMMAND_COLS} FROM commands WHERE vehicle_id = :v "  # noqa: S608
            "AND status = ANY(:open) ORDER BY created_at",
            v=vehicle_id,
            open=list(OPEN_STATUSES),
        )
        return [command_of(r) for r in rows]

    async def add_command(
        self,
        cmd: dict[str, Any],
        gate: GateDecision,
        *,
        phase: str,
        state_seq: int | None,
        now: dt.datetime,
        approval: dict[str, Any] | None,
        consume_clearance: str | None,
    ) -> None:
        """The order, its gate decision, its card and a consumed clearance: one
        transaction."""
        async with self.engine.begin() as conn:
            await self._x(
                conn,
                "INSERT INTO commands (id, mission_id, vehicle_id, charter_id, charter_version, "
                "issuer_kind, issuer_ref, backend, ai_config_hash, skill, params, rationale_sv, "
                "in_reply_to, based_on_state_epoch, based_on_state_seq, status, status_reason, "
                "approval_id, command_hash, expires_at, target, created_at, updated_at) VALUES "
                "(:id, :mission_id, :vehicle_id, :charter_id, :charter_version, :issuer_kind, "
                ":issuer_ref, :backend, :ai_config_hash, :skill, CAST(:params AS jsonb), "
                ":rationale_sv, :in_reply_to, :epoch, :seq, :status, :status_reason, NULL, "
                ":command_hash, :expires_at, CASE WHEN CAST(:target AS text) IS NULL THEN NULL "
                "ELSE ST_SetSRID(ST_GeomFromGeoJSON(:target), 4326) END, :t, :t)",
                **{
                    **cmd,
                    "params": json.dumps(cmd["params"]),
                    "target": None if cmd.get("target") is None else json.dumps(cmd["target"]),
                    "t": now,
                },
            )
            await self._gate(conn, cmd, gate, phase=phase, state_seq=state_seq, now=now)
            if approval is not None:
                await self._insert_approval(conn, approval)
                await self._x(
                    conn,
                    "UPDATE commands SET approval_id = :a WHERE id = :id",
                    a=approval["id"],
                    id=cmd["id"],
                )
            if consume_clearance is not None:
                await self._consume(conn, consume_clearance, cmd["id"], now)
            await self._event(
                conn,
                "command.created",
                t=now,
                mission_id=cmd["mission_id"],
                vehicle_id=cmd["vehicle_id"],
                payload={
                    "command_id": cmd["id"],
                    "skill": cmd["skill"],
                    "status": cmd["status"],
                    "issuer": cmd["issuer_ref"],
                    "decision": gate.decision,
                    "reason_codes": gate.reason_codes,
                },
            )

    async def _gate(
        self,
        conn: AsyncConnection,
        cmd: dict[str, Any],
        gate: GateDecision,
        *,
        phase: str,
        state_seq: int | None,
        now: dt.datetime,
    ) -> None:
        await self._x(
            conn,
            "INSERT INTO gate_decisions (id, mission_id, vehicle_id, command_id, side, phase, "
            "decision, reason_codes, rule_ids, message_sv, details, charter_version, state_seq, "
            "created_at) VALUES (:id, :m, :v, :c, :side, :phase, :d, :codes, :rules, :msg, "
            "CAST(:details AS jsonb), :cv, :seq, :t)",
            id=str(new_id()),
            m=cmd["mission_id"],
            v=cmd["vehicle_id"],
            c=cmd["id"],
            side=gate.side,
            phase=phase,
            d=gate.decision,
            codes=list(gate.reason_codes),
            rules=list(gate.rule_ids),
            msg=gate.message_sv,
            details=json.dumps(gate.details, default=str),
            cv=cmd.get("charter_version"),
            seq=state_seq,
            t=now,
        )

    async def add_gate_decision(
        self,
        cmd: dict[str, Any],
        gate: GateDecision,
        *,
        phase: str,
        state_seq: int | None,
        now: dt.datetime,
    ) -> None:
        async with self.engine.begin() as conn:
            await self._gate(conn, cmd, gate, phase=phase, state_seq=state_seq, now=now)

    async def gate_decisions(
        self, mission_id: str, limit: int, command_id: str | None = None
    ) -> list[GateDecision]:
        rows = await self._all(
            "SELECT decision, reason_codes, rule_ids, message_sv, details, side FROM "
            "gate_decisions WHERE mission_id = :m AND (CAST(:c AS uuid) IS NULL OR "
            "command_id = CAST(:c AS uuid)) ORDER BY created_at DESC LIMIT :n",
            m=mission_id,
            c=command_id,
            n=limit,
        )
        return [
            GateDecision.model_validate({**dict(r), "details": _json(r["details"])}) for r in rows
        ]

    async def first_gate(self, command_id: str) -> GateDecision | None:
        """The decision an order got when it was submitted."""
        row = await self._one(
            "SELECT decision, reason_codes, rule_ids, message_sv, details, side FROM "
            "gate_decisions WHERE command_id = :c AND phase = 'submit' ORDER BY created_at "
            "LIMIT 1",
            c=command_id,
        )
        if row is None:
            return None
        return GateDecision.model_validate({**dict(row), "details": _json(row["details"])})

    async def set_command_status(
        self,
        command_id: str,
        status: str,
        *,
        reason: str | None,
        now: dt.datetime,
        only_from: Sequence[str] | None = None,
        approval_id: str | None = None,
        command_hash: str | None = None,
        charter_version: int | None = None,
        charter_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Change an order's status (only from ``only_from`` when given) and write the
        ``command.status`` event in the same transaction. Returns the old and new row,
        or ``None`` when nothing changed."""
        async with self.engine.begin() as conn:
            old = (
                (
                    await self._x(
                        conn,
                        "SELECT status FROM commands WHERE id = :id FOR UPDATE",
                        id=command_id,
                    )
                )
                .mappings()
                .first()
            )
            if old is None or (only_from is not None and old["status"] not in only_from):
                return None
            if old["status"] == status:
                return None
            row = (
                (
                    await self._x(
                        conn,
                        "UPDATE commands SET status = :s, status_reason = :r, updated_at = :t, "  # noqa: S608
                        "approval_id = COALESCE(:a, approval_id), "
                        "command_hash = COALESCE(:h, command_hash), "
                        "charter_version = COALESCE(:cv, charter_version), "
                        "charter_id = COALESCE(:cid, charter_id) "
                        f"WHERE id = :id RETURNING {_COMMAND_COLS}",
                        s=status,
                        r=reason,
                        t=now,
                        a=approval_id,
                        h=command_hash,
                        cv=charter_version,
                        cid=charter_id,
                        id=command_id,
                    )
                )
                .mappings()
                .first()
            )
            if row is None:
                raise RuntimeError("row vanished after write")
            await self._event(
                conn,
                "command.status",
                t=now,
                mission_id=str(row["mission_id"]),
                vehicle_id=row["vehicle_id"],
                payload={
                    "command_id": command_id,
                    "from": old["status"],
                    "to": status,
                    "reason": reason,
                },
            )
            return {"from": old["status"], "row": row}

    # --- clearances --------------------------------------------------------------------

    async def add_clearance(
        self,
        *,
        clearance_id: str,
        mission_id: str,
        vehicle_id: str,
        charter_id: str,
        charter_version: int,
        by: str,
        now: dt.datetime,
        valid_until: dt.datetime,
    ) -> None:
        """A new "clear around the drone"; an older active one for the vehicle ends."""
        async with self.engine.begin() as conn:
            await self._x(
                conn,
                "UPDATE clearances SET status = 'invalidated', ended_at = :t, end_reason = "
                "'REPLACED' WHERE vehicle_id = :v AND status = 'active'",
                v=vehicle_id,
                t=now,
            )
            await self._x(
                conn,
                "INSERT INTO clearances (id, mission_id, vehicle_id, charter_id, charter_version, "
                "status, confirmed_by, confirmed_at, valid_until) VALUES (:id, :m, :v, :c, :cv, "
                "'active', :by, :t, :until)",
                id=clearance_id,
                m=mission_id,
                v=vehicle_id,
                c=charter_id,
                cv=charter_version,
                by=by,
                t=now,
                until=valid_until,
            )
            await self._event(
                conn,
                "clearance.confirmed",
                t=now,
                mission_id=mission_id,
                vehicle_id=vehicle_id,
                payload={
                    "clearance_id": clearance_id,
                    "by": by,
                    "charter_version": charter_version,
                },
            )

    async def _consume(
        self, conn: AsyncConnection, clearance_id: str, command_id: str, now: dt.datetime
    ) -> None:
        await self._x(
            conn,
            "UPDATE clearances SET status = 'consumed', consumed_by_command_id = :c, "
            "ended_at = :t, end_reason = 'TAKEOFF' WHERE id = :id AND status = 'active'",
            c=command_id,
            t=now,
            id=clearance_id,
        )
        await self._event(
            conn,
            "clearance.consumed",
            t=now,
            mission_id=None,
            vehicle_id=None,
            payload={"clearance_id": clearance_id, "command_id": command_id},
        )

    async def consume_clearance(self, clearance_id: str, command_id: str, now: dt.datetime) -> None:
        async with self.engine.begin() as conn:
            await self._consume(conn, clearance_id, command_id, now)

    async def clearance(self, clearance_id: str) -> Clearance | None:
        row = await self._one(
            "SELECT id, mission_id, vehicle_id, charter_id, charter_version, status, "
            "confirmed_by, confirmed_at, valid_until FROM clearances WHERE id = :id",
            id=clearance_id,
        )
        return None if row is None else Clearance.model_validate(dict(row))

    async def end_clearances(
        self, vehicle_id: str, status: str, reason: str, now: dt.datetime
    ) -> int:
        return await self._exec(
            "UPDATE clearances SET status = :s, ended_at = :t, end_reason = :r "
            "WHERE vehicle_id = :v AND status = 'active'",
            s=status,
            t=now,
            r=reason,
            v=vehicle_id,
        )

    async def invalidate_all_clearances(self, now: dt.datetime) -> int:
        return await self._exec(
            "UPDATE clearances SET status = 'invalidated', ended_at = :t, end_reason = "
            "'SERVER_RESTART' WHERE status = 'active'",
            t=now,
        )

    # --- latches -----------------------------------------------------------------------

    async def mirror_latch(
        self, vehicle_id: str, latch: OperatorLatch | None, action_seq: int, mission_id: str | None
    ) -> None:
        await self._exec(
            "INSERT INTO operator_latches (vehicle_id, action_seq, action, underlying, set_by, "
            "set_at, mission_id) VALUES (:v, :seq, :a, :u, :by, :t, :m) ON CONFLICT (vehicle_id) "
            "DO UPDATE SET action_seq = "
            "GREATEST(operator_latches.action_seq, EXCLUDED.action_seq), "
            "action = EXCLUDED.action, underlying = EXCLUDED.underlying, set_by = EXCLUDED.set_by, "
            "set_at = EXCLUDED.set_at, mission_id = EXCLUDED.mission_id",
            v=vehicle_id,
            seq=action_seq,
            a=None if latch is None else latch.action,
            u=None if latch is None else latch.underlying,
            by=None if latch is None else latch.set_by,
            t=None if latch is None else latch.set_at,
            m=None if latch is None else mission_id,
        )

    async def latch_seq(self, vehicle_id: str) -> int:
        row = await self._one(
            "SELECT action_seq FROM operator_latches WHERE vehicle_id = :v", v=vehicle_id
        )
        return 0 if row is None else int(row["action_seq"])

    # --- annotations -------------------------------------------------------------------

    async def add_annotation(
        self,
        *,
        mission_id: str,
        vehicle_id: str,
        kind: str,
        label: str | None,
        geometry: dict[str, Any],
        created_by: str,
        now: dt.datetime,
    ) -> str | None:
        """A pin, area or path. ``label = None`` gives the next free ``M<n>`` (pins the AI
        marks). ``None`` when the label is taken."""
        async with self.engine.begin() as conn:
            if label is None:
                n = (
                    await self._x(
                        conn,
                        "SELECT count(*) FROM map_annotations WHERE mission_id = :m "
                        "AND label ~ '^M[0-9]+$'",
                        m=mission_id,
                    )
                ).scalar_one()
                label = f"M{int(n) + 1}"
            try:
                async with conn.begin_nested():
                    new = str(new_id())
                    await self._x(
                        conn,
                        "INSERT INTO map_annotations (id, mission_id, vehicle_id, kind, label, "
                        "geom, created_by, created_at) VALUES (:id, :m, :v, :k, :l, "
                        "ST_SetSRID(ST_GeomFromGeoJSON(:g), 4326), :by, :t)",
                        id=new,
                        m=mission_id,
                        v=vehicle_id,
                        k=kind,
                        l=label,
                        g=json.dumps(geometry),
                        by=created_by,
                        t=now,
                    )
            except IntegrityError:
                return None
            return new

    async def annotation(self, annotation_id: str) -> dict[str, Any] | None:
        row = await self._one(
            "SELECT id, mission_id, vehicle_id, kind, label, ST_AsGeoJSON(geom) AS geometry, "
            "created_by, created_at, CASE WHEN kind = 'area' THEN ST_Area(geom::geography) END "
            "AS area_m2, CASE WHEN kind = 'path' THEN ST_Length(geom::geography) END AS length_m, "
            "deleted_at FROM map_annotations WHERE id = :id",
            id=annotation_id,
        )
        return None if row is None else {**dict(row), "geometry": _json(row["geometry"])}

    async def delete_annotation(self, annotation_id: str, now: dt.datetime) -> bool:
        return (
            await self._exec(
                "UPDATE map_annotations SET deleted_at = :t WHERE id = :id AND deleted_at IS NULL",
                t=now,
                id=annotation_id,
            )
            == 1
        )

    # --- telemetry, files and the mission file -----------------------------------------

    async def track(
        self, mission_id: str, since: dt.datetime | None = None
    ) -> list[dict[str, Any]]:
        rows = await self._all(
            "SELECT t, ST_Y(pos) AS lat, ST_X(pos) AS lon, ST_Z(pos) AS alt_amsl_m, "
            "alt_rel_home_m, groundspeed_ms, battery_pct, flight_mode, armed FROM telemetry "
            "WHERE mission_id = :m AND (CAST(:since AS timestamptz) IS NULL OR t > :since) "
            "ORDER BY t",
            m=mission_id,
            since=since,
        )
        return [dict(r) for r in rows]

    async def add_file(
        self,
        *,
        mission_id: str,
        kind: str,
        rel_path: str,
        size: int,
        sha256: str,
        started_at: dt.datetime | None,
        ended_at: dt.datetime | None,
        notes: str | None,
    ) -> None:
        await self._exec(
            "INSERT INTO mission_files (id, mission_id, kind, rel_path, bytes, sha256, started_at, "
            "ended_at, notes) VALUES (:id, :m, :k, :p, :b, :h, :s, :e, :n) ON CONFLICT "
            "(mission_id, rel_path) DO UPDATE SET bytes = EXCLUDED.bytes, "
            "sha256 = EXCLUDED.sha256, "
            "ended_at = EXCLUDED.ended_at, notes = EXCLUDED.notes",
            id=str(new_id()),
            m=mission_id,
            k=kind,
            p=rel_path,
            b=size,
            h=sha256,
            s=started_at,
            e=ended_at,
            n=notes,
        )

    async def mission_export(self, mission_id: str) -> dict[str, Any]:
        """Everything ``mission.json`` holds: the mission, every charter version, the
        orders and the approvals."""
        mission = await self.mission(mission_id)
        return {
            "mission": None if mission is None else mission.model_dump(mode="json"),
            "charters": [c.model_dump(mode="json") for c in await self.charters(mission_id)],
            "commands": [c.model_dump(mode="json") for c in await self.commands(mission_id, None)],
            "approvals": [
                a.model_dump(mode="json") for a in await self.approvals(mission_id, None)
            ],
        }

    async def add_event(
        self,
        type_: str,
        *,
        now: dt.datetime,
        mission_id: str | None,
        vehicle_id: str | None,
        payload: dict[str, Any],
        severity: str = "info",
    ) -> None:
        async with self.engine.begin() as conn:
            await self._event(
                conn,
                type_,
                t=now,
                mission_id=mission_id,
                vehicle_id=vehicle_id,
                payload=payload,
                severity=severity,
            )
