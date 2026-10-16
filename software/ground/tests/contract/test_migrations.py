"""Migration 0001_initial: round trip, grants and the append-only guards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncpg
import pytest

if TYPE_CHECKING:
    from conftest import Db  # fixtures live in conftest.py (not importable at runtime)

pytestmark = [pytest.mark.contract, pytest.mark.db]

TABLES = {
    "vehicles",
    "missions",
    "charters",
    "mission_sectors",
    "commands",
    "approvals",
    "gate_decisions",
    "clearances",
    "operator_latches",
    "telemetry",
    "chat_messages",
    "map_annotations",
    "events",
    "mission_files",
    "auth_sessions",
    "brain_memory",
    "brain_decisions",
}
APPEND_ONLY = ("events", "gate_decisions", "chat_messages", "brain_decisions")

# Public CMAC test field (ArduPilot's default SITL home), never a private place.
LON, LAT = 149.165230, -35.363261


def _tables(db: Db) -> set[str]:
    rows = db.run(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
    )
    return {r["table_name"] for r in rows} - {"spatial_ref_sys", "alembic_version"}


def _has_postgis(db: Db) -> bool:
    return bool(db.run("SELECT 1 FROM pg_extension WHERE extname = 'postgis'"))


def test_roundtrip(fresh_db: Db) -> None:
    db = fresh_db
    db.upgrade()
    db.downgrade()
    assert _tables(db) == set()
    assert _has_postgis(db), "downgrade must keep PostGIS"
    db.upgrade()

    assert _tables(db) == TABLES
    assert not {"detections", "candidates"} & _tables(db)
    assert "3.6" in db.run("SELECT postgis_full_version() AS v")[0]["v"]
    head = db.run("SELECT version_num FROM alembic_version", role="sd_app")
    assert [r["version_num"] for r in head] == ["0001_initial"]


def test_app_role_has_no_delete_anywhere(migrated_db: Db) -> None:
    rows = migrated_db.run(
        "SELECT table_name, privilege_type FROM information_schema.role_table_grants "
        "WHERE grantee = 'sd_app' "
        "AND privilege_type IN ('DELETE', 'TRUNCATE', 'REFERENCES', 'TRIGGER')"
    )
    assert rows == []
    owners = migrated_db.run(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tableowner = 'sd_app'"
    )
    assert owners == []


def _seed(db: Db) -> str:
    """One vehicle, mission, charter and one row in every append-only table."""
    db.run("INSERT INTO vehicles (id, name, kind, mav_sysid) VALUES ('sim-01', 'Sim 01', 'sim', 1)")
    mission = db.run(
        "INSERT INTO missions (vehicle_id, title, limits, created_by, data_dir) "
        "VALUES ('sim-01', 'Test', '{}', 'operator:demo', 'missions/x') RETURNING id::text"
    )[0]["id"]
    db.run(
        "INSERT INTO charters (mission_id, vehicle_id, version, status, body, area, home, "
        "pilot_position, proposed_by) VALUES ($1::uuid, 'sim-01', 1, 'proposed', '{}', "
        "ST_Expand(ST_SetSRID(ST_MakePoint($2, $3), 4326), 0.001), "
        "ST_SetSRID(ST_MakePoint($2, $3, 584), 4326), "
        "ST_SetSRID(ST_MakePoint($2, $3), 4326), 'brain:ground')",
        mission,
        LON,
        LAT,
        role="sd_app",
    )
    db.run(
        "INSERT INTO events (mission_id, vehicle_id, source, type) "
        "VALUES ($1::uuid, 'sim-01', 'ground', 'mission.created')",
        mission,
        role="sd_app",
    )
    db.run(
        "INSERT INTO gate_decisions (mission_id, vehicle_id, side, phase, decision, message_sv) "
        "VALUES ($1::uuid, 'sim-01', 'ground', 'submit', 'REJECT', 'nej')",
        mission,
        role="sd_app",
    )
    db.run(
        "INSERT INTO chat_messages (id, mission_id, vehicle_id, role, author_ref, text) "
        "VALUES (uuidv7(), $1::uuid, 'sim-01', 'operator', 'operator:demo', 'hej')",
        mission,
        role="sd_app",
    )
    db.run(
        "INSERT INTO brain_decisions (id, mission_id, vehicle_id, turn_id, body) "
        "VALUES (uuidv7(), $1::uuid, 'sim-01', uuidv7(), '{}')",
        mission,
        role="sd_app",
    )
    return mission


@pytest.mark.parametrize("table", APPEND_ONLY)
@pytest.mark.parametrize("role", ["sd_app", "sd_owner"])
def test_append_only(migrated_db: Db, table: str, role: str) -> None:
    db = migrated_db
    _seed(db)
    # sd_app is stopped by missing grants, sd_owner (superuser) by the triggers.
    # `table` comes from the fixed APPEND_ONLY tuple, never from input.
    for sql in (
        f"UPDATE {table} SET id = id",  # noqa: S608
        f"DELETE FROM {table}",  # noqa: S608
        f"TRUNCATE {table} CASCADE",
    ):
        with pytest.raises(asyncpg.PostgresError):
            db.run(sql, role=role)
    assert db.run(f"SELECT count(*) AS n FROM {table}")[0]["n"] == 1  # noqa: S608


def test_app_role_cannot_bypass_guards(migrated_db: Db) -> None:
    db = migrated_db
    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        db.run("SET session_replication_role = replica", role="sd_app")
    with pytest.raises(asyncpg.InsufficientPrivilegeError):
        db.run("ALTER TABLE events DISABLE TRIGGER ALL", role="sd_app")


def test_charter_body_frozen_status_free(migrated_db: Db) -> None:
    db = migrated_db
    _seed(db)
    with pytest.raises(asyncpg.RaiseError, match="immutable"):
        db.run("UPDATE charters SET body = '{\"x\": 1}'", role="sd_app")
    with pytest.raises(asyncpg.RaiseError, match="immutable"):
        db.run("UPDATE charters SET version = 2", role="sd_app")
    db.run("UPDATE charters SET status = 'rejected'", role="sd_app")
    assert db.run("SELECT status FROM charters")[0]["status"] == "rejected"
