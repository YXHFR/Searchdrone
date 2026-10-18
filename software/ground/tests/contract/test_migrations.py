"""Migration 0001_initial: round trip, grants and the append-only guards."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import asyncpg
import pytest

if TYPE_CHECKING:
    from conftest import Db  # fixtures live in conftest.py (not importable at runtime)

pytestmark = [pytest.mark.contract, pytest.mark.db]
# Briefly changing the shared sd_app role runs in CI, or locally only on request.
INVASIVE_OK = os.environ.get("CI") == "true" or os.environ.get("SD_TEST_DB_INVASIVE") == "1"

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
    "detections",
}
APPEND_ONLY = ("events", "gate_decisions", "chat_messages", "brain_decisions", "detections")

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
    assert "candidates" not in _tables(db)
    assert "3.6" in db.run("SELECT postgis_full_version() AS v")[0]["v"]
    head = db.run("SELECT version_num FROM alembic_version", role="sd_app")
    assert [r["version_num"] for r in head] == ["0003_detections"]


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
    db.run(
        "INSERT INTO detections (id, mission_id, vehicle_id, find_id, still_id, captured_at, "
        "label, kind, source, score, bbox, lat, lon, sigma95_m, size_m, synthetic) "
        "VALUES (uuidv7(), $1::uuid, 'sim-01', uuidv7(), uuidv7(), now(), 'red_block', "
        "'target', 'color', 0.7, '[1, 2, 3, 4]', $2, $3, 4.0, 0.6, true)",
        mission,
        LAT,
        LON,
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


def test_charter_proposal_and_approval_frozen(migrated_db: Db) -> None:
    db = migrated_db
    _seed(db)
    db.run("INSERT INTO vehicles (id, name, kind, mav_sysid) VALUES ('sim-02', 'Sim 02', 'sim', 2)")
    for column, value in (
        ("area", "ST_Expand(area, 0.01)"),
        ("home", f"ST_SetSRID(ST_MakePoint({LON}, {LAT}, 600), 4326)"),
        ("pilot_position", f"ST_SetSRID(ST_MakePoint({LON + 0.001}, {LAT}), 4326)"),
        ("vehicle_id", "'sim-02'"),
        ("proposed_by", "'brain:other'"),
    ):
        with pytest.raises(asyncpg.RaiseError, match="immutable"):
            db.run(f"UPDATE charters SET {column} = {value}", role="sd_app")  # noqa: S608
    db.run(
        "UPDATE charters SET status = 'approved', approved_by = 'operator:demo', "
        "approved_at = now(), autonomy_mode = 'WITHIN_CHARTER'",
        role="sd_app",
    )
    with pytest.raises(asyncpg.RaiseError, match="approval is immutable"):
        db.run("UPDATE charters SET approved_by = 'operator:other'", role="sd_app")
    with pytest.raises(asyncpg.RaiseError, match="back to proposed"):
        db.run("UPDATE charters SET status = 'proposed'", role="sd_app")
    db.run("UPDATE charters SET status = 'superseded'", role="sd_app")
    with pytest.raises(asyncpg.RaiseError, match="final"):
        db.run("UPDATE charters SET status = 'approved'", role="sd_app")


def test_only_named_operators_approve(migrated_db: Db) -> None:
    db = migrated_db
    mission = _seed(db)
    insert = (
        "INSERT INTO approvals (mission_id, vehicle_id, kind, subject_id, status, summary_sv, "
        "requested_by, decided_by, decided_at) "
        "VALUES ($1::uuid, 'sim-01', 'resume', uuidv7(), $2, 'Fortsätt', 'system:ground', $3, $4)"
    )
    now = datetime.now(UTC)
    for decided_by, decided_at in ((None, None), ("operator:demo", None), (None, now)):
        with pytest.raises(asyncpg.CheckViolationError):
            db.run(insert, mission, "approved", decided_by, decided_at, role="sd_app")
    for who in ("brain:ground", "operator:", "system:ground"):
        with pytest.raises(asyncpg.CheckViolationError):
            db.run(insert, mission, "approved", who, now, role="sd_app")
    db.run(insert, mission, "pending", None, None, role="sd_app")
    with pytest.raises(asyncpg.CheckViolationError):
        db.run("UPDATE approvals SET status = 'approved'", role="sd_app")
    db.run(
        "UPDATE approvals SET status = 'approved', decided_by = 'operator:demo', "
        "decided_at = now()",
        role="sd_app",
    )
    with pytest.raises(asyncpg.CheckViolationError):
        db.run("UPDATE charters SET approved_by = 'brain:ground'", role="sd_app")


def test_stop_intent_and_unique_rules(migrated_db: Db) -> None:
    db = migrated_db
    mission = _seed(db)
    chat = (
        "INSERT INTO chat_messages (id, mission_id, vehicle_id, role, author_ref, text, "
        "parsed_intent, handled_by) VALUES (uuidv7(), $1::uuid, 'sim-01', 'operator', "
        "'operator:demo', 'x', $2, $3)"
    )
    for intent, handled in (("resume", None), (None, "ground")):
        with pytest.raises(asyncpg.CheckViolationError):
            db.run(chat, mission, intent, handled, role="sd_app")
    db.run(chat, mission, "stop", "ground", role="sd_app")
    with pytest.raises(asyncpg.UniqueViolationError):
        db.run(
            "INSERT INTO missions (vehicle_id, title, limits, created_by, data_dir) "
            "VALUES ('sim-01', 'Two', '{}', 'operator:demo', 'missions/y')",
            role="sd_app",
        )
    with pytest.raises(asyncpg.CheckViolationError):
        db.run("UPDATE missions SET status = 'PAUSED'", role="sd_app")


def test_failed_upgrade_rolls_back(fresh_db: Db) -> None:
    # A clashing function makes the second script fail after the tables were created.
    fresh_db.run(
        "CREATE FUNCTION sd_forbid_change() RETURNS trigger LANGUAGE plpgsql "
        "AS $$ BEGIN RETURN NULL; END $$"
    )
    with pytest.raises(Exception, match="already exists"):
        fresh_db.upgrade()
    assert _tables(fresh_db) == set()
    assert not fresh_db.run("SELECT 1 FROM pg_tables WHERE tablename = 'alembic_version'")


def test_app_role_is_unprivileged(fresh_db: Db) -> None:
    rows = fresh_db.run(
        "SELECT rolsuper, rolcreaterole, rolcreatedb, rolbypassrls, rolreplication "
        "FROM pg_roles WHERE rolname = 'sd_app'"
    )
    assert rows and not any(rows[0].values())


@pytest.mark.skipif(not INVASIVE_OK, reason="set SD_TEST_DB_INVASIVE=1 (changes sd_app)")
def test_migration_refuses_privileged_app_role(fresh_db: Db) -> None:
    # sd_app is cluster-wide: every change is undone in `finally`. If this test is killed
    # midway, `ALTER ROLE sd_app NOCREATEDB` restores it (the migration refuses until then).
    group = f"sd_test_grp_{os.getpid()}"
    fresh_db.run(f"CREATE ROLE {group} NOLOGIN")
    try:
        fresh_db.run(f"GRANT {group} TO sd_app")
        with pytest.raises(Exception, match="must not be a member"):
            fresh_db.upgrade()
    finally:
        fresh_db.run(f"DROP ROLE {group}")
    try:
        fresh_db.run("ALTER ROLE sd_app CREATEDB")
        with pytest.raises(Exception, match="must be NOSUPERUSER"):
            fresh_db.upgrade()
    finally:
        fresh_db.run("ALTER ROLE sd_app NOCREATEDB")
    fresh_db.upgrade()
    assert _tables(fresh_db) == TABLES


def test_one_approved_charter_and_one_active_clearance(migrated_db: Db) -> None:
    db = migrated_db
    mission = _seed(db)
    charter = db.run("SELECT id::text FROM charters")[0]["id"]
    db.run(
        "INSERT INTO charters (mission_id, vehicle_id, version, status, body, area, home, "
        "pilot_position, proposed_by) SELECT mission_id, vehicle_id, 2, 'proposed', body, "
        "area, home, pilot_position, proposed_by FROM charters",
        role="sd_app",
    )
    approve = (
        "UPDATE charters SET status = 'approved', approved_by = 'operator:demo', "
        "approved_at = now(), autonomy_mode = 'APPROVE_EACH' WHERE version = $1"
    )
    db.run(approve, 1, role="sd_app")
    with pytest.raises(asyncpg.UniqueViolationError):
        db.run(approve, 2, role="sd_app")
    clearance = (
        "INSERT INTO clearances (mission_id, vehicle_id, charter_id, charter_version, status, "
        "confirmed_by, valid_until) VALUES ($1::uuid, 'sim-01', $2::uuid, 1, 'active', "
        "'operator:demo', now() + interval '120 seconds')"
    )
    db.run(clearance, mission, charter, role="sd_app")
    with pytest.raises(asyncpg.UniqueViolationError):
        db.run(clearance, mission, charter, role="sd_app")


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("label", "'Red Block'"),
        ("source", "'guess'"),
        ("kind", "'other'"),
        ("score", "1.5"),
        ("crop_path", "'../x.jpg'"),
        ("lat", "91"),
    ],
)
def test_detection_rows_are_checked(migrated_db: Db, column: str, value: str) -> None:
    db = migrated_db
    mission = _seed(db)
    row = {
        "label": "'car'",
        "kind": "'context'",
        "source": "'oracle'",
        "score": "1",
        "lat": "0",
        "crop_path": "NULL",
    } | {column: value}
    # Column names and values come from the fixed lists above, never from input.
    sql = (
        "INSERT INTO detections (id, mission_id, vehicle_id, find_id, still_id, captured_at, "  # noqa: S608
        f"bbox, lon, sigma95_m, size_m, synthetic, {', '.join(row)}) VALUES (uuidv7(), "
        "$1::uuid, 'sim-01', uuidv7(), uuidv7(), now(), '[]', 0, 0, 0, true, "
        f"{', '.join(row.values())})"
    )
    with pytest.raises(asyncpg.CheckViolationError):
        db.run(sql, mission, role="sd_app")
