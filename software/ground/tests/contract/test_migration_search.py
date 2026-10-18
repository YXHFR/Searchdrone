"""Migration 0003_search (B2-01): the stage 2 tables are append-only and go up and down."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import asyncpg
import pytest

if TYPE_CHECKING:
    from conftest import Db  # fixtures live in conftest.py (not importable at runtime)

pytestmark = [pytest.mark.contract, pytest.mark.db]

SEARCH_TABLES = (
    "search_profiles",
    "perception_stills",
    "detections",
    "candidates",
    "candidate_observations",
    "candidate_assessments",
    "candidate_decisions",
    "probability_maps",
    "cloud_usage",
)

# Public CMAC test field (ArduPilot's default SITL home), never a private place.
POINT = "POINT(149.16523 -35.363261)"
SQUARE = (
    "POLYGON((149.1647 -35.3637, 149.1657 -35.3637, 149.1657 -35.3628, "
    "149.1647 -35.3628, 149.1647 -35.3637))"
)


def _tables(db: Db) -> set[str]:
    rows = db.run(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
    )
    return {r["table_name"] for r in rows}


def _columns(db: Db, table: str) -> set[str]:
    rows = db.run("SELECT column_name FROM information_schema.columns WHERE table_name = $1", table)
    return {r["column_name"] for r in rows}


def _seed(db: Db) -> str:
    """A vehicle, a mission and one row in every stage 2 table, written as sd_app."""
    db.run("INSERT INTO vehicles (id, name, kind, mav_sysid) VALUES ('sim-01', 'Sim 01', 'sim', 1)")
    mission = db.run(
        "INSERT INTO missions (vehicle_id, title, limits, created_by, data_dir) "
        "VALUES ('sim-01', 'Test', '{}', 'operator:demo', 'missions/x') RETURNING id::text"
    )[0]["id"]
    app = {"role": "sd_app"}
    db.run(
        "INSERT INTO search_profiles (mission_id, profile_id, profile_version, body, proposed_by) "
        "VALUES ($1::uuid, uuidv7(), 1, '{}', 'brain:ground')",
        mission,
        **app,
    )
    still = db.run(
        "INSERT INTO perception_stills (mission_id, vehicle_id, still_id, captured_utc, sensor, "
        "pose, footprint, px_per_m, detector) VALUES ($1::uuid, 'sim-01', uuidv7(), now(), "
        "'rgb', '{}', ST_GeomFromText($2, 4326), 50, '{}') RETURNING still_id::text",
        mission,
        SQUARE,
        **app,
    )[0]["still_id"]
    detection = db.run(
        "INSERT INTO detections (mission_id, vehicle_id, detection_id, still_id, score, bbox, "
        "geom, sigma95_m, px_size) VALUES ($1::uuid, 'sim-01', uuidv7(), $2::uuid, 0.6, "
        "'{}', ST_GeomFromText($3, 4326), 4.5, 90) RETURNING detection_id::text",
        mission,
        still,
        POINT,
        **app,
    )[0]["detection_id"]
    candidate = db.run(
        "INSERT INTO candidates (candidate_id, rev, mission_id, seq, state, geom, sigma95_m, "
        "n_seen, n_in_view, best_detector_score, priority, first_seen_utc, last_seen_utc) "
        "VALUES (uuidv7(), 1, $1::uuid, 1, 'new', ST_GeomFromText($2, 4326), 4.5, 1, 1, 0.6, "
        "0.4, now(), now()) RETURNING candidate_id::text",
        mission,
        POINT,
        **app,
    )[0]["candidate_id"]
    db.run(
        "INSERT INTO candidate_observations (candidate_id, detection_id, distance_m) "
        "VALUES ($1::uuid, $2::uuid, 0)",
        candidate,
        detection,
        **app,
    )
    db.run(
        "INSERT INTO candidate_assessments (candidate_id, profile_version, engine, model_id, "
        "body, latency_ms) VALUES ($1::uuid, 1, 'code', 'colors_v1', '{}', 3)",
        candidate,
        **app,
    )
    db.run(
        "INSERT INTO candidate_decisions (candidate_id, decision, subject_rev, operator_ref, "
        "session_id) VALUES ($1::uuid, 'unsure', 1, 'operator:demo', 's1')",
        candidate,
        **app,
    )
    db.run(
        "INSERT INTO probability_maps (mission_id, version, reason, grid, p, pod_cum, row, "
        "event_seq) VALUES ($1::uuid, 1, 'prior', $2::jsonb, '\\x00', '\\x00', 0.1, 0)",
        mission,
        json.dumps({"cell_m": 25}),
        **app,
    )
    db.run(
        "INSERT INTO cloud_usage (reservation_id, mission_id, process, purpose, model_id, "
        "est_sek, status) VALUES (uuidv7(), $1::uuid, 'search', 'match', 'm', 0.1, 'reserved')",
        mission,
        **app,
    )
    return mission


def test_upgrade_and_downgrade(fresh_db: Db) -> None:
    db = fresh_db
    db.upgrade("0003_path_annotations")
    assert not set(SEARCH_TABLES) & _tables(db)
    db.upgrade()
    assert set(SEARCH_TABLES) <= _tables(db)
    assert "found_candidate_id" in _columns(db, "missions")
    assert "poa_hint" in _columns(db, "map_annotations")
    _seed(db)
    db.downgrade("0003_path_annotations")
    assert not set(SEARCH_TABLES) & _tables(db)
    assert "found_candidate_id" not in _columns(db, "missions")
    assert "poa_hint" not in _columns(db, "map_annotations")
    assert db.run("SELECT count(*) AS n FROM missions")[0]["n"] == 1  # base data kept
    db.upgrade()
    assert set(SEARCH_TABLES) <= _tables(db)


@pytest.mark.parametrize("table", SEARCH_TABLES)
@pytest.mark.parametrize("role", ["sd_app", "sd_owner"])
def test_append_only_search_tables(migrated_db: Db, table: str, role: str) -> None:
    db = migrated_db
    _seed(db)
    # sd_app is stopped by missing grants, sd_owner (superuser) by sd_forbid_change().
    # `table` comes from the fixed SEARCH_TABLES tuple, never from input.
    for sql in (
        f"UPDATE {table} SET id = id",  # noqa: S608
        f"DELETE FROM {table}",  # noqa: S608
        f"TRUNCATE {table} CASCADE",
    ):
        with pytest.raises(asyncpg.PostgresError) as err:
            db.run(sql, role=role)
        if role == "sd_owner":
            assert "append-only" in str(err.value)
    assert db.run(f"SELECT count(*) AS n FROM {table}")[0]["n"] == 1  # noqa: S608


def test_sd_app_may_read_and_add_only(migrated_db: Db) -> None:
    rows = migrated_db.run(
        "SELECT table_name, privilege_type FROM information_schema.role_table_grants "
        "WHERE grantee = 'sd_app' AND table_name = ANY($1::text[])",
        list(SEARCH_TABLES),
    )
    grants: dict[str, set[str]] = {}
    for r in rows:
        grants.setdefault(r["table_name"], set()).add(r["privilege_type"])
    assert grants == {t: {"SELECT", "INSERT"} for t in SEARCH_TABLES}


def test_only_a_named_operator_decides_or_approves(migrated_db: Db) -> None:
    db = migrated_db
    mission = _seed(db)
    for who in ("brain:ground", "search", "system:ground", "operator:"):
        with pytest.raises(asyncpg.CheckViolationError):
            db.run(
                "INSERT INTO candidate_decisions (candidate_id, decision, subject_rev, "
                "operator_ref, session_id) VALUES (uuidv7(), 'confirm', 1, $1, 's1')",
                who,
                role="sd_app",
            )
        with pytest.raises(asyncpg.CheckViolationError):
            db.run(
                "INSERT INTO search_profiles (mission_id, profile_id, profile_version, body, "
                "proposed_by, approved_by, approved_at) "
                "VALUES ($1::uuid, uuidv7(), 2, '{}', 'brain:ground', $2, now())",
                mission,
                who,
                role="sd_app",
            )
    with pytest.raises(asyncpg.CheckViolationError):  # approved without a time
        db.run(
            "INSERT INTO search_profiles (mission_id, profile_id, profile_version, body, "
            "proposed_by, approved_by) VALUES ($1::uuid, uuidv7(), 2, '{}', 'x', 'operator:demo')",
            mission,
            role="sd_app",
        )


def test_one_approved_profile_per_version(migrated_db: Db) -> None:
    db = migrated_db
    mission = _seed(db)
    approve = (
        "INSERT INTO search_profiles (mission_id, profile_id, profile_version, body, "
        "proposed_by, approved_by, approved_at) "
        "VALUES ($1::uuid, uuidv7(), 1, '{}', 'brain:ground', 'operator:demo', now())"
    )
    db.run(approve, mission, role="sd_app")
    with pytest.raises(asyncpg.UniqueViolationError):
        db.run(approve, mission, role="sd_app")


def test_candidate_rows_check_their_values(migrated_db: Db) -> None:
    db = migrated_db
    mission = _seed(db)
    insert = (
        "INSERT INTO candidates (candidate_id, rev, mission_id, seq, state, geom, sigma95_m, "
        "n_seen, n_in_view, best_detector_score, priority, merged_into, first_seen_utc, "
        "last_seen_utc) VALUES (uuidv7(), 1, $1::uuid, 2, $2, ST_GeomFromText($5, 4326), 4.5, "
        "1, 1, 0.6, $3, $4, now(), now())"
    )
    for state, priority, merged in (
        ("found", 0.5, None),  # unknown state
        ("ready", 1.5, None),  # priority above 1
        ("merged", 0.5, None),  # merged needs merged_into
        ("ready", 0.5, "0192f49e-6000-7000-8000-000000000006"),  # only merged has it
    ):
        with pytest.raises(asyncpg.CheckViolationError):
            db.run(insert, mission, state, priority, merged, POINT, role="sd_app")


def test_detection_needs_its_still_and_a_safe_crop_path(migrated_db: Db) -> None:
    db = migrated_db
    mission = _seed(db)
    still = db.run("SELECT still_id::text AS s FROM perception_stills")[0]["s"]
    insert = (
        "INSERT INTO detections (mission_id, vehicle_id, detection_id, still_id, score, bbox, "
        "geom, sigma95_m, px_size, crop_path) VALUES ($1::uuid, 'sim-01', uuidv7(), $2::uuid, "
        "0.6, '{}', ST_GeomFromText($4, 4326), 4.5, 90, $3)"
    )
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        db.run(insert, mission, "0192f49e-4000-7000-8000-000000000099", None, POINT, role="sd_app")
    for path in ("/etc/passwd", "crops/../../x.jpg"):
        with pytest.raises(asyncpg.CheckViolationError):
            db.run(insert, mission, still, path, POINT, role="sd_app")
    db.run(insert, mission, still, "crops/d1.jpg", POINT, role="sd_app")


def test_poa_hint_only_on_areas(migrated_db: Db) -> None:
    db = migrated_db
    mission = _seed(db)
    insert = (
        "INSERT INTO map_annotations (mission_id, vehicle_id, kind, label, geom, created_by, "
        "poa_hint) VALUES ($1::uuid, 'sim-01', $2, $3, "
        "ST_GeomFromText($5, 4326), 'operator:demo', $4)"
    )
    db.run(insert, mission, "area", "Y1", "high", SQUARE, role="sd_app")
    db.run(insert, mission, "pin", "P1", None, POINT, role="sd_app")
    with pytest.raises(asyncpg.CheckViolationError):
        db.run(insert, mission, "pin", "P2", "low", POINT, role="sd_app")
    with pytest.raises(asyncpg.CheckViolationError):
        db.run(insert, mission, "area", "Y2", "maybe", SQUARE, role="sd_app")


def test_found_candidate_is_set_on_the_mission(migrated_db: Db) -> None:
    db = migrated_db
    mission = _seed(db)
    candidate = db.run("SELECT candidate_id::text AS c FROM candidates")[0]["c"]
    db.run(
        "UPDATE missions SET found_candidate_id = $1::uuid WHERE id = $2::uuid",
        candidate,
        mission,
        role="sd_app",
    )
    row = db.run("SELECT found_candidate_id::text AS c, status FROM missions")[0]
    assert row["c"] == candidate and row["status"] == "DRAFT"


def test_no_local_engine_and_no_thermal_sensor(migrated_db: Db) -> None:
    db = migrated_db
    mission = _seed(db)
    with pytest.raises(asyncpg.CheckViolationError):
        db.run(
            "INSERT INTO candidate_assessments (candidate_id, profile_version, engine, model_id, "
            "body, latency_ms) VALUES (uuidv7(), 1, 'local', 'm', '{}', 3)",
            role="sd_app",
        )
    with pytest.raises(asyncpg.CheckViolationError):
        db.run(
            "INSERT INTO perception_stills (mission_id, vehicle_id, still_id, captured_utc, "
            "sensor, pose, footprint, px_per_m, detector) VALUES ($1::uuid, 'sim-01', uuidv7(), "
            "now(), 'thermal', '{}', ST_GeomFromText($2, 4326), 50, '{}')",
            mission,
            SQUARE,
            role="sd_app",
        )
