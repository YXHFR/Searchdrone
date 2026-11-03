"""Migration 0007_inspections: the closer look's table is append-only, a start names the
pilot, the charter keeps its operation mode, and the decision "resume" exists."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncpg
import pytest

if TYPE_CHECKING:
    from conftest import Db  # fixtures live in conftest.py (not importable at runtime)

pytestmark = [pytest.mark.contract, pytest.mark.db]

APP = {"role": "sd_app"}


def _seed(db: Db) -> tuple[str, str]:
    """A vehicle, a mission and an inspect_detection order: (mission id, order id)."""
    db.run("INSERT INTO vehicles (id, name, kind, mav_sysid) VALUES ('sim-01', 'Sim 01', 'sim', 1)")
    mission = db.run(
        "INSERT INTO missions (vehicle_id, title, limits, created_by, data_dir) "
        "VALUES ('sim-01', 'Test', '{}', 'operator:demo', 'missions/x') RETURNING id::text"
    )[0]["id"]
    command = db.run(
        "INSERT INTO commands (id, mission_id, vehicle_id, issuer_kind, issuer_ref, skill, "
        "params, status) VALUES (uuidv7(), $1::uuid, 'sim-01', 'operator', 'operator:demo', "
        "'inspect_detection', '{}', 'dispatched') RETURNING id::text",
        mission,
    )[0]["id"]
    return mission, command


def _start(db: Db, mission: str, command: str, by: str = "operator:demo") -> None:
    db.run(
        "INSERT INTO inspections (command_id, candidate_id, mission_id, row_kind, started_at, "
        "started_by, operation_mode, overflight) VALUES ($1::uuid, uuidv7(), $2::uuid, "
        "'start', now(), $3, 'exercise', true)",
        command,
        mission,
        by,
        **APP,
    )


def _columns(db: Db, table: str) -> set[str]:
    rows = db.run("SELECT column_name FROM information_schema.columns WHERE table_name = $1", table)
    return {r["column_name"] for r in rows}


def test_up_and_down(fresh_db: Db) -> None:
    db = fresh_db
    db.upgrade()
    mission, command = _seed(db)
    _start(db, mission, command)
    assert {"operation_mode", "overflight_allowed"} <= _columns(db, "charters")
    db.downgrade("0006_candidate_views")
    assert not _columns(db, "inspections")
    assert "operation_mode" not in _columns(db, "charters")
    db.upgrade()
    assert "row_kind" in _columns(db, "inspections")


@pytest.mark.parametrize("role", ["sd_app", "sd_owner"])
def test_a_look_is_never_rewritten(migrated_db: Db, role: str) -> None:
    db = migrated_db
    mission, command = _seed(db)
    _start(db, mission, command)
    for sql in (
        "UPDATE inspections SET overflight = false",
        "DELETE FROM inspections",
        "TRUNCATE inspections",
    ):
        with pytest.raises(asyncpg.PostgresError) as err:
            db.run(sql, role=role)
        if role == "sd_owner":
            assert "append-only" in str(err.value)
    assert db.run("SELECT count(*) AS n FROM inspections")[0]["n"] == 1


def test_one_start_and_one_end_per_order(migrated_db: Db) -> None:
    db = migrated_db
    mission, command = _seed(db)
    _start(db, mission, command)
    with pytest.raises(asyncpg.UniqueViolationError):
        _start(db, mission, command)
    end = (
        "INSERT INTO inspections (command_id, candidate_id, mission_id, row_kind, ended_at, "
        "outcome) VALUES ($1::uuid, uuidv7(), $2::uuid, 'end', now(), $3)"
    )
    db.run(end, command, mission, "timeout", **APP)
    with pytest.raises(asyncpg.UniqueViolationError):
        db.run(end, command, mission, "confirmed", **APP)


def test_a_start_names_the_pilot_and_an_end_its_outcome(migrated_db: Db) -> None:
    db = migrated_db
    mission, command = _seed(db)
    for who in ("brain:ground", "rule:ground", "operator:"):
        with pytest.raises(asyncpg.CheckViolationError):
            _start(db, mission, command, by=who)
    for outcome in (None, "maybe"):
        with pytest.raises(asyncpg.CheckViolationError):
            db.run(
                "INSERT INTO inspections (command_id, candidate_id, mission_id, row_kind, "
                "ended_at, outcome) VALUES ($1::uuid, uuidv7(), $2::uuid, 'end', now(), $3)",
                command,
                mission,
                outcome,
                **APP,
            )


def test_overflight_needs_an_operation_mode_and_resume_is_a_decision(migrated_db: Db) -> None:
    db = migrated_db
    checks = db.run(
        "SELECT conname, pg_get_constraintdef(oid) AS def FROM pg_constraint "
        "WHERE conname IN ('charters_overflight_needs_mode', 'candidate_decisions_decision_check')"
    )
    found = {r["conname"]: r["def"] for r in checks}
    assert "overflight_allowed" in found["charters_overflight_needs_mode"]
    assert "'resume'" in found["candidate_decisions_decision_check"]
