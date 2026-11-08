"""Migration 0009_reject_reason: a rejection may say "not a person"; no other decision has
a reason, and every older row has none (it keeps the cylinder)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import Db  # fixtures live in conftest.py (not importable at runtime)

pytestmark = [pytest.mark.contract, pytest.mark.db]


def _columns(db: Db) -> set[str]:
    rows = db.run(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'candidate_decisions'"
    )
    return {r["column_name"] for r in rows}


def test_up_and_down(fresh_db: Db) -> None:
    db = fresh_db
    db.upgrade()
    assert "reason" in _columns(db)
    db.downgrade("0008_floor_pack_choice")
    assert "reason" not in _columns(db)
    db.upgrade()
    assert "reason" in _columns(db)


def test_only_a_rejection_says_not_a_person(migrated_db: Db) -> None:
    checks = migrated_db.run(
        "SELECT pg_get_constraintdef(oid) AS def FROM pg_constraint "
        "WHERE conrelid = 'candidate_decisions'::regclass AND contype = 'c'"
    )
    defs = " ".join(r["def"] for r in checks)
    assert "not_a_person" in defs and "reason IS NULL" in defs
    assert "'reject'" in defs.split("not_a_person", 1)[1]
