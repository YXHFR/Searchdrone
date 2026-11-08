"""Migration 0008_floor_pack_choice: the charter keeps the pilot's map pack, both the name
and the hash or neither, in the form ``MissionCharter`` checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import Db  # fixtures live in conftest.py (not importable at runtime)

pytestmark = [pytest.mark.contract, pytest.mark.db]

COLUMNS = {"floor_pack_id", "floor_pack_sha256", "floor_pack_set"}


def _columns(db: Db) -> set[str]:
    rows = db.run(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'charters'"
    )
    return {r["column_name"] for r in rows}


def test_up_and_down(fresh_db: Db) -> None:
    db = fresh_db
    db.upgrade()
    assert _columns(db) >= COLUMNS
    db.downgrade("0007_inspections")
    assert not COLUMNS & _columns(db)
    db.upgrade()
    assert _columns(db) >= COLUMNS


def test_the_name_and_the_hash_go_together(migrated_db: Db) -> None:
    checks = migrated_db.run(
        "SELECT conname, pg_get_constraintdef(oid) AS def FROM pg_constraint "
        "WHERE conrelid = 'charters'::regclass"
    )
    found = {r["conname"]: r["def"] for r in checks}
    assert "floor_pack_sha256 IS NULL" in found["charters_floor_pack_both"]
    assert "floor_pack_set" in found["charters_floor_pack_set"]
    defs = " ".join(found.values())
    assert "[a-z0-9][a-z0-9_-]{0,63}" in defs and "[0-9a-f]{64}" in defs
