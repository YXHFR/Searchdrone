"""The events table: a write the database took but whose call still failed is not a
reason for every later flush to fail (``Repo.add_events``, ``Ground.flush_events``)."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_ground_server import _ground, _q  # pyright: ignore[reportMissingImports]

from sd_ground.db.repo import EventRow, Repo, new_event
from sd_testing import FakeClock

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]


async def test_a_row_written_twice_is_kept_once(migrated_db: Db, tmp_path: Path) -> None:
    g = await _ground(migrated_db, tmp_path)
    assert g.repo is not None
    t = FakeClock().utc_now()
    rows = [new_event("a", t=t), new_event("b", t=t)]
    await g.repo.add_events(rows)
    await g.repo.add_events([*rows, new_event("c", t=t)])  # a and b again: no error
    found = await _q(migrated_db, "SELECT type FROM events ORDER BY type")
    assert [r["type"] for r in found] == ["a", "b", "c"]


class _LostReply:
    """The database takes the first write, but its reply is lost: the call fails."""

    def __init__(self, repo: Repo) -> None:
        self.repo = repo
        self.calls = 0

    async def add_events(self, rows: Sequence[EventRow]) -> None:
        self.calls += 1
        await self.repo.add_events(rows)
        if self.calls == 1:
            raise ConnectionResetError("reply lost")


async def test_a_write_whose_reply_was_lost_does_not_stop_later_flushes(
    migrated_db: Db, tmp_path: Path
) -> None:
    g = await _ground(migrated_db, tmp_path)
    assert g.repo is not None
    lost = _LostReply(g.repo)
    ground: Any = g
    ground.repo = lost
    g.add_event("a")
    await g.flush_events()  # written, but the call failed: "a" stays queued
    assert [r["type"] for r in g.queued_events()] == ["a"]
    g.add_event("b")
    await g.flush_events()
    assert g.queued_events() == []
    found = await _q(migrated_db, "SELECT type FROM events ORDER BY type")
    assert [r["type"] for r in found] == ["a", "b"]
