"""The ground's event queue: one flush at a time (``Ground.flush_events``)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from sd_ground.context import Ground
from sd_ground.db.repo import EventRow


class SlowRepo:
    """Writes rows like the database: an id another write holds (uncommitted) makes this
    write wait for that one, then fail as a whole on the duplicate key. The first write
    commits only once ``go`` is set, so a second flush can start meanwhile."""

    def __init__(self) -> None:
        self.rows: dict[str, EventRow] = {}
        self.go = asyncio.Event()
        self.committed = asyncio.Event()
        self.held: set[str] = set()
        self.calls = 0

    async def add_events(self, rows: Sequence[EventRow]) -> None:
        self.calls += 1
        ids = [str(r["id"]) for r in rows]
        if set(ids) & self.held:
            await self.committed.wait()
        if set(ids) & set(self.rows):
            raise RuntimeError("IntegrityError: duplicate key")
        first = self.calls == 1
        self.held |= set(ids)
        if first:
            await self.go.wait()
        self.rows.update(zip(ids, rows, strict=True))
        self.held -= set(ids)
        if first:
            self.committed.set()


async def test_a_flush_during_another_writes_the_events_queued_since(ground: Ground) -> None:
    repo = SlowRepo()
    g: Any = ground
    g.repo = repo
    ground.add_event("first")
    first = asyncio.create_task(ground.flush_events())
    await asyncio.sleep(0)  # the first flush is writing
    ground.add_event("second")  # e.g. a queued onboard event arriving meanwhile
    second = asyncio.create_task(ground.flush_events())
    await asyncio.sleep(0)
    repo.go.set()
    await second
    # The reader that flushed second finds both, and nothing is written twice.
    assert sorted(r["type"] for r in repo.rows.values()) == ["first", "second"]
    await first
    assert ground.queued_events() == []
    assert repo.calls == 2
