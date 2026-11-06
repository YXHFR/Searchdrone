"""The ground's event queue: one flush at a time (``Ground.flush_events``)."""

from __future__ import annotations

import asyncio
from collections import deque
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


class GatedRepo:
    """Writes every row it is given. The first write waits for ``go``, and fails if
    ``fail`` is set; later writes go straight through."""

    def __init__(self, *, fail: bool = False) -> None:
        self.rows: dict[str, EventRow] = {}
        self.go = asyncio.Event()
        self.fail = fail
        self.writing = asyncio.Event()
        self.calls = 0

    async def add_events(self, rows: Sequence[EventRow]) -> None:
        self.calls += 1
        if self.calls == 1:
            self.writing.set()
            await self.go.wait()
            if self.fail:
                raise RuntimeError("database down")
        self.rows.update((str(r["id"]), r) for r in rows)


def _types(rows: Sequence[EventRow]) -> list[str]:
    return [r["type"] for r in rows]


async def test_a_queue_that_overflows_during_a_write_keeps_the_unwritten_rows(
    ground: Ground,
) -> None:
    """The queue drops its oldest row when full. A row dropped meanwhile was already in
    the write: what is left after the write is every row queued since, none lost."""
    repo = GatedRepo()
    g: Any = ground
    g.repo = repo
    g._events = deque[EventRow](maxlen=3)
    ground.add_event("a")
    ground.add_event("b")
    first = asyncio.create_task(ground.flush_events())
    await repo.writing.wait()  # writing a and b
    ground.add_event("c")
    ground.add_event("d")  # full: a goes from the queue
    assert _types(ground.queued_events()) == ["b", "c", "d"]
    repo.go.set()
    await first
    assert _types(ground.queued_events()) == ["c", "d"]
    await ground.flush_events()
    assert sorted(_types(list(repo.rows.values()))) == ["a", "b", "c", "d"]
    assert ground.queued_events() == []


async def test_a_failed_write_lets_the_next_flush_write_every_row(ground: Ground) -> None:
    repo = GatedRepo(fail=True)
    g: Any = ground
    g.repo = repo
    ground.add_event("a")
    first = asyncio.create_task(ground.flush_events())
    await repo.writing.wait()
    ground.add_event("b")
    waiting = asyncio.create_task(ground.flush_events())  # waits for the lock
    await asyncio.sleep(0)
    repo.go.set()
    await first  # failed, logged: the rows stay queued
    await asyncio.wait_for(waiting, 1)
    assert sorted(_types(list(repo.rows.values()))) == ["a", "b"]
    assert ground.queued_events() == []


async def test_a_cancelled_flush_lets_a_waiting_flush_write_every_row(ground: Ground) -> None:
    repo = GatedRepo()
    g: Any = ground
    g.repo = repo
    ground.add_event("a")
    first = asyncio.create_task(ground.flush_events())
    await repo.writing.wait()
    ground.add_event("b")
    waiting = asyncio.create_task(ground.flush_events())  # waits for the lock
    await asyncio.sleep(0)
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    assert first.cancelled()
    await asyncio.wait_for(waiting, 1)
    assert sorted(_types(list(repo.rows.values()))) == ["a", "b"]
    assert ground.queued_events() == []
