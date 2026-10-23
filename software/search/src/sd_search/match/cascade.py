"""The matching cascade, steps 1-2: the code always assesses; the fast cloud model next.

The code's colour score runs in the search loop for every candidate. A candidate the
code has seen goes on to the cloud model in the background, unless it is clearly not a
person (``refined_person_score < 0.1`` and ``best_detector_score < 0.35`` and seen once).
At most ``max_concurrent`` calls run at a time and at most ``queue_max`` wait; a full
queue takes nothing new (the candidate is offered again on its next change).

When the cloud fails (no network, a timeout, a server error) or the budget refuses, the
matching pauses: running calls finish, nothing new is sent, and the cloud is tried again
after ``retry_after_s``. The heartbeat says ``down`` (and the panel that only the code
assesses) until a call gets through again, retries refused by the budget included.
Without a cloud engine the state is ``off``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, Literal

from sd_brain.runtime.base import ModelError, ModelTimeoutError
from sd_brain.runtime.budget import BudgetDeniedError
from sd_brain.runtime.images import CloudImageRejected
from sd_schemas.search import CandidateAssessment
from sd_search.match.engines import MatchEngine, MatchRequest

__all__ = ["VLM_QUEUE_MAX", "Cascade", "MatchResult", "skip_cloud"]

log = logging.getLogger("sd_search.match")
VLM_QUEUE_MAX: Final = 8
MAX_CONCURRENT: Final = 4
RETRY_AFTER_S: Final = 60.0


def skip_cloud(refined: float | None, best_detector_score: float, n_seen: int) -> bool:
    """Clearly not a person: the code's assessment is enough."""
    return refined is not None and refined < 0.1 and best_detector_score < 0.35 and n_seen == 1


@dataclass(frozen=True)
class MatchResult:
    mission_id: str
    candidate_id: str
    key: tuple[int, tuple[str, ...]]  # (profile version, crops) the answer is about
    assessment: CandidateAssessment | None  # None: the cloud paused, ask again later


def _pauses(exc: ModelError) -> bool:
    """A failure of the cloud (or of the budget), not of this one candidate."""
    if isinstance(exc, CloudImageRejected):
        return False
    return isinstance(exc, (BudgetDeniedError, ModelTimeoutError)) or exc.kind == "http_error"


@dataclass
class Cascade:
    engine: MatchEngine | None
    max_concurrent: int = MAX_CONCURRENT
    queue_max: int = VLM_QUEUE_MAX
    retry_after_s: float = RETRY_AFTER_S
    monotonic: Callable[[], float] = time.monotonic
    paused_until: float | None = None
    down: bool = False  # from a pause until a call gets through again
    results: list[MatchResult] = field(default_factory=list[MatchResult])
    _pending: dict[str, asyncio.Task[None]] = field(default_factory=dict[str, asyncio.Task[None]])
    _slots: asyncio.Semaphore | None = None

    @property
    def state(self) -> Literal["off", "ok", "down"]:
        if self.engine is None:
            return "off"
        return "down" if self.down or self.paused else "ok"

    @property
    def paused(self) -> bool:
        if self.paused_until is None:
            return False
        if self.monotonic() >= self.paused_until:
            self.paused_until = None
            log.info("molnmatchningen försöker igen")
            return False
        return True

    @property
    def accepting(self) -> bool:
        return self.engine is not None and not self.paused and len(self._pending) < self.queue_max

    def pending(self, candidate_id: str) -> bool:
        return candidate_id in self._pending

    def submit(self, req: MatchRequest, key: tuple[int, tuple[str, ...]]) -> bool:
        """Start a cloud assessment in the background; False when off, paused, full or
        already asked for this candidate."""
        if self.engine is None or self.paused or req.candidate_id in self._pending:
            return False
        if len(self._pending) >= self.queue_max:
            return False
        if self._slots is None:
            self._slots = asyncio.Semaphore(self.max_concurrent)
        task = asyncio.create_task(self._run(self.engine, self._slots, req, key))
        self._pending[req.candidate_id] = task
        task.add_done_callback(lambda _: self._pending.pop(req.candidate_id, None))
        return True

    async def _run(
        self,
        engine: MatchEngine,
        slots: asyncio.Semaphore,
        req: MatchRequest,
        key: tuple[int, tuple[str, ...]],
    ) -> None:
        again = MatchResult(req.mission_id, req.candidate_id, key, None)
        async with slots:
            if self.paused:
                self.results.append(again)  # paused while waiting: offered after the pause
                return
            try:
                a = await engine.assess(req)
            except ModelError as exc:
                if _pauses(exc):
                    self._pause(exc)
                    self.results.append(again)
                else:  # this candidate's crops or answer: not asked again until they change
                    log.warning("molnbedömningen misslyckades (%s): %s", exc.kind, exc)
                return
            except Exception:  # a bug in the engine: logged, the search goes on
                log.exception("molnbedömningen kraschade")
                return
        self.down = False
        self.results.append(MatchResult(req.mission_id, req.candidate_id, key, a))

    def _pause(self, exc: ModelError) -> None:
        self.down = True
        if self.paused_until is None:
            why = "budgeten" if isinstance(exc, BudgetDeniedError) else exc.kind
            log.warning(
                "molnmatchningen pausas (%s); nytt försök om %.0f s", why, self.retry_after_s
            )
        self.paused_until = self.monotonic() + self.retry_after_s

    def take(self) -> list[MatchResult]:
        out, self.results = self.results, []
        return out

    async def drain(self) -> None:
        """Wait for every call in flight (tests, shutdown)."""
        while self._pending:
            await asyncio.gather(*list(self._pending.values()), return_exceptions=True)

    def cancel(self) -> None:
        for task in self._pending.values():
            task.cancel()
