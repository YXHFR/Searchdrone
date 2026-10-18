"""``DecisionService``: the best available hosted engine decides.

All configured engines are asked at once. The answer used is the one from the highest
ranked engine that answers within the time limit; when a better engine fails or is too
slow, the next one in rank decides. Every engine's answer, also the late ones, is passed
to ``on_answer`` so the engines can be compared afterwards.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable, Sequence

from sd_brain.clock import utc_now
from sd_brain.decide.engines import DecideError, EngineAnswer, QuestionEngine
from sd_schemas.decide import DecideRequest, DecideResult, DecisionEngine

__all__ = ["DecisionService", "EngineOutcome"]

log = logging.getLogger("sd_brain.decide")


class EngineOutcome:
    def __init__(
        self, engine: DecisionEngine, answer: EngineAnswer | None, error: str | None, used: bool
    ) -> None:
        self.engine = engine
        self.answer = answer
        self.error = error
        self.used = used


OnAnswer = Callable[[DecideRequest, list[EngineOutcome]], Awaitable[None] | None]


class DecisionService:
    def __init__(
        self,
        engines: Sequence[QuestionEngine],
        *,
        on_answer: OnAnswer | None = None,
        late_answer_s: float = 5.0,
    ) -> None:
        self.engines = [e for e in engines if e.configured]
        self.on_answer = on_answer
        self.late_answer_s = late_answer_s
        self._background: set[asyncio.Task[None]] = set()

    def available(self) -> list[DecisionEngine]:
        return [e.name for e in self.engines]

    async def decide(self, req: DecideRequest, timeout_s: float) -> DecideResult | None:
        """The best engine's answer within ``timeout_s``, or ``None`` when none answered."""
        if not self.engines:
            return None
        start = time.monotonic()
        late = max(timeout_s, self.late_answer_s)
        tasks = [asyncio.create_task(e.answer(req, late)) for e in self.engines]
        chosen: int | None = None
        try:
            while True:
                for i, task in enumerate(tasks):
                    if not task.done():
                        break  # a better engine may still answer in time
                    if task.exception() is None:
                        chosen = i
                        break
                else:
                    break  # every engine has failed
                if chosen is not None:
                    break
                remaining = timeout_s - (time.monotonic() - start)
                if remaining <= 0:
                    chosen = next(
                        (i for i, t in enumerate(tasks) if t.done() and t.exception() is None),
                        None,
                    )
                    break
                pending = [t for t in tasks if not t.done()]
                await asyncio.wait(pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise
        self._log_when_done(req, tasks, chosen)
        if chosen is None:
            return None
        ans = tasks[chosen].result()
        fallback = self.engines[0].name if chosen > 0 else None
        return DecideResult(
            request_id=req.request_id,
            engine=ans.engine,
            location=ans.location,
            model_id=ans.model_id,
            latency_ms=int((time.monotonic() - start) * 1000),
            fallback_from=fallback,
            answers=ans.answers,
            created_at=utc_now(),
        )

    def _log_when_done(
        self, req: DecideRequest, tasks: list[asyncio.Task[EngineAnswer]], chosen: int | None
    ) -> None:
        async def collect() -> None:
            with contextlib.suppress(Exception):
                await asyncio.wait(tasks, timeout=self.late_answer_s + 1)
            outcomes: list[EngineOutcome] = []
            for i, task in enumerate(tasks):
                name = self.engines[i].name
                if not task.done():
                    task.cancel()
                    outcomes.append(EngineOutcome(name, None, "too_slow", False))
                elif task.cancelled():
                    outcomes.append(EngineOutcome(name, None, "cancelled", False))
                elif (exc := task.exception()) is not None:
                    reason = str(exc) if isinstance(exc, DecideError) else type(exc).__name__
                    outcomes.append(EngineOutcome(name, None, reason, False))
                else:
                    outcomes.append(EngineOutcome(name, task.result(), None, i == chosen))
            if self.on_answer is not None:
                try:
                    res = self.on_answer(req, outcomes)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:  # logging must never break a decision
                    log.exception("kunde inte logga beslutssvaren")

        bg = asyncio.create_task(collect())
        self._background.add(bg)
        bg.add_done_callback(self._background.discard)

    async def drain(self) -> None:
        """Wait for the background logging (tests, shutdown)."""
        while self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)
