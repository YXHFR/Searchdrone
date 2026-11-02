"""Keeps a SITL test's onboard side from going quiet without a trace.

The onboard executor, its SD-link client and perception run as tasks in the test's own
event loop. If one of them dies (an exception) or simply returns while the flight is on,
the drone keeps flying whatever the flight controller last had, and nothing in the test
says why. ``guarded`` runs the test's flight next to a check of those tasks and fails the
test at once, with the task's own error, when one ends; ``ExecutorRecord`` keeps the
executor's log lines (its events, ``executor.<type> {...}``, and its warnings) and its
state at the end in the run's ``logs/`` folder, which CI uploads when a test fails.

Nothing here touches the flight stack: it only reads.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import traceback
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any, Final

__all__ = [
    "EXECUTOR_LOG",
    "EXECUTOR_STATE",
    "ExecutorRecord",
    "OnboardDiedError",
    "dead_task",
    "executor_state",
    "guarded",
    "task_name",
]

EXECUTOR_LOG: Final = "onboard_executor.log"  # in the run's logs/ (CI keeps logs/*.log)
EXECUTOR_STATE: Final = "onboard_executor_state.log"  # JSON, one block per onboard side
PERIOD_S: Final = 0.25
_LOGGERS: Final = ("sd_onboard",)  # the executor, the skills, the link client, perception


class OnboardDiedError(AssertionError):
    """An onboard task ended while the flight was on."""


def task_name(task: asyncio.Future[Any]) -> str:
    """What the task runs (``Executor.run``, ``OnboardLink.run``, ...)."""
    get = getattr(task, "get_coro", None)
    coro = get() if get is not None else None
    return getattr(coro, "__qualname__", None) or repr(task)


def _error(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc)).strip()


def dead_task(tasks: Iterable[asyncio.Future[Any]]) -> str | None:
    """The first task that has ended, and how; None while all run."""
    for task in tasks:
        if not task.done():
            continue
        name = task_name(task)
        if task.cancelled():
            return f"{name} was cancelled"
        exc = task.exception()
        if exc is None:
            return f"{name} returned while the flight was on"
        return f"{name} died: {_error(exc)}"
    return None


async def guarded[T](
    work: Awaitable[T], check: Callable[[], str | None], *, period_s: float = PERIOD_S
) -> T:
    """``work``, failed at once with ``OnboardDiedError`` when ``check`` reports a dead
    onboard task (then ``work`` is cancelled); checked once more when ``work`` is done."""
    job = asyncio.ensure_future(work)

    async def watch() -> None:
        while True:
            why = check()
            if why is not None:
                raise OnboardDiedError(why)
            await asyncio.sleep(period_s)

    guard = asyncio.ensure_future(watch())
    try:
        await asyncio.wait({job, guard}, return_when=asyncio.FIRST_COMPLETED)
        if not job.done():
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)
            guard.result()  # raises OnboardDiedError
        result = job.result()
        why = check()
        if why is not None:
            raise OnboardDiedError(why)
        return result
    finally:
        for t in (job, guard):
            t.cancel()
        await asyncio.gather(job, guard, return_exceptions=True)


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list | tuple | set | frozenset):
        return [_plain(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
    return str(value)


def _task_state(task: asyncio.Future[Any] | None) -> dict[str, Any] | None:
    if task is None:
        return None
    out: dict[str, Any] = {"name": task_name(task), "done": task.done()}
    if not task.done():
        stack = task.get_stack() if isinstance(task, asyncio.Task) else []
        out["awaiting"] = [f"{f.f_code.co_filename}:{f.f_lineno} {f.f_code.co_name}" for f in stack]
    elif task.cancelled():
        out["cancelled"] = True
    elif task.exception() is not None:
        out["error"] = _error(task.exception())  # pyright: ignore[reportArgumentType]
    return out


def executor_state(ex: Any, tasks: Iterable[asyncio.Future[Any]] = ()) -> dict[str, Any]:
    """What the executor holds now (``sd_onboard.executor.Executor``, read duck-typed):
    its state and who holds it, the running order and its task (where it waits), the
    authority's pending mode and the last notice, and the onboard side's own tasks."""
    active = getattr(ex, "active", None)
    auth = getattr(ex, "auth", None)
    t = getattr(getattr(ex, "vehicle", None), "telemetry", None)
    now = ex.now_ms() if hasattr(ex, "now_ms") else None
    last_tick = getattr(ex, "_last_tick_ms", None)
    state: dict[str, Any] = {
        "state": getattr(ex, "state", None),
        "hold_reason": getattr(ex, "hold_reason", None),
        "held_by": getattr(ex, "held_by", None),
        "control_suspended": getattr(ex, "control_suspended", None),
        "taken_by": getattr(ex, "_taken_by", None),
        "loop_alive": ex.loop_alive() if hasattr(ex, "loop_alive") else None,
        "tick_age_ms": None if now is None or last_tick is None else now - last_tick,
        "ai_silent_after_step": getattr(ex, "ai_silent_after_step", None),
        "resume_point": getattr(ex, "resume_point", None),
        "keepouts": len(getattr(ex, "keepouts", ())),
        "local_keepouts": len(getattr(ex, "local_keepouts", {})),
        "auth": None
        if auth is None
        else {
            "pending_mode": getattr(auth, "pending_mode", None),
            "notice_mono_ms": getattr(auth, "notice_mono_ms", None),
            "change": getattr(auth, "change", None),
        },
        "fc": None
        if t is None
        else {
            "connected": getattr(t, "fc_connected", None),
            "mode": getattr(t, "fc_mode", None),
            "armed": getattr(t, "armed", None),
            "mission_seq": getattr(t, "mission_seq_current", None),
            "mission_total": getattr(t, "mission_total", None),
        },
        "active": None
        if active is None
        else {
            "skill": getattr(active, "skill", None),
            "command_id": getattr(active, "command_id", None),
            "origin": getattr(active, "origin", None),
            "ack_done": getattr(active, "ack_done", None),
            "task": _task_state(getattr(active, "task", None)),
        },
        "side_tasks": [_task_state(s) for s in getattr(ex, "_side_tasks", ())],
        "onboard_tasks": [_task_state(s) for s in tasks],
    }
    return _plain(state)


class ExecutorRecord:
    """The ``sd_onboard`` log lines in ``<run>/logs/onboard_executor.log`` from ``start``
    to ``close``, and the executor's state at ``close`` (or ``snapshot``) in
    ``onboard_executor_state.log``. Several onboard sides in one run append."""

    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir / "logs"
        self._handler: logging.Handler | None = None
        self._levels: dict[str, int] = {}

    def start(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(self.dir / EXECUTOR_LOG, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        line = "%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s %(message)s"
        fmt = logging.Formatter(line, "%H:%M:%S")
        fmt.converter = time.gmtime  # UTC, as the CI logs and the tlogs
        handler.setFormatter(fmt)
        for name in _LOGGERS:
            logger = logging.getLogger(name)
            self._levels[name] = logger.level
            if logger.getEffectiveLevel() > logging.INFO:
                logger.setLevel(logging.INFO)  # the executor's events are INFO lines
            logger.addHandler(handler)
        self._handler = handler

    def snapshot(self, why: str, ex: Any, tasks: Iterable[asyncio.Future[Any]] = ()) -> None:
        with contextlib.suppress(OSError), (self.dir / EXECUTOR_STATE).open("a") as f:
            stamp = time.strftime("%H:%M:%S", time.gmtime())
            f.write(f"--- {stamp}Z {why}\n")
            f.write(json.dumps(executor_state(ex, tasks), indent=2, ensure_ascii=False) + "\n")

    def close(self) -> None:
        handler, self._handler = self._handler, None
        if handler is None:
            return
        for name, level in self._levels.items():
            logger = logging.getLogger(name)
            logger.removeHandler(handler)
            logger.setLevel(level)
        handler.close()
