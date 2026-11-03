"""A dead onboard task fails a SITL test at once, with its own error, instead of the
drone flying on in silence; the executor's log lines and state go to the run's logs/."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from sd_sim.onboard_watch import (
    EXECUTOR_LOG,
    EXECUTOR_STATE,
    ExecutorRecord,
    OnboardDiedError,
    dead_task,
    executor_state,
    guarded,
)


class FakeExecutor:
    """The executor's shape as the record reads it, with a main loop that can crash."""

    def __init__(self, *, crash_after_s: float | None) -> None:
        self.crash_after_s = crash_after_s
        self.state = "SURVEY"
        self.held_by = None
        self.active = None
        self.ticks = 0

    def now_ms(self) -> int:
        return 1_000

    async def run(self, stop: asyncio.Event) -> None:
        log = logging.getLogger("sd_onboard")
        log.info("executor.state_changed {'to': 'SURVEY'}")
        loop = asyncio.get_running_loop()
        start = loop.time()
        while not stop.is_set():
            self.ticks += 1
            if self.crash_after_s is not None and loop.time() - start >= self.crash_after_s:
                raise RuntimeError("tick blew up")
            await asyncio.sleep(0.01)


async def _flight(seconds: float, flown: list[str]) -> str:
    await asyncio.sleep(seconds)
    flown.append("landed")
    return "done"


async def test_an_executor_crash_fails_the_flight_at_once() -> None:
    stop = asyncio.Event()
    ex = FakeExecutor(crash_after_s=0.05)
    tasks = [asyncio.ensure_future(ex.run(stop))]
    flown: list[str] = []
    loop = asyncio.get_running_loop()
    start = loop.time()
    with pytest.raises(OnboardDiedError) as err:
        await guarded(_flight(30.0, flown), lambda: dead_task(tasks), period_s=0.01)
    assert loop.time() - start < 5.0  # not after the flight's own 30 s
    assert flown == []  # the flight was cancelled, not finished
    assert "FakeExecutor.run died" in str(err.value)
    assert "RuntimeError: tick blew up" in str(err.value)
    stop.set()


async def test_an_executor_that_returns_early_fails_too() -> None:
    stop = asyncio.Event()
    ex = FakeExecutor(crash_after_s=None)
    tasks = [asyncio.ensure_future(ex.run(stop))]
    stop.set()  # the loop ends, but nobody closed the onboard side
    with pytest.raises(OnboardDiedError, match="returned while the flight was on"):
        await guarded(_flight(30.0, []), lambda: dead_task(tasks), period_s=0.01)


async def test_a_healthy_onboard_side_lets_the_flight_finish() -> None:
    stop = asyncio.Event()
    ex = FakeExecutor(crash_after_s=None)
    tasks = [asyncio.ensure_future(ex.run(stop))]
    flown: list[str] = []
    assert await guarded(_flight(0.05, flown), lambda: dead_task(tasks), period_s=0.01) == "done"
    assert flown == ["landed"] and ex.ticks > 0
    stop.set()
    await asyncio.gather(*tasks)


async def test_the_flights_own_error_is_kept() -> None:
    async def failing() -> None:
        await asyncio.sleep(0.01)
        raise AssertionError("no confirmed find")

    with pytest.raises(AssertionError, match="no confirmed find"):
        await guarded(failing(), lambda: None, period_s=0.01)


async def test_the_record_keeps_the_log_lines_and_the_state(tmp_path: Path) -> None:
    record = ExecutorRecord(tmp_path)
    record.start()
    stop = asyncio.Event()
    ex = FakeExecutor(crash_after_s=0.0)
    tasks = [asyncio.ensure_future(ex.run(stop))]
    await asyncio.gather(*tasks, return_exceptions=True)
    record.snapshot("onboard task ended", ex, tasks)
    record.close()
    logging.getLogger("sd_onboard").info("after close: not in the file")

    lines = (tmp_path / "logs" / EXECUTOR_LOG).read_text(encoding="utf-8")
    assert "executor.state_changed" in lines and "after close" not in lines
    text = (tmp_path / "logs" / EXECUTOR_STATE).read_text(encoding="utf-8")
    head, body = text.split("\n", 1)
    assert head.endswith("onboard task ended")
    state: dict[str, Any] = json.loads(body)
    assert state["state"] == "SURVEY" and state["active"] is None
    assert "RuntimeError: tick blew up" in state["onboard_tasks"][0]["error"]


def test_the_state_of_a_waiting_order() -> None:
    """Where the running order's task waits: the line that holds it up."""

    async def main() -> dict[str, Any]:
        class Active:
            skill = "survey_sector"
            command_id = "c-1"
            origin = "skill"
            ack_done = True

        async def stuck() -> None:
            await asyncio.Event().wait()

        ex = FakeExecutor(crash_after_s=None)
        active = Active()
        active.task = asyncio.ensure_future(stuck())  # type: ignore[attr-defined]
        ex.active = active  # type: ignore[assignment]
        await asyncio.sleep(0)
        state = executor_state(ex)
        active.task.cancel()  # type: ignore[attr-defined]
        return state

    state = asyncio.run(main())
    task = state["active"]["task"]
    assert state["active"]["skill"] == "survey_sector"
    assert task["done"] is False and any("stuck" in line for line in task["awaiting"])
