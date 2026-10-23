"""The process stops (exit 1) when a main task dies, so it never looks alive frozen; a
dying perception task is logged and never stops the executor or the link."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sd_onboard import app
from sd_onboard.config import OnboardConfig


class _Stub:
    def __init__(self, *_: Any, **__: Any) -> None:
        self.executor: Any = None

    async def run(self, stop: asyncio.Event) -> None:
        await stop.wait()

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def on_detections(self, detections: Any) -> None:
        return None


class _DyingExecutor(_Stub):
    async def run(self, stop: asyncio.Event) -> None:
        raise RuntimeError("tick crashed")


def _patch(monkeypatch: pytest.MonkeyPatch, executor: type[_Stub]) -> None:
    monkeypatch.setattr(app, "read_token", lambda _l: "x" * 40)
    monkeypatch.setattr(app, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(app, "connect", lambda *_a, **_k: object())
    monkeypatch.setattr(app, "Vehicle", _Stub)
    monkeypatch.setattr(app, "OnboardLink", _Stub)
    monkeypatch.setattr(app, "Executor", executor)
    monkeypatch.setattr(app, "StateStore", _Stub)
    monkeypatch.setattr(app, "effective_params", lambda _f: {})


def _loaded(tmp_path: Any) -> Any:
    class _Loaded:
        config = OnboardConfig()
        state_dir = tmp_path
        root = tmp_path

        def param_files(self) -> list[Any]:
            return []

    return _Loaded()


async def test_a_crashing_tick_stops_the_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _patch(monkeypatch, _DyingExecutor)
    assert await asyncio.wait_for(app.serve(_loaded(tmp_path)), 5) == 1


class _TickingExecutor(_Stub):
    ticks = 0

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            type(self).ticks += 1
            await asyncio.sleep(0.01)


class _DyingPerception:
    ran = False
    on_targets: Any = None

    async def capture(self, command_id: Any) -> None:
        return None

    async def run(self, stop: asyncio.Event) -> None:
        type(self).ran = True
        raise RuntimeError("detector crashed")


async def test_a_crashing_perception_task_never_stops_flying(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _patch(monkeypatch, _TickingExecutor)
    monkeypatch.setattr(app, "_perception", lambda *_a: _DyingPerception())
    stop = asyncio.Event()
    serve = asyncio.ensure_future(app.serve(_loaded(tmp_path), stop))
    for _ in range(200):
        if _DyingPerception.ran:
            break
        await asyncio.sleep(0.01)
    assert _DyingPerception.ran
    before = _TickingExecutor.ticks
    await asyncio.sleep(0.2)
    assert not serve.done()  # perception died; the process lives on
    assert _TickingExecutor.ticks > before  # and the executor keeps ticking
    stop.set()
    assert await asyncio.wait_for(serve, 5) == 0  # a clean stop, no failure recorded
