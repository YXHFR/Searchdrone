"""The process stops (exit 1) when a main task dies, so it never looks alive frozen."""

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


class _DyingExecutor(_Stub):
    async def run(self, stop: asyncio.Event) -> None:
        raise RuntimeError("tick crashed")


async def test_a_crashing_tick_stops_the_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setattr(app, "read_token", lambda _l: "x" * 40)
    monkeypatch.setattr(app, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(app, "connect", lambda *_a, **_k: object())
    monkeypatch.setattr(app, "Vehicle", _Stub)
    monkeypatch.setattr(app, "OnboardLink", _Stub)
    monkeypatch.setattr(app, "Executor", _DyingExecutor)
    monkeypatch.setattr(app, "StateStore", _Stub)
    monkeypatch.setattr(app, "effective_params", lambda _f: {})

    class _Loaded:
        config = OnboardConfig()
        state_dir = tmp_path
        root = tmp_path

        def param_files(self) -> list[Any]:
            return []

    assert await asyncio.wait_for(app.serve(_Loaded()), 5) == 1  # type: ignore[arg-type]
