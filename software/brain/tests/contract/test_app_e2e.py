"""The whole sd-brain process against the mock commander API (startup, stream, orders)."""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from brain_helpers import REG, fixture
from mock_commander import TOKEN, MockCommander

from sd_brain.app import StartError, _startup, build, run
from sd_brain.clock import utc_now
from sd_brain.config import BrainConfig
from sd_brain.evals.fake_ground import FakeGround
from sd_brain.secrets import Secrets
from sd_schemas.skills import SKILL_REGISTRY

pytestmark = pytest.mark.contract
KEY = "k" * 40


def _cfg(url: str) -> BrainConfig:
    return BrainConfig.model_validate(
        {
            "backend": {"kind": "grammar"},
            "ground": {"commander_url": url},
            "decide": {"engines": ["jev"]},
        }
    )


async def test_startup_refuses_a_different_registry() -> None:
    fg = FakeGround(fixture("airborne_modeB.json")["situation"], reg=REG, now=utc_now)
    async with MockCommander(fg) as mock:
        b = build(_cfg(mock.url), Secrets(TOKEN, KEY), SKILL_REGISTRY)
        with pytest.raises(StartError, match="registret"):
            await _startup(b, SKILL_REGISTRY)
        await b.http.aclose()


async def test_run_takes_orders_from_the_stream() -> None:
    fg = FakeGround(fixture("airborne_modeB.json")["situation"], reg=REG, now=utc_now)
    async with MockCommander(fg) as mock:
        task = asyncio.create_task(run(_cfg(mock.url), Secrets(TOKEN, KEY), REG))
        try:
            for _ in range(200):
                if mock.heartbeats:
                    break
                await asyncio.sleep(0.02)
            assert mock.hellos and mock.hellos[0]["backend"] == "grammar"
            evt = fg.operator_message("flyg till P1")
            evt.pop("seq")
            mock.push(evt)
            for _ in range(200):
                if fg.commands:
                    break
                await asyncio.sleep(0.02)
            assert [c.skill for c in fg.commands] == ["goto"]
            assert fg.commands[0].backend == "grammar"
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
