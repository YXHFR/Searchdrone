"""The whole sd-brain process against the mock commander API (startup, stream, orders)."""

from __future__ import annotations

import asyncio
import contextlib
import logging

import pytest
from brain_helpers import REG, fixture, with_stub
from mock_commander import TOKEN, MockCommander

from sd_brain.app import StartError, _startup, build, follow_ground_profile, run
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
    other = with_stub("report")  # the ground station runs another build
    fg = FakeGround(fixture("airborne_modeB.json")["situation"], reg=other, now=utc_now)
    async with MockCommander(fg) as mock:
        b = build(_cfg(mock.url), Secrets(TOKEN, KEY), SKILL_REGISTRY)
        with pytest.raises(StartError, match="registret"):
            await _startup(b, SKILL_REGISTRY)
        await b.http.aclose()


async def test_run_takes_orders_from_the_stream() -> None:
    fg = FakeGround(fixture("airborne_modeB.json")["situation"], reg=REG, now=utc_now)
    async with MockCommander(fg) as mock:
        cfg = _cfg(mock.url)
        assert cfg.cloud.provider.zdr_on  # until the ground station has said its profile
        task = asyncio.create_task(run(cfg, Secrets(TOKEN, KEY), REG))
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
            assert not cfg.cloud.provider.zdr_on  # the mock ground runs the sim profile
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


async def test_zdr_and_the_ai_hash_follow_the_ground_stations_profile(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = BrainConfig.model_validate(
        {"profile": "sim", "ground": {"commander_url": "http://127.0.0.1:9/api/v1/commander"}}
    )
    b = build(cfg, Secrets(TOKEN, KEY), SKILL_REGISTRY)
    try:
        before = b.loop.ai_hash
        assert cfg.cloud.provider.zdr_on
        with caplog.at_level(logging.INFO, logger="sd_brain"):
            follow_ground_profile(b, "sim")
            follow_ground_profile(b, "sim")  # already off: no second line
        assert caplog.text.count("zdr av") == 1
        assert not cfg.cloud.provider.zdr_on and b.loop.ai_hash != before
        follow_ground_profile(b, None)  # the stream is down: the profile is unknown again
        assert cfg.cloud.provider.zdr_on and b.loop.ai_hash == before
        follow_ground_profile(b, "sim")
        follow_ground_profile(b, "field")  # a sim brain against a field ground
        assert cfg.cloud.provider.zdr_on and b.loop.ai_hash == before
        cfg.cloud.provider.zdr = False
        follow_ground_profile(b, "field")
        assert "zdr = false" in caplog.text
    finally:
        await b.http.aclose()
        await b.budget.aclose()
        if b.cloud_http is not None:
            await b.cloud_http.aclose()
