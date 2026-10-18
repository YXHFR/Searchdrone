"""``measure_cloud`` (``just perf``) against the fake OpenAI-compatible server on 127.0.0.1."""

from __future__ import annotations

import httpx
import pytest
from fake_openai import FakeOpenAI

from sd_brain import perf
from sd_brain.config import CloudConfig
from sd_brain.perf import measure_cloud

KEY = "sk-test-" + "k" * 40
MAIN = "z-ai/glm-5.3-flash"
FALLBACK = "anthropic/claude-haiku-5.5"


async def _measure(fake: FakeOpenAI, runs: int) -> dict[str, object]:
    async with httpx.AsyncClient(trust_env=False) as client:
        return await measure_cloud(CloudConfig(base_url=fake.base_url), KEY, client, runs=runs)


async def test_all_answered() -> None:
    with FakeOpenAI() as fake:
        report = await _measure(fake, 5)
    assert report["answered"] == 5 and report["errors"] == {}
    assert report["answered_by"] == {MAIN: 5}
    p50, p95, top = report["latency_p50_ms"], report["latency_p95_ms"], report["latency_max_ms"]
    assert isinstance(p50, int) and isinstance(p95, int) and isinstance(top, int)
    assert 0 <= p50 <= p95 <= top
    assert report["ok"] is True
    assert KEY not in str(report)  # the report never holds the key


async def test_the_fallback_model_counts_as_answered() -> None:
    with FakeOpenAI() as fake:
        fake.behaviour[MAIN] = "500"
        report = await _measure(fake, 3)
    assert report["answered_by"] == {FALLBACK: 3}
    assert report["errors"] == {} and report["ok"] is True


async def test_errors_fail_the_measurement() -> None:
    with FakeOpenAI() as fake:
        fake.behaviour[MAIN] = "500"
        fake.behaviour[FALLBACK] = "500"
        report = await _measure(fake, 3)
    assert report["answered"] == 0 and report["errors"] == {"http_error": 3}
    assert report["latency_p50_ms"] is None and report["latency_p95_ms"] is None
    assert report["ok"] is False


async def test_slow_median_fails_the_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(perf, "TARGET_P50_MS", -1)
    with FakeOpenAI() as fake:
        report = await _measure(fake, 2)
    assert report["answered"] == 2 and report["errors"] == {}
    assert report["ok"] is False
