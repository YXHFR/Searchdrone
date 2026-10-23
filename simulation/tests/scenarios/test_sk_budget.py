"""SK11: the cloud budget runs out during a search on ArduCopter SITL.

The search service runs as in SK01 (synthetic person reports, the real SD-link and
perception path), now with the fast cloud model behind its cascade: a fake
OpenAI-compatible server (no network, no key) whose every answer costs 0.002 USD, and the
ground's real budget guard on the served commander listener. Each new still of the
person is a little larger, so each one is a new crop for the cloud to look at. The
mission cap is set to 0.05 kr by the pilot.

* The matching stops within one call of the cap: the budget refuses, nothing more
  reaches the cloud while the reports keep coming (the cascade retries every 2 s and is
  refused before any call), the health part ``search`` says ``degraded`` and ``ready``
  stays.
* The panel's budget says ``exhausted``, ``/system/status`` too (the panel then shows the
  command mode), the mission chat says so, and the commander's own reservation for the
  mission is refused (it falls back to the command mode).
* Every call that reached the cloud has its ``reserved`` and ``committed`` rows in
  ``cloud_usage``, and nothing else is booked.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "software/ground/tests/helpers"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "software/search/tests"))

import flow_rig  # pyright: ignore[reportMissingImports]
from fake_cloud import FakeCloud, runtime  # pyright: ignore[reportMissingImports]
from scenario_rig import (  # pyright: ignore[reportMissingImports]
    Scenario,
    fresh_db,
    migrated_db,
    passwords,
    scenario,
    sim,
)
from test_sk_search import (  # pyright: ignore[reportMissingImports]
    RED,
    People,
    Profile,
    _crop,
    _rows,
    home_and_land,
)

from sd_brain.runtime.budget import HttpBudgetClient
from sd_ground.db.cloud_usage import CloudUsageRepo
from sd_ground.services.cloud_budget import EXHAUSTED_TEXT, CloudBudget, load_prices
from sd_safety.targets import Plane
from sd_search.app import SearchService
from sd_search.client import GroundClient
from sd_search.match.cascade import Cascade
from sd_search.match.engines import CloudMatchEngine
from sd_sim.report import measure

pytestmark = [pytest.mark.sitl, pytest.mark.slow]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

ROOT = Path(__file__).resolve().parents[3]

CAP_SEK = 0.05
COST_USD = 0.002  # about 0.02 kr a call at the example rate: the cap holds two or three


class GrowingPeople(People):
    """SK01's reports, each detection a pixel larger than the one before, so the cascade
    always has a new largest crop to ask about."""

    def _report(self) -> Any:
        out = super()._report()
        if out is None:
            return None
        report = out.report.model_copy(
            update={
                "detections": [
                    d.model_copy(update={"px_size": 20.0 + self.total()})
                    for d in out.report.detections
                ]
            }
        )
        return type(out)(report, out.crops)


@contextlib.asynccontextmanager
async def search_with_cloud(
    sc: Scenario, profile: Profile, cloud: FakeCloud
) -> AsyncIterator[Cascade]:
    url = f"http://127.0.0.1:{sc.port}"
    client = GroundClient(url, flow_rig.SEARCH)
    budget = HttpBudgetClient(url, flow_rig.SEARCH)
    cascade = Cascade(CloudMatchEngine(runtime(cloud, budget)), retry_after_s=2.0)
    service = SearchService(client, profile, cascade=cascade)
    task = asyncio.ensure_future(service.run(asyncio.Event(), poll_s=0.25))
    try:
        yield cascade
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        cascade.cancel()
        await budget.aclose()
        await client.aclose()


async def booked(sc: Scenario) -> list[tuple[str, str, str, float | None]]:
    rows = await _rows(
        sc,
        "SELECT reservation_id::text, process, status, actual_sek FROM cloud_usage "
        "WHERE mission_id = CAST(:m AS uuid) ORDER BY created_at, id",
    )
    return [(str(a), str(b), str(c), None if d is None else float(d)) for a, b, c, d in rows]


@pytest.mark.budget_s(480)
async def test_sk11_the_budget_runs_out_mid_search(
    sim: Any, migrated_db: Any, tmp_path: Path
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        g = sc.r.g
        g.cloud_budget = CloudBudget(g, load_prices(ROOT), CloudUsageRepo(sc.r.flow.repo.engine))
        await sc.airborne()
        res = await sc.r.panel.put(
            f"/api/v1/missions/{sc.mid}/cloud-budget", json={"cap_sek": CAP_SEK}, headers=flow_rig.H
        )
        assert res.status_code == 200, res.text

        red = Plane(*sc.r.home).latlon(0.0, 50.0)
        people = GrowingPeople(sc, {"red": (red, _crop(RED))})
        cloud = FakeCloud(cost_usd=COST_USD)
        sender = asyncio.ensure_future(people.run())
        try:
            res = await sc.order("goto", {"lat": red[0], "lon": red[1], "alt_m": 40, "speed_ms": 4})
            assert res.get("decision") == "ALLOW", res
            async with search_with_cloud(sc, Profile(sc), cloud) as cascade:
                await sc.until(lambda: cascade.state == "down", 120, "the matching paused")
                calls = len(cloud.requests)
                sent = people.total()
                # The person stays in view and the cascade retries: no call gets through.
                await sc.until(lambda: people.total() >= sent + 16, 30, "more reports")
                assert len(cloud.requests) == calls, "a cloud call after the budget ran out"
                health = (await sc.r.panel.get("/api/v1/health")).json()
                assert health["parts"]["search"] == "degraded", health["parts"]
                assert health["ready"] is True, health
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

        # Within one call of the cap: the spend before the last call was under it.
        rows = await booked(sc)
        committed = [r for r in rows if r[2] == "committed"]
        assert calls >= 2 and len(committed) == calls, (calls, rows)
        assert {r[1] for r in rows} == {"search"}
        assert sorted(r[0] for r in rows if r[2] == "reserved") == sorted(r[0] for r in committed)
        spent = [r[3] or 0.0 for r in committed]
        assert sum(spent[:-1]) <= CAP_SEK < sum(spent) + 0.01, spent
        measure("cloud calls before the cap", calls, calls)

        status = (
            await sc.r.panel.get("/api/v1/cloud-budget", params={"mission_id": sc.mid})
        ).json()
        assert status["state"] == "exhausted", status
        system = (await sc.r.panel.get("/api/v1/system/status")).json()
        assert system["cloud_budget"] == "exhausted", system
        assert any(m.get("text") == EXHAUSTED_TEXT for m in await sc.chat())

        # The commander's chat would be the next cloud call: refused the same way.
        res = await sc.r.commander.post(
            "/internal/v1/cloud-budget/reserve",
            json={
                "purpose": "chat",
                "mission_id": sc.mid,
                "model_id": "anthropic/claude-haiku-5.5",
                "est_prompt_tokens": 2000,
                "est_completion_tokens": 300,
            },
        )
        assert res.json()["granted"] is False and res.json()["reason"] == "MISSION_CAP"
        assert len(await booked(sc)) == len(rows)
        await home_and_land(sc)
