"""B2-06: the cloud budget's books against a migrated database.

Both cloud callers (the commander's token and the search token) reserve, commit and
release over the commander listener; every step is a ``cloud_usage`` row. The pilot's
mission cap is an event, so a restarted ground station (a new ``CloudBudget`` on the same
database) knows both what was spent and the cap.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from flow_rig import H, Rig, rig  # pyright: ignore[reportMissingImports]

from sd_ground.db.cloud_usage import CloudUsageRepo
from sd_ground.services.cloud_budget import CloudBudget, load_prices

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]

ROOT = Path(__file__).resolve().parents[4]
MODEL = "anthropic/claude-haiku-5.5"
BASE = "/internal/v1/cloud-budget"


def budget(r: Rig) -> CloudBudget:
    assert r.g.repo is not None
    return CloudBudget(r.g, load_prices(ROOT), CloudUsageRepo(r.g.repo.engine))


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        built.g.cloud_budget = budget(built)
        yield built


def reserve_body(mid: str | None, purpose: str = "match") -> dict[str, Any]:
    return {
        "purpose": purpose,
        "mission_id": mid,
        "model_id": MODEL,
        "est_prompt_tokens": 1200,
        "est_completion_tokens": 300,
        "image_count": 2,
    }


async def rows(r: Rig) -> list[tuple[str, str, str]]:
    assert r.g.repo is not None
    async with r.g.repo.engine.connect() as conn:
        res = await conn.execute(
            text(
                "SELECT reservation_id::text, process, status FROM cloud_usage "
                "ORDER BY created_at, id"
            )
        )
        return [(a, b, c) for a, b, c in res.all()]


async def test_reserve_commit_release_rows(r: Rig) -> None:
    mid = await r.open_mission()
    res = await r.search.post(f"{BASE}/reserve", json=reserve_body(mid))
    assert res.status_code == 200, res.text
    grant = res.json()
    assert grant["granted"] is True and grant["est_sek"] > 0
    rid = grant["reservation_id"]
    done = {"reservation_id": rid, "prompt_tokens": 900, "completion_tokens": 60}
    res = await r.search.post(f"{BASE}/commit", json=done | {"cost_usd": 0.0004})
    assert res.status_code == 200, res.text
    assert res.json()["actual_sek"] == pytest.approx(0.0004 * 10.5)

    # The commander's reservation is its own: the search token may not book it.
    res = await r.commander.post(f"{BASE}/reserve", json=reserve_body(mid, "chat"))
    chat = res.json()["reservation_id"]
    res = await r.search.post(f"{BASE}/release", json={"reservation_id": chat})
    assert res.status_code == 404
    res = await r.commander.post(f"{BASE}/release", json={"reservation_id": chat})
    assert res.status_code == 204

    assert await rows(r) == [
        (rid, "search", "reserved"),
        (rid, "search", "committed"),
        (chat, "commander", "reserved"),
        (chat, "commander", "released"),
    ]
    # The app role may only add rows (the books are append-only).
    with pytest.raises(Exception, match="permission denied"):
        assert r.g.repo is not None
        async with r.g.repo.engine.begin() as conn:
            await conn.execute(text("UPDATE cloud_usage SET est_sek = 0"))

    status = (await r.panel.get("/api/v1/cloud-budget", params={"mission_id": mid})).json()
    assert status["mission_sek"] == pytest.approx(0.0042)
    assert status["state"] == "ok"


async def test_pilot_cap_survives_restart(r: Rig) -> None:
    mid = await r.open_mission()
    res = await r.panel.put(
        f"/api/v1/missions/{mid}/cloud-budget", json={"cap_sek": 0.001}, headers=H
    )
    assert res.status_code == 200, res.text
    res = await r.search.post(f"{BASE}/reserve", json=reserve_body(mid))
    assert res.json() | {"remaining_sek": 0} == {
        "granted": False,
        "reservation_id": None,
        "reason": "MISSION_CAP",
        "est_sek": res.json()["est_sek"],
        "remaining_sek": 0,
    }
    assert await rows(r) == []  # a refused call leaves no row
    status = (await r.panel.get("/api/v1/cloud-budget", params={"mission_id": mid})).json()
    assert status["state"] == "exhausted" and status["mission_cap_sek"] == 0.001
    assert "cloud_budget.state" in r.bus_types()

    # The event reaches the database; a new budget on it knows the cap.
    async def stored() -> bool:
        return bool(await CloudUsageRepo(r.g.repo.engine).mission_caps())  # type: ignore[union-attr]

    for _ in range(100):
        if await stored():
            break
        await asyncio.sleep(0.05)
    again = budget(r)
    r.g.cloud_budget = again
    res = await r.search.post(f"{BASE}/reserve", json=reserve_body(mid))
    assert res.json()["reason"] == "MISSION_CAP"
    assert again.mission_cap(mid) == 0.001

    # The pilot raises it: calls go through again, and the spend is there after a restart.
    res = await r.panel.put(f"/api/v1/missions/{mid}/cloud-budget", json={"cap_sek": 50}, headers=H)
    assert res.status_code == 200
    grant = (await r.search.post(f"{BASE}/reserve", json=reserve_body(mid))).json()
    assert grant["granted"] is True
    done = {"reservation_id": grant["reservation_id"], "prompt_tokens": 1, "completion_tokens": 1}
    assert (
        await r.search.post(f"{BASE}/commit", json=done | {"cost_usd": 0.01})
    ).status_code == 200
    r.g.cloud_budget = budget(r)
    status = (await r.panel.get("/api/v1/cloud-budget", params={"mission_id": mid})).json()
    assert status["mission_sek"] == pytest.approx(0.105)
