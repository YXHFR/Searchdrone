"""A sector that turns done with a part no track point saw (``services.coverage``): the
ground tells the operator once (alert ``SECTOR_GAP``) and the AI in its situation, both
in square metres and as a share of the sector, over the real SD-link server with the
fake drone."""

from __future__ import annotations

import asyncio
import sys
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flow_rig import Rig, rig  # pyright: ignore[reportMissingImports]
from test_orders import (  # pyright: ignore[reportMissingImports]
    driver,
    order,
    ready_with_sector,
    take_off,
    wait_order_status,
)

from sd_schemas.api import AlertCode

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


def gap_alerts(r: Rig) -> list[dict[str, Any]]:
    return [
        e.data
        for e in r.g.bus.replay(r.g.bus.server_epoch, 0) or []
        if e.type == "alert" and e.data.get("code") == str(AlertCode.SECTOR_GAP)
    ]


async def test_a_sector_done_with_an_unsearched_part_is_told_to_the_operator_and_the_ai(
    r: Rig,
) -> None:
    """The drone hovers in the middle of sector A (80 x 80 m) and the search ends done:
    only the disc it saw (20 / 2 + 2 m around the hover point, about 450 m²) is searched,
    so about 5 950 m² (93 %) is told as not searched."""
    mid, _sector = await ready_with_sector(r)
    d = driver(r, mid)
    await take_off(r, mid, d)
    r.onboard.order_outcome = "running"
    out = await order(r, d, "survey_sector", {"sector_id": "A", "alt_m": 40, "spacing_m": 20})
    sid = str(out["command_id"])
    await wait_order_status(r, mid, sid, "running")
    assert r.fc is not None
    r.fc.lat, r.fc.lon = r.at(50.0, 0.0)

    async def covered() -> bool:
        features = (await r.mission(mid))["sectors"]["features"]
        return next(f for f in features if f["id"] == "A")["properties"]["covered"] is not None

    await r.until(covered, 20, "hover point covered")
    assert not gap_alerts(r)  # nothing is told while the search runs

    assert r.onboard.session is not None
    await r.onboard.session.ack(uuid.UUID(sid), "done")
    await wait_order_status(r, mid, sid, "done")
    await r.until(lambda: gap_alerts(r), 10, "the gap told to the operator")

    (alert,) = gap_alerts(r)
    assert alert["severity"] == "warning"
    details = alert  # the details sit flat in the payload
    assert details["sector_id"] == "A"
    assert 5_500 <= details["unsearched_m2"] <= 6_100, details
    assert 86.0 <= details["unsearched_pct"] <= 95.0, details
    assert "Sektor A är klar men cirka" in alert["message_sv"]
    assert "m²" in alert["message_sv"] and "%" in alert["message_sv"]

    rows = {s["sector_id"]: s for s in (await d.situation())["sectors"]}
    assert rows["A"]["status"] == "done"
    assert rows["A"]["unsearched_m2"] == details["unsearched_m2"]
    assert rows["A"]["unsearched_pct"] == details["unsearched_pct"]

    # Told once: later ticks (one a second) leave it be.
    await asyncio.sleep(2.5)
    assert len(gap_alerts(r)) == 1

    # The ground restarts (its coverage memory is gone): the track is counted again and
    # the gap is kept for the AI, but the operator was told already.
    r.flow.coverage.forget(mid)

    async def rebuilt() -> bool:
        return r.flow.coverage.of(mid).gaps.get("A", (0.0, 0.0))[0] > 0

    await r.until(rebuilt, 10, "the gap worked out again")
    rows = {s["sector_id"]: s for s in (await d.situation())["sectors"]}
    assert rows["A"]["unsearched_m2"] == details["unsearched_m2"]
    await asyncio.sleep(1.5)
    assert len(gap_alerts(r)) == 1
