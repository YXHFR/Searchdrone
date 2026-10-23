"""``inspect_detection`` through the ground: the ground fills in where the detection lies
and its keep-out radius before the gate, and says ``ONB_INSPECT_GEOMETRY`` when no
height in the charter sees the target steeply enough."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flow_rig import Rig, rig  # pyright: ignore[reportMissingImports]
from test_candidates import person, send  # pyright: ignore[reportMissingImports]
from test_orders import (  # pyright: ignore[reportMissingImports]
    driver,
    no_rate_limit,
    order,
    take_off,
)

from sd_ground.services import commands
from sd_safety.constants import PERSON_KEEPOUT_M
from sd_schemas.base import new_id

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


def look(det: Any) -> dict[str, Any]:
    return {"detection_id": str(det.detection_id), "standoff_m": 20}


async def flying_with(r: Rig, *dets: Any) -> tuple[str, Any]:
    mid = await r.ready_mission()
    d = driver(r, mid)
    await take_off(r, mid, d)
    await send(r, *dets)
    return mid, d


async def test_the_ground_fills_in_the_target(r: Rig) -> None:
    seen = person(r, 0, 100)
    _, d = await flying_with(r, seen)
    r.onboard.order_outcome = "running"
    await order(r, d, "inspect_detection", look(seen))
    await r.until(
        lambda: [o for o in r.onboard.orders if o.skill == "inspect_detection"], 5, "inspect"
    )
    sent = next(o for o in r.onboard.orders if o.skill == "inspect_detection")
    assert sent.params["lat"] == pytest.approx(seen.lat)
    assert sent.params["lon"] == pytest.approx(seen.lon)
    assert sent.params["target_radius_m"] == pytest.approx(PERSON_KEEPOUT_M + seen.sigma95_m)


async def test_a_target_no_height_sees_steeply_is_refused(r: Rig) -> None:
    vague = person(r, 0, 100, sigma95_m=40.0)  # 70 m around it: too wide for 120 m up
    _, d = await flying_with(r, vague)
    out = await order(r, d, "inspect_detection", look(vague), want="REJECT")
    assert out["reason_codes"] == ["ONB_INSPECT_GEOMETRY"], out
    assert not [o for o in r.onboard.orders if o.skill == "inspect_detection"]
    # The estimate resolves the target the same way as the order.
    no_rate_limit(r)
    situation = await d.situation()
    body = {
        "command_id": str(new_id()),
        "skill": "inspect_detection",
        "params": look(vague),
        "rationale_sv": "Går det?",
        "in_reply_to": None,
        "backend": "scripted",
        "ai_config_hash": "script-driver",
        "based_on": {
            "state_epoch": situation["snapshot_state_epoch"],
            "state_seq": situation["snapshot_state_seq"],
        },
        "charter_version": situation["charter"]["version"],
    }
    res = await r.commander.post(f"/commander/v1/missions/{d.mission_id}/estimate", json=body)
    assert res.status_code == 200, res.text
    assert res.json()["gate"]["reason_codes"] == ["ONB_INSPECT_GEOMETRY"], res.text


async def test_an_unknown_detection_or_a_bad_order_is_refused_as_such(r: Rig) -> None:
    _, d = await flying_with(r, person(r, 0, 100))
    # The panel's own lat/lon are dropped: the ground alone says where the target is.
    unknown = {"detection_id": str(new_id()), "standoff_m": 20, "lat": 1.0, "lon": 1.0}
    out = await order(r, d, "inspect_detection", unknown, want="REJECT")
    assert out["reason_codes"] == ["SCHEMA_INVALID"], out
    out = await order(r, d, "inspect_detection", {"standoff_m": -1}, want="REJECT")
    assert out["reason_codes"] == ["SCHEMA_INVALID"], out


async def test_the_geometry_check_needs_a_target_and_a_vehicle(r: Rig) -> None:
    seen = person(r, 0, 100)
    mid, _ = await flying_with(r, seen)
    ctx = await r.flow.gate_context(await r.flow.open_mission_of(mid))
    assert ctx.vehicle is not None
    found = await commands._inspect_params(  # pyright: ignore[reportPrivateUsage]
        r.flow, mid, look(seen)
    )
    check = commands._inspect_geometry_check  # pyright: ignore[reportPrivateUsage]
    assert check(found, ctx) is None  # a target the drone can look at
    assert check(found, ctx.model_copy(update={"vehicle": None})) is None
    assert check(look(seen), ctx) is None
    assert check({"standoff_m": -1}, ctx) is None
    far = found | {"target_radius_m": 200.0}
    refused = check(far, ctx)
    assert refused is not None and refused.reason_codes == ["ONB_INSPECT_GEOMETRY"]
