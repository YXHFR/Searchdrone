"""``inspect_detection`` through the ground: the ground fills in where the candidate lies,
its uncertainty and its keep-out radius before the gate, and says
``ONB_INSPECT_GEOMETRY`` when no height in the charter sees the target steeply enough.
The pilot clicks; the AI's order is only a proposal (an approval card)."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flow_rig import H, Rig, rig  # pyright: ignore[reportMissingImports]
from test_candidates import person, send  # pyright: ignore[reportMissingImports]
from test_orders import (  # pyright: ignore[reportMissingImports]
    driver,
    no_rate_limit,
    order,
    take_off,
)

from sd_ground.services import charters, commands
from sd_safety.constants import PERSON_KEEPOUT_M
from sd_schemas.base import new_id
from sd_schemas.skills import SKILL_REGISTRY

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


def look(det: Any) -> dict[str, Any]:
    # A candidate's id is its first detection's.
    return {"candidate_id": str(det.detection_id)}


async def flying_with(r: Rig, *dets: Any) -> tuple[str, Any]:
    skills = [*charters.default_skills(SKILL_REGISTRY), "inspect_detection"]
    mid = await r.ready_mission(allowed_skills=skills)
    d = driver(r, mid)
    await take_off(r, mid, d)
    await send(r, *dets)
    return mid, d


async def pilot(r: Rig, mid: str, params: dict[str, Any], want: str = "ALLOW") -> dict[str, Any]:
    """The pilot's "Titta närmare"; the gate's answer must be ``want``."""
    res = await r.panel.post(
        f"/api/v1/missions/{mid}/commands",
        json={"client_command_id": str(new_id()), "skill": "inspect_detection", "params": params},
        headers=H,
    )
    assert res.status_code == 201, res.text
    gate = res.json()["gate"]
    assert gate["decision"] == want, gate
    return gate


async def sent_onboard(r: Rig) -> Any:
    def looks() -> list[Any]:
        return [o for o in r.onboard.orders if o.skill == "inspect_detection"]

    await r.until(looks, 5, "inspect")
    return looks()[0]


async def test_the_ground_fills_in_the_target(r: Rig) -> None:
    seen = person(r, 0, 100)
    mid, _ = await flying_with(r, seen)
    r.onboard.order_outcome = "running"
    await pilot(r, mid, look(seen))
    sent = await sent_onboard(r)
    assert sent.params["lat"] == pytest.approx(seen.lat)
    assert sent.params["lon"] == pytest.approx(seen.lon)
    assert sent.params["sigma95_m"] == pytest.approx(seen.sigma95_m)
    assert sent.params["target_radius_m"] == pytest.approx(PERSON_KEEPOUT_M + seen.sigma95_m)
    assert sent.params["candidate_id"] == str(seen.detection_id)


async def test_the_ais_look_is_only_a_proposal(r: Rig) -> None:
    seen = person(r, 0, 100)
    _, d = await flying_with(r, seen)
    out = await order(r, d, "inspect_detection", look(seen), want="PENDING_APPROVAL")
    assert out["reason_codes"] == ["NEEDS_APPROVAL_SKILL"], out
    assert not [o for o in r.onboard.orders if o.skill == "inspect_detection"]


async def test_an_order_naming_a_detection_is_read_as_its_candidate(r: Rig) -> None:
    # Version 1 named a detection; for now the ground reads it as the candidate holding it.
    seen = person(r, 0, 100)
    mid, _ = await flying_with(r, seen)
    r.onboard.order_outcome = "running"
    await pilot(r, mid, {"detection_id": str(seen.detection_id)})
    sent = await sent_onboard(r)
    assert "detection_id" not in sent.params
    assert sent.params["candidate_id"] == str(seen.detection_id)


async def test_a_target_no_height_sees_steeply_is_refused(r: Rig) -> None:
    vague = person(r, 0, 100, sigma95_m=40.0)  # 70 m around it: too wide for 120 m up
    mid, d = await flying_with(r, vague)
    gate = await pilot(r, mid, look(vague), want="REJECT")
    assert gate["reason_codes"] == ["ONB_INSPECT_GEOMETRY"], gate
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


async def test_an_unknown_candidate_or_a_bad_order_is_refused_as_such(r: Rig) -> None:
    mid, _ = await flying_with(r, person(r, 0, 100))
    # The panel's own lat/lon are dropped: the ground alone says where the target is.
    unknown = {"candidate_id": str(new_id()), "lat": 1.0, "lon": 1.0, "sigma95_m": 1.0}
    gate = await pilot(r, mid, unknown, want="REJECT")
    assert gate["reason_codes"] == ["SCHEMA_INVALID"], gate
    gate = await pilot(r, mid, {"standoff_m": 20}, want="REJECT")
    assert gate["reason_codes"] == ["SCHEMA_INVALID"], gate


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
    assert check({"standoff_m": 20}, ctx) is None
    far = found | {"target_radius_m": 200.0}
    refused = check(far, ctx)
    assert refused is not None and refused.reason_codes == ["ONB_INSPECT_GEOMETRY"]
