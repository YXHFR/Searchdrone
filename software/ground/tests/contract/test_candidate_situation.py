"""B2-10: the AI sees the candidates and the confirmed finds in its situation, and is
woken by a high candidate and by a person's decision. It never decides one."""

from __future__ import annotations

import re
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flow_rig import BRAIN, Rig, rig  # pyright: ignore[reportMissingImports]
from test_candidate_decisions import (  # pyright: ignore[reportMissingImports]
    candidate,
    decide,
    person,
    view,
)

from sd_ground.api.commander import to_commander

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]

AS_BRAIN = {"Authorization": f"Bearer {BRAIN}"}


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


async def situation(r: Rig, mid: str) -> dict[str, object]:
    res = await r.commander.get(f"/commander/v1/missions/{mid}/situation", headers=AS_BRAIN)
    assert res.status_code == 200, res.text
    return res.json()


async def test_candidate_and_find_lines(r: Rig) -> None:
    mid = await r.open_mission()
    s = await situation(r, mid)
    assert s["candidates_line"] is None and s["finds_line"] is None

    cid, det = await candidate(r, mid)
    s = await situation(r, mid)
    line = str(s["candidates_line"])
    assert line.startswith("Kandidater: K1 HÖG ("), line
    assert line.endswith(" | att granska 1, låg 0"), line
    assert s["finds_line"] == "Fynd: inga bekräftade"

    async with person(r, "pilot", "kalle") as pilot:
        await view(pilot, det)
        res = await decide(pilot, cid, "confirm")
        assert res.status_code == 200, res.text
    s = await situation(r, mid)
    assert s["candidates_line"] == "Kandidater: att granska 0, låg 0"
    assert re.fullmatch(r"Fynd: K1 BEKRÄFTAD \d\d:\d\d, cylinder 33 m", str(s["finds_line"]))

    out = [to_commander(e) for e in r.g.bus.replay(r.g.bus.server_epoch, 0) or []]
    ai = [e for e in out if e is not None and e["type"].startswith("event.candidate")]
    assert [e["type"] for e in ai] == ["event.candidate", "event.candidate_decided"]
    assert ai[0]["body"]["detection_id"] == det
    assert ai[1]["body"]["decision"] == "confirm" and ai[1]["body"]["after"] == "confirmed"
    assert "kalle" not in str(ai)
