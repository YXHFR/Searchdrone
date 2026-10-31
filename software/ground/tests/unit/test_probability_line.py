"""The AI's probability-map situation line, and its fallback when the map cannot be read."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sd_ground.db.probmap import ProbabilityMapRepo
from sd_ground.services import probability_map as map_service
from sd_schemas.probmap import PSR_MAX

MISSION = "00000000-0000-0000-0000-000000000001"
SECTOR = {
    "sector_id": "A",
    "poa": 0.4,
    "covered_pct": 10.0,
    "mean_pod": 0.1,
    "pos_plan": 0.2,
    "est_min": 4.0,
    "psr": 0.05,
    "spacing_m": 30.0,
}


def _flow() -> Any:
    return SimpleNamespace(repo=SimpleNamespace(engine=object()))


def _flow_with_mission() -> Any:
    async def mission(mission_id: str) -> None:
        return None

    return SimpleNamespace(repo=SimpleNamespace(engine=object()), mission=mission)


def _head(monkeypatch: pytest.MonkeyPatch, head: Any) -> None:
    async def fake(self: ProbabilityMapRepo, mission_id: str) -> Any:
        if isinstance(head, Exception):
            raise head
        return head

    monkeypatch.setattr(ProbabilityMapRepo, "head", fake)


async def test_the_line_lists_the_sectors(monkeypatch: pytest.MonkeyPatch) -> None:
    _head(monkeypatch, {"grid": {"sectors": [SECTOR]}})
    line = await map_service.probability_line(_flow(), MISSION)
    assert line is not None and "A " in line


@pytest.mark.parametrize("head", [None, {"grid": {}}, {"grid": {"sectors": []}}])
async def test_no_line_before_a_map_with_sectors(
    monkeypatch: pytest.MonkeyPatch, head: Any
) -> None:
    _head(monkeypatch, head)
    assert await map_service.probability_line(_flow(), MISSION) is None


@pytest.mark.parametrize(
    "head",
    [
        RuntimeError("databasen svarar inte"),
        {"grid": {"sectors": [{**SECTOR, "poa": -1}]}},  # a stored row the schema refuses
    ],
    ids=["db-error", "bad-row"],
)
async def test_a_fault_gives_no_line_and_is_logged(
    monkeypatch: pytest.MonkeyPatch, head: Any
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        map_service.log, "exception", lambda event, **kw: calls.append(event), raising=False
    )
    _head(monkeypatch, head)
    assert await map_service.probability_line(_flow(), MISSION) is None
    assert calls == ["probability_map.situation_line_failed"]


async def test_a_row_stored_before_psr_had_a_bound_is_read_clamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _head(monkeypatch, {"grid": {"sectors": [{**SECTOR, "psr": 250.0}]}})
    [score] = await map_service.sector_scores(_flow_with_mission(), MISSION)
    assert score.psr == PSR_MAX
