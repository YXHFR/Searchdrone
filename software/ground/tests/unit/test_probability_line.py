"""The AI's probability-map situation line, and its fallback when the map cannot be read."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from sd_ground.db.probmap import ProbabilityMapRepo
from sd_ground.services import probability_map as map_service

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
        {"grid": {"sectors": [{**SECTOR, "psr": 1e9}]}},  # a stored row the schema refuses
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
