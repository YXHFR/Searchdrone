"""E7: the search service has no path to an order, an approval, STOP / RETURN HOME or a
decision on a candidate. It talks only to ``/search/v1`` and imports nothing that steers."""

from __future__ import annotations

import ast
import datetime as dt
import json
from pathlib import Path

import httpx
import pytest
from search_helpers import MISSION  # pyright: ignore[reportMissingImports]

import sd_search
from sd_schemas.search import CandidateBatch, CandidateRevision, SearchHeartbeat
from sd_search.client import GroundClient

SRC = Path(sd_search.__file__).parent
FORBIDDEN_ROOTS = {
    "sd_ground", "sd_safety", "sd_mav", "sd_link", "sd_onboard", "sd_sim", "pymavlink",
    "serial", "mavsdk", "dronekit", "MAVProxy", "sqlalchemy", "asyncpg", "psycopg",
    "subprocess", "socket", "websockets",
}  # fmt: skip
CID = "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a22"
T = dt.datetime(2026, 10, 1, tzinfo=dt.UTC)


def _revision(**kw: object) -> dict[str, object]:
    return {
        "candidate_id": CID,
        "base_rev": 0,
        "state": "new",
        "lat": 0.0,
        "lon": 0.0,
        "sigma95_m": 3.0,
        "n_seen": 1,
        "best_detector_score": 0.5,
        "priority": 0.4,
        "first_seen_utc": T,
        "last_seen_utc": T,
    } | kw


async def test_search_has_no_command_path() -> None:
    seen: list[tuple[str, str]] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        assert request.headers["authorization"] == "Bearer t"
        body: object = []
        if request.url.path.endswith("/search-profile"):
            return httpx.Response(404, json={"code": "NOT_FOUND"})
        if request.url.path.endswith("/crop"):
            return httpx.Response(404)
        if request.method == "POST" and request.url.path.endswith("/candidates"):
            body = [{"candidate_id": CID, "rev": 1, "seq": 1, "state": "new"}]
        if request.url.path.endswith("/heartbeat"):
            return httpx.Response(204)
        return httpx.Response(200, content=json.dumps(body))

    client = GroundClient("http://127.0.0.1:7700", "t", transport=httpx.MockTransport(answer))
    await client.heartbeat(SearchHeartbeat())
    await client.missions()
    await client.detections(MISSION, 0, 10)
    await client.crop(CID)
    await client.tracks(MISSION)
    await client.write(
        MISSION, CandidateBatch(revisions=[CandidateRevision.model_validate(_revision())])
    )
    await client.profile(MISSION)
    await client.aclose()
    assert len(seen) == 7
    assert all(path.startswith("/search/v1/") for _, path in seen), seen
    for word in ("command", "approval", "decision", "actions", "commander", "stop", "return_home"):
        assert not any(word in path for _, path in seen), word

    # The only writes are the heartbeat and candidates; neither can decide.
    assert {m for m, _ in seen} == {"GET", "POST"}
    assert {p for m, p in seen if m == "POST"} == {
        "/search/v1/heartbeat",
        f"/search/v1/missions/{MISSION}/candidates",
    }
    for state in ("confirmed", "rejected", "inspecting"):
        with pytest.raises(ValueError, match="state"):
            CandidateRevision.model_validate(_revision(state=state))


def test_search_imports_nothing_that_steers() -> None:
    for path in sorted(SRC.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] not in FORBIDDEN_ROOTS, (path.name, name)
