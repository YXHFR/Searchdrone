"""B1-14: the stage 2 stubs answer empty, say they are not built, and never act."""

from __future__ import annotations

from fastapi.testclient import TestClient

from sd_ground.api.panel import STAGE2_HEADER, STAGE2_VALUE

MISSION = "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a21"


def test_candidates_are_an_empty_stub(logged_in: TestClient) -> None:
    res = logged_in.get(f"/api/v1/missions/{MISSION}/candidates")
    assert res.status_code == 200, res.text
    assert res.json() == {"items": []}
    assert res.headers[STAGE2_HEADER] == STAGE2_VALUE


def test_detections_are_an_empty_stub(logged_in: TestClient) -> None:
    res = logged_in.get(f"/api/v1/missions/{MISSION}/detections")
    assert res.status_code == 200, res.text
    assert res.json() == []
    assert res.headers[STAGE2_HEADER] == STAGE2_VALUE


def test_stubs_need_a_login(panel: TestClient) -> None:
    for what in ("candidates", "detections"):
        assert panel.get(f"/api/v1/missions/{MISSION}/{what}").status_code == 401


def test_stubs_have_no_write_route(logged_in: TestClient) -> None:
    """No route can confirm or reject a find yet: only a person will, in stage 2."""
    h = {"X-SD-Client": "panel", "Content-Type": "application/json"}
    for path in (
        f"/api/v1/missions/{MISSION}/candidates",
        f"/api/v1/missions/{MISSION}/detections",
        f"/api/v1/missions/{MISSION}/candidates/{MISSION}/decision",
    ):
        assert logged_in.post(path, json={}, headers=h).status_code in (404, 405)
