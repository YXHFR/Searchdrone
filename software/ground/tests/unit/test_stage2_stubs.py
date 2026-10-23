"""B1-14 / B2-04: the candidate queue is built now (contract tests in
test_candidates.py): it no longer says it is not built, needs a login, and has no route
that confirms or rejects (only a person will, in a later ticket)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from sd_ground.api.panel import STAGE2_HEADER

MISSION = "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a21"


def test_candidates_are_no_stub(logged_in: TestClient) -> None:
    res = logged_in.get(f"/api/v1/missions/{MISSION}/candidates")
    assert res.status_code == 503 and res.json()["code"] == "DB_DOWN"  # no database here
    assert STAGE2_HEADER not in res.headers
    assert logged_in.get(f"/api/v1/candidates/{MISSION}").status_code == 503


def test_stubs_need_a_login(panel: TestClient) -> None:
    for what in ("candidates", "detections", "finds", f"detections/{MISSION}/crop"):
        assert panel.get(f"/api/v1/missions/{MISSION}/{what}").status_code == 401, what
    assert panel.get(f"/api/v1/candidates/{MISSION}").status_code == 401


def test_stubs_have_no_write_route(logged_in: TestClient) -> None:
    """No route can confirm or reject a find yet: only a person will, in stage 2."""
    h = {"X-SD-Client": "panel", "Content-Type": "application/json"}
    for path in (
        f"/api/v1/missions/{MISSION}/candidates",
        f"/api/v1/missions/{MISSION}/detections",
        f"/api/v1/missions/{MISSION}/candidates/{MISSION}/decision",
        f"/api/v1/candidates/{MISSION}",
        f"/api/v1/candidates/{MISSION}/decision",
    ):
        assert logged_in.post(path, json={}, headers=h).status_code in (404, 405)
