"""``/health`` and ``ready`` with faked links (B1-05 point 1)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from sd_ground.context import Ground


def test_health_answers_without_login(panel: TestClient) -> None:
    res = panel.get("/api/v1/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert set(body["parts"]) == {"db", "mavlink", "sdlink", "onboard", "brain", "mediamtx"}
    assert body["test_rig"] is False


def test_ready_needs_db_mavlink_and_sdlink(
    panel: TestClient, ground: Ground, vehicle_link: object
) -> None:
    ground.db_ok = False
    assert panel.get("/api/v1/health").json()["ready"] is False
    ground.db_ok = True
    assert panel.get("/api/v1/health").json()["ready"] is True  # fakes report mavlink and sdlink ok
    vehicle_link.mav = "down"  # type: ignore[attr-defined]
    body = panel.get("/api/v1/health").json()
    assert body["ready"] is False
    assert body["parts"]["mavlink"] == "down"
    vehicle_link.mav = "ok"  # type: ignore[attr-defined]
    ground.sdlink.link = "degraded"  # type: ignore[attr-defined]
    assert panel.get("/api/v1/health").json()["ready"] is False


def test_brain_is_not_needed_for_ready(panel: TestClient, ground: Ground) -> None:
    ground.db_ok = True
    body = panel.get("/api/v1/health").json()
    assert body["parts"]["brain"] == "down"
    assert body["ready"] is True


def test_test_rig_is_visible(make_ground: object) -> None:
    from sd_ground.app import create_panel_app, panel_asgi

    g = make_ground(auth={"cli_may_confirm": True})  # type: ignore[operator]
    with TestClient(panel_asgi(g, create_panel_app(g)), base_url="http://127.0.0.1:8080") as c:  # pyright: ignore[reportArgumentType]
        assert c.get("/api/v1/health").json()["test_rig"] is True


def test_db_down_routes_say_503(logged_in: TestClient, ground: Ground) -> None:
    ground.db_ok = False
    res = logged_in.get("/api/v1/vehicles")
    assert res.status_code == 503
    assert res.json()["code"] == "DB_DOWN"


def test_emergency_without_path_is_503_not_silent(make_ground: object) -> None:
    from sd_ground.app import create_panel_app, panel_asgi
    from sd_ground.sources import NoEmergencyPath

    g = make_ground()  # type: ignore[operator]
    g.emergency = NoEmergencyPath()
    with TestClient(panel_asgi(g, create_panel_app(g)), base_url="http://127.0.0.1:8080") as c:  # pyright: ignore[reportArgumentType]
        code, _ = g.login_link()
        h = {"X-SD-Client": "panel", "Content-Type": "application/json"}
        c.post("/api/v1/session", json={"login_code": code}, headers=h)
        res = c.post(
            "/api/v1/vehicles/sim-01/actions/stop",
            json={"client_action_id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a11"},
            headers=h,
        )
        assert res.status_code == 503
        assert res.json()["code"] == "NO_PATH"


def test_panel_config(logged_in: TestClient) -> None:
    body = logged_in.get("/api/v1/panel/config").json()
    assert body["default_vehicle_id"] == "sim-01"
    assert body["remote"] is False
    assert body["limits_default"]["max_alt_agl_m"] == 120
    assert body["limits_absolute"]["max_alt_agl_m"] == 300  # outer cap without a local file
    assert body["video"]["whep_url"] == "/api/v1/video/sim-01/main/whep"
