"""Regression tests for the review findings on the first version of the server."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from sd_ground.app import create_panel_app, panel_asgi
from sd_ground.auth import SessionStore
from sd_ground.config import check, load
from sd_ground.context import Ground
from sd_ground.sources import NoEmergencyPath, NoVehicleLink
from sd_testing import FakeClock

H = {"X-SD-Client": "panel", "Content-Type": "application/json"}
ACT = {"client_action_id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a11"}


def test_cookie_has_no_fixed_lifetime(panel: TestClient, ground: Ground) -> None:
    code, _ = ground.login_link()
    cookie = panel.post("/api/v1/session", json={"login_code": code}, headers=H).headers[
        "set-cookie"
    ]
    assert "max-age" not in cookie.lower() and "expires" not in cookie.lower()


def test_logout_is_not_undone_by_a_database_reload(clock: FakeClock) -> None:
    store = SessionStore(clock)
    token, rec = store.create_session(role="pilot", issuer_ref="operator:demo", auth_mode="local")
    store.revoke(rec.token_sha256)
    store.load([rec])  # the database still lists it (the revoke was not saved yet)
    assert store.lookup(token) is None
    assert store.unsaved_revocations() == [rec.token_sha256]
    store.revocation_saved(rec.token_sha256)
    assert store.unsaved_revocations() == []


def test_stop_all_without_vehicles_is_never_success(make_ground: Callable[..., Ground]) -> None:
    g = make_ground()
    g.emergency = NoEmergencyPath()
    with TestClient(panel_asgi(g, create_panel_app(g)), base_url="http://127.0.0.1:8080") as c:  # pyright: ignore[reportArgumentType]
        code, _ = g.login_link()
        c.post("/api/v1/session", json={"login_code": code}, headers=H)
        for path in ("/api/v1/actions/stop_all", "/api/v1/actions/return_home_all"):
            res = c.post(path, json=ACT, headers=H)
            assert res.status_code == 503, path
            assert res.json()["code"] == "NO_PATH"


def test_in_flight_extension_follows_a_swapped_link(ground: Ground, vehicle_link: object) -> None:
    ground.vehicle_link = NoVehicleLink()
    ground.vehicle_link = vehicle_link  # type: ignore[assignment]
    vehicle_link.flying_since = ground.now()  # type: ignore[attr-defined]
    token, _ = ground.sessions.create_session(
        role="pilot", issuer_ref="operator:demo", auth_mode="local"
    )
    ground.clock.advance(13 * 3600)  # type: ignore[attr-defined]
    assert ground.sessions.lookup(token) is not None


def test_bad_limits_file_stops_the_server(tmp_path: Path) -> None:
    (tmp_path / "config" / "local").mkdir(parents=True)
    (tmp_path / "config" / "local" / "limits.toml").write_text("max_alt_agl_m = 50\n")
    assert any("limits.toml" in e for e in check(load(tmp_path, {})))


def test_observer_may_stop_on_the_main_app_too(ground: Ground, clock: FakeClock) -> None:
    """The documented routes use the same rule as the ingress thread."""
    from sd_ground.auth import SessionRecord, hash_token

    now = clock.utc_now()
    ground.sessions.load(
        [
            SessionRecord(
                id="o",
                kind="token",
                role="observer",
                issuer_ref="observer:script",
                auth_mode="cli",
                token_sha256=hash_token("observer-token-xyz"),
                created_at=now,
                last_used_at=now,
                expires_at=now + dt.timedelta(hours=1),
                name="script",
            )
        ]
    )
    auth = {"Authorization": "Bearer observer-token-xyz"}
    with TestClient(create_panel_app(ground), base_url="http://127.0.0.1:8080") as c:
        assert (
            c.post("/api/v1/vehicles/sim-01/actions/stop", json=ACT, headers=auth).status_code
            == 202
        )
        assert (
            c.post(
                "/api/v1/vehicles/sim-01/actions/return_home", json=ACT, headers=auth
            ).status_code
            == 403
        )


def test_stop_all_reaches_every_vehicle_even_if_one_fails(clock: FakeClock) -> None:
    import pytest

    from sd_ground import emergency
    from sd_ground.auth import Principal
    from sd_ground.sources import EmergencyUnavailableError
    from sd_schemas.api import ActionDispatched
    from sd_schemas.base import new_id

    calls: list[str] = []

    class TwoVehicles:
        def vehicles(self) -> list[str]:
            return ["sim-01", "sim-02"]

        def dispatch(self, action: str, vehicle_id: str, **_: object) -> ActionDispatched:
            calls.append(vehicle_id)
            if vehicle_id == "sim-01":
                raise EmergencyUnavailableError(503, "NO_PATH", "nere")
            return ActionDispatched(action_id=new_id(), action_seq=1, paths={"mavlink": "queued"})

    who = Principal(role="pilot", issuer_ref="operator:demo", auth_mode="local")
    with pytest.raises(EmergencyUnavailableError) as exc:
        emergency.dispatch_all(TwoVehicles(), "stop", str(new_id()), who)  # type: ignore[arg-type]
    assert calls == ["sim-01", "sim-02"]  # the second vehicle still got STOP
    assert "sim-01" in exc.value.detail_sv and "sim-02" in exc.value.detail_sv


def test_live_ws_closes_after_logout(
    logged_in: TestClient, ground: Ground, clock: FakeClock
) -> None:
    import pytest
    from starlette.websockets import WebSocketDisconnect

    with logged_in.websocket_connect("ws://127.0.0.1:8080/ws/v1/live") as ws:
        ws.receive_json()
        ws.send_json(
            {"op": "subscribe", "topics": ["mission"], "server_epoch": None, "since_seq": None}
        )
        ws.receive_json()
        for rec in ground.sessions.all():
            ground.sessions.revoke(rec.token_sha256)
        with pytest.raises(WebSocketDisconnect) as closed:
            for _ in range(200):
                clock.advance(1.1)
                ws.receive_json()
        assert closed.value.code == 4401
