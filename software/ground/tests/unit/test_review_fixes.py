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
from sd_obs.clock import FakeClock

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
