"""Login, PIN pairing and sessions (B1-05 points 5 and 5b, 04 §5.2, T26, T41)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

from fastapi.testclient import TestClient

from sd_ground.app import create_panel_app, panel_asgi
from sd_ground.auth import (
    ATTEMPT_LIMIT,
    LoginError,
    PendingLogins,
    SessionStore,
    parse_duration,
    token_expiry,
)
from sd_ground.context import Ground
from sd_testing import FakeClock

H = {"X-SD-Client": "panel", "Content-Type": "application/json"}
STOP = {"client_action_id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a11"}


def _client(g: Ground, ip: str) -> TestClient:
    return TestClient(
        panel_asgi(g, create_panel_app(g)), base_url="http://127.0.0.1:8080", client=(ip, 40000)
    )  # pyright: ignore[reportArgumentType]


def _raises(fn: Callable[[], object]) -> LoginError:
    try:
        fn()
    except LoginError as exc:
        return exc
    raise AssertionError("expected LoginError")


def test_login_code_expires_after_ttl(clock: FakeClock) -> None:
    logins = PendingLogins(clock, code_ttl_s=600)
    code = logins.create_login_code()
    clock.advance(601)
    assert _raises(lambda: logins.redeem("a", login_code=code, pin=None)).code == "LOGIN_FAILED"


def test_pin_and_code_belong_together(clock: FakeClock) -> None:
    logins = PendingLogins(clock)
    code, pin = logins.create_pair()
    assert logins.redeem("a", login_code=None, pin=pin) == "pin"
    # using the PIN used up the pair's code as well
    assert _raises(lambda: logins.redeem("a", login_code=code, pin=None)).status == 401


def test_pin_global_lock_from_six_addresses(clock: FakeClock) -> None:
    locked: list[bool] = []
    logins = PendingLogins(clock, on_pin_locked=lambda: locked.append(True))
    code, pin = logins.create_pair()
    wrong = "000000" if pin != "000000" else "111111"
    for i in range(6):
        _raises(lambda i=i: logins.redeem(f"10.0.0.{i}", login_code=None, pin=wrong))
    assert locked == [True]
    err = _raises(lambda: logins.redeem("10.0.0.99", login_code=None, pin=pin))
    assert (err.status, err.code) == (401, "PIN_INVALIDATED")
    assert _raises(lambda: logins.redeem("10.0.0.99", login_code=code, pin=None)).status == 401
    _new_code, new_pin = logins.create_pair()
    assert logins.redeem("10.0.0.99", login_code=None, pin=new_pin) == "pin"


def test_attempt_window_is_per_address(clock: FakeClock) -> None:
    logins = PendingLogins(clock)
    for _ in range(ATTEMPT_LIMIT):
        _raises(lambda: logins.redeem("10.0.0.1", login_code="bad", pin=None))
    assert _raises(lambda: logins.redeem("10.0.0.1", login_code="bad", pin=None)).status == 429
    good = logins.create_login_code()
    assert logins.redeem("10.0.0.2", login_code=good, pin=None) == "local"
    clock.advance(301)
    assert _raises(lambda: logins.redeem("10.0.0.1", login_code="bad", pin=None)).status == 401


def test_pin_lockout_does_not_affect_existing_session(
    ground: Ground, logged_in: TestClient, emergency: object
) -> None:
    _code, pin, _url = ground.pair()
    wrong = "000000" if pin != "000000" else "111111"
    with _client(ground, "192.168.1.50") as other:
        statuses = [
            other.post("/api/v1/session", json={"pin": wrong}, headers=H).status_code
            for _ in range(6)
        ]
    assert statuses[-1] == 429
    res = logged_in.post("/api/v1/vehicles/sim-01/actions/stop", json=STOP, headers=H)
    assert res.status_code == 202, res.text
    assert emergency.calls == [("stop", "sim-01", "operator:demo")]  # type: ignore[attr-defined]
    with logged_in.websocket_connect("ws://127.0.0.1:8080/ws/v1/live") as ws:
        assert ws.receive_json()["type"] == "hello"


def test_session_idle_and_cap(clock: FakeClock) -> None:
    store = SessionStore(clock)
    token, _ = store.create_session(role="pilot", issuer_ref="operator:demo", auth_mode="local")
    clock.advance(11 * 3600)
    assert store.lookup(token) is not None  # use moves the 12 h idle limit
    clock.advance(11 * 3600)
    assert store.lookup(token) is not None
    clock.advance(12 * 3600 + 1)
    assert store.lookup(token) is None
    token2, _ = store.create_session(role="pilot", issuer_ref="operator:demo", auth_mode="local")
    for _ in range(16):  # used every 11 h for a week
        clock.advance(11 * 3600)
        store.lookup(token2)
    assert store.lookup(token2) is None  # 7-day cap


def test_session_extended_in_flight(clock: FakeClock) -> None:
    since: list[dt.datetime | None] = [None]
    extended: list[str] = []
    store = SessionStore(
        clock, airborne_since=lambda: since[0], on_extended=lambda r: extended.append(r.id)
    )
    token, rec = store.create_session(role="pilot", issuer_ref="operator:demo", auth_mode="local")
    for _ in range(167 // 10):  # in use every 10 h until 1 h before the 7-day cap
        clock.advance(10 * 3600)
        store.lookup(token)
    clock.advance((167 - 160) * 3600)
    store.lookup(token)
    since[0] = clock.utc_now()  # takes off 1 h before the 7-day cap
    clock.advance(5 * 3600)
    assert store.lookup(token) is not None  # past the cap, but airborne
    assert store.expires_at(rec) is None
    clock.advance(20 * 3600)
    assert store.lookup(token) is not None  # past 12 h idle too
    assert extended == [rec.id]
    since[0] = None  # landed and disarmed
    assert store.lookup(token) is None


def test_session_expired_before_flight_stays_expired(clock: FakeClock) -> None:
    since: list[dt.datetime | None] = [None]
    store = SessionStore(clock, airborne_since=lambda: since[0])
    token, _ = store.create_session(role="pilot", issuer_ref="operator:demo", auth_mode="local")
    clock.advance(13 * 3600)
    since[0] = clock.utc_now()
    assert store.lookup(token) is None


def test_session_expiring_blocks_takeoff(
    ground: Ground, logged_in: TestClient, clock: FakeClock
) -> None:
    body = {
        "client_command_id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a12",
        "skill": "takeoff",
        "params": {"alt_m": 40},
    }
    mission = "0192f49e-0b3a-7c55-8e2f-1a2b3c4d5e6f"
    for _ in range(16):  # in use every 10 h: 160 h of the 168 h cap
        clock.advance(10 * 3600)
        assert logged_in.get("/api/v1/session").status_code == 200
    ok = logged_in.post(f"/api/v1/missions/{mission}/commands", json=body, headers=H)
    assert ok.status_code == 501  # 8 h left: passes the session check (the order flow is B1-12)
    clock.advance(6.5 * 3600)
    assert ground.sessions.headroom(ground.sessions.all()[0]) < dt.timedelta(hours=2)
    res = logged_in.post(f"/api/v1/missions/{mission}/commands", json=body, headers=H)
    assert res.status_code == 409
    assert res.json() == {
        **res.json(),
        "code": "SESSION_EXPIRING",
        "detail_sv": "Logga in igen innan start.",
    }
    approval = logged_in.post(
        "/api/v1/approvals/0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a13/decision",
        json={"decision": "approve", "subject_version": 1},
        headers=H,
    )
    assert approval.json()["code"] == "SESSION_EXPIRING"


def test_cli_token_not_extended(clock: FakeClock) -> None:
    from sd_ground.auth import SessionRecord, hash_token

    now = clock.utc_now()
    store = SessionStore(clock, airborne_since=lambda: now)
    rec = SessionRecord(
        id="t",
        kind="token",
        role="pilot",
        issuer_ref="operator:ci",
        auth_mode="cli",
        token_sha256=hash_token("tok"),
        created_at=now,
        last_used_at=now,
        expires_at=now + dt.timedelta(hours=1),
        name="ci",
    )
    store.load([rec])
    clock.advance(3601)
    assert store.lookup("tok") is None


def test_token_expiry_rules() -> None:
    now = dt.datetime(2026, 10, 15, tzinfo=dt.UTC)
    assert token_expiry(now, None) == now + dt.timedelta(hours=24)
    assert token_expiry(now, parse_duration("30d")) == now + dt.timedelta(days=30)
    for bad in ("31d", "0h"):
        try:
            token_expiry(now, parse_duration(bad))
        except ValueError:
            continue
        raise AssertionError(bad)


def test_logout_ends_session(logged_in: TestClient) -> None:
    assert logged_in.get("/api/v1/session").status_code == 200
    assert logged_in.post("/api/v1/session/logout", json={}, headers=H).status_code == 204
    logged_in.cookies.clear()
    assert logged_in.get("/api/v1/session").status_code == 401


def test_observer_cannot_change(ground: Ground, panel: TestClient, clock: FakeClock) -> None:
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
    assert panel.get("/api/v1/system/status", headers=auth).status_code == 200
    res = panel.post("/api/v1/vehicles/sim-01/actions/stop", json=STOP, headers={**H, **auth})
    assert res.status_code == 202  # every logged-in user may press STOP
    home = panel.post(
        "/api/v1/vehicles/sim-01/actions/return_home", json=STOP, headers={**H, **auth}
    )
    assert home.status_code == 403  # RETURN HOME is for the pilot only
    home_all = panel.post("/api/v1/actions/return_home_all", json=STOP, headers={**H, **auth})
    assert home_all.status_code == 403
    assert (
        panel.post("/api/v1/actions/stop_all", json=STOP, headers={**H, **auth}).status_code == 202
    )
    res = panel.put(
        "/api/v1/missions/0192f49e-0b3a-7c55-8e2f-1a2b3c4d5e6f/ai",
        json={"enabled": False},
        headers={**H, **auth},
    )
    assert res.status_code == 403


def test_interactive_only_with_cli_token(panel: TestClient, pilot_token: str) -> None:
    res = panel.post(
        "/api/v1/missions/0192f49e-0b3a-7c55-8e2f-1a2b3c4d5e6f/clearance",
        json={},
        headers={**H, "Authorization": f"Bearer {pilot_token}"},
    )
    assert res.status_code == 403
    assert res.json()["code"] == "INTERACTIVE_SESSION_REQUIRED"
