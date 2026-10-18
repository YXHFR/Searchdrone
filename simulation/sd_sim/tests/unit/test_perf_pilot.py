"""sd_sim.perf.pilot: the perf run's operator, against a fake panel API (no stack)."""

from __future__ import annotations

import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from sd_sim import fixture
from sd_sim.errors import InfraError, VehicleError
from sd_sim.perf import pilot as pilot_mod
from sd_sim.perf.pilot import (
    DEMO_REQUEST,
    DEMO_YES,
    Pilot,
    flight_done,
    is_brake,
    latencies_s,
    new_uuid7,
    newest_brain_command,
    not_brake,
    pin_latlon,
    remaining_s,
)

VID = "sim-01"


def fake_stack(tmp_path: Path) -> Any:
    return SimpleNamespace(
        ports={"ground_panel": 1},
        secrets_dir=tmp_path / "secrets",
        repo=tmp_path,
        cfg=SimpleNamespace(instance=0),
    )


def live(mode: str = "LOITER", **over: Any) -> dict[str, Any]:
    tel = {
        "flight_mode": mode,
        "armed": True,
        "landed_state": "in_air",
        "alt_rel_home_m": 40.0,
    }
    tel.update(over.pop("tel", {}))
    return {"telemetry": tel, "snapshot": {"preflight_ok": True}, "airborne": True, **over}


class FakePanel:
    """The panel API as a scripted demo: what the pilot asked for, and the replies."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None, dict[str, str] | None]] = []
        self.call_ns: list[int] = []
        self.state = live()
        self.status = "DRAFT"
        self.commands: list[dict[str, Any]] = []
        self.stop_status = 202
        self.chat: list[dict[str, Any]] = []
        self.card = True  # the AI's charter card is pending
        self.card_after_yes = False  # the card appears once the pilot answers "Ja"

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> tuple[int, Any]:
        self.call_ns.append(time.monotonic_ns())
        self.calls.append((method, path, body, params))
        if path == f"/api/v1/vehicles/{VID}/state":
            return 200, self.state
        if path == f"/api/v1/vehicles/{VID}/missions/open":
            return 200, {"id": "m1"}
        if path == "/api/v1/missions/m1/chat" and method == "GET":
            return 200, {"items": self.chat, "next_before": None}
        if path == "/api/v1/missions/m1/chat":
            if self.card_after_yes and body is not None and body["text"] == DEMO_YES:
                self.card = True
            return 201, {}
        if path == "/api/v1/missions/m1/annotations":
            return 201, {}
        if path == "/api/v1/missions/m1/approvals":
            card = [{"id": "a1", "kind": "charter", "subject_version": 3}] if self.card else []
            return 200, [{"id": "other", "kind": "command", "subject_version": 9}, *card]
        if path == "/api/v1/approvals/a1/decision":
            self.status = "READY"
            return 200, {}
        if path == "/api/v1/missions/m1":
            return 200, {"mission": {"status": self.status}}
        if path == "/api/v1/missions/m1/clearance":
            self.commands.append(
                {"skill": "takeoff", "issuer_kind": "brain", "status": "done", "created_at": "1"}
            )
            return 200, {}
        if path == "/api/v1/missions/m1/commands":
            return 200, self.commands
        if path.startswith(f"/api/v1/vehicles/{VID}/actions/"):
            return self.stop_status, {"action_id": "x"}
        return 404, {"detail": path}

    def posts(self) -> list[tuple[str, dict[str, Any] | None]]:
        return [(p, b) for m, p, b, _ in self.calls if m == "POST"]


def make(tmp_path: Path, panel: FakePanel | None = None) -> tuple[Pilot, FakePanel]:
    panel = panel or FakePanel()
    return Pilot(fake_stack(tmp_path), panel=panel), panel


# --- pure parts ---------------------------------------------------------------------------


def test_new_uuid7_is_version_7_and_time_ordered() -> None:
    a = new_uuid7()
    time.sleep(0.002)
    b = new_uuid7()
    assert a.version == 7 and a.variant == uuid.RFC_4122
    assert a.int >> 80 < b.int >> 80  # the millisecond prefix
    assert abs((a.int >> 80) - time.time_ns() // 1_000_000) < 5_000
    assert len({new_uuid7() for _ in range(100)}) == 100


def test_mode_predicates() -> None:
    assert is_brake("BRAKE") and not is_brake("GUIDED") and not is_brake(None)
    assert not_brake("GUIDED") and not not_brake("BRAKE")
    assert not not_brake(None)  # no telemetry is not "not braking"


def test_flight_done() -> None:
    on_ground = live(tel={"armed": False, "landed_state": "on_ground"}, airborne=False)
    assert flight_done(on_ground, None, seen_airborne=True)
    assert not flight_done(on_ground, None, seen_airborne=False)  # still before take-off
    assert not flight_done(live(), "ACTIVE", seen_airborne=True)
    assert flight_done(live(), "LANDED", seen_airborne=False)
    assert flight_done({}, "CLOSED", seen_airborne=False)
    assert not flight_done({}, None, seen_airborne=True)
    armed = live(tel={"armed": True, "landed_state": "on_ground"}, airborne=False)
    assert not flight_done(armed, None, seen_airborne=True)


def test_latencies_s() -> None:
    t = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        (t, t + timedelta(seconds=2.5)),
        (t, None),
        (None, t),
        (t, t + timedelta(milliseconds=800)),
    ]
    assert latencies_s(rows) == [2.5, 0.8]
    assert latencies_s([]) == []


def test_pin_latlon_is_metres_from_home() -> None:
    fx = fixture.load()
    lat, lon = pin_latlon(fx, "P1")  # 200 m north of home
    assert lon == pytest.approx(fx.home.lon)
    assert (lat - fx.home.lat) * 111_320 == pytest.approx(200.0, rel=1e-2)
    lat, lon = pin_latlon(fx, "OUT")  # 250 m east
    assert lat == pytest.approx(fx.home.lat)
    assert lon > fx.home.lon


def test_remaining_s() -> None:
    assert remaining_s(10.0, now=4.0) == 6.0
    assert remaining_s(10.0, now=12.0) == 0.0


def test_newest_brain_command_ignores_the_operator_and_other_skills() -> None:
    cmds = [
        {"skill": "takeoff", "issuer_kind": "brain", "created_at": "2026-01-01T00:00:01Z"},
        {"skill": "takeoff", "issuer_kind": "brain", "created_at": "2026-01-01T00:00:03Z"},
        {"skill": "takeoff", "issuer_kind": "operator", "created_at": "2026-01-01T00:00:09Z"},
        {"skill": "goto", "issuer_kind": "brain", "created_at": "2026-01-01T00:00:09Z"},
    ]
    found = newest_brain_command(cmds, "takeoff")
    assert found is not None and found["created_at"].endswith("03Z")
    assert newest_brain_command(cmds, "land") is None


def test_chat_latency_sql_reads_the_brains_replies() -> None:
    sql = " ".join(pilot_mod.CHAT_LATENCY_SQL.split())
    assert "r ON r.in_reply_to = m.id" in sql and "m.role = 'operator'" in sql
    assert "min(r.created_at) AS replied_at" in sql
    # Both kinds of reply: the demo's request is answered with a chat line, not an order.
    assert "FROM commands c WHERE c.issuer_kind = 'brain'" in sql
    assert "FROM chat_messages b WHERE b.role = 'brain'" in sql


# --- the pilot ----------------------------------------------------------------------------


def test_needs_the_pilot_token(tmp_path: Path) -> None:
    with pytest.raises(InfraError, match="test_rig=True"):
        Pilot(fake_stack(tmp_path))


def test_reads_the_pilot_token(tmp_path: Path) -> None:
    stack = fake_stack(tmp_path)
    stack.secrets_dir.mkdir()
    (stack.secrets_dir / "pilot_token").write_text("tok\n", encoding="utf-8")
    p = Pilot(stack)
    panel = cast(pilot_mod.HttpPanel, p.panel)
    assert panel.base_url == "http://127.0.0.1:1"


def test_press_stop_takes_the_time_before_the_post(tmp_path: Path) -> None:
    p, panel = make(tmp_path)
    t0 = p.press_stop()
    t1 = p.press_resume()
    (stop_path, stop_body), (resume_path, _) = panel.posts()
    assert stop_path == f"/api/v1/vehicles/{VID}/actions/stop"
    assert resume_path == f"/api/v1/vehicles/{VID}/actions/resume"
    assert stop_body is not None
    assert uuid.UUID(stop_body["client_action_id"]).version == 7
    assert t0 <= panel.call_ns[0] and t1 <= panel.call_ns[1] and t0 < t1
    panel.stop_status = 409
    with pytest.raises(VehicleError, match="STOP: HTTP 409"):
        p.press_stop()


def test_mode_and_airborne_read_the_state_without_the_watcher(tmp_path: Path) -> None:
    p, panel = make(tmp_path)
    assert p.mode() == "LOITER" and p.airborne()
    panel.state = live("BRAKE", airborne=False)
    assert p.mode() == "BRAKE" and not p.airborne()
    panel.state = {"telemetry": None, "airborne": False}
    assert p.mode() is None


def test_wait_mode(tmp_path: Path) -> None:
    p, panel = make(tmp_path)
    panel.state = live("BRAKE")
    assert p.wait_mode(is_brake, 1.0) == "BRAKE"
    with pytest.raises(VehicleError, match="flygläget"):
        p.wait_mode(not_brake, 0.2)


def test_fly_demo_until_airborne(tmp_path: Path) -> None:
    p, panel = make(tmp_path)
    try:
        assert p.fly_demo_until_airborne(5.0) == "m1"
    finally:
        p.close()
    posts = panel.posts()
    assert [path for path, _ in posts] == [
        f"/api/v1/vehicles/{VID}/missions/open",
        "/api/v1/missions/m1/annotations",
        "/api/v1/missions/m1/chat",
        "/api/v1/approvals/a1/decision",
        "/api/v1/missions/m1/clearance",
    ]
    pin = posts[1][1]
    assert pin is not None and pin["kind"] == "pin" and pin["label"] == "P1"
    lon, lat = pin["geometry"]["coordinates"]
    assert (lat, lon) == pytest.approx(pin_latlon(fixture.load(), "P1"))
    chat = posts[2][1]
    assert chat is not None and chat["text"] == DEMO_REQUEST
    assert uuid.UUID(chat["client_msg_id"]).version == 7
    assert posts[3][1] == {
        "decision": "approve",
        "subject_version": 3,
        "autonomy_mode": "WITHIN_CHARTER",
    }
    approvals = [c for c in panel.calls if c[1].endswith("/approvals")]
    assert approvals[0][3] == {"status": "pending"}


def test_a_refused_takeoff_fails_at_once(tmp_path: Path) -> None:
    p, panel = make(tmp_path)
    p.mid = "m1"
    panel.commands = [
        {
            "skill": "takeoff",
            "issuer_kind": "brain",
            "status": "rejected_by_gate",
            "status_reason": "X",
            "created_at": "1",
        }
    ]
    with pytest.raises(VehicleError, match="takeoff: rejected_by_gate X"):
        p.wait_command("takeoff", "done", 5.0)


def test_start_without_telemetry_stops_its_threads(tmp_path: Path) -> None:
    p, panel = make(tmp_path)
    panel.state = {"telemetry": None, "airborne": False}
    # As in ``with pilot:``, whose __exit__ never runs when __enter__ (start) raises.
    with pytest.raises(InfraError, match="Ingen telemetri"):
        p.start(telemetry_s=0.2)
    assert not p._threads  # pyright: ignore[reportPrivateUsage]
    assert not [t for t in threading.enumerate() if t.name.startswith("perf-")]
    # A later start() waits for telemetry again instead of returning at once.
    with pytest.raises(InfraError, match="Ingen telemetri"):
        p.start(telemetry_s=0.2)
    panel.state = live()
    try:
        assert p.start(telemetry_s=2.0) is p and p.tel["flight_mode"] == "LOITER"
    finally:
        p.close()


def test_wait_flight_done(tmp_path: Path) -> None:
    p, panel = make(tmp_path)
    p.mid = "m1"
    panel.status = "ACTIVE"
    try:
        p.start()
        p.refresh()  # airborne seen
        panel.state = live(tel={"armed": False, "landed_state": "on_ground"}, airborne=False)
        p.wait_flight_done(5.0)
    finally:
        p.close()
    assert not p._threads  # pyright: ignore[reportPrivateUsage]


def test_chat_latencies_uses_the_pilots_mission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str | None] = []

    def fake(stack: Any, mission_id: str | None = None) -> list[float]:
        seen.append(mission_id)
        return [1.0]

    monkeypatch.setattr(pilot_mod, "chat_latencies_s", fake)
    p, _ = make(tmp_path)
    assert p.chat_latencies_s() == [1.0]
    p.mid = "m1"
    assert p.chat_latencies_s(p.stack) == [1.0]
    assert seen == [None, "m1"]


def test_http_panel_sends_the_token_and_reads_errors() -> None:
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: list[tuple[str, str, str, str, bytes]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            seen.append(
                (
                    self.command,
                    self.path,
                    self.headers.get("Authorization") or "",
                    self.headers.get("X-SD-Client") or "",
                    self.rfile.read(n),
                )
            )
            code = 202 if self.path.endswith("/stop") else 409
            raw = json.dumps({"ok": code == 202}).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        panel = pilot_mod.HttpPanel(f"http://127.0.0.1:{server.server_port}", "tok", 5.0)
        assert panel.request("POST", "/a/stop", {"x": 1}) == (202, {"ok": True})
        assert panel.request("POST", "/a/resume", {}) == (409, {"ok": False})
    finally:
        server.shutdown()
        server.server_close()
    assert seen[0] == ("POST", "/a/stop", "Bearer tok", "panel", b'{"x": 1}')
    with pytest.raises(InfraError, match="sd-ground svarar inte"):
        pilot_mod.HttpPanel(f"http://127.0.0.1:{server.server_port}", "tok", 1.0).request(
            "GET", "/x"
        )


def test_the_pilot_answers_once_when_the_ai_asks_instead_of_proposing(tmp_path: Path) -> None:
    panel = FakePanel()
    panel.card, panel.card_after_yes = False, True
    panel.chat = [
        {"seq": 2, "role": "brain", "kind": "question", "text": "Vill du godkänna detta?"},
        {"seq": 1, "role": "operator", "kind": "text", "text": DEMO_REQUEST},
    ]
    p, _ = make(tmp_path, panel)
    try:
        assert p.fly_demo_until_airborne(5.0) == "m1"
    finally:
        p.close()
    chats = [b for path, b in panel.posts() if path == "/api/v1/missions/m1/chat"]
    assert [b["text"] for b in chats if b is not None] == [DEMO_REQUEST, DEMO_YES]


def test_the_pilot_does_not_answer_its_own_line(tmp_path: Path) -> None:
    panel = FakePanel()
    panel.chat = [{"seq": 1, "role": "operator", "kind": "text", "text": DEMO_REQUEST}]
    p, _ = make(tmp_path, panel)
    p.mid = "m1"
    assert not p.brain_asked()
    panel.chat.append({"seq": 2, "role": "brain", "kind": "text", "text": "Ok."})
    assert not p.brain_asked()
