"""The ground server against a real database (B1-05 points 1-5b, 7, 9).

Each test gets its own migrated ``searchdrone_test_*`` database.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from sqlalchemy import URL

from sd_ground.api.commander import clean_text
from sd_ground.app import (
    Background,
    commander_asgi,
    create_commander_app,
    create_panel_app,
    panel_asgi,
)
from sd_ground.auth import ServiceToken
from sd_ground.config import GroundConfig, Loaded
from sd_ground.context import Ground
from sd_ground.db.engine import make_engine
from sd_ground.db.repo import Repo
from sd_ground.services.telemetry import TelemetryService
from sd_schemas.api import Telemetry
from sd_schemas.mission import MissionState
from sd_testing import FakeClock

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]

ROOT = Path(__file__).resolve().parents[4]
BRAIN = "brain-token-for-db-tests-0123456789abcd"
H = {"X-SD-Client": "panel", "Content-Type": "application/json"}
CMAC = (-35.363261, 149.165230)
VEHICLE_BOUND = {
    "missions",
    "charters",
    "mission_sectors",
    "commands",
    "approvals",
    "gate_decisions",
    "clearances",
    "operator_latches",
    "telemetry",
    "chat_messages",
    "map_annotations",
    "events",
    "brain_decisions",
}


def _app_url(db: Db) -> URL:
    return URL.create(
        "postgresql+asyncpg",
        username="sd_app",
        password=db.app_password,
        host="127.0.0.1",
        port=5433,
        database=db.name,
    )


def _owner_url(db: Db) -> str:
    return f"postgresql+asyncpg://sd_owner@127.0.0.1:5433/{db.name}"


def _mission(db: Db, status: str = "DRAFT", vehicle: str = "sim-01", sysid: int = 1) -> str:
    db.run(
        "INSERT INTO vehicles (id, name, kind, mav_sysid) VALUES ($1, $1, 'sim', $2) "
        "ON CONFLICT DO NOTHING",
        vehicle,
        sysid,
    )
    paused = "operator" if status == "PAUSED" else None
    returning = "brain" if status == "RETURNING" else None
    rows = db.run(
        "INSERT INTO missions (vehicle_id, title, status, limits, created_by, data_dir, "
        "paused_by, returning_cause) VALUES ($1, 'Uppdrag test', $2, '{}', 'operator:demo', "
        "'missions/x', $3, $4) RETURNING id",
        vehicle,
        status,
        paused,
        returning,
    )
    return str(rows[0]["id"])


async def _ground(db: Db, tmp_path: Path, clock: Any = None, **sections: Any) -> Ground:
    loaded = Loaded(
        config=GroundConfig.model_validate({"data_dir": str(tmp_path / "data"), **sections}),
        root=tmp_path,
        env={},
    )
    g = Ground(
        loaded=loaded, clock=clock or FakeClock(), brain_token=ServiceToken(BRAIN), version="0.0.0"
    )
    g.repo = Repo(make_engine(_app_url(db)))
    g.db_ok = True
    return g


async def _q(db: Db, sql: str, *args: Any) -> list[Any]:
    """``Db.run`` uses its own event loop, so async tests call it in a thread."""
    return await asyncio.to_thread(db.run, sql, *args)


@contextlib.asynccontextmanager
async def _clients(g: Ground) -> AsyncIterator[tuple[httpx.AsyncClient, httpx.AsyncClient]]:
    panel = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=panel_asgi(g, create_panel_app(g))),  # pyright: ignore[reportArgumentType]
        base_url="http://127.0.0.1:8080",
    )
    commander = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=commander_asgi(g, create_commander_app(g))),  # pyright: ignore[reportArgumentType]
        base_url="http://127.0.0.1:7700",
        headers={"Authorization": f"Bearer {BRAIN}"},
    )
    try:
        yield panel, commander
    finally:
        await panel.aclose()
        await commander.aclose()
        if g.repo is not None:
            await g.repo.close()


async def _login(g: Ground, panel: httpx.AsyncClient) -> None:
    code, _ = g.login_link()
    res = await panel.post("/api/v1/session", json={"login_code": code}, headers=H)
    assert res.status_code == 200, res.text
    assert panel.cookies["sd_session"]  # httpx keeps the cookie like a browser


# --- schema ------------------------------------------------------------------------


def test_vehicle_bound_tables_have_vehicle_id(migrated_db: Db) -> None:
    rows = migrated_db.run(
        "SELECT table_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND column_name = 'vehicle_id'"
    )
    assert {r["table_name"] for r in rows} >= VEHICLE_BOUND


def test_mission_status_values_match_mission_state(migrated_db: Db) -> None:
    for state in MissionState:
        _mission(migrated_db, state.value, vehicle="sim-02", sysid=2)
        migrated_db.run(
            "UPDATE missions SET status = 'CLOSED', paused_by = NULL, returning_cause = NULL "
            "WHERE vehicle_id = 'sim-02'"
        )
    with pytest.raises(Exception):  # noqa: B017 - a CHECK violation
        _mission(migrated_db, "ENDED", vehicle="sim-02", sysid=2)


# --- sessions ----------------------------------------------------------------------


async def test_session_survives_restart(migrated_db: Db, tmp_path: Path) -> None:
    g = await _ground(migrated_db, tmp_path)
    async with _clients(g) as (panel, _):
        await _login(g, panel)
        cookie = panel.cookies["sd_session"]
    rows = await _q(
        migrated_db, "SELECT kind, role, issuer_ref, auth_mode, token_sha256 FROM auth_sessions"
    )
    assert [(r["kind"], r["role"], r["issuer_ref"], r["auth_mode"]) for r in rows] == [
        ("session", "pilot", "operator:demo", "local")
    ]
    assert cookie not in json.dumps([dict(r) for r in rows])  # only the hash is stored
    g2 = await _ground(migrated_db, tmp_path, clock=g.clock)
    assert g2.repo is not None
    g2.sessions.load(await g2.repo.sessions())
    async with _clients(g2) as (panel2, _):
        panel2.cookies.set("sd_session", cookie)
        assert (await panel2.get("/api/v1/session")).status_code == 200


async def test_login_while_db_down_is_saved_later(migrated_db: Db, tmp_path: Path) -> None:
    g = await _ground(migrated_db, tmp_path)
    g.db_ok = False
    async with _clients(g) as (panel, _):
        await _login(g, panel)  # the session lives in memory
        assert (await _q(migrated_db, "SELECT count(*) AS n FROM auth_sessions"))[0]["n"] == 0
        await Background(g).db_tick()  # the database answers again
        assert g.db_ok
        assert (await _q(migrated_db, "SELECT count(*) AS n FROM auth_sessions"))[0]["n"] == 1
        assert (await panel.get("/api/v1/session")).status_code == 200


async def test_login_codes_never_in_database(migrated_db: Db, tmp_path: Path) -> None:
    g = await _ground(migrated_db, tmp_path)
    code, _ = g.login_link()
    _c, pin, _u = g.pair()
    async with _clients(g):
        pass
    dump = json.dumps(
        [dict(r) for r in await _q(migrated_db, "SELECT * FROM auth_sessions")], default=str
    )
    dump += json.dumps(
        [dict(r) for r in await _q(migrated_db, "SELECT * FROM events")], default=str
    )
    assert code not in dump and pin not in dump


# --- telemetry ---------------------------------------------------------------------


def _telemetry(t: dt.datetime) -> Telemetry:
    return Telemetry(
        t=t,
        time_boot_ms=1,
        lat=CMAC[0],
        lon=CMAC[1],
        alt_amsl_m=624.0,
        alt_rel_home_m=40.0,
        heading_deg=10.0,
        groundspeed_ms=5.0,
        climb_ms=0.0,
        battery_pct=80.0,
        battery_v=15.9,
        flight_mode="GUIDED",
        armed=True,
        landed_state="in_air",
        gps_fix=3,
        gps_sats=14,
        home=None,
        dist_to_pilot_m=30.0,
    )


class _Link:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock

    def state(self) -> str:
        return "ok"

    def connected(self, vehicle_id: str) -> bool:
        return True

    def telemetry(self, vehicle_id: str) -> Telemetry:
        return _telemetry(self.clock.utc_now())

    def in_flight(self, vehicle_id: str) -> bool:
        return True

    def airborne_since(self) -> dt.datetime:
        return self.clock.utc_now()

    def onboard_state(self) -> str:
        return "ok"


async def test_telemetry_written_at_2hz(migrated_db: Db, tmp_path: Path) -> None:
    mission = await asyncio.to_thread(_mission, migrated_db, "ACTIVE")
    clock = FakeClock()
    g = await _ground(migrated_db, tmp_path, clock=clock)
    g.vehicle_link = _Link(clock)  # type: ignore[assignment]
    service = TelemetryService(g)
    task = asyncio.create_task(service.run())
    try:
        for _ in range(400):  # 20 simulated seconds in 50 ms steps
            clock.advance(0.05)
            await asyncio.sleep(0.003)
        await asyncio.sleep(0.2)
        await service.flush(force=True)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert g.repo is not None
        await g.repo.close()
    rows = await _q(
        migrated_db,
        "SELECT count(*) AS n, min(t) AS a, max(t) AS b FROM telemetry WHERE mission_id = $1::uuid",
        mission,
    )
    assert 38 <= rows[0]["n"] <= 42
    pos = (
        await _q(
            migrated_db,
            "SELECT ST_X(pos) AS lon, ST_Y(pos) AS lat, ST_Z(pos) AS z FROM telemetry LIMIT 1",
        )
    )[0]
    assert (round(pos["lat"], 6), round(pos["lon"], 6), pos["z"]) == (CMAC[0], CMAC[1], 624.0)


async def test_no_telemetry_rows_before_active(migrated_db: Db, tmp_path: Path) -> None:
    await asyncio.to_thread(_mission, migrated_db, "READY")
    clock = FakeClock()
    g = await _ground(migrated_db, tmp_path, clock=clock)
    g.vehicle_link = _Link(clock)  # type: ignore[assignment]
    service = TelemetryService(g)
    await service.refresh_missions()
    for _ in range(20):
        clock.advance(0.5)
        service.sample()
    await service.flush(force=True)
    assert g.repo is not None
    await g.repo.close()
    assert (await _q(migrated_db, "SELECT count(*) AS n FROM telemetry"))[0]["n"] == 0


async def test_telemetry_buffers_while_db_down(migrated_db: Db, tmp_path: Path) -> None:
    mission = await asyncio.to_thread(_mission, migrated_db, "ACTIVE")
    clock = FakeClock()
    g = await _ground(migrated_db, tmp_path, clock=clock)
    g.vehicle_link = _Link(clock)  # type: ignore[assignment]
    service = TelemetryService(g)
    await service.refresh_missions()
    good = g.repo
    g.repo = Repo(
        make_engine(
            URL.create(
                "postgresql+asyncpg",
                username="sd_app",
                password="x",
                host="127.0.0.1",
                port=1,
                database="none",
            )
        )
    )
    for _ in range(30):
        clock.advance(0.5)
        service.sample()
        await service.flush()
    assert len(service.buffer) == 30
    await g.repo.close()
    g.repo = good
    await service.flush()
    assert good is not None
    await good.close()
    assert (
        await _q(
            migrated_db, "SELECT count(*) AS n FROM telemetry WHERE mission_id = $1::uuid", mission
        )
    )[0]["n"] == 30


# --- panel and commander routes ----------------------------------------------------


async def test_mission_routes(migrated_db: Db, tmp_path: Path) -> None:
    mission = await asyncio.to_thread(_mission, migrated_db)
    g = await _ground(migrated_db, tmp_path)
    async with _clients(g) as (panel, commander):
        await _login(g, panel)
        active = await panel.get("/api/v1/missions/active", params={"vehicle_id": "sim-01"})
        assert active.status_code == 200 and active.json()["id"] == mission
        assert (await commander.get("/commander/v1/missions/active")).json() == {
            "mission_id": mission
        }
        res = await panel.patch(
            f"/api/v1/missions/{mission}", json={"title": "Sök vid sjön"}, headers=H
        )
        assert res.json()["title"] == "Sök vid sjön"
        res = await panel.put(f"/api/v1/missions/{mission}/ai", json={"enabled": False}, headers=H)
        assert res.json()["ai_enabled"] is False
        await g.flush_events()
        events = await panel.get(f"/api/v1/missions/{mission}/events")
        assert [e["type"] for e in events.json()] == ["mission.updated", "mission.ai_toggled"]
        assert events.json()[1]["payload"]["by"] == "operator:demo"
        listed = await panel.get("/api/v1/missions")
        assert [m["id"] for m in listed.json()] == [mission]
        vehicles = await panel.get("/api/v1/vehicles")
        assert vehicles.json()[0]["id"] == "sim-01"
        missing = await panel.get("/api/v1/missions/0192f49e-0b3a-7c55-8e2f-1a2b3c4d5e6f/events")
        assert missing.status_code == 404


async def test_commander_messages(migrated_db: Db, tmp_path: Path) -> None:
    mission = await asyncio.to_thread(_mission, migrated_db)
    g = await _ground(migrated_db, tmp_path)
    sub = g.bus.subscribe(["chat"])
    async with _clients(g) as (panel, commander):
        body = {
            "client_msg_id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a11",
            "text": "a\u0007b\nc",
            "kind": "text",
        }
        first = await commander.post(f"/commander/v1/missions/{mission}/messages", json=body)
        again = await commander.post(f"/commander/v1/missions/{mission}/messages", json=body)
        assert first.status_code == again.status_code == 201
        assert first.json() == again.json()
        chat = (await commander.get(f"/commander/v1/missions/{mission}/chat")).json()
        assert [(m["role"], m["author_ref"], m["text"]) for m in chat] == [
            ("brain", "brain:ground", "ab\nc")
        ]
        assert sub.queue.qsize() == 1  # one live event, not two
        await _login(g, panel)
        page = (await panel.get(f"/api/v1/missions/{mission}/chat")).json()
        assert page["items"][0]["text"] == "ab\nc"
        long = {**body, "client_msg_id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a12", "text": "x" * 4001}
        assert (
            await commander.post(f"/commander/v1/missions/{mission}/messages", json=long)
        ).status_code == 422
        reply = {
            **body,
            "client_msg_id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a13",
            "in_reply_to": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8aff",
        }
        res = await commander.post(f"/commander/v1/missions/{mission}/messages", json=reply)
        assert res.json()["code"] == "REPLY_TARGET_UNKNOWN"
    assert clean_text("ab") == "ab"


async def test_commander_memory_and_decisions(migrated_db: Db, tmp_path: Path) -> None:
    mission = await asyncio.to_thread(_mission, migrated_db)
    g = await _ground(migrated_db, tmp_path)
    async with _clients(g) as (_, commander):
        empty = (await commander.get(f"/commander/v1/missions/{mission}/memory")).json()
        assert empty["summary_sv"] == "" and empty["updated_at"] is None
        memory = {
            "mission_id": mission,
            "summary_sv": "Sökt\u0000 sektor A",
            "plan_note": "B sen",
            "summarized_up_to_seq": 3,
        }
        assert (
            await commander.put(f"/commander/v1/missions/{mission}/memory", json=memory)
        ).status_code == 204
        stored = (await commander.get(f"/commander/v1/missions/{mission}/memory")).json()
        assert stored["summary_sv"] == "Sökt sektor A" and stored["updated_at"] is not None
        turn = "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a21"
        decision = {
            "id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a20",
            "turn_id": turn,
            "trigger": "operator_message",
            "backend": "scripted",
            "ai_config_hash": "abc",
            "prompt_sha256": "0" * 64,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "latency_ms": 5,
            "rounds": 1,
            "repairs": 0,
            "tool_calls": [],
            "command_id": None,
            "gate_decision": None,
            "error": None,
            "prompt_json": {"messages": ["hej"]},
        }
        res = await commander.post(f"/commander/v1/missions/{mission}/decisions", json=decision)
        assert res.status_code == 204, res.text
    body = (await _q(migrated_db, "SELECT body FROM brain_decisions"))[0]["body"]
    stored_body = json.loads(body) if isinstance(body, str) else body
    assert stored_body["prompt_file"] == f"brain/{turn}.json"
    assert "prompt_json" not in stored_body
    prompt = tmp_path / "data" / "missions" / mission / "brain" / f"{turn}.json"
    assert json.loads(prompt.read_text()) == {"messages": ["hej"]}


async def test_prompt_file_is_never_rewritten(migrated_db: Db, tmp_path: Path) -> None:
    mission = await asyncio.to_thread(_mission, migrated_db)
    g = await _ground(migrated_db, tmp_path)
    turn = "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a41"
    base = {
        "turn_id": turn,
        "trigger": "operator_message",
        "backend": "scripted",
        "ai_config_hash": "abc",
        "prompt_sha256": "0" * 64,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "latency_ms": 5,
        "rounds": 1,
        "repairs": 0,
        "tool_calls": [],
        "command_id": None,
        "gate_decision": None,
        "error": None,
    }
    async with _clients(g) as (_, commander):
        url = f"/commander/v1/missions/{mission}/decisions"
        first = {**base, "id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a42", "prompt_json": {"v": 1}}
        assert (await commander.post(url, json=first)).status_code == 204
        second = {**base, "id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a43", "prompt_json": {"v": 2}}
        res = await commander.post(url, json=second)
        assert res.status_code == 409 and res.json()["code"] == "TURN_EXISTS"
    prompt = tmp_path / "data" / "missions" / mission / "brain" / f"{turn}.json"
    assert json.loads(prompt.read_text()) == {"v": 1}


async def test_closed_mission_refuses_commander_text(migrated_db: Db, tmp_path: Path) -> None:
    mission = await asyncio.to_thread(_mission, migrated_db, "CLOSED")
    g = await _ground(migrated_db, tmp_path)
    async with _clients(g) as (_, commander):
        body = {"client_msg_id": "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a31", "text": "hej"}
        res = await commander.post(f"/commander/v1/missions/{mission}/messages", json=body)
        assert res.status_code == 409 and res.json()["code"] == "MISSION_CLOSED"


# --- the command line and the running server ----------------------------------------


def _cli(
    db: Db, *args: str, env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "SD_GROUND__DB__URL": f"postgresql+asyncpg://sd_app@127.0.0.1:5433/{db.name}",
        "SD_GROUND__DB__OWNER_URL": _owner_url(db),
        "SD_GROUND__DB_PASSWORD": db.owner_password,
        "SD_GROUND__DB_APP_PASSWORD": db.app_password,
        **(env_extra or {}),
    }
    return subprocess.run(
        [sys.executable, "-m", "sd_ground", *args],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_cli_tokens_and_vehicles(migrated_db: Db) -> None:
    res = _cli(migrated_db, "token", "create", "--name", "ci", "--role", "pilot")
    assert res.returncode == 0, res.stderr
    token = res.stdout.strip()
    assert len(token) >= 40 and "\n" not in token  # only the value when stdout is no terminal
    row = migrated_db.run(
        "SELECT expires_at - created_at AS span, auth_mode, token_sha256 FROM auth_sessions"
    )[0]
    assert row["span"] == dt.timedelta(hours=24) or abs(
        row["span"] - dt.timedelta(hours=24)
    ) < dt.timedelta(seconds=5)
    assert row["auth_mode"] == "cli" and token not in row["token_sha256"]
    assert _cli(migrated_db, "token", "create", "--name", "ci", "--role", "pilot").returncode == 1
    assert (
        _cli(
            migrated_db, "token", "create", "--name", "x", "--role", "pilot", "--expires", "31d"
        ).returncode
        == 1
    )
    assert (
        _cli(migrated_db, "token", "create", "--name", "y", "--role", "commander").returncode == 2
    )
    assert _cli(migrated_db, "token", "revoke", "--name", "ci").returncode == 0
    assert _cli(migrated_db, "vehicle", "add", "sim-01", "--name", "Sim 1").returncode == 0
    assert _cli(migrated_db, "vehicle", "add", "sim-01").returncode == 1
    assert _cli(migrated_db, "vehicle", "add", "sim-01", "--if-missing").returncode == 0


def test_db_setup_is_idempotent(migrated_db: Db) -> None:
    res = _cli(migrated_db, "db", "setup")
    assert res.returncode == 0, res.stderr
    assert "uppdaterad" in res.stdout


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _lan_ip() -> str | None:
    """This machine's LAN address, found in a child process (tests may not leave loopback)."""
    probe = (
        "import socket\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "try:\n"
        "    s.connect(('10.255.255.255', 1))\n"  # UDP connect sends no packet
        "    print(s.getsockname()[0])\n"
        "except OSError:\n"
        "    pass\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    ip = out.stdout.strip()
    return None if not ip or ip.startswith("127.") else ip


def _wait(url: str, host: str, seconds: float = 30.0) -> httpx.Response:
    deadline = time.monotonic() + seconds
    while True:
        try:
            return httpx.get(url, headers={"Host": host}, timeout=1)
        except httpx.TransportError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.2)


def test_serve_binds_loopback_only(migrated_db: Db, tmp_path: Path) -> None:
    panel_port, commander_port = _free_port(), _free_port()
    data = tmp_path / "data"
    token_file = tmp_path / "secrets" / "brain_token"
    token_file.parent.mkdir(mode=0o700)
    token_file.write_text(BRAIN)
    token_file.chmod(0o600)
    env = {
        "SD_GROUND__HTTP__PANEL_BIND": f"127.0.0.1:{panel_port}",
        "SD_GROUND__HTTP__COMMANDER_BIND": f"127.0.0.1:{commander_port}",
        "SD_GROUND__DATA_DIR": str(data),
        "SD_DATA_DIR": str(data),
        "SD_GROUND__AUTH__BRAIN_TOKEN_FILE": str(token_file),
        "SD_GROUND__VIDEO_PUBLISH_PASSWORD": "video-pass-for-tests",
        "SD_GROUND__VIDEO_API_PASSWORD": "video-api-pass-for-tests",
        "XDG_RUNTIME_DIR": str(tmp_path),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "sd_ground", "serve"],
        env={
            **os.environ,
            **env,
            **{
                "SD_GROUND__DB__URL": f"postgresql+asyncpg://sd_app@127.0.0.1:5433/{migrated_db.name}",
                "SD_GROUND__DB_PASSWORD": migrated_db.owner_password,
                "SD_GROUND__DB_APP_PASSWORD": migrated_db.app_password,
            },
        },
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        health = _wait(f"http://127.0.0.1:{panel_port}/api/v1/health", f"127.0.0.1:{panel_port}")
        assert health.status_code == 200
        body = health.json()
        assert body["parts"]["db"] == "ok" and body["ready"] is False  # no MAVLink/SD-link yet
        skills = httpx.get(
            f"http://127.0.0.1:{commander_port}/commander/v1/skills",
            headers={"Authorization": f"Bearer {BRAIN}", "Host": f"127.0.0.1:{commander_port}"},
        )
        assert skills.status_code == 200
        lan = _lan_ip()
        if lan is not None:
            for port in (panel_port, commander_port):
                probe = subprocess.run(  # outside pytest-socket: a real connect to the LAN address
                    [
                        sys.executable,
                        "-c",
                        "import socket,sys;s=socket.socket();s.settimeout(2);"
                        f"sys.exit(s.connect_ex(('{lan}',{port})))",
                    ],
                    check=False,
                )
                assert probe.returncode != 0, f"answers on {lan}:{port}"
        reply = _cli(migrated_db, "login-link", "--print-code", env_extra=env)
        assert reply.returncode == 0, reply.stderr
        code = reply.stdout.strip()
        session = httpx.post(
            f"http://127.0.0.1:{panel_port}/api/v1/session",
            json={"login_code": code},
            headers={**H, "Host": f"127.0.0.1:{panel_port}"},
        )
        assert session.status_code == 200
    finally:
        proc.send_signal(signal.SIGTERM)
        out, err = proc.communicate(timeout=20)
    assert proc.returncode == 0, err
    assert "#login=" in out  # the login link is printed for the operator
    log = (data / "logs" / "sd-ground.jsonl").read_text()
    assert code not in log and code not in err


def test_serve_refuses_unmigrated_database(fresh_db: Db, tmp_path: Path) -> None:
    fresh_db.run("CREATE TABLE alembic_version (version_num varchar(32) PRIMARY KEY)")
    fresh_db.run("INSERT INTO alembic_version VALUES ('0000_old')")
    fresh_db.run("GRANT SELECT ON alembic_version TO sd_app")
    data = tmp_path / "data"
    token_file = tmp_path / "brain_token"
    token_file.write_text(BRAIN)
    token_file.chmod(0o600)
    res = subprocess.run(
        [sys.executable, "-m", "sd_ground", "serve"],
        env={
            **os.environ,
            "SD_GROUND__DB__URL": f"postgresql+asyncpg://sd_app@127.0.0.1:5433/{fresh_db.name}",
            "SD_GROUND__DB_APP_PASSWORD": fresh_db.app_password,
            "SD_GROUND__DATA_DIR": str(data),
            "SD_DATA_DIR": str(data),
            "SD_GROUND__AUTH__BRAIN_TOKEN_FILE": str(token_file),
            "SD_GROUND__VIDEO_PUBLISH_PASSWORD": "video-pass-for-tests",
            "SD_GROUND__VIDEO_API_PASSWORD": "video-api-pass-for-tests",
            "SD_GROUND__HTTP__PANEL_BIND": f"127.0.0.1:{_free_port()}",
            "SD_GROUND__HTTP__COMMANDER_BIND": f"127.0.0.1:{_free_port()}",
            "XDG_RUNTIME_DIR": str(tmp_path),
        },
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert res.returncode != 0
    assert "Kör: just migrate" in res.stderr
