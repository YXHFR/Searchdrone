"""SC18: the APIs refuse everyone who may not act, with the vehicle on the ground.

A mission with an approved charter; the panel API is served on a loopback port. The test
logs in as the panel does (a one-time login code, ``POST /api/v1/session``, the
``sd_session`` cookie) and then tries every way in that it may not have:

* no session or a wrong token: 401 on every changing endpoint, on the read paths and on
  the live WebSocket;
* the AI's token on the operator's endpoints: 403 on the panel API, 404 on the
  commander API; ``/api/v1/brain/*`` does not exist;
* with the cookie: a foreign ``Host`` (421), a foreign ``Origin`` (403), no
  ``X-SD-Client`` or ``text/plain`` (403 ``CSRF``); the WebSocket with a foreign
  ``Origin`` is refused and a bearer token together with an ``Origin`` is closed;
* the login code works once, and not after its lifetime;
* an observer may change nothing;
* nothing changed in the database and the motors were never armed.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
import websockets
from brain_rig import free_port  # pyright: ignore[reportMissingImports]
from scenario_rig import Scenario, scenario  # pyright: ignore[reportMissingImports]

from sd_ground.app import create_panel_app, panel_asgi
from sd_ground.auth import SessionRecord, new_token
from sd_ground.auth import hash_token as hash_session_token
from sd_schemas.base import new_id

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.script("SC18")]

H = {"X-SD-Client": "panel", "Content-Type": "application/json"}
CODE_TTL_S = 5
VID = "sim-01"


def _changing(mid: str, approval: str) -> list[tuple[str, str]]:
    """Every endpoint that changes something, with the mission's ids filled in."""
    m = f"/api/v1/missions/{mid}"
    v = f"/api/v1/vehicles/{VID}"
    return [
        ("POST", f"{v}/missions/open"),
        ("POST", f"{m}/charters"),
        ("POST", f"/api/v1/approvals/{approval}/decision"),
        ("POST", f"{m}/clearance"),
        ("PUT", f"{m}/limits"),
        ("PUT", f"{m}/autonomy"),
        ("PUT", f"{m}/ai"),
        ("POST", f"{m}/commands"),
        ("POST", f"{m}/chat"),
        ("POST", f"{m}/annotations"),
        ("POST", f"{m}/end"),
        ("POST", f"{v}/actions/stop"),
        ("POST", f"{v}/actions/return_home"),
        ("POST", f"{v}/actions/resume"),
        ("POST", "/api/v1/actions/stop_all"),
        ("POST", "/api/v1/actions/return_home_all"),
    ]


def _body(path: str) -> dict[str, Any]:
    """A body that passes validation, so the answer is about who asks, not what."""
    if "/actions/" in path:
        return {"client_action_id": str(new_id())}
    return {}


OPERATOR_ONLY = (
    "/decision",
    "/autonomy",
    "/clearance",
    "/limits",
    "/actions/",
    "/end",
)


@contextlib.asynccontextmanager
async def _served_panel(sc: Scenario, port: int) -> AsyncIterator[str]:
    app = panel_asgi(sc.r.g, create_panel_app(sc.r.g))
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="off", log_level="warning")
    )
    task = asyncio.ensure_future(server.serve())
    try:
        await sc.until(lambda: server.started, 10, "panel-API:t uppe")
        yield f"127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)


async def _token(sc: Scenario, role: str) -> str:
    """A CLI token as ``sd-ground token create`` makes it."""
    repo = sc.r.g.repo
    assert repo is not None
    token = new_token()
    now = dt.datetime.now(dt.UTC)
    await repo.save_session(
        SessionRecord(
            id="",
            kind="token",
            role=role,
            issuer_ref=f"operator:ci-{role}",
            auth_mode="cli",
            token_sha256=hash_session_token(token),
            created_at=now,
            last_used_at=now,
            expires_at=now + dt.timedelta(hours=24),
            name=f"ci-{role}",
        )
    )
    sc.r.g.sessions.load(await repo.sessions())
    return token


async def _counts(sc: Scenario) -> dict[str, int]:
    m = f"/api/v1/missions/{sc.mission_id}"
    events = (await sc.r.panel.get(f"{m}/events")).json()
    return {
        "charters": len((await sc.r.panel.get(f"{m}/charters")).json()),
        "commands": len(await sc.commands()),
        "missions": len((await sc.r.panel.get("/api/v1/missions")).json()),
        "events": sum(1 for e in events if not str(e["type"]).startswith(("session", "auth"))),
    }


async def _ws_refused(url: str, **headers: str) -> bool:
    """The live WebSocket is refused at the upgrade or closed with 4401/4403."""
    try:
        async with websockets.connect(url, additional_headers=headers, open_timeout=5) as ws:
            try:
                await asyncio.wait_for(ws.recv(), 5)
            except websockets.ConnectionClosed as exc:
                return exc.rcvd is not None and exc.rcvd.code in (4401, 4403)
            return False
    except websockets.InvalidStatus as exc:
        return exc.response.status_code in (401, 403)


@pytest.mark.budget_s(150)
async def test_sc18(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    port = free_port()
    config = {
        "http": {"panel_bind": f"127.0.0.1:{port}"},
        "auth": {"cli_may_confirm": True, "login_code_ttl_s": CODE_TTL_S},
    }
    async with scenario(
        "test_sc18", sim, migrated_db, tmp_path, "SC18", ground_config=config
    ) as sc:
        sc.r.panel.base_url = f"http://127.0.0.1:{port}"
        async with _served_panel(sc, port) as host:
            await _run(sc, host)


async def _run(sc: Scenario, host: str) -> None:
    await sc.ready()
    approval = (await sc.approvals("charter"))[-1]["id"]
    before = await _counts(sc)
    base = f"http://{host}"
    endpoints = _changing(sc.mission_id, approval)

    async with httpx.AsyncClient(base_url=base, timeout=10) as anon:
        # 1. No session, a wrong token.
        for method, path in endpoints:
            for auth in ({}, {"Authorization": "Bearer wrong-token-0123456789abcdef"}):
                res = await anon.request(method, path, json={}, headers={**H, **auth})
                sc.criterion(
                    "SC18.1", res.status_code == 401, f"{method} {path}", got=res.status_code
                )
        for method, path in [
            ("GET", "/api/v1/missions"),
            ("POST", f"/api/v1/video/{VID}/main/whep"),
        ]:
            res = await anon.request(method, path, headers=H)
            sc.criterion("SC18.1r", res.status_code == 401, f"{method} {path}", got=res.status_code)
        ws = f"ws://{host}/ws/v1/live"
        sc.criterion("SC18.1w", await _ws_refused(ws), "WS utan inloggning nekas")

        # 2. The AI's token on the operator's paths.
        brain = {"Authorization": sc.r.commander.headers["Authorization"]}
        for method, path in endpoints:
            if not any(p in path for p in OPERATOR_ONLY):
                continue
            res = await anon.request(method, path, json=_body(path), headers={**H, **brain})
            sc.criterion(
                "SC18.2", res.status_code == 403, f"AI: {method} {path}", got=res.status_code
            )
            cpath = path.replace("/api/v1/", "/commander/v1/")
            res = await sc.r.commander.request(method, cpath, json={})
            sc.criterion("SC18.2c", res.status_code == 404, f"AI: {cpath}", got=res.status_code)
        # No AI paths on the panel API: the AI's token is refused before routing, and
        # with the pilot's token the path does not exist.
        res = await anon.get("/api/v1/brain/state", headers=brain)
        sc.criterion("SC18.2b", res.status_code in (403, 404), "AI-token", got=res.status_code)
        pilot = {"Authorization": sc.r.panel.headers["Authorization"]}
        res = await anon.get("/api/v1/brain/state", headers=pilot)
        sc.criterion(
            "SC18.2d", res.status_code == 404, "brain-vägar finns inte", got=res.status_code
        )

        # 4. The login code: once, and only within its lifetime.
        code, _ = sc.r.g.login_link()
        res = await anon.post("/api/v1/session", json={"login_code": code}, headers=H)
        assert res.status_code == 200, res.text
        cookie = {"sd_session": res.cookies["sd_session"]}
        res = await anon.post("/api/v1/session", json={"login_code": code}, headers=H)
        sc.criterion(
            "SC18.4a", res.status_code == 401, "koden går inte två gånger", got=res.status_code
        )
        old, _ = sc.r.g.login_link()
        await asyncio.sleep(CODE_TTL_S + 1)
        res = await anon.post("/api/v1/session", json={"login_code": old}, headers=H)
        sc.criterion("SC18.4b", res.status_code == 401, "gammal kod nekas", got=res.status_code)

    # 3. With the cookie: the browser protections.
    stop = f"/api/v1/vehicles/{VID}/actions/stop"
    async with httpx.AsyncClient(base_url=base, cookies=cookie, timeout=10) as browser:
        res = await browser.post(stop, json={}, headers={**H, "Host": "evil.example"})
        sc.criterion("SC18.3a", res.status_code == 421, "främmande Host", got=res.status_code)
        res = await browser.post(stop, json={}, headers={**H, "Origin": "https://evil.example"})
        sc.criterion("SC18.3b", res.status_code == 403, "främmande Origin", got=res.status_code)
        res = await browser.post(stop, json={}, headers={"Content-Type": "application/json"})
        sc.criterion("SC18.3c", res.status_code == 403 and "CSRF" in res.text, "utan X-SD-Client")
        res = await browser.post(
            stop, content="{}", headers={"X-SD-Client": "panel", "Content-Type": "text/plain"}
        )
        sc.criterion("SC18.3d", res.status_code == 403 and "CSRF" in res.text, "text/plain")
    session = f"sd_session={cookie['sd_session']}"
    refused = await _ws_refused(ws, Cookie=session, Origin="https://evil.example")
    sc.criterion("SC18.3e", refused, "WS med cookie och främmande Origin nekas")
    pilot = sc.r.panel.headers["Authorization"]
    refused = await _ws_refused(ws, Authorization=pilot, Origin=f"http://{host}")
    sc.criterion("SC18.3f", refused, "bearer-token med Origin på WS stängs")

    # 5. The observer.
    observer = {"Authorization": f"Bearer {await _token(sc, 'observer')}"}
    async with httpx.AsyncClient(base_url=base, timeout=10) as client:
        for method, path in endpoints:
            if path.endswith("/actions/stop") or path.endswith("/stop_all"):
                continue  # everyone logged in may press STOP (the owner's decision)
            res = await client.request(method, path, json=_body(path), headers={**H, **observer})
            sc.criterion(
                "SC18.5", res.status_code == 403, f"observer: {method} {path}", got=res.status_code
            )

    # 7. Nothing changed; never armed.
    after = await _counts(sc)
    sc.criterion(
        "SC18.7", after == before, "inga ändringar i databasen", before=before, after=after
    )
    arms = [m for m in sc.tl.sent if m.command == 400 and m.params and m.params[0] == 1]
    sc.criterion("SC18.7b", not arms and not sc.tel.armed, "aldrig armerad")
    await sc.finish()
