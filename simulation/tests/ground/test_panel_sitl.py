"""The panel in a real browser against the whole stack on ArduCopter SITL (builtin physics).

Started by ``just test-panel-sitl``, never by the pull-request or the nightly suite
(marker ``panel``): the browser test needs Playwright next to the stack. Everything is
real but the language model: ``sd-ground`` with its database, the gate, the emergency
path and SD-link, the onboard executor against SITL (as ``test_mission_flow_sitl``),
``sd-brain`` over the commander API with the scripted backend
(``simulation/scenarios/scripts/panel_demo.yaml``), and the panel's test build served by
the ground station on a loopback port.

The browser side lives in ``software/panel/e2e/sitl/`` and logs in like a person, with a
one-time login code. The two sides talk through files in ``build/panel-sitl/``:

* ``stack.json`` (written here): the panel's URL and one login code;
* ``stop-brain`` (written by the browser test): stop ``sd-brain`` mid-flight;
* ``done.json`` (written by the browser test): its exit code.

With ``SD_PANEL_SITL_PLAYWRIGHT=1`` this test starts the browser test itself (a machine
with Node and Chromium); otherwise it waits for another process to run it (the
simulator container, with Playwright on the host).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "onboard"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "software/ground/tests/helpers"))

from brain_rig import (  # pyright: ignore[reportMissingImports]
    loopback_listener,
    run_brain,
    serve_commander,
)
from flow_rig import VID, Rig, WallClock, ground_stack  # pyright: ignore[reportMissingImports]
from test_mission_flow_sitl import (  # pyright: ignore[reportMissingImports]
    SitlOnboard,
    fresh_db,
    migrated_db,
    passwords,
)
from test_onboard_sitl import Sim, _vehicle, sim  # pyright: ignore[reportMissingImports]

from sd_ground.__main__ import make_lan_cert
from sd_ground.app import create_panel_app, panel_asgi
from sd_mav import Role
from sd_obs.clock import MonotonicClock

pytestmark = [pytest.mark.sitl, pytest.mark.slow, pytest.mark.panel]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2].parent
DIST_TEST = ROOT / "software/panel/dist-test"
SCRIPT = ROOT / "simulation/scenarios/scripts/panel_demo.yaml"
EXCHANGE = ROOT / "build/panel-sitl"
BROWSER_LIMIT_S = 900.0


@contextlib.asynccontextmanager
async def serve_panel(
    r: Rig, sock: socket.socket, *, tls: Path | None = None
) -> AsyncIterator[None]:
    """A panel listener (guard, ingress, API, live socket, static test build): plain HTTP
    on loopback, or with ``tls`` the LAN listener with the ground's self-signed certificate."""
    import uvicorn

    extra: dict[str, Any] = {}
    if tls is not None:
        extra = {"ssl_keyfile": str(tls / "server.key"), "ssl_certfile": str(tls / "server.crt")}
    server = uvicorn.Server(
        uvicorn.Config(
            panel_asgi(r.g, create_panel_app(r.g), listener="local" if tls is None else "lan"),
            lifespan="off",
            log_level="warning",
            ws_ping_interval=None,
            timeout_graceful_shutdown=2,
            **extra,
        )
    )
    task = asyncio.ensure_future(server.serve(sockets=[sock]))
    try:
        await r.until(lambda: server.started, 10, "panel listener up")
        yield
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)


async def _rc_pilot(r: Rig, sim: Sim) -> None:
    """The RC pilot's left stick: down on the ground (arming), centred in the air."""
    while True:
        snap = r.core.snapshot()
        tel = None if snap is None else snap.telemetry
        if tel is not None:
            in_air = tel.armed and tel.landed_state != "on_ground"
            sim.rc.throttle = 1500 if in_air and tel.alt_rel_home_m > 2.0 else 1000
        await asyncio.sleep(0.2)


def _exchange_file(name: str) -> Path | None:
    """A file from the browser side, if it is there yet (a plain read; async code polls)."""
    path = EXCHANGE / name
    return path if path.exists() else None


async def _brain_until_stopped(r: Rig, port: int) -> None:
    """``sd-brain`` with the scripted backend until the browser test asks to stop it."""
    async with run_brain(
        r,
        port,
        backend={"kind": "scripted", "scripted_file": str(SCRIPT)},
        profile="sim",
    ):
        await r.until(lambda: _exchange_file("stop-brain") is not None, 3_600, "stop-brain")


async def _browser_done(limit_s: float) -> int:
    proc = None
    if os.environ.get("SD_PANEL_SITL_PLAYWRIGHT") == "1":
        proc = await asyncio.create_subprocess_exec(
            "pnpm",
            "--filter",
            "@searchdrone/panel",
            "e2e:sitl",
            cwd=ROOT,
            env={**os.environ, "SD_PANEL_SITL_EXCHANGE": str(EXCHANGE)},
        )
    start = time.monotonic()
    while (done := _exchange_file("done.json")) is None:
        if proc is not None and proc.returncode is not None:
            return proc.returncode
        if time.monotonic() - start > limit_s:
            if proc is not None:
                proc.kill()
            raise AssertionError(f"the browser test did not finish within {limit_s} s")
        if proc is not None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), 0.5)
        else:
            await asyncio.sleep(0.5)
    return int(json.loads(done.read_text(encoding="utf-8")).get("code", 1))


async def _report(r: Rig, port: int) -> None:
    """What the AI and the ground did, for a failed run (chat, orders, cards per mission)."""
    out = [f"onboard snapshot: {r.flow.snapshot(VID)[0]!r}"]
    host = {"Host": f"127.0.0.1:{port}"}  # the panel listener's own port
    listed = await r.panel.get("/api/v1/missions", headers=host)
    out.append(f"missions: {listed.status_code} {listed.text[:500]}")
    missions = listed.json() if listed.status_code == 200 else []
    for m in missions if isinstance(missions, list) else []:
        mid = m["id"]
        out.append(f"--- uppdrag {m['title']} {m['status']}")
        for path in ("chat", "commands", "approvals", "charters"):
            res = await r.panel.get(f"/api/v1/missions/{mid}/{path}", headers=host)
            out.append(f"{path}: {res.text[:4000]}")
        res = await r.panel.get(f"/api/v1/missions/{mid}/events?limit=1000", headers=host)
        events = res.json() if res.status_code == 200 else []
        out.append(f"events: {res.status_code}")
        out.extend(
            f"  {e.get('t', '')[11:23]} {e.get('type')} {json.dumps(e.get('payload'))[:300]}"
            for e in (events if isinstance(events, list) else [])
        )
    sys.stderr.write("\n".join(out) + "\n")


@pytest.mark.budget_s(900)
async def test_panel_in_a_browser_on_sitl(sim: Sim, migrated_db: Any, tmp_path: Path) -> None:
    if not (DIST_TEST / "index.html").is_file():
        raise AssertionError(
            "software/panel/dist-test saknas: kör `pnpm --filter @searchdrone/panel build:test`"
        )
    EXCHANGE.mkdir(parents=True, exist_ok=True)
    for name in ("stack.json", "stop-brain", "done.json"):
        (EXCHANGE / name).unlink(missing_ok=True)
    ground = _vehicle(f"udpin:127.0.0.1:{sim.ports['ground_mavlink']}", Role.GROUND)
    panel, commander, lan = loopback_listener(), loopback_listener(), loopback_listener()
    panel_port, commander_port, lan_port = (s.getsockname()[1] for s in (panel, commander, lan))
    tls = tmp_path / "tls"
    make_lan_cert(tls)  # the same self-signed certificate `sd-ground init` makes for LAN mode

    def onboard(_clock: WallClock, url: str, token: str) -> SitlOnboard:
        return SitlOnboard(sim, url, token)

    async with (
        ground_stack(
            migrated_db.name,
            migrated_db.app_password,
            tmp_path,
            vehicle=ground,
            mav_clock=MonotonicClock(),
            onboard=onboard,
            http={
                "commander_bind": f"127.0.0.1:{commander_port}",
                "panel_bind": f"127.0.0.1:{panel_port}",
                "static_dir": str(DIST_TEST),
            },
            lan={"enabled": True, "bind": f"127.0.0.1:{lan_port}", "tls_dir": str(tls)},
        ) as r,
        serve_commander(r, commander),
        serve_panel(r, panel),
        serve_panel(r, lan, tls=tls),
    ):
        r.g.lan_hosts = ["127.0.0.1"]
        tasks = [
            asyncio.ensure_future(_rc_pilot(r, sim)),
            asyncio.ensure_future(_brain_until_stopped(r, commander_port)),
        ]
        try:
            await r.until(lambda: r.g.brain_state == "ready", 30, "sd-brain ready")
            code, _link = r.g.login_link()
            _pair_code, pin, _pair_link = r.g.pair()
            stack = {
                "url": f"http://127.0.0.1:{panel_port}",
                "login_code": code,
                "lan_url": f"https://127.0.0.1:{lan_port}",
                "pin": pin,
            }
            # Readable only by this user from the first byte: a login code and a PIN.
            fd = os.open(EXCHANGE / "stack.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(stack))
            result = await _browser_done(BROWSER_LIMIT_S)
            if result != 0:
                await _report(r, panel_port)
            assert result == 0, (
                f"the browser test failed (exit {result}); see software/panel/test-results"
            )
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            (EXCHANGE / "stack.json").unlink(missing_ok=True)
