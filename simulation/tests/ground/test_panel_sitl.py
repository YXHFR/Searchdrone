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
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "onboard"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "software/ground/tests/helpers"))

from brain_rig import free_port, run_brain, serve_commander  # pyright: ignore[reportMissingImports]
from flow_rig import Rig, WallClock, ground_stack  # pyright: ignore[reportMissingImports]
from test_mission_flow_sitl import (  # pyright: ignore[reportMissingImports]
    SitlOnboard,
    fresh_db,
    migrated_db,
    passwords,
)
from test_onboard_sitl import Sim, _vehicle, sim  # pyright: ignore[reportMissingImports]

from sd_ground.__main__ import _make_cert  # pyright: ignore[reportPrivateUsage]
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
async def serve_panel(r: Rig, port: int, *, tls: Path | None = None) -> AsyncIterator[None]:
    """A panel listener (guard, ingress, API, live socket, static test build): plain HTTP
    on loopback, or with ``tls`` the LAN listener with the ground's self-signed certificate."""
    import uvicorn

    extra: dict[str, Any] = {}
    if tls is not None:
        extra = {"ssl_keyfile": str(tls / "server.key"), "ssl_certfile": str(tls / "server.crt")}
    server = uvicorn.Server(
        uvicorn.Config(
            panel_asgi(r.g, create_panel_app(r.g), listener="local" if tls is None else "lan"),
            host="127.0.0.1",
            port=port,
            lifespan="off",
            log_level="warning",
            ws_ping_interval=None,
            timeout_graceful_shutdown=2,
            **extra,
        )
    )
    task = asyncio.ensure_future(server.serve())
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
            in_air = tel.armed and str(tel.landed_state) not in (
                "ON_GROUND",
                "LandedState.ON_GROUND",
            )
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


async def _report(r: Rig) -> None:
    """What the AI and the ground did, for a failed run (chat, orders, cards per mission)."""
    missions = (await r.panel.get("/api/v1/missions")).json()
    for m in missions if isinstance(missions, list) else []:
        mid = m["id"]
        log.warning("uppdrag %s %s", m["title"], m["status"])
        for path in ("chat", "commands", "approvals", "charters"):
            res = await r.panel.get(f"/api/v1/missions/{mid}/{path}")
            log.warning("%s: %s", path, res.text[:3000])


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
    panel_port, commander_port, lan_port = free_port(), free_port(), free_port()
    tls = tmp_path / "tls"
    _make_cert(tls)  # the same self-signed certificate `sd-ground init` makes for LAN mode

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
        serve_commander(r, commander_port),
        serve_panel(r, panel_port),
        serve_panel(r, lan_port, tls=tls),
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
            path = EXCHANGE / "stack.json"
            path.write_text(json.dumps(stack), encoding="utf-8")
            path.chmod(0o600)
            result = await _browser_done(BROWSER_LIMIT_S)
            if result != 0:
                await _report(r)
            assert result == 0, (
                f"the browser test failed (exit {result}); see software/panel/test-results"
            )
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            (EXCHANGE / "stack.json").unlink(missing_ok=True)
