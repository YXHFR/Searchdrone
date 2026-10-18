"""The whole stack as separate programs, the way ``just demo`` starts it, for the
end-to-end test (``just e2e``).

``live_stack`` starts ``sd_sim.launch.Stack`` with every service (database, MediaMTX when
it runs, ``sd-ground serve``, SITL with builtin physics and the test picture, linksim,
``sd-onboard``, ``sd-brain run`` with the ``ScriptedBackend``) for this worker's
instance, with its own fresh database. :class:`Pilot` is the operator at the panel: it
logs in with the one-time login code the launcher printed (a session cookie, like a
person), keeps the live WebSocket open (a pilot who is present) and records the
telemetry the panel gets (5 Hz) as the track. Nothing here reaches into a service's
process: everything goes through the panel API, the database and the mission folder.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import asyncpg
import httpx
import websockets

from sd_schemas.base import new_id
from sd_sim import config
from sd_sim.invariants import Sample
from sd_sim.launch import Stack, read_env_value
from sd_sim.pytest_plugin import worker_instance
from sd_sim.sitl import CMAC_HOME
from sd_sim.util import new_run_dir

__all__ = ["VID", "Pilot", "live_stack"]

VID = "sim-01"
PANEL = {"X-SD-Client": "panel", "Content-Type": "application/json"}
PING_S = 5.0


@dataclass
class Pilot:
    stack: Stack
    http: httpx.AsyncClient
    track: list[Sample] = field(default_factory=list[Sample])
    telemetry: dict[str, Any] = field(default_factory=dict[str, Any])
    _tasks: list[asyncio.Task[Any]] = field(default_factory=list[asyncio.Task[Any]])

    @property
    def port(self) -> int:
        return self.stack.ports["ground_panel"]

    # --- the panel ----------------------------------------------------------------------

    async def login(self) -> None:
        code = urlsplit(self.stack.login_url).fragment.removeprefix("login=")
        assert code, f"no login code in {self.stack.login_url!r}"
        res = await self.http.post("/api/v1/session", json={"login_code": code}, headers=PANEL)
        assert res.status_code == 200, res.text
        assert res.json()["role"] == "pilot"

    async def get(self, path: str, **params: Any) -> Any:
        res = await self.http.get(path, params=params or None)
        assert res.status_code == 200, f"GET {path}: {res.status_code} {res.text[:300]}"
        return res.json()

    async def post(self, path: str, body: dict[str, Any], status: int = 200) -> Any:
        res = await self.http.post(path, json=body, headers=PANEL)
        assert res.status_code == status, f"POST {path}: {res.status_code} {res.text[:300]}"
        return res.json() if res.content else None

    async def open_mission(self) -> str:
        mission = await self.post(f"/api/v1/vehicles/{VID}/missions/open", {})
        return str(mission["id"])

    async def say(self, mid: str, text: str) -> dict[str, Any]:
        return await self.post(
            f"/api/v1/missions/{mid}/chat", {"client_msg_id": str(new_id()), "text": text}, 201
        )

    async def status(self, mid: str) -> str:
        return str((await self.get(f"/api/v1/missions/{mid}"))["mission"]["status"])

    async def commands(self, mid: str) -> list[dict[str, Any]]:
        return list(await self.get(f"/api/v1/missions/{mid}/commands"))

    async def gate(self, mid: str) -> list[dict[str, Any]]:
        return list(await self.get(f"/api/v1/missions/{mid}/gate_decisions"))

    async def events(self, mid: str) -> list[dict[str, Any]]:
        return list(await self.get(f"/api/v1/missions/{mid}/events", limit="1000"))

    async def chat(self, mid: str) -> list[dict[str, Any]]:
        data = await self.get(f"/api/v1/missions/{mid}/chat")
        return list(data["items"] if isinstance(data, dict) else data)

    async def until(self, cond: Callable[[], Any], limit_s: float, what: str) -> float:
        """Wait until ``cond()`` (or its awaitable) is true; the time it took."""
        start = time.monotonic()
        while True:
            value = cond()
            if asyncio.iscoroutine(value):
                value = await value
            if value:
                return time.monotonic() - start
            if time.monotonic() - start > limit_s:
                t = self.telemetry
                raise AssertionError(
                    f"{what}: not within {limit_s} s (mode {t.get('flight_mode')}, "
                    f"armed {t.get('armed')}, {t.get('alt_rel_home_m', 0):.1f} m)"
                )
            await asyncio.sleep(0.1)

    async def command(
        self, mid: str, skill: str, statuses: tuple[str, ...], limit_s: float
    ) -> dict[str, Any]:
        """The AI's newest order of ``skill`` once it is in one of ``statuses``."""
        found: dict[str, Any] = {}

        async def reached() -> bool:
            cmds = [c for c in await self.commands(mid) if c["skill"] == skill]
            if not cmds:
                return False
            c = max(cmds, key=lambda c: c["created_at"])
            found.clear()
            found.update(c)
            return c["status"] in statuses

        try:
            await self.until(reached, limit_s, f"{skill} {'/'.join(statuses)}")
        except AssertionError as exc:
            seen = [(c["skill"], c["status"], c["status_reason"]) for c in await self.commands(mid)]
            raise AssertionError(f"{exc}; orders {seen}") from None
        return found

    # --- the live socket ----------------------------------------------------------------

    async def _live(self) -> None:
        cookie = "; ".join(f"{k}={v}" for k, v in self.http.cookies.items())
        url = f"ws://127.0.0.1:{self.port}/ws/v1/live"
        async with websockets.connect(url, additional_headers={"Cookie": cookie}, proxy=None) as ws:
            json.loads(await ws.recv())  # hello
            await ws.send(
                json.dumps({"op": "subscribe", "topics": ["telemetry"], "vehicle_id": VID})
            )

            async def ping() -> None:
                while True:
                    await ws.send(json.dumps({"op": "ping"}))
                    await asyncio.sleep(PING_S)

            pinger = asyncio.ensure_future(ping())
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "telemetry":
                        continue
                    t = msg["data"]
                    self.telemetry = t
                    self.track.append(
                        Sample(
                            t=time.monotonic(),
                            lat=float(t["lat"]),
                            lon=float(t["lon"]),
                            alt_m=float(t["alt_rel_home_m"]),
                            groundspeed_ms=float(t["groundspeed_ms"]),
                            mode=str(t["flight_mode"]),
                            armed=bool(t["armed"]),
                        )
                    )
            finally:
                pinger.cancel()

    async def start_live(self) -> None:
        self._tasks.append(asyncio.ensure_future(self._live()))
        await self.until(lambda: bool(self.telemetry), 15, "telemetry on the live socket")

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.http.aclose()

    # --- the database (read only, as its owner) -----------------------------------------

    @contextlib.asynccontextmanager
    async def db(self) -> AsyncIterator[asyncpg.Connection]:
        assert self.stack.layout is not None
        conn = await asyncpg.connect(
            host="127.0.0.1",
            port=self.stack.ports["postgres_local"],
            user="sd_owner",
            password=read_env_value(self.stack.repo / ".env", "SD_GROUND__DB_PASSWORD"),
            database=self.stack.layout.db_name,
        )
        try:
            yield conn
        finally:
            await conn.close()

    def mission_dir(self, mid: str) -> Path:
        assert self.stack.layout is not None
        return self.stack.layout.data_dir / "missions" / mid


@contextlib.asynccontextmanager
async def live_stack(name: str, script: str) -> AsyncIterator[Pilot]:
    """The whole stack with ``simulation/scenarios/scripts/<script>.yaml``, logged in."""
    instance = worker_instance()
    cfg = config.load(
        {
            "physics": "builtin",
            "gui": "false",
            "video": "test",
            "brain": "scripted",
            "script": f"simulation/scenarios/scripts/{script}.yaml",
            "instance": str(instance),
        }
    )
    cfg.home, cfg.personal_home, cfg.open_browser = CMAC_HOME, False, False
    stack = Stack(cfg, new_run_dir(instance, name), with_services=True, fresh_db=True)
    pilot: Pilot | None = None
    try:
        await asyncio.to_thread(stack.up)
        pilot = Pilot(
            stack,
            httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{stack.ports['ground_panel']}",
                timeout=10,
                trust_env=False,  # loopback only, never through a proxy
            ),
        )
        await pilot.login()
        await pilot.start_live()
        yield pilot
    finally:
        if pilot is not None:
            await pilot.close()
        await asyncio.to_thread(stack.down)
