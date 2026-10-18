"""The scenario rig with a ground station that can die and come back, and an observer.

``scenario_x`` is ``scenario_rig.scenario`` with two additions for the link and failure
scenarios:

* the ground station (``sd-ground``: mission flow, gate, emergency path, SD-link server,
  both APIs) can be killed (``kill_ground``) and started again (``start_ground``) on the
  same database while SITL, linksim and the onboard executor keep running. While it is
  dead its MAVLink heartbeat (250/190) is gone and SD-link is closed; the onboard
  executor reconnects through linksim when it is back. The AI is stopped with it and
  started again against the new commander API.
* an observer on the harness port (253, reads only: heartbeats and stream requests)
  sees the vehicle when the ground station is dead or the radio is down: ``sc.tel``
  reads it, and its samples go into the timeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from brain_rig import free_port, serve_commander  # pyright: ignore[reportMissingImports]
from flow_rig import VID, Rig, WallClock, ground_stack  # pyright: ignore[reportMissingImports]
from scenario_rig import (  # pyright: ignore[reportMissingImports]
    SCRIPTS,
    Scenario,
    _Onboard,  # pyright: ignore[reportPrivateUsage]
    _record_writes,  # pyright: ignore[reportPrivateUsage]
    _sticks,  # pyright: ignore[reportPrivateUsage]
)
from test_onboard_sitl import Sim, _vehicle  # pyright: ignore[reportMissingImports]

from sd_mav import CopterMode, Role
from sd_obs.clock import MonotonicClock
from sd_safety.targets import distance_m
from sd_sim import expect, invariants
from sd_sim.errors import InfraError
from sd_sim.invariants import Sample

__all__ = ["Observer", "Watched", "scenario_x"]


@dataclass(frozen=True)
class View:
    """The observer's view of the vehicle, with the names ``sc.tel`` callers use."""

    lat: float
    lon: float
    alt_rel_home_m: float
    groundspeed_ms: float
    fc_custom_mode: int | None
    armed: bool
    battery_pct: float | None

    @property
    def fc_mode(self) -> str:
        return "UNKNOWN" if self.fc_custom_mode is None else CopterMode(self.fc_custom_mode).name

    @property
    def landed_state(self) -> str:
        return "on_ground" if not self.armed or self.alt_rel_home_m < 0.3 else "in_air"


class Observer:
    """Pumps the harness in a thread: the vehicle as the flight controller reports it,
    its STATUSTEXTs and mode changes. Harness writes (fault injection) go through
    ``call`` so the two never read the same socket at once."""

    def __init__(self, sim: Sim, on_sample: Callable[[View], None]) -> None:
        self.h = sim.harness
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._on_sample = on_sample
        self.view: View | None = None
        self.battery_pct: float | None = None
        self.texts: list[tuple[float, str]] = []
        self.modes: list[tuple[float, int]] = []  # (host time, custom_mode) on each change
        apply = self.h._apply  # pyright: ignore[reportPrivateUsage]

        def recording(msg: Any) -> None:
            kind = msg.get_type()
            if msg.get_srcSystem() == 1 and msg.get_srcComponent() == 1:
                if kind == "STATUSTEXT":
                    raw = msg.text
                    text = raw.decode(errors="replace") if isinstance(raw, bytes) else str(raw)
                    self.texts.append((time.monotonic(), text.rstrip("\x00")))
                elif kind == "SYS_STATUS" and int(msg.battery_remaining) >= 0:
                    self.battery_pct = float(msg.battery_remaining)
                elif kind == "HEARTBEAT" and int(msg.type) != 6:  # not a GCS
                    mode = int(msg.custom_mode)
                    if not self.modes or self.modes[-1][1] != mode:
                        self.modes.append((time.monotonic(), mode))
            apply(msg)
            if kind in ("GLOBAL_POSITION_INT", "HEARTBEAT"):
                self._update()

        self.h._apply = recording  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]
        self._thread = threading.Thread(target=self._run, name="observer", daemon=True)

    def _update(self) -> None:
        s = self.h.state
        mode = self.modes[-1][1] if self.modes else None
        self.view = View(
            s.lat, s.lon, s.rel_alt_m, s.ground_speed_ms, mode, s.armed, self.battery_pct
        )
        if s.lat:
            self._on_sample(self.view)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(3)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                self.h.pump(0.2)
            time.sleep(0.005)

    def call(self, fn: Callable[[], Any]) -> Any:
        with self._lock:
            return fn()

    def mode_since(self, t: float) -> list[tuple[float, int]]:
        return [(tt, m) for tt, m in self.modes if tt >= t]

    def text_since(self, t: float, pattern: Any) -> list[tuple[float, str]]:
        return [(tt, s) for tt, s in self.texts if tt >= t and pattern.search(s)]


class _Kept:
    """What ``ground_stack`` sees as its onboard side: the real executor, which outlives
    every ground station. A new ground station's link token is handed to it."""

    def __init__(self, side: _Onboard) -> None:
        self.side = side
        self.started = False

    async def start(self, home: tuple[float, float]) -> None:
        if not self.started:
            self.started = True
            await self.side.start(home)

    async def close(self) -> None:
        return None  # the onboard computer does not die with the ground station


@dataclass
class Watched(Scenario):
    observer: Observer | None = None
    ground_alive: bool = True
    _db: Any = None
    _tmp: Path | None = None
    _ground: contextlib.AsyncExitStack | None = None
    _kept: _Kept | None = None
    link_log: list[tuple[float, str, str]] = field(default_factory=list[tuple[float, str, str]])

    @property
    def tel(self) -> Any:
        if self.observer is not None and self.observer.view is not None:
            return self.observer.view
        return self.r.core.vehicle.telemetry

    def from_home_m(self) -> float:
        return distance_m(self.r.home[0], self.r.home[1], self.tel.lat, self.tel.lon)

    async def battery(self, pct: float) -> None:
        self._harness_sent("COMMAND_LONG", command=42651, params=(1.0, pct))
        assert self.observer is not None
        obs = self.observer
        await asyncio.to_thread(obs.call, lambda: obs.h.battery_set_remaining(pct))

    async def snapshot_now(self) -> None:
        """Read orders, gate decisions and charters now, while the ground station is
        alive, for ``finish_offline`` (the ground station will not come back)."""
        await self._snapshot_db()  # pyright: ignore[reportPrivateUsage]

    def finish_offline(self, exempt: Any = ()) -> None:
        """``finish`` on the database read by ``snapshot_now``; the vehicle's track,
        MAVLink and events cover the whole test."""
        self.tl.events = sorted(
            [*self.tl.events, *(e for e in self._bus if e not in self.tl.events)],  # pyright: ignore[reportPrivateUsage]
            key=lambda e: e[0],
        )
        results = invariants.check_all(self.tl, exempt)
        self._write_result(results)  # pyright: ignore[reportPrivateUsage]
        bad = invariants.failed(results)
        assert not bad, "invarianter bröts: " + "; ".join(f"{r.id} {r.detail}" for r in bad)

    # --- the ground station ---------------------------------------------------------

    async def kill_ground(self) -> float:
        """``sd-ground`` gone: no heartbeat from 250, no SD-link, no APIs. The AI goes
        with it (it has nobody to talk to). Returns the host time."""
        await self.kill_brain()
        t = time.monotonic()
        if self._ground is not None:
            # The rig's pilot token is made anew by the next ground station (its name
            # is unique); the dead one's is retired as `sd-ground token revoke` would.
            if self.r.g.repo is not None:
                await self.r.g.repo.revoke_token("ci-harness")
            await self._ground.aclose()
            self._ground = None
        self.ground_alive = False
        return t

    async def start_ground(self, *, brain: bool = True) -> float:
        """A new ``sd-ground`` on the same database and data folder; returns the host
        time when it is up (MAVLink and SD-link)."""
        stack = contextlib.AsyncExitStack()
        port = free_port()
        sim = self.sim

        def onboard(_clock: WallClock, _url: str, token: str) -> _Kept:
            assert self._kept is not None
            # The new ground station's token; the executor reconnects with it.
            self._kept.side.link._client._token = token  # pyright: ignore[reportPrivateUsage]
            return self._kept

        ground = _vehicle(f"udpin:127.0.0.1:{sim.ports['ground_mavlink']}", Role.GROUND)
        r: Rig = await stack.enter_async_context(
            ground_stack(
                self._db.name,
                self._db.app_password,
                self._tmp,
                vehicle=ground,
                mav_clock=MonotonicClock(),
                onboard=onboard,
                sdlink_bind=f"127.0.0.1:{sim.ports['ground_sdlink']}",
                http={"commander_bind": f"127.0.0.1:{port}"},
            )
        )
        await stack.enter_async_context(serve_commander(r, port))
        r.commander.base_url = f"http://127.0.0.1:{port}"
        self._ground = stack
        self.r = r
        self.commander_port = port
        self.ground_alive = True
        _hook_ground(self)
        if brain:
            await self.start_brain()
        return time.monotonic()

    def watch_links(self) -> None:
        """Log every change of the ground's view of SD-link and MAVLink."""

        async def run() -> None:
            last: tuple[str, str] | None = None
            while True:
                if self.ground_alive:
                    now = (
                        self.r.link._state_of(VID),  # pyright: ignore[reportPrivateUsage]
                        str(self.r.g.vehicle_link.state()),
                    )
                    if now != last:
                        self.link_log.append((time.monotonic(), now[0], now[1]))
                        last = now
                await asyncio.sleep(0.05)

        self._tasks.append(asyncio.ensure_future(run()))

    def link_change(self, since: float, part: str, state: str) -> float | None:
        """Host time of the first change of ``part`` to ``state`` after ``since``:
        ``sdlink`` is up/degraded/lost, ``mavlink`` ok/down."""
        idx = 1 if part == "sdlink" else 2
        for row in self.link_log:
            if row[0] >= since and row[idx] == state:
                return row[0]
        return None


def _hook_ground(sc: Watched) -> None:
    """The timeline hooks of ``Scenario._record`` for a (new) ground station."""

    def sample(t: Any) -> None:
        sc.tl.samples.append(
            Sample(
                time.monotonic(),
                t.lat,
                t.lon,
                t.alt_rel_home_m,
                t.groundspeed_ms,
                t.fc_custom_mode,
                t.armed,
                t.landed_state == "on_ground",
            )
        )

    sc.r.core.vehicle.subscribe(sample)
    _record_writes(sc.r.core.vehicle, "ground", sc)
    add = sc.r.g.add_event

    def add_event(type_: str, **kw: Any) -> None:
        sc._bus.append((time.monotonic(), type_, dict(kw.get("payload") or {})))  # pyright: ignore[reportPrivateUsage]
        add(type_, **kw)

    sc.r.g.add_event = add_event  # type: ignore[method-assign]


@contextlib.asynccontextmanager
async def scenario_x(
    name: str, sim: Sim, db: Any, tmp_path: Path, script: str, *, brain: bool = True
) -> AsyncIterator[Watched]:
    """``scenario`` with a restartable ground station and the harness observer."""
    path = SCRIPTS / f"{script}.yaml"
    if not path.is_file():
        raise InfraError(f"manuset {path.name} saknas")
    holder: list[_Onboard] = []
    kept: list[_Kept] = []

    def first_onboard(_clock: WallClock, _url: str, token: str) -> _Kept:
        url = f"ws://127.0.0.1:{sim.ports['linksim_sdlink']}/sdlink/v1/control"
        side = _Onboard(sim, url, token)
        holder.append(side)
        kept.append(_Kept(side))
        return kept[0]

    stack = contextlib.AsyncExitStack()
    port = free_port()
    ground = _vehicle(f"udpin:127.0.0.1:{sim.ports['ground_mavlink']}", Role.GROUND)
    try:
        r: Rig = await stack.enter_async_context(
            ground_stack(
                db.name,
                db.app_password,
                tmp_path,
                vehicle=ground,
                mav_clock=MonotonicClock(),
                onboard=first_onboard,
                sdlink_bind=f"127.0.0.1:{sim.ports['ground_sdlink']}",
                http={"commander_bind": f"127.0.0.1:{port}"},
            )
        )
        await stack.enter_async_context(serve_commander(r, port))
        r.commander.base_url = f"http://127.0.0.1:{port}"
        sc = Watched(name, sim, r, holder[0], port, path)
        sc._db, sc._tmp, sc._ground, sc._kept = db, tmp_path, stack, kept[0]
        sc._record()  # pyright: ignore[reportPrivateUsage]

        def on_view(v: View) -> None:
            sc.tl.samples.append(
                Sample(
                    time.monotonic(),
                    v.lat,
                    v.lon,
                    v.alt_rel_home_m,
                    v.groundspeed_ms,
                    v.fc_custom_mode,
                    v.armed,
                    v.landed_state == "on_ground",
                )
            )

        sc.observer = Observer(sim, on_view)
        sc.observer.start()
        sc._tasks.append(asyncio.ensure_future(_sticks(sc)))  # pyright: ignore[reportPrivateUsage]
        sc.watch_links()
        try:
            if brain:
                await sc.start_brain()
            yield sc
        finally:
            for task in sc._tasks:  # pyright: ignore[reportPrivateUsage]
                task.cancel()
            await asyncio.gather(*sc._tasks, return_exceptions=True)  # pyright: ignore[reportPrivateUsage]
            await sc.kill_brain()
            sc.observer.close()
            if sc._ground is not None:
                await sc._ground.aclose()
                sc._ground = None
            await holder[0].close()
    finally:
        if stack is not None:
            await stack.aclose()


# --- shared checks -----------------------------------------------------------------


async def wait_mode(sc: Watched, modes: Any, limit_s: float, what: str) -> tuple[float, int]:
    """Until the flight controller reports one of ``modes``; (host time, mode)."""
    wanted = {int(m) for m in modes}
    await sc.until(lambda: sc.tel.fc_custom_mode in wanted, limit_s, what)
    return time.monotonic(), int(sc.tel.fc_custom_mode)


def first_choice_or_proof(sc: Watched, mode: int, chain: Any, since: float) -> bool:
    """The first mode of a failsafe chain, or a later one only after the flight
    controller said SmartRTL cannot be used (``expect.SRTL_UNAVAILABLE``) just before."""
    chain = [int(m) for m in chain]
    if mode == chain[0]:
        return True
    if mode not in chain or sc.observer is None:
        return False
    changed = next((t for t, m in sc.observer.mode_since(since) if m == mode), None)
    if changed is None:
        return False
    return bool(
        [t for t, _ in sc.observer.text_since(since, expect.SRTL_UNAVAILABLE) if t <= changed + 0.2]
    )


async def landed_home(sc: Watched, id_: str, since: float) -> None:
    """L-LAND: disarmed after landing within the limit, this close to home."""
    await sc.until(lambda: not sc.tel.armed, expect.L_LAND_S, "landad och desarmerad")
    took = time.monotonic() - since
    off = sc.from_home_m()
    sc.criterion(
        id_,
        took <= expect.L_LAND_S and off <= expect.L_LAND_M,
        "landar hemma (L-LAND)",
        measured=[round(took, 1), round(off, 2)],
        limit=[expect.L_LAND_S, expect.L_LAND_M],
    )
