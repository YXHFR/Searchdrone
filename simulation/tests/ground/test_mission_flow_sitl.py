"""The ground station's mission flow against ArduCopter SITL (builtin physics, headless).

Everything but the AI is real: ``sd-ground``'s mission flow with its database (a fresh
migrated ``searchdrone_test_*`` database, as in the contract tests), ``GroundMav`` on the
ground's MAVLink channel (250/190, through linksim), the SD-link server, and the onboard
executor in this process on 1/191 (through mavtap), connected to that server. The AI is
``helpers/script_driver.py`` on ``/commander/v1``; the pilot is the panel API with a
``ci-harness`` pilot token.

* ``/api/v1/health`` is ready with real MAVLink and real SD-link;
* after the pilot approves the charter, the ground uploads the fence polygon to the
  flight controller, reads it back and marks it verified; the onboard computer accepts it;
* a scripted flight: take off, fly to P1, fly home and land; the mission closes with its
  files;
* limits raised by the pilot on the ground move ArduCopter's fence parameters, and the
  onboard computer's preflight still passes.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "onboard"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "software/ground/tests/helpers"))

from flow_rig import VID, H, Rig, WallClock, ground_stack  # pyright: ignore[reportMissingImports]
from script_driver import ScriptDriver, load_steps  # pyright: ignore[reportMissingImports]
from test_onboard_sitl import (  # pyright: ignore[reportMissingImports]
    PARM,
    Sim,
    _before_executor,
    _vehicle,
    sim,
)

from sd_ground.mav.fence import read_params
from sd_mav import Role, SendOrigin
from sd_mav.mission import FENCE, download, fence_hash
from sd_obs.clock import MonotonicClock
from sd_onboard.executor.executor import Executor, ExecutorOptions
from sd_onboard.io.params import effective_params
from sd_onboard.io.persistence import StateStore
from sd_onboard.link_client import OnboardLink
from sd_safety.limits import fc_fence_params
from sd_safety.targets import distance_m
from sd_schemas.executor import ExecutorState as S
from sd_schemas.safety import SafetyLimits

ROOT = Path(__file__).resolve().parents[3]


def _db_fixtures() -> Any:
    """The contract tests' database fixtures, loaded under their own module name."""
    path = ROOT / "software/ground/tests/contract/conftest.py"
    spec = importlib.util.spec_from_file_location("ground_db_fixtures", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_db = _db_fixtures()
passwords, fresh_db, migrated_db = _db.passwords, _db.fresh_db, _db.migrated_db

pytestmark = [pytest.mark.sitl, pytest.mark.slow]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

SCRIPT = ROOT / "software/ground/tests/helpers/scripts/sitl_flight.yaml"
AREA_HALF_M = 200.0
P1 = (100.0, 50.0)  # east, north of home (m)


class SitlOnboard:
    """The onboard executor on 1/191 against SITL, with SD-link to the ground's server."""

    def __init__(
        self,
        sim: Sim,
        url: str,
        token: str,
        *,
        param_layers: tuple[Path, ...] = (),
        state_dir: Path | None = None,
        **options: Any,
    ) -> None:
        """``state_dir``: the onboard computer's saved state and find hovers, as on 1/191."""
        self.clock = MonotonicClock()
        holder: list[Executor] = []
        self.vehicle = _vehicle(
            f"udpin:127.0.0.1:{sim.ports['onboard_mavlink']}",
            Role.ONBOARD,
            on_event=lambda k, d: holder[0].on_vehicle_event(k, d) if holder else None,
            heartbeat_allowed=lambda: bool(holder) and holder[0].loop_alive(),
            send_gate=lambda o: holder[0].send_gate(o) if holder else _before_executor(o),
        )
        self.link = OnboardLink(url=url, token=token, vehicle_id=VID, clock=self.clock)
        self.ex = Executor(
            self.vehicle,
            self.clock,
            self.link,
            ExecutorOptions(
                vehicle_id=VID,
                expected_params=effective_params([PARM, *param_layers]),
                **{"require_rc": False, **options},  # no RC transmitter in SITL and CI
            ),
            None if state_dir is None else StateStore(state_dir, self.clock),
        )
        holder.append(self.ex)
        self.link.executor = self.ex
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Future[Any]] = []

    async def start(self, home: tuple[float, float]) -> None:
        await self.vehicle.start()
        if not await self.vehicle.wait_connected(20):
            raise AssertionError("1/191 has no MAVLink")
        self._tasks += [
            asyncio.ensure_future(self.ex.run(self._stop)),
            asyncio.ensure_future(self.link.run(self._stop)),
        ]

    async def close(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.ex.close()
        await self.vehicle.close()


@pytest.fixture
async def stack(sim: Sim, migrated_db: Any, tmp_path: Path) -> AsyncIterator[Rig]:
    ground = _vehicle(f"udpin:127.0.0.1:{sim.ports['ground_mavlink']}", Role.GROUND)

    def onboard(_clock: WallClock, url: str, token: str) -> SitlOnboard:
        return SitlOnboard(sim, url, token)

    async with ground_stack(
        migrated_db.name,
        migrated_db.app_password,
        tmp_path,
        vehicle=ground,
        mav_clock=MonotonicClock(),
        onboard=onboard,
    ) as r:
        yield r


async def _ready(r: Rig) -> str:
    """Open a mission; the pilot proposes and approves the charter: fence up, READY."""
    mid = await r.open_mission()
    res = await r.panel.post(
        f"/api/v1/missions/{mid}/charters",
        json={"area": r.square(AREA_HALF_M), "reason_sv": "SITL"},
        headers=H,
    )
    assert res.status_code == 201, res.text
    card = res.json()
    res = await r.approve(str(card["approval_id"]), card["version"])
    assert res.status_code == 200, res.text
    await r.wait_status(mid, "READY", 90)
    await r.wait_snapshot(S.PREFLIGHT, 60)
    ex = r.onboard.ex
    await r.until(ex.preflight_ok, 90, "onboard preflight")
    await r.until(
        lambda: (snap := r.flow.snapshot(VID)[0]) is not None and snap.preflight_ok,
        10,
        "preflight ok in the ground's snapshot",
    )
    return mid


async def _command(r: Rig, mid: str, command_id: str) -> dict[str, Any]:
    cmds = (await r.panel.get(f"/api/v1/missions/{mid}/commands")).json()
    return next(c for c in cmds if c["id"] == command_id)


async def _until_done(r: Rig, mid: str, command_id: str, limit_s: float) -> None:
    start = time.monotonic()
    while True:
        cmd = await _command(r, mid, command_id)
        if cmd["status"] == "done":
            return
        if cmd["status"] in ("failed", "rejected_by_onboard", "rejected_by_gate", "cancelled"):
            raise AssertionError(f"{cmd['skill']}: {cmd['status']} {cmd['status_reason']}")
        if time.monotonic() - start > limit_s:
            raise AssertionError(f"{cmd['skill']}: not done within {limit_s} s ({cmd['status']})")
        await asyncio.sleep(0.2)


@pytest.mark.budget_s(180)
async def test_health_ready_and_fence_on_sitl(stack: Rig, sim: Sim) -> None:
    r = stack
    await r.until(
        lambda: r.g.vehicle_link.state() == "ok" and r.g.sdlink.state() == "ok",
        30,
        "MAVLink and SD-link ok",
    )
    health = await r.panel.get("/api/v1/health")
    assert health.status_code == 200 and health.json()["ready"] is True, health.text

    mid = await _ready(r)
    charter = (await r.mission(mid))["charter"]
    assert charter["fence_verified_at"] is not None
    assert charter["install_status"] == "installed"
    # What the flight controller holds is what the ground verified and the onboard
    # computer accepted.
    stored = await r.flow.mav.call(lambda v: download(v, FENCE, origin=SendOrigin.GROUND), 10)
    assert len(stored) >= 3
    assert fence_hash(stored) == charter["body"]["fence_hash"]


@pytest.mark.budget_s(300)
async def test_scripted_flight_on_sitl(stack: Rig, sim: Sim) -> None:
    r = stack
    mid = await _ready(r)
    d = ScriptDriver(r.commander, mid, {"p1": r.at(*P1)})
    steps = load_steps(SCRIPT)

    res = await r.clearance(mid)
    assert res.status_code == 200, res.text
    takeoff = await d.step(steps[0])
    await _until_done(r, mid, takeoff.json()["command_id"], 90)
    sim.rc.throttle = 1500  # in the air: the pilot centres the throttle stick
    await r.wait_status(mid, "ACTIVE")
    assert (await r.mission(mid))["clearance"] is None  # consumed by the take-off

    await asyncio.sleep(3.1)  # the gate's rate limit: one AI action per 3 s
    goto = await d.step(steps[1])
    await _until_done(r, mid, goto.json()["command_id"], 90)
    tel = r.core.vehicle.telemetry
    p1 = r.at(*P1)
    assert distance_m(p1[0], p1[1], tel.lat, tel.lon) < 10.0

    await asyncio.sleep(3.1)
    await d.step(steps[2])
    await r.wait_status(mid, "RETURNING", 10)
    sim.rc.throttle = 1000
    await r.wait_status(mid, "LANDED", 180)
    await r.wait_status(mid, "CLOSED", 60)
    tel = r.core.vehicle.telemetry
    assert distance_m(r.home[0], r.home[1], tel.lat, tel.lon) < 5.0
    files = (await r.panel.get(f"/api/v1/missions/{mid}/files")).json()
    assert {"mission.json", "events.jsonl", "exports/track.geojson"} <= {
        f["rel_path"] for f in files
    }
    gates = (await r.panel.get(f"/api/v1/missions/{mid}/gate_decisions")).json()
    assert [g["decision"] for g in gates] == ["ALLOW", "ALLOW", "ALLOW"]


@pytest.mark.budget_s(180)
async def test_raised_limits_move_the_fc_fence_on_sitl(stack: Rig, sim: Sim) -> None:
    r = stack
    raised = {
        "max_alt_agl_m": 150,
        "max_speed_ms": 15,
        "max_range_from_pilot_m": 800,
        "battery_rtl_pct": 25,
    }
    mid = await r.open_mission()
    res = await r.panel.put(f"/api/v1/missions/{mid}/limits", json=raised, headers=H)
    assert res.status_code == 200, res.text
    res = await r.panel.post(
        f"/api/v1/missions/{mid}/charters",
        json={"area": r.square(AREA_HALF_M), "reason_sv": "SITL"},
        headers=H,
    )
    assert res.status_code == 201, res.text
    card = res.json()
    res = await r.approve(str(card["approval_id"]), card["version"])
    assert res.status_code == 200, res.text
    await r.wait_status(mid, "READY", 90)
    want = fc_fence_params(SafetyLimits.model_validate(raised))
    assert r.flow.mav is not None
    got = await r.flow.mav.call(lambda v: read_params(v, want), 20)
    for name, value in want.items():
        assert got[name] is not None and abs(got[name] - value) < 0.01, (name, got[name])
    await r.until(r.onboard.ex.preflight_ok, 90, "onboard preflight with the raised fence")
