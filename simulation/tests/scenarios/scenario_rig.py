"""The scenario framework on ArduCopter SITL with builtin physics.

Everything runs for real except the language model: ``sd-ground``'s mission flow with
its database, the gate and the emergency path; the simulated radio (``linksim``) carries
the ground's MAVLink channel (250/190) and SD-link; the onboard executor on 1/191
(through ``mavtap``); and ``sd-brain`` with ``ScriptedBackend`` playing the scenario's
script (``simulation/scenarios/scripts/SCxx.yaml``) over the real commander API. The
ground station, the executor and the commander run in the test process, so "killing" a
process stops it there: its tasks end, its sockets close, no more heartbeats.

* ``operator``: what the panel does, through the panel API with a pilot token;
* ``commander``: the commander API with the AI's token (``sd-brain``'s own path);
* ``faults``: kill the AI or the onboard computer, cut the radio or one flow, set the
  simulated battery, fly with the harness past the gate;
* the timeline: the track, every MAVLink message the three senders wrote, the ground's
  audit events, orders, gate decisions and charters; ``finish`` checks INV1-INV11 on it
  and writes ``result.json``.

Only test-rig failures raise ``InfraError`` (retried once); every other failure is real.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import sys
import time
import urllib.request
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "onboard"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ground"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "software/ground/tests/helpers"))

from brain_rig import free_port, run_brain, serve_commander  # pyright: ignore[reportMissingImports]
from flow_rig import VID, H, Rig, WallClock, ground_stack  # pyright: ignore[reportMissingImports]
from test_mission_flow_sitl import SitlOnboard  # pyright: ignore[reportMissingImports]
from test_onboard_sitl import Sim, _vehicle  # pyright: ignore[reportMissingImports]

from sd_mav import CopterMode, Role
from sd_obs.clock import MonotonicClock
from sd_safety.targets import Plane, distance_m
from sd_schemas.base import new_id
from sd_sim import expect, invariants
from sd_sim.errors import InfraError
from sd_sim.invariants import Charter, Sample, Sent, Timeline
from sd_sim.util import LOOPBACK_OPENER

__all__ = ["SCRIPTS", "VID", "Scenario", "scenario"]

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "simulation/scenarios/scripts"
FIXTURE = ROOT / "simulation/scenarios/demo_fixture.yaml"
TERMINAL = frozenset(
    {
        "done",
        "failed",
        "cancelled",
        "expired",
        "superseded",
        "rejected_by_gate",
        "rejected_onboard",
        "rejected_by_operator",
        "approval_expired",
    }
)
RAN = frozenset({"accepted", "running", "done", "failed"})
AI_RATE_S = 3.1  # the gate lets the AI act once per 3 s


def _fixture() -> dict[str, Any]:
    return dict(yaml.safe_load(FIXTURE.read_text(encoding="utf-8")))


@dataclass
class Scenario:
    name: str
    sim: Sim
    r: Rig
    onboard: SitlOnboard
    commander_port: int
    script: Path
    tl: Timeline = field(default_factory=Timeline)
    fixture: dict[str, Any] = field(default_factory=_fixture)
    mission_id: str = ""
    criteria: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    _brain: contextlib.AsyncExitStack | None = None
    brain: Any = None
    _onboard_alive: bool = True
    _tasks: list[asyncio.Future[Any]] = field(default_factory=list[asyncio.Future[Any]])
    _bus: list[tuple[float, str, dict[str, Any]]] = field(
        default_factory=list[tuple[float, str, dict[str, Any]]]
    )
    _started: float = field(default_factory=time.monotonic)

    # --- places -------------------------------------------------------------------

    @property
    def plane(self) -> Plane:
        return self.r.plane

    def at(self, east: float, north: float) -> tuple[float, float]:
        return self.r.at(east, north)

    def pin(self, name: str) -> tuple[float, float]:
        p = next(p for p in self.fixture["pins"] if p["id"] == name)
        return self.at(float(p["east_m"]), float(p["north_m"]))

    @property
    def tel(self) -> Any:
        return self.r.core.vehicle.telemetry

    def mode(self) -> int | None:
        return self.tel.fc_custom_mode

    def from_home_m(self) -> float:
        return distance_m(self.r.home[0], self.r.home[1], self.tel.lat, self.tel.lon)

    def distance_to(self, lat: float, lon: float) -> float:
        return distance_m(lat, lon, self.tel.lat, self.tel.lon)

    # --- waiting ------------------------------------------------------------------

    async def until(self, cond: Callable[[], Any], limit_s: float, what: str) -> float:
        start = time.monotonic()
        while True:
            value = cond()
            if asyncio.iscoroutine(value):
                value = await value
            if value:
                return time.monotonic() - start
            if time.monotonic() - start > limit_s:
                t = self.tel
                raise AssertionError(
                    f"{what}: inte inom {limit_s:.0f} s (läge {t.fc_mode}, "
                    f"{t.alt_rel_home_m:.1f} m, {t.groundspeed_ms:.1f} m/s, "
                    f"{self.from_home_m():.0f} m från hem, "
                    f"utförare {self.onboard.ex.state})"
                )
            await asyncio.sleep(0.05)

    async def still(self, cond: Callable[[], bool], hold_s: float, what: str) -> None:
        """``cond`` holds for the whole of ``hold_s``."""
        end = time.monotonic() + hold_s
        while time.monotonic() < end:
            assert cond(), f"{what} (läge {self.tel.fc_mode}, {self.tel.alt_rel_home_m:.1f} m)"
            await asyncio.sleep(0.1)

    def criterion(self, id_: str, passed: bool, text: str, **measure: Any) -> None:
        self.criteria.append({"id": id_, "text": text, "passed": passed, **measure})
        assert passed, f"{id_}: {text} {measure}"

    # --- the operator (panel API) -------------------------------------------------

    async def open_mission(self) -> str:
        self.mission_id = await self.r.open_mission()
        return self.mission_id

    async def say(self, text: str) -> dict[str, Any]:
        res = await self.r.panel.post(
            f"/api/v1/missions/{self.mission_id}/chat",
            json={"client_msg_id": str(new_id()), "text": text},
            headers=H,
        )
        assert res.status_code == 201, res.text
        return dict(res.json())

    async def press(self, action: str) -> float:
        """STOP / FLY HOME / Resume; returns the host time of the call."""
        t = time.monotonic()
        res = await self.r.panel.post(
            f"/api/v1/vehicles/{VID}/actions/{action}",
            json={"client_action_id": str(new_id())},
            headers=H,
        )
        assert res.status_code == 202, res.text
        return t

    async def order(self, skill: str, **params: Any) -> Any:
        """The pilot's own order (an order button)."""
        return await self.r.panel.post(
            f"/api/v1/missions/{self.mission_id}/commands",
            json={"client_command_id": str(new_id()), "skill": skill, "params": params},
            headers=H,
        )

    async def approvals(self, kind: str | None = None) -> list[dict[str, Any]]:
        res = await self.r.panel.get(f"/api/v1/missions/{self.mission_id}/approvals")
        assert res.status_code == 200, res.text
        return [a for a in res.json() if kind is None or a["kind"] == kind]

    async def pending(self, kind: str, limit_s: float = 30) -> dict[str, Any]:
        found: list[dict[str, Any]] = []

        async def seen() -> bool:
            found[:] = [a for a in await self.approvals(kind) if a["status"] == "pending"]
            return bool(found)

        try:
            await self.until(seen, limit_s, f"ett väntande {kind}-kort")
        except AssertionError as exc:
            raise AssertionError(f"{exc}; {await self.describe()}") from None
        return found[-1]

    async def describe(self) -> str:
        """The mission, the last chat lines and orders, for a failure message."""
        if not self.mission_id:
            return "inget uppdrag"
        status = await self.r.status(self.mission_id)
        chat = [(m["role"], m["text"][:80]) for m in (await self.chat())[-6:]]
        cmds = [(c["skill"], c["status"], c["status_reason"]) for c in await self.commands()]
        ai = "" if self.brain is None else f" (AI:ns uppdrag {self.brain.loop.mission_id})"
        return f"uppdrag {status}, AI {self.r.g.brain_state}{ai}, chatt {chat}, order {cmds}"

    async def decide(self, approval: dict[str, Any], decision: str, **extra: Any) -> Any:
        return await self.r.panel.post(
            f"/api/v1/approvals/{approval['id']}/decision",
            json={"decision": decision, "subject_version": approval["subject_version"], **extra},
            headers=H,
        )

    async def approve_charter(self, autonomy_mode: str = "WITHIN_CHARTER") -> None:
        card = await self.pending("charter")
        res = await self.decide(card, "approve", autonomy_mode=autonomy_mode)
        assert res.status_code == 200, res.text
        await self.r.wait_status(self.mission_id, "READY", 60)
        await self.until(self.onboard.ex.preflight_ok, 90, "förflygningskontrollen ombord")
        await self.until(
            lambda: (snap := self.r.flow.snapshot(VID)[0]) is not None and snap.preflight_ok,
            10,
            "förflygningskontrollen i markens lägesbild",
        )

    async def confirm_clear(self) -> None:
        res = await self.r.clearance(self.mission_id)
        assert res.status_code == 200, res.text

    async def commands(self) -> list[dict[str, Any]]:
        res = await self.r.panel.get(f"/api/v1/missions/{self.mission_id}/commands")
        assert res.status_code == 200, res.text
        return list(res.json())

    async def command(self, command_id: str) -> dict[str, Any]:
        return next(c for c in await self.commands() if c["id"] == command_id)

    async def wait_command(
        self,
        skill: str,
        status: str | Sequence[str],
        limit_s: float,
        *,
        issuer_kind: str | None = None,
        after: int = 0,
    ) -> dict[str, Any]:
        """The first order with ``skill`` (from the ``after``-th on) that reaches
        ``status``; fails at once when it ends in another terminal status."""
        want = {status} if isinstance(status, str) else set(status)
        found: list[dict[str, Any]] = []

        async def reached() -> bool:
            cmds = [
                c
                for c in await self.commands()
                if c["skill"] == skill and (issuer_kind is None or c["issuer_kind"] == issuer_kind)
            ]
            cmds.sort(key=lambda c: c["created_at"])
            for c in cmds[after:]:
                if c["status"] in want:
                    found.append(c)
                    return True
                if c["status"] in TERMINAL:
                    raise AssertionError(
                        f"{skill}: {c['status']} {c['status_reason']} (väntade {sorted(want)})"
                    )
            return False

        await self.until(reached, limit_s, f"{skill} {sorted(want)}")
        return found[0]

    async def wait_done(self, skill: str, limit_s: float, **kw: Any) -> dict[str, Any]:
        return await self.wait_command(skill, "done", limit_s, **kw)

    async def chat(self) -> list[dict[str, Any]]:
        res = await self.r.panel.get(f"/api/v1/missions/{self.mission_id}/chat")
        assert res.status_code == 200, res.text
        body = res.json()
        return list(body["items"] if isinstance(body, dict) else body)

    async def wait_chat(self, pred: Callable[[dict[str, Any]], bool], limit_s: float) -> None:
        async def seen() -> bool:
            return any(pred(m) for m in await self.chat())

        await self.until(seen, limit_s, "chattrad")

    async def gate(self, command_id: str) -> list[Any]:
        """The command's gate decisions (both sides), oldest first."""
        rows = await self.r.flow.repo.gate_decisions(self.mission_id, 50, command_id=command_id)
        return list(reversed(rows))

    async def codes(self, command_id: str) -> list[str]:
        return [c for g in await self.gate(command_id) for c in g.reason_codes]

    def steering_since(self, index: int, senders: Sequence[str] = ("onboard",)) -> list[Sent]:
        """Messages since ``tl.sent[index]`` that steer or change the mode."""
        return [
            m
            for m in self.tl.sent[index:]
            if m.sender in senders and (m.steers() or m.mode is not None)
        ]

    async def drift(self, hold_s: float = expect.L_DRIFT_S) -> tuple[float, float]:
        """Horizontal and vertical movement over the next ``hold_s`` (max from start)."""
        lat, lon, alt = self.tel.lat, self.tel.lon, self.tel.alt_rel_home_m
        h = v = 0.0
        end = time.monotonic() + hold_s
        while time.monotonic() < end:
            h = max(h, distance_m(lat, lon, self.tel.lat, self.tel.lon))
            v = max(v, abs(self.tel.alt_rel_home_m - alt))
            await asyncio.sleep(0.1)
        return h, v

    async def assert_drift(self, id_: str, hold_s: float = expect.L_DRIFT_S) -> None:
        h, v = await self.drift(hold_s)
        self.criterion(
            id_,
            h <= expect.L_HOLD_H_M and v <= expect.L_HOLD_V_M,
            "står still (L-DRIFT)",
            measured=[round(h, 2), round(v, 2)],
            limit=[expect.L_HOLD_H_M, expect.L_HOLD_V_M],
            unit="m",
        )

    # --- the AI's own path (commander API, brain token) ---------------------------

    async def situation(self) -> dict[str, Any]:
        res = await self.r.commander.get(f"/commander/v1/missions/{self.mission_id}/situation")
        assert res.status_code == 200, res.text
        return dict(res.json())

    async def commander_order(self, skill: str, **params: Any) -> Any:
        """An order on the AI's own path (commander API, brain token), as ``sd-brain``
        sends it: the gate decides (201 for every outcome)."""
        situation = await self.situation()
        based_on = None
        if situation["snapshot_state_epoch"] is not None:
            based_on = {
                "state_epoch": situation["snapshot_state_epoch"],
                "state_seq": situation["snapshot_state_seq"],
            }
        charter = situation["charter"]
        return await self.r.commander.post(
            f"/commander/v1/missions/{self.mission_id}/commands",
            json={
                "command_id": str(new_id()),
                "skill": skill,
                "params": params,
                "rationale_sv": "Testkoden.",
                "in_reply_to": None,
                "based_on": based_on,
                "charter_version": None if charter is None else charter["version"],
                "backend": "scripted",
                "ai_config_hash": "scenario-test",
            },
        )

    # --- common steps -------------------------------------------------------------

    async def ready(self, charter: str = "charter_test", autonomy: str = "WITHIN_CHARTER") -> None:
        """Open, let the AI propose the fixture's charter, approve it: READY."""
        await self.open_mission()
        await self.say(f"Använd ramen {charter}.")
        await self.approve_charter(autonomy)

    async def arrange_airborne(self, charter: str = "charter_test") -> None:
        """07's shared steps: in LOITER at 40 +- 2 m above home, pilot WS open."""
        await self.ready(charter)
        await self.confirm_clear()
        await self.say("Starta.")
        await self.wait_done("takeoff", 90, issuer_kind="brain")
        await self.until(lambda: abs(self.tel.alt_rel_home_m - 40) <= 2, 30, "40 m över hem")
        await self.r.wait_status(self.mission_id, "ACTIVE", 10)

    # --- faults ---------------------------------------------------------

    def _linksim(self, path: str, body: dict[str, Any]) -> None:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.sim.ports['linksim_control']}{path}",
            data=json.dumps(body).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.sim.stack.linksim_token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with LOOPBACK_OPENER.open(req, timeout=2) as resp:
                if resp.status not in (200, 204):
                    raise InfraError(f"linksim {path}: {resp.status}")
        except OSError as exc:
            raise InfraError(f"linksim {path}: {exc}") from exc

    async def radio(self, up: bool) -> None:
        """All three flows of the simulated radio on or off."""
        await asyncio.to_thread(self._linksim, "/radio", {"state": "up" if up else "down"})

    async def flow(self, name: str, up: bool) -> None:
        await asyncio.to_thread(self._linksim, f"/flows/{name}", {"state": "up" if up else "down"})

    async def battery(self, pct: float) -> None:
        self._harness_sent("COMMAND_LONG", command=42651, params=(1.0, pct))
        await asyncio.to_thread(self.sim.harness.battery_set_remaining, pct)

    def _harness_sent(self, type_: str, **kw: Any) -> None:
        t = self.tel
        self.tl.sent.append(
            Sent(
                time.monotonic(),
                "harness",
                type_,
                alt_m=t.alt_rel_home_m,
                armed=t.armed,
                landed=t.landed_state == "on_ground",
                **kw,
            )
        )

    async def kill_brain(self) -> None:
        """``sd-brain`` gone: its stream, turns and HTTP client end."""
        if self._brain is not None:
            await self._brain.aclose()
            self._brain = None

    async def start_brain(self) -> None:
        stack = contextlib.AsyncExitStack()
        self.brain = await stack.enter_async_context(
            run_brain(
                self.r,
                self.commander_port,
                backend={"kind": "scripted", "scripted_file": str(self.script)},
            )
        )
        self._brain = stack

    async def kill_onboard(self) -> None:
        """``sd-onboard`` gone: no executor, no SD-link, no heartbeat from 1/191."""
        if self._onboard_alive:
            self._onboard_alive = False
            await self.onboard.close()

    # --- recording ----------------------------------------------------------------

    def _record(self) -> None:
        tl = self.tl

        def sample(t: Any) -> None:
            tl.samples.append(
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

        self.r.core.vehicle.subscribe(sample)
        _record_writes(self.r.core.vehicle, "ground", self)
        _record_writes(self.onboard.vehicle, "onboard", self)
        add = self.r.g.add_event

        def add_event(type_: str, **kw: Any) -> None:
            self._bus.append((time.monotonic(), type_, dict(kw.get("payload") or {})))
            add(type_, **kw)

        self.r.g.add_event = add_event  # type: ignore[method-assign]

    async def _snapshot_db(self) -> None:
        """Orders, gate decisions and charters from the database for the invariants."""
        if not self.mission_id:
            return
        tl = self.tl
        # Mission events from the database (charter approvals, clearances), on the host
        # clock; the vehicle's events (STOP, FLY HOME, ...) as the ground added them.
        offset = time.monotonic() - self.r.clock.utc_now().timestamp()
        res = await self.r.panel.get(f"/api/v1/missions/{self.mission_id}/events")
        assert res.status_code == 200, res.text
        stored = [
            (dt.datetime.fromisoformat(e["t"]).timestamp() + offset, e["type"], e["payload"])
            for e in res.json()
            if e["type"] in ("charter.approved", "clearance.confirmed")
        ]
        tl.events = sorted([*self._bus, *stored], key=lambda e: e[0])
        tl.clearances = [t for t, kind, _ in tl.events if kind == "clearance.confirmed"]
        tl.commands = await self.commands()
        repo = self.r.flow.repo
        gates: list[dict[str, Any]] = []
        for cmd in tl.commands:
            for g in await repo.gate_decisions(self.mission_id, 50, command_id=cmd["id"]):
                gates.append({"command_id": cmd["id"], "side": g.side, "decision": g.decision})
            if cmd["status"] in RAN or cmd["status_reason"] in ("OPERATOR_RTL", "OPERATOR_STOP"):
                tl.accepted[str(cmd["id"])] = 0.0
        tl.gates = gates
        res = await self.r.panel.get(f"/api/v1/missions/{self.mission_id}/charters")
        rows = sorted(
            (c for c in res.json() if c.get("approved_at")), key=lambda c: c["approved_at"]
        )
        for c in rows:
            since = dt.datetime.fromisoformat(c["approved_at"]).timestamp() + offset
            body = c["body"]
            pilot = body["pilot_position"]
            tl.charters.append(
                Charter(
                    since=since,
                    area=body["area"],
                    pilot=(pilot["lat"], pilot["lon"]),
                    alt_max_m=body["alt_max_agl_m"],
                    speed_max_ms=body["max_speed_ms"],
                    range_m=body["max_range_from_pilot_m"],
                    buffer_m=body["geofence_buffer_m"],
                    autonomy_mode=str(body["autonomy_mode"]),
                    approved_by=c.get("approved_by"),
                    status=c.get("status", "approved"),
                    # Installed once its fence was verified; a reinstall after an
                    # SD-link outage makes the row briefly "pending" again.
                    install_status=(
                        "installed" if c.get("fence_verified_at") else c["install_status"]
                    ),
                )
            )

    async def finish(
        self,
        exempt: Sequence[str] = (),
        *,
        harness_extra: Sequence[str] = (),
        allow_mismatch: bool = False,
        inside_m: float | None = None,
    ) -> None:
        """INV1-INV11 on the recorded timeline, then ``result.json``."""
        await self._snapshot_db()
        self.tl.inside_m = inside_m
        results = invariants.check_all(
            self.tl, exempt, harness_extra=harness_extra, allow_mismatch=allow_mismatch
        )
        self._write_result(results)
        bad = invariants.failed(results)
        assert not bad, "invarianter bröts: " + "; ".join(f"{r.id} {r.detail}" for r in bad)

    def _write_result(self, results: list[invariants.Result]) -> None:
        built = [c for c in self.criteria if not c.get("not_built")]
        out = {
            "schema": 1,
            "scenario": self.name.split("[")[0].removeprefix("test_").upper(),
            "variant": self.name.split("[")[1].rstrip("]") if "[" in self.name else "default",
            "passed": all(c["passed"] for c in built) and not invariants.failed(results),
            "not_built": len(self.criteria) - len(built),
            "duration_s": round(time.monotonic() - self._started, 1),
            "physics": "builtin",
            "brain": "scripted",
            "criteria": self.criteria,
            "invariants": [r.__dict__ for r in results],
            "infra_failure": None,
        }
        run_dir = self.sim.stack.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))


class _Onboard(SitlOnboard):
    """``SitlOnboard`` that a test may stop early (a dead ``sd-onboard``)."""

    closed = False

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            await super().close()


def _record_writes(vehicle: Any, sender: str, sc: Scenario) -> None:
    write = vehicle._transport.write_message  # pyright: ignore[reportPrivateUsage]

    def recording(msg: Any) -> None:
        t = sc.tel
        kind = msg.get_type()
        params: tuple[float, ...] = ()
        command = None
        if kind == "COMMAND_LONG":
            command = int(msg.command)
            params = tuple(float(getattr(msg, f"param{i}")) for i in range(1, 8))
        param_id = None
        if kind in ("PARAM_SET", "PARAM_EXT_SET"):
            raw = msg.param_id
            param_id = (raw.decode() if isinstance(raw, bytes) else str(raw)).rstrip("\x00")
        sc.tl.sent.append(
            Sent(
                time.monotonic(),
                sender,
                kind,
                command=command,
                params=params,
                mission_type=getattr(msg, "mission_type", None),
                param_id=param_id,
                alt_m=t.alt_rel_home_m,
                armed=t.armed,
                landed=t.landed_state == "on_ground",
            )
        )
        write(msg)

    vehicle._transport.write_message = recording  # pyright: ignore[reportPrivateUsage]


async def _sticks(sc: Scenario) -> None:
    """The pilot's RC transmitter: throttle centred while flying, down on the ground."""
    while True:
        t = sc.tel
        sc.sim.rc.throttle = 1500 if t.armed and t.alt_rel_home_m > 1.0 else 1000
        await asyncio.sleep(0.1)


@contextlib.asynccontextmanager
async def scenario(
    name: str,
    sim: Sim,
    db: Any,
    tmp_path: Path,
    script: str,
    *,
    brain: bool = True,
    ground_config: dict[str, Any] | None = None,
) -> AsyncIterator[Scenario]:
    """One scenario: ground, onboard and (unless ``brain=False``) the AI with ``script``.
    ``ground_config`` adds to the ground station's test configuration."""
    path = SCRIPTS / f"{script}.yaml"
    if not path.is_file():
        raise InfraError(f"manuset {path.name} saknas")
    ground = _vehicle(f"udpin:127.0.0.1:{sim.ports['ground_mavlink']}", Role.GROUND)
    port = free_port()
    holder: list[SitlOnboard] = []

    def onboard(_clock: WallClock, _url: str, token: str) -> SitlOnboard:
        # Through the simulated radio: linksim's SD-link port forwards to the ground.
        url = f"ws://127.0.0.1:{sim.ports['linksim_sdlink']}/sdlink/v1/control"
        side = _Onboard(sim, url, token)
        holder.append(side)
        return side

    async with (
        ground_stack(
            db.name,
            db.app_password,
            tmp_path,
            vehicle=ground,
            mav_clock=MonotonicClock(),
            onboard=onboard,
            sdlink_bind=f"127.0.0.1:{sim.ports['ground_sdlink']}",
            **_merged(ground_config or {}, {"http": {"commander_bind": f"127.0.0.1:{port}"}}),
        ) as r,
        serve_commander(r, port),
    ):
        # The commander API's host check expects the port it is served on.
        r.commander.base_url = f"http://127.0.0.1:{port}"
        sc = Scenario(name, sim, r, holder[0], port, path)
        sc._record()
        sc._tasks.append(asyncio.ensure_future(_sticks(sc)))
        try:
            if brain:
                await sc.start_brain()
            yield sc
        finally:
            for task in sc._tasks:
                task.cancel()
            await asyncio.gather(*sc._tasks, return_exceptions=True)
            await sc.kill_brain()


def _merged(extra: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def mode_name(mode: int | None) -> str:
    return "?" if mode is None else CopterMode(mode).name
