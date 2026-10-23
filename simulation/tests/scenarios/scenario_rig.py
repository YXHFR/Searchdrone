"""The rig for the scenario tests: the whole stack but the language model, on ArduCopter SITL.

Real: ArduCopter SITL (builtin physics) with the project's parameter layers, the ground
station's mission flow with its database, gate, emergency path and SD-link server
(``flow_rig.ground_stack``, MAVLink 250/190 through linksim), the onboard executor on
1/191 (through mavtap) and, where a scenario needs one, ``sd-brain`` itself on the
commander API over real HTTP and WebSocket, with the ``ScriptedBackend`` and the
scenario's script from ``simulation/scenarios/scripts/``. The pilot is the panel API with
a pilot CLI token (``operator:ci-harness``); an observer has its own token.

``Scenario`` records what the ground station sees of the flight controller (one sample a
tenth of a second) and checks the invariants in ``sd_sim.invariants`` after the scenario
(``await sc.check_invariants()``). Sticks go to ``VehicleActions.manual_input``, the
service behind the panel's stick input. ``sd-brain`` dying is played by cancelling its
tasks without a goodbye, so the ground station notices only that its heartbeat stops.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import sys
import time
import urllib.request
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import yaml
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "onboard"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ground"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "software/ground/tests/helpers"))

import flow_rig  # pyright: ignore[reportMissingImports]
import test_mission_flow_sitl  # pyright: ignore[reportMissingImports]
import test_onboard_sitl  # pyright: ignore[reportMissingImports]
from brain_rig import loopback_listener, serve_commander  # pyright: ignore[reportMissingImports]
from flow_rig import BRAIN, Rig, WallClock, ground_stack  # pyright: ignore[reportMissingImports]
from test_mission_flow_sitl import SitlOnboard  # pyright: ignore[reportMissingImports]
from test_onboard_sitl import Sim, _vehicle  # pyright: ignore[reportMissingImports]

from sd_brain.app import Brain, _idle_ticker, _startup, _stream, build
from sd_brain.config import BrainConfig
from sd_brain.secrets import Secrets
from sd_geo.coverage import coverage_fraction
from sd_ground.auth import Principal, SessionRecord, new_token
from sd_ground.auth import hash_token as hash_session_token
from sd_ground.mav.thread import vehicle_events
from sd_link_testing import LossyProxy
from sd_mav import Role
from sd_obs.clock import MonotonicClock
from sd_safety.targets import Plane, distance_m
from sd_schemas.api import ManualControlInput
from sd_schemas.base import new_id
from sd_schemas.skills import SKILL_REGISTRY
from sd_sim.invariants import Sample, check_all
from sd_sim.report import measure
from sd_sim.util import LOOPBACK_OPENER

__all__ = [
    "FIXTURE",
    "PILOT",
    "SCRIPTS",
    "VID",
    "H",
    "Scenario",
    "fresh_db",
    "migrated_db",
    "passwords",
    "scenario",
    "sim",
    "sim_stack",
]

# Re-exported for the scenario tests (the fixtures by name, as pytest finds them).
VID: str = flow_rig.VID
H: dict[str, str] = flow_rig.H
fresh_db: Any = test_mission_flow_sitl.fresh_db
migrated_db: Any = test_mission_flow_sitl.migrated_db
passwords: Any = test_mission_flow_sitl.passwords
sim: Any = test_onboard_sitl.sim
sim_stack: Any = test_onboard_sitl.sim_stack

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "simulation/scenarios/scripts"
FIXTURE: dict[str, Any] = yaml.safe_load(
    (ROOT / "simulation/scenarios/demo_fixture.yaml").read_text(encoding="utf-8")
)
PILOT = Principal(role="pilot", issuer_ref="operator:ci-harness", auth_mode="cli")
LANDING_MODES = frozenset({"RTL", "SMART_RTL", "LAND"})
# Channel 5 PWM in the middle of each mode-switch band (ArduCopter: <= 1230 FLTMODE1,
# <= 1360 FLTMODE2, <= 1490 FLTMODE3, <= 1620 FLTMODE4, <= 1749 FLTMODE5, else FLTMODE6).
RC_PWM: dict[str, int] = {"LAND": 1295, "RTL": 1425, "LOITER": 1685}
TERMINAL = frozenset(
    {
        "rejected_by_gate",
        "rejected_by_operator",
        "approval_expired",
        "rejected_onboard",
        "done",
        "failed",
        "superseded",
        "expired",
        "cancelled",
    }
)


def charter_body(name: str = "charter_test", **overrides: Any) -> dict[str, Any]:
    """A charter from the demo fixture as the pilot's proposal (east/north -> WGS84)."""
    c = dict(FIXTURE["charters"][name])
    base = c.pop("base", None)
    if base is not None:
        c = {**FIXTURE["charters"][base], **c}
        c.pop("base", None)
    home = FIXTURE["home"]
    plane = Plane(float(home["lat"]), float(home["lon"]))
    ring = [list(reversed(plane.latlon(e, n))) for e, n in c["area_enu"]]
    body: dict[str, Any] = {
        "area": {"type": "Polygon", "coordinates": [[*ring, ring[0]]]},
        "reason_sv": f"Ramen {name} ur fixturen.",
    }
    for key in (
        "geofence_buffer_m",
        "alt_min_agl_m",
        "alt_max_agl_m",
        "default_alt_agl_m",
        "max_speed_ms",
        "max_range_from_pilot_m",
        "battery_rtl_pct",
        "allowed_skills",
    ):
        if key in c:
            body[key] = c[key]
    return {**body, **overrides}


def pin(name: str) -> tuple[float, float]:
    p = next(p for p in FIXTURE["pins"] if p["id"] == name)
    home = FIXTURE["home"]
    return Plane(float(home["lat"]), float(home["lon"])).latlon(p["east_m"], p["north_m"])


def survey_box_enu() -> list[tuple[float, float]]:
    box = FIXTURE["survey_box"]
    p = next(p for p in FIXTURE["pins"] if p["id"] == box["center_pin"])
    w, h = (float(x) / 2 for x in box["size_m"])
    cx, cy = float(p["east_m"]), float(p["north_m"])
    return [(cx - w, cy - h), (cx + w, cy - h), (cx + w, cy + h), (cx - w, cy + h)]


class ProxiedOnboard:
    """``SitlOnboard`` with its SD-link through a proxy that can cut it (the radio)."""

    def __init__(self, sim: Sim, url: str, token: str, **options: Any) -> None:
        self.proxy = LossyProxy(url)
        self._args = (sim, token, options)
        self.inner: SitlOnboard | None = None

    @property
    def ex(self) -> Any:
        assert self.inner is not None
        return self.inner.ex

    async def start(self, home: tuple[float, float]) -> None:
        await self.proxy.start()
        sim, token, options = self._args
        inner = SitlOnboard(sim, self.proxy.url, token, **options)
        self.inner = inner
        await inner.start(home)

    async def close(self) -> None:
        if self.inner is not None:
            await self.inner.close()
        await self.proxy.stop()


@dataclass
class Scenario:
    r: Rig
    sim: Sim
    port: int
    track: list[Sample] = field(default_factory=list[Sample])
    brain: Brain | None = None
    brain_tasks: list[asyncio.Future[Any]] = field(default_factory=list[asyncio.Future[Any]])
    mid: str = ""
    observer_token: str = ""
    sticks_seq: int = 0
    harness_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    bus_log: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    # --- what the ground station sees ---------------------------------------------------

    @property
    def tel(self) -> Any:
        return self.r.core.vehicle.telemetry

    def mode(self) -> str:
        """The flight mode as the ground station sees it."""
        return str(self.tel.fc_mode)

    def fc_mode(self) -> str:
        """The flight mode on the test rig's own channel (also with the radio down)."""
        return str(self.sim.harness.state.mode)

    async def harness(self, fn: Callable[..., Any], *args: Any) -> Any:
        """A harness call (read a parameter, inject a fault) between the rig's reads."""
        async with self.harness_lock:
            return await asyncio.to_thread(fn, *args)

    def modes(self) -> list[str]:
        """The flight modes seen so far, each change once (for failure messages)."""
        out: list[str] = []
        for sample in self.track:
            if not out or out[-1] != sample.mode:
                out.append(sample.mode)
        return out

    def from_home_m(self) -> float:
        return distance_m(self.r.home[0], self.r.home[1], self.tel.lat, self.tel.lon)

    def coverage(self, since: float, alt_m: float = 40.0, spacing_m: float = 30.0) -> float:
        """The share of the survey box seen from the track flown armed since ``since``."""
        plane = Plane(*self.r.home)
        track = [(*plane.xy(s.lat, s.lon), s.alt_m) for s in self.track if s.t >= since and s.armed]
        return coverage_fraction(survey_box_enu(), track, alt_m, spacing_m)

    def at(self, east: float, north: float) -> tuple[float, float]:
        return self.r.at(east, north)

    async def until(self, cond: Callable[[], Any], limit_s: float, what: str) -> float:
        start = time.monotonic()
        while True:
            value = cond()
            if asyncio.iscoroutine(value):
                value = await value
            if value:
                took = time.monotonic() - start
                measure(what, took, limit_s, passed=True)  # the condition held: passed
                return took
            if time.monotonic() - start > limit_s:
                measure(what, time.monotonic() - start, limit_s)
                t = self.tel
                raise AssertionError(
                    f"{what}: not within {limit_s} s (mode {self.mode()}, armed {t.armed}, "
                    f"{t.alt_rel_home_m:.1f} m, {t.groundspeed_ms:.1f} m/s, "
                    f"{self.from_home_m():.0f} m from home)"
                )
            await asyncio.sleep(0.05)

    async def hold_for(self, seconds: float, cond: Callable[[], bool], what: str) -> None:
        """``cond`` stays true the whole time."""
        start = time.monotonic()
        end = start + seconds
        while time.monotonic() < end:
            if not cond():
                measure(what, time.monotonic() - start, seconds, at_most=False)
                raise AssertionError(
                    f"{what} (mode {self.mode()}, {self.from_home_m():.0f} m from home)"
                )
            await asyncio.sleep(0.1)
        measure(what, time.monotonic() - start, seconds, at_most=False)

    # --- the pilot (panel API) -----------------------------------------------------------

    async def ready(self, charter: str = "charter_test", **overrides: Any) -> str:
        """Open a mission; the pilot proposes and approves a fixture charter: READY."""
        r = self.r
        self.mid = await r.open_mission()
        res = await r.panel.post(
            f"/api/v1/missions/{self.mid}/charters",
            json=charter_body(charter, **overrides),
            headers=H,
        )
        assert res.status_code == 201, res.text
        card = res.json()
        res = await r.approve(str(card["approval_id"]), card["version"])
        assert res.status_code == 200, res.text
        await r.wait_status(self.mid, "READY", 90)
        await r.until(r.onboard.ex.preflight_ok, 90, "onboard preflight")
        await r.until(
            lambda: (snap := r.flow.snapshot(VID)[0]) is not None and snap.preflight_ok,
            10,
            "preflight ok in the ground's snapshot",
        )
        return self.mid

    async def open(self) -> str:
        self.mid = await self.r.open_mission()
        return self.mid

    async def approve_proposed(self, limit_s: float = 30.0, **extra: Any) -> dict[str, Any]:
        """The pilot approves the charter card the AI proposed: READY, preflight ok."""
        r = self.r
        found: list[dict[str, Any]] = []

        async def card() -> bool:
            res = await r.panel.get(
                f"/api/v1/missions/{self.mid}/approvals", params={"status": "pending"}
            )
            found[:] = [a for a in res.json() if a["kind"] == "charter"]
            return bool(found)

        await self.until(card, limit_s, "the AI's charter card")
        res = await r.approve(str(found[0]["id"]), found[0]["subject_version"], **extra)
        assert res.status_code == 200, res.text
        await r.wait_status(self.mid, "READY", 90)
        await r.until(r.onboard.ex.preflight_ok, 90, "onboard preflight")
        await r.until(
            lambda: (snap := r.flow.snapshot(VID)[0]) is not None and snap.preflight_ok,
            10,
            "preflight ok in the ground's snapshot",
        )
        return found[0]

    async def airborne_with_ai(self, script: str, **brain: Any) -> None:
        """The scenarios' common start with the AI: the pilot names the charter, approves
        the AI's proposal, confirms "clear around the drone" and asks for the start; the
        AI takes off to 40 m."""
        await self.open()
        await self.start_brain(script, **brain)
        await self.say("Använd ramen charter_test.")
        await self.approve_proposed()
        await self.clearance()
        await self.say("Starta.")
        await self.wait_command("takeoff", "done", 90, issuer_kind="brain")

    async def airborne(self, charter: str = "charter_test", **overrides: Any) -> str:
        """Without the AI process: the pilot's charter and "clear around the drone", and a
        take-off order on the commander API (the AI's path) to 40 m."""
        await self.ready(charter, **overrides)
        await self.clearance()
        res = await self.order("takeoff", {"alt_m": 40})
        assert res.get("decision") == "ALLOW", res
        await self.wait_command("takeoff", "done", 90)
        await asyncio.sleep(3.1)  # the gate's rate limit: one AI action per 3 s
        return self.mid

    async def operator_order(self, skill: str, params: dict[str, Any]) -> dict[str, Any]:
        """The pilot's own order (an order button): the same gate as the AI's."""
        res = await self.r.panel.post(
            f"/api/v1/missions/{self.mid}/commands",
            json={"client_command_id": str(new_id()), "skill": skill, "params": params},
            headers=H,
        )
        return {"status_code": res.status_code, **res.json()}

    async def clearance(self) -> None:
        res = await self.r.clearance(self.mid)
        assert res.status_code == 200, res.text

    async def say(self, text: str, *, token: str | None = None) -> dict[str, Any]:
        headers = dict(H)
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        res = await self.r.panel.post(
            f"/api/v1/missions/{self.mid}/chat",
            json={"client_msg_id": str(new_id()), "text": text},
            headers=headers,
        )
        assert res.status_code == 201, res.text
        return res.json()

    async def press(self, action: str, *, token: str | None = None) -> httpx.Response:
        headers = dict(H)
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        return await self.r.panel.post(
            f"/api/v1/vehicles/{VID}/actions/{action}",
            json={"client_action_id": str(new_id())},
            headers=headers,
        )

    async def sticks(self, *, pitch: float, roll: float, takeover_id: str) -> bool:
        self.sticks_seq += 1
        value = ManualControlInput(
            takeover_id=UUID(takeover_id),
            input_seq=self.sticks_seq,
            pitch=pitch,
            roll=roll,
            throttle=0.5,
            yaw=0.0,
            source="gamepad",
        )
        return self.r.actions.manual_input(VID, value, PILOT)

    # --- the AI's side ------------------------------------------------------------------

    async def order(self, skill: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """An order on the commander API with the commander token, as ``sd-brain`` sends it
        (``based_on`` and the charter version from the ground's situation)."""
        situation = (
            await self.r.commander.get(f"/commander/v1/missions/{self.mid}/situation")
        ).json()
        based_on = None
        if situation["snapshot_state_epoch"] is not None:
            based_on = {
                "state_epoch": situation["snapshot_state_epoch"],
                "state_seq": situation["snapshot_state_seq"],
            }
        charter = situation["charter"]
        body = {
            "command_id": str(new_id()),
            "skill": skill,
            "params": params or {},
            "rationale_sv": "Scenariotest",
            "in_reply_to": None,
            "based_on": based_on,
            "charter_version": None if charter is None else charter["version"],
            "backend": "scripted",
            "ai_config_hash": "scenario",
        }
        res = await self.r.commander.post(f"/commander/v1/missions/{self.mid}/commands", json=body)
        assert res.status_code in (200, 201, 409, 422), res.text
        return {"status_code": res.status_code, **res.json()}

    async def commands(self) -> list[dict[str, Any]]:
        res = await self.r.panel.get(f"/api/v1/missions/{self.mid}/commands")
        return list(res.json())

    async def command(self, command_id: str) -> dict[str, Any]:
        return next(c for c in await self.commands() if c["id"] == command_id)

    async def wait_command(
        self,
        skill: str,
        status: str,
        limit_s: float,
        issuer_kind: str | None = None,
        *,
        besides: Iterable[str] = (),
    ) -> dict[str, Any]:
        """The newest order of ``skill`` (not one of the ids in ``besides``) reaches
        ``status``; a different end fails."""
        found: dict[str, Any] = {}
        old = set(besides)

        async def reached() -> bool:
            cmds = [
                c
                for c in await self.commands()
                if c["skill"] == skill
                and (issuer_kind is None or c["issuer_kind"] == issuer_kind)
                and c["id"] not in old
            ]
            if not cmds:
                return False
            c = max(cmds, key=lambda c: c["created_at"])
            if c["status"] in TERMINAL and c["status"] != status:
                raise AssertionError(
                    f"{skill}: {c['status']} {c['status_reason']} (modes {self.modes()})"
                )
            found.update(c)
            return c["status"] == status

        try:
            await self.until(reached, limit_s, f"{skill} {status}")
        except AssertionError as exc:
            seen = [(c["skill"], c["status"], c["status_reason"]) for c in await self.commands()]
            raise AssertionError(
                f"{exc}; orders {seen}; modes {self.modes()}; AI turns {self.brain_turns()}"
                f"{self.brain_failures()}"
            ) from None
        return found

    def brain_turns(self) -> list[tuple[str, str, str | None, str | None]]:
        """The AI's turns so far (trigger, backend, action, error), for failure messages."""
        if self.brain is None:
            return []
        return [
            (
                r.trigger,
                r.backend,
                None if r.ai_action is None else r.ai_action.skill,
                r.error or (None if r.result is None else str(r.result.status)),
            )
            for r in self.brain.loop.records
        ]

    def brain_failures(self) -> str:
        """The AI's tasks that ended with an error (stream, turn loop, idle ticker), for
        failure messages: a task that dies here leaves the AI silent without a trace."""
        names = ("event stream", "turn loop", "idle ticker")
        dead = [
            f"{name}: {t.exception()!r}"
            for name, t in zip(names, self.brain_tasks, strict=False)
            if t.done() and not t.cancelled() and t.exception() is not None
        ]
        return f"; AI tasks ended: {dead}" if dead else ""

    def actions(self, action: str) -> list[dict[str, Any]]:
        """STOP / FLY HOME / Resume outcomes the ground station published, by action."""
        return [a for a in self.published("vehicle.action") if a.get("action") == action]

    def published(self, type_: str) -> list[dict[str, Any]]:
        """What the ground station put on its live stream (the panel's view), by type."""
        return [data for t, data in self.bus_log if t == type_]

    async def onboard_state(self) -> dict[str, Any]:
        """The onboard executor's state as the panel API shows it (the latest snapshot)."""
        live = (await self.r.panel.get(f"/api/v1/vehicles/{VID}/state")).json()
        assert live["snapshot"] is not None, live
        return dict(live["snapshot"])

    async def gate(self, command_id: str) -> list[dict[str, Any]]:
        res = await self.r.panel.get(
            f"/api/v1/missions/{self.mid}/gate_decisions", params={"command_id": command_id}
        )
        return list(res.json())

    async def events(self, type_: str | None = None) -> list[dict[str, Any]]:
        await self.r.g.flush_events()  # the ground writes events in batches
        params = {"limit": "1000"} | ({} if type_ is None else {"type": type_})
        res = await self.r.panel.get(f"/api/v1/missions/{self.mid}/events", params=params)
        return list(res.json())

    async def chat(self) -> list[dict[str, Any]]:
        res = await self.r.panel.get(f"/api/v1/missions/{self.mid}/chat")
        data = res.json()
        return list(data["items"] if isinstance(data, dict) else data)

    # --- sd-brain -----------------------------------------------------------------------

    async def start_brain(self, script: str, **config: Any) -> Brain:
        """``sd-brain`` with the scenario's script, on the served commander API."""
        cfg = BrainConfig.model_validate(
            {
                "backend": {"kind": "scripted", "scripted_file": str(SCRIPTS / f"{script}.yaml")},
                "ground": {
                    "commander_url": f"http://127.0.0.1:{self.port}/commander/v1",
                    "vehicle_id": VID,
                },
                "decide": {"engines": []},
                **config,
            }
        )
        b = build(cfg, Secrets(BRAIN, "k" * 40), SKILL_REGISTRY)
        await _startup(b, SKILL_REGISTRY)
        self.brain_tasks = [
            asyncio.ensure_future(_stream(b)),
            asyncio.ensure_future(b.loop.run_forever()),
            asyncio.ensure_future(_idle_ticker(b)),
        ]
        self.brain = b
        await self.until(lambda: self.r.g.brain_state == "ready", 10, "sd-brain ready")
        return b

    async def kill_brain(self) -> None:
        """``sd-brain`` dies: its tasks stop at once, without a goodbye."""
        for task in self.brain_tasks:
            task.cancel()
        await asyncio.gather(*self.brain_tasks, return_exceptions=True)
        self.brain_tasks = []
        if self.brain is not None:
            with contextlib.suppress(Exception):
                await self.brain.http.aclose()
        self.brain = None

    # --- a confirmed find and the RC transmitter -----------------------------------------

    async def confirm_find(self, lat: float, lon: float, sigma95_m: float = 5.0) -> str:
        """A candidate that a person confirmed, written as the search service writes it
        (the panel has no confirm route yet), and wait until the onboard executor holds its
        keep-out cylinder (``KeepoutInstall``). Returns the candidate id."""
        async with self.r.flow.repo.engine.begin() as conn:
            row = (
                await conn.execute(
                    text(
                        "INSERT INTO candidates (candidate_id, rev, mission_id, seq, state, "
                        "geom, sigma95_m, n_seen, n_in_view, best_detector_score, priority, "
                        "first_seen_utc, last_seen_utc) SELECT uuidv7(), 1, CAST(:m AS uuid), "
                        "COALESCE(max(seq), 0) + 1, 'confirmed', "
                        "ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), :sigma, 3, 3, 0.8, 0.9, "
                        "now(), now() FROM candidates WHERE mission_id = CAST(:m AS uuid) "
                        "RETURNING candidate_id::text"
                    ),
                    {"m": self.mid, "lat": lat, "lon": lon, "sigma": sigma95_m},
                )
            ).one()
        cid = str(row[0])
        ex = self.r.onboard.ex
        await self.until(
            lambda: any(str(k.source_id) == cid for k in ex.keepouts), 15, "keep-out onboard"
        )
        return cid

    async def rc_switch(self, mode: str) -> None:
        """The pilot moves the flight-mode switch (channel 5, FLTMODE1-6 in
        searchdrone-safety.parm). ArduCopter acts on a change of position only, so the
        LOITER position is reached through RTL's (as a pilot flicks the switch)."""
        pwm = RC_PWM[mode]
        if mode == "LOITER":
            self.sim.rc.mode_pwm = RC_PWM["RTL"]
            await asyncio.sleep(0.5)  # longer than the flight controller's switch debounce
        self.sim.rc.mode_pwm = pwm

    # --- faults ---------------------------------------------------------------------------

    async def kill_onboard(self) -> None:
        """``sd-onboard`` dies: no executor, no heartbeat from 1/191 (closed only once)."""
        side = self.r.onboard
        await side.close()
        side.close = _closed  # the rig's teardown closes it again

    def radio(self, *, mavlink: bool | None = None, sdlink: bool | None = None) -> None:
        """The radio between the ground and the drone: the ground's MAVLink through
        linksim, SD-link through the proxy (``scenario(proxied_link=True)``)."""
        if mavlink is not None:
            body = json.dumps({"state": "up" if mavlink else "down"}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{self.sim.ports['linksim_control']}/flows/mavlink",
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.sim.stack.linksim_token}",
                    "Content-Type": "application/json",
                },
            )
            with LOOPBACK_OPENER.open(req, timeout=2) as resp:
                assert resp.status in (200, 204)
        if sdlink is not None:
            proxy = getattr(self.r.onboard, "proxy", None)
            assert proxy is not None, "scenario(proxied_link=True) needed"
            proxy.broken = not sdlink

    # --- invariants ---------------------------------------------------------------------

    async def check_invariants(self, *, exempt: Iterable[str] = ()) -> None:
        r = self.r
        charters = (await r.panel.get(f"/api/v1/missions/{self.mid}/charters")).json()
        cmds = await self.commands()
        gates = {c["id"]: await self.gate(c["id"]) for c in cmds}
        events = await self.events()
        broken = check_all(self.track, charters, cmds, gates, events, exempt=exempt)
        assert not broken, broken


async def _closed() -> None:
    return None


async def _observe(sc: Scenario) -> None:
    """Read the test rig's own channel to the flight controller (serial0, independent of
    the radio) and record the track; play the pilot's throttle stick: centred in the air
    (LOITER holds its height), down on the ground and while landing (arming needs it low)."""
    h = sc.sim.harness

    def record(kind: str, st: Any) -> None:
        if kind != "GLOBAL_POSITION_INT":
            return
        sc.track.append(
            Sample(
                t=time.monotonic(),
                lat=st.lat,
                lon=st.lon,
                alt_m=st.rel_alt_m,
                groundspeed_ms=st.ground_speed_ms,
                mode=st.mode,
                armed=st.armed,
            )
        )
        airborne = st.armed and st.rel_alt_m > 3.0
        sc.sim.rc.throttle = 1500 if airborne and st.mode not in LANDING_MODES else 1000

    h.listeners.append(record)
    try:
        while True:
            async with sc.harness_lock:
                await asyncio.to_thread(h.pump, 0.2)
            await asyncio.sleep(0)
    finally:
        h.listeners.remove(record)


async def _observer_token(r: Rig) -> str:
    """An observer's CLI token: may watch and press STOP, nothing else."""
    token = new_token()
    now = dt.datetime.now(dt.UTC)
    rec = SessionRecord(
        id="",
        kind="token",
        role="observer",
        issuer_ref="operator:ci-observer",
        auth_mode="cli",
        token_sha256=hash_session_token(token),
        created_at=now,
        last_used_at=now,
        expires_at=now + dt.timedelta(hours=24),
        name="ci-observer",
    )
    assert r.g.repo is not None
    await r.g.repo.save_session(rec)
    r.g.sessions.load(await r.g.repo.sessions())
    return token


@contextlib.asynccontextmanager
async def scenario(
    sim: Sim,
    db: Any,
    tmp_path: Path,
    *,
    proxied_link: bool = False,
    onboard_factory: Callable[[Sim, str, str], Any] | None = None,
    **onboard_options: Any,
) -> AsyncIterator[Scenario]:
    """``onboard_factory(sim, url, token)`` replaces the onboard side (e.g. one that also
    runs perception); otherwise ``SitlOnboard`` with ``onboard_options``."""
    # The ground's vehicle events (status texts, the GPS texts' alarms) as sd_ground.app
    # wires them; the sink is the rig's VehicleActions, which exists once the stack is up.
    sink: list[Callable[[str, dict[str, Any]], None]] = []
    ground = _vehicle(
        f"udpin:127.0.0.1:{sim.ports['ground_mavlink']}",
        Role.GROUND,
        on_event=lambda k, d: sink[0](k, d) if sink else None,
    )
    sock = loopback_listener()
    port = sock.getsockname()[1]

    def onboard(_clock: WallClock, url: str, token: str) -> Any:
        if onboard_factory is not None:
            return onboard_factory(sim, url, token)
        if proxied_link:
            return ProxiedOnboard(sim, url, token, **onboard_options)
        return SitlOnboard(sim, url, token, **onboard_options)

    async with (
        ground_stack(
            db.name,
            db.app_password,
            tmp_path,
            vehicle=ground,
            mav_clock=MonotonicClock(),
            onboard=onboard,
            http={"commander_bind": f"127.0.0.1:{port}"},
        ) as r,
        serve_commander(r, sock),
    ):
        r.commander.base_url = httpx.URL(f"http://127.0.0.1:{port}")  # the bound port
        sink.append(vehicle_events(r.actions.on_event))
        sc = Scenario(r, sim, port)
        publish = r.g.bus.publish

        def recording(topic: str, type_: str, data: dict[str, Any], **kw: Any) -> Any:
            sc.bus_log.append((type_, data))
            return publish(topic, type_, data, **kw)

        r.g.bus.publish = recording
        sc.observer_token = await _observer_token(r)
        watch = asyncio.ensure_future(_observe(sc))
        try:
            yield sc
        finally:
            await sc.kill_brain()
            watch.cancel()
            await asyncio.gather(watch, return_exceptions=True)
