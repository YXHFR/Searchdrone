"""The closer look's scenarios (spec "Titta närmare" §16.4) on the onboard SITL rig.

The onboard rig of ``test_onboard_sitl`` (the executor in this process, SD-link to a test
ground, builtin physics) in a scenario's world (``sd_sim.inspect_world``):

* the world's tree stands, power lines and light ground are the truth (``Stack.obstacles``,
  ``sd_sim.obstacles.Scene``); the map pack is built from it with the scenario's map
  errors (``Stack.write_floor_pack``) and given to the executor;
* the laser, when the scenario has one, is the simulated one on SITL's serial port, read
  by the onboard ``LaserReader`` through the flight controller as on the drone;
* the cameras and the detector are ``inspect_world.Oracle``: every round of a look gives
  one still per camera (main and tele) from where the drone is, each reported up with the
  order's ``command_id`` and ``inspect_step`` and its target detections handed to the
  executor's keep-out cylinders, as the perception task does.

``TnLook`` flies one: the charter in the scenario's mode, the take-off to its start
height, the pilot's order, the pilot's answers, and the numbers of §16.3 against the truth
(``sd_sim.inspect_metrics``), written to the run's folder (``search_report.md``).
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from test_onboard_sitl import (  # pyright: ignore[reportMissingImports]
    READY_S,
    Flight,
    RcSticks,
    Sim,
    _install,
    _takeoff,
)

from sd_onboard.config import RangefinderConfig
from sd_onboard.perception.pipeline import Perception
from sd_onboard.rangefinder import LaserReader
from sd_safety.constants import PERSON_KEEPOUT_M
from sd_schemas.base import LatLon, new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.inspect import InspectProgress
from sd_schemas.link import (
    BBox,
    DetectorInfo,
    GeoDetectionMsg,
    InspectLower,
    InspectOutcome,
    InspectStay,
    PerceptionReport,
    Pose,
    SkillCommand,
    StateSnapshot,
)
from sd_sim import config, rangefinder
from sd_sim.harness import Harness
from sd_sim.inspect_metrics import InspectMeter, InspectMetrics, write_report
from sd_sim.inspect_world import Camera, InspectWorld, Oracle
from sd_sim.launch import Stack
from sd_sim.pytest_plugin import worker_instance
from sd_sim.sitl import CMAC_HOME
from sd_sim.util import new_run_dir

__all__ = ["SKILLS", "TnLook", "tn_stack"]

SKILLS = ["takeoff", "goto", "hold", "resume", "return_home", "capture_stills", "inspect_detection"]
ROUND_S = 0.2  # how long a round of stills takes (the oracle answers at once otherwise)
RISE_GRACE_S = 12.0  # a floor that rises (the laser lost) holds only after the climb to it
_TELE_BOX = BBox(x=10, y=10, w=20, h=40)


@contextlib.contextmanager
def tn_stack(world: InspectWorld) -> Iterator[Sim]:
    """SITL in the scenario's world: its obstacles and light ground, its laser with its
    faults, the project's parameter layers, checked strictly, and the pilot's sticks."""
    instance = worker_instance()
    cfg = config.load(
        {
            "physics": "builtin",
            "gui": "false",
            "video": "off",
            "instance": str(instance),
            "rangefinder": world.laser,
        }
    )
    cfg.home, cfg.personal_home = CMAC_HOME, False
    st = Stack(
        cfg,
        new_run_dir(instance, f"{world.scenario.lower()}-s{world.seed}"),
        observer=False,
        obstacles=world.obstacles,
        rangefinder_faults=world.faults,
        light_areas=world.light,
    )
    harness = Harness(st.ports["harness_mavlink"])
    try:
        st.up()
        harness.wait(lambda s: s.heartbeat, 60, "HEARTBEAT från flygkontrollen")
        harness.wait(lambda s: s.ready, READY_S, "position från EKF och GPS")
        broken = harness.safety_violations()
        assert not broken, f"flygkontrollens säkerhetsparametrar: {broken}"
        with RcSticks(st.ports["sitl_rc_in"]) as rc:
            yield Sim(st, harness, rc)
    finally:
        harness.close()
        st.down()


@dataclass
class Shot:
    t: float
    command_id: UUID
    step: int
    camera: str
    still_id: UUID
    detections: list[GeoDetectionMsg]
    target_px: float | None


@dataclass
class TnLook:
    """One closer look in a scenario's world, flown on ``f``."""

    f: Flight
    world: InspectWorld
    events: list[tuple[float, str, dict[str, Any]]] = field(default_factory=list)
    shots: list[Shot] = field(default_factory=list)
    cmd: SkillCommand | None = None
    t_order: float = 0.0
    t_yes: float | None = None
    laser: LaserReader | None = None
    _rounds: dict[int, int] = field(default_factory=dict)

    # --- setting up -----------------------------------------------------------------------

    async def start(self, **charter: Any) -> None:
        """Charter, take-off to the start height, the map pack, the laser, the cameras."""
        f, w = self.f, self.world
        await _install(
            f,
            allowed_skills=SKILLS,
            operation_mode=w.mode,
            overflight_allowed=w.overflight,
            alt_min_agl_m=w.alt_min_agl_m,
            alt_max_agl_m=w.alt_max_agl_m,
            **charter,
        )
        await _takeoff(f, w.start_alt_m)
        await f.until(lambda: f.ex.state == S.HOLD, 60, "takeoff done")
        name = f"{w.scenario.lower()}-s{w.seed}"
        f.ex.floor_pack = await asyncio.to_thread(f.sim.stack.write_floor_pack, name, w.map_errors)
        if w.laser != "none":
            low, high = rangefinder.PROFILES[w.laser].tested_m
            self.laser = LaserReader(
                RangefinderConfig(model="other", min_tested_m=low, max_tested_m=high)
            )
            f.onboard.subscribe(self.laser.observe)
            f.ex.laser = self.laser
        self._record_events()
        oracle = Oracle(w)

        async def capture_round(
            command_id: UUID, step: int
        ) -> list[tuple[str, UUID, list[GeoDetectionMsg]]]:
            return await self._round(oracle, command_id, step)

        async def capture(command_id: UUID) -> UUID | None:
            """The side look's still (live): the main camera only, as ``capture``."""
            shots = await self._round(oracle, command_id, 0, cameras=oracle.cameras[:1])
            return shots[0][1]

        f.ex.capture_round = capture_round
        f.ex.camera = capture

    def _record_events(self) -> None:
        f = self.f
        send_inspect, send = f.ex.emit_inspect, f.ex.emit

        def inspect(kind: str, data: dict[str, Any]) -> None:
            self.events.append((time.monotonic(), kind, data))
            send_inspect(kind, data)

        def executor(kind: str, data: dict[str, Any] | None = None) -> None:
            self.events.append((time.monotonic(), f"executor.{kind}", dict(data or {})))
            send(kind, data)

        f.ex.emit_inspect = inspect  # type: ignore[method-assign]
        f.ex.emit = executor  # type: ignore[method-assign]

    async def _round(
        self,
        oracle: Oracle,
        command_id: UUID,
        step: int,
        cameras: tuple[Camera, ...] | None = None,
    ) -> list[tuple[str, UUID, list[GeoDetectionMsg]]]:
        """One round of stills, as ``PerceptionLoop.capture_round``: a still per camera,
        reported up with the order and the step; its people to the keep-out cylinders."""
        f = self.f
        n = self._rounds.get(step, 0)
        self._rounds[step] = n + 1
        tel = f.onboard.telemetry
        east, north = f.plane.xy(tel.lat, tel.lon)
        drone = (east, north, tel.alt_rel_home_m)
        shots: list[tuple[str, UUID, list[GeoDetectionMsg]]] = []
        for cam in oracle.cameras if cameras is None else cameras:
            seen = oracle.still(cam, drone, (step, n))
            dets = []
            target_px = None
            for s in seen:
                lat, lon = f.plane.latlon(s.east, s.north)
                dets.append(
                    GeoDetectionMsg(
                        detection_id=new_id(),
                        bbox=_TELE_BOX,
                        score=s.score,
                        lat=lat,
                        lon=lon,
                        sigma95_m=s.sigma95_m,
                        px_size=s.px,
                        crop_inline=False,
                        label="person",
                        source="model",
                    )
                )
                if s.person.id == "target":
                    target_px = s.px
            report = self._report(command_id, step, dets)
            assert f.link is not None
            f.link.perception(Perception(report, []))
            if dets:
                f.ex.on_detections(dets)  # the perception task's on_targets
            self.shots.append(
                Shot(time.monotonic(), command_id, step, cam.name, report.still_id, dets, target_px)
            )
            shots.append((cam.name, report.still_id, dets))
        await asyncio.sleep(ROUND_S)
        return shots

    def _report(self, command_id: UUID, step: int, dets: list[GeoDetectionMsg]) -> PerceptionReport:
        f = self.f
        tel = f.onboard.telemetry
        home = f.ground.telemetry.home
        assert home is not None
        east, north = f.plane.xy(tel.lat, tel.lon)
        ring = [
            f.plane.latlon(east + e, north + n)
            for e, n in ((-20, 15), (20, 15), (20, -15), (-20, -15))
        ]
        return PerceptionReport(
            still_id=new_id(),
            command_id=command_id,
            inspect_step=step,
            captured_utc=f.ex.clock.utc_now(),
            sensor="rgb",
            pose=Pose(
                lat=tel.lat,
                lon=tel.lon,
                alt_amsl_m=home.alt_amsl_m + tel.alt_rel_home_m,
                agl_m=tel.alt_rel_home_m,
                roll_deg=0,
                pitch_deg=0,
                yaw_deg=0,
                source_ts_ms=0,
            ),
            footprint=[LatLon(lat=la, lon=lo) for la, lo in ring],
            px_per_m=24.0,
            synthetic=True,
            detector=DetectorInfo(
                name="sd_sim:inspect-oracle",
                license="Apache-2.0",
                version="0",
                placement="onboard",
                ok=True,
                tiles=1,
                latency_ms=30,
                rate_hz=1,
            ),
            detections=dets,
        )

    # --- the pilot ------------------------------------------------------------------------

    async def order(self) -> SkillCommand:
        """The pilot's "Titta närmare" on the candidate the ground knows (its estimate)."""
        f, w = self.f, self.world
        lat, lon = f.plane.latlon(*w.estimate_enu)
        self.t_order = time.monotonic()
        cmd = await f.order(
            "inspect_detection",
            {
                "candidate_id": str(new_id()),
                "lat": lat,
                "lon": lon,
                "sigma95_m": w.sigma95_m,
                "target_radius_m": PERSON_KEEPOUT_M + w.sigma95_m,
            },
        )
        ack = await f.stage(cmd, {"running"}, 10)
        assert ack.stage == "running", ack
        self.cmd = cmd
        return cmd

    async def _send(self, body: Any) -> None:
        assert self.f.session is not None
        await self.f.session.send(body, ttl_ms=15_000)

    async def yes(self) -> None:
        assert self.cmd is not None
        self.t_yes = time.monotonic()
        await self._send(InspectLower(command_id=self.cmd.command_id))

    async def no(self) -> None:
        assert self.cmd is not None
        await self._send(InspectStay(command_id=self.cmd.command_id))

    async def decide(self, outcome: str) -> None:
        assert self.cmd is not None
        await self._send(InspectOutcome(command_id=self.cmd.command_id, outcome=outcome))  # type: ignore[arg-type]

    # --- what happened --------------------------------------------------------------------

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [d for _, k, d in self.events if k == kind]

    def first(self, kind: str) -> tuple[float, dict[str, Any]] | None:
        return next(((t, d) for t, k, d in self.events if k == kind), None)

    async def wait_event(self, kind: str, limit_s: float) -> dict[str, Any]:
        await self.f.until(lambda: bool(self.of(kind)), limit_s, kind)
        return self.of(kind)[0]

    async def waiting(self, limit_s: float = 240.0) -> dict[str, Any]:
        """The steps are over and the drone hovers for the decision: ``inspect.done``."""
        f = self.f
        await f.until(lambda: f.ex.hold_reason == "inspect_await", limit_s, "the wait")
        done = self.of("inspect.done")
        assert len(done) == 1, done
        return done[0]

    def snapshots(self) -> list[tuple[float, StateSnapshot]]:
        return [(t, b) for t, b in self.f.inbound if isinstance(b, StateSnapshot)]

    def progress(self) -> list[tuple[float, InspectProgress]]:
        return [(t, s.inspect) for t, s in self.snapshots() if s.inspect is not None]

    def xy(self) -> tuple[float, float]:
        t = self.f.onboard.telemetry
        return self.f.plane.xy(t.lat, t.lon)

    def alt(self) -> float:
        return self.f.onboard.telemetry.alt_rel_home_m

    def track(
        self, since: float, until: float | None = None
    ) -> list[tuple[float, float, float, float]]:
        """(t, east, north, height above home) of the flight since ``since``."""
        end = math.inf if until is None else until
        return [
            (t, *self.f.plane.xy(lat, lon), alt)
            for t, lat, lon, alt in self.f.track
            if since <= t <= end
        ]

    def floors(self) -> list[tuple[float, float]]:
        """The floor in force (above ground) over time: the automatic one until the
        pilot's yes, the lowest after it; none once the drone flies back."""
        out = []
        for t, p in self.progress():
            if p.phase == "return":
                continue
            yes = self.t_yes is not None and t >= self.t_yes
            floor = p.floor_lowest_above_ground_m if yes else p.floor_auto_above_ground_m
            if floor is not None:
                out.append((t, floor))
        return out

    def floor_at(self, t: float, floors: list[tuple[float, float]]) -> float | None:
        """The floor the drone must be above at ``t``: a rise holds only after the climb."""
        window = [f for ft, f in floors if t - RISE_GRACE_S <= ft <= t]
        before = [f for ft, f in floors if ft <= t]
        if not before:
            return None
        return min([before[-1], *window])

    def metrics(self, until: float | None = None) -> InspectMetrics:
        """§16.3 against the world's truth, from the order to ``until`` (now)."""
        f, w = self.f, self.world
        scene = f.sim.stack.obstacle_scene()
        meter = InspectMeter(scene, w.target.enu)
        floors = self.floors()
        for t, east, north, z in self.track(self.t_order, until):
            meter.sample(t, east, north, z, self.floor_at(t, floors))
        px: dict[int, dict[str, float]] = {}
        for s in self.shots:
            if s.target_px is not None:
                cams = px.setdefault(s.step, {})
                cams[s.camera] = max(cams.get(s.camera, 0.0), s.target_px)
        done = self.of("inspect.done")
        aborted = self.of("inspect.aborted")
        outcome = done[0]["end"] if done else aborted[0]["reason"] if aborted else ""
        return meter.result(
            outcome=outcome,
            px_first=px[min(px)] if px else None,
            px_last=px[max(px)] if px else None,
        )

    def report(self, until: float | None = None) -> InspectMetrics:
        """The numbers, checked against acceptance 7 (every TN run), and written down."""
        m = self.metrics(until)
        run_dir: Path = self.f.sim.stack.run_dir
        write_report(run_dir, self.world.scenario, self.world.seed, m)
        assert m.min_clearance_m >= 5.0, m
        assert m.floor_breach_m <= 1.5, m
        return m

    def steps_reported(self) -> dict[int, set[str]]:
        """Acceptance 20: every step's cameras, among the stills of this order."""
        assert self.cmd is not None
        out: dict[int, set[str]] = {}
        for s in self.shots:
            if s.command_id == self.cmd.command_id:
                out.setdefault(s.step, set()).add(s.camera)
        return out

    def check_steps(self) -> None:
        """Every step gave a still from every camera with the order's id and its step,
        and the steps count 0, 1, 2 ... with none left out (acceptance 20)."""
        steps = self.steps_reported()
        assert steps, "no stills"
        assert sorted(steps) == list(range(max(steps) + 1)), sorted(steps)
        for step, cams in steps.items():
            assert cams == {"main", "tele"}, (step, cams)
        assert self.cmd is not None
        sent = [
            b
            for _, b in self.f.inbound
            if isinstance(b, PerceptionReport) and b.command_id == self.cmd.command_id
        ]
        assert sent and {b.inspect_step for b in sent} == set(steps), "the reports up"
