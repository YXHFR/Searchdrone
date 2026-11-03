"""The perception task: poll the still source, run the pipeline, queue the report and crops.

Runs beside the executor in its own task and only reads: the flight controller's latest
telemetry (``Vehicle.telemetry``) and the still source. It never sends anything to the
flight controller and never touches the executor. Reports go to the link's
lowest-priority queue (``OnboardLink.perception``), each report followed by its crops.
Reading the still and the detector work run in a worker thread so a slow picture never
blocks the event loop the executor ticks on. A failing still is logged and skipped; the
loop keeps going. The detector sources (a trained model among them) are built in a
worker thread when the task starts, after the executor and link run; if that fails,
perception is off, the reason is logged and sent to the ground (``off``), and flying
goes on. Only a reason code and the error's class go to the ground, never the error's
text (it may hold paths on the on-board computer). ``FAIL_REPORT`` stills failing in a
row (the detector, the still or the telemetry) are reported the same way, once, as
``stills_failing``; when the detector then runs on a still again, the ground is told it
is back (``on``). A still skipped before detection (no pose, on the ground, turning)
neither ends nor adds to a streak. Kept stills (for a run's record) are capped at
``KEEP_MAX``, the oldest removed first.

An order can ask for a still of its own (``capture``: ``capture_stills`` and
``inspect_detection``): the next still from the source, reported with the order's
``command_id``, at most ``CAPTURE_WAIT_S`` after the asking. Target detections go to
``on_targets`` too: the executor puts a keep-out cylinder around each that is sure enough.

A second camera (``extra``: the tele camera beside the main one, looking the same way)
has its own still source and pinhole camera. Each round the loop first polls every
camera's source, one after the other, and stamps each still with its capture time (the
source's own, else the clock right after the poll, before any detection), then runs
the detector on each. The stills of one round are close in time but not synchronised:
each source gives its newest still, and the cameras are not triggered together. A tele
still goes through the same steps as a main one, georeferenced with its own camera,
and its report says so only through its smaller footprint and higher ``px_per_m``.
A failing streak (``stills_failing``) is counted per camera, so a dead main camera is
reported even while the tele camera works. An order's still (``capture``) comes from
the main camera, so does the side look of ``inspect_detection`` (its ``live`` look): it
sees its target 24-28 degrees from straight down, outside the fixed tele camera's view
(about +-10 by +-7.5 degrees). Neither camera tilts. The step-wise closer look straight
over the target (``sd_schemas.inspect``) takes one still from each camera per round.

The pose is the telemetry read when a still arrives, a moment after the picture was
taken. In steady flight that moment does not matter; while the attitude changes quickly
(a turn, braking, yawing towards a find) it does: 10 degrees of attitude at 40 m move a
detection about 7 m. The loop follows the attitude rate between telemetry samples and
skips stills taken while it is above ``MAX_TURN_RATE_DPS``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import UUID

from sd_detect import ClassRegistry, Still, build_sources, default_registry, detect_all
from sd_detect.sources import BuiltSources
from sd_geo import PinholeCamera
from sd_mav.vehicle import FcTelemetry
from sd_obs.clock import Clock
from sd_onboard.perception.pipeline import MIN_AGL_M, Perception, Pose, process
from sd_onboard.perception.sources import CapturedStill, StillSource
from sd_schemas.base import new_id
from sd_schemas.link import GeoDetectionMsg

__all__ = [
    "CAPTURE_WAIT_S",
    "FAIL_REPORT",
    "KEEP_MAX",
    "MAIN",
    "MAX_POSE_AGE_MS",
    "MAX_TURN_RATE_DPS",
    "AttitudeRate",
    "Feed",
    "PerceptionLoop",
    "PerceptionStats",
    "pose_from",
]

log = logging.getLogger("sd_onboard.perception")
MAX_POSE_AGE_MS = 1000  # older position or attitude: the still is skipped
KEEP_MAX = 200  # kept stills (JPEG + JSON pairs) in ``keep_dir``
MAX_TURN_RATE_DPS = 10.0  # faster attitude change: the still is skipped
CAPTURE_WAIT_S = 3.0  # an order's still: no still from the source this long -> none
_CAPTURE_POLL_S = 0.1
FAIL_REPORT = 10  # stills failing in a row before the ground is told
MAIN = "main"  # the main camera's name (``source`` and ``camera``)


@dataclass(frozen=True)
class Feed:
    """A camera beyond the main one: where its stills come from and its pinhole model."""

    source: StillSource
    camera: PinholeCamera
    name: str = "tele"


@dataclass(frozen=True)
class _Polled:
    """A still just taken from a camera's source, before the detector."""

    camera: str
    captured: CapturedStill
    t: FcTelemetry
    now_ms: int
    rate: float  # the attitude rate around the poll (degrees per second)
    utc: datetime  # when it was taken


@dataclass
class PerceptionStats:
    stills: int = 0
    reports: int = 0
    detections: int = 0
    skipped_no_pose: int = 0
    skipped_on_ground: int = 0
    skipped_turning: int = 0
    skipped_no_footprint: int = 0  # the camera's view does not reach the ground
    failed: int = 0
    by_label: dict[str, int] = field(default_factory=dict[str, int])
    by_camera: dict[str, int] = field(default_factory=dict[str, int])  # reports per camera


def pose_from(t: FcTelemetry, now_ms: int) -> Pose | None:
    """The pose from fresh telemetry; ``None`` when position or attitude is stale."""
    if t.position_mono_ms is None or t.attitude_mono_ms is None:
        return None
    if (
        now_ms - t.position_mono_ms > MAX_POSE_AGE_MS
        or now_ms - t.attitude_mono_ms > MAX_POSE_AGE_MS
    ):
        return None
    return Pose(
        lat=t.lat,
        lon=t.lon,
        alt_amsl_m=t.alt_amsl_m,
        agl_m=t.alt_rel_home_m,  # flat ground at home height (the simulator's world)
        roll_deg=t.roll_deg,
        pitch_deg=t.pitch_deg,
        yaw_deg=t.yaw_deg,
        speed_ms=t.groundspeed_ms,
        source_ts_ms=t.attitude_mono_ms,
    )


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


class AttitudeRate:
    """The largest of roll, pitch and yaw rate (degrees per second) between the two most
    recent attitude samples; 0 until two samples are known."""

    def __init__(self) -> None:
        self._last: tuple[int, float, float, float] | None = None
        self.dps = 0.0

    def update(self, t: FcTelemetry) -> float:
        if t.attitude_mono_ms is None:
            return self.dps
        now = (t.attitude_mono_ms, t.roll_deg, t.pitch_deg, t.yaw_deg)
        last = self._last
        if last is not None and now[0] > last[0]:
            dt = (now[0] - last[0]) / 1000.0
            self.dps = max(_angle_diff(a, b) for a, b in zip(now[1:], last[1:], strict=True)) / dt
        if last is None or now[0] > last[0]:
            self._last = now
        return self.dps


class PerceptionLoop:
    def __init__(
        self,
        source: StillSource,
        telemetry: Callable[[], FcTelemetry],
        send: Callable[[Perception], None],
        clock: Clock,
        camera: PinholeCamera,
        *,
        registry: ClassRegistry | None = None,
        period_s: float = 0.5,
        keep_dir: Path | None = None,
        env: dict[str, str] | None = None,
        on_targets: Callable[[list[GeoDetectionMsg]], None] | None = None,
        off: Callable[[str, str], None] | None = None,
        on: Callable[[], None] | None = None,
        sources: BuiltSources | None = None,
        extra: Sequence[Feed] = (),
    ) -> None:
        """``sources``: detector sources built by the caller (the simulator's oracle);
        without them they are built from ``registry`` on first use. ``extra``: cameras
        beside ``source`` and ``camera`` (the tele camera), polled after it each round
        with the same detector sources."""
        if any(f.name == MAIN for f in extra) or len({f.name for f in extra}) != len(extra):
            raise ValueError("every extra camera needs its own name, not 'main'")
        self.extra = tuple(extra)
        self.source = source
        self.telemetry = telemetry
        self.send = send
        self.clock = clock
        self.camera = camera
        self.registry = registry or default_registry()
        self.period_s = period_s
        self.keep_dir = keep_dir
        self.env = env
        self.off = off  # (reason code, error class) when perception cannot work
        self.on = on  # perception works again after ``stills_failing``
        self._failing: dict[str, int] = {}  # stills failing in a row, per camera
        self._reported: set[str] = set()  # cameras whose streak the ground was told of
        self._built: BuiltSources | None = sources
        self._build_failed = False  # ``run`` could not build the sources: no stills at all
        self.stats = PerceptionStats()
        self.rate = AttitudeRate()
        self._last_still_ms: dict[str, int] = {}  # per camera
        self.on_targets = on_targets
        self._lock = asyncio.Lock()  # one still at a time: the loop's or an order's

    @property
    def built(self) -> BuiltSources:
        """The detector sources, built on first use (``run`` builds them off the event
        loop)."""
        if self._built is None:
            self._built = build_sources(self.registry, env=self.env)
        return self._built

    def _camera(self, name: str) -> PinholeCamera:
        if name == MAIN:
            return self.camera
        return next(f.camera for f in self.extra if f.name == name)

    def _source(self, name: str) -> StillSource:
        if name == MAIN:
            return self.source
        return next(f.source for f in self.extra if f.name == name)

    @property
    def cameras(self) -> list[str]:
        """Every camera's name, the main one first: the order the loop polls them in."""
        return [MAIN, *(f.name for f in self.extra)]

    def handle(
        self,
        captured: CapturedStill,
        t: FcTelemetry,
        now_ms: int,
        turn_rate_dps: float = 0.0,
        command_id: UUID | None = None,
        camera: str = MAIN,
        captured_utc: datetime | None = None,
    ) -> Perception | None:
        """One still and the telemetry read when it arrived (with the attitude rate then):
        a report and its crops, or ``None`` when it cannot be placed on the ground well
        enough. ``camera``: which camera took it (``MAIN`` or an ``extra`` feed's name).
        ``captured_utc``: when it was taken (``_poll``); without it the source's time, else
        now, before the detector runs."""
        stamp = captured_utc or captured.taken_utc or self.clock.utc_now()
        pinhole = self._camera(camera)
        self.stats.stills += 1
        last = self._last_still_ms.get(camera)
        self._last_still_ms[camera] = now_ms
        rate_hz = 1000.0 / (now_ms - last) if last is not None and now_ms > last else 0.0
        pose = pose_from(t, now_ms)
        if pose is None:
            self.stats.skipped_no_pose += 1
            return None
        if pose.agl_m < MIN_AGL_M:
            self.stats.skipped_on_ground += 1
            return None
        if turn_rate_dps > MAX_TURN_RATE_DPS:
            self.stats.skipped_turning += 1
            return None
        start = self.clock.monotonic_ms()
        found = detect_all(captured.still, self.built.sources, self.registry)
        out = process(
            captured.still.frame,
            found,
            pose,
            pinhole,
            self.registry,
            still_id=new_id(),
            captured_utc=stamp,
            synthetic=captured.synthetic,
            sources=sorted(self.built.sources),
            latency_ms=max(0, self.clock.monotonic_ms() - start),
            rate_hz=rate_hz,
            command_id=command_id,
        )
        if out is None:
            self.stats.skipped_no_footprint += 1
            return None
        report = out.report
        self.stats.reports += 1
        self.stats.by_camera[camera] = self.stats.by_camera.get(camera, 0) + 1
        self.stats.detections += len(report.detections)
        for d in report.detections:
            label = d.label or "?"
            self.stats.by_label[label] = self.stats.by_label.get(label, 0) + 1
        if report.detections and self.keep_dir is not None:
            self._keep(captured.still, out, self.keep_dir)
        return out

    def _keep(self, still: Still, out: Perception, folder: Path) -> None:
        """Save a still that had detections (JPEG, no metadata) for the run's record."""
        from PIL import Image

        folder.mkdir(parents=True, exist_ok=True)
        f = still.frame
        report = out.report
        Image.frombytes("RGB", (f.width, f.height), f.rgb).save(
            folder / f"{report.still_id}.jpg", format="JPEG", quality=90
        )
        (folder / f"{report.still_id}.json").write_text(report.model_dump_json(), encoding="utf-8")
        kept = sorted(folder.glob("*.json"))  # still ids are time-ordered (UUIDv7)
        for old in kept[: max(0, len(kept) - KEEP_MAX)]:
            old.with_suffix(".jpg").unlink(missing_ok=True)
            old.unlink(missing_ok=True)

    def _deliver(self, out: Perception) -> None:
        self.send(out)
        if self.on_targets is None:
            return
        targets = [
            d
            for d in out.report.detections
            if d.label is not None
            and d.label in self.registry
            and self.registry.get(d.label).kind == "target"
        ]
        if targets:
            self.on_targets(targets)

    async def _poll(self, camera: str) -> _Polled | None:
        """A camera's newest still, stamped with when it was taken (the source's time,
        else now: right after the poll, before any detector runs) and the telemetry then."""
        rate = self.rate.update(self.telemetry())
        captured = await asyncio.to_thread(self._source(camera).poll)
        if captured is None:
            return None
        stamp = captured.taken_utc or self.clock.utc_now()
        t, now_ms = self.telemetry(), self.clock.monotonic_ms()
        rate = max(rate, self.rate.update(t))
        return _Polled(camera, captured, t, now_ms, rate, stamp)

    async def _detect(self, p: _Polled, command_id: UUID | None = None) -> Perception | None:
        """The detector on a polled still; its report, if it could be placed."""
        ran = self._detected()
        out = await asyncio.to_thread(
            self.handle, p.captured, p.t, p.now_ms, p.rate, command_id, p.camera, p.utc
        )
        if out is not None:
            self._deliver(out)
        if self._detected() > ran:  # a skipped still says nothing either way
            self._back(p.camera)
        return out

    async def _one(self, command_id: UUID | None = None, camera: str = MAIN) -> Perception | None:
        """Poll a camera's source once; the still's report, if there was one and it could
        be placed."""
        polled = await self._poll(camera)
        return None if polled is None else await self._detect(polled, command_id)

    async def capture(self, command_id: UUID) -> UUID | None:
        """One still for an order: its ``still_id``, or ``None`` when none came in
        ``CAPTURE_WAIT_S`` or it could not be placed on the ground."""
        if self._build_failed:
            return None
        deadline = self.clock.monotonic() + CAPTURE_WAIT_S
        # A still stuck in the loop must not hold the order past its wait: a watchdog, so on
        # the event loop's (wall) clock, which runs even when the stuck still holds the lock.
        try:
            await asyncio.wait_for(self._lock.acquire(), CAPTURE_WAIT_S)
        except TimeoutError:
            log.warning("perception busy: no still for the order")
            return None
        try:
            while True:
                try:
                    out = await self._one(command_id)
                except Exception:
                    self.stats.failed += 1
                    log.exception("perception failed on an order's still")
                    return None
                if out is not None:
                    return out.report.still_id
                if self.clock.monotonic() >= deadline:
                    return None
                await self.clock.sleep(_CAPTURE_POLL_S)
        finally:
            self._lock.release()

    def _detected(self) -> int:
        """Stills the detector ran on (a report, or a view that misses the ground)."""
        return self.stats.reports + self.stats.skipped_no_footprint

    def _back(self, camera: str) -> None:
        """The detector ran on ``camera``'s still: its streak ends. The ground hears
        detection is back once no camera's streak it was told of is left."""
        self._failing[camera] = 0
        if camera not in self._reported:
            return
        self._reported.discard(camera)
        if not self._reported and self.on is not None:
            try:  # telling the ground never counts against the still
                self.on()
            except Exception:
                log.exception("perception: could not tell the ground detection is back")

    def _fail(self, camera: str, exc: Exception) -> None:
        """A still of ``camera`` failed: its streak grows; at ``FAIL_REPORT`` the ground
        hears ``stills_failing`` (once while any camera's streak is told)."""
        self.stats.failed += 1
        n = self._failing.get(camera, 0) + 1
        self._failing[camera] = n
        log.exception("perception failed on a still (%s camera)", camera, exc_info=exc)
        if n != FAIL_REPORT:
            return
        first = not self._reported
        self._reported.add(camera)
        if first and self.off is not None:
            try:  # telling the ground never ends perception
                self.off("stills_failing", type(exc).__name__)
            except Exception:
                log.exception("perception: could not tell the ground detection is off")

    async def run(self, stop: asyncio.Event) -> None:
        try:
            await asyncio.to_thread(lambda: self.built)
        except Exception as exc:
            log.exception("perception off: the detector sources could not be built")
            self._build_failed = True
            if self.off is not None:
                self.off("sources_failed", type(exc).__name__)
            return
        log.info(
            "perception on: cameras %s, sources %s, skipped %s",
            self.cameras,
            sorted(self.built.sources),
            self.built.skipped,
        )
        while not stop.is_set():
            await self._round()
            await self.clock.sleep(self.period_s)

    async def _round(self) -> None:
        """Every camera's newest still first (stamped as it is polled), then the detector
        on each, so a slow detector on one still never delays the next still's stamp. An
        order's still may come between two detections (each holds the lock alone)."""
        polled: list[_Polled] = []
        async with self._lock:
            for camera in self.cameras:
                try:  # one bad still must not end perception
                    got = await self._poll(camera)
                except Exception as exc:
                    self._fail(camera, exc)
                    continue
                if got is not None:
                    polled.append(got)
        for p in polled:
            try:
                async with self._lock:
                    await self._detect(p)
            except Exception as exc:
                self._fail(p.camera, exc)
