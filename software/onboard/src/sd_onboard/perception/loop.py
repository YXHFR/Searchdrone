"""The perception task: poll the still source, run the pipeline, queue the report and crops.

Runs beside the executor in its own task and only reads: the flight controller's latest
telemetry (``Vehicle.telemetry``) and the still source. It never sends anything to the
flight controller and never touches the executor. Reports go to the link's
lowest-priority queue (``OnboardLink.perception``), each report followed by its crops.
Reading the still and the detector work run in a worker thread so a slow picture never
blocks the event loop the executor ticks on. A failing still is logged and skipped; the
loop keeps going. The detector sources (a trained model among them) are built in a
worker thread when the task starts, after the executor and link run; if that fails,
perception is off and the reason is logged, and flying goes on. Kept stills (for a
run's record) are capped at ``KEEP_MAX``, the oldest removed first.

The pose is the telemetry read when a still arrives, a moment after the picture was
taken. In steady flight that moment does not matter; while the attitude changes quickly
(a turn, braking, yawing towards a find) it does: 10 degrees of attitude at 40 m move a
detection about 7 m. The loop follows the attitude rate between telemetry samples and
skips stills taken while it is above ``MAX_TURN_RATE_DPS``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sd_detect import ClassRegistry, Still, build_sources, default_registry, detect_all
from sd_detect.sources import BuiltSources
from sd_geo import PinholeCamera
from sd_mav.vehicle import FcTelemetry
from sd_obs.clock import Clock
from sd_onboard.perception.pipeline import MIN_AGL_M, Perception, Pose, process
from sd_onboard.perception.sources import CapturedStill, StillSource
from sd_schemas.base import new_id

__all__ = [
    "KEEP_MAX",
    "MAX_POSE_AGE_MS",
    "MAX_TURN_RATE_DPS",
    "AttitudeRate",
    "PerceptionLoop",
    "PerceptionStats",
    "pose_from",
]

log = logging.getLogger("sd_onboard.perception")
MAX_POSE_AGE_MS = 1000  # older position or attitude: the still is skipped
KEEP_MAX = 200  # kept stills (JPEG + JSON pairs) in ``keep_dir``
MAX_TURN_RATE_DPS = 10.0  # faster attitude change: the still is skipped


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
    ) -> None:
        self.source = source
        self.telemetry = telemetry
        self.send = send
        self.clock = clock
        self.camera = camera
        self.registry = registry or default_registry()
        self.period_s = period_s
        self.keep_dir = keep_dir
        self.env = env
        self._built: BuiltSources | None = None
        self.stats = PerceptionStats()
        self.rate = AttitudeRate()
        self._last_still_ms: int | None = None

    @property
    def built(self) -> BuiltSources:
        """The detector sources, built on first use (``run`` builds them off the event
        loop)."""
        if self._built is None:
            self._built = build_sources(self.registry, env=self.env)
        return self._built

    def handle(
        self, captured: CapturedStill, t: FcTelemetry, now_ms: int, turn_rate_dps: float = 0.0
    ) -> Perception | None:
        """One still and the telemetry read when it arrived (with the attitude rate then):
        a report and its crops, or ``None`` when it cannot be placed on the ground well
        enough."""
        self.stats.stills += 1
        last, self._last_still_ms = self._last_still_ms, now_ms
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
            self.camera,
            self.registry,
            still_id=new_id(),
            captured_utc=self.clock.utc_now(),
            synthetic=captured.synthetic,
            sources=sorted(self.built.sources),
            latency_ms=max(0, self.clock.monotonic_ms() - start),
            rate_hz=rate_hz,
        )
        if out is None:
            self.stats.skipped_no_footprint += 1
            return None
        report = out.report
        self.stats.reports += 1
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

    async def run(self, stop: asyncio.Event) -> None:
        try:
            await asyncio.to_thread(lambda: self.built)
        except Exception:
            log.exception("perception off: the detector sources could not be built")
            return
        log.info(
            "perception on: sources %s, skipped %s", sorted(self.built.sources), self.built.skipped
        )
        while not stop.is_set():
            try:  # one bad still must not end perception
                rate = self.rate.update(self.telemetry())
                captured = await asyncio.to_thread(self.source.poll)
                if captured is not None:
                    t, now_ms = self.telemetry(), self.clock.monotonic_ms()
                    rate = max(rate, self.rate.update(t))
                    out = await asyncio.to_thread(self.handle, captured, t, now_ms, rate)
                    if out is not None:
                        self.send(out)
            except Exception:
                self.stats.failed += 1
                log.exception("perception failed on a still")
            await self.clock.sleep(self.period_s)
