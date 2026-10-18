"""The perception task: poll the still source, run the pipeline, queue the report.

Runs beside the executor in its own task and only reads: the flight controller's latest
telemetry (``Vehicle.telemetry``) and the still source. It never sends anything to the
flight controller and never touches the executor. Reports go to the link's
lowest-priority queue (``OnboardLink.perception``). The detector work runs in a worker
thread so a slow picture never blocks the event loop the executor ticks on. A failing
still is logged and skipped; the loop keeps going.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sd_detect import ClassRegistry, Still, build_sources, default_registry, detect_all
from sd_geo import PinholeCamera
from sd_mav.vehicle import FcTelemetry
from sd_obs.clock import Clock
from sd_onboard.perception.pipeline import MIN_AGL_M, Pose, process
from sd_onboard.perception.sources import CapturedStill, StillSource
from sd_schemas.base import new_id
from sd_schemas.link import PerceptionReport

__all__ = ["MAX_POSE_AGE_MS", "PerceptionLoop", "PerceptionStats", "pose_from"]

log = logging.getLogger("sd_onboard.perception")
MAX_POSE_AGE_MS = 1000  # older position or attitude: the still is skipped


@dataclass
class PerceptionStats:
    stills: int = 0
    reports: int = 0
    detections: int = 0
    skipped_no_pose: int = 0
    skipped_on_ground: int = 0
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
    )


class PerceptionLoop:
    def __init__(
        self,
        source: StillSource,
        telemetry: Callable[[], FcTelemetry],
        send: Callable[[PerceptionReport], None],
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
        self.built = build_sources(self.registry, env=env)
        self.stats = PerceptionStats()

    def handle(
        self, captured: CapturedStill, t: FcTelemetry, now_ms: int
    ) -> PerceptionReport | None:
        """One still and the telemetry read when it arrived: a report, or ``None`` when
        it cannot be placed on the ground."""
        self.stats.stills += 1
        pose = pose_from(t, now_ms)
        if pose is None:
            self.stats.skipped_no_pose += 1
            return None
        if pose.agl_m < MIN_AGL_M:
            self.stats.skipped_on_ground += 1
            return None
        start = self.clock.monotonic_ms()
        found = detect_all(captured.still, self.built.sources, self.registry)
        report = process(
            captured.still.frame,
            found,
            pose,
            self.camera,
            self.registry,
            still_id=new_id(),
            captured_utc=self.clock.utc_now(),
            synthetic=captured.synthetic,
            sources=sorted(self.built.sources),
            skipped=dict(self.built.skipped),
            latency_ms=max(0, self.clock.monotonic_ms() - start),
        )
        self.stats.reports += 1
        self.stats.detections += len(report.detections)
        for d in report.detections:
            self.stats.by_label[d.label] = self.stats.by_label.get(d.label, 0) + 1
        if report.detections and self.keep_dir is not None:
            self._keep(captured.still, report, self.keep_dir)
        return report

    def _keep(self, still: Still, report: PerceptionReport, folder: Path) -> None:
        """Save a still that had detections (JPEG, no metadata) for the run's record."""
        from PIL import Image

        folder.mkdir(parents=True, exist_ok=True)
        f = still.frame
        Image.frombytes("RGB", (f.width, f.height), f.rgb).save(
            folder / f"{report.still_id}.jpg", format="JPEG", quality=90
        )
        (folder / f"{report.still_id}.json").write_text(
            report.model_dump_json(exclude={"detections": {"__all__": {"crop_jpeg_b64"}}}),
            encoding="utf-8",
        )

    async def run(self, stop: asyncio.Event) -> None:
        log.info(
            "perception on: sources %s, skipped %s", sorted(self.built.sources), self.built.skipped
        )
        while not stop.is_set():
            captured = self.source.poll()
            if captured is not None:
                t, now_ms = self.telemetry(), self.clock.monotonic_ms()
                try:
                    report = await asyncio.to_thread(self.handle, captured, t, now_ms)
                except Exception:  # one bad still must not end perception
                    self.stats.failed += 1
                    log.exception("perception failed on still %s", captured.name)
                    report = None
                if report is not None:
                    self.send(report)
            await self.clock.sleep(self.period_s)
