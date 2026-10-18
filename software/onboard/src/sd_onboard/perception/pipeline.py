"""One still and the pose at that moment in, one ``PerceptionReport`` out.

Steps: every detector source the class register names (``sd_detect.detect_all``), the
size check in metres (``keep_plausible``), georeferencing of each box's centre on flat
ground below the vehicle (``sd_geo``), and a small JPEG thumbnail around each target
(a search target, not context such as roads). At most ``PERCEPTION_MAX_DETECTIONS``
detections go in a report: targets first, then by score; the rest are counted.

Pure: no clock, no I/O. The pose is the flight controller's at about the moment the
still arrived; its age is part of the position error.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from PIL import Image

from sd_detect import ClassRegistry, Detection, Frame, keep_plausible
from sd_geo import Attitude, PinholeCamera, footprint, ground_offset, px_per_m, sigma95_m
from sd_schemas.base import new_id
from sd_schemas.geo import plane_offset
from sd_schemas.link import (
    CROP_B64_MAX,
    PERCEPTION_MAX_DETECTIONS,
    PerceptionDetection,
    PerceptionReport,
    StillPose,
)

__all__ = ["CROP_BUDGET_B64", "MIN_AGL_M", "Pose", "crop_jpeg", "process"]

MIN_AGL_M = 2.0  # below this the vehicle is on or near the ground: no report
CROP_BUDGET_B64 = 40_000  # all thumbnails of one report together
CROP_MIN_PX, CROP_MAX_PX, THUMB_PX = 48, 256, 128


@dataclass(frozen=True)
class Pose:
    lat: float
    lon: float
    alt_amsl_m: float
    agl_m: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    speed_ms: float = 0.0


def crop_jpeg(frame: Frame, d: Detection) -> bytes:
    """A square around the box (three times its size, 48-256 px), at most 128 px wide."""
    side = int(min(CROP_MAX_PX, max(CROP_MIN_PX, 3 * max(d.bbox.w, d.bbox.h))))
    cx, cy = d.bbox.x + d.bbox.w / 2, d.bbox.y + d.bbox.h / 2
    x0 = int(min(max(0, cx - side / 2), max(0, frame.width - side)))
    y0 = int(min(max(0, cy - side / 2), max(0, frame.height - side)))
    img = Image.frombytes("RGB", (frame.width, frame.height), frame.rgb)
    img = img.crop((x0, y0, min(frame.width, x0 + side), min(frame.height, y0 + side)))
    img.thumbnail((THUMB_PX, THUMB_PX))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=80)  # no EXIF: nothing but the pixels
    return out.getvalue()


def process(
    frame: Frame,
    detections: list[Detection],
    pose: Pose,
    camera: PinholeCamera,
    registry: ClassRegistry,
    *,
    still_id: UUID,
    captured_utc: datetime,
    synthetic: bool,
    sources: list[str],
    skipped: dict[str, str],
    latency_ms: int,
) -> PerceptionReport:
    att = Attitude(pose.roll_deg, pose.pitch_deg, pose.yaw_deg)
    h = pose.agl_m
    ppm = px_per_m(camera, h)
    kept = keep_plausible(detections, ppm, registry)
    kept.sort(key=lambda d: (registry.get(d.label).kind != "target", -d.score))
    out: list[PerceptionDetection] = []
    budget = CROP_BUDGET_B64
    for d in kept[:PERCEPTION_MAX_DETECTIONS]:
        cx, cy = d.bbox.x + d.bbox.w / 2, d.bbox.y + d.bbox.h / 2
        off = ground_offset(cx, cy, camera, att, h)
        if off is None:
            continue
        lat, lon = plane_offset(pose.lat, pose.lon, off[0], off[1])
        crop: str | None = None
        if registry.get(d.label).kind == "target":
            b64 = base64.b64encode(crop_jpeg(frame, d)).decode("ascii")
            if len(b64) <= min(CROP_B64_MAX, budget):
                crop, budget = b64, budget - len(b64)
        out.append(
            PerceptionDetection(
                detection_id=new_id(),
                label=d.label,
                source=d.source or "model",
                score=d.score,
                bbox=(d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h),
                lat=lat,
                lon=lon,
                sigma95_m=round(sigma95_m(math.hypot(*off), h, pose.speed_ms), 2),
                size_m=round(max(d.bbox.w, d.bbox.h) / ppm, 2),
                crop_jpeg_b64=crop,
            )
        )
    corners = footprint(camera, att, h) or []
    return PerceptionReport(
        still_id=still_id,
        captured_utc=captured_utc,
        pose=StillPose(
            lat=pose.lat,
            lon=pose.lon,
            alt_amsl_m=pose.alt_amsl_m,
            agl_m=h,
            roll_deg=max(-180.0, min(180.0, pose.roll_deg)),
            pitch_deg=max(-90.0, min(90.0, pose.pitch_deg)),
            yaw_deg=pose.yaw_deg % 360.0,
        ),
        footprint=[plane_offset(pose.lat, pose.lon, e, n) for e, n in corners],
        px_per_m=round(ppm, 3),
        synthetic=synthetic,
        sources=sources,
        skipped_sources=skipped,
        latency_ms=latency_ms,
        detections=out,
        detections_dropped=max(0, len(kept) - len(out)),
    )
