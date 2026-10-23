"""One still and the pose at that moment in, one ``PerceptionReport`` and its crops out.

Steps: every detector source the class register names (``sd_detect.detect_all``), the
size check in metres (``keep_plausible``), georeferencing of each box's centre on flat
ground below the vehicle (``sd_geo``), and a small JPEG crop (``CropData``) around each
target (a search target, not context such as roads), at most ``CROPS_PER_STILL``. At
most ``PERCEPTION_DETECTIONS_MAX`` detections go in a report: targets first, then by
score; the rest are counted.

Pure: no clock, no I/O. The pose is the flight controller's at about the moment the
still arrived; its age is part of the position error.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from PIL import Image

from sd_detect import ClassRegistry, Detection, Frame, __version__, keep_plausible
from sd_geo import Attitude, PinholeCamera, footprint, ground_offset, px_per_m, sigma95_m
from sd_schemas.base import LatLon, new_id
from sd_schemas.geo import plane_offset
from sd_schemas.link import (
    PERCEPTION_DETECTIONS_MAX,
    BBox,
    CropData,
    DetectorInfo,
    GeoDetectionMsg,
    PerceptionReport,
)
from sd_schemas.link import Pose as StillPose

__all__ = ["CROPS_PER_STILL", "MIN_AGL_M", "Perception", "Pose", "crop_jpeg", "process"]

MIN_AGL_M = 2.0  # below this the vehicle is on or near the ground: no report
CROPS_PER_STILL = 4  # crops sent with one report; more targets go without
CROP_MIN_PX, CROP_SIDE_MAX_PX, THUMB_PX = 48, 256, 128


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
    source_ts_ms: int = 0  # the telemetry's time (monotonic ms)


@dataclass(frozen=True)
class Perception:
    """A report and the crops that follow it on the link."""

    report: PerceptionReport
    crops: list[CropData] = field(default_factory=list[CropData])


def crop_jpeg(frame: Frame, d: Detection) -> tuple[BBox, bytes, int, int]:
    """A square around the box (three times its size, 48-256 px), at most 128 px wide:
    the square in the still, the JPEG and its size."""
    side = int(min(CROP_SIDE_MAX_PX, max(CROP_MIN_PX, 3 * max(d.bbox.w, d.bbox.h))))
    cx, cy = d.bbox.x + d.bbox.w / 2, d.bbox.y + d.bbox.h / 2
    x0 = int(min(max(0, cx - side / 2), max(0, frame.width - side)))
    y0 = int(min(max(0, cy - side / 2), max(0, frame.height - side)))
    x1, y1 = min(frame.width, x0 + side), min(frame.height, y0 + side)
    img = Image.frombytes("RGB", (frame.width, frame.height), frame.rgb).crop((x0, y0, x1, y1))
    img.thumbnail((THUMB_PX, THUMB_PX))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=80)  # no EXIF: nothing but the pixels
    return BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0), out.getvalue(), img.width, img.height


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
    latency_ms: float,
    rate_hz: float,
    command_id: UUID | None = None,
) -> Perception | None:
    """``None`` when the camera's view does not reach the ground (no footprint)."""
    att = Attitude(pose.roll_deg, pose.pitch_deg, pose.yaw_deg)
    h = pose.agl_m
    corners = footprint(camera, att, h)
    if corners is None:
        return None
    ppm = px_per_m(camera, h)
    kept = keep_plausible(detections, ppm, registry)
    kept.sort(key=lambda d: (registry.get(d.label).kind != "target", -d.score))
    out: list[GeoDetectionMsg] = []
    crops: list[CropData] = []
    for d in kept[:PERCEPTION_DETECTIONS_MAX]:
        cx, cy = d.bbox.x + d.bbox.w / 2, d.bbox.y + d.bbox.h / 2
        off = ground_offset(cx, cy, camera, att, h)
        if off is None:
            continue
        lat, lon = plane_offset(pose.lat, pose.lon, off[0], off[1])
        did = new_id()
        crop = registry.get(d.label).kind == "target" and len(crops) < CROPS_PER_STILL
        if crop:
            square, jpeg, w, hpx = crop_jpeg(frame, d)
            crops.append(
                CropData(
                    ref_id=did,
                    purpose="detection",
                    still_id=still_id,
                    bbox_in_still=square,
                    width=w,
                    height=hpx,
                    jpeg=jpeg,
                )
            )
        out.append(
            GeoDetectionMsg(
                detection_id=did,
                bbox=BBox(x=d.bbox.x, y=d.bbox.y, w=d.bbox.w, h=d.bbox.h),
                score=d.score,
                lat=lat,
                lon=lon,
                sigma95_m=min(1000.0, round(sigma95_m(math.hypot(*off), h, pose.speed_ms), 2)),
                px_size=max(d.bbox.w, d.bbox.h),
                crop_inline=crop,
                label=d.label,
                source=d.source or "model",
            )
        )
    report = PerceptionReport(
        still_id=still_id,
        command_id=command_id,
        captured_utc=captured_utc,
        sensor="rgb",
        pose=StillPose(
            lat=pose.lat,
            lon=pose.lon,
            alt_amsl_m=pose.alt_amsl_m,
            agl_m=h,
            roll_deg=max(-180.0, min(180.0, pose.roll_deg)),
            pitch_deg=max(-180.0, min(180.0, pose.pitch_deg)),
            yaw_deg=pose.yaw_deg % 360.0,
            source_ts_ms=max(0, pose.source_ts_ms),
        ),
        footprint=[
            LatLon(lat=la, lon=lo)
            for la, lo in (plane_offset(pose.lat, pose.lon, e, n) for e, n in corners)
        ],
        px_per_m=round(ppm, 3),
        synthetic=synthetic,
        detector=DetectorInfo(
            name=("sd_detect:" + "+".join(sources))[:64],
            license="Apache-2.0",
            version=__version__,
            placement="onboard",
            ok=True,
            tiles=1,
            latency_ms=round(max(0.0, latency_ms), 1),
            rate_hz=round(max(0.0, rate_hz), 3),
        ),
        detections=out,
        detections_dropped=max(0, len(kept) - len(out)),
    )
    return Perception(report, crops)
