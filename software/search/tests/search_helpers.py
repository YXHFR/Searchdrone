"""Shared builders for the sd-search tests: sightings, profiles and synthetic crops."""

from __future__ import annotations

import datetime as dt
import io
from typing import Any

from PIL import Image, ImageDraw

from sd_schemas.base import new_id
from sd_schemas.geo import plane_offset
from sd_schemas.search import SearchProfile
from sd_search.fusion import Sighting

# Public CMAC test field (ArduPilot's default SITL home), never a private place.
HOME = (-35.363261, 149.165230)
T0 = dt.datetime(2026, 10, 1, 12, 0, tzinfo=dt.UTC)
MISSION = "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a21"

GRASS = (70, 120, 40)
RED = (200, 30, 30)
BLUE = (30, 60, 190)
YELLOW = (230, 210, 30)


def at(east: float, north: float) -> tuple[float, float]:
    return plane_offset(*HOME, east, north)


def sighting(
    east: float,
    north: float,
    *,
    seq: int = 1,
    still: str | None = None,
    label: str = "person",
    sigma: float = 3.0,
    score: float = 0.6,
    t: float = 0.0,
    px: float = 40.0,
    crop: bool = False,
) -> Sighting:
    lat, lon = at(east, north)
    return Sighting(
        detection_id=str(new_id()),
        seq=seq,
        still_id=still or str(new_id()),
        label=label,
        lat=lat,
        lon=lon,
        sigma95_m=sigma,
        score=score,
        px_size=px,
        captured_utc=T0 + dt.timedelta(seconds=t),
        has_crop=crop,
    )


def profile(**kw: Any) -> SearchProfile:
    lat, lon = HOME
    base: dict[str, Any] = {
        "profile_id": str(new_id()),
        "profile_version": 1,
        "mission_id": MISSION,
        "subject_category": "hiker",
        "garments": [{"part": "upper", "colors": ["red"], "certainty": "certain"}],
        "last_seen": {"lat": lat, "lon": lon, "at_utc": T0.isoformat()},
        "approved_by": "operator:demo",
        "approved_at": T0.isoformat(),
    }
    return SearchProfile.model_validate(base | kw)


def crop_jpeg(
    body: tuple[int, int, int] | list[tuple[int, int, int]],
    *,
    size: int = 96,
    box: int = 32,
    background: tuple[int, int, int] = GRASS,
) -> bytes:
    """A crop as onboard cuts it: the box in the middle, a third of the side, on grass.
    ``body`` as a list paints horizontal bands from the top."""
    img = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(img)
    x0 = (size - box) // 2
    bands = body if isinstance(body, list) else [body]
    step = box / len(bands)
    for i, color in enumerate(bands):
        draw.rectangle(
            (x0, x0 + round(i * step), x0 + box - 1, x0 + round((i + 1) * step) - 1), fill=color
        )
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=92)
    return out.getvalue()
