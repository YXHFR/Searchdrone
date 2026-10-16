"""Building blocks shared by every schema: the model base class, ids, positions and GeoJSON."""

from __future__ import annotations

import json
import math
import uuid
from typing import Annotated, Any, cast

import uuid_utils
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, WithJsonSchema

__all__ = [
    "ISSUER_PATTERN",
    "MAX_POLYGON_POINTS",
    "MIN_POLYGON_POINTS",
    "SKILL_NAME_PATTERN",
    "VEHICLE_ID_PATTERN",
    "GeoJsonPoint",
    "GeoJsonPolygon",
    "Id",
    "LatLon",
    "LatLonAlt",
    "SdModel",
    "VehicleId",
    "json_bytes_at_most",
    "new_id",
]


class SdModel(BaseModel):
    """Base for every message: unknown fields are rejected, instances are immutable and
    floats must be finite (no inf or NaN)."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


_UUID7_PATTERN = (
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-7[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)


def _require_v7(value: uuid.UUID) -> uuid.UUID:
    if value.version != 7 or value.variant != uuid.RFC_4122:
        raise ValueError("ids are UUIDv7")
    return value


Id = Annotated[
    uuid.UUID,
    AfterValidator(_require_v7),
    WithJsonSchema({"type": "string", "format": "uuid", "pattern": _UUID7_PATTERN}),
]
"""Every id is a UUIDv7 (time-ordered). Create new ones with :func:`new_id`."""


def new_id() -> uuid.UUID:
    """Return a new UUIDv7 as a standard :class:`uuid.UUID`."""
    return uuid.UUID(bytes=uuid_utils.uuid7().bytes)


VEHICLE_ID_PATTERN = r"^[a-z0-9][a-z0-9-]{2,31}$"
SKILL_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
ISSUER_PATTERN = r"^(operator:[a-z0-9][a-z0-9_-]{0,31}|brain:ground|system:ground|rule:onboard)$"
VehicleId = Annotated[str, StringConstraints(pattern=VEHICLE_ID_PATTERN)]
"""Vehicle id such as ``sim-01``. Never a name that reveals a person or a place."""


class LatLon(SdModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class LatLonAlt(LatLon):
    alt_amsl_m: float = Field(ge=-500, le=9000)


MIN_POLYGON_POINTS = 4  # a triangle plus the closing point
MAX_POLYGON_POINTS = 201  # 200 corners plus the closing point


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("a coordinate is a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("coordinates must be finite")
    return number


def _position(value: object) -> tuple[float, float]:
    if not isinstance(value, list | tuple):
        raise ValueError("a position is [lon, lat]")
    items = list(cast("list[object] | tuple[object, ...]", value))
    if len(items) != 2:
        raise ValueError("a position is [lon, lat]")
    lon, lat = _number(items[0]), _number(items[1])
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError("coordinates out of range (GeoJSON order is [lon, lat])")
    return lon, lat


def _check_point(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != {"type", "coordinates"} or value["type"] != "Point":
        raise ValueError('expected {"type": "Point", "coordinates": [lon, lat]}')
    _position(value["coordinates"])
    return value


def _check_polygon(value: dict[str, Any]) -> dict[str, Any]:
    """Shape checks only. Self-intersection is checked by the gate and the charter validator."""
    if set(value) != {"type", "coordinates"} or value["type"] != "Polygon":
        raise ValueError('expected {"type": "Polygon", "coordinates": [[[lon, lat], ...]]}')
    rings: object = value["coordinates"]
    if not isinstance(rings, list) or len(cast("list[object]", rings)) != 1:
        raise ValueError("a polygon has exactly one outer ring and no holes")
    ring: object = cast("list[object]", rings)[0]
    if not isinstance(ring, list):
        raise ValueError("the ring is a list of positions")
    positions = cast("list[object]", ring)
    if not MIN_POLYGON_POINTS <= len(positions) <= MAX_POLYGON_POINTS:
        raise ValueError(
            f"the ring has {MIN_POLYGON_POINTS}-{MAX_POLYGON_POINTS} positions (closed)"
        )
    points = [_position(p) for p in positions]
    if points[0] != points[-1]:
        raise ValueError("the ring must be closed (first position == last position)")
    if len(set(points[:-1])) < 3:
        raise ValueError("the ring needs at least three distinct corners")
    return value


GeoJsonPoint = Annotated[dict[str, Any], AfterValidator(_check_point)]
"""GeoJSON Point ``{"type": "Point", "coordinates": [lon, lat]}``."""

GeoJsonPolygon = Annotated[dict[str, Any], AfterValidator(_check_polygon)]
"""GeoJSON Polygon with one closed outer ring of 4-201 positions and no holes."""


def json_bytes_at_most(limit: int) -> AfterValidator:
    """Reject a value whose compact JSON form is larger than ``limit`` bytes."""

    def check(value: Any) -> Any:
        size = len(json.dumps(value, separators=(",", ":"), default=str).encode())
        if size > limit:
            raise ValueError(f"too large: {size} bytes, at most {limit}")
        return value

    return AfterValidator(check)
