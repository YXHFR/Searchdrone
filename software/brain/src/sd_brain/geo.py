"""Small local geometry: east/north metres around an origin (flat-earth tangent plane).

Good to well under a metre within the few kilometres a mission covers.
"""

from __future__ import annotations

import math

__all__ = ["EARTH_RADIUS_M", "bearing_word", "from_enu", "to_enu"]

EARTH_RADIUS_M = 6_378_137.0


def to_enu(lat: float, lon: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    """(east_m, north_m) of a point relative to the origin."""
    east = math.radians(lon - origin_lon) * EARTH_RADIUS_M * math.cos(math.radians(origin_lat))
    north = math.radians(lat - origin_lat) * EARTH_RADIUS_M
    return east, north


def from_enu(
    east: float, north: float, origin_lat: float, origin_lon: float
) -> tuple[float, float]:
    """(lat, lon) of a point given in metres east and north of the origin."""
    lat = origin_lat + math.degrees(north / EARTH_RADIUS_M)
    lon = origin_lon + math.degrees(east / (EARTH_RADIUS_M * math.cos(math.radians(origin_lat))))
    return lat, lon


_WORDS = ("norr", "nordost", "öster", "sydost", "söder", "sydväst", "väster", "nordväst")


def bearing_word(east: float, north: float) -> str:
    """The Swedish compass word for a direction (eight sectors)."""
    deg = math.degrees(math.atan2(east, north)) % 360
    return _WORDS[round(deg / 45) % 8]
