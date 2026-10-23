"""The local east/north tangent plane on the WGS84 ellipsoid, shared by every package.

``sd_safety.targets.Plane`` measures in it, and the schemas build shapes in it (an orbit's
circle), so both use these functions and agree to the millimetre.
"""

from __future__ import annotations

import math
from typing import Final

__all__ = [
    "WGS84_A",
    "WGS84_E2",
    "WGS84_F",
    "plane_distance_m",
    "plane_offset",
    "tangent_radii",
    "wrap_lon",
]

WGS84_A: Final = 6_378_137.0
WGS84_F: Final = 1 / 298.257223563
WGS84_E2: Final = WGS84_F * (2 - WGS84_F)


def wrap_lon(dlon: float) -> float:
    return (dlon + 180.0) % 360.0 - 180.0


def tangent_radii(lat0: float) -> tuple[float, float]:
    """Metres per radian east and north at latitude ``lat0``."""
    phi = math.radians(lat0)
    s = math.sin(phi)
    w = math.sqrt(1 - WGS84_E2 * s * s)
    n_radius = WGS84_A / w  # prime vertical
    m_radius = WGS84_A * (1 - WGS84_E2) / (w * w * w)  # meridian
    return n_radius * math.cos(phi), m_radius


def plane_offset(lat0: float, lon0: float, east: float, north: float) -> tuple[float, float]:
    """(lat, lon) of the point ``east``/``north`` metres from (lat0, lon0) in the plane."""
    east_r, north_r = tangent_radii(lat0)
    lat = lat0 + math.degrees(north / north_r)
    lon = wrap_lon(lon0 + math.degrees(east / east_r))
    return lat, lon


def plane_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """The distance in the plane at the two points' mean latitude: exact enough for the
    tens of metres that candidate fusion works with."""
    east_r, north_r = tangent_radii((lat1 + lat2) / 2)
    east = math.radians(wrap_lon(lon2 - lon1)) * east_r
    north = math.radians(lat2 - lat1) * north_r
    return math.hypot(east, north)
