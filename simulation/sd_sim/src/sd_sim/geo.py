"""Local east/north metres around home to WGS84 and back.

A local tangent plane on the WGS84 ellipsoid: the radii of curvature at home's latitude
turn metres into degrees. Within the simulator's 1.2 km world the error is well under a
centimetre, which is far below GPS noise, so no projection library is needed.
"""

from __future__ import annotations

import math

WGS84_A = 6_378_137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


def _radii(lat_deg: float) -> tuple[float, float]:
    """Meridional (north) and prime-vertical (east) radius of curvature in metres."""
    s = math.sin(math.radians(lat_deg))
    w = math.sqrt(1 - WGS84_E2 * s * s)
    meridional = WGS84_A * (1 - WGS84_E2) / w**3
    prime_vertical = WGS84_A / w
    return meridional, prime_vertical


def enu_to_wgs84(
    home_lat: float, home_lon: float, east_m: float, north_m: float
) -> tuple[float, float]:
    """Latitude and longitude (degrees) of a point ``east_m``/``north_m`` from home."""
    m, n = _radii(home_lat)
    lat = home_lat + math.degrees(north_m / m)
    lon = home_lon + math.degrees(east_m / (n * math.cos(math.radians(home_lat))))
    return lat, lon


def wgs84_to_enu(home_lat: float, home_lon: float, lat: float, lon: float) -> tuple[float, float]:
    """East and north metres of ``lat``/``lon`` from home (inverse of ``enu_to_wgs84``)."""
    m, n = _radii(home_lat)
    north = math.radians(lat - home_lat) * m
    east = math.radians(lon - home_lon) * n * math.cos(math.radians(home_lat))
    return east, north
