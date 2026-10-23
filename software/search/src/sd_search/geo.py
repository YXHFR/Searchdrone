"""Distances between two WGS84 points in the local tangent plane (metres)."""

from __future__ import annotations

import math

from sd_schemas.geo import tangent_radii, wrap_lon

__all__ = ["distance_m"]


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """The plane distance; exact enough for the tens of metres fusion works with."""
    east_r, north_r = tangent_radii((lat1 + lat2) / 2)
    east = math.radians(wrap_lon(lon2 - lon1)) * east_r
    north = math.radians(lat2 - lat1) * north_r
    return math.hypot(east, north)
