"""Where an order would take the vehicle, and the exact geometry the gate checks it with.

Distances are geodesic on WGS84 (Vincenty's inverse formula, sub-millimetre at these
ranges). Containment is checked in a local east/north plane around the charter's home
point. Within the 2 km a charter can span the plane is off by at most about a metre at
high latitudes (centimetres near the test site), far inside the 20 m fence buffer.

Routes are not sampled: a straight leg lies inside a polygon exactly when every piece
between its crossings with the polygon's edges does, so concave areas cannot hide a leg
that leaves them. Every function here is pure and fails closed: a result that cannot be
worked out counts as outside or too far.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from sd_schemas.base import LatLon

__all__ = [
    "Plane",
    "Polygon",
    "Xy",
    "distance_m",
    "polygon_from_geojson",
]

Xy = tuple[float, float]  # east, north in metres

# WGS84
_A = 6_378_137.0
_F = 1 / 298.257223563
_B = _A * (1 - _F)
_E2 = _F * (2 - _F)

_EPS_M = 1e-6  # a point this close to an edge counts as on it


def distance_m(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    """Geodesic distance in metres on WGS84; ``inf`` if the formula does not converge."""
    if a_lat == b_lat and a_lon == b_lon:
        return 0.0
    big_l = math.radians(_wrap_lon(b_lon - a_lon))
    u1 = math.atan((1 - _F) * math.tan(math.radians(a_lat)))
    u2 = math.atan((1 - _F) * math.tan(math.radians(b_lat)))
    sin_u1, cos_u1 = math.sin(u1), math.cos(u1)
    sin_u2, cos_u2 = math.sin(u2), math.cos(u2)
    lam = big_l
    for _ in range(200):
        sin_lam, cos_lam = math.sin(lam), math.cos(lam)
        sin_sigma = math.hypot(cos_u2 * sin_lam, cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam)
        if sin_sigma == 0.0:
            return 0.0  # the same point written differently
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cos_u1 * cos_u2 * sin_lam / sin_sigma
        cos2_alpha = 1 - sin_alpha * sin_alpha
        cos_2sm = cos_sigma - 2 * sin_u1 * sin_u2 / cos2_alpha if cos2_alpha != 0.0 else 0.0
        c = _F / 16 * cos2_alpha * (4 + _F * (4 - 3 * cos2_alpha))
        lam_prev = lam
        lam = big_l + (1 - c) * _F * sin_alpha * (
            sigma + c * sin_sigma * (cos_2sm + c * cos_sigma * (-1 + 2 * cos_2sm * cos_2sm))
        )
        if abs(lam - lam_prev) < 1e-12:
            u_sq = cos2_alpha * (_A * _A - _B * _B) / (_B * _B)
            k_a = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
            k_b = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
            d_sigma = (
                k_b
                * sin_sigma
                * (
                    cos_2sm
                    + k_b
                    / 4
                    * (
                        cos_sigma * (-1 + 2 * cos_2sm * cos_2sm)
                        - k_b
                        / 6
                        * cos_2sm
                        * (-3 + 4 * sin_sigma * sin_sigma)
                        * (-3 + 4 * cos_2sm * cos_2sm)
                    )
                )
            )
            return _B * k_a * (sigma - d_sigma)
    return math.inf  # nearly antipodal points: never near a drone, so too far


def _wrap_lon(dlon: float) -> float:
    return (dlon + 180.0) % 360.0 - 180.0


@dataclass(frozen=True, slots=True)
class Plane:
    """A local east/north plane (metres) tangent to the ellipsoid at a reference point."""

    lat0: float
    lon0: float

    def xy(self, lat: float, lon: float) -> Xy:
        phi = math.radians(self.lat0)
        s = math.sin(phi)
        w = math.sqrt(1 - _E2 * s * s)
        n_radius = _A / w  # prime vertical
        m_radius = _A * (1 - _E2) / (w * w * w)  # meridian
        east = math.radians(_wrap_lon(lon - self.lon0)) * n_radius * math.cos(phi)
        north = math.radians(lat - self.lat0) * m_radius
        return east, north

    def point(self, p: LatLon) -> Xy:
        return self.xy(p.lat, p.lon)


@dataclass(frozen=True, slots=True)
class Polygon:
    """A closed ring in a :class:`Plane` (first corner not repeated)."""

    corners: tuple[Xy, ...]

    def edges(self) -> list[tuple[Xy, Xy]]:
        c = self.corners
        return [(c[i], c[(i + 1) % len(c)]) for i in range(len(c))]

    def is_simple(self) -> bool:
        """No edge crosses or touches another except its neighbours at the shared corner,
        and the ring encloses some area."""
        if abs(_signed_area(self.corners)) < 1e-6:
            return False
        edges = self.edges()
        n = len(edges)
        for i in range(n):
            for j in range(i + 1, n):
                adjacent = j == i + 1 or (i == 0 and j == n - 1)
                if adjacent:
                    if _overlap_beyond_corner(edges[i], edges[j]):
                        return False
                elif _segments_touch(*edges[i], *edges[j]):
                    return False
        return True

    def contains(self, p: Xy) -> bool:
        """Inside or on the boundary."""
        if self.boundary_distance(p) <= _EPS_M:
            return True
        x, y = p
        inside = False
        for (x1, y1), (x2, y2) in self.edges():
            if (y1 > y) != (y2 > y):
                x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_cross:
                    inside = not inside
        return inside

    def boundary_distance(self, p: Xy) -> float:
        return min(_point_segment_distance(p, a, b) for a, b in self.edges())

    def leg_outside_by(self, p: Xy, q: Xy) -> float:
        """How far the straight leg p->q leaves the polygon (0.0 when it stays inside). The
        distance is measured at the first piece found outside, so it is a lower bound."""
        ts = {0.0, 1.0}
        for a, b in self.edges():
            ts.update(_crossings(p, q, a, b))
        ordered = sorted(ts)
        probes = [*ordered, *((s + t) / 2 for s, t in pairwise(ordered))]
        for t in probes:
            point = (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
            if not self.contains(point):
                return self.boundary_distance(point)  # the first piece outside is enough
        return 0.0

    def outside_by(self, other: Polygon) -> float:
        """How far ``other`` reaches outside this polygon (0.0 when it lies inside); stops at
        the first edge found outside."""
        for a, b in other.edges():
            by = self.leg_outside_by(a, b)
            if by > 0.0:
                return by
        return 0.0


def polygon_from_geojson(polygon: dict[str, Any], plane: Plane) -> Polygon:
    """The schema has already checked the shape: one closed ring of [lon, lat] positions."""
    ring: Sequence[Sequence[float]] = polygon["coordinates"][0]
    return Polygon(tuple(plane.xy(lat, lon) for lon, lat in ring[:-1]))


# --- plane geometry ------------------------------------------------------------------------


def _cross(o: Xy, a: Xy, b: Xy) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _signed_area(corners: Sequence[Xy]) -> float:
    n = len(corners)
    total = 0.0
    for i in range(n):
        (x1, y1), (x2, y2) = corners[i], corners[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total / 2


def _point_segment_distance(p: Xy, a: Xy, b: Xy) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0.0 else ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2
    t = min(1.0, max(0.0, t))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def _segments_touch(a: Xy, b: Xy, c: Xy, d: Xy) -> bool:
    """Whether segments ab and cd share any point."""
    return (
        _point_segment_distance(a, c, d) <= _EPS_M
        or _point_segment_distance(b, c, d) <= _EPS_M
        or _point_segment_distance(c, a, b) <= _EPS_M
        or _point_segment_distance(d, a, b) <= _EPS_M
        or (
            (_cross(a, b, c) > 0) != (_cross(a, b, d) > 0)
            and (_cross(c, d, a) > 0) != (_cross(c, d, b) > 0)
        )
    )


def _overlap_beyond_corner(e1: tuple[Xy, Xy], e2: tuple[Xy, Xy]) -> bool:
    """Neighbouring edges share one corner; they must not fold back over each other."""
    (a, b), (c, d) = e1, e2
    shared = b if b == c else a  # e2 follows e1, or e1 is the last edge and e2 the first
    far1 = a if shared == b else b
    far2 = d if shared == c else c
    return (
        _point_segment_distance(far1, c, d) <= _EPS_M
        or _point_segment_distance(far2, a, b) <= _EPS_M
    )


def _crossings(p: Xy, q: Xy, a: Xy, b: Xy) -> list[float]:
    """Parameters t in [0, 1] where p + t(q - p) meets the edge ab."""
    d = (q[0] - p[0], q[1] - p[1])
    e = (b[0] - a[0], b[1] - a[1])
    denom = d[0] * e[1] - d[1] * e[0]
    ap = (a[0] - p[0], a[1] - p[1])
    dd = d[0] * d[0] + d[1] * d[1]
    if dd == 0.0:
        return []
    if abs(denom) > 1e-12 * dd:
        t = (ap[0] * e[1] - ap[1] * e[0]) / denom
        u = (ap[0] * d[1] - ap[1] * d[0]) / denom
        if 0.0 <= t <= 1.0 and -1e-9 <= u <= 1.0 + 1e-9:
            return [t]
        return []
    # parallel: the ends of a collinear edge split the leg
    out: list[float] = []
    for corner in (a, b):
        if _point_segment_distance(corner, p, q) <= _EPS_M:
            t = ((corner[0] - p[0]) * d[0] + (corner[1] - p[1]) * d[1]) / dd
            out.append(min(1.0, max(0.0, t)))
    return out
