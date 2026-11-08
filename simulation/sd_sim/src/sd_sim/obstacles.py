"""What the drone can hit in a simulated world: the ground, the trees and power lines.

The truth the closer look (``inspect_detection`` v2) is measured against. The same
scene gives the simulated laser its distance (``sd_sim.rangefinder``), the map pack its
tree heights (``sd_sim.floor_pack``) and a run its clearance (``sd_sim.inspect_metrics``),
so the laser, the map and the facit agree unless a test puts an error in on purpose.

Every solid is a vertical frustum (a cylinder when both radii are equal, a cone when one
is zero) or a wire (a capsule between two points). A tree is a few of them, as the world
draws it:

* ``lake``: ``sd_lake_forest``'s spruce, exactly as ``sd_lake_forest.sdf.jinja`` draws it
  (a trunk of radius ``world.TRUNK_RADIUS_M`` to the top and two crown cylinders); the
  stands a scenario adds (``TreeStand``) are drawn the same way;
* ``spruce``, ``pine``, ``birch``: a site's trees (``sd_sim.terrain``) as cones, the
  generated trees' own shapes; a real tree's mesh (``sd_sim.forest``) has about the same
  outline. Every crown reaches the tree's full height, so the map's top is the tree's.

A power line (``PowerLine``) is two poles and one straight wire between their tops (no
sag). Coordinates are the world's: metres east and north of home, z up from the ground
at home. Water and light ground are areas, for the laser (no return over water when
that fault is on; a TF02-Pro reaches further over light ground).

``Scene.save`` / ``Scene.load`` keep a scene in a folder (``scene.json`` and, on a site,
``ground.npy``), so a separate process (the laser) reads the scene the launcher built.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import Polygon

from sd_sim import objects, paths, world

__all__ = [
    "SHAPES",
    "Ground",
    "Hit",
    "PowerLine",
    "Scene",
    "SimTree",
    "TreeStand",
    "for_world",
    "lake_forest",
    "site_scene",
]

Shape = Literal["lake", "spruce", "pine", "birch"]
SHAPES: Final[tuple[Shape, ...]] = ("lake", "spruce", "pine", "birch")
Kind = Literal["ground", "tree", "line"]
F = NDArray[np.float64]
Xy = tuple[float, float]

BUCKET_M: Final = 20.0  # the spatial index's square
POLE_RADIUS_M: Final = 0.15
WIRE_RADIUS_M: Final = 0.02
LINE_HEIGHT_M: Final = (5.0, 40.0)  # a power line's allowed height above the ground
# The ray's step: no solid is thinner than this along a beam that tilts less than
# 60 degrees, so none is stepped over; the hit is then found to RAY_TOL_M.
RAY_STEP_M: Final = 0.05
RAY_TOL_M: Final = 0.002
RAY_CHUNK: Final = 512
SCENE_FORMAT: Final = 1


# --- the ground -----------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Ground:
    """The ground's height above home (m): flat at 0 without a grid, else bilinear on
    nodes every ``step`` m from ``-half`` to ``half``, row 0 northernmost (the site
    world's own mesh, ``sd_sim.terrain.Ground``). Outside the grid the edge holds."""

    step: float = 1.0
    half: float = 0.0
    z: F | None = None

    def at(self, east: Any, north: Any) -> F:
        e = np.asarray(east, dtype=np.float64)
        n_ = np.asarray(north, dtype=np.float64)
        if self.z is None:
            return np.zeros(np.broadcast(e, n_).shape)
        n = self.z.shape[0]
        col = np.clip((e + self.half) / self.step, 0, n - 1 - 1e-9)
        row = np.clip((self.half - n_) / self.step, 0, n - 1 - 1e-9)
        c0, r0 = np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)
        tc, tr = col - c0, row - r0
        z = self.z
        top = z[r0, c0] * (1 - tc) + z[r0, c0 + 1] * tc
        bottom = z[r0 + 1, c0] * (1 - tc) + z[r0 + 1, c0 + 1] * tc
        return top * (1 - tr) + bottom * tr

    def at1(self, east: float, north: float) -> float:
        return float(self.at(east, north))

    @property
    def max_slope(self) -> float:
        """The steepest rise per metre between neighbouring nodes (0 when flat)."""
        if self.z is None or self.z.shape[0] < 2:
            return 0.0
        dz = max(
            float(np.abs(np.diff(self.z, axis=0)).max()),
            float(np.abs(np.diff(self.z, axis=1)).max()),
        )
        return dz / self.step

    def span(self, west: float, south: float, east: float, north: float) -> tuple[float, float]:
        """The lowest and highest ground in a box (the nodes in and around it)."""
        if self.z is None:
            return 0.0, 0.0
        n = self.z.shape[0]
        c0 = int(np.clip(math.floor((west + self.half) / self.step), 0, n - 1))
        c1 = int(np.clip(math.ceil((east + self.half) / self.step), 0, n - 1))
        r0 = int(np.clip(math.floor((self.half - north) / self.step), 0, n - 1))
        r1 = int(np.clip(math.ceil((self.half - south) / self.step), 0, n - 1))
        patch = self.z[r0 : r1 + 1, c0 : c1 + 1]
        return float(patch.min()), float(patch.max())


# --- trees and lines ------------------------------------------------------------------


@dataclass(frozen=True)
class SimTree:
    """A tree standing on the ground at ``(east, north)`` (``base`` = the ground's height
    there). ``map_height`` is how tall the map pack says it is: ``None`` = as tall as it
    is, 0 = not in the map at all (a tree the map missed)."""

    east: float
    north: float
    height: float
    shape: Shape = "lake"
    base: float = 0.0
    map_height: float | None = None

    def parts(self, height: float | None = None) -> list[tuple[float, float, float, float]]:
        """The tree's frustums ``(z0, r0, z1, r1)`` above the world's zero, for ``height``
        (the tree's own by default; the map draws it at its map height)."""
        h = self.height if height is None else height
        b = self.base
        if h <= 0:
            return []
        if self.shape == "lake":
            trunk = world.TRUNK_RADIUS_M
            low, high = 0.15 * h, 0.08 * h
            return [
                (b, trunk, b + h, trunk),
                (b + 0.25 * h, low, b + 0.65 * h, low),
                (b + 0.625 * h, high, b + 0.975 * h, high),
            ]
        if self.shape == "spruce":
            return [(b, 0.02 * h, b + 0.25 * h, 0.02 * h), (b + 0.08 * h, 0.19 * h, b + h, 0.0)]
        if self.shape == "pine":
            r, zc = 0.2 * h, 0.8 * h
            return [
                (b, 0.022 * h, b + 0.7 * h, 0.022 * h),
                (b + zc - 0.12 * h, 0.0, b + zc, r),
                (b + zc, r, b + h, 0.0),
            ]
        r, zc = 0.21 * h, 0.64 * h  # birch
        return [
            (b, 0.02 * h, b + 0.55 * h, 0.02 * h),
            (b + zc - 0.192 * h, 0.0, b + zc, r),
            (b + zc, r, b + h, 0.0),
        ]


@dataclass(frozen=True)
class Wire:
    """A line's span between two pole tops, or a pole (``a`` straight under ``b``)."""

    a: tuple[float, float, float]
    b: tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class LineSpec:
    """A power line in the scene: its pole positions and the wire's height above the
    ground at each pole."""

    a: Xy
    b: Xy
    height: float
    name: str = ""


@dataclass(frozen=True)
class Hit:
    distance: float
    kind: Kind


# --- the scene ------------------------------------------------------------------------


@dataclass(eq=False)
class Scene:
    """A world's solids. ``origin`` is home (WGS84) and ``home_alt`` its ground height
    above sea level: world z = height above sea level - ``home_alt``. ``half`` is how far
    the world reaches from home."""

    origin: tuple[float, float]
    home_alt: float
    half: float
    ground: Ground = field(default_factory=Ground)
    trees: tuple[SimTree, ...] = ()
    lines: tuple[LineSpec, ...] = ()
    water: tuple[Any, ...] = ()  # shapely polygons in the world's plane
    light: tuple[Any, ...] = ()
    name: str = "scene"

    def __post_init__(self) -> None:
        self._index()

    # -- building ---------------------------------------------------------------------

    def with_trees(self, trees: Iterable[SimTree]) -> Scene:
        """The scene with more trees, each standing on the ground where it is."""
        placed = [replace(t, base=self.ground.at1(t.east, t.north)) for t in trees]
        return replace(self, trees=(*self.trees, *placed))

    def with_lines(self, lines: Iterable[LineSpec]) -> Scene:
        return replace(self, lines=(*self.lines, *lines))

    def with_areas(self, *, water: Iterable[Any] = (), light: Iterable[Any] = ()) -> Scene:
        return replace(self, water=(*self.water, *water), light=(*self.light, *light))

    def wires(self) -> list[Wire]:
        out = []
        for line in self.lines:
            za = self.ground.at1(*line.a)
            zb = self.ground.at1(*line.b)
            top_a, top_b = (*line.a, za + line.height), (*line.b, zb + line.height)
            out.append(Wire((*line.a, za), top_a, POLE_RADIUS_M))
            out.append(Wire((*line.b, zb), top_b, POLE_RADIUS_M))
            out.append(Wire(top_a, top_b, WIRE_RADIUS_M))
        return out

    def _index(self) -> None:
        rows = [p for t in self.trees for p in placed_parts(t, None)]
        arr = np.array(rows, dtype=np.float64).reshape(-1, 6)
        self._parts = arr  # x, y, z0, r0, z1, r1
        self._rmax = np.maximum(arr[:, 3], arr[:, 5]) if len(arr) else np.zeros(0)
        buckets: dict[tuple[int, int], list[int]] = {}
        for i, (x, y, r) in enumerate(zip(arr[:, 0], arr[:, 1], self._rmax, strict=True)):
            for key in _keys(x - r, y - r, x + r, y + r):
                buckets.setdefault(key, []).append(i)
        self._buckets = {k: np.array(v, dtype=np.int64) for k, v in buckets.items()}
        wires = self.wires()
        self._wires = np.array([(*w.a, *w.b, w.radius) for w in wires], dtype=np.float64).reshape(
            -1, 7
        )
        self._water = shapely.union_all(list(self.water)) if self.water else Polygon()
        self._light = shapely.union_all(list(self.light)) if self.light else Polygon()

    def _near(self, west: float, south: float, east: float, north: float) -> NDArray[np.int64]:
        found = [self._buckets[k] for k in _keys(west, south, east, north) if k in self._buckets]
        if not found:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(found))

    # -- queries ----------------------------------------------------------------------

    def is_water(self, east: float, north: float) -> bool:
        return bool(shapely.contains_xy(self._water, east, north))

    def is_light(self, east: float, north: float) -> bool:
        return bool(shapely.contains_xy(self._light, east, north))

    def top(self, east: float, north: float, radius: float = 0.0, *, lines: bool = True) -> float:
        """The highest surface (z) within ``radius`` of a point: the ground, a crown or
        a line."""
        if radius > 0:
            best = self.ground.span(east - radius, north - radius, east + radius, north + radius)[1]
        else:
            best = self.ground.at1(east, north)
        idx = self._near(east - radius, north - radius, east + radius, north + radius)
        if len(idx):
            p = self._parts[idx]
            d = np.maximum(np.hypot(p[:, 0] - east, p[:, 1] - north) - radius, 0.0)
            tops = frustum_top(d, p[:, 2], p[:, 3], p[:, 4], p[:, 5])
            if np.isfinite(tops).any():
                best = max(best, float(np.nanmax(tops)))
        if lines and len(self._wires):
            point = np.array([east, north])
            for ax, ay, az, bx, by, bz, r in self._wires:
                t = _closest_t(point, np.array([ax, ay]), np.array([bx, by]))
                q = np.array([ax + (bx - ax) * t, ay + (by - ay) * t])
                if float(np.hypot(*(q - point))) <= radius + r:
                    best = max(best, az + (bz - az) * t + r)
        return best

    def clearance(self, east: float, north: float, z: float) -> float:
        """The shortest distance from a point to anything solid; negative inside one.
        To the ground it is the height above it, divided by how steep the ground can
        be (never more than the true distance)."""
        g = (z - self.ground.at1(east, north)) / math.hypot(1.0, self.ground.max_slope)
        best = g
        reach = max(min(abs(best), 200.0), 1.0)
        idx = self._near(east - reach, north - reach, east + reach, north + reach)
        if len(idx):
            p = self._parts[idx]
            best = min(best, float(_frustum_sdf(east, north, z, p).min()))
        for w in self._wires:
            best = min(best, _capsule_distance(np.array([east, north, z]), w))
        return best

    def ray(
        self,
        start: Sequence[float],
        direction: Sequence[float],
        max_m: float,
        *,
        crowns: bool = True,
    ) -> Hit | None:
        """The first solid along a beam from ``start`` (world x, y, z) in ``direction``
        (a unit vector), within ``max_m``; ``crowns=False`` sees through the trees (a
        gap in the canopy) to what lies below."""
        p0 = np.asarray(start, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        d = d / float(np.linalg.norm(d))
        p1 = p0 + d * max_m
        west, east = float(min(p0[0], p1[0])), float(max(p0[0], p1[0]))
        south, north = float(min(p0[1], p1[1])), float(max(p0[1], p1[1]))
        idx = self._near(west, south, east, north) if crowns else np.zeros(0, dtype=np.int64)
        parts = self._parts[idx]
        zlo, zhi = self.ground.span(west, south, east, north)
        if len(parts):
            zhi = max(zhi, float(parts[:, 4].max()))
        if len(self._wires):
            zhi = max(zhi, float(np.maximum(self._wires[:, 2], self._wires[:, 5]).max()) + 1.0)
        t0, t1 = 0.0, max_m
        if d[2] < 0:  # pointing down: nothing above the highest top, nothing below the ground
            t0 = max(0.0, (p0[2] - zhi) / -d[2] - RAY_STEP_M)
            t1 = min(max_m, (p0[2] - zlo) / -d[2] + RAY_STEP_M)
        step = RAY_STEP_M
        if len(self._wires):  # a beam crossing a wire is inside it for about its width
            step = min(step, 2 * float(self._wires[:, 6].min()))
        prev: float | None = None
        t = t0
        while t <= t1:
            ts = np.minimum(t + step * np.arange(RAY_CHUNK), t1)
            kinds = self._solid(p0[None, :] + ts[:, None] * d[None, :], parts)
            hit = np.flatnonzero(kinds >= 0)
            if len(hit):
                k = int(hit[0])
                lo = float(ts[k - 1]) if k > 0 else prev
                if lo is None:  # the beam starts inside something
                    return Hit(0.0, _KINDS[int(kinds[k])])
                hi = float(ts[k])
                while hi - lo > RAY_TOL_M:
                    mid = (lo + hi) / 2
                    if self._solid((p0 + mid * d)[None, :], parts)[0] >= 0:
                        hi = mid
                    else:
                        lo = mid
                kind = int(self._solid((p0 + hi * d)[None, :], parts)[0])
                return Hit(hi, _KINDS[max(kind, 0)])
            prev = float(ts[-1])
            if ts[-1] >= t1:
                return None
            t = float(ts[-1]) + step
        return None

    def _solid(self, pts: F, parts: F) -> NDArray[np.int64]:
        """Per point: -1 in the air, else what it is inside (0 ground, 1 tree, 2 line)."""
        out = np.full(len(pts), -1, dtype=np.int64)
        if len(self._wires):
            for w in self._wires:
                inside = _segment_distance(pts, w[:3], w[3:6]) <= w[6]
                out[inside] = 2
        if len(parts):
            dx = pts[:, 0, None] - parts[None, :, 0]
            dy = pts[:, 1, None] - parts[None, :, 1]
            z = pts[:, 2, None]
            z0, r0, z1, r1 = (parts[None, :, i] for i in (2, 3, 4, 5))
            frac = np.clip((z - z0) / np.maximum(z1 - z0, 1e-9), 0.0, 1.0)
            r = r0 + (r1 - r0) * frac
            inside = ((dx * dx + dy * dy) <= r * r) & (z >= z0) & (z <= z1)
            out[inside.any(axis=1) & (out < 0)] = 1
        ground = pts[:, 2] <= self.ground.at(pts[:, 0], pts[:, 1])
        out[ground & (out < 0)] = 0
        return out

    # -- files ------------------------------------------------------------------------

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "format": SCENE_FORMAT,
            "name": self.name,
            "origin": list(self.origin),
            "home_alt": self.home_alt,
            "half": self.half,
            "ground": None,
            "trees": [
                [t.east, t.north, t.height, t.shape, t.base, t.map_height] for t in self.trees
            ],
            "lines": [[*ln.a, *ln.b, ln.height, ln.name] for ln in self.lines],
            "water": [shapely.to_geojson(g) for g in self.water],
            "light": [shapely.to_geojson(g) for g in self.light],
        }
        if self.ground.z is not None:
            np.save(folder / "ground.npy", self.ground.z, allow_pickle=False)
            data["ground"] = {"step": self.ground.step, "half": self.ground.half}
        path = folder / "scene.json"
        path.write_text(json.dumps(data) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, folder: Path) -> Scene:
        data = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
        if data.get("format") != SCENE_FORMAT:
            raise ValueError(f"{folder}: scene.json har inte format {SCENE_FORMAT}")
        ground = Ground()
        if data["ground"] is not None:
            z = np.load(folder / "ground.npy", allow_pickle=False).astype(np.float64)
            ground = Ground(float(data["ground"]["step"]), float(data["ground"]["half"]), z)
        return cls(
            origin=(float(data["origin"][0]), float(data["origin"][1])),
            home_alt=float(data["home_alt"]),
            half=float(data["half"]),
            ground=ground,
            trees=tuple(
                SimTree(
                    float(e),
                    float(n),
                    float(h),
                    _shape(s),
                    float(b),
                    None if m is None else float(m),
                )
                for e, n, h, s, b, m in data["trees"]
            ),
            lines=tuple(
                LineSpec((float(ax), float(ay)), (float(bx), float(by)), float(h), str(nm))
                for ax, ay, bx, by, h, nm in data["lines"]
            ),
            water=tuple(shapely.from_geojson(g) for g in data["water"]),
            light=tuple(shapely.from_geojson(g) for g in data["light"]),
            name=str(data["name"]),
        )


_KINDS: Final[tuple[Kind, ...]] = ("ground", "tree", "line")


def _keys(west: float, south: float, east: float, north: float) -> list[tuple[int, int]]:
    i0, i1 = math.floor(west / BUCKET_M), math.floor(east / BUCKET_M)
    j0, j1 = math.floor(south / BUCKET_M), math.floor(north / BUCKET_M)
    return [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]


def placed_parts(
    tree: SimTree, height: float | None
) -> list[tuple[float, float, float, float, float, float]]:
    return [(tree.east, tree.north, *part) for part in tree.parts(height)]


def frustum_top(d: F, z0: F | float, r0: F | float, z1: F | float, r1: F | float) -> F:
    """The highest point of each frustum at horizontal distance ``d`` from its axis
    (NaN where it does not reach)."""
    flat = r1 >= r0  # widest at the top (a cylinder or an upturned cone)
    slope = np.where(r0 > r1, (z1 - z0) / np.maximum(r0 - r1, 1e-9), 0.0)
    top = np.where(d <= r1, z1, z1 - slope * (d - r1))
    reach = np.where(flat, r1, r0)
    return np.where(d <= reach, top, np.nan)


def _frustum_sdf(x: float, y: float, z: float, p: F) -> F:
    """Signed distance from a point to each vertical frustum (Inigo Quilez's capped
    cone, exact)."""
    d = np.hypot(p[:, 0] - x, p[:, 1] - y)
    z0, ra, z1, rb = p[:, 2], p[:, 3], p[:, 4], p[:, 5]
    hh = np.maximum(z1 - z0, 1e-9)
    baba = hh * hh
    paba = (z - z0) / hh
    rba = rb - ra
    cax = np.maximum(0.0, d - np.where(paba < 0.5, ra, rb))
    cay = np.abs(paba - 0.5) - 0.5
    k = rba * rba + baba
    f = np.clip((rba * (d - ra) + paba * baba) / k, 0.0, 1.0)
    cbx = d - ra - f * rba
    cby = paba - f
    s = np.where((cbx < 0.0) & (cay < 0.0), -1.0, 1.0)
    return s * np.sqrt(np.minimum(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba))


def _closest_t(p: F, a: F, b: F) -> float:
    ab = b - a
    denom = float(ab @ ab)
    return 0.0 if denom == 0 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))


def _capsule_distance(p: F, w: F) -> float:
    a, b = w[:3], w[3:6]
    t = _closest_t(p, a, b)
    return float(np.linalg.norm(p - (a + t * (b - a)))) - float(w[6])


def _segment_distance(pts: F, a: F, b: F) -> F:
    ab = b - a
    denom = float(ab @ ab)
    t = np.zeros(len(pts)) if denom == 0 else np.clip((pts - a) @ ab / denom, 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


# --- world objects ----------------------------------------------------------------------


def _num(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class PowerLine:
    """A world object (``sd_sim.objects.WorldObject``): two poles at ``a`` and ``b``
    (metres east/north of home) and a wire between their tops, ``height_m`` above the
    ground. It stands in Gazebo's world, the laser sees it, and the map pack lists it in
    ``lines.geojson`` (``LineSpec``)."""

    name: str
    a: Xy
    b: Xy
    height_m: float = 20.0
    label: int = 0
    up_m: float = 0.0
    yaw_deg: float = 0.0
    truth: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        low, high = LINE_HEIGHT_M
        if not low <= self.height_m <= high:
            raise objects.ObjectError(
                f"{self.name}: height_m ska ligga mellan {low:g} och {high:g}"
            )
        if math.dist(self.a, self.b) < 1.0:
            raise objects.ObjectError(f"{self.name}: stolparna ska stå minst 1 m isär")

    @property
    def east_m(self) -> float:
        return (self.a[0] + self.b[0]) / 2

    @property
    def north_m(self) -> float:
        return (self.a[1] + self.b[1]) / 2

    @property
    def footprint_m(self) -> float:
        return math.dist(self.a, self.b) / 2 + POLE_RADIUS_M

    def spec(self) -> LineSpec:
        return LineSpec(self.a, self.b, self.height_m, self.name)

    def sdf(self, pose: objects.Pose) -> str:
        """Poles and wire relative to the midpoint on the ground (the pose's yaw is not
        used: the line runs from ``a`` to ``b``)."""
        h = self.height_m
        dx, dy = self.b[0] - self.a[0], self.b[1] - self.a[1]
        span = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)
        hx, hy = dx / 2, dy / 2

        def pole(name: str, x: float, y: float) -> str:
            geom = (
                f"<cylinder><radius>{POLE_RADIUS_M}</radius><length>{_num(h)}</length></cylinder>"
            )
            return f"""    <link name="{name}">
      <pose>{_num(x)} {_num(y)} {_num(h / 2)} 0 0 0</pose>
      <collision name="collision"><geometry>{geom}</geometry></collision>
      <visual name="visual"><geometry>{geom}</geometry>
        <material><ambient>0.35 0.27 0.18 1</ambient><diffuse>0.35 0.27 0.18 1</diffuse></material>
      </visual>
    </link>"""

        wire = f"<cylinder><radius>{WIRE_RADIUS_M}</radius><length>{_num(span)}</length></cylinder>"
        return f"""<model name="{self.name}">
    <static>true</static>
    <pose>{pose.x:.3f} {pose.y:.3f} {pose.z:.3f} 0 0 0</pose>
{pole("pole_a", -hx, -hy)}
{pole("pole_b", hx, hy)}
    <link name="wire">
      <pose>0 0 {_num(h)} 0 {_num(math.pi / 2)} {yaw:.6f}</pose>
      <visual name="visual"><geometry>{wire}</geometry>
        <material><ambient>0.15 0.15 0.15 1</ambient><diffuse>0.15 0.15 0.15 1</diffuse></material>
      </visual>
    </link>
  </model>"""


@dataclass(frozen=True)
class TreeStand:
    """A world object: a stand of ``sd_lake_forest``'s spruces (``lake`` shape) in a
    circle of ``radius_m`` around (east, north), about ``spacing_m`` apart, each
    ``height_m`` tall give or take ``jitter_m`` (fixed by ``seed``). For a scenario that
    needs forest where the flat test world has meadow ("granskog 24 m"). ``map_short_m``:
    the map pack has every tree of the stand that much too short (a tree grown since the
    scan, or one the map missed when it is as tall as the tree)."""

    name: str
    east_m: float
    north_m: float
    radius_m: float
    height_m: float
    spacing_m: float = 6.0
    jitter_m: float = 0.0
    seed: int = 0
    map_short_m: float = 0.0
    label: int = 0
    up_m: float = 0.0
    yaw_deg: float = 0.0
    truth: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.radius_m <= 200 or not 1 <= self.height_m <= 40:
            raise objects.ObjectError(f"{self.name}: radius_m 0-200 och height_m 1-40")
        if not 2 <= self.spacing_m <= 50 or not 0 <= self.jitter_m < self.height_m:
            raise objects.ObjectError(f"{self.name}: spacing_m 2-50, jitter_m under height_m")
        if not 0 <= self.map_short_m <= 40:
            raise objects.ObjectError(f"{self.name}: map_short_m 0-40")

    @property
    def footprint_m(self) -> float:
        return self.radius_m + 0.15 * (self.height_m + self.jitter_m)

    def trees(self) -> list[SimTree]:
        """The stand's trees: a jittered square grid, cut to the circle."""
        rng = random.Random(self.seed)  # noqa: S311 - scenery, not security
        out = []
        n = math.ceil(self.radius_m / self.spacing_m)
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                dx = (i + rng.uniform(-0.3, 0.3)) * self.spacing_m
                dy = (j + rng.uniform(-0.3, 0.3)) * self.spacing_m
                h = self.height_m + rng.uniform(-self.jitter_m, self.jitter_m)
                if math.hypot(dx, dy) <= self.radius_m:
                    short = None if self.map_short_m == 0 else max(h - self.map_short_m, 0.0)
                    out.append(
                        SimTree(
                            round(self.east_m + dx, 2),
                            round(self.north_m + dy, 2),
                            round(h, 2),
                            map_height=None if short is None else round(short, 2),
                        )
                    )
        return out

    def sdf(self, pose: objects.Pose) -> str:
        """The same three cylinders per tree as ``sd_lake_forest.sdf.jinja``."""
        links = []
        for k, t in enumerate(self.trees()):
            x, y, h = t.east - self.east_m, t.north - self.north_m, t.height
            cyl = [
                ("trunk", world.TRUNK_RADIUS_M, h / 2, h, "0.35 0.25 0.15"),
                ("crown_low", 0.15 * h, 0.45 * h, 0.4 * h, "0.10 0.30 0.15"),
                ("crown_high", 0.08 * h, 0.8 * h, 0.35 * h, "0.10 0.35 0.15"),
            ]
            visuals = "".join(
                f"""
      <visual name="{nm}"><pose>0 0 {_num(zc)} 0 0 0</pose>
        <geometry><cylinder><radius>{_num(r)}</radius><length>{_num(ln)}</length></cylinder></geometry>
        <material><ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse></material>
      </visual>"""
                for nm, r, zc, ln, rgb in cyl
            )
            links.append(
                f"""    <link name="t{k:03d}">
      <pose>{_num(x)} {_num(y)} 0 0 0 0</pose>
      <collision name="trunk_collision"><pose>0 0 {_num(h / 2)} 0 0 0</pose>
        <geometry><cylinder><radius>{world.TRUNK_RADIUS_M}</radius><length>{_num(h)}</length></cylinder></geometry>
      </collision>{visuals}
    </link>"""
            )
        body = "\n".join(links)
        return f"""<model name="{self.name}">
    <static>true</static>
    <pose>{pose.x:.3f} {pose.y:.3f} {pose.z:.3f} 0 0 0</pose>
{body}
  </model>"""


# --- builders ---------------------------------------------------------------------------


def _from_objects(scene: Scene, placed: Sequence[Any]) -> Scene:
    """Add the obstacles among a run's world objects: stands and power lines. Other
    objects (people, boxes) are not obstacles."""
    trees = [t for obj in placed if isinstance(obj, TreeStand) for t in obj.trees()]
    lines = [obj.spec() for obj in placed if isinstance(obj, PowerLine)]
    return scene.with_trees(trees).with_lines(lines)


def lake_forest(origin: tuple[float, float], home_alt: float, placed: Sequence[Any] = ()) -> Scene:
    """``sd_lake_forest``: flat ground, its spruce band, the lake, and the run's objects."""
    trees = tuple(SimTree(t.x, t.y, t.height) for t in world.trees())
    scene = Scene(
        origin=origin,
        home_alt=home_alt,
        half=world.GROUND_SIZE_M / 2,
        trees=trees,
        water=(Polygon(world.lake_polygon()),),
        name=world.WORLD_NAME,
    )
    return _from_objects(scene, placed)


def site_scene(name: str, placed: Sequence[Any] = (), repo: Path = paths.REPO) -> Scene:
    """A site's world (``sd_sim.site``): its terrain, its trees exactly where the world
    puts them (the real ones after ``just trees-fetch``, else the generated forest), its
    water, and the run's objects."""
    from sd_sim import site as site_mod
    from sd_sim import terrain

    s = site_mod.load(name, repo)
    g = terrain.ground_grid(s)
    trees = tuple(
        SimTree(t.east, t.north, t.height, _site_shape(t.species), t.z)
        for t in terrain.site_trees(s, g)
    )
    scene = Scene(
        origin=(s.lat, s.lon),
        home_alt=s.home_alt,
        half=s.half_size_m,
        ground=Ground(g.step, g.half, g.z.astype(np.float64)),
        trees=trees,
        water=tuple(layer.geom for layer in s.of_kind("water")),
        name=s.world,
    )
    return _from_objects(scene, placed)


def _site_shape(species: str) -> Shape:
    """A site tree's shape: its species, any broadleaf drawn as the birch."""
    if species == "spruce":
        return "spruce"
    return "pine" if species == "pine" else "birch"


def _shape(value: Any) -> Shape:
    for shape in SHAPES:
        if value == shape:
            return shape
    raise ValueError(f"okänd trädform {value!r}")


def for_world(
    world_name: str,
    home: str,
    placed: Sequence[Any] = (),
    repo: Path = paths.REPO,
) -> Scene:
    """The scene of a run's world. ``home`` is SITL's ``lat,lon,alt,heading``: the
    flat world's home is wherever the run puts it."""
    from sd_sim import site as site_mod

    sites = site_mod.world_sites(repo)
    if world_name in sites:
        return site_scene(sites[world_name], placed, repo)
    if world_name != world.WORLD_NAME:
        raise ValueError(f"okänd värld {world_name!r}")
    lat, lon, alt = (float(v) for v in home.split(",")[:3])
    return lake_forest((lat, lon), alt, placed)
