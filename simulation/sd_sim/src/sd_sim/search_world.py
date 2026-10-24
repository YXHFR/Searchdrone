"""A search scenario's world: who and what lies where, for one seed (``SKxx_*.yaml``).

A scenario file (``simulation/scenarios/SKxx_<name>.yaml``) is also the scripted
commander's fixture: ``home``, ``pins``, ``areas`` and ``charters`` in the demo fixture's
shape (metres east/north of home), so a script can name ``fixture:<area>`` and
``charter_proposal: {from_fixture: <charter>}``. The ``search`` block is the simulator's
own and never leaves it:

* ``profile``: the search profile as the operator approves it (the scripted operator
  posts it; the AI never does).
* ``people``: the persons in the world, the missing person (``target: true``) among
  them, with their clothing colours (``sd_schemas.colors`` names), ``pose`` (lying,
  sitting or standing), a heading, an occlusion share and ``jitter_m`` (each seed moves
  them up to that far, and turns them).
* ``decoys``: things that are not persons (rocks, backpacks); ``count`` of them at
  random in ``area`` (an area name) or one ``at_enu``. ``lure`` is the chance that the
  detector takes one for a person in a still it is in.
* ``limit_s``: the time the nightly requirement allows to the confirmed find.
* ``trail``: a path through the area (``[[east, north], ...]``), the map's terrain; a
  person with ``near_trail_m: [lo, hi]`` lies that far from a random point of it.
* ``sector_order``: the order the sectors are searched in without the map.

``load(scenario, seed)`` gives a ``World``: the same seed, the same world. Positions are
kept inside the charter's area. Nothing here touches the flight stack.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from shapely.geometry import LineString, Point, Polygon

from sd_schemas.colors import COLOR_TABLE
from sd_schemas.geo import plane_offset
from sd_sim import paths

__all__ = [
    "SCENARIOS",
    "Thing",
    "World",
    "WorldError",
    "load",
    "scenario_path",
    "seed_offset",
    "seeds",
]

SCENARIOS = paths.SIMULATION / "scenarios"
MAX_SEEDS = 200
Pose = Literal["lying", "sitting", "standing"]
POSES: tuple[Pose, ...] = ("lying", "sitting", "standing")
DECOY_KINDS = ("rock", "backpack", "log", "tarp")
_TOP = {"scenario", "title", "home", "pilot", "pins", "areas", "charters", "search"}
_SEARCH = {"limit_s", "survey", "profile", "last_seen_enu", "people", "decoys", "trail",
           "sector_order"}  # fmt: skip
_PERSON = {"id", "target", "at_enu", "jitter_m", "pose", "upper", "lower", "backpack",
           "heading_deg", "occlusion_pct", "area", "near_trail_m"}  # fmt: skip
_DECOY = {"kind", "at_enu", "count", "area", "color", "lure", "size_m"}


class WorldError(ValueError):
    """The scenario file cannot be used."""


@dataclass(frozen=True)
class Thing:
    """One object in the world. ``person`` and ``target`` are the truth the oracles and
    the scripted operator go by; the flight stack never sees them."""

    thing_id: str
    kind: str  # "person" or a decoy kind
    east: float
    north: float
    heading_deg: float = 0.0
    person: bool = False
    target: bool = False
    pose: Pose = "lying"
    upper: str | None = None
    lower: str | None = None
    backpack: str | None = None
    color: str | None = None  # a decoy's colour
    size_m: float = 0.8  # a decoy's longest side
    occlusion_pct: float = 0.0
    lure: float = 0.0  # a decoy: the chance a still takes it for a person

    @property
    def length_m(self) -> float:
        """The longest side seen from above."""
        if not self.person:
            return self.size_m
        return {"lying": 1.7, "sitting": 0.9, "standing": 0.5}[self.pose]


@dataclass(frozen=True)
class World:
    scenario: str
    seed: int
    home: tuple[float, float, float]  # lat, lon, alt AMSL
    things: tuple[Thing, ...]
    limit_s: float
    survey: Mapping[str, float]
    profile: Mapping[str, Any]
    last_seen_enu: tuple[float, float]
    areas: Mapping[str, tuple[tuple[float, float], ...]]
    trail: tuple[tuple[float, float], ...] = ()
    sector_order: tuple[str, ...] = ()
    fixture_path: Path = field(default=Path())

    @property
    def target(self) -> Thing | None:
        return next((t for t in self.things if t.target), None)

    @property
    def people(self) -> list[Thing]:
        return [t for t in self.things if t.person]

    def latlon(self, east: float, north: float) -> tuple[float, float]:
        return plane_offset(self.home[0], self.home[1], east, north)

    def where(self, thing: Thing) -> tuple[float, float]:
        return self.latlon(thing.east, thing.north)

    def nearest(self, east: float, north: float, within_m: float) -> Thing | None:
        """The person or decoy nearest to a point, if one is within ``within_m``."""
        best = min(self.things, key=lambda t: math.hypot(t.east - east, t.north - north),
                   default=None)  # fmt: skip
        if best is None or math.hypot(best.east - east, best.north - north) > within_m:
            return None
        return best

    def terrain(self) -> list[dict[str, Any]]:
        """The trail as the map's terrain layer (GeoJSON features, WGS84 lon/lat)."""
        if not self.trail:
            return []
        coords = [list(reversed(self.latlon(e, n))) for e, n in self.trail]
        return [
            {
                "type": "Feature",
                "properties": {"kind": "path"},
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        ]

    def profile_draft(self, at_utc: str) -> dict[str, Any]:
        """The approved profile's body (``SearchProfileDraft``), last seen at the start."""
        lat, lon = self.latlon(*self.last_seen_enu)
        return {**self.profile, "last_seen": {"lat": lat, "lon": lon, "at_utc": at_utc}}


def seed_offset(seed: int, radius_m: float, salt: str = "") -> tuple[float, float]:
    """A seeded east/north offset within ``radius_m`` (the same seed, the same offset):
    for the scenarios that move a fixed place with the seed."""
    rng = random.Random(f"offset:{salt}:{seed}")  # noqa: S311 - a simulation
    r, a = radius_m * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
    return r * math.cos(a), r * math.sin(a)


def scenario_path(scenario: str) -> Path:
    """``SK01`` -> ``simulation/scenarios/SK01_<name>.yaml`` (exactly one)."""
    found = sorted(SCENARIOS.glob(f"{scenario}_*.yaml"))
    if len(found) != 1:
        raise WorldError(f"scenariot {scenario}: {len(found)} filer i {SCENARIOS}")
    return found[0]


def seeds(spec: str) -> list[int]:
    """``"1-50"`` or ``"1,2,3"`` -> the seeds (positive, at most ``MAX_SEEDS``)."""
    out: list[int] = []
    try:
        for part in spec.replace(" ", "").split(","):
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                out += range(lo, hi + 1)
            else:
                out.append(int(part))
    except ValueError:
        raise WorldError(f"frön ska skrivas 1-50 eller 1,2,3, inte {spec!r}") from None
    if not out or min(out) < 1 or len(out) > MAX_SEEDS or len(set(out)) != len(out):
        raise WorldError(f"frön: 1-{MAX_SEEDS} olika positiva heltal, inte {spec!r}")
    return out


def _ring(raw: Any, what: str) -> tuple[tuple[float, float], ...]:
    try:
        ring = tuple((float(e), float(n)) for e, n in raw)
    except (TypeError, ValueError):
        raise WorldError(f"{what}: area_enu ska vara [[öst, norr], ...]") from None
    if len(ring) < 3:
        raise WorldError(f"{what}: minst tre hörn")
    return ring


def _line(raw: Any, what: str) -> tuple[tuple[float, float], ...]:
    try:
        line = tuple((float(e), float(n)) for e, n in raw)
    except (TypeError, ValueError):
        raise WorldError(f"{what}: ska vara [[öst, norr], ...]") from None
    if len(line) < 2:
        raise WorldError(f"{what}: minst två punkter")
    return line


def _keys(data: Mapping[str, Any], allowed: set[str], what: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise WorldError(f"{what}: okända nycklar {sorted(unknown)}")


def _color(value: Any, what: str) -> str | None:
    if value is None:
        return None
    if value not in COLOR_TABLE:
        raise WorldError(f"{what}: okänd färg {value!r}")
    return str(value)


def _inside(poly: Polygon, rng: random.Random) -> tuple[float, float]:
    xmin, ymin, xmax, ymax = poly.bounds
    for _ in range(1000):
        p = (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))
        if poly.contains(Point(p)):
            return p
    raise WorldError("hittar ingen punkt i ytan")


def _jittered(
    at: Sequence[float], jitter_m: float, keep: Polygon, rng: random.Random
) -> tuple[float, float]:
    for _ in range(1000):
        r, a = jitter_m * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
        p = (float(at[0]) + r * math.cos(a), float(at[1]) + r * math.sin(a))
        if keep.contains(Point(p)):
            return p
    raise WorldError(f"{list(at)} ± {jitter_m} m hamnar utanför ytan")


def _near_trail(
    trail: LineString, span: Sequence[float], keep: Polygon, rng: random.Random
) -> tuple[float, float]:
    lo, hi = (float(x) for x in span)
    for _ in range(1000):
        d = rng.uniform(0, trail.length)
        a, b = trail.interpolate(d), trail.interpolate(min(trail.length, d + 1.0))
        if a.equals(b):
            a, b = trail.interpolate(d - 1.0), a
        ux, uy = b.x - a.x, b.y - a.y
        norm = math.hypot(ux, uy)
        off = rng.uniform(lo, hi) * rng.choice((-1.0, 1.0))
        p = (a.x - uy / norm * off, a.y + ux / norm * off)
        if keep.contains(Point(p)) and lo - 0.5 <= trail.distance(Point(p)) <= hi + 0.5:
            return p
    raise WorldError(f"ingen punkt {lo}-{hi} m från stigen i ytan")


def load(scenario: str | Path, seed: int) -> World:
    path = Path(scenario) if Path(scenario).suffix == ".yaml" else scenario_path(str(scenario))
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WorldError(f"{path.name}: {exc}") from None
    if not isinstance(data, dict):
        raise WorldError(f"{path.name}: ska vara en mappning")
    _keys(data, _TOP, path.name)
    search = data.get("search") or {}
    _keys(search, _SEARCH, f"{path.name}: search")
    home = data["home"]
    areas = {
        name: _ring(a["area_enu"], f"{path.name}: {name}")
        for name, a in (data.get("areas") or {}).items()
    }
    charters = data.get("charters") or {}
    if len(charters) < 1:
        raise WorldError(f"{path.name}: minst en ram")
    # Everything stays inside the first charter's area, a few metres from its edge.
    keep = Polygon(_ring(next(iter(charters.values()))["area_enu"], path.name)).buffer(-5.0)

    def area(name: str) -> Polygon:
        if name not in areas:
            raise WorldError(f"{path.name}: okänd yta {name!r}")
        inside = Polygon(areas[name]).intersection(keep)
        if not isinstance(inside, Polygon) or inside.is_empty:
            raise WorldError(f"{path.name}: ytan {name!r} ligger inte i ramen")
        return inside

    # One generator per seed and scenario: the same seed gives the same world.
    rng = random.Random(f"{data.get('scenario', path.stem)}:{seed}")  # noqa: S311 - a simulation
    trail = _line(search["trail"], f"{path.name}: trail") if "trail" in search else ()
    things: list[Thing] = []
    for i, p in enumerate(search.get("people") or []):
        what = f"{path.name}: people[{i}]"
        _keys(p, _PERSON, what)
        pose = p.get("pose", "lying")
        if pose not in POSES:
            raise WorldError(f"{what}: pose ska vara en av {POSES}")
        if "near_trail_m" in p:  # within ``area`` when it has one
            if not trail:
                raise WorldError(f"{what}: near_trail_m utan trail")
            inside = area(p["area"]) if "area" in p else keep
            east, north = _near_trail(LineString(trail), p["near_trail_m"], inside, rng)
        elif "area" in p:
            east, north = _inside(area(p["area"]), rng)
        else:
            east, north = _jittered(p["at_enu"], float(p.get("jitter_m", 0.0)), keep, rng)
        things.append(
            Thing(
                thing_id=str(p.get("id", f"person-{i}")),
                kind="person",
                east=east,
                north=north,
                heading_deg=float(p.get("heading_deg", rng.uniform(0, 360))),
                person=True,
                target=bool(p.get("target", False)),
                pose=pose,
                upper=_color(p.get("upper"), what),
                lower=_color(p.get("lower"), what),
                backpack=_color(p.get("backpack"), what),
                occlusion_pct=float(p.get("occlusion_pct", 0.0)),
            )
        )
    for i, d in enumerate(search.get("decoys") or []):
        what = f"{path.name}: decoys[{i}]"
        _keys(d, _DECOY, what)
        kind = d.get("kind")
        if kind not in DECOY_KINDS:
            raise WorldError(f"{what}: kind ska vara en av {DECOY_KINDS}")
        for j in range(int(d.get("count", 1))):
            if "area" in d:
                east, north = _inside(area(d["area"]), rng)
            else:
                east, north = (float(x) for x in d["at_enu"])
            things.append(
                Thing(
                    thing_id=f"{kind}-{i}-{j}",
                    kind=kind,
                    east=east,
                    north=north,
                    heading_deg=rng.uniform(0, 360),
                    color=_color(d.get("color", "gray"), what),
                    size_m=float(d.get("size_m", 0.8)) * rng.uniform(0.7, 1.3),
                    lure=float(d.get("lure", 0.03)),
                )
            )
    if sum(t.target for t in things) > 1:
        raise WorldError(f"{path.name}: högst en saknad person")
    last_seen = search.get("last_seen_enu", [0.0, 0.0])
    return World(
        scenario=str(data.get("scenario", path.stem)),
        seed=seed,
        home=(float(home["lat"]), float(home["lon"]), float(home["alt_amsl_m"])),
        things=tuple(things),
        limit_s=float(search.get("limit_s", 900.0)),
        survey={k: float(v) for k, v in (search.get("survey") or {}).items()},
        profile=dict(search.get("profile") or {}),
        last_seen_enu=(float(last_seen[0]), float(last_seen[1])),
        areas=areas,
        trail=trail,
        sector_order=tuple(str(x) for x in search.get("sector_order") or ()),
        fixture_path=path,
    )
