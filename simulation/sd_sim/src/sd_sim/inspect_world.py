"""The closer look's scenarios (spec "Titta närmare" §16.4) and the camera oracle they fly.

A scenario file is ``simulation/scenarios/TNxx_<name>.yaml``: where the person is, what
stands around it (tree stands, power lines, light ground), what the map pack has wrong,
which laser the drone carries and the laser's faults, the charter's mode and heights, and
the seeds. Everything is east/north of home, so no file names a real place::

    scenario: TN02
    title: "Granskog 24 m, ingen laser"
    mode: exercise            # sim | exercise | live (the charter's operation mode)
    overflight: true          # the charter's overflight_allowed (live without a permit: false)
    laser: none               # none | tf02 | grf250
    start_alt_m: 60           # the look starts from here (above home)
    charter: { alt_min_agl_m: 20, alt_max_agl_m: 120 }
    target: { at_enu: [40, 0], jitter_m: 5, visible_m: 0.4, score: 0.45 }
    estimate: { error_m: 3, sigma95_m: 5 }   # the candidate the ground knows
    others: [{ id: other, at_enu: [65, 0], visible_m: 0.4, score: 0.9 }]
    stands: [{ name: granar, at_enu: [40, 0], radius_m: 30, height_m: 24, spacing_m: 6 }]
    lines: [{ name: ledning, a: [20, -60], b: [20, 60], height_m: 15 }]
    light: [[[-10, -10], [80, -10], [80, 10], [-10, 10]]]
    map: { scan_year: 2021, tree_bias_m: 0, missing_trees: 0, holes: [[40, 0, 6]] }
    faults: { rf_spikes: 0.0 }
    seeds: [1, 2, 3]

``InspectWorld.load(name, seed)`` places the people with the seed (``jitter_m``), and
``Oracle`` is the cameras' and the detector's stand-in: from the drone's true position it
says, per camera, whether a person is in the picture, how big (in the sensor's full pixels,
the scale ``CLEAR_PX`` is in), how sure the detector is, and where the georeferenced
detection lands (with the seed's noise, growing with the distance). ``visible_m`` is the
person's visible extent: a lying adult in the open is about 1.7 m, one in tall grass or
under branches much less. A fixed ``score`` below ``CLEAR_SCORE`` gives a person the
detector sees but is never sure of, so the look comes down to its floor.

Only the simulator uses this; nothing here flies.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import yaml
from shapely.geometry import Polygon

from sd_schemas.charter import OperationMode
from sd_sim import cameras, objects, paths
from sd_sim.floor_pack import MapErrors
from sd_sim.obstacles import PowerLine, TreeStand
from sd_sim.search_world import seed_offset

__all__ = [
    "CAMERAS",
    "DETECT_MIN_PX",
    "SCENARIOS",
    "Camera",
    "InspectWorld",
    "Oracle",
    "Person",
    "Sighting",
    "WorldError",
    "load",
    "scenario_path",
]

SCENARIOS = paths.SIMULATION / "scenarios"
Laser = Literal["none", "tf02", "grf250"]
LASERS: Final[tuple[Laser, ...]] = ("none", "tf02", "grf250")
MODES: Final[tuple[OperationMode, ...]] = ("sim", "exercise", "live")
DETECT_MIN_PX: Final = 32.0  # the detector's smallest person (sd_detect)
NOISE_PER_M: Final = 0.02  # a detection's position error, 1 sigma, per metre of range
SIGMA95_MIN_M: Final = 1.0
_TOP = {
    "scenario", "title", "mode", "overflight", "laser", "start_alt_m", "charter", "target",
    "estimate", "others", "stands", "lines", "light", "map", "faults", "seeds",
}  # fmt: skip
_PERSON = {"id", "at_enu", "jitter_m", "visible_m", "score"}
_STAND = {"name", "at_enu", "radius_m", "height_m", "spacing_m", "jitter_m", "map_short_m"}
_LINE = {"name", "a", "b", "height_m"}
_MAP = {"scan_year", "tree_bias_m", "missing_trees", "holes"}
_CHARTER = {"alt_min_agl_m", "alt_max_agl_m"}
_ESTIMATE = {"error_m", "sigma95_m"}

Xy = tuple[float, float]


class WorldError(ValueError):
    """A scenario file that does not fit the format."""


@dataclass(frozen=True)
class Camera:
    """A stills camera pointing straight down, in the sensor's full pixels."""

    name: str
    width_px: int
    height_px: int
    hfov_deg: float

    @property
    def focal_px(self) -> float:
        return self.width_px / (2.0 * math.tan(math.radians(self.hfov_deg) / 2.0))

    @property
    def half_view_deg(self) -> float:
        """Half the narrower field of view: a person this far off straight down is in."""
        vfov = 2.0 * math.atan(math.tan(math.radians(self.hfov_deg) / 2.0) * self.aspect)
        return math.degrees(vfov) / 2.0

    @property
    def aspect(self) -> float:
        return self.height_px / self.width_px


# The main camera (Raspberry Pi Camera Module 3, 4608 x 2592, 66 degrees) and the tele
# (HQ Camera, 4056 x 3040, the 20-degree M12 lens), as their sensors are.
CAMERAS: Final[tuple[Camera, ...]] = (
    Camera("main", 4608, 2592, 66.0),
    Camera("tele", *cameras.TELE_SENSOR_PX, cameras.TELE_HFOV_DEG),
)


@dataclass(frozen=True)
class Person:
    """Someone on the ground: the target or another person."""

    id: str
    east: float
    north: float
    visible_m: float = 0.4
    score: float = 0.45

    @property
    def enu(self) -> Xy:
        return (self.east, self.north)


@dataclass(frozen=True)
class Sighting:
    """A person in one camera's still: how big, how sure, and where it was placed."""

    person: Person
    camera: str
    px: float
    score: float
    east: float
    north: float
    sigma95_m: float


@dataclass(frozen=True)
class InspectWorld:
    scenario: str
    title: str
    seed: int
    mode: OperationMode
    overflight: bool
    laser: Laser
    start_alt_m: float
    alt_min_agl_m: float
    alt_max_agl_m: float
    target: Person
    estimate_enu: Xy  # the candidate's position as the ground knows it
    sigma95_m: float
    others: tuple[Person, ...] = ()
    stands: tuple[TreeStand, ...] = ()
    lines: tuple[PowerLine, ...] = ()
    light: tuple[Polygon, ...] = ()
    map_errors: MapErrors = field(default_factory=MapErrors)
    faults: Mapping[str, Any] = field(default_factory=dict[str, Any])
    seeds: tuple[int, ...] = (1, 2, 3)

    @property
    def obstacles(self) -> list[objects.WorldObject]:
        return [*self.stands, *self.lines]

    @property
    def people(self) -> tuple[Person, ...]:
        return (self.target, *self.others)


def scenario_path(scenario: str) -> Path:
    """``TN02`` -> ``simulation/scenarios/TN02_<name>.yaml`` (exactly one)."""
    found = sorted(SCENARIOS.glob(f"{scenario}_*.yaml"))
    if len(found) != 1:
        raise WorldError(f"scenariot {scenario}: {len(found)} filer i {SCENARIOS}")
    return found[0]


def load(scenario: str, seed: int, path: Path | None = None) -> InspectWorld:
    """The scenario with its people placed for ``seed``."""
    data = yaml.safe_load((path or scenario_path(scenario)).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise WorldError(f"{scenario}: inte en mapp")
    raw: dict[str, Any] = data
    _keys(raw, _TOP, scenario)
    if raw.get("scenario") != scenario:
        raise WorldError(f"{scenario}: filen säger scenario {raw.get('scenario')!r}")
    mode = raw.get("mode", "exercise")
    laser = raw.get("laser", "none")
    if mode not in MODES or laser not in LASERS:
        raise WorldError(f"{scenario}: mode ska vara {MODES}, laser {LASERS}")
    charter = dict(raw.get("charter") or {})
    _keys(charter, _CHARTER, f"{scenario}.charter")
    estimate = dict(raw.get("estimate") or {})
    _keys(estimate, _ESTIMATE, f"{scenario}.estimate")
    seeds = tuple(int(s) for s in raw.get("seeds", (1, 2, 3)))
    if not seeds or len(set(seeds)) != len(seeds):
        raise WorldError(f"{scenario}: seeds ska vara olika och minst ett")
    target = _person(dict(raw["target"]), "target", seed, scenario)
    err = float(estimate.get("error_m", 3.0))
    de, dn = seed_offset(seed, err, f"{scenario}:estimate")
    mapping = dict(raw.get("map") or {})
    _keys(mapping, _MAP, f"{scenario}.map")
    holes = tuple((float(e), float(n), float(r)) for e, n, r in mapping.get("holes", ()))
    world = InspectWorld(
        scenario=scenario,
        title=str(raw.get("title", scenario)),
        seed=seed,
        mode=mode,
        overflight=bool(raw.get("overflight", mode != "live")),
        laser=laser,
        start_alt_m=float(raw.get("start_alt_m", 50.0)),
        alt_min_agl_m=float(charter.get("alt_min_agl_m", 20.0)),
        alt_max_agl_m=float(charter.get("alt_max_agl_m", 120.0)),
        target=target,
        estimate_enu=(target.east + de, target.north + dn),
        sigma95_m=float(estimate.get("sigma95_m", 5.0)),
        others=tuple(
            _person(dict(p), str(p.get("id", f"other{i}")), seed, scenario)
            for i, p in enumerate(raw.get("others") or ())
        ),
        stands=tuple(_stand(dict(s), seed, scenario) for s in raw.get("stands") or ()),
        lines=tuple(_line(dict(ln), scenario) for ln in raw.get("lines") or ()),
        light=tuple(
            Polygon([(float(x), float(y)) for x, y in ring]) for ring in raw.get("light") or ()
        ),
        map_errors=MapErrors(
            tree_bias_m=float(mapping.get("tree_bias_m", 0.0)),
            missing_trees=float(mapping.get("missing_trees", 0.0)),
            holes=holes,
            scan_year=None if mapping.get("scan_year") is None else int(mapping["scan_year"]),
            seed=seed,
        ),
        faults=dict(raw.get("faults") or {}),
        seeds=seeds,
    )
    if not world.alt_min_agl_m < world.start_alt_m <= world.alt_max_agl_m:
        raise WorldError(f"{scenario}: start_alt_m ska ligga inom ramens höjder")
    return world


def _keys(raw: Mapping[str, Any], allowed: set[str], where: str) -> None:
    extra = set(raw) - allowed
    if extra:
        raise WorldError(f"{where}: okända nycklar {sorted(extra)}")


def _xy(value: Any, where: str) -> Xy:
    if not isinstance(value, Sequence) or len(value) != 2:  # pyright: ignore[reportUnknownArgumentType]
        raise WorldError(f"{where}: ska vara [öst, nord]")
    return (float(value[0]), float(value[1]))  # pyright: ignore[reportUnknownArgumentType]


def _person(raw: dict[str, Any], pid: str, seed: int, scenario: str) -> Person:
    _keys(raw, _PERSON, f"{scenario}.{pid}")
    east, north = _xy(raw.get("at_enu"), f"{scenario}.{pid}.at_enu")
    de, dn = seed_offset(seed, float(raw.get("jitter_m", 0.0)), f"{scenario}:{pid}")
    visible, score = float(raw.get("visible_m", 0.4)), float(raw.get("score", 0.45))
    if not 0.05 <= visible <= 2.0 or not 0.0 < score <= 1.0:
        raise WorldError(f"{scenario}.{pid}: visible_m 0,05-2, score 0-1")
    return Person(pid, round(east + de, 2), round(north + dn, 2), visible, score)


def _stand(raw: dict[str, Any], seed: int, scenario: str) -> TreeStand:
    _keys(raw, _STAND, f"{scenario}.stands")
    east, north = _xy(raw.get("at_enu"), f"{scenario}.stands.at_enu")
    try:
        return TreeStand(
            name=str(raw["name"]),
            east_m=east,
            north_m=north,
            radius_m=float(raw["radius_m"]),
            height_m=float(raw["height_m"]),
            spacing_m=float(raw.get("spacing_m", 6.0)),
            jitter_m=float(raw.get("jitter_m", 0.0)),
            seed=seed,
            map_short_m=float(raw.get("map_short_m", 0.0)),
        )
    except (KeyError, objects.ObjectError) as exc:
        raise WorldError(f"{scenario}.stands: {exc}") from exc


def _line(raw: dict[str, Any], scenario: str) -> PowerLine:
    _keys(raw, _LINE, f"{scenario}.lines")
    try:
        return PowerLine(
            name=str(raw["name"]),
            a=_xy(raw.get("a"), f"{scenario}.lines.a"),
            b=_xy(raw.get("b"), f"{scenario}.lines.b"),
            height_m=float(raw.get("height_m", 20.0)),
        )
    except (KeyError, objects.ObjectError) as exc:
        raise WorldError(f"{scenario}.lines: {exc}") from exc


class Oracle:
    """The cameras and the detector for a closer look: what each camera's still shows of
    the world's people from where the drone truly is. The same seed, step, round and
    camera always give the same noise."""

    def __init__(self, world: InspectWorld, cameras_: Sequence[Camera] = CAMERAS) -> None:
        self.world = world
        self.cameras = tuple(cameras_)

    def px(self, camera: Camera, drone: tuple[float, float, float], person: Person) -> float | None:
        """The person's size in ``camera``'s still (full pixels), or None out of view."""
        east, north, up = drone
        off = math.hypot(person.east - east, person.north - north)
        if up <= 0.5 or math.degrees(math.atan2(off, up)) > camera.half_view_deg:
            return None
        return person.visible_m * camera.focal_px / math.hypot(off, up)

    def still(
        self, camera: Camera, drone: tuple[float, float, float], key: tuple[int, int]
    ) -> list[Sighting]:
        """The detections in one still; ``key`` = (step, round) for the noise."""
        out = []
        for person in self.world.people:
            px = self.px(camera, drone, person)
            if px is None or px < DETECT_MIN_PX:
                continue
            rng = random.Random(  # noqa: S311 - a simulation
                f"{self.world.scenario}:{self.world.seed}:{key}:{camera.name}:{person.id}"
            )
            rng_m = math.hypot(person.east - drone[0], person.north - drone[1], drone[2])
            sigma = NOISE_PER_M * rng_m
            out.append(
                Sighting(
                    person=person,
                    camera=camera.name,
                    px=round(px, 1),
                    score=person.score,
                    east=person.east + rng.gauss(0.0, sigma),
                    north=person.north + rng.gauss(0.0, sigma),
                    sigma95_m=round(max(SIGMA95_MIN_M, 2.45 * sigma), 2),
                )
            )
        return out
