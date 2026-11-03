"""The photo farm: training pictures of people on a real site, without a flight.

    uv run --package sd-sim python -m sd_sim.farm stills [scenes=N] [seed=S] [site=NAME]
        [shots=N] [instance=N] [gi=off|vct] [out=DIR] [profile=heridal|kios]

The search-round farm (``just sim-farm``) flies a whole search in SITL to get its
pictures; most of them show no one, and a round takes ten minutes. The detector only
needs the pictures, so this farm places the camera instead: one **scene** is one Gazebo
world of a real site (``sd_sim.site_world``: the terrain, the site's real trees from
``just trees-fetch`` with ``just tree-pictures-fetch``'s textures, and the orthophoto on
open ground when the owner has fetched it) with people placed in it, under one drawn
sun and weather, and a free camera rig (``Rig``) that ``gz_stills.py shots`` moves to
every planned pose. Each pose aims at a person from a drawn height so the person lands
anywhere in the picture; some poses look at nothing.

Real proportions, so the pictures teach the same sizes the real data shows:

* **people** are scaled to a drawn body height (adults 1.50-1.95 m, sometimes a child of
  1.10-1.45 m, ``draw_height``) and stand, walk, sit or lie; their clothes are the
  model's own or recoloured to a drawn top and bottom (``sd_sim.farm.looks``);
* **cameras** (``PROFILES``) give the ground sample distance (cm per pixel) of the real
  sets the detector is measured on: ``heridal`` the centre of a 12 MP drone camera at
  25-45 m, as HERIDAL's 4000 x 3000 pictures; ``kios`` a 1920 x 1080 video camera at
  45-85 m, as KIOS (heights set so the person boxes come out the size of the real
  sets'). The camera is never lower than ``TREE_CLEARANCE_M`` over the trees
  under it;
* **places** (``PLACES``): in the open, at a forest edge, under a tree's crown (partly
  hidden) and on a path; never in water or a house.

The labels are the bounding-box camera's boxes of what is visible of each person
(``labels.facit`` pairs them with the projected box), so a person half under a crown is
boxed as far as it shows, as in the real sets' hand-drawn boxes. A person the camera
does not see at all gets no box.

Output, one folder per scene under ``out`` (default ``build/simfarm/stills``), the
layout the image bank reads (``tools/bench/layouts.py``, ``simfarm``)::

    <scene>/images/<scene>_<n>.jpg   the pictures (JPEG, as the real sets)
    <scene>/labels/<scene>_<n>.txt   YOLO, class 0 = person
    <scene>/meta/<scene>_<n>.json    camera, height, ground sample distance, people
    <scene>/coco.json                COCO for the scene
    <scene>/scene.json               the draw, the counts and the check, written last
    <scene>/logs/                    Gazebo's and the shots' whole logs

A scene that fails keeps its ``scene.json`` with the error and its logs, never its
pictures.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import shapely

from sd_sim import _args, forest, gazebo, paths, procs, site, site_world, terrain, world
from sd_sim.farm import ground as ground_mod
from sd_sim.farm import labels, looks, people, scenario
from sd_sim.farm.camera import Box, CameraPose, Intrinsics, project, quat_from_rpy
from sd_sim.farm.people import CATALOG, Person

SCENE_SCHEMA = "simfarm.stills_scene/v1"
DEFAULT_SITE = "hellasgarden"
DEFAULT_OUT = Path("build") / "simfarm" / "stills"
ALLOWED_KEYS = ("scenes", "seed", "site", "shots", "instance", "gi", "out", "profile")
RIG_TEMPLATE = "objects/sd_still_rig.sdf.jinja"
RIG_RATE_HZ = 4
GZ_START_S = 240.0
SHOT_BUDGET_S = 20.0  # per planned shot, before the shots process is stopped
JPEG_QUALITY = 92
# The camera's development (``develop``): the picture's mean grey level after the
# automatic exposure, the highest exposure gain, the lens softness and the sensor noise
# (grey levels), each drawn per picture; JPEG quality likewise.
EXPOSURE_MEAN = (95.0, 135.0)
MAX_GAIN = 12.0
KNEE = 0.8
BLUR_SIGMA_PX = (0.3, 0.8)
NOISE_SD = (1.0, 3.0)
JPEG_QUALITY_RANGE = (85, 95)


@dataclass(frozen=True)
class Profile:
    """A camera the farm imitates: picture size, field of view across and the heights
    above the ground under the person it is flown at."""

    name: str
    width: int
    height: int
    hfov_deg: float
    heights_m: tuple[float, float]
    about: str

    @property
    def intrinsics(self) -> Intrinsics:
        return Intrinsics(self.width, self.height, math.radians(self.hfov_deg))

    def gsd_cm(self, height_m: float) -> float:
        """Ground sample distance straight down from ``height_m``: centimetres per pixel."""
        return 100.0 * 2.0 * height_m * math.tan(math.radians(self.hfov_deg) / 2) / self.width


PROFILES: dict[str, Profile] = {
    # HERIDAL: 4000 x 3000 from a DJI Phantom-class camera (about 81 degrees across).
    # The centre 2000 x 1500 of it has the same pixels at a quarter of the cost: 44
    # degrees across. 25-45 m gives 1.0-1.8 cm per pixel.
    "heridal": Profile("heridal", 2000, 1500, 44.0, (25.0, 45.0), "centre of a 12 MP drone camera"),
    # KIOS: 1920 x 1080 video with a wide lens; 75 degrees at 45-85 m gives 3.6-6.8 cm
    # per pixel.
    "kios": Profile("kios", 1920, 1080, 75.0, (45.0, 85.0), "full HD drone video"),
}
PROFILE_SHARE = {"heridal": 0.5, "kios": 0.5}

PEOPLE_PER_SCENE = (12, 18)
GROUP_SHARE = 0.15  # a second person a few metres from the first
RADIUS_M = 280.0  # people stand this close to home (the detailed trees are near home)
MIN_GAP_M = 4.0
# Where people are put, with the share of each.
PLACES = {"open": 0.35, "edge": 0.25, "under": 0.25, "path": 0.15}
PATH_KINDS = ("path", "footway", "track", "cycleway", "bridleway", "service", "unclassified")
WATER_GAP_M = 2.0
BUILDING_GAP_M = 2.0
# Poses: of the standing models, how many lie down; the sitting model always sits.
LYING_SHARE = 0.35
SITTING_SHARE = 0.15
OWN_CLOTHES_SHARE = 0.5
ADULT_M = (1.72, 0.09, 1.50, 1.95)  # mean, spread, lowest, highest
CHILD_SHARE = 0.08
CHILD_M = (1.10, 1.45)
# Shots
SHOTS_PER_PERSON = 3
EMPTY_SHARE = 0.1
# A person whose seen pixels are less than this share of what it would cover unhidden
# (under a crown) gets no label: the real sets' hand-drawn boxes leave such people out.
MIN_SHOWN = 0.3
# The share of its turned model box's outline a body covers in the picture, per pose
# (0.85 of the median for unhidden people in a test render, so they count as wholly seen;
# see ``silhouette_px``).
FILL = {"standing": 0.3, "sitting": 0.28, "lying": 0.42}
EDGE_KEEP = 0.04  # a person's centre at least this share of the picture from its edge
TILT_SD_DEG = 3.0
TILT_MAX_DEG = 10.0
TREE_CLEARANCE_M = 5.0  # over the highest tree top within the camera's reach below it
LIGHT_GAIN = (1.0, 1.0)  # sun, ambient


class StillsError(RuntimeError):
    """A scene could not be made."""


# --- the draw --------------------------------------------------------------------------


@dataclass(frozen=True)
class Shot:
    name: str
    profile: str
    east_m: float
    north_m: float
    z_m: float  # camera height above home
    height_m: float  # above the ground under the aimed person (or the camera)
    roll_deg: float
    pitch_deg: float
    yaw_deg: float  # compass heading of the picture's top
    aim: str | None  # the person aimed at

    def gz(self) -> dict[str, Any]:
        """The pose as ``gz_stills shots`` reads it: Gazebo's frame (x east, y north, z
        up), radians, the model's front (the picture's top) along ``yaw_deg``."""
        return {
            "name": self.name,
            "x": round(self.east_m, 3),
            "y": round(self.north_m, 3),
            "z": round(self.z_m, 3),
            "roll": round(math.radians(self.roll_deg), 6),
            "pitch": round(-math.radians(self.pitch_deg), 6),
            "yaw": round(math.radians(90.0 - self.yaw_deg), 6),
        }

    def camera_pose(self) -> CameraPose:
        g = self.gz()
        body = quat_from_rpy(g["roll"], g["pitch"], g["yaw"])
        return CameraPose.mounted((g["x"], g["y"], g["z"]), body)


@dataclass(frozen=True)
class Scene:
    scene_id: str
    seed: int
    site: str
    profile: str
    day_of_year: int
    solar_hour: float
    sun: scenario.Sun
    weather: str
    people: tuple[Person, ...]
    places: dict[str, str]  # person name -> place kind
    shots: tuple[Shot, ...]

    @property
    def light(self) -> dict[str, Any]:
        """The weather's light, made brighter as a camera's exposure would (``LIGHT_GAIN``):
        Gazebo's picture of the scenario light is far darker than a photo of the day."""
        light = scenario.light_of(self.weather, self.sun)
        sun_gain, ambient_gain = LIGHT_GAIN
        return {
            **light,
            "sun_intensity": round(min(1.0, light["sun_intensity"] * sun_gain), 3),
            "ambient": round(min(1.0, light["ambient"] * ambient_gain), 3),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCENE_SCHEMA,
            "scene_id": self.scene_id,
            "seed": self.seed,
            "site": self.site,
            "profile": asdict(PROFILES[self.profile]),
            "day_of_year": self.day_of_year,
            "solar_hour": self.solar_hour,
            "sun": asdict(self.sun),
            "weather": self.weather,
            "light": self.light,
            "people": [
                {
                    "name": p.name,
                    "model": p.model,
                    "pose": p.pose,
                    "place": self.places[p.name],
                    "east_m": p.east_m,
                    "north_m": p.north_m,
                    "yaw_deg": p.yaw_deg,
                    "height_m": round(p.height_m, 2),
                    "scale": p.scale,
                    "look": p.look or None,
                }
                for p in self.people
            ],
            "shots": [asdict(s) for s in self.shots],
        }


@dataclass
class Land:
    """What the draw needs of a site: where people may stand and how high things are."""

    ground: Callable[[float, float], float]
    trees: list[forest.RealTree]
    open_land: Any  # shapely: meadow, grass, farmland, pitch
    blocked: Any  # shapely: water and buildings, with a margin
    paths: list[Any]  # shapely lines
    tree_xy: Any = field(init=False)

    def __post_init__(self) -> None:
        self.tree_xy = np.array([(t.east, t.north) for t in self.trees]).reshape(-1, 2)

    def free(self, e: float, n: float) -> bool:
        return math.hypot(e, n) > 8.0 and not shapely.contains_xy(self.blocked, e, n)

    def top_near(self, e: float, n: float, reach: float) -> float:
        """Highest ground or tree top within ``reach`` metres, above home."""
        top = self.ground(e, n)
        if len(self.tree_xy):
            d = np.hypot(self.tree_xy[:, 0] - e, self.tree_xy[:, 1] - n)
            for i in np.flatnonzero(d < reach + 8.0):
                t = self.trees[i]
                if d[i] <= reach + t.crown_radius:
                    top = max(top, self.ground(t.east, t.north) + t.height)
        return top


def land_of(s: site.Site, ground: terrain.Ground) -> Land:
    def at(e: float, n: float) -> float:
        return float(ground.at(np.array([e]), np.array([n]))[0])

    trees = forest.load_trees(s) or []
    paths_ = [
        layer.geom
        for layer in s.of_kind("road")
        if str(layer.props.get("highway", layer.osm)) in PATH_KINDS
        or any(k in str(layer.osm) for k in PATH_KINDS)
    ]
    blocked = shapely.union_all(
        [s.union("water").buffer(WATER_GAP_M), s.union("building").buffer(BUILDING_GAP_M)]
    )
    return Land(
        at,
        [t for t in trees if math.hypot(t.east, t.north) < RADIUS_M + 60.0],
        s.union("meadow", "grass", "farmland", "pitch"),
        blocked,
        paths_,
    )


def draw_height(rng: random.Random) -> float:
    if rng.random() < CHILD_SHARE:
        return round(rng.uniform(*CHILD_M), 2)
    mean, sd, low, high = ADULT_M
    return round(min(high, max(low, rng.gauss(mean, sd))), 2)


def _spot(rng: random.Random, land: Land, kind: str) -> tuple[float, float] | None:
    """A free spot of ``kind`` within ``RADIUS_M`` of home, or ``None``."""
    for _ in range(200):
        if kind in ("under", "edge"):
            if not land.trees:
                return None
            t = rng.choice(land.trees)
            r = t.crown_radius
            dist = rng.uniform(0.2, 0.8) * r if kind == "under" else rng.uniform(r, r + 4.0)
            ang = rng.uniform(0, 2 * math.pi)
            e, n = t.east + dist * math.cos(ang), t.north + dist * math.sin(ang)
        elif kind == "path":
            if not land.paths:
                return None
            line = rng.choice(land.paths)
            p = line.interpolate(rng.uniform(0, line.length))
            e, n = p.x + rng.uniform(-1.0, 1.0), p.y + rng.uniform(-1.0, 1.0)
        else:
            if land.open_land.is_empty:
                return None
            ang, rad = rng.uniform(0, 2 * math.pi), RADIUS_M * math.sqrt(rng.random())
            e, n = rad * math.cos(ang), rad * math.sin(ang)
            if not shapely.contains_xy(land.open_land, e, n):
                continue
        if math.hypot(e, n) < RADIUS_M and land.free(e, n):
            return (round(e, 2), round(n, 2))
    return None


def _person(rng: random.Random, i: int, spot: tuple[float, float]) -> Person:
    sitting = [m for m in CATALOG if m.pose == "sitting"]
    standing = [m for m in CATALOG if m.pose != "sitting"]
    if sitting and rng.random() < SITTING_SHARE:
        model, pose = rng.choice(sitting), "sitting"
    else:
        model = rng.choice(standing)
        pose = "lying" if rng.random() < LYING_SHARE else "standing"
    look = ""
    if rng.random() >= OWN_CLOTHES_SHARE:
        top = rng.choice(sorted(looks.TOPS))
        bottom = rng.choice(sorted(looks.BOTTOMS))
        look = looks.name(model.id, top, bottom)
    scale = round(draw_height(rng) / model.body_height_m, 4)
    low, high = people.SCALE_RANGE
    return Person(
        name=f"person_{i}",
        model=model.id,
        pose=pose,
        east_m=spot[0],
        north_m=spot[1],
        yaw_deg=round(rng.uniform(0.0, 360.0), 1),
        scale=min(high, max(low, scale)),
        look=look,
    )


def _weighted(rng: random.Random, shares: Mapping[str, float]) -> str:
    x, acc = rng.random() * sum(shares.values()), 0.0
    for k, v in shares.items():
        acc += v
        if x < acc:
            return k
    return list(shares)[-1]


def _tilt(rng: random.Random) -> float:
    return round(max(-TILT_MAX_DEG, min(TILT_MAX_DEG, rng.gauss(0.0, TILT_SD_DEG))), 2)


def _aimed(
    rng: random.Random, prof: Profile, land: Land, name: str, target: tuple[float, float] | None
) -> Shot:
    """A shot from a drawn height whose picture holds ``target`` (east, north) at a drawn
    place inside it, or a shot of nothing in particular."""
    h = round(rng.uniform(*prof.heights_m), 1)
    yaw = round(rng.uniform(0.0, 360.0), 1)
    if target is None:
        ang, rad = rng.uniform(0, 2 * math.pi), RADIUS_M * math.sqrt(rng.random())
        target = (rad * math.cos(ang), rad * math.sin(ang))
    # Where in the picture the target lands, as a share of the footprint from its centre.
    fx = rng.uniform(-0.5 + EDGE_KEEP, 0.5 - EDGE_KEEP)
    fy = rng.uniform(-0.5 + EDGE_KEEP, 0.5 - EDGE_KEEP)
    half_w = h * math.tan(math.radians(prof.hfov_deg) / 2)
    foot_w, foot_h = 2 * half_w, 2 * half_w * prof.height / prof.width
    right = (math.cos(math.radians(yaw)), -math.sin(math.radians(yaw)))  # east, north
    up = (math.sin(math.radians(yaw)), math.cos(math.radians(yaw)))
    # The target sits fx right and fy up of the centre, so the centre is the other way.
    ce = target[0] - fx * foot_w * right[0] - fy * foot_h * up[0]
    cn = target[1] - fx * foot_w * right[1] - fy * foot_h * up[1]
    ground = land.ground(target[0], target[1])
    z = ground + h
    top = land.top_near(ce, cn, 3.0)
    if z < top + TREE_CLEARANCE_M:
        z = top + TREE_CLEARANCE_M
    shot = Shot(
        name=name,
        profile=prof.name,
        east_m=round(ce, 2),
        north_m=round(cn, 2),
        z_m=round(z, 2),
        height_m=round(z - ground, 2),
        roll_deg=_tilt(rng),
        pitch_deg=_tilt(rng),
        yaw_deg=yaw,
        aim=None,
    )
    # The tilt moves the picture over the ground: move the camera back so the target
    # lands where it was drawn to (the picture's v grows downwards, against ``up``).
    want = ((0.5 + fx) * prof.width, (0.5 - fy) * prof.height)
    for _ in range(4):
        uv = project(prof.intrinsics, shot.camera_pose(), (target[0], target[1], ground))
        if uv is None:
            break
        g = (z - ground) * 2 * math.tan(math.radians(prof.hfov_deg) / 2) / prof.width
        du, dv = (uv[0] - want[0]) * g, (uv[1] - want[1]) * g
        shot = replace(
            shot,
            east_m=round(shot.east_m + du * right[0] - dv * up[0], 2),
            north_m=round(shot.north_m + du * right[1] - dv * up[1], 2),
        )
    return shot


def draw(
    seed: int,
    site_name: str,
    land: Land,
    lat_deg: float,
    shots: int = 0,
    profile: str | None = None,
) -> Scene:
    """The scene for ``seed``: the same seed and site always give the same scene.
    ``shots``: how many pictures (0: ``SHOTS_PER_PERSON`` per person plus the empty
    ones); ``profile``: the camera (default: drawn by ``PROFILE_SHARE``)."""
    rng = random.Random(f"simfarm-stills:{site_name}:{seed}")  # noqa: S311 - scenery
    day = rng.randint(*scenario.DAYS)
    for _ in range(200):
        hour = round(rng.uniform(5.0, 19.5), 2)
        sun = scenario.sun_position(lat_deg, day, hour)
        if sun.elevation_deg >= scenario.MIN_SUN_ELEVATION_DEG + 3.0:
            break
    else:
        hour, sun = 12.0, scenario.sun_position(lat_deg, day, 12.0)
    weather = rng.choice(sorted(scenario.WEATHER))
    drawn = _weighted(rng, PROFILE_SHARE)
    prof = PROFILES[profile or drawn]
    count = rng.randint(*PEOPLE_PER_SCENE)
    placed: list[Person] = []
    places: dict[str, str] = {}
    tries = 0
    while len(placed) < count and tries < count * 20:
        tries += 1
        kind = _weighted(rng, PLACES)
        spot = _spot(rng, land, kind)
        if spot is None:
            continue
        group = [spot]
        if rng.random() < GROUP_SHARE:
            a, d = rng.uniform(0, 2 * math.pi), rng.uniform(1.2, 3.0)
            buddy = (round(spot[0] + d * math.cos(a), 2), round(spot[1] + d * math.sin(a), 2))
            if land.free(*buddy):
                group.append(buddy)
        for e, n in group:
            if len(placed) >= count:
                break
            near = [p for p in placed if math.hypot(p.east_m - e, p.north_m - n) < 1.0]
            far_enough = all(math.hypot(p.east_m - e, p.north_m - n) >= MIN_GAP_M for p in placed)
            if near or not (far_enough or len(group) > 1):
                continue
            person = _person(rng, len(placed), (e, n))
            placed.append(person)
            places[person.name] = kind
    if not placed:
        raise StillsError(f"{site_name}: ingen plats för personer hittades")
    plan: list[Shot] = []
    wanted = shots or len(placed) * SHOTS_PER_PERSON + max(1, round(len(placed) * 0.3))
    n = 0
    while len(plan) < wanted:
        if rng.random() < EMPTY_SHARE:
            shot = _aimed(rng, prof, land, f"s{n:03d}", None)
        else:
            p = (
                placed[n % len(placed)]
                if n < len(placed) * SHOTS_PER_PERSON
                else rng.choice(placed)
            )
            shot = _aimed(rng, prof, land, f"s{n:03d}", (p.east_m, p.north_m))
            shot = Shot(**{**asdict(shot), "aim": p.name})
        plan.append(shot)
        n += 1
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return Scene(
        scene_id=f"{stamp}-{site_name}-s{seed}",
        seed=seed,
        site=site_name,
        profile=prof.name,
        day_of_year=day,
        solar_hour=hour,
        sun=sun,
        weather=weather,
        people=tuple(placed),
        places=places,
        shots=tuple(plan),
    )


# --- the rig ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rig:
    """The free camera with its bounding-box camera, as a ``WorldObject``; it starts a few
    metres up near home and ``gz_stills shots`` moves it."""

    instance: int
    profile: str
    east_m: float = 10.0
    north_m: float = -10.0
    up_m: float = 5.0
    yaw_deg: float = 0.0
    label: int = 0
    footprint_m: float = 0.1
    truth: dict[str, Any] = field(default_factory=lambda: {"role": "camera"})

    @property
    def name(self) -> str:
        return f"sd_stills_rig_i{self.instance}"

    @property
    def topic_prefix(self) -> str:
        return f"/sd_stills_i{self.instance}"

    def sdf(self, pose: Any) -> str:
        prof = PROFILES[self.profile]
        template = world._env().get_template(RIG_TEMPLATE)
        return template.render(
            name=self.name,
            pose=pose.sdf(),
            topic_prefix=self.topic_prefix,
            width=prof.width,
            height=prof.height,
            hfov_rad=round(math.radians(prof.hfov_deg), 6),
            rate_hz=RIG_RATE_HZ,
            labels_map=True,
        ).strip()


# --- the picture's facit ---------------------------------------------------------------


def silhouette_px(person: Person, ground_m: float, cam: Intrinsics, pose: CameraPose) -> float:
    """The picture area (pixels) inside the outline of the person's turned model box,
    clipped to the picture: what the person would cover with nothing in the way, times
    ``FILL`` (a body does not fill its box)."""
    pts = [project(cam, pose, c) for c in person.corners(ground_m)]
    if any(q is None for q in pts):
        return 0.0
    hull = shapely.MultiPoint([q for q in pts if q is not None]).convex_hull
    frame = shapely.box(0.0, 0.0, float(cam.width), float(cam.height))
    return float(hull.intersection(frame).area) * FILL[person.pose]


def shown_shares(
    truth: Mapping[str, Any],
    fac: labels.ImageFacit,
    cam: Intrinsics,
    pose: CameraPose,
    label_map: np.ndarray | None,
) -> dict[str, float | None]:
    """How much of each person the picture shows: the person pixels of the segmentation
    camera's label map inside the person's box, over the pixels the person would cover
    unhidden (``silhouette_px``). ``None`` without a label map."""
    if label_map is None:
        return {p.name: None for p in fac.people}
    where = {person.name: (person, ground) for person, ground in labels.people_in(truth)}
    out: dict[str, float | None] = {}
    for p in fac.people:
        person, ground = where[p.name]
        b = p.box
        x0, y0 = max(int(b.x0), 0), max(int(b.y0), 0)
        x1, y1 = min(math.ceil(b.x1), cam.width), min(math.ceil(b.y1), cam.height)
        seen = int(np.count_nonzero(label_map[y0:y1, x0:x1] == people.PERSON_LABEL))
        whole = silhouette_px(person, ground, cam, pose)
        out[p.name] = round(min(1.0, seen / whole), 3) if whole > 0 else 0.0
    return out


def _shown_enough(share: float | None) -> bool:
    return share is None or share >= MIN_SHOWN


def label_boxes(fac: labels.ImageFacit, shown: Mapping[str, float | None]) -> list[Box]:
    """The boxes to train on: what the camera sees of each labelled person, clipped. A
    person shown less than ``MIN_SHOWN`` (mostly under a crown) gets none: no one could
    point at it in the picture either."""
    out = []
    for p in fac.labels:
        if not _shown_enough(shown.get(p.name)):
            continue
        b = p.seen if p.seen is not None else p.box
        if b.w >= labels.MIN_BOX_PX and b.h >= labels.MIN_BOX_PX:
            out.append(b)
    return out


def picture_facit(
    truth: Mapping[str, Any],
    shot: Shot,
    cam: Intrinsics,
    raw: Mapping[str, Any],
    label_map: np.ndarray | None = None,
) -> tuple[labels.ImageFacit, list[Box], dict[str, float | None]]:
    pose = shot.camera_pose()
    fac = labels.facit(truth, cam, pose, labels.camera_boxes(raw.get("boxes")))
    shown = shown_shares(truth, fac, cam, pose, label_map)
    return fac, label_boxes(fac, shown), shown


def read_pgm(path: Path) -> np.ndarray:
    """A one-byte-per-pixel PGM (``gz_stills`` label map) as an array, rows first."""
    data = path.read_bytes()
    head = data.split(maxsplit=4)
    if len(head) < 5 or head[0] != b"P5" or head[3] != b"255":
        raise StillsError(f"{path.name}: ingen PGM")
    width, height = int(head[1]), int(head[2])
    pixels = data[len(data) - width * height :]
    return np.frombuffer(pixels, np.uint8).reshape(height, width)


def _gsd(prof: Profile, shot: Shot) -> float:
    return round(prof.gsd_cm(shot.height_m), 3)


# --- running a scene ------------------------------------------------------------------


def gz_env(instance: int, repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    dirs = [
        str(repo / "build" / "sim" / "models"),
        str(looks.looks_dir(repo)),
        *map(str, people.search_dirs(env, repo)),
    ]
    env.update(
        {
            "GZ_IP": "127.0.0.1",
            "GZ_PARTITION": f"sd_stills_i{instance}",
            "GZ_SIM_RESOURCE_PATH": os.pathsep.join(dict.fromkeys(d for d in dirs if d)),
        }
    )
    return env


def _write_json(path: Path, body: Any) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def _to_linear(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1 / 2.4) - 0.055)


def develop(image: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict[str, float]]:
    """What a drone camera does to the light, in place of Gazebo's plain picture: an
    automatic exposure to a drawn brightness (``EXPOSURE_MEAN``; Gazebo's pictures are far
    darker than a photo of the same day), a soft roll-off of the highlights, a little lens
    softness and sensor noise. The boxes do not move."""
    target = rng.uniform(*EXPOSURE_MEAN)
    lin = _to_linear(image.astype(np.float32) / 255.0)
    small = lin[::8, ::8]
    gain = 1.0
    for _ in range(6):  # the gain that gives the drawn mean, after the roll-off
        mean = float(_to_srgb(_rolloff(small * gain)).mean() * 255.0)
        gain *= (target / max(mean, 1.0)) ** 2.2
        gain = min(max(gain, 0.5), MAX_GAIN)
    out = _to_srgb(_rolloff(lin * gain)) * 255.0
    blur = rng.uniform(*BLUR_SIGMA_PX)
    noise = rng.uniform(*NOISE_SD)
    from PIL import Image, ImageFilter  # the dev group's Pillow, as render_stills

    pic = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    pic = pic.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(pic).astype(np.float32)
    gen = np.random.default_rng(rng.getrandbits(32))
    arr += gen.normal(0.0, noise, arr.shape[:2])[..., None]  # brightness noise
    arr += gen.normal(0.0, noise * 0.5, arr.shape)  # colour noise
    done = np.clip(arr + 0.5, 0, 255).astype(np.uint8)
    return done, {
        "target_mean": round(target, 1),
        "gain": round(gain, 3),
        "blur_px": round(blur, 2),
        "noise": round(noise, 2),
    }


def _rolloff(v: np.ndarray) -> np.ndarray:
    """Linear light to linear light with a shoulder over ``KNEE``, so a bright roof or
    a white shirt saturates softly rather than clipping flat."""
    k = KNEE
    over = np.maximum(v - k, 0.0)
    return np.where(v < k, v, k + (1 - k) * (1 - np.exp(-over / (1 - k))))


def _save_jpeg(image: np.ndarray, path: Path, quality: int = JPEG_QUALITY) -> None:
    from PIL import Image  # the dev group's Pillow, as render_stills

    tmp = path.with_name(f".{path.name}.tmp")
    Image.fromarray(image).save(tmp, format="JPEG", quality=quality)
    tmp.replace(path)


def finalize(scene: Scene, truth: Mapping[str, Any], shots_dir: Path, out: Path) -> dict[str, Any]:
    """The scene's dataset from the raw shots; returns the counts and the check."""
    prof = PROFILES[scene.profile]
    cam = prof.intrinsics
    for sub in ("images", "labels", "meta"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    credits = sorted({json.dumps(p.spec.attribution(), sort_keys=True) for p in scene.people})
    coco = labels.Coco(
        {
            "description": "Searchdrone photo farm: pictures of a real site with person boxes",
            "scene_id": scene.scene_id,
            "credits": [json.loads(c) for c in credits],
        }
    )
    tally = {"pictures": 0, "black": 0, "missing": 0, "labels": 0, "hidden": 0, "partly": 0}
    sizes: list[float] = []
    ious: list[float] = []
    for i, shot in enumerate(scene.shots):
        meta_path = shots_dir / f"{shot.name}.json"
        ppm = shots_dir / f"{shot.name}.ppm"
        if not (meta_path.is_file() and ppm.is_file()):
            tally["missing"] += 1
            continue
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        image = labels.read_ppm(ppm)
        if image.shape[1] != cam.width or image.shape[0] != cam.height:
            raise StillsError(f"{ppm.name}: storleken stämmer inte med kameran")
        if labels.is_black(image):
            tally["black"] += 1
            continue
        label_map = None
        if raw.get("labels_map"):
            label_map = read_pgm(shots_dir / str(raw["labels_map"]))
        fac, boxes, shown = picture_facit(truth, shot, cam, raw, label_map)
        stem = f"{scene.scene_id}_{i:04d}"
        mean, std = labels.brightness(image)
        rng = random.Random(f"develop:{scene.scene_id}:{i}")  # noqa: S311 - not a secret
        image, developed = develop(image, rng)
        developed["jpeg_quality"] = rng.randint(*JPEG_QUALITY_RANGE)
        _save_jpeg(image, out / "images" / f"{stem}.jpg", developed["jpeg_quality"])
        rows = []
        for b in boxes:
            cx, cy = (b.x0 + b.x1) / 2 / cam.width, (b.y0 + b.y1) / 2 / cam.height
            rows.append(f"0 {cx:.6f} {cy:.6f} {b.w / cam.width:.6f} {b.h / cam.height:.6f}\n")
            sizes.append(max(b.w, b.h))
        (out / "labels" / f"{stem}.txt").write_text("".join(rows), encoding="utf-8")
        coco_id = len(coco.data["images"]) + 1
        coco.data["images"].append(
            {
                "id": coco_id,
                "file_name": f"images/{stem}.jpg",
                "width": cam.width,
                "height": cam.height,
            }
        )
        for b in boxes:
            coco.data["annotations"].append(
                {
                    "id": len(coco.data["annotations"]) + 1,
                    "image_id": coco_id,
                    "category_id": labels.COCO_CATEGORY["id"],
                    "bbox": [round(b.x0, 2), round(b.y0, 2), round(b.w, 2), round(b.h, 2)],
                    "area": round(b.area, 2),
                    "iscrowd": 0,
                }
            )
        tally["pictures"] += 1
        tally["labels"] += len(boxes)
        in_view = [p for p in fac.people if p.in_view >= labels.MIN_IN_VIEW]
        tally["hidden"] += sum(1 for p in in_view if p.seen is None)
        tally["partly"] += sum(1 for p in fac.labels if p.visible < labels.UNHIDDEN)
        ious += [p.iou for p in fac.labels if p.iou is not None and p.visible >= 0.99]
        _write_json(
            out / "meta" / f"{stem}.json",
            {
                "schema": labels.IMAGE_SCHEMA,
                "scene_id": scene.scene_id,
                "image": f"images/{stem}.jpg",
                "site": scene.site,
                "profile": scene.profile,
                "shot": asdict(shot),
                "gsd_cm": _gsd(prof, shot),
                "sim_time_ns": raw.get("sim_time_ns"),
                "day_of_year": scene.day_of_year,
                "solar_hour": scene.solar_hour,
                "sun": asdict(scene.sun),
                "weather": scene.weather,
                "camera": {
                    **shot.camera_pose().as_dict(),
                    "width": cam.width,
                    "height": cam.height,
                    "hfov_rad": cam.hfov_rad,
                },
                "brightness": {"mean": round(mean, 1), "std": round(std, 1)},
                "develop": developed,
                "people": [
                    {
                        **p.as_dict(),
                        "shown": shown.get(p.name),
                        "labelled": p.labelled and _shown_enough(shown.get(p.name)),
                    }
                    for p in fac.people
                ],
                "boxes_xyxy": [[round(v, 2) for v in (b.x0, b.y0, b.x1, b.y1)] for b in boxes],
                "label_source": (
                    "bounding-box camera (visible part), paired with the projected facit; "
                    f"people with less than {MIN_SHOWN:g} of their body seen are left out"
                ),
                "credits": coco.data["info"]["credits"],
            },
        )
    coco.write(out / "coco.json")
    sizes.sort()
    q = lambda p: round(sizes[int(p * (len(sizes) - 1))], 1) if sizes else None  # noqa: E731
    reasons = []
    if tally["pictures"] == 0:
        reasons.append("inga bilder")
    elif tally["black"] > 0.1 * (tally["pictures"] + tally["black"]):
        reasons.append(f"{tally['black']} svarta bilder")
    if tally["pictures"] and tally["labels"] == 0:
        reasons.append("inga personer i bild")
    if ious and float(np.median(ious)) < 0.3:
        reasons.append(f"facit och etikettkamera träffar dåligt (median-IoU {np.median(ious):.2f})")
    return {
        **tally,
        "box_long_side_px": {"p10": q(0.1), "p50": q(0.5), "p90": q(0.9)},
        "open_iou_median": round(float(np.median(ious)), 3) if ious else None,
        "ok": not reasons,
        "reasons": reasons,
    }


def ground_patches(
    scene: Scene, ground: terrain.Ground, instance: int, seed: int
) -> list[ground_mod.Patch]:
    """The detailed ground under every shot of the scene."""
    prof = PROFILES[scene.profile]
    feet = []
    for sh in scene.shots:
        z = float(ground.at(np.array([sh.east_m]), np.array([sh.north_m]))[0])
        poly = ground_mod.footprint(
            sh.east_m, sh.north_m, sh.z_m - z, prof.hfov_deg, prof.height / prof.width, sh.yaw_deg
        )
        feet.append((poly, prof.gsd_cm(sh.height_m)))
    return ground_mod.plan(feet, scene.profile, instance, seed)


def run_scene(
    seed: int,
    site_name: str,
    out_root: Path,
    instance: int = 6,
    gi: str = "off",
    shots: int = 0,
    repo: Path = paths.REPO,
    say: Callable[[str], None] = print,
    profile: str | None = None,
) -> dict[str, Any]:
    """Draw, render and finalize one scene; returns its ``scene.json``."""
    started = time.monotonic()
    s = site.load(site_name, repo)
    terrain.build(s, repo, progress=lambda m: say(f"  {m}"))
    ground = terrain.load_ground(s, repo)
    land = land_of(s, ground)
    scene = draw(seed, site_name, land, s.lat, shots, profile)
    out = out_root / scene.scene_id
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = scene.as_dict()
    _write_json(out / "draw.json", body)
    for p in scene.people:
        if p.look:
            model, top, bottom = p.look.split("-")
            looks.make(model, top, bottom, repo)
    patches: list[ground_mod.Patch] = []
    if ground_mod.textures_ready(repo):
        patches = ground_patches(scene, ground, instance, seed)
        ground_mod.make(patches, s, ground, repo, say)
        scene = replace(
            scene,
            people=tuple(
                replace(p, up_m=ground_mod.lift_at(patches, p.east_m, p.north_m))
                for p in scene.people
            ),
        )
    else:
        say(
            "Obs: inga markbilder, marken blir den målade. "
            "Hämta dem med: just ground-pictures-fetch"
        )
    body["ground_patches"] = [
        {
            "east_m": p.east_m,
            "north_m": p.north_m,
            "side_m": p.side_m,
            "cm_per_px": round(p.cell_m * 100, 2),
        }
        for p in patches
    ]
    rig = Rig(instance, scene.profile)
    world_path = site_world.render(
        site_name,
        instance,
        repo,
        placed=[*scene.people, *patches, rig],
        stills=False,
        video=False,
        light=scene.light,
        gi=gi,
    )
    truth = json.loads(world.truth_path(s.world, instance, repo).read_text(encoding="utf-8"))
    env = gz_env(instance, repo)
    shots_dir = out / ".shots"
    (out / ".poses.json").write_text(json.dumps([sh.gz() for sh in scene.shots]), encoding="utf-8")
    say(
        f"scen {scene.scene_id}: {len(scene.people)} personer, {len(scene.shots)} bilder, "
        f"kamera {scene.profile}, {scene.weather}, sol {scene.sun.elevation_deg:.0f} grader"
    )
    status, error = "ok", None
    gz_log = (logs / "gz.log").open("w", encoding="utf-8")
    shots_log = logs / "shots.log"
    gz_proc = subprocess.Popen(  # noqa: S603 - fixed argv, our own world file
        ["gz", "sim", "-s", "-r", "--headless-rendering", "-v2", str(world_path)],  # noqa: S607
        env=env,
        stdout=gz_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # its own group, so its renderer goes with it
    )
    try:
        script = Path(world.__file__).with_name("gz_stills.py")
        argv = [
            gazebo.GZ_PYTHON,
            str(script),
            "shots",
            "--world",
            s.world,
            "--model",
            rig.name,
            "--poses",
            str(out / ".poses.json"),
            "--image-topic",
            f"{rig.topic_prefix}/stills",
            "--boxes-topic",
            f"{rig.topic_prefix}/boxes",
            "--labels-topic",
            f"{rig.topic_prefix}/labels/labels_map",
            "--out",
            str(shots_dir),
            "--keep-going",
            "--start-wait",
            str(GZ_START_S),
        ]
        with shots_log.open("w", encoding="utf-8") as log:
            done = subprocess.run(  # noqa: S603 - fixed argv
                argv,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=GZ_START_S + SHOT_BUDGET_S * len(scene.shots),
                check=False,
            )
        if done.returncode != 0:
            status, error = (
                "error",
                f"gz_stills slutade med kod {done.returncode} (se logs/shots.log)",
            )
    except subprocess.TimeoutExpired:
        status, error = "error", "bilderna tog för lång tid (se logs/shots.log)"
    finally:
        procs.stop_pid(gz_proc.pid, gz_proc)
        gz_log.close()
    result: dict[str, Any] = {}
    if shots_dir.is_dir():
        try:
            result = finalize(scene, truth, shots_dir, out)
        except (StillsError, OSError, ValueError) as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
    if status == "ok" and not result.get("ok"):
        status, error = "failed", "; ".join(result.get("reasons", [])) or "kontrollen föll"
    shutil.rmtree(shots_dir, ignore_errors=True)
    (out / ".poses.json").unlink(missing_ok=True)
    if status != "ok":
        for sub in ("images", "labels", "meta"):
            shutil.rmtree(out / sub, ignore_errors=True)
        (out / "coco.json").unlink(missing_ok=True)
    body |= {
        "status": status,
        "error": error,
        "result": result,
        "gi": gi,
        "seconds": round(time.monotonic() - started, 1),
        "finished_utc": dt.datetime.now(dt.UTC).isoformat(),
        "ground_texture": truth.get("ground_texture", "painted"),
    }
    _write_json(out / "scene.json", body)
    return body


def main(argv: list[str] | None = None) -> int:
    values = _args.parse(
        list(sys.argv[2:] if argv is None else argv), ALLOWED_KEYS, "sd_sim.farm stills"
    )
    try:
        scenes = int(values.get("scenes", "1"))
        seed = int(values.get("seed", str(random.SystemRandom().randrange(1, 2**31))))
        shots = int(values.get("shots", "0"))
        instance = int(values.get("instance", "6"))
    except ValueError:
        _args.fail("sd_sim.farm stills: scenes, seed, shots och instance ska vara heltal")
    gi = values.get("gi", "off")
    if gi not in site_world.GI_MODES:
        _args.fail(f"sd_sim.farm stills: gi ska vara en av {', '.join(site_world.GI_MODES)}")
    site_name = values.get("site", DEFAULT_SITE)
    profile = values.get("profile")
    if profile is not None and profile not in PROFILES:
        _args.fail(f"sd_sim.farm stills: profile ska vara en av {', '.join(PROFILES)}")
    out_root = Path(values.get("out", str(paths.REPO / DEFAULT_OUT)))
    missing = people.missing(paths.REPO, os.environ)
    if missing:
        _args.fail(f"personmodellerna saknas ({', '.join(missing)}). Kör: just sim-farm-setup")
    # Stopped from outside (kill, a timeout): leave through ``finally`` so Gazebo stops too.
    signal.signal(signal.SIGTERM, lambda _sig, _frame: sys.exit(143))
    failures = 0
    total = 0
    for k in range(scenes):
        body = run_scene(seed + k, site_name, out_root, instance, gi, shots, profile=profile)
        res = body.get("result", {})
        total += int(res.get("pictures", 0)) if body["status"] == "ok" else 0
        print(  # noqa: T201 - command-line output
            f"{body['scene_id']}: {body['status']}"
            + (f" ({body['error']})" if body["error"] else "")
            + f", {res.get('pictures', 0)} bilder, {res.get('labels', 0)} personrutor, "
            f"{body['seconds']:.0f} s"
        )
        failures += body["status"] != "ok"
    print(f"klart: {total} bilder i {scenes - failures} av {scenes} scener till {out_root}")  # noqa: T201
    return 0 if failures < scenes else 1
