"""Search targets and labelled objects in the Gazebo world, with their ground truth.

``layout(scenario, seed)`` places the objects of a search scenario on seeded positions
inside the demo's search box: red blocks (the targets), a red car and a red house as
decoys, and context objects (a car, a building, a road) for the classes the simulator's
labels report. ``render_targets`` writes every object as a static Gazebo model of its own
(``build/sim/models/sd_targets_i<N>_<object>``, included by the world: the bounding-box
camera gives one box per top-level model) and the ground truth next to the world
(``build/sim/worlds/ground_truth_i<N>.json``): every object's class, place in metres
east/north of home and in WGS84, size and label number.

Every object carries the simulator's label system with its class's ``gz_label`` from the
class register, so the bounding-box camera can name it. Only the evaluator and the
simulator-only oracle source read the labels and the ground truth; the colour detector
sees the rendered pixels and nothing else.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sd_detect.classes import ClassRegistry, default_registry
from sd_safety.targets import Plane
from sd_sim import fixture, paths, world

__all__ = [
    "SCENARIOS",
    "TARGETS_MODEL",
    "Target",
    "ground_truth",
    "layout",
    "render_targets",
    "targets_sdf",
]

TARGETS_MODEL = "sd_targets"

RED = (0.80, 0.05, 0.05)
FALU_RED = (0.50, 0.11, 0.08)  # the Swedish red house paint: a second red decoy
CAR_BLUE = (0.10, 0.20, 0.55)
ASPHALT = (0.25, 0.25, 0.27)
ROOF_GREY = (0.35, 0.35, 0.38)

BLOCK_SIZE_M = (0.5, 0.5, 0.5)
CAR_SIZE_M = (4.5, 1.8, 1.5)
BUILDING_SIZE_M = (10.0, 7.0, 5.0)
ROAD_WIDTH_M = 6.0
MARGIN_M = 10.0  # targets stay this far inside the search box
MIN_GAP_M = 12.0  # between any two objects' centres
ROAD_CLEAR_M = 7.0  # targets stay this far from the road's centre line

SCENARIOS = ("SK-RED",)


@dataclass(frozen=True)
class Target:
    name: str
    cls: str  # a class id from the register
    role: str  # "target" | "decoy" | "context"
    east_m: float
    north_m: float
    yaw_deg: float  # compass degrees of the long side, clockwise from north
    size_m: tuple[float, float, float]  # length, width, height
    rgb: tuple[float, float, float]
    gz_label: int


def _box(f: fixture.Fixture) -> tuple[float, float, float, float]:
    (x0, y0), _, (x1, y1), _ = f.survey_box.corners()
    return x0, y0, x1, y1


def layout(
    scenario: str = "SK-RED",
    seed: int = 1,
    fix: fixture.Fixture | None = None,
    registry: ClassRegistry | None = None,
) -> list[Target]:
    """The objects of ``scenario`` for ``seed``: the same seed always gives the same world."""
    if scenario not in SCENARIOS:
        raise ValueError(f"okänt scenario {scenario!r} (finns: {', '.join(SCENARIOS)})")
    fix = fixture.load() if fix is None else fix
    labels = {spec.id: spec.params.get("gz_label", 0) for spec in (registry or default_registry())}
    rng = random.Random(f"{scenario}:{seed}")  # noqa: S311 - scenery, not security
    x0, y0, x1, y1 = _box(fix)
    road_north = round(rng.uniform(y0 + 15.0, y1 - 15.0), 2)
    placed: list[Target] = [
        Target(
            "road_0",
            "road",
            "context",
            (x0 + x1) / 2,
            road_north,
            90.0,
            (x1 - x0 + 40.0, ROAD_WIDTH_M, 0.02),
            ASPHALT,
            labels["road"],
        )
    ]

    def free_spot(clear_road: float) -> tuple[float, float]:
        for _ in range(10_000):
            e = rng.uniform(x0 + MARGIN_M, x1 - MARGIN_M)
            n = rng.uniform(y0 + MARGIN_M, y1 - MARGIN_M)
            if abs(n - road_north) < clear_road:
                continue
            if all(math.hypot(e - t.east_m, n - t.north_m) >= MIN_GAP_M for t in placed[1:]):
                return round(e, 2), round(n, 2)
        raise RuntimeError("hittade ingen ledig plats i sökrutan")

    def add(
        name: str,
        cls: str,
        role: str,
        size: tuple[float, float, float],
        rgb: tuple[float, float, float],
        clear_road: float,
    ) -> None:
        e, n = free_spot(clear_road)
        yaw = round(rng.uniform(0.0, 180.0), 1)
        placed.append(Target(name, cls, role, e, n, yaw, size, rgb, labels[cls]))

    for i in range(1 + rng.randrange(3)):
        add(f"red_block_{i}", "red_block", "target", BLOCK_SIZE_M, RED, ROAD_CLEAR_M)
    add("red_car_0", "car", "decoy", CAR_SIZE_M, RED, ROAD_WIDTH_M)
    add("car_1", "car", "context", CAR_SIZE_M, CAR_BLUE, ROAD_WIDTH_M)
    add("building_0", "building", "decoy", BUILDING_SIZE_M, FALU_RED, ROAD_CLEAR_M + 3.0)
    return placed


def targets_sdf(target: Target, name: str) -> str:
    """One object as a static model with its label."""
    row = {
        **asdict(target),
        # Gazebo's yaw is counter-clockwise from east; the box's length lies along x.
        "yaw_rad": round(math.radians(90.0 - target.yaw_deg), 6),
        "z": round(target.size_m[2] / 2, 4),
    }
    template = world._env().get_template(f"models/{TARGETS_MODEL}/model.sdf.jinja")
    return template.render(name=name, t=row, roof=ROOF_GREY)


def ground_truth(
    targets: list[Target], scenario: str, seed: int, fix: fixture.Fixture
) -> dict[str, Any]:
    plane = Plane(fix.home.lat, fix.home.lon)
    objects = []
    for t in targets:
        lat, lon = plane.latlon(t.east_m, t.north_m)
        objects.append({**asdict(t), "lat": round(lat, 8), "lon": round(lon, 8)})
    return {
        "scenario": scenario,
        "seed": seed,
        "home": {"lat": fix.home.lat, "lon": fix.home.lon, "alt_amsl_m": fix.home.alt_amsl_m},
        "search_box_enu": list(_box(fix)),
        "objects": objects,
    }


def render_targets(
    instance: int, scenario: str = "SK-RED", seed: int = 1, repo: Path = paths.REPO
) -> tuple[tuple[str, ...], Path]:
    """Write one model per object and the ground truth; returns the model names (for the
    world's ``includes``) and the ground truth's path."""
    fix = fixture.load()
    targets = layout(scenario, seed, fix)
    models = repo / "build" / "sim" / "models"
    config = (world.GAZEBO_DIR / "models" / TARGETS_MODEL / "model.config").read_text(
        encoding="utf-8"
    )
    prefix = f"{TARGETS_MODEL}_i{instance}_"
    for old in models.glob(f"{prefix}*"):  # a previous seed's objects
        for f in old.iterdir():
            f.unlink()
        old.rmdir()
    names: list[str] = []
    for t in targets:
        name = f"{prefix}{t.name}"
        model_dir = models / name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.sdf").write_text(targets_sdf(t, name), encoding="utf-8")
        (model_dir / "model.config").write_text(
            config.replace(f"<name>{TARGETS_MODEL}</name>", f"<name>{name}</name>"),
            encoding="utf-8",
        )
        names.append(name)
    worlds = repo / "build" / "sim" / "worlds"
    worlds.mkdir(parents=True, exist_ok=True)
    truth = worlds / f"ground_truth_i{instance}.json"
    truth.write_text(
        json.dumps(ground_truth(targets, scenario, seed, fix), indent=1) + "\n", encoding="utf-8"
    )
    return tuple(names), truth
