"""Search targets and labelled objects in the Gazebo world, with their ground truth.

``layout(scenario, seed)`` places the objects of a search scenario on seeded positions
inside the demo's search box: red blocks (the targets), a red car and a red house as
decoys, and context objects (a car, a building, a road) for the classes the simulator's
labels report. Every ``Target`` is a ``sd_sim.objects.WorldObject``: pass the layout to
``world.render_world(..., placed=layout(...))``, which puts each object at world level
as a model of its own (the bounding-box camera gives one box per top-level model) and
writes the facit (``build/sim/worlds/<world>_i<N>.truth.json``) with each object's place
and its ``truth``: class, role, size and colour.

Every object carries the simulator's label system with its class's ``gz_label`` from the
class register, so the bounding-box camera can name it. Only the evaluator and the
simulator-only oracle source read the labels and the ground truth; the colour detector
sees the rendered pixels and nothing else.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sd_detect.classes import ClassRegistry, default_registry
from sd_sim import fixture, objects, world

__all__ = ["SCENARIOS", "TEMPLATE", "Target", "layout", "search_box"]

TEMPLATE = "objects/sd_target.sdf.jinja"

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
    """One object of a scenario; a ``WorldObject`` standing on the ground."""

    name: str
    cls: str  # a class id from the register
    role: str  # "target" | "decoy" | "context"
    east_m: float
    north_m: float
    yaw_deg: float  # compass degrees of the long side, clockwise from north
    size_m: tuple[float, float, float]  # length, width, height
    rgb: tuple[float, float, float]
    gz_label: int
    up_m: float = 0.0

    @property
    def label(self) -> int:
        return self.gz_label

    @property
    def footprint_m(self) -> float:
        return math.hypot(self.size_m[0], self.size_m[1]) / 2

    @property
    def truth(self) -> Mapping[str, Any]:
        return {
            "cls": self.cls,
            "role": self.role,
            "size_m": list(self.size_m),
            "rgb": list(self.rgb),
        }

    def sdf(self, pose: objects.Pose) -> str:
        """The object as one static model with its label, its base on the ground."""
        template = world._env().get_template(TEMPLATE)
        return template.render(t=self, pose=pose.sdf(), roof=ROOF_GREY).strip()


def search_box(f: fixture.Fixture) -> tuple[float, float, float, float]:
    """The fixture's search box in metres east/north of home: x0, y0, x1, y1."""
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
    x0, y0, x1, y1 = search_box(fix)
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
