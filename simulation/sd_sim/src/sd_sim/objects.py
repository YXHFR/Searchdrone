"""Objects placed in a generated world, with their ground truth (the "facit").

This is the hook for scenario code: anything that should appear in the world for the
camera to find (people, decoys, a red test block) is a ``WorldObject``. The world
renderer (``sd_sim.world.render_world(..., placed=...)``) puts each one on the ground
of that world at metres east/north of home, writes its SDF at world level, and writes
``build/sim/worlds/<world>_i<N>.truth.json`` with every object's exact position in the
world frame and in WGS84. Detection tests compare against that file, never against the
scenario file, so the facit is what Gazebo was actually given.

``WorldObject`` is a protocol: scenario code can bring its own classes (a person mesh,
an ``<actor>`` that walks) as long as they have these fields and ``sdf(pose)``; ``sdf``
must escape any text it puts into the XML. The renderer parses every fragment and
refuses one that is not a single ``<model>`` or ``<actor>`` with the object's name or
that names a remote URI. ``Box`` is the built-in one. ``load(path)`` reads a YAML list
of boxes, for ``just sim objects=<file>``::

    - { name: red_block, kind: box, east_m: 10, north_m: 200, size_m: [1, 1, 0.5],
        color: [0.8, 0.05, 0.05], label: 1, truth: { is_target: true } }

``label`` is Gazebo's semantic label (the Label system), which segmentation and
bounding-box cameras report; 0 means none. ``truth`` is free-form and copied as is (it
must be JSON). No object may stand within ``HOME_KEEP_OUT_M`` of home (the landing pad
and the vehicle's spawn point), and every object must lie inside the world.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import yaml

# Names the world templates already use at world level.
RESERVED_NAMES = frozenset({"ground", "landing_pad", "lake", "path", "forest", "vehicle", "sun"})
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
BOX_REQUIRED = {"name", "kind", "east_m", "north_m", "size_m", "color"}
BOX_OPTIONAL = {"up_m", "yaw_deg", "label", "truth"}
# Nothing stands this close to home: the landing pad and the vehicle start there.
HOME_KEEP_OUT_M = 5.0
SIZE_RANGE_M = (0.01, 100.0)
# Scenery stays near the ground, well below survey heights: the origin at most this far
# above or below the ground, and a box's top at most MAX_TOP_M above it.
UP_RANGE_M = (-2.0, 10.0)
MAX_TOP_M = 30.0


class ObjectError(ValueError):
    """An object list does not have the expected shape."""


@dataclass(frozen=True)
class Pose:
    """Where the renderer puts an object, in the world frame (x east, y north, z up)."""

    x: float
    y: float
    z: float
    yaw_rad: float

    def sdf(self) -> str:
        return f"{self.x:.3f} {self.y:.3f} {self.z:.3f} 0 0 {self.yaw_rad:.6f}"


class WorldObject(Protocol):
    """Anything a scenario places in the world.

    ``up_m`` is the height of the object's origin above the ground at (east, north);
    ``yaw_deg`` is clockwise from north, like a compass heading.
    """

    @property
    def name(self) -> str: ...
    @property
    def east_m(self) -> float: ...
    @property
    def north_m(self) -> float: ...
    @property
    def up_m(self) -> float: ...
    @property
    def yaw_deg(self) -> float: ...
    @property
    def label(self) -> int: ...
    @property
    def footprint_m(self) -> float:
        """Horizontal radius around (east, north) that the object covers."""
        ...

    @property
    def truth(self) -> Mapping[str, Any]: ...

    def sdf(self, pose: Pose) -> str:
        """One or more world-level SDF elements (``<model>``, ``<actor>``) at ``pose``."""
        ...


def yaw_enu(yaw_deg: float) -> float:
    """Compass heading (clockwise from north) to the world's yaw (counter-clockwise from east)."""
    return math.radians(90.0 - yaw_deg)


def _xml_number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class Box:
    """A static coloured box standing on the ground (origin at the bottom face's centre)."""

    name: str
    east_m: float
    north_m: float
    size_m: tuple[float, float, float]
    color: tuple[float, float, float]
    up_m: float = 0.0
    yaw_deg: float = 0.0
    label: int = 0
    truth: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        low, high = SIZE_RANGE_M
        if len(self.size_m) != 3 or not all(low <= v <= high for v in self.size_m):
            raise ObjectError(f"{self.name}: size_m ska vara tre tal mellan {low:g} och {high:g}")
        if len(self.color) != 3 or not all(0.0 <= c <= 1.0 for c in self.color):
            raise ObjectError(f"{self.name}: color ska vara tre tal mellan 0 och 1")
        if self.up_m + self.size_m[2] > MAX_TOP_M:
            raise ObjectError(
                f"{self.name}: lådans topp ska vara högst {MAX_TOP_M:g} m över marken"
            )

    @property
    def footprint_m(self) -> float:
        return math.hypot(self.size_m[0], self.size_m[1]) / 2

    def sdf(self, pose: Pose) -> str:
        sx, sy, sz = (_xml_number(v) for v in self.size_m)
        rgb = " ".join(_xml_number(c) for c in self.color)
        label = (
            '\n    <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">'
            f"<label>{self.label}</label></plugin>"
            if self.label
            else ""
        )
        half = _xml_number(self.size_m[2] / 2)
        return f"""<model name="{self.name}">
    <static>true</static>
    <pose>{pose.sdf()}</pose>{label}
    <link name="link">
      <collision name="collision">
        <pose>0 0 {half} 0 0 0</pose>
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
      </collision>
      <visual name="visual">
        <pose>0 0 {half} 0 0 0</pose>
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <material><ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse></material>
      </visual>
    </link>
  </model>"""


def check(objects: Sequence[WorldObject], world_half_m: float = math.inf) -> None:
    """Names must be unique, safe as SDF names and not taken by the world itself; every
    object stands outside the keep-out circle around home and inside the world."""
    seen: set[str] = set()
    for obj in objects:
        if not _NAME.match(obj.name):
            raise ObjectError(f"{obj.name!r}: namnet ska vara bokstäver, siffror och _")
        if obj.name in RESERVED_NAMES or obj.name.startswith("sd_quad14"):
            raise ObjectError(f"{obj.name!r}: namnet används redan av världen")
        if obj.name in seen:
            raise ObjectError(f"{obj.name!r}: namnet finns två gånger")
        seen.add(obj.name)
        values = (obj.east_m, obj.north_m, obj.up_m, obj.yaw_deg)
        if not all(math.isfinite(v) for v in values):
            raise ObjectError(f"{obj.name}: position och riktning ska vara ändliga tal")
        if not UP_RANGE_M[0] <= obj.up_m <= UP_RANGE_M[1]:
            low, high = UP_RANGE_M
            raise ObjectError(f"{obj.name}: up_m ska ligga mellan {low:g} och {high:g}")
        if obj.label < 0:
            raise ObjectError(f"{obj.name}: label ska vara 0 eller större")
        reach = obj.footprint_m
        if math.hypot(obj.east_m, obj.north_m) - reach < HOME_KEEP_OUT_M:
            raise ObjectError(f"{obj.name}: står närmare hem än {HOME_KEEP_OUT_M:g} m")
        if max(abs(obj.east_m), abs(obj.north_m)) + reach > world_half_m:
            raise ObjectError(f"{obj.name}: ligger utanför världen")
        try:
            json.dumps(dict(obj.truth), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ObjectError(f"{obj.name}: truth går inte att skriva som JSON") from exc


def _num(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ObjectError(f"{where}: väntade ett tal, fick {value!r}")
    if not math.isfinite(float(value)):
        raise ObjectError(f"{where}: talet måste vara ändligt")
    return float(value)


def _triple(value: Any, where: str, *, low: float, high: float) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ObjectError(f"{where}: väntade tre tal")
    out = tuple(_num(v, where) for v in value)
    if not all(low <= v <= high for v in out):
        raise ObjectError(f"{where}: talen ska ligga mellan {low:g} och {high:g}")
    return (out[0], out[1], out[2])


def parse(data: Any) -> list[Box]:
    if not isinstance(data, list):
        raise ObjectError("objektfilen ska vara en lista")
    out: list[Box] = []
    for i, raw in enumerate(data):
        where = f"objekt[{i}]"
        if not isinstance(raw, dict):
            raise ObjectError(f"{where}: väntade en tabell")
        unknown = sorted(set(raw) - BOX_REQUIRED - BOX_OPTIONAL)
        if unknown:
            raise ObjectError(f"{where}: okända nycklar {', '.join(unknown)}")
        missing = sorted(BOX_REQUIRED - set(raw))
        if missing:
            raise ObjectError(f"{where}: saknar {', '.join(missing)}")
        if raw["kind"] != "box":
            raise ObjectError(f"{where}: kind ska vara box, inte {raw['kind']!r}")
        label = raw.get("label", 0)
        if isinstance(label, bool) or not isinstance(label, int):
            raise ObjectError(f"{where}.label: väntade ett heltal")
        truth = raw.get("truth", {})
        if not isinstance(truth, dict):
            raise ObjectError(f"{where}.truth: väntade en tabell")
        out.append(
            Box(
                name=str(raw["name"]),
                east_m=_num(raw["east_m"], f"{where}.east_m"),
                north_m=_num(raw["north_m"], f"{where}.north_m"),
                up_m=_num(raw.get("up_m", 0.0), f"{where}.up_m"),
                yaw_deg=_num(raw.get("yaw_deg", 0.0), f"{where}.yaw_deg"),
                size_m=_triple(
                    raw["size_m"], f"{where}.size_m", low=SIZE_RANGE_M[0], high=SIZE_RANGE_M[1]
                ),
                color=_triple(raw["color"], f"{where}.color", low=0.0, high=1.0),
                label=label,
                truth=truth,
            )
        )
    check(out)
    return out


def load(path: Path) -> list[Box]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ObjectError(f"{path}: {exc}") from exc
    return parse(data)
