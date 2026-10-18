"""Render the Gazebo model ``sd_quad14`` and the world ``sd_lake_forest`` for one instance.

The templates live in ``simulation/gazebo/``. ``render_model(instance)`` writes
``build/sim/models/sd_quad14_i<N>/model.sdf`` and ``model.config``; ``render_world(name,
instance)`` writes ``build/sim/worlds/<name>_i<N>.sdf`` (which includes
``model://sd_quad14_i<N>``) and ``build/sim/worlds/<name>.geojson`` with the lake, the path
and the forest band in WGS84. Scenario objects (``sd_sim.objects``) are placed on the
ground and their facit is written to ``build/sim/worlds/<name>_i<N>.truth.json``.
Everything is deterministic (fixed seed) and offline: no template may name a remote URI.

The airframe's mass and propeller size come from ``hardware/drone.yaml`` (read and
validated by ``sd_hwcalc``). In Gazebo each rotor is a joint that the ArduPilot plugin
spins; two LiftDrag surfaces per rotor turn its speed into thrust. Their area is chosen
so that full rotor speed lifts ``THRUST_TO_WEIGHT`` times the all-up weight.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2

from sd_hwcalc.calc import compute
from sd_hwcalc.model import Drone, load_drone
from sd_safety.targets import Plane
from sd_sim import fixture, objects, paths, ports

GAZEBO_DIR = paths.REPO / "simulation" / "gazebo"
DRONE_YAML = paths.REPO / "hardware" / "drone.yaml"
MODEL_NAME = "sd_quad14"
WORLD_NAME = "sd_lake_forest"
SEED = 42

# The vehicle.
ARM_M = 0.29  # half the 0.58 m motor-to-motor diagonal
ROTOR_XY_M = ARM_M / math.sqrt(2)  # 0.205 m in x and y for an X frame
INERTIA_KG_M2 = (0.045, 0.045, 0.080)  # mass in the middle of the frame; CAD values later
IMU_MASS_KG = 0.01
# Body origin above the ground so the landing legs (0.2 m below it) just touch it. The
# world's <include> pose replaces the model's own pose, so both use this height.
SPAWN_Z_M = 0.2
MAX_ROTOR_RAD_S = 650.0  # about 6 200 rpm for a 14-inch propeller
THRUST_TO_WEIGHT = 2.0
# LiftDrag surfaces: lift = 0.5 * rho * area * cl * v^2 with cl = CLA * angle of attack and
# v = rotor speed * BLADE_CP_M. The angle of attack is the blade's fixed pitch
# BLADE_A0_RAD; the area follows from the thrust each rotor must give at full speed.
AIR_DENSITY = 1.2041
BLADE_CP_M = 0.084
CLA = 4.25
BLADE_A0_RAD = 0.3
CAMERA = {"width": 640, "height": 480, "fps": 15, "hfov_deg": 80.0}
# The stills camera a search scenario adds (the real camera's 66 degrees; 1280 x 720 in CI).
STILLS = {"width": 1280, "height": 720, "hfov_deg": 66.0, "rate_hz": 1}

# The world, in metres east/north of the landing pad.
GROUND_SIZE_M = 1200.0
PAD_SIZE_M = 3.0
LAKE_CENTER = (150.0, 120.0)
LAKE_SEMI_AXES = (90.0, 60.0)
LAKE_POINTS = 48
PATH_WIDTH_M = 3.0
PATH_POINTS = ((0.0, 0.0), (-50.0, 150.0), (-120.0, 300.0))
FOREST_X = (-400.0, -200.0)
FOREST_Y = (-300.0, 400.0)
FOREST_TREES = 250
TREE_HEIGHT_M = (15.0, 25.0)
TRUNK_RADIUS_M = 0.2

Xy = tuple[float, float]


@dataclass(frozen=True)
class Tree:
    name: str
    x: float
    y: float
    height: float


def blade_area_m2(total_mass_kg: float) -> float:
    """LiftDrag area per surface so that full rotor speed gives THRUST_TO_WEIGHT x weight.

    Each rotor has two surfaces; the lift of one is 0.5 * rho * area * CLA * a0 * v^2.
    """
    thrust_per_surface = THRUST_TO_WEIGHT * total_mass_kg * 9.80665 / 4 / 2
    v = MAX_ROTOR_RAD_S * BLADE_CP_M
    return thrust_per_surface / (0.5 * AIR_DENSITY * CLA * BLADE_A0_RAD * v * v)


def stills_topic_prefix(instance: int) -> str:
    """Where the instance's stills and labelled boxes are published (Gazebo Transport)."""
    return f"/sd_i{instance}"


def model_context(
    instance: int, drone: Drone | None = None, *, stills: bool = False, video: bool = True
) -> dict[str, Any]:
    """``stills`` adds the stills camera and its bounding-box camera; ``video = False``
    leaves out the video camera (it costs rendering time and a search does not need it)."""
    drone = load_drone(DRONE_YAML) if drone is None else drone
    if drone.motors.count != 4:
        raise ValueError("sd_quad14 är en quad: fyra motorer och fyra propellrar")
    total = compute(drone).total_mass_kg
    rotor_mass = drone.props.mass_kg
    camera_mass = drone.payload_mass_kg
    base_mass = total - IMU_MASS_KG - 4 * rotor_mass - camera_mass
    prop_radius = drone.props.diameter_in * 0.0254 / 2
    r = ROTOR_XY_M
    # ArduPilot quad X order: front right, back left (both CCW), front left, back right (CW).
    rotors: list[dict[str, Any]] = [
        {"x": r, "y": -r, "ccw": True},
        {"x": -r, "y": r, "ccw": True},
        {"x": r, "y": r, "ccw": False},
        {"x": -r, "y": -r, "ccw": False},
    ]
    for rotor in rotors:
        rotor["yaw"] = round(math.atan2(rotor["y"], rotor["x"]), 6)
        # A CCW rotor spins with positive joint speed; its blades meet the air along +y
        # at +x. A CW rotor mirrors both.
        rotor["sign"] = 1 if rotor["ccw"] else -1
    ports_i = ports.for_instance(instance)
    return {
        "name": f"{MODEL_NAME}_i{instance}",
        "spawn_z": SPAWN_Z_M,
        "instance": instance,
        "fdm_port_in": ports_i["sitl_json_fdm"],
        "video_port": ports_i["video_raw_udp"],
        "base_mass": round(base_mass, 6),
        "inertia": INERTIA_KG_M2,
        "imu_mass": IMU_MASS_KG,
        "rotor_mass": rotor_mass,
        # A thin disc of the propeller's radius around z.
        "rotor_ixx": round(rotor_mass * prop_radius**2 / 4, 9),
        "rotor_izz": round(rotor_mass * prop_radius**2 / 2, 9),
        "prop_radius": round(prop_radius, 4),
        "camera_mass": camera_mass,
        "rotors": rotors,
        "multiplier": MAX_ROTOR_RAD_S,
        "blade_area": round(blade_area_m2(total), 6),
        "blade_cp": BLADE_CP_M,
        "blade_a0": BLADE_A0_RAD,
        "cla": CLA,
        "air_density": AIR_DENSITY,
        "camera": {**CAMERA, "hfov_rad": round(math.radians(CAMERA["hfov_deg"]), 6)},
        "video": video,
        "stills": (
            {
                **STILLS,
                "hfov_rad": round(math.radians(STILLS["hfov_deg"]), 6),
                "topic_prefix": stills_topic_prefix(instance),
            }
            if stills
            else None
        ),
    }


def lake_polygon() -> list[Xy]:
    cx, cy = LAKE_CENTER
    ax, ay = LAKE_SEMI_AXES
    return [
        (
            round(cx + ax * math.cos(2 * math.pi * i / LAKE_POINTS), 3),
            round(cy + ay * math.sin(2 * math.pi * i / LAKE_POINTS), 3),
        )
        for i in range(LAKE_POINTS)
    ]


def path_segments() -> list[dict[str, float]]:
    out = []
    for (x0, y0), (x1, y1) in itertools.pairwise(PATH_POINTS):
        out.append(
            {
                "x": (x0 + x1) / 2,
                "y": (y0 + y1) / 2,
                "length": math.hypot(x1 - x0, y1 - y0),
                "yaw": math.atan2(y1 - y0, x1 - x0),
            }
        )
    return out


def trees(seed: int = SEED) -> list[Tree]:
    rng = random.Random(seed)  # noqa: S311 - scenery, not security
    return [
        Tree(
            f"tree_{i:03d}",
            round(rng.uniform(*FOREST_X), 2),
            round(rng.uniform(*FOREST_Y), 2),
            round(rng.uniform(*TREE_HEIGHT_M), 2),
        )
        for i in range(FOREST_TREES)
    ]


def ground_z(east_m: float, north_m: float) -> float:
    """Height of the ground above home at a point; ``sd_lake_forest`` is flat."""
    del east_m, north_m
    return 0.0


def object_pose(obj: objects.WorldObject) -> objects.Pose:
    return objects.Pose(
        obj.east_m,
        obj.north_m,
        ground_z(obj.east_m, obj.north_m) + obj.up_m,
        objects.yaw_enu(obj.yaw_deg),
    )


LABEL_PLUGIN = "gz-sim-label-system"
REMOTE = re.compile(r"(?i)https?://|fuel\.")


def checked_fragment(obj: objects.WorldObject) -> str:
    """The object's SDF, refused unless it is one ``<model>``/``<actor>`` named like the
    object and names no remote URI (Gazebo would download it)."""
    text = obj.sdf(object_pose(obj))
    try:
        root = ET.fromstring(text)  # noqa: S314 - SDF built by our own object classes
    except ET.ParseError as exc:
        raise objects.ObjectError(f"{obj.name}: SDF går inte att läsa ({exc})") from exc
    if root.tag not in ("model", "actor") or root.get("name") != obj.name:
        raise objects.ObjectError(
            f"{obj.name}: SDF ska vara en <model> eller <actor> med objektets namn"
        )
    for uri in root.iter("uri"):
        if not (uri.text or "").strip().startswith("model://"):
            raise objects.ObjectError(f"{obj.name}: bara model://-adresser är tillåtna")
    for el in root.iter():
        if el.tag == "plugin" and el.get("filename") != LABEL_PLUGIN:
            raise objects.ObjectError(f"{obj.name}: bara Label-pluginen är tillåten")
        names = [el.get("filename") or ""] + ([el.text or ""] if el.tag == "filename" else [])
        if any(REMOTE.search(n) for n in names):
            raise objects.ObjectError(f"{obj.name}: filer ska ligga lokalt, inte på nätet")
    return text


def world_context(
    name: str,
    instance: int,
    fix: fixture.Fixture,
    placed: Sequence[objects.WorldObject] = (),
) -> dict[str, Any]:
    objects.check(placed, GROUND_SIZE_M / 2)
    home = fix.home
    return {
        "name": name,
        "model_uri": f"model://{MODEL_NAME}_i{instance}",
        "home": home,
        # ENU yaw of the vehicle so that it faces the fixture's heading (clockwise from north).
        "vehicle_yaw": round(math.radians(90.0 - home.heading_deg), 6),
        "spawn_z": SPAWN_Z_M,
        "ground_size": GROUND_SIZE_M,
        "pad_size": PAD_SIZE_M,
        "lake": lake_polygon(),
        "path": path_segments(),
        "path_width": PATH_WIDTH_M,
        "trees": trees(),
        "trunk_radius": TRUNK_RADIUS_M,
        "objects": [checked_fragment(obj) for obj in placed],
    }


def _env() -> jinja2.Environment:
    # SDF is XML, so values are escaped like any other XML text.
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(GAZEBO_DIR)),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=True,
    )


def model_sdf(
    instance: int, drone: Drone | None = None, *, stills: bool = False, video: bool = True
) -> str:
    template = _env().get_template(f"models/{MODEL_NAME}/model.sdf.jinja")
    return template.render(**model_context(instance, drone, stills=stills, video=video))


def model_config(instance: int) -> str:
    template = (GAZEBO_DIR / "models" / MODEL_NAME / "model.config").read_text(encoding="utf-8")
    return template.replace(f"<name>{MODEL_NAME}</name>", f"<name>{MODEL_NAME}_i{instance}</name>")


def world_sdf(
    name: str,
    instance: int,
    fix: fixture.Fixture | None = None,
    placed: Sequence[objects.WorldObject] = (),
) -> str:
    fix = fixture.load() if fix is None else fix
    template = _env().get_template(f"worlds/{name}.sdf.jinja")
    return template.render(**world_context(name, instance, fix, placed))


def world_truth(
    name: str, placed: Sequence[objects.WorldObject], fix: fixture.Fixture | None = None
) -> dict[str, Any]:
    """The facit: where every object really is, in the world frame and in WGS84."""
    fix = fixture.load() if fix is None else fix
    plane = Plane(fix.home.lat, fix.home.lon)
    rows = []
    for obj in placed:
        pose = object_pose(obj)
        lat, lon = plane.latlon(pose.x, pose.y)
        rows.append(
            {
                "name": obj.name,
                "east_m": round(pose.x, 3),
                "north_m": round(pose.y, 3),
                "ground_m": round(ground_z(obj.east_m, obj.north_m), 3),
                "up_m": round(obj.up_m, 3),
                "lat": round(lat, 8),
                "lon": round(lon, 8),
                "alt_amsl_m": round(fix.home.alt_amsl_m + pose.z, 3),
                "yaw_deg": round(obj.yaw_deg, 3),
                "label": obj.label,
                "truth": dict(obj.truth),
            }
        )
    return {
        "world": name,
        "home": {"lat": fix.home.lat, "lon": fix.home.lon, "alt_amsl_m": fix.home.alt_amsl_m},
        "objects": rows,
    }


def world_geojson(fix: fixture.Fixture | None = None) -> dict[str, Any]:
    fix = fixture.load() if fix is None else fix
    plane = Plane(fix.home.lat, fix.home.lon)

    def lonlat(points: list[Xy]) -> list[list[float]]:
        out = []
        for x, y in points:
            lat, lon = plane.latlon(x, y)
            out.append([round(lon, 7), round(lat, 7)])
        return out

    lake = lake_polygon()
    forest = [
        (FOREST_X[0], FOREST_Y[0]),
        (FOREST_X[1], FOREST_Y[0]),
        (FOREST_X[1], FOREST_Y[1]),
        (FOREST_X[0], FOREST_Y[1]),
    ]

    def feature(kind: str, geometry: dict[str, Any]) -> dict[str, Any]:
        return {"type": "Feature", "properties": {"kind": kind}, "geometry": geometry}

    return {
        "type": "FeatureCollection",
        "features": [
            feature("lake", {"type": "Polygon", "coordinates": [lonlat([*lake, lake[0]])]}),
            feature("path", {"type": "LineString", "coordinates": lonlat(list(PATH_POINTS))}),
            feature("forest", {"type": "Polygon", "coordinates": [lonlat([*forest, forest[0]])]}),
        ],
    }


def render_model(
    instance: int, repo: Path = paths.REPO, *, stills: bool = False, video: bool = True
) -> Path:
    """Write the instance's model; returns its folder (``model://sd_quad14_i<N>``)."""
    out = repo / "build" / "sim" / "models" / f"{MODEL_NAME}_i{instance}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.sdf").write_text(
        model_sdf(instance, stills=stills, video=video), encoding="utf-8"
    )
    (out / "model.config").write_text(model_config(instance), encoding="utf-8")
    return out


def truth_path(name: str, instance: int, repo: Path = paths.REPO) -> Path:
    return repo / "build" / "sim" / "worlds" / f"{name}_i{instance}.truth.json"


def clear_truth(instance: int, repo: Path = paths.REPO) -> None:
    """Remove the instance's facit files, for a run that renders no world."""
    for stale in (repo / "build" / "sim" / "worlds").glob(f"*_i{instance}.truth.json"):
        stale.unlink()


def render_world(
    name: str,
    instance: int,
    repo: Path = paths.REPO,
    placed: Sequence[objects.WorldObject] = (),
    *,
    stills: bool = False,
    video: bool = True,
) -> Path:
    """Write the instance's world and model, the objects' facit and the GeoJSON layer.
    ``stills`` and ``video`` choose the vehicle's cameras.

    The GeoJSON file has no instance suffix: every instance writes the same content, since
    the layer only depends on the fixture's home and the fixed scenery. The facit
    (``<name>_i<N>.truth.json``) is per instance, since each may run other objects."""
    render_model(instance, repo, stills=stills, video=video)
    fix = fixture.load()
    out = repo / "build" / "sim" / "worlds"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_i{instance}.sdf"
    sdf = world_sdf(name, instance, fix, placed)
    truth = json.dumps(world_truth(name, placed, fix), indent=1) + "\n"
    path.write_text(sdf, encoding="utf-8")
    truth_path(name, instance, repo).write_text(truth, encoding="utf-8")
    (out / f"{name}.geojson").write_text(
        json.dumps(world_geojson(fix), indent=1) + "\n", encoding="utf-8"
    )
    return path
