"""Render detector test pictures in the Gazebo world: ``python -m sd_sim.render_stills``.

    uv run --package sd-sim python -m sd_sim.render_stills [out=DIR] [seed=N] [instance=N]

Builds the world ``sd_lake_forest`` with a scenario's targets (``sd_sim.targets``) and a
free stills camera (``StillRig``: the vehicle's stills camera plus a bounding-box camera
with the same picture), all placed with ``world.render_world(..., placed=...)``; the
vehicle stands on the pad as in every world. Gazebo runs as a headless server;
``gz_stills.py shots`` (system Python) moves the camera to every pose in ``shot_poses``
and saves a picture with the labelled boxes of the same moment. The
pictures become PNG files and ``stills.json`` describes them: the camera, every pose
(Gazebo's frame and the vehicle attitude it equals), the labelled boxes by class and the
ground truth of the objects.

The default output is the detector's test fixtures,
``software/detector/tests/testdata/stills/`` (run ``just fmt`` afterwards so the JSON
file has the repository's format). Needs Gazebo Harmonic (``just sim-setup``).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sd_detect.classes import default_registry
from sd_safety.targets import Plane
from sd_sim import _args, fixture, gazebo, objects, paths, targets, world

__all__ = ["CAMERA", "RIG_MODEL", "StillRig", "boxes_by_class", "render", "shot_poses"]

RIG_MODEL = "sd_still_rig"
RIG_WORLD = "sd_lake_forest"
RIG_TEMPLATE = "objects/sd_still_rig.sdf.jinja"
# The stills camera in CI: 1280 x 720, 66 degrees across (the real camera's field of view).
CAMERA = {"width": 1280, "height": 720, "hfov_deg": 66.0}
DEFAULT_OUT = paths.REPO / "software" / "detector" / "tests" / "testdata" / "stills"
DEFAULT_SEED = 11
GZ_START_S = 120.0
SHOTS_TIMEOUT_S = 600.0
ALLOWED_KEYS = ("out", "seed", "instance")


def shot_poses(scene: list[targets.Target]) -> list[dict[str, Any]]:
    """Camera poses over the scenario: every target from 20, 30 and 40 m (one near the
    picture's edge, one tilted), the decoys, context objects and places with nothing red.

    Each pose is ``name, east, north, height, compass yaw, roll, pitch`` (degrees, the
    vehicle's attitude convention); ``gz`` holds the same pose in Gazebo's frame."""
    plans: list[tuple[str, float, float, float, float, float, float]] = []
    blocks = [t for t in scene if t.cls == "red_block"]
    for i, b in enumerate(blocks):
        e, n = b.east_m, b.north_m
        plans += [
            (f"block{i}_h20", e + 3.0, n - 2.0, 20.0, 15.0 + 40 * i, 0.0, 0.0),
            (f"block{i}_h30", e - 4.0, n + 5.0, 30.0, 100.0 + 40 * i, 0.0, 0.0),
            (f"block{i}_h40", e + 8.0, n + 3.0, 40.0, 200.0 + 40 * i, 0.0, 0.0),
        ]
    b0 = blocks[0]
    plans += [
        ("block0_edge", b0.east_m - 21.0, b0.north_m + 2.0, 35.0, 0.0, 0.0, 0.0),
        ("block0_tilted", b0.east_m + 2.0, b0.north_m - 3.0, 35.0, 300.0, 6.0, -8.0),
    ]
    for t in scene:
        if t.role == "decoy" or t.cls == "car":
            plans.append((f"{t.name}_h30", t.east_m + 2.0, t.north_m - 1.0, 30.0, 45.0, 0.0, 0.0))
    road = next(t for t in scene if t.cls == "road")
    plans += [
        ("road_h30", road.east_m + 20.0, road.north_m, 30.0, 0.0, 0.0, 0.0),
        ("overview_h60", road.east_m, road.north_m + 8.0, 60.0, 0.0, 0.0, 0.0),
        ("empty_forest_h40", -205.0, 40.0, 40.0, 90.0, 0.0, 0.0),
        ("empty_lake_h40", 150.0, 120.0, 40.0, 0.0, 0.0, 0.0),
        ("empty_path_h30", -40.0, 125.0, 30.0, 20.0, 0.0, 0.0),
        ("pad_h25", 0.0, 0.0, 25.0, 353.0, 0.0, 0.0),  # the vehicle on the pad, nothing red
    ]
    out = []
    for name, e, n, h, yaw, roll, pitch in plans:
        out.append(
            {
                "name": name,
                "east_m": round(e, 2),
                "north_m": round(n, 2),
                "height_m": h,
                "attitude": {"roll_deg": roll, "pitch_deg": pitch, "yaw_deg": yaw},
                # Gazebo: x east, y north, z up, rotations about x, y, z (front-left-up).
                "gz": {
                    "name": name,
                    "x": round(e, 2),
                    "y": round(n, 2),
                    "z": h,
                    "roll": math.radians(roll),
                    "pitch": -math.radians(pitch),
                    "yaw": math.radians(90.0 - yaw),
                },
            }
        )
    return out


def boxes_by_class(raw: list[dict[str, Any]], labels: dict[int, str]) -> list[dict[str, Any]]:
    """The bounding-box camera's corner boxes as class id and x, y, w, h; labels that are
    not in the register are left out."""
    return [
        {
            "cls": labels[b["label"]],
            "x": b["x0"],
            "y": b["y0"],
            "w": b["x1"] - b["x0"],
            "h": b["y1"] - b["y0"],
        }
        for b in raw
        if b["label"] in labels
    ]


@dataclass(frozen=True)
class StillRig:
    """The free stills camera as a ``WorldObject``. It starts a few metres up beside the
    search box; ``gz_stills.py shots`` moves it to each shot's pose."""

    instance: int
    east_m: float = 0.0
    north_m: float = 120.0
    up_m: float = 5.0
    yaw_deg: float = 0.0
    label: int = 0
    footprint_m: float = 0.1
    truth: dict[str, Any] = field(default_factory=lambda: {"role": "camera"})

    @property
    def name(self) -> str:
        return f"{RIG_MODEL}_i{self.instance}"

    @property
    def topic_prefix(self) -> str:
        return f"/sd_rig_i{self.instance}"

    def sdf(self, pose: objects.Pose) -> str:
        template = world._env().get_template(RIG_TEMPLATE)
        return template.render(
            name=self.name,
            pose=pose.sdf(),
            topic_prefix=self.topic_prefix,
            width=CAMERA["width"],
            height=CAMERA["height"],
            hfov_rad=round(math.radians(CAMERA["hfov_deg"]), 6),
            rate_hz=2,
        ).strip()


def object_rows(objs: list[targets.Target], fix: fixture.Fixture) -> list[dict[str, Any]]:
    """The scenario's objects for ``stills.json``: every field and the WGS84 position."""
    plane = Plane(fix.home.lat, fix.home.lon)
    rows = []
    for t in objs:
        lat, lon = plane.latlon(t.east_m, t.north_m)
        row = {k: v for k, v in asdict(t).items() if k != "up_m"}
        rows.append(row | {"lat": round(lat, 8), "lon": round(lon, 8)})
    return rows


def _gz_env(instance: int, repo: Path) -> dict[str, str]:
    env = dict(os.environ)
    models = str(repo / "build" / "sim" / "models")
    env.update(
        {
            "GZ_IP": "127.0.0.1",
            "GZ_PARTITION": f"sd_rig_i{instance}",
            "GZ_SIM_RESOURCE_PATH": os.pathsep.join(
                p for p in (models, env.get("GZ_SIM_RESOURCE_PATH", "")) if p
            ),
        }
    )
    return env


def _to_png(ppm: Path, png: Path) -> None:
    from PIL import Image  # test tooling only (dev dependency)

    with Image.open(ppm) as im:
        im.save(png, optimize=True)


def render(
    out: Path = DEFAULT_OUT, seed: int = DEFAULT_SEED, instance: int = 7, repo: Path = paths.REPO
) -> Path:
    fix = fixture.load()
    scene = targets.layout("SK-RED", seed, fix)
    rig = StillRig(instance)
    world_path = world.render_world(RIG_WORLD, instance, repo, placed=[*scene, rig])
    poses = shot_poses(scene)
    env = _gz_env(instance, repo)
    labels = {n: spec.id for n, spec in default_registry().by_gz_label().items()}
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sd_stills_") as tmp_name:
        tmp = Path(tmp_name)
        (tmp / "poses.json").write_text(json.dumps([p["gz"] for p in poses]), encoding="utf-8")
        gz = subprocess.Popen(  # noqa: S603 - fixed argv, our own world file
            ["gz", "sim", "-s", "-r", "--headless-rendering", "-v1", str(world_path)],  # noqa: S607
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(3.0)
            script = Path(__file__).with_name("gz_stills.py")
            subprocess.run(  # noqa: S603 - fixed argv
                [
                    gazebo.GZ_PYTHON,
                    str(script),
                    "shots",
                    "--world",
                    RIG_WORLD,
                    "--model",
                    rig.name,
                    "--poses",
                    str(tmp / "poses.json"),
                    "--image-topic",
                    f"{rig.topic_prefix}/stills",
                    "--boxes-topic",
                    f"{rig.topic_prefix}/boxes",
                    "--out",
                    str(tmp / "shots"),
                ],
                env=env,
                check=True,
                timeout=SHOTS_TIMEOUT_S + GZ_START_S,
            )
        finally:
            gz.terminate()
            try:
                gz.wait(10)
            except subprocess.TimeoutExpired:
                gz.kill()
        shots = []
        for pose in poses:
            name = pose["name"]
            meta = json.loads((tmp / "shots" / f"{name}.json").read_text(encoding="utf-8"))
            _to_png(tmp / "shots" / f"{name}.ppm", out / f"{name}.png")
            boxes = boxes_by_class(meta["boxes"] or [], labels)
            shots.append(
                {k: v for k, v in pose.items() if k != "gz"}
                | {"file": f"{name}.png", "boxes": boxes}
            )
    doc = {
        "about": (
            "Pictures rendered by Gazebo Harmonic (ogre2, headless) in the world "
            "sd_lake_forest with the SK-RED targets; regenerate with "
            "`uv run --package sd-sim python -m sd_sim.render_stills`."
        ),
        "camera": CAMERA,
        "scenario": "SK-RED",
        "seed": seed,
        "objects": object_rows(scene, fix),
        "shots": shots,
    }
    (out / "stills.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    values = _args.parse(
        list(sys.argv[1:] if argv is None else argv), ALLOWED_KEYS, "sd_sim.render_stills"
    )
    out = Path(values.get("out", str(DEFAULT_OUT)))
    path = render(out, int(values.get("seed", DEFAULT_SEED)), int(values.get("instance", "7")))
    sys.stdout.write(f"{path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
