"""Pictures and measurements of a site world: ``python -m sd_sim.snapshot [world=<name>]``.

Renders the world with extra fixed cameras, runs Gazebo headless until each camera has
saved frames, and writes to ``runs/<run>/snapshots/``:

- ``nadir_40m_road.png``, ``nadir_40m_block.png``, ``nadir_100m.png``: straight down, with
  exactly the vehicle's camera (640 x 480, 80 degrees; ``world.CAMERA``), north up;
- ``oblique.png``: a 1280 x 720 view over the area from the south-west;
- ``metrics.json`` and ``summary.md``: the road's width measured in the 40 m picture
  against its width on the map, and how far the road in the picture lies from the
  map's road line (the map the panel shows comes from the same OpenStreetMap data).

The 40 m road picture is centred on a point of the map's road line, so the painted road
must cross the picture's centre: ``offset_m`` is the distance between the two, and
``width_m`` the asphalt band's width across the road. Exit code 0 when both are within
their limits (10 % and 2 m), 1 otherwise, 2 for bad arguments.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import MultiLineString
from shapely.ops import nearest_points

from sd_sim import _args, gazebo, objects, paths, png, site, site_world, terrain, world
from sd_sim.util import new_run_dir, say

ALLOWED_KEYS = ("world", "instance")
WIDTH_TOLERANCE = 0.10
ROAD_CLEAR_M = 3.0
OFFSET_LIMIT_M = 2.0
FRAMES_WANTED = 3
RUN_LIMIT_S = 240.0
RED_BLOCK = paths.SIMULATION / "scenarios" / "objects" / "red_block.yaml"


@dataclass(frozen=True)
class View:
    name: str
    x: float
    y: float
    z: float
    yaw: float  # radians, ENU
    pitch: float  # radians, positive looks down
    width: int
    height: int
    hfov_deg: float

    def sdf(self, folder: Path) -> str:
        return f"""<model name="snap_{self.name}">
    <static>true</static>
    <pose>{self.x:.3f} {self.y:.3f} {self.z:.3f} 0 {self.pitch:.6f} {self.yaw:.6f}</pose>
    <link name="link">
      <sensor name="camera" type="camera">
        <always_on>1</always_on>
        <update_rate>2</update_rate>
        <camera>
          <horizontal_fov>{math.radians(self.hfov_deg):.6f}</horizontal_fov>
          <image><width>{self.width}</width><height>{self.height}</height><format>R8G8B8</format></image>
          <clip><near>0.5</near><far>3000</far></clip>
          <save enabled="true"><path>{folder / self.name}</path></save>
        </camera>
      </sensor>
    </link>
  </model>"""


def nadir(name: str, x: float, y: float, ground: float, alt: float) -> View:
    cam = world.CAMERA
    # Pitch down a quarter turn, then yaw a quarter turn: the picture's top is north.
    return View(
        name,
        x,
        y,
        ground + alt,
        math.pi / 2,
        math.pi / 2,
        int(cam["width"]),
        int(cam["height"]),
        float(cam["hfov_deg"]),
    )


def road_point(s: site.Site) -> tuple[float, float, float]:
    """A point on a marked road's centre line where nothing else that is paved or a path
    lies within ``ROAD_CLEAR_M`` of the road's edge, nearest home; and the road's heading.
    The grey band across the picture's centre is then this road only."""
    kinds = terrain.MARKED_ROADS
    candidates = []
    for layer in s.of_kind("road"):
        if layer.props.get("highway") not in kinds or layer.props.get("layer"):
            continue
        geom = layer.geom
        for line in list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]:
            candidates.append((line, float(layer.props.get("width_m", 0.0)), layer.osm))
    if not candidates:
        raise ValueError(f"{s.name} har ingen väg med vägmarkering att mäta på")
    found: list[tuple[float, float, float, float]] = []
    for line, width, osm in candidates:
        others = shapely.union_all(
            [
                r.geom.buffer(float(r.props.get("width_m", 2.0)) / 2)
                for r in s.of_kind("road")
                if r.osm != osm
            ]
            + [s.union("parking", "building")]
        )
        reach = width / 2 + ROAD_CLEAR_M
        for at in np.arange(20.0, line.length - 20.0, 5.0):
            p = line.interpolate(at)
            q = line.interpolate(at + 2.0)
            heading = math.atan2(q.y - p.y, q.x - p.x)
            dx, dy = -math.sin(heading) * reach, math.cos(heading) * reach
            across = shapely.LineString([(p.x + dx, p.y + dy), (p.x - dx, p.y - dy)])
            if max(abs(p.x), abs(p.y)) < s.half_size_m - 60 and not across.intersects(others):
                found.append((math.hypot(p.x, p.y), p.x, p.y, heading))
    if not found:
        raise ValueError(f"{s.name}: ingen fri vägsträcka att mäta på")
    _, x, y, heading = min(found)
    return x, y, heading


def road_width(s: site.Site, x: float, y: float) -> float:
    for layer in s.of_kind("road"):
        if layer.geom.distance(shapely.Point(x, y)) < 0.01:
            return float(layer.props.get("width_m", 0.0))
    return 0.0


def asphalt_mask(img: NDArray[np.uint8]) -> NDArray[np.bool_]:
    """Asphalt and its white markings: grey (low saturation), unlike grass and forest."""
    rgb = img[..., :3].astype(np.int32)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    value = rgb.max(axis=2)
    return (spread < 18) & (value > 25)


def measure_road(
    img: NDArray[np.uint8], view: View, ground_z: float, heading: float
) -> dict[str, Any]:
    """Width of the grey band across the road through the picture's centre, and how far
    its middle lies from the centre (where the map's road line is)."""
    h, w = img.shape[:2]
    metres_per_px = 2 * (view.z - ground_z) * math.tan(math.radians(view.hfov_deg) / 2) / w
    grey = asphalt_mask(img)
    # Walk across the road: perpendicular to its heading, in picture coordinates
    # (column grows east, row grows south).
    nx, ny = -math.sin(heading), math.cos(heading)
    offsets = np.arange(-15.0, 15.0, metres_per_px / 2)
    cols = np.round(w / 2 - 0.5 + offsets * nx / metres_per_px).astype(int)
    rows = np.round(h / 2 - 0.5 - offsets * ny / metres_per_px).astype(int)
    ok = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    line = np.zeros(len(offsets), dtype=bool)
    line[ok] = grey[rows[ok], cols[ok]]
    centre = int(np.argmin(np.abs(offsets)))
    # The grey run nearest the centre (it should contain it).
    idx = np.flatnonzero(line)
    if idx.size == 0:
        return {"metres_per_px": metres_per_px, "width_m": 0.0, "offset_m": float("inf")}
    nearest = int(idx[np.argmin(np.abs(idx - centre))])
    lo = hi = nearest
    while lo > 0 and line[lo - 1]:
        lo -= 1
    while hi < len(line) - 1 and line[hi + 1]:
        hi += 1
    width = float(offsets[hi] - offsets[lo] + metres_per_px / 2)
    middle = float((offsets[hi] + offsets[lo]) / 2)
    return {
        "metres_per_px": round(metres_per_px, 4),
        "width_m": round(width, 2),
        "offset_m": round(abs(middle), 2),
    }


def last_frame(folder: Path) -> Path | None:
    frames = sorted(folder.glob("*.png"))
    return frames[-1] if len(frames) >= FRAMES_WANTED else None


def run(world_name: str, instance: int, repo: Path = paths.REPO) -> tuple[int, Path]:
    sites = site.world_sites(repo)
    if world_name not in sites:
        _args.fail(
            f"sd_sim.snapshot: {world_name} är ingen platsvärld ({', '.join(sorted(sites))})"
        )
    s = site.load(sites[world_name], repo)
    say("==> bygger världen")
    terrain.build(s, repo, progress=lambda m: say(f"    {m}"))
    ground = terrain.load_ground(s, repo)

    def gz(x: float, y: float) -> float:
        return float(ground.at(np.array([x]), np.array([y]))[0])

    run_dir = new_run_dir(instance, "snapshot")
    out = run_dir / "snapshots"
    frames = run_dir / "frames"
    rx, ry, heading = road_point(s)
    placed = objects.load(RED_BLOCK)
    block = placed[0]
    lake = s.union("water")
    poi = (60.0, 120.0)
    if not lake.is_empty:  # look at the shore nearest home: lake, forest and buildings
        near = nearest_points(lake, shapely.Point(0, 0))[0]
        poi = (round(near.x / 2), round(near.y / 2))
    views = [
        nadir("nadir_40m_road", rx, ry, gz(rx, ry), 40.0),
        nadir(
            "nadir_40m_block", block.east_m, block.north_m, gz(block.east_m, block.north_m), 40.0
        ),
        nadir("nadir_100m", poi[0], poi[1], gz(*poi), 100.0),
        View(
            "oblique",
            -330.0,
            -420.0,
            gz(-330.0, -420.0) + 160.0,
            math.atan2(560.0, 380.0),
            math.radians(22.0),
            1280,
            720,
            75.0,
        ),
    ]
    for v in views:
        (frames / v.name).mkdir(parents=True, exist_ok=True)
    path = site_world.render(s.name, instance, repo, placed, cameras=[v.sdf(frames) for v in views])
    env = gazebo.gz_env(instance, gui=False)
    log = (run_dir / "gazebo.log").open("w", encoding="utf-8")
    say("==> Gazebo (headless) tills varje kamera har sparat bilder")
    proc = subprocess.Popen(  # noqa: S603 - fixed program, our own world file
        [shutil.which("gz") or "gz", "sim", "-s", "-r", "--headless-rendering", str(path)],
        env={**env, **({"HOME": os.environ["HOME"]} if "HOME" in os.environ else {})},
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    # Gazebo renders a camera only while something subscribes to its pictures; saving to
    # disk alone does not count. One discarded subscriber per camera.
    gz_exe = shutil.which("gz") or "gz"
    readers = [
        subprocess.Popen(  # noqa: S603 - fixed program and topic
            [
                gz_exe,
                "topic",
                "-e",
                "-t",
                f"/world/{s.world}/model/snap_{v.name}/link/link/sensor/camera/image",
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for v in views
    ]
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < RUN_LIMIT_S:
            if all(last_frame(frames / v.name) for v in views) or proc.poll() is not None:
                break
            time.sleep(2.0)
    finally:
        for reader in readers:
            reader.terminate()
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "world": s.world,
        "site": s.name,
        "home_alt_m": s.home_alt,
        "stats": json.loads((terrain.out_dir(s, repo) / "manifest.json").read_text())["stats"],
        "views": {},
        "checks": {},
    }
    for v in views:
        frame = last_frame(frames / v.name)
        result["views"][v.name] = {
            "x": v.x,
            "y": v.y,
            "z": round(v.z, 2),
            "saved": frame is not None,
        }
        if frame is not None:
            shutil.copyfile(frame, out / f"{v.name}.png")
    road_png = out / "nadir_40m_road.png"
    if road_png.is_file():
        measured = measure_road(png.read(road_png), views[0], gz(rx, ry), heading)
        expected = road_width(s, rx, ry)
        measured["expected_width_m"] = expected
        result["road"] = measured
        result["checks"]["road_width"] = (
            expected > 0 and abs(measured["width_m"] - expected) <= WIDTH_TOLERANCE * expected
        )
        result["checks"]["map_overlap"] = measured["offset_m"] <= OFFSET_LIMIT_M
    result["checks"]["pictures"] = all(r["saved"] for r in result["views"].values())
    result["passed"] = all(result["checks"].values()) and bool(result["checks"])
    (out / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "summary.md").write_text(summary(result), encoding="utf-8")
    shutil.rmtree(frames, ignore_errors=True)
    return (0 if result["passed"] else 1), out


def summary(r: dict[str, Any]) -> str:
    road = r.get("road", {})
    lines = [
        f"## World {r['world']}",
        "",
        f"- Trees {r['stats']['trees']}, buildings {r['stats']['buildings']}, "
        f"cars {r['stats']['cars']}; "
        f"ground {r['stats']['ground_min_m']:.0f} to {r['stats']['ground_max_m']:.0f} m from home "
        f"(home {r['home_alt_m']} m above sea level).",
        f"- Road width in the 40 m picture: {road.get('width_m', '-')} m, "
        f"map {road.get('expected_width_m', '-')} m "
        f"({'ok' if r['checks'].get('road_width') else 'FAIL'}, limit 10 %).",
        f"- Road in the picture from the map's road line: {road.get('offset_m', '-')} m "
        f"({'ok' if r['checks'].get('map_overlap') else 'FAIL'}, limit 2 m).",
        f"- Pictures: {', '.join(k for k, v in r['views'].items() if v['saved'])}.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    values = _args.parse(
        list(sys.argv[1:] if argv is None else argv), ALLOWED_KEYS, "sd_sim.snapshot"
    )
    try:
        instance = int(values.get("instance", "0"))
    except ValueError:
        _args.fail("sd_sim.snapshot: instance= ska vara ett heltal")
    code, out = run(values.get("world", "sd_hellasgarden"), instance)
    say((out / "summary.md").read_text(encoding="utf-8"))
    say(f"Bilder och mätningar: {out}")
    return code


if __name__ == "__main__":
    sys.exit(main())
