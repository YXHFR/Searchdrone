"""Render a Gazebo world from a site (``sd_sim.site``) for one simulator instance.

``render(site_name, instance)`` builds the site's model if needed (``sd_sim.terrain``,
cached in ``build/sim/sites/``) and writes:

- ``build/sim/worlds/<world>_i<N>.sdf``: the site model, the landing pad at home,
  scenario objects on the terrain (``sd_sim.objects``) and the vehicle on the pad;
  ``<spherical_coordinates>`` is home with the pad's height above sea level, the same
  point SITL gets as ``--home`` (``Site.home_arg``) and the panel's map is centred on;
- ``build/sim/worlds/<world>_i<N>.truth.json``: the objects' facit;
- ``build/sim/worlds/<world>.fixture.yaml``: the scenario fixture for this site (home,
  pilot, pins, the search box and the test charters, as ``demo_fixture.yaml`` has them
  for the flat world), with the search box over open ground and its heights checked
  against the terrain and the forest.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import shapely
import yaml

from sd_sim import objects, paths, site, terrain, world

TEMPLATE = "worlds/sd_site.sdf.jinja"
PAD_SIZE_M = 3.0
# The fixture's search height must clear the tallest thing under the charter by this much.
SEARCH_CLEARANCE_M = 25.0


def frame(s: site.Site, ground: terrain.Ground) -> world.Frame:
    def ground_at(east: float, north: float) -> float:
        return float(ground.at(np.array([east]), np.array([north]))[0])

    return world.Frame(s.lat, s.lon, s.home_alt, s.half_size_m, ground_at)


def world_sdf(
    s: site.Site,
    ground: terrain.Ground,
    instance: int,
    placed: Sequence[objects.WorldObject] = (),
    cameras: Sequence[str] = (),
    light: Mapping[str, Any] | None = None,
) -> str:
    fr = frame(s, ground)
    objects.check(placed, s.half_size_m)
    return (
        world._env()
        .get_template(TEMPLATE)
        .render(
            name=s.world,
            lat=s.lat,
            lon=s.lon,
            alt=s.home_alt,
            site_uri=f"model://{terrain.model_name(s)}",
            pad_size=PAD_SIZE_M,
            objects=[world.checked_fragment(obj, fr) for obj in placed],
            cameras=list(cameras),
            model_uri=f"model://{world.MODEL_NAME}_i{instance}",
            vehicle_z=world.STAND_M,
            vehicle_yaw=round(math.radians(90.0 - s.heading_deg), 6),
            light=world.light_context(light),
        )
    )


def render(
    site_name: str,
    instance: int,
    repo: Path = paths.REPO,
    placed: Sequence[objects.WorldObject] = (),
    *,
    cameras: Sequence[str] = (),
    progress: Callable[[str], None] = lambda _: None,
    stills: bool = False,
    video: bool = True,
    light: Mapping[str, Any] | None = None,
) -> Path:
    s = site.load(site_name, repo)
    terrain.build(s, repo, progress=progress)
    ground = terrain.load_ground(s, repo)
    world.render_model(instance, repo, stills=stills, video=video)
    out = repo / "build" / "sim" / "worlds"
    out.mkdir(parents=True, exist_ok=True)
    sdf = world_sdf(s, ground, instance, placed, cameras, light)
    truth = world.world_truth(s.world, placed, frame=frame(s, ground))
    path = out / f"{s.world}_i{instance}.sdf"
    path.write_text(sdf, encoding="utf-8")
    world.truth_path(s.world, instance, repo).write_text(
        json.dumps(truth, indent=1) + "\n", encoding="utf-8"
    )
    (out / f"{s.world}.fixture.yaml").write_text(fixture_yaml(s, ground, repo), encoding="utf-8")
    return path


# --- the site's fixture ------------------------------------------------------------------


def tallest(s: site.Site, ground: terrain.Ground, area: Any) -> float:
    """Highest ground inside ``area`` in metres above home, plus the tallest tree when any
    forest reaches into it."""
    minx, miny, maxx, maxy = area.bounds
    ex, ny = np.meshgrid(np.arange(minx, maxx + 5, 5.0), np.arange(miny, maxy + 5, 5.0))
    inside = shapely.contains_xy(area, ex, ny)
    top = float(ground.at(ex[inside], ny[inside]).max()) if inside.any() else 0.0
    forest = s.union("forest")
    if not forest.is_empty and forest.intersects(area):
        top += max(hi for _, _, (_, hi) in terrain.SPECIES)
    return top


def search_box_center(s: site.Site) -> tuple[float, float]:
    """The open meadow closest to 200 m from home, or (0, 200) when the site has none."""
    open_land = s.union("meadow", "farmland", "grass")
    best: tuple[float, float, float] | None = None
    for poly in getattr(open_land, "geoms", [open_land]):
        if poly.is_empty or poly.area < 2000:
            continue
        p = poly.representative_point()
        score = abs(math.hypot(p.x, p.y) - 200.0)
        if best is None or score < best[0]:
            best = (score, round(p.x / 10) * 10.0, round(p.y / 10) * 10.0)
    return (best[1], best[2]) if best else (0.0, 200.0)


def fixture_yaml(s: site.Site, ground: terrain.Ground, repo: Path = paths.REPO) -> str:
    base = yaml.safe_load(
        (repo / "simulation" / "scenarios" / "demo_fixture.yaml").read_text(encoding="utf-8")
    )
    cx, cy = search_box_center(s)
    base["home"] = {
        "lat": s.lat,
        "lon": s.lon,
        "alt_amsl_m": s.home_alt,
        "heading_deg": s.heading_deg,
    }
    for pin in base["pins"]:
        if pin["id"] == "P1":
            pin["east_m"], pin["north_m"] = cx, cy
    test = base["charters"]["charter_test"]
    # charter_test around home and the search box, the same shape as in the flat world.
    west, east = min(-150.0, cx - 100.0), max(150.0, cx + 100.0)
    south, north = min(-100.0, cy - 100.0), max(300.0, cy + 100.0)
    test["area_enu"] = [[west, south], [east, south], [east, north], [west, north]]
    top = tallest(s, ground, shapely.box(west, south, east, north))
    alt = max(float(base["survey_box"]["alt_m"]), math.ceil((top + SEARCH_CLEARANCE_M) / 5) * 5)
    base["survey_box"]["alt_m"] = alt
    test["default_alt_agl_m"] = alt
    header = (
        f"# Generated by sd_sim.site_world for the site {s.name}; do not edit.\n"
        f"# Heights are above home. The tallest ground plus trees under charter_test is\n"
        f"# {top:.0f} m above home, so the search height is {alt:g} m.\n"
    )
    return header + yaml.safe_dump(base, sort_keys=False, allow_unicode=True)
