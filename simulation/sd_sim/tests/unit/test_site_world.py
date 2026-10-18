"""Worlds from real sites: heights, generator output, the site world and its home."""

from __future__ import annotations

import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import shapely
import yaml

from sd_safety.targets import Plane, distance_m
from sd_sim import config, fixture, objects, paths, png, raster, site, site_world, snapshot, terrain
from sd_sim.sitl import CMAC_HOME

PUBLIC = "hellasgarden"
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


# --- a small synthetic site near CMAC, built in a temporary repository ------------------

CMAC_LAT, CMAC_LON = (float(v) for v in CMAC_HOME.split(",")[:2])


def _ring(plane: Plane, pts: list[tuple[float, float]]) -> list[list[float]]:
    out = []
    for x, y in [*pts, pts[0]]:
        lat, lon = plane.latlon(x, y)
        out.append([round(lon, 7), round(lat, 7)])
    return out


def make_site(repo: Path, name: str = "testsite") -> Path:
    """A 200 x 200 m site with a hill, a lake, a forest, a road, a house and a car park."""
    folder = repo / "simulation" / "gazebo" / "sites" / name
    folder.mkdir(parents=True)
    (folder / "site.toml").write_text(
        f'name = "{name}"\nworld = "sd_{name}"\ntitle = "test"\n'
        f"[home]\nlat = {CMAC_LAT}\nlon = {CMAC_LON}\nheading_deg = 0\n"
        "[area]\nhalf_size_m = 100\n"
        '[dem]\nsource = "copernicus-glo30"\ncell_m = 20\nsurface = "dsm"\ncanopy_m = 10\n',
        encoding="utf-8",
    )
    n = 15
    xs = (np.arange(n) - n // 2) * 20.0
    ex, ny = np.meshgrid(xs, xs[::-1])
    z = 100.0 + 0.1 * ex + 20.0 * np.exp(-((ex - 80) ** 2 + (ny + 80) ** 2) / 2000)
    rows = [" ".join(f"{v:.2f}" for v in row) for row in z]
    (folder / "dem.asc").write_text(f"# test\ncell_m 20\nnodes {n}\n" + "\n".join(rows) + "\n")
    plane = Plane(CMAC_LAT, CMAC_LON)

    def poly(kind: str, pts: list[tuple[float, float]], **props: Any) -> dict[str, Any]:
        geom = {"type": "Polygon", "coordinates": [_ring(plane, pts)]}
        return {
            "type": "Feature",
            "properties": {"kind": kind, "osm": f"w{len(feats)}", **props},
            "geometry": geom,
        }

    feats: list[dict[str, Any]] = []
    feats.append(poly("forest", [(-90, 20), (-20, 20), (-20, 90), (-90, 90)]))
    feats.append(poly("water", [(30, 30), (90, 30), (90, 90), (30, 90)]))
    feats.append(
        poly(
            "building",
            [(-60, -60), (-48, -60), (-48, -52), (-60, -52)],
            building="house",
            roof="gabled",
        )
    )
    feats.append(poly("parking", [(20, -70), (60, -70), (60, -45), (20, -45)]))
    road = [plane.latlon(x, -20.0) for x in (-95.0, 95.0)]
    feats.append(
        {
            "type": "Feature",
            "properties": {"kind": "road", "osm": "w99", "highway": "secondary", "width_m": 7.0},
            "geometry": {
                "type": "LineString",
                "coordinates": [[round(lon, 7), round(lat, 7)] for lat, lon in road],
            },
        }
    )
    (folder / "osm.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats})
    )
    return folder


@pytest.fixture
def small(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(terrain, "TEXTURE_PX", 64)
    monkeypatch.setattr(terrain, "HEIGHTMAP_PX", 33)
    make_site(tmp_path)
    shutil.copytree(paths.REPO / "simulation" / "scenarios", tmp_path / "simulation" / "scenarios")
    return tmp_path


def _hashes(repo: Path) -> dict[str, str]:
    return json.loads((repo / "build/sim/sites/testsite/manifest.json").read_text())["files"]


def test_generator_gives_the_same_files_twice(small: Path) -> None:
    s = site.load("testsite", small)
    terrain.build(s, small, force=True)
    first = _hashes(small)
    terrain.build(s, small, force=True)
    assert _hashes(small) == first
    names = set(first)
    assert any(n.endswith("terrain_0_0.png") for n in names)
    assert any(n.endswith("buildings.obj") for n in names) and any(
        n.endswith("cars.obj") for n in names
    )
    assert any(n.endswith(terrain.HEIGHTMAP) for n in names)


def test_build_is_reused_until_inputs_change(small: Path) -> None:
    s = site.load("testsite", small)
    model = terrain.build(s, small)
    stamp = (model / "model.sdf").stat().st_mtime_ns
    terrain.build(s, small)
    assert (model / "model.sdf").stat().st_mtime_ns == stamp
    (small / "simulation/gazebo/sites/testsite/site.toml").write_text(
        s.raw_toml.replace("canopy_m = 10", "canopy_m = 8"), encoding="utf-8"
    )
    terrain.build(site.load("testsite", small), small)
    assert (model / "model.sdf").stat().st_mtime_ns != stamp


def test_model_collides_with_a_heightmap_and_names_only_local_files(small: Path) -> None:
    model = terrain.build(site.load("testsite", small), small)
    root = ET.parse(model / "model.sdf").getroot()  # noqa: S314 - our own generated file
    collisions = root.findall(".//link[@name='terrain']/collision")
    assert [c.get("name") for c in collisions] == ["ground"]
    assert collisions[0].find("geometry/heightmap") is not None
    for uri in root.iter("uri"):
        text = uri.text or ""
        assert text.startswith("model://sd_site_testsite/meshes/")
        assert (model / text.removeprefix("model://sd_site_testsite/")).is_file()
    # The forest has no collision; it is rendered only.
    assert root.find(".//link[@name='forest']/collision") is None


def test_heightmap_matches_the_terrain_mesh(small: Path) -> None:
    s = site.load("testsite", small)
    model = terrain.build(s, small)
    ground = terrain.load_ground(s, small)
    img = png_gray16(model / "meshes" / terrain.HEIGHTMAP)
    sdf = ET.parse(model / "model.sdf").getroot()  # noqa: S314 - our own generated file
    hm = sdf.find(".//collision[@name='ground']")
    assert hm is not None
    zmin = float((hm.findtext("pose") or "").split()[2]) + 0.1
    zrange = float((hm.findtext("geometry/heightmap/size") or "").split()[2])
    n = img.shape[0]
    for r, c in ((0, 0), (n - 1, n - 1), (n // 2, n // 4), (3, n - 5)):
        x = -s.half_size_m + c * 2 * s.half_size_m / (n - 1)
        y = s.half_size_m - r * 2 * s.half_size_m / (n - 1)  # row 0 is north
        z = zmin + img[r, c] / 65535 * zrange
        assert abs(z - float(ground.at(np.array([x]), np.array([y]))[0])) < 0.01


def png_gray16(path: Path) -> np.ndarray:
    import struct
    import zlib

    data = path.read_bytes()
    pos, idat, width, height = 8, b"", 0, 0
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height = struct.unpack(">II", body[:8])
        elif kind == b"IDAT":
            idat += body
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(height, 1 + 2 * width)
    return raw[:, 1:].copy().view(">u2").astype(np.int64)


# --- heights ----------------------------------------------------------------------------


def test_heights_flat_pad_canopy_and_lake(small: Path) -> None:
    s = site.load("testsite", small)
    assert s.home_alt == pytest.approx(100.0, abs=0.1)  # the hill tail reaches home
    g = terrain.ground_grid(s)
    at = lambda x, y: float(g.at(np.array([x]), np.array([y]))[0])  # noqa: E731
    assert abs(at(0, 0)) < 0.01 and abs(at(8, -6)) < 0.01  # the flat pad
    # The lake lies under its level; the forest has its canopy taken off.
    lake_level = s.water_levels[0][1] - s.home_alt
    assert at(60, 60) <= lake_level - site.LAKE_DEPTH_M + 0.01
    raw = s.dem.sample(np.array([-55.0]), np.array([55.0]))[0] - s.home_alt
    assert at(-55, 55) == pytest.approx(raw - 10.0, abs=0.6)


def test_dem_sampling_hits_the_nodes() -> None:
    z = np.arange(25, dtype=np.float64).reshape(5, 5)
    dem = site.DemGrid(10.0, z)
    assert dem.sample(np.array([-20.0]), np.array([20.0]))[0] == pytest.approx(0.0)
    assert dem.sample(np.array([10.0]), np.array([0.0]))[0] == pytest.approx(13.0)


def test_raster_mask_fills_with_holes() -> None:
    square = shapely.Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)], [[(3, 3), (7, 3), (7, 7), (3, 7)]]
    )
    m = raster.mask(square, -5.0, 15.0, 1.0, 20, 20)
    assert m.sum() == 100 - 16
    assert not m[10, 10] and m[13, 6]  # the hole's centre, and a cell in the ring


def test_png_round_trip() -> None:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(7, 9, 3), dtype=np.uint8)
    assert np.array_equal(png.decode(png.encode(img)), img)


# --- the public site --------------------------------------------------------------------


def _enu_from_wgs84(
    lat: float, lon: float, alt: float, lat0: float, lon0: float, alt0: float
) -> tuple[float, float]:
    """East/north the way Gazebo's spherical coordinates compute it (WGS84, ECEF, ENU)."""

    def ecef(la: float, lo: float, h: float) -> np.ndarray:
        la, lo = math.radians(la), math.radians(lo)
        n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(la) ** 2)
        return np.array(
            [
                (n + h) * math.cos(la) * math.cos(lo),
                (n + h) * math.cos(la) * math.sin(lo),
                (n * (1 - WGS84_E2) + h) * math.sin(la),
            ]
        )

    d = ecef(lat, lon, alt) - ecef(lat0, lon0, alt0)
    la, lo = math.radians(lat0), math.radians(lon0)
    east = -math.sin(lo) * d[0] + math.cos(lo) * d[1]
    north = (
        -math.sin(la) * math.cos(lo) * d[0]
        - math.sin(la) * math.sin(lo) * d[1]
        + math.cos(la) * d[2]
    )
    return east, north


def test_osm_point_is_within_2_m_in_world_and_map() -> None:
    """A building corner from OpenStreetMap (the map data the panel shows) lands within 2 m
    of the same corner of the building in the Gazebo world."""
    s = site.load(PUBLIC)
    raw = json.loads((s.folder / "osm.geojson").read_text(encoding="utf-8"))
    corners = [f for f in raw["features"] if f["properties"]["kind"] == "building"]
    assert corners
    ground = terrain.ground_grid(s)
    buildings = terrain.plan_buildings(s, ground)
    world_pts = np.array([c for b in buildings for c in b.footprint.exterior.coords])
    worst = 0.0
    for feature in corners:
        for lon, lat in feature["geometry"]["coordinates"][0]:
            # Where Gazebo puts that latitude/longitude, with the world's spherical coordinates.
            east, north = _enu_from_wgs84(lat, lon, s.home_alt, s.lat, s.lon, s.home_alt)
            nearest = float(np.min(np.hypot(world_pts[:, 0] - east, world_pts[:, 1] - north)))
            worst = max(worst, nearest)
    assert worst < 2.0, worst


def test_site_world_home_matches_sitl_and_the_map() -> None:
    s = site.load(PUBLIC)
    ground = terrain.ground_grid(s)
    root = ET.fromstring(site_world.world_sdf(s, ground, 0))  # noqa: S314 - our own rendered file
    coords = root.find("world/spherical_coordinates")
    assert coords is not None
    lat = float(coords.findtext("latitude_deg") or 0)
    lon = float(coords.findtext("longitude_deg") or 0)
    alt = float(coords.findtext("elevation") or 0)
    sitl_lat, sitl_lon, sitl_alt, _ = (float(v) for v in s.home_arg.split(","))
    assert distance_m(lat, lon, sitl_lat, sitl_lon) < 0.2
    assert abs(alt - sitl_alt) <= 0.05
    assert [u.text for u in root.iter("uri")] == [
        "model://sd_site_hellasgarden",
        "model://sd_quad14_i0",
    ]


def test_objects_stand_on_the_terrain() -> None:
    s = site.load(PUBLIC)
    ground = terrain.ground_grid(s)
    box = objects.Box("probe", 150.0, -300.0, (1.0, 1.0, 1.0), (1.0, 0.0, 0.0))
    root = ET.fromstring(site_world.world_sdf(s, ground, 0, [box]))  # noqa: S314 - our own file
    pose = root.findtext("world/model[@name='probe']/pose") or ""
    z = float(pose.split()[2])
    assert z == pytest.approx(float(ground.at(np.array([150.0]), np.array([-300.0]))[0]), abs=0.01)
    assert abs(z) > 1.0  # the ground there is not at home's height


def test_site_fixture_parses_and_clears_the_forest() -> None:
    s = site.load(PUBLIC)
    ground = terrain.ground_grid(s)
    text = site_world.fixture_yaml(s, ground)
    fix = fixture.parse(yaml.safe_load(text))
    assert (fix.home.lat, fix.home.lon, fix.home.alt_amsl_m) == (s.lat, s.lon, s.home_alt)
    charter = fix.charters["charter_test"]
    area = shapely.Polygon(charter.area_enu)
    assert area.contains(shapely.Point(fix.survey_box.center))
    tallest = site_world.tallest(s, ground, area)
    assert charter.default_alt_agl_m >= tallest + site_world.SEARCH_CLEARANCE_M
    assert charter.default_alt_agl_m <= charter.alt_max_agl_m


# --- config and smoke ------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """The example config and the public sites, without the developer's local config."""
    (tmp_path / "config").mkdir()
    shutil.copyfile(
        paths.REPO / "config" / "sim.example.toml", tmp_path / "config" / "sim.example.toml"
    )
    shutil.copytree(
        paths.REPO / "simulation" / "gazebo" / "sites", tmp_path / "simulation" / "gazebo" / "sites"
    )
    return tmp_path


def test_launch_world_brings_its_home_tests_keep_cmac(repo: Path) -> None:
    assert config.load({}, repo=repo, environ={}).home == CMAC_HOME
    cfg = config.load({}, repo=repo, environ={}, default_world=config.LAUNCH_WORLD)
    assert cfg.world == "sd_hellasgarden" and cfg.home == site.load(PUBLIC).home_arg
    args = {"world": "sd_hellasgarden", "home_override": "1,2,3,4"}
    personal = config.load(args, repo=repo, environ={})
    assert personal.home == "1,2,3,4" and personal.personal_home


def test_public_site_world_in_the_demo_uses_its_home(repo: Path) -> None:
    cfg = config.load({"world": "sd_hellasgarden"}, repo=repo, demo=True, environ={})
    assert cfg.home == site.load(PUBLIC).home_arg and not cfg.personal_home


def _local_site(repo: Path) -> None:
    src = repo / "simulation" / "gazebo" / "sites" / PUBLIC
    dest = repo / "build" / "sites" / "mine"
    shutil.copytree(src, dest)
    toml = (dest / "site.toml").read_text(encoding="utf-8")
    toml = toml.replace('name = "hellasgarden"', 'name = "mine"')
    (dest / "site.toml").write_text(toml.replace('world = "sd_hellasgarden"', 'world = "sd_mine"'))


def test_local_site_is_a_personal_home(repo: Path) -> None:
    _local_site(repo)
    cfg = config.load({"world": "sd_mine"}, repo=repo, environ={})
    assert cfg.personal_home
    with pytest.raises(config.DemoHomeError):
        config.load({"world": "sd_mine"}, repo=repo, demo=True, environ={})
    with pytest.raises(config.DemoHomeError):
        config.load({}, repo=repo, demo=True, environ={"SD_SIM__WORLD": "sd_mine"})


def test_scripted_commander_only_in_the_flat_world(repo: Path) -> None:
    with pytest.raises(config.ConfigError, match="brain=scripted"):
        config.load({"world": "sd_hellasgarden", "brain": "scripted"}, repo=repo, environ={})
    assert config.load({"brain": "scripted"}, repo=repo, environ={}).world == config.TEST_WORLD


# --- the snapshot's road measurement ----------------------------------------------------


def test_measure_road_on_a_drawn_picture() -> None:
    view = snapshot.nadir("t", 0.0, 0.0, 0.0, 40.0)
    h, w = view.height, view.width
    m_px = 2 * 40.0 * math.tan(math.radians(view.hfov_deg) / 2) / w
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (70, 110, 50)
    rows = np.arange(h)[:, None]
    # A 7 m grey road running east, its middle 1 m north of the picture's centre.
    north = (h / 2 - 0.5 - rows) * m_px
    band = np.abs(north - 1.0) <= 3.5
    img[np.broadcast_to(band, (h, w))] = (80, 80, 82)
    out = snapshot.measure_road(img, view, 0.0, 0.0)
    assert out["width_m"] == pytest.approx(7.0, abs=2 * m_px)
    assert out["offset_m"] == pytest.approx(1.0, abs=2 * m_px)
