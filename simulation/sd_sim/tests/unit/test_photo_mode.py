"""Photo mode: a fetched orthophoto draped over the open ground, global illumination for
the cameras, and the world exactly as before when there is no photo.

Every photo here is synthetic, made at run time: two flat colours and a marker. No real
orthophoto is ever part of the tests."""

from __future__ import annotations

import importlib.util
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sd_sim import config, png, site, site_world, snapshot, terrain

_spec = importlib.util.spec_from_file_location(
    "site_world_tests", Path(__file__).with_name("test_site_world.py")
)
assert _spec is not None and _spec.loader is not None
_sw: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sw)
_fs = importlib.util.spec_from_file_location(
    "forest_tests", Path(__file__).with_name("test_forest.py")
)
assert _fs is not None and _fs.loader is not None
_forest: Any = importlib.util.module_from_spec(_fs)
_fs.loader.exec_module(_forest)
small = _sw.small
wooded = _forest.wooded

OPEN = (200, 40, 200)  # the synthetic photo's ground: magenta, which nothing paints
MARK = (40, 200, 40)  # a 6 m square, green
MARK_AT = (10.0, -60.0)  # east, north: open grass between the road and the car park
PHOTO_PX = 400  # 0.5 m per pixel over the 200 m test site


def synthetic_photo(half_size_m: float = 100.0) -> np.ndarray:
    photo = np.empty((PHOTO_PX, PHOTO_PX, 3), np.uint8)
    photo[:] = OPEN
    cell = 2 * half_size_m / PHOTO_PX
    e, n = MARK_AT
    c0, c1 = round((e - 3 + half_size_m) / cell), round((e + 3 + half_size_m) / cell)
    r0, r1 = round((half_size_m - n - 3) / cell), round((half_size_m - n + 3) / cell)
    photo[r0:r1, c0:c1] = MARK
    return photo


def put_photo(repo: Path) -> Path:
    folder = repo / "build" / "ortho" / "testsite"
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "ortho.npy", synthetic_photo())
    return folder


def tile(repo: Path, east: float, north: float) -> tuple[np.ndarray, int, int]:
    """The ground texture under (east, north), and the pixel there."""
    size = 200.0 / terrain.TILES
    tj, ti = int((east + 100) // size), int((100 - north) // size)
    s = site.load("testsite", repo)
    meshes = repo / "build" / "sim" / "models" / terrain.model_name(s) / "meshes"
    img = png.read(meshes / f"terrain_{ti}_{tj}.png")
    step = size / img.shape[1]
    col = int((east - (-100 + tj * size)) / step)
    row = int(((100 - ti * size) - north) / step)
    return img, row, col


def _hashes(repo: Path) -> dict[str, str]:
    manifest = json.loads((repo / "build/sim/sites/testsite/manifest.json").read_text())
    return dict(manifest["files"])


def _stats(repo: Path) -> dict[str, Any]:
    manifest = json.loads((repo / "build/sim/sites/testsite/manifest.json").read_text())
    return dict(manifest["stats"])


def test_photo_lands_where_it_was_taken(small: Path) -> None:
    put_photo(small)
    terrain.build(site.load("testsite", small), small, force=True)
    assert _stats(small)["ground_texture"] == "orthophoto"
    img, _, _ = tile(small, *MARK_AT)
    rgb = img[..., :3].astype(int)
    green = np.argwhere((np.abs(rgb - MARK) < 3).all(axis=2))
    assert len(green) > 0
    # The marker's centre in the texture is the marker's place on the ground, north up.
    size = 200.0 / terrain.TILES
    step = size / img.shape[1]
    row, col = green.mean(axis=0)
    west, north = 0.0, -50.0  # the tile holding MARK_AT
    east_m, north_m = west + (col + 0.5) * step, north - (row + 0.5) * step
    assert abs(east_m - MARK_AT[0]) < step and abs(north_m - MARK_AT[1]) < step
    side = (green[:, 1].max() - green[:, 1].min() + 1) * step
    assert side == pytest.approx(6.0, abs=2 * step)
    # Open ground elsewhere is the photo too.
    img, row, col = tile(small, -10.0, 5.0)
    assert tuple(img[row, col, :3]) == OPEN


def test_tree_crowns_keep_the_painted_floor(wooded: Path) -> None:
    put_photo(wooded)
    terrain.build(site.load("testsite", wooded), wooded, force=True)
    # The pine at (-30, 40): its own model stands there, so the photo's tree must not lie
    # flat under it.
    img, row, col = tile(wooded, -30.0, 40.0)
    assert tuple(img[row, col, :3]) != OPEN
    img, row, col = tile(wooded, -10.0, 5.0)
    assert tuple(img[row, col, :3]) == OPEN


def test_photo_cover_is_a_soft_disc_a_little_wider_than_the_crown() -> None:
    xs = np.arange(-10.0, 10.0, 0.1) + 0.05
    ex, ny = np.meshgrid(xs, xs[::-1])
    cover = terrain.photo_cover([(0.0, 0.0, 2.0)], ex, ny, -10.0, 10.0, 0.1)
    reach = 2.0 * terrain.PHOTO_CROWN_SCALE + terrain.PHOTO_CROWN_MARGIN_M
    d = np.hypot(ex, ny)
    assert (cover[d < reach - terrain.PHOTO_EDGE_M] == 1.0).all()
    assert (cover[d > reach] == 0.0).all()
    assert cover.min() == 0.0 and 0 < cover[(d > reach - 0.5) & (d < reach - 0.1)].mean() < 1


def test_without_the_photo_the_world_is_exactly_as_before(small: Path) -> None:
    s = site.load("testsite", small)
    terrain.build(s, small, force=True)
    before = _hashes(small)
    assert "ground_texture" not in _stats(small)
    folder = put_photo(small)
    terrain.build(site.load("testsite", small), small)
    assert _hashes(small) != before
    shutil.rmtree(folder)
    terrain.build(site.load("testsite", small), small)
    assert _hashes(small) == before
    # A photo in a public site's committed folder is never used.
    np.save(small / "simulation/gazebo/sites/testsite/ortho.npy", synthetic_photo())
    assert site.load("testsite", small).ortho is None
    terrain.build(site.load("testsite", small), small, force=True)
    assert _hashes(small) == before


def test_a_broken_photo_says_fetch_again(small: Path) -> None:
    folder = small / "build" / "ortho" / "testsite"
    folder.mkdir(parents=True)
    np.save(folder / "ortho.npy", np.zeros((40, 40), np.uint8))
    with pytest.raises(ValueError, match="just ortho-fetch"):
        terrain.build(site.load("testsite", small), small, force=True)


def _gi(sdf: str) -> ET.Element | None:
    root = ET.fromstring(sdf)  # noqa: S314 - our own rendered file
    return root.find(".//plugin[@name='gz::sim::systems::Sensors']/global_illumination")


def test_global_illumination_only_when_asked(small: Path) -> None:
    s = site.load("testsite", small)
    ground = terrain.ground_grid(s)
    assert _gi(site_world.world_sdf(s, ground, 0)) is None
    gi = _gi(site_world.world_sdf(s, ground, 0, gi="vct"))
    assert gi is not None and gi.get("type") == "vct"
    assert gi.findtext("enabled") == "true"
    assert gi.findtext("resolution") == site_world.GI_VCT["resolution"]
    with pytest.raises(ValueError, match="gi"):
        site_world.world_sdf(s, ground, 0, gi="rtx")


def test_gi_needs_a_site_world_and_gazebo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    sites = repo / "simulation" / "gazebo" / "sites"
    shutil.copytree(_sw.paths.REPO / "simulation" / "gazebo" / "sites", sites)
    (repo / "config" / "local").mkdir(parents=True)
    example = repo / "config" / "sim.example.toml"
    shutil.copyfile(_sw.paths.REPO / "config" / "sim.example.toml", example)
    cfg = config.load({"world": "sd_hellasgarden", "gi": "vct"}, repo=repo, environ={})
    assert cfg.gi == "vct"
    assert config.load({}, repo=repo, environ={}).gi == "off"
    with pytest.raises(config.ConfigError, match="platsvärld"):
        config.load({"world": "sd_lake_forest", "gi": "vct"}, repo=repo, environ={})
    with pytest.raises(config.ConfigError, match="gi"):
        config.load({"world": "sd_hellasgarden", "gi": "yes"}, repo=repo, environ={})


def test_snapshot_credits_the_photo_only_when_used() -> None:
    assert "Lantmäteriet's orthophoto" not in snapshot.attribution({"trees": 1})
    text = snapshot.attribution({"ground_texture": "orthophoto"})
    assert "Lantmäteriet's orthophoto" in text and "CC BY 4.0" in text


def test_sim_photo_says_what_to_do_without_the_photo(
    small: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from sd_sim import photo

    with pytest.raises(SystemExit) as stop:
        photo.launch_argv(["site=testsite"], small)
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert "Ortofoto saknas" in err and "just ortho-fetch site=testsite" in err
    assert "geotorget.example.toml" in err  # no login file yet: step one is the login
    put_photo(small)
    argv = photo.launch_argv(["site=testsite", "gui=false"], small)
    assert argv == ["up", "world=sd_testsite", "gi=vct", "gui=false"]
    assert photo.launch_argv(["site=testsite", "gi=off"], small)[2:] == ["gi=off"]
    with pytest.raises(SystemExit):
        photo.launch_argv(["world=sd_lake_forest"], small)


def test_snapshot_summary_reports_lighting_rtf_and_memory() -> None:
    r: dict[str, Any] = {
        "world": "w",
        "home_alt_m": 1,
        "stats": {"trees": 1, "buildings": 0, "cars": 0, "ground_min_m": 0, "ground_max_m": 1},
        "checks": {},
        "views": {},
        "gi": "vct",
        "real_time_factor": 0.47,
        "gpu_memory_used_mib": None,
    }
    text = snapshot.summary(r)
    assert "global illumination vct" in text and "0.47" in text and "not measured" in text
    r["gpu_memory_used_mib"] = 2100
    assert "2100 MiB" in snapshot.summary(r)
