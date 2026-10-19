"""The real trees: glTF files, tree models, and a site world built from a tree list."""

from __future__ import annotations

import importlib.util
import json
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sd_sim import forest, gltf, paths, png, site, site_world, snapshot, terrain

_spec = importlib.util.spec_from_file_location(
    "site_world_tests", Path(__file__).with_name("test_site_world.py")
)
assert _spec is not None and _spec.loader is not None
_sw: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sw)


def _chunks(data: bytes) -> tuple[dict[str, Any], bytes]:
    magic, version, total = struct.unpack_from("<III", data, 0)
    assert (magic, version, total) == (0x46546C67, 2, len(data))
    jlen, jkind = struct.unpack_from("<II", data, 12)
    assert jkind == 0x4E4F534A
    doc = json.loads(data[20 : 20 + jlen])
    blen, bkind = struct.unpack_from("<II", data, 20 + jlen)
    assert bkind == 0x004E4942
    return doc, data[28 + jlen : 28 + jlen + blen]


def test_glb_holds_one_textured_primitive_z_up() -> None:
    p = gltf.Primitive()
    pos = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 2], [0, 0, 2]], float)
    p.add(
        pos,
        np.tile([0, -1, 0], (4, 1)),
        np.tile([1, 0, 0, 1], (4, 1)),
        np.array([[0, 1], [1, 1], [1, 0], [0, 0]]),
        np.array([[0, 1, 2], [0, 2, 3]]),
    )
    mat = gltf.PbrMaterial("m", "c.png", normal="n.png", orm="o.png")
    data = gltf.glb({"m": p}, [mat], "t")
    assert data == gltf.glb({"m": p}, [mat], "t")  # the same bytes every time
    doc, blob = _chunks(data)
    assert [i["uri"] for i in doc["images"]] == ["c.png", "o.png", "n.png"]
    m = doc["materials"][0]
    assert m["alphaMode"] == "MASK" and m["doubleSided"] is True
    prim = doc["meshes"][0]["primitives"][0]
    acc = doc["accessors"][prim["attributes"]["POSITION"]]
    assert acc["count"] == 4 and acc["max"] == [1.0, 0.0, 2.0]  # Z up, as the world
    view = doc["bufferViews"][acc["bufferView"]]
    first = np.frombuffer(blob, "<f4", 12, view["byteOffset"]).reshape(4, 3)
    assert np.allclose(first, pos)
    assert doc["accessors"][prim["indices"]]["count"] == 6


ATLAS = {
    f"{sp}_{part}": (0.25 * i, 0.25 * j, 0.25 * i + 0.24, 0.25 * j + 0.24)
    for i, sp in enumerate(forest.SPECIES)
    for j, part in enumerate(("card", "side", "top", "bark"))
}


@pytest.mark.parametrize("species", forest.SPECIES)
def test_models_are_one_metre_tall_and_cheap_far_away(species: str) -> None:
    near = forest.near_model(species, ATLAS)
    far = forest.far_model(species, ATLAS)
    for model in (near, far):
        pos, nor, _tan, _uv, faces = model.arrays()
        assert pos[:, 2].max() == pytest.approx(1.0)
        assert np.hypot(pos[:, 0], pos[:, 1]).max() <= forest.CROWN[species] + 0.12
        assert np.allclose(np.linalg.norm(nor, axis=1), 1.0)
        assert faces.max() < len(pos)
    tri = lambda m: sum(len(f) for f in m.faces)  # noqa: E731
    assert 400 <= tri(near) <= 2000 and tri(far) <= 8


def test_trees_are_scaled_to_their_height_and_sunk_into_the_ground() -> None:
    tree = forest.RealTree(10.0, -5.0, 23.4, "spruce")
    pos, *_ = forest.place(forest.far_model("spruce", ATLAS), tree, 2.0)
    assert pos[:, 2].max() == pytest.approx(2.0 + 23.4, abs=0.01)
    assert pos[:, 2].min() == pytest.approx(2.0 - forest.SINK_M, abs=0.01)
    assert np.allclose(pos[:, :2].mean(axis=0), [10.0, -5.0], atol=forest.CROWN["spruce"] * 23.4)


def test_near_home_is_detailed_and_tiles_split_the_forest() -> None:
    trees = [
        forest.RealTree(5.0, 50.0, 15.0, "pine"),
        forest.RealTree(-500.0, 500.0, 20.0, "deciduous"),
        forest.RealTree(500.0, -500.0, 18.0, "spruce"),
    ]
    woods = forest.build_forest(trees, np.zeros(3), 600.0, ATLAS)
    assert woods.near == 1 and len(woods.tiles) == 3
    assert woods.triangles == woods.model_triangles["pine_near"] + 2 * 6


def test_height_check_against_the_raster() -> None:
    chm = np.zeros((20, 20))
    chm[9, 9] = 14.5  # the cell east -0.5, north 0.5 of a 20 m site
    t = forest.RealTree(-0.5, 0.5, 15.0, "pine")
    assert forest.height_check([t], chm, 10.0) == pytest.approx(0.5)
    assert forest.height_check([t], None, 10.0) is None


# --- a site world built from a tree list ----------------------------------------------


@pytest.fixture
def wooded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(terrain, "TEXTURE_PX", 64)
    monkeypatch.setattr(terrain, "HEIGHTMAP_PX", 33)
    _sw.make_site(tmp_path)
    shutil.copytree(paths.REPO / "simulation" / "scenarios", tmp_path / "simulation" / "scenarios")
    trees = tmp_path / "build" / "trees" / "testsite"
    trees.mkdir(parents=True)
    (trees / "trees.csv").write_text(
        "east_m,north_m,height_m,species\n"
        "-50.0,60.0,21.5,spruce\n"
        "-30.0,40.0,12.0,pine\n"
        "-54.0,-56.0,9.0,deciduous\n"  # on the house's roof: the raster sees buildings too
        "60.0,60.0,8.0,deciduous\n"  # in the lake
        "-70.0,-15.0,30.0,deciduous\n",
        encoding="ascii",
    )
    chm = np.zeros((200, 200), np.uint16)
    chm[100 - 60, 100 - 50] = 215
    np.save(trees / "chm_dm.npy", chm)
    atlas = tmp_path / "build" / "tree_models"
    atlas.mkdir(parents=True)
    for name in forest.ATLAS_FILES:
        png.write(atlas / name, np.full((8, 8, 4), 200, np.uint8))
    regions = {k: list(v) for k, v in ATLAS.items()}
    (atlas / "atlas.json").write_text(json.dumps({"regions": regions}), encoding="utf-8")
    return tmp_path


def test_world_has_the_real_trees_and_bare_heights(wooded: Path) -> None:
    s = site.load("testsite", wooded)
    model = terrain.build(s, wooded, force=True)
    manifest = json.loads((wooded / "build/sim/sites/testsite/manifest.json").read_text())
    stats = manifest["stats"]
    assert stats["tree_source"] == "raster" and stats["tree_models"] == "poly-haven-cc0"
    assert stats["trees_raster"] == 5 and stats["trees"] == 3 and stats["trees_left_out"] == 2
    assert stats["trees_near"] == 3
    root = ET.parse(model / "model.sdf").getroot()  # noqa: S314 - our own generated file
    uris = [u.text or "" for u in root.findall(".//link[@name='forest']/visual/geometry/mesh/uri")]
    assert uris and all(u.endswith(".glb") for u in uris)
    for name in forest.ATLAS_FILES:
        assert (model / "meshes" / name).is_file()
    # With the real trees there is no canopy taken off: the heights are the source's.
    g = terrain.ground_grid(s)
    raw = s.dem.sample(np.array([-55.0]), np.array([55.0]))[0] - s.home_alt
    assert float(g.at(np.array([-55.0]), np.array([55.0]))[0]) == pytest.approx(raw, abs=0.05)
    # The fixture's search height clears the tallest real tree top under the charter.
    top = float(g.at(np.array([-70.0]), np.array([-15.0]))[0]) + 30.0
    assert site_world.tallest(s, g, s.union("forest").buffer(100)) == pytest.approx(top)
    # The same files again, and the build is redone when the tree list changes.
    first = manifest["files"]
    terrain.build(s, wooded, force=True)
    again = json.loads((wooded / "build/sim/sites/testsite/manifest.json").read_text())
    assert again["files"] == first
    before = s.input_digest()
    csv = wooded / "build/trees/testsite/trees.csv"
    csv.write_text(csv.read_text().replace("12.0,pine", "13.0,pine"))
    assert site.load("testsite", wooded).input_digest() != before


def test_world_without_the_tree_pictures_has_plain_real_trees(wooded: Path) -> None:
    shutil.rmtree(wooded / "build" / "tree_models")
    s = site.load("testsite", wooded)
    model = terrain.build(s, wooded, force=True)
    stats = json.loads((wooded / "build/sim/sites/testsite/manifest.json").read_text())["stats"]
    assert stats["tree_source"] == "raster" and stats["tree_models"] == "plain"
    assert stats["trees"] == 3
    assert any(
        p.suffix == ".obj" and p.stem.startswith("trees_") for p in (model / "meshes").iterdir()
    )


def test_search_height_above_the_charters_cap_is_an_error(wooded: Path) -> None:
    s = site.load("testsite", wooded)
    g = terrain.ground_grid(s)
    yaml_text = site_world.fixture_yaml(s, g)
    assert "default_alt_agl_m" in yaml_text
    csv = wooded / "build/trees/testsite/trees.csv"
    csv.write_text(csv.read_text().replace("-70.0,-15.0,30.0", "-70.0,-15.0,110.0"))
    s = site.load("testsite", wooded)
    with pytest.raises(ValueError, match="charter_test"):
        site_world.fixture_yaml(s, terrain.ground_grid(s))


def test_canopy_follows_the_tree_list_not_the_raster(wooded: Path) -> None:
    def forest_drop(s: site.Site) -> float:
        g = terrain.ground_grid(s)
        raw = s.dem.sample(np.array([-55.0]), np.array([55.0]))[0] - s.home_alt
        return float(raw - g.at(np.array([-55.0]), np.array([55.0]))[0])

    assert forest_drop(site.load("testsite", wooded)) == pytest.approx(0.0, abs=0.05)
    (wooded / "build/trees/testsite/trees.csv").unlink()  # the raster alone
    s = site.load("testsite", wooded)
    assert not s.has_real_trees and forest.load_trees(s) is None
    assert forest_drop(s) == pytest.approx(10.0, abs=0.6)  # generated trees, canopy off


# --- the snapshot's notices and summary ------------------------------------------------

RASTER_STATS = {
    "trees": 14669,
    "buildings": 3,
    "cars": 1,
    "ground_min_m": -2.0,
    "ground_max_m": 30.0,
    "tree_source": "raster",
    "tree_models": "poly-haven-cc0",
    "trees_left_out": 421,
    "trees_near": 482,
    "tree_triangles": 527414,
    "tree_height_vs_raster_mean_abs_m": 0.325,
    "tree_height_max_m": 33.5,
}


def test_attribution_names_the_tree_sources_only_when_used() -> None:
    generated = snapshot.attribution({"trees": 10})
    assert generated == snapshot.ATTRIBUTION
    full = snapshot.attribution(RASTER_STATS)
    assert full == snapshot.ATTRIBUTION + snapshot.TREES_ATTRIBUTION + (
        snapshot.TREE_MODELS_ATTRIBUTION
    )
    assert "Lantmäteriet, CC BY 4.0" in full and "Poly Haven" in full
    plain = snapshot.attribution({**RASTER_STATS, "tree_models": "plain"})
    assert plain == snapshot.ATTRIBUTION + snapshot.TREES_ATTRIBUTION


def _result(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "world": "sd_testsite",
        "home_alt_m": 20.0,
        "stats": stats,
        "views": {"nadir_40m_road": {"saved": True}, "oblique": {"saved": False}},
        "checks": {"road_width": True, "map_overlap": False},
        "road": {"width_m": 6.9, "expected_width_m": 7.0, "offset_m": 2.5},
    }


def test_summary_tells_real_trees_from_generated_ones() -> None:
    text = snapshot.summary(_result(RASTER_STATS))
    assert "14669 real trees from the tree height raster (421 on roads" in text
    assert "482 detailed near home; 527414 triangles; models poly-haven-cc0." in text
    assert "mean error 0.325 m, tallest 33.5 m." in text
    assert "6.9 m, map 7.0 m (ok" in text and "2.5 m (FAIL" in text
    assert "Pictures: nadir_40m_road." in text
    stats = {k: RASTER_STATS[k] for k in ("trees", "buildings", "cars")}
    stats |= {"ground_min_m": 0.0, "ground_max_m": 1.0, "tree_triangles": 900}
    text = snapshot.summary(_result(stats))
    assert "- Forest: generated (900 triangles)." in text and "real trees" not in text
