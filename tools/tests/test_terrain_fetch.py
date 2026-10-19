"""tools/terrain/fetch_site.py without the network: settings, OSM parsing and the DEM file."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from shapely.geometry import box, shape

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "fetch_site", ROOT / "tools" / "terrain" / "fetch_site.py"
)
assert _spec is not None and _spec.loader is not None
fetch: Any = importlib.util.module_from_spec(_spec)
sys.modules["fetch_site"] = fetch
_spec.loader.exec_module(fetch)

# Made-up nodes around (0, 0), built at run time (in units of 0.0001 degrees); nothing
# here is a real place.
_NODES = {
    1: (0, 0),
    2: (0, 10),
    3: (10, 10),
    4: (10, 0),
    5: (2, 2),
    6: (2, 4),
    7: (4, 4),
    8: (4, 2),
    9: (-5, -5),
    10: (5, 5),
}
_NODE_XML = "".join(
    f'<node id="{i}" lat="{la * 1e-4:.4f}" lon="{lo * 1e-4:.4f}"/>'
    for i, (la, lo) in _NODES.items()
)
HALF = 5e-4
OSM = (
    '<?xml version="1.0"?>\n<osm version="0.6">\n'
    + _NODE_XML
    + """
  <way id="11"><nd ref="1"/><nd ref="2"/><nd ref="3"/></way>
  <way id="12"><nd ref="3"/><nd ref="4"/><nd ref="1"/></way>
  <way id="13"><nd ref="5"/><nd ref="6"/><nd ref="7"/><nd ref="8"/><nd ref="5"/></way>
  <way id="14"><nd ref="9"/><nd ref="10"/>
    <tag k="highway" v="secondary"/><tag k="width" v="6.5"/></way>
  <way id="15"><nd ref="5"/><nd ref="6"/><nd ref="7"/><nd ref="8"/><nd ref="5"/>
    <tag k="building" v="house"/><tag k="building:levels" v="2"/></way>
  <way id="16"><nd ref="9"/><nd ref="10"/><tag k="highway" v="path"/></way>
  <relation id="20">
    <member type="way" ref="11" role="outer"/><member type="way" ref="12" role="outer"/>
    <member type="way" ref="13" role="inner"/>
    <tag k="type" v="multipolygon"/><tag k="natural" v="water"/>
  </relation>
</osm>
"""
).encode()


def test_osm_features_join_rings_keep_holes_and_tags() -> None:
    data = fetch.parse_osm(OSM)
    feats = fetch.osm_features(data, box(-0.01, -0.01, 0.01, 0.01))
    kinds = {f["properties"]["osm"]: f["properties"] for f in feats}
    lake: Any = shape(next(f["geometry"] for f in feats if f["properties"]["osm"] == "r20"))
    assert kinds["r20"]["kind"] == "water"
    assert len(lake.interiors) == 1 and lake.area == pytest.approx(1e-6 - 4e-8, rel=1e-6)
    assert kinds["w14"] == {
        "kind": "road",
        "osm": "w14",
        "highway": "secondary",
        "width_m": 6.5,
        "width_tagged": True,
    }
    assert kinds["w16"]["width_m"] == fetch.ROAD_WIDTH_M["path"]
    assert kinds["w15"]["building"] == "house" and kinds["w15"]["levels"] == 2.0
    assert [f["properties"]["kind"] for f in feats] == sorted(
        f["properties"]["kind"] for f in feats
    )


def test_features_are_clipped_to_the_site() -> None:
    feats = fetch.osm_features(fetch.parse_osm(OSM), box(0.0, 0.0, HALF, HALF))
    for f in feats:
        minx, miny, maxx, maxy = shape(f["geometry"]).bounds
        assert minx >= -1e-9 and miny >= -1e-9 and maxx <= HALF + 1e-9 and maxy <= HALF + 1e-9


def test_write_dem_keeps_a_square_odd_grid(tmp_path: Path) -> None:
    src = tmp_path / "in.asc"
    src.write_text(
        "ncols 3\nnrows 3\nxllcorner -30\nyllcorner -30\ncellsize 20\nNODATA_value -9999\n"
        "1 2 3\n4 5 6\n7 8 9\n"
    )
    site = fetch.SiteDef("t", 0.0, 0.0, 0.0, 100.0, "copernicus-glo30", 20.0, False, tmp_path, True)
    assert fetch.write_dem(src, tmp_path / "dem.asc", site) == 5.0
    lines = (tmp_path / "dem.asc").read_text().splitlines()
    assert "cell_m 20" in lines and "nodes 3" in lines and lines[-1] == "7.00 8.00 9.00"
    src.write_text(src.read_text().replace("5 6", "-9999 6"))
    with pytest.raises(fetch.FetchError, match="hål"):
        fetch.write_dem(src, tmp_path / "dem.asc", site)


def test_public_site_loads_and_refuses_photos(tmp_path: Path) -> None:
    site, login = fetch.load_site("hellasgarden", ROOT)
    assert site.public and site.dem_source == "copernicus-glo30" and login == {}
    assert site.out == ROOT / "simulation" / "gazebo" / "sites" / "hellasgarden"
    folder = tmp_path / "simulation" / "gazebo" / "sites" / "x"
    folder.mkdir(parents=True)
    (folder / "site.toml").write_text(
        "ortho = true\n[home]\nlat = 0\nlon = 0\n[area]\nhalf_size_m = 500\n[dem]\ncell_m = 20\n"
    )
    with pytest.raises(fetch.FetchError, match="lokala"):
        fetch.load_site("x", tmp_path)


def test_local_site_needs_a_login_and_writes_to_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SD_LM_USER", raising=False)
    monkeypatch.delenv("SD_LM_PASSWORD", raising=False)
    (tmp_path / "config").mkdir()
    conf = tmp_path / "config" / "site.local.toml"
    body = "[sites.mine]\northo = true\n[sites.mine.home]\nlat = 0\nlon = 0\n"
    body += "[sites.mine.area]\nhalf_size_m = 300\n[sites.mine.dem]\ncell_m = 2\n"
    conf.write_text(body)
    with pytest.raises(fetch.FetchError, match="inloggning"):
        fetch.load_site("mine", tmp_path)
    conf.write_text(body.replace("ortho = true\n", 'ortho = true\nlm_password = "x"\n'))
    with pytest.raises(fetch.FetchError, match="SD_LM_USER"):
        fetch.load_site("mine", tmp_path)
    conf.write_text(body)
    monkeypatch.setenv("SD_LM_USER", "u")
    monkeypatch.setenv("SD_LM_PASSWORD", "p")
    site, login = fetch.load_site("mine", tmp_path)
    assert not site.public and site.out == tmp_path / "build" / "sites" / "mine"
    assert site.dem_source == "lantmateriet-mhm" and login == {"user": "u", "password": "p"}
    # The login goes to GDAL's environment, never to a command line or the written files.
    assert fetch.lm_env(login)["GDAL_HTTP_USERPWD"] == "u:p"
    site.out.mkdir(parents=True)
    fetch.write_local_site_toml(site)
    written = (site.out / "site.toml").read_text()
    assert "lantmateriet-mhm" in written and 'surface = "dtm"' in written
    assert (
        "u"
        not in written.split("\n", 1)[1]
        .replace("surface", "")
        .replace("source", "")
        .split("title")[0]
    )
    assert "lm_user" not in written and "lm_password" not in written and ":p" not in written


@pytest.mark.parametrize("argv", [[], ["site=../x"], ["site=Big"], ["foo=1"]])
def test_bad_arguments_exit_2(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert fetch.main(argv) == 2
    assert capsys.readouterr().err


def test_stac_hrefs_must_be_lantmateriet_https() -> None:
    good = {"assets": {"data": {"href": f"https://{fetch.LM_HOST}/hojd/data/x.tif"}}}
    assert fetch._lm_href(good).endswith("x.tif")
    for href in ("http://dl1.lantmateriet.se/x.tif", "https://evil.example/x.tif"):
        with pytest.raises(fetch.FetchError):
            fetch._lm_href({"assets": {"data": {"href": href}}})


def test_public_site_may_take_lantmateriet_heights_as_bare_ground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SD_LM_USER", raising=False)
    monkeypatch.delenv("SD_LM_PASSWORD", raising=False)
    folder = tmp_path / "simulation" / "gazebo" / "sites" / "x"
    folder.mkdir(parents=True)
    body = "[home]\nlat = 0\nlon = 0\n[area]\nhalf_size_m = 500\n[dem]\ncell_m = 20\n"
    body += 'source = "lantmateriet-mhm"\n'
    (folder / "site.toml").write_text(body)
    with pytest.raises(fetch.FetchError, match="dtm"):
        fetch.load_site("x", tmp_path)
    (folder / "site.toml").write_text(body + 'surface = "dtm"\n')
    with pytest.raises(fetch.FetchError, match="inloggning"):
        fetch.load_site("x", tmp_path)
    monkeypatch.setenv("SD_LM_USER", "u")
    monkeypatch.setenv("SD_LM_PASSWORD", "p")
    site, login = fetch.load_site("x", tmp_path)
    assert site.public and site.dem_source == "lantmateriet-mhm" and login["user"] == "u"
    assert site.out == folder
