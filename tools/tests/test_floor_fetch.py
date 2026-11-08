"""tools/terrain/fetch_floor_map.py without the network: the projection, the area, the
catalogue, the downloads, the grids, the power lines and a whole run against a fake
Lantmäteriet and a fake Overpass. Nothing here is real map data."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from sd_safety.floor import floor_at
from sd_safety.floor_pack import load_floor_pack

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "terrain" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("fetch_site")
ff: Any = _load("fetch_floor_map")

# The public simulator site (tools/allowed_coordinates.toml): a small area on its meadow.
HOME_LAT, HOME_LON = 59.28842, 18.15953
USER, PASSWORD = "test-user", "not-a-real-password"


def _area(path: Path, half_e_m: float = 30.0, half_n_m: float = 20.0) -> Path:
    dlat = half_n_m / 111_000
    dlon = half_e_m / (111_000 * np.cos(np.radians(HOME_LAT)))
    ring = [
        [HOME_LON - dlon, HOME_LAT - dlat],
        [HOME_LON + dlon, HOME_LAT - dlat],
        [HOME_LON + dlon, HOME_LAT + dlat],
        [HOME_LON - dlon, HOME_LAT + dlat],
        [HOME_LON - dlon, HOME_LAT - dlat],
    ]
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}), "utf-8")
    return path


# --- SWEREF 99 TM ---------------------------------------------------------------------------

# Reference values from PROJ (pyproj 3.7.1, EPSG:4326 -> EPSG:3006), 2026-11-03.
REFERENCE = [
    (59.28842, 18.15953, 6576436.043594649, 679958.2997596966),
    (66.0, 20.0, 7328955.759433632, 726739.9778164843),
    (56.0, 13.0, 6207884.602847646, 375273.8482556883),
]


@pytest.mark.parametrize(("lat", "lon", "north", "east"), REFERENCE)
def test_sweref_matches_proj(lat: float, lon: float, north: float, east: float) -> None:
    n, e = ff.to_sweref(np.array([lat]), np.array([lon]))
    assert abs(float(n[0]) - north) < 0.001 and abs(float(e[0]) - east) < 0.001
    la, lo = ff.from_sweref(np.array([north]), np.array([east]))
    assert abs(float(la[0]) - lat) < 1e-8 and abs(float(lo[0]) - lon) < 1e-8


def test_sweref_round_trip_across_sweden() -> None:
    rng = np.random.default_rng(1)
    lat, lon = rng.uniform(55, 69.5, 500), rng.uniform(10, 24.5, 500)
    la, lo = ff.from_sweref(*ff.to_sweref(lat, lon))
    assert np.abs(la - lat).max() < 1e-9 and np.abs(lo - lon).max() < 1e-9


# --- the area -------------------------------------------------------------------------------


def test_area_becomes_a_grid_with_a_margin(tmp_path: Path) -> None:
    grid = ff.read_area(_area(tmp_path / "a.geojson", 30.0, 20.0))
    assert grid.origin.lat == pytest.approx(HOME_LAT) and grid.origin.lon == pytest.approx(HOME_LON)
    # 60 x 40 m (a little more: 111 km per degree is rounded) plus 20 m each side, even.
    assert grid.cols % 2 == 0 and grid.rows % 2 == 0
    assert 100 <= grid.cols <= 102 and 80 <= grid.rows <= 82


def test_grid_cells_are_the_floors_cells(tmp_path: Path) -> None:
    """A point in a cell lands in that cell here and in sd_safety.floor alike."""
    grid = ff.read_area(_area(tmp_path / "a.geojson"))
    rows = np.array([0, 5, grid.rows - 1])
    cols = np.array([0, 7, grid.cols - 1])
    east = cols + 0.3 - grid.cols / 2
    north = grid.rows / 2 - rows - 0.7
    lat, lon = grid.latlon(east, north)
    r, c, inside = grid.cell_of(*grid.plane(lat, lon))
    assert inside.all() and (r == rows).all() and (c == cols).all()
    centre_lat, centre_lon = grid.centres(5, 6)
    assert centre_lat.shape == (1, grid.cols)
    r, c, _ = grid.cell_of(*grid.plane(centre_lat, centre_lon))
    assert (r == 5).all() and (c == np.arange(grid.cols)).all()


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"type": "Point", "coordinates": [18.2, 59.3]}, "Polygon"),
        ({"type": "FeatureCollection", "features": []}, "ingen polygon"),
        (
            {"type": "Polygon", "coordinates": [[[59.3, 18.2], [59.4, 18.2], [59.4, 18.3]]]},
            "Sverige",
        ),
        (
            {"type": "Polygon", "coordinates": [[[18.0, 59.0], [18.3, 59.0], [18.3, 59.05]]]},
            "10 x 10",
        ),
        ({"type": "Polygon", "coordinates": [[["x", 59.0]]]}, "koordinater"),
        ({"type": "MultiPolygon", "coordinates": None}, "koordinater"),
    ],
)
def test_area_refused(tmp_path: Path, data: Any, message: str) -> None:
    path = tmp_path / "a.geojson"
    path.write_text(json.dumps(data), "utf-8")
    with pytest.raises(ff.FetchError, match=message) as err:
        ff.read_area(path)
    assert err.value.code == 2


def test_area_unreadable(tmp_path: Path) -> None:
    with pytest.raises(ff.FetchError, match="går inte att läsa") as err:
        ff.read_area(tmp_path / "missing.geojson")
    assert err.value.code == 2


def test_multipolygon_area(tmp_path: Path) -> None:
    ring = [[18.159, 59.288], [18.160, 59.288], [18.160, 59.289], [18.159, 59.288]]
    path = tmp_path / "a.geojson"
    path.write_text(json.dumps({"type": "MultiPolygon", "coordinates": [[ring]]}), "utf-8")
    assert ff.read_area(path).cols > 40


# --- a fake network ---------------------------------------------------------------------------


class Response(io.BytesIO):
    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class FakeNet:
    """Answers requests by URL; records each request."""

    def __init__(self) -> None:
        self.routes: dict[str, Callable[[urllib.request.Request], bytes]] = {}
        self.seen: list[urllib.request.Request] = []

    def __call__(self, req: urllib.request.Request) -> Response:
        self.seen.append(req)
        url = req.full_url
        for prefix, answer in self.routes.items():
            if url.startswith(prefix):
                return Response(answer(req))
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]


@pytest.fixture
def net(monkeypatch: pytest.MonkeyPatch) -> FakeNet:
    fake = FakeNet()
    monkeypatch.setattr(ff, "_urlopen", fake)
    monkeypatch.setattr(ff.time, "sleep", lambda _s: None)
    return fake


def _item(
    item_id: str, href: str, bbox: list[float], when: str, data: bytes | None = None
) -> dict[str, Any]:
    asset: dict[str, Any] = {"href": href, "proj:bbox": bbox, "proj:code": "EPSG:5845"}
    if data is not None:
        asset["file:size"] = len(data)
        asset["file:checksum"] = "1220" + hashlib.sha256(data).hexdigest()
    return {"id": item_id, "assets": {"data": asset}, "properties": {"datetime": when}}


LM = "https://dl1.lantmateriet.se/hojd/data"


# --- the catalogue ----------------------------------------------------------------------------


def test_stac_search_follows_the_next_page(net: FakeNet) -> None:
    pages = [
        {
            "features": [_item("a", f"{LM}/a.laz", [0, 0, 1, 1], "2021-01-01")],
            "links": [
                {
                    "rel": "next",
                    "method": "POST",
                    "href": "https://api.lantmateriet.se/stac-hojd/v1/search",
                    "body": {"token": "next:abc"},
                }
            ],
        },
        {"features": [_item("b", f"{LM}/b.laz", [0, 0, 1, 1], "2022-01-01")], "links": []},
    ]
    net.routes[ff.STAC] = lambda req: json.dumps(pages.pop(0)).encode()
    items = ff.stac_search("dsm-skoglig-copc", (18.0, 59.0, 18.1, 59.1))
    assert [i["id"] for i in items] == ["a", "b"]
    first, second = (json.loads(cast(bytes, r.data)) for r in net.seen)
    assert first["collections"] == ["dsm-skoglig-copc"] and first["bbox"] == [
        18.0,
        59.0,
        18.1,
        59.1,
    ]
    assert second == {"token": "next:abc"}
    assert not any(r.has_header("Authorization") for r in net.seen)  # the catalogue is open


@pytest.mark.parametrize(
    "link",
    [
        {"rel": "next", "method": "POST", "href": "https://elsewhere.example/search", "body": {}},
        {"rel": "next", "method": "GET", "href": "https://api.lantmateriet.se/stac-hojd/v1/x"},
    ],
)
def test_stac_search_refuses_odd_pages(net: FakeNet, link: dict[str, Any]) -> None:
    net.routes[ff.STAC] = lambda req: json.dumps({"features": [], "links": [link]}).encode()
    with pytest.raises(ff.FetchError):
        ff.stac_search("dtm-cog", (18.0, 59.0, 18.1, 59.1))


def test_stac_search_gives_up_on_endless_pages(net: FakeNet) -> None:
    link = {"rel": "next", "method": "POST", "href": f"{ff.STAC}/search", "body": {"t": 1}}
    net.routes[ff.STAC] = lambda req: json.dumps({"features": [], "links": [link]}).encode()
    with pytest.raises(ff.FetchError, match="sidor"):
        ff.stac_search("dtm-cog", (18.0, 59.0, 18.1, 59.1))


def test_stac_search_not_json(net: FakeNet) -> None:
    net.routes[ff.STAC] = lambda req: b"<html>"
    with pytest.raises(ff.FetchError, match="JSON"):
        ff.stac_search("dtm-cog", (18.0, 59.0, 18.1, 59.1))


def test_choose_tiles_newest_scan_per_tile_within_the_area() -> None:
    within = (1000.0, 1000.0, 3000.0, 3000.0)
    items = [
        _item("old", f"{LM}/old.laz", [0, 0, 2500, 2500], "2019-04-01", b"x"),
        _item("new", f"{LM}/new.laz", [0, 0, 2500, 2500], "2023-04-01", b"y"),
        _item("east", f"{LM}/east.laz", [2500, 0, 5000, 2500], "2021-04-01"),
        _item("far", f"{LM}/far.laz", [5000, 0, 7500, 2500], "2024-04-01"),
    ]
    tiles = ff.choose_tiles(items, within)
    assert [t.item_id for t in tiles] == ["new", "east"]
    assert tiles[0].year == 2023 and tiles[0].size == 1 and tiles[0].file_name == "new.laz"
    assert tiles[0].sha256 == hashlib.sha256(b"y").hexdigest()
    assert tiles[1].sha256 is None and tiles[1].size is None


@pytest.mark.parametrize(
    "item",
    [
        _item("x", "https://elsewhere.example/x.laz", [0, 0, 1, 1], "2021-01-01"),
        _item("x", "http://dl1.lantmateriet.se/x.laz", [0, 0, 1, 1], "2021-01-01"),
        _item("x", f"{LM}/x.laz", [0, 0, 1], "2021-01-01"),
        _item("x", f"{LM}/x.laz", [0, 0, 1, 1], ""),
        {
            "id": "x",
            "assets": {
                "data": {"href": f"{LM}/x.laz", "proj:bbox": [0, 0, 1, 1], "proj:code": "EPSG:4326"}
            },
        },
    ],
)
def test_odd_catalogue_items_refused(item: dict[str, Any]) -> None:
    with pytest.raises(ff.FetchError):
        ff.choose_tiles([item], (0.0, 0.0, 1.0, 1.0))


# --- downloads --------------------------------------------------------------------------------


def _tile(data: bytes, name: str = "t.copc.laz") -> Any:
    return ff.tile_of(_item("t", f"{LM}/{name}", [0, 0, 1, 1], "2021-04-01", data))


LOGIN = {"user": USER, "password": PASSWORD}


def test_download_with_the_login_to_lantmateriet_only(tmp_path: Path, net: FakeNet) -> None:
    net.routes[LM] = lambda req: b"points"
    path = ff.download(_tile(b"points"), tmp_path / "cache", LOGIN)
    assert path.read_bytes() == b"points"
    req = net.seen[0]
    token = base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()
    # Unredirected: urllib never copies it onto a redirect.
    assert req.unredirected_hdrs == {"Authorization": f"Basic {token}"}
    assert "Authorization" not in req.headers
    assert sorted(p.name for p in (tmp_path / "cache").iterdir()) == ["t.copc.laz"]


def test_an_intact_cached_file_is_not_fetched_again(tmp_path: Path, net: FakeNet) -> None:
    (tmp_path / "t.copc.laz").write_bytes(b"points")
    assert ff.download(_tile(b"points"), tmp_path, LOGIN).read_bytes() == b"points"
    assert net.seen == []
    (tmp_path / "t.copc.laz").write_bytes(b"broken")
    net.routes[LM] = lambda req: b"points"
    assert ff.download(_tile(b"points"), tmp_path, LOGIN).read_bytes() == b"points"


def test_a_file_that_does_not_match_the_catalogue_is_refused(tmp_path: Path, net: FakeNet) -> None:
    net.routes[LM] = lambda req: b"altered"
    with pytest.raises(ff.FetchError, match="SHA-256"):
        ff.download(_tile(b"points"), tmp_path, LOGIN)
    assert list(tmp_path.iterdir()) == []


def _fail(code: int) -> Callable[[urllib.request.Request], bytes]:
    def answer(req: urllib.request.Request) -> bytes:
        raise urllib.error.HTTPError(req.full_url, code, "x", {}, None)  # type: ignore[arg-type]

    return answer


@pytest.mark.parametrize("code", [401, 403])
def test_a_refused_login_says_what_to_do(
    tmp_path: Path, net: FakeNet, code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    net.routes[LM] = _fail(code)
    with pytest.raises(ff.FetchError, match="beställt Laserdata skog") as err:
        ff.download(_tile(b"points"), tmp_path, LOGIN)
    assert err.value.code == 1
    assert PASSWORD not in str(err.value) and PASSWORD not in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_download_tries_again_then_gives_up(tmp_path: Path, net: FakeNet) -> None:
    calls: list[int] = []

    def flaky(req: urllib.request.Request) -> bytes:
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError("connection reset")
        if len(calls) == 2:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, None)  # type: ignore[arg-type]
        return b"points"

    net.routes[LM] = flaky
    assert ff.download(_tile(b"points"), tmp_path, LOGIN).read_bytes() == b"points"
    net.routes[LM] = _fail(500)
    with pytest.raises(ff.FetchError, match="kunde inte hämta"):
        ff.download(_tile(b"other", "u.copc.laz"), tmp_path, LOGIN)
    net.routes[LM] = _fail(404)
    with pytest.raises(ff.FetchError, match="kunde inte hämta"):
        ff.download(_tile(b"other", "u.copc.laz"), tmp_path, LOGIN)


def test_fetch_errors(net: FakeNet) -> None:
    net.routes[ff.OVERPASS] = _fail(403)  # no login sent: not the Geotorget message
    with pytest.raises(ff.FetchError, match="HTTP Error 403"):
        ff._fetch(ff._request(ff.OVERPASS))
    net.routes[ff.OVERPASS] = _fail(400)
    with pytest.raises(ff.FetchError, match="kunde inte hämta"):
        ff._fetch(ff._request(ff.OVERPASS))
    net.routes[ff.OVERPASS] = _fail(504)
    with pytest.raises(ff.FetchError, match="kunde inte hämta"):
        ff._fetch(ff._request(ff.OVERPASS))


def test_redirects_only_to_https() -> None:
    handler = ff._HttpsRedirectsOnly()
    req = ff._request(f"{LM}/x", login=LOGIN)
    with pytest.raises(ff.FetchError, match="https"):
        handler.redirect_request(req, None, 302, "Found", {}, "http://dl1.lantmateriet.se/x")
    moved = handler.redirect_request(req, None, 302, "Found", {}, "https://cdn.example/x")
    assert moved is not None and not moved.has_header("Authorization")


# --- the grids --------------------------------------------------------------------------------


def _small_grid() -> Any:
    return ff.Grid(origin=ff.LatLon(lat=HOME_LAT, lon=HOME_LON), cols=10, rows=8)


def _points_at(grid: Any, cells: list[tuple[int, int, float, int]]) -> tuple[Any, Any, Any, Any]:
    """Points (easting, northing, z, class) at the centres of (row, col) cells."""
    rows = np.array([c[0] for c in cells], dtype=np.float64)
    cols = np.array([c[1] for c in cells], dtype=np.float64)
    lat, lon = grid.latlon(cols + 0.5 - grid.cols / 2, grid.rows / 2 - rows - 0.5)
    north, east = ff.to_sweref(lat, lon)
    z = np.array([c[2] for c in cells])
    cls = np.array([c[3] for c in cells], dtype=np.uint8)
    return east, north, z, cls


def test_canopy_takes_the_highest_point_but_never_noise() -> None:
    grid = _small_grid()
    canopy = ff.Canopy(grid)
    canopy.add(*_points_at(grid, [(1, 1, 50.0, 1), (1, 1, 55.21, 1), (1, 1, 300.0, 7)]), year=2021)
    canopy.add(*_points_at(grid, [(1, 1, 51.0, 2), (2, 2, 40.0, 18), (3, 3, 30.0, 2)]), year=2019)
    # A point far outside the pack is dropped.
    canopy.add(np.array([1.0]), np.array([1.0]), np.array([10.0]), np.array([1]), year=2019)
    assert canopy.top[1, 1] == pytest.approx(55.21, abs=1e-4)
    assert np.isneginf(canopy.top[2, 2])  # only noise there
    assert canopy.year[1, 1] == 2019 and canopy.year[3, 3] == 2019 and canopy.year[2, 2] == 0
    assert canopy.points == 4

    dtm = np.full((8, 10), 3000, np.int32)  # 30.00 m
    dtm[3, 3] = ff.DTM_NODATA_CM
    chm = ff.tree_heights(canopy.top, dtm)
    assert chm.dtype == np.int16
    assert chm[1, 1] == 253  # 25.21 m above the ground: rounded up to 25.3
    assert chm[2, 2] == -1 and chm[0, 0] == -1  # no points
    assert chm[3, 3] == -1  # no ground


def test_tree_heights_round_up_and_never_below_ground() -> None:
    dtm = np.array([[3000, 3000, 3000]], np.int32)
    top = np.array([[30.0, 29.0, 30.4]], np.float32)
    assert ff.tree_heights(top, dtm).tolist() == [[0, 0, 4]]


def test_scan_squares_take_the_oldest_known_year() -> None:
    year = np.zeros((5, 5), np.uint16)
    year[0, 0], year[1, 1], year[4, 4] = 2021, 2019, 2023
    squares = ff.scan_squares(year, 3)
    assert squares.dtype == np.uint16
    assert squares.tolist() == [[2019, 0], [0, 2023]]


def test_ground_from_the_rasters_at_the_cell_centres() -> None:
    grid = _small_grid()
    west, south, east, north = grid.sweref_bbox(5.0)
    w, h = int(east - west) + 1, int(north - south) + 1
    values = np.full((h, w), 31.234, np.float32)
    values[:, : w // 2] = -9999.0  # the west half has no ground in this tile
    first = ff.Raster(values, float(int(west)), float(north), 1.0, -9999.0)
    second = ff.Raster(np.full((h, w), 12.0, np.float32), float(int(west)), north, 1.0, None)
    dtm = ff.sample_ground(grid, [first, second])
    assert dtm[:, -1].tolist() == [3123] * 8  # the first raster wins where it has ground
    assert dtm[:, 0].tolist() == [1200] * 8  # the second fills its gap
    assert (ff.sample_ground(grid, [first])[:, 0] == ff.DTM_NODATA_CM).all()
    far = ff.Raster(values, 0.0, 100.0, 1.0, None)
    assert (ff.sample_ground(grid, [far]) == ff.DTM_NODATA_CM).all()


# --- power lines --------------------------------------------------------------------------------


def test_overpass_query_asks_for_lines_towers_and_poles() -> None:
    q = ff.overpass_query((18.155, 59.286, 18.165, 59.291))
    assert "(59.286000,18.155000,59.291000,18.165000)" in q
    assert 'way["power"~"^(line|minor_line)$"]' in q and 'node["power"~"^(tower|pole)$"]' in q
    assert q.startswith("[out:json]") and q.endswith("out geom;")


def test_parse_overpass() -> None:
    raw = json.dumps(
        {
            "elements": [
                {
                    "type": "way",
                    "id": 1,
                    "tags": {"power": "line", "voltage": "20000"},
                    "geometry": [{"lat": 59.288, "lon": 18.159}, {"lat": 59.289, "lon": 18.160}],
                },
                {
                    "type": "node",
                    "id": 2,
                    "lat": 59.2885,
                    "lon": 18.1595,
                    "tags": {"power": "pole"},
                },
                {"type": "way", "id": 3, "tags": {"power": "line"}},  # no geometry
                {"type": "relation", "id": 4},
            ]
        }
    ).encode()
    lines = ff.parse_overpass(raw)
    assert [len(line.points) for line in lines] == [2, 1]
    assert lines[0].properties == {
        "source": "OpenStreetMap",
        "osm": "way/1",
        "power": "line",
        "voltage": "20000",
    }
    assert lines[1].properties["osm"] == "node/2"


@pytest.mark.parametrize(
    "raw",
    [b"<xml/>", b"[]", json.dumps({"remark": "runtime error: Query timed out"}).encode()],
)
def test_parse_overpass_refuses_errors(raw: bytes) -> None:
    with pytest.raises(ff.FetchError, match="Overpass"):
        ff.parse_overpass(raw)


# --- a whole run ----------------------------------------------------------------------------------


GROUND_M = 31.0
TREE_M = 18.0


@pytest.fixture
def lantmateriet(tmp_path: Path, net: FakeNet, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A fake Lantmäteriet for the area in ``tmp_path/a.geojson``: flat ground at
    ``GROUND_M``, one tree ``TREE_M`` tall 5 m east of the origin, scanned in 2021, and
    noise; a fake Overpass with one power line 25 m north of the origin."""
    area = _area(tmp_path / "a.geojson")
    grid = ff.read_area(area)
    west, south, east, north = grid.sweref_bbox(50.0)
    copc_bbox = [west - 10, south - 10, east + 10, north + 10]
    copc_bytes, dtm_bytes = b"fake copc", b"fake cog"
    stac = {
        ff.COPC_COLLECTION: [
            _item(
                "21c031-1",
                f"{LM}/pointcloud/a.copc.laz",
                copc_bbox,
                "2021-03-23T00:00:00Z",
                copc_bytes,
            ),
            _item(
                "17c001-1", f"{LM}/pointcloud/b.copc.laz", copc_bbox, "2017-03-23T00:00:00Z", b"old"
            ),
        ],
        ff.DTM_COLLECTION: [
            _item("658_68", f"{LM}/grid/m.tif", copc_bbox, "2023-04-27T00:00:00Z", dtm_bytes)
        ],
    }

    def search(req: urllib.request.Request) -> bytes:
        body = json.loads(cast(bytes, req.data))
        return json.dumps({"features": stac[body["collections"][0]], "links": []}).encode()

    files = {"a.copc.laz": copc_bytes, "m.tif": dtm_bytes}
    net.routes[ff.STAC] = search
    net.routes[LM] = lambda req: files[req.full_url.rsplit("/", 1)[-1]]
    lat0, lon0 = grid.latlon(np.array([-40.0, 40.0]), np.array([25.0, 25.0]))
    overpass = {
        "elements": [
            {
                "type": "way",
                "id": 7,
                "tags": {"power": "minor_line"},
                "geometry": [
                    {"lat": float(a), "lon": float(o)} for a, o in zip(lat0, lon0, strict=True)
                ],
            }
        ]
    }
    net.routes[ff.OVERPASS] = lambda req: json.dumps(overpass).encode()

    w, s = int(west) - 1, int(south) - 1
    size = (int(north) + 2 - s, int(east) + 2 - w)
    raster = ff.Raster(
        np.full(size, GROUND_M, np.float32), float(w), float(s + size[0]), 1.0, -9999.0
    )
    monkeypatch.setattr(ff, "read_raster", lambda path: raster)

    def points(path: Path) -> Iterator[tuple[Any, Any, Any, Any]]:
        assert path.read_bytes() == copc_bytes
        cells = [(r, c, GROUND_M + 0.2, 2) for r in range(grid.rows) for c in range(grid.cols)]
        tree_r, tree_c = grid.rows // 2 - 1, grid.cols // 2 + 5
        cells += [(tree_r, tree_c, GROUND_M + TREE_M, 1), (0, 0, 900.0, 7)]
        e, n, z, cls = _points_at(grid, cells)
        half = len(z) // 2
        yield e[:half], n[:half], z[:half], cls[:half]
        yield e[half:], n[half:], z[half:], cls[half:]

    monkeypatch.setattr(ff, "read_points", points)
    monkeypatch.setenv("SD_LM_USER", USER)
    monkeypatch.setenv("SD_LM_PASSWORD", PASSWORD)
    return {"area": area, "grid": grid, "out": tmp_path / "floor"}


def _floor_m(pack: Any, east: float, north: float) -> Any:
    lat, lon = pack.plane.latlon(east, north)
    return floor_at(
        pack, lat, lon, flight_date=date(2026, 6, 1), laser=None, start_alt_amsl_m=GROUND_M + 80
    )


def _run(lm: dict[str, Any], *extra: str) -> int:
    argv = [f"area={lm['area']}", "name=meadow", f"out={lm['out']}", *extra]
    return ff.main(argv)


def test_a_whole_run(
    lantmateriet: dict[str, Any], net: FakeNet, capsys: pytest.CaptureFixture[str]
) -> None:
    lm = lantmateriet
    assert _run(lm) == 0
    out = capsys.readouterr()
    folder = lm["out"] / "meadow"
    pack = load_floor_pack(folder)
    grid = lm["grid"]
    assert pack.name == "meadow" and pack.dtm_cm.shape == (grid.rows, grid.cols)
    assert pack.origin == grid.origin
    assert (pack.dtm_cm == round(GROUND_M * 100)).all()
    # Ground points 0.2 m up everywhere, one tree, noise left out.
    assert pack.chm_dm[grid.rows // 2 - 1, grid.cols // 2 + 5] == round(TREE_M * 10)
    assert int(np.count_nonzero(pack.chm_dm == 2)) == pack.chm_dm.size - 1
    assert pack.scan_cell_m == ff.SCAN_CELL_M and (pack.scan_year == 2021).all()
    assert "Lantmäteriet" in pack.attribution and "OpenStreetMap" in pack.attribution
    assert len(pack.lines) == 1
    # The floor reads it: the tree (+ 2 m, + 2.5 m growth) and the line 15 m from (0, 10).
    tree = _floor_m(pack, 5.0, -9.0)
    assert tree.tree_m == pytest.approx(TREE_M + 2.0 + 2.5)
    assert _floor_m(pack, 0.0, 10.0).reason == "powerline"
    assert tree.reason == "map"
    meta = (folder / "floor_pack.toml").read_text("utf-8")
    assert "21c031-1 (2021-03-23)" in meta and "17c001-1" not in meta
    assert pack.sha256 in out.out
    # The point cloud is gone; the ground model stays in the cache.
    assert sorted(p.name for p in (lm["out"] / ".cache").iterdir()) == ["m.tif"]
    # The login went to Lantmäteriet's download host only, and nowhere on disk or screen.
    for req in net.seen:
        sent = req.has_header("Authorization")
        assert sent == (urllib.parse.urlparse(req.full_url).hostname == ff.LM_HOST)
    assert PASSWORD not in out.out + out.err
    for path in lm["out"].rglob("*"):
        if path.is_file():
            assert PASSWORD.encode() not in path.read_bytes()


def test_keep_cache_keeps_the_point_cloud_and_warns(
    lantmateriet: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(lantmateriet, "keep_cache=1") == 0
    assert "personuppgifter" in capsys.readouterr().err
    assert (lantmateriet["out"] / ".cache" / "a.copc.laz").is_file()


def test_without_lines_says_so(
    lantmateriet: dict[str, Any], net: FakeNet, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(lantmateriet, "lines=0") == 0
    assert "inga kraftledningar" in capsys.readouterr().err
    pack = load_floor_pack(lantmateriet["out"] / "meadow")
    assert pack.lines == () and "OpenStreetMap" not in pack.attribution
    assert not any(r.full_url.startswith(ff.OVERPASS) for r in net.seen)


def test_a_failed_overpass_stops_the_run(lantmateriet: dict[str, Any], net: FakeNet) -> None:
    net.routes[ff.OVERPASS] = lambda req: json.dumps(
        {"remark": "runtime error: out of memory"}
    ).encode()
    assert _run(lantmateriet) == 1
    assert not (lantmateriet["out"] / "meadow").exists()


def test_the_point_cloud_is_deleted_even_when_reading_fails(
    lantmateriet: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(path: Path) -> Iterator[Any]:
        raise ff.FetchError("trasig fil")
        yield

    monkeypatch.setattr(ff, "read_points", broken)
    assert _run(lantmateriet) == 1
    assert not (lantmateriet["out"] / ".cache" / "a.copc.laz").exists()


def test_no_data_for_the_area(lantmateriet: dict[str, Any], net: FakeNet) -> None:
    net.routes[ff.STAC] = lambda req: json.dumps({"features": [], "links": []}).encode()
    assert _run(lantmateriet) == 1


def test_a_pack_that_does_not_load_is_not_written(
    lantmateriet: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ff,
        "sample_ground",
        lambda grid, rasters: np.full((grid.rows, grid.cols), ff.DTM_NODATA_CM, np.int32),
    )
    assert _run(lantmateriet) == 1
    assert not (lantmateriet["out"] / "meadow").exists()


# --- settings ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["name=meadow"],  # no area
        ["area=a.geojson"],  # no name
        ["area=a.geojson", "name=Meadow"],
        ["area=a.geojson", "name=meadow", "colour=red"],
        ["area=a.geojson", "name=meadow", "keep_cache=yes"],
        ["area=a.geojson", "name=meadow", "lines=2"],
        ["push", "name=meadow", "host=-oProxyCommand=x", "dir=/srv"],
        ["push", "name=meadow", "host=pi@drone", "dir=$(reboot)"],
    ],
)
def test_bad_arguments(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert ff.main(argv) == 2
    assert capsys.readouterr().err


def test_floor_dir(tmp_path: Path) -> None:
    assert ff.floor_dir({"out": str(tmp_path)}, {}, tmp_path) == tmp_path.resolve()
    assert (
        ff.floor_dir({}, {"SD_DATA_DIR": str(tmp_path / "d")}, tmp_path)
        == (tmp_path / "d" / "floor").resolve()
    )
    (tmp_path / ".env").write_text("OTHER=1\nSD_DATA_DIR=/srv/sd\n", "utf-8")
    assert ff.floor_dir({}, {}, tmp_path) == Path("/srv/sd/floor")
    (tmp_path / ".env").write_text("OTHER=1\n", "utf-8")
    with pytest.raises(ff.FetchError, match="setup-config") as err:
        ff.floor_dir({}, {}, tmp_path)
    assert err.value.code == 2


def test_login_is_asked_for_only_in_a_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SD_LM_USER", raising=False)
    monkeypatch.delenv("SD_LM_PASSWORD", raising=False)
    monkeypatch.setattr(ff.sys.stdin, "isatty", lambda: False, raising=False)
    with pytest.raises(ff.FetchError, match="geotorget") as err:
        ff.get_login(tmp_path)
    assert err.value.code == 2
    monkeypatch.setattr(ff.sys, "stdin", type("Tty", (), {"isatty": staticmethod(lambda: True)})())
    monkeypatch.setattr("builtins.input", lambda _p: USER)
    monkeypatch.setattr(ff.getpass, "getpass", lambda _p: PASSWORD)
    assert ff.get_login(tmp_path) == {"user": USER, "password": PASSWORD}
    monkeypatch.setattr(ff.getpass, "getpass", lambda _p: "")
    with pytest.raises(ff.FetchError):
        ff.get_login(tmp_path)


# --- push -----------------------------------------------------------------------------------------


def test_push_checks_the_pack_then_copies_it(
    lantmateriet: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = lantmateriet["out"]
    calls: list[list[str]] = []
    monkeypatch.setattr(
        ff.subprocess,
        "run",
        lambda argv, check: calls.append(argv) or subprocess.CompletedProcess(argv, 0),
    )
    push = ["push", "name=meadow", f"out={out}", "host=pi@drone.local", "dir=/home/pi/state/"]
    assert ff.main(push) == 1  # no pack yet
    assert calls == []
    assert _run(lantmateriet) == 0
    assert ff.main(push) == 0
    assert calls == [
        ["ssh", "pi@drone.local", "mkdir", "-p", "/home/pi/state/floor"],
        ["scp", "-q", "-r", str(out / "meadow"), "pi@drone.local:/home/pi/state/floor/"],
    ]
    assert load_floor_pack(out / "meadow").sha256 in capsys.readouterr().out
    monkeypatch.setattr(
        ff.subprocess, "run", lambda argv, check: subprocess.CompletedProcess(argv, 255)
    )
    assert ff.main(push) == 1


# --- the real readers (only where laspy and tifffile are installed, never in CI) -------------


def test_read_points_with_laspy(tmp_path: Path) -> None:
    laspy = pytest.importorskip("laspy")
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets, header.scales = [680000.0, 6576000.0, 0.0], [0.01, 0.01, 0.01]
    las = laspy.LasData(header)
    las.x, las.y, las.z = (
        np.array([680001.0, 680002.0]),
        np.array([6576001.0, 6576002.0]),
        np.array([40.0, 41.5]),
    )
    las.classification = np.array([2, 7], dtype=np.uint8)
    las.write(tmp_path / "p.laz")  # compressed, like the COPC files
    chunks = list(ff.read_points(tmp_path / "p.laz"))
    e, _n, z, cls = (np.concatenate(parts) for parts in zip(*chunks, strict=True))
    assert (
        e.tolist() == [680001.0, 680002.0] and z.tolist() == [40.0, 41.5] and cls.tolist() == [2, 7]
    )


def test_read_raster_with_tifffile(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    tifffile.imwrite(
        tmp_path / "m.tif",
        values,
        extratags=[
            (33550, 12, 3, (1.0, 1.0, 0.0)),
            (33922, 12, 6, (0.0, 0.0, 0.0, 680000.0, 6580000.0, 0.0)),
            (42113, 2, 0, "-9999"),
        ],
    )
    r = ff.read_raster(tmp_path / "m.tif")
    assert (r.west, r.north, r.cell, r.nodata) == (680000.0, 6580000.0, 1.0, -9999.0)
    assert r.values.tolist() == values.tolist()


def test_the_example_area_is_an_area() -> None:
    grid = ff.read_area(ROOT / "config" / "floor_area.example.geojson")
    assert 600 <= grid.cols <= 700 and 560 <= grid.rows <= 650
