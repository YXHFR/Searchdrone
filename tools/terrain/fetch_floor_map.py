"""Fetch the height map of a closer look (a floor pack): ``just floor-fetch area= name=``.

The height floor (``sd_safety.floor``) needs to know, offline, how high the ground and
the trees are where the drone goes down for a closer look. This tool makes that map from
Lantmäteriet's forest laser scan (Laserdata Nedladdning, skog: a point cloud, COPC) and
ground model (Markhöjdmodell grid 1: 1 m, COG), with the owner's own Geotorget login,
plus the known power lines from OpenStreetMap. It writes a floor pack in the format
``sd_safety.floor_pack`` reads, to ``${SD_DATA_DIR}/floor/<name>/`` (``out=<folder>``
instead of ``${SD_DATA_DIR}/floor``), never into the repository.

1. The area is a polygon in a GeoJSON file the owner points at (outside the repository),
   at most 10 x 10 km. The pack is the rectangle around it plus ``MARGIN_M`` on every
   side (the floor looks 10 m around a point; outside the pack counts as unknown).
2. Lantmäteriet's open STAC catalogue (no login) names the tiles that cover the area:
   ``dsm-skoglig-copc`` (2.5 x 2.5 km) and ``dtm-cog`` (10 x 10 km). Where a tile was
   scanned more than once, the newest scan is used.
3. Only those tiles are downloaded, with the login, to ``<out>/.cache/`` (resumed from
   there when the tool is run again), and each is checked against the catalogue's
   SHA-256.
4. The tree height of a 1 m cell is its highest laser point (every class but noise)
   minus the ground model at the cell's centre, rounded up to the next decimetre. A cell
   without points, or without ground, is unknown (-1). Nothing is smoothed: the caution
   is in the floor (``sd_safety.floor``: highest tree within 10 m, unknown = 30 m
   forest, growth since the scan).
5. The scan's year is kept per ``SCAN_CELL_M`` square: the oldest scan among the cells
   that have points (``sd_safety.floor`` adds growth from that year).
6. Power lines and their towers and poles (``power`` = ``line``, ``minor_line``,
   ``tower``, ``pole``) come from OpenStreetMap through Overpass: one request per run.
   ``lines=0`` writes no lines and says so; there is no silent fallback.
7. The point clouds are deleted as soon as each is read (Lantmäteriet's terms of
   2026-05-20: they may hold personal data). ``keep_cache=1`` keeps them, with a
   warning. The ground model tiles stay in the cache.

The grids are 1 m cells in the plane tangent at the pack's origin, exactly as
``sd_safety.floor`` looks them up. Lantmäteriet's data is in SWEREF 99 TM (EPSG:3006,
heights in RH 2000); the conversion to and from latitude/longitude is Lantmäteriet's own
Gauss-Krüger formulas (``to_sweref``, ``from_sweref``), so no projection library is
needed.

The login is read from ``SD_LM_USER``/``SD_LM_PASSWORD`` or the gitignored
``config/geotorget.local.toml`` (``config/geotorget.example.toml``), as for
``just world-fetch``; when neither has one and the tool runs in a terminal, it asks. It
is sent only to Lantmäteriet's download host, never along a redirect, and never printed
or written. Lantmäteriet's data is CC BY 4.0: the pack's ``floor_pack.toml`` carries
the attribution, which the panel shows when the pack is used.

``just floor-push name=<name> host=<user@pi> dir=<state_dir>`` copies a pack, after the
loader has checked it, to ``<state_dir>/floor/<name>/`` on the onboard computer.

Runs on the owner's computer, never in CI. Needs internet and, only while it runs,
``laspy`` with ``lazrs`` (the point clouds) and ``tifffile`` (the ground model), which
the ``just`` recipes add. Exit codes: 0 done, 1 a download or a check failed, 2 bad
arguments.
"""

from __future__ import annotations

import base64
import datetime as dt
import getpass
import hashlib
import http.client
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_site
from fetch_site import ROOT, FetchError

from sd_safety.floor_pack import (
    CHM_UNKNOWN_DM,
    DTM_MAX_CM,
    DTM_MIN_CM,
    DTM_NODATA_CM,
    FloorPack,
    FloorPackError,
    PackLine,
    load_floor_pack,
    write_floor_pack,
)
from sd_schemas.base import LatLon
from sd_schemas.charter import FLOOR_PACK_NAME_PATTERN
from sd_schemas.geo import tangent_radii

SCRIPT_VERSION = 1
FETCH_KEYS = ("area", "name", "out", "keep_cache", "lines")
PUSH_KEYS = ("name", "out", "host", "dir")
STAC = fetch_site.LM_STAC_DEM
STAC_HOST = "api.lantmateriet.se"
LM_HOST = fetch_site.LM_HOST
COPC_COLLECTION = "dsm-skoglig-copc"
DTM_COLLECTION = "dtm-cog"
OVERPASS = "https://overpass-api.de/api/interpreter"
USER_AGENT = "searchdrone-floor-fetch (+https://github.com/YXHFR/Searchdrone)"
TIMEOUT_S = 120
TRIES = 3
STAC_PAGES_MAX = 50
CHUNK = 1024 * 1024
POINTS_PER_CHUNK = 2_000_000
ROWS_PER_BLOCK = 500

MAX_SIDE_M = 10_000.0
MARGIN_M = 20  # the pack reaches this far outside the area on every side
LINE_MARGIN_M = 100.0  # power lines are fetched this far outside the pack
SCAN_CELL_M = 100
NOISE_CLASSES = (7, 18)  # ASPRS low and high noise
# Sweden, generously: Lantmäteriet's data covers nothing outside.
LAT_RANGE = (55.0, 69.5)
LON_RANGE = (10.0, 24.5)

ATTRIBUTION = (
    "© Lantmäteriet, Laserdata Nedladdning, skog och Markhöjdmodell (grid 1), CC BY 4.0. "
    "Bearbetad (trädhöjd) av Searchdrone."
)
OSM_ATTRIBUTION = " Kraftledningar © OpenStreetMap-bidragsgivare, ODbL 1.0."
LOGIN_REFUSED = (
    "Inloggningen nekades. Har du beställt Laserdata skog och Markhöjdmodell i Geotorget?"
)
NO_LINES_WARNING = (
    "VARNING: lines=0, inga kraftledningar i kartpaketet. Golvet vet då inte var några "
    "ledningar går."
)
KEEP_CACHE_WARNING = (
    "VARNING: keep_cache=1, punktmolnen ligger kvar i {cache}. De kan innehålla "
    "personuppgifter (Lantmäteriets villkor): radera dem när du inte behöver dem."
)

# --- SWEREF 99 TM (Lantmäteriet's Gauss-Krüger formulas, GRS 80) -----------------------

_A = 6378137.0
_F = 1 / 298.257222101
_K0 = 0.9996
_LON0 = 15.0
_FE = 500000.0
_E2 = _F * (2 - _F)
_N = _F / (2 - _F)
_AR = _A / (1 + _N) * (1 + _N**2 / 4 + _N**4 / 64)
_BETA = (
    _N / 2 - 2 * _N**2 / 3 + 5 * _N**3 / 16 + 41 * _N**4 / 180,
    13 * _N**2 / 48 - 3 * _N**3 / 5 + 557 * _N**4 / 1440,
    61 * _N**3 / 240 - 103 * _N**4 / 140,
    49561 * _N**4 / 161280,
)
_DELTA = (
    _N / 2 - 2 * _N**2 / 3 + 37 * _N**3 / 96 - _N**4 / 360,
    _N**2 / 48 + _N**3 / 15 - 437 * _N**4 / 1440,
    17 * _N**3 / 480 - 37 * _N**4 / 840,
    4397 * _N**4 / 161280,
)

Floats = NDArray[np.float64]


def to_sweref(lat: Floats, lon: Floats) -> tuple[Floats, Floats]:
    """(northing, easting) in SWEREF 99 TM of WGS84 latitude/longitude (degrees)."""
    phi = np.radians(lat)
    s = np.sin(phi)
    a, b = _E2, (5 * _E2**2 - _E2**3) / 6
    c, d = (104 * _E2**3 - 45 * _E2**4) / 120, 1237 * _E2**4 / 1260
    conf = phi - s * np.cos(phi) * (a + b * s**2 + c * s**4 + d * s**6)
    dlon = np.radians(lon - _LON0)
    xi = np.arctan(np.tan(conf) / np.cos(dlon))
    eta = np.arctanh(np.cos(conf) * np.sin(dlon))
    x, y = xi.copy(), eta.copy()
    for j, beta in enumerate(_BETA, 1):
        x += beta * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        y += beta * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    return _K0 * _AR * x, _K0 * _AR * y + _FE


def from_sweref(north: Floats, east: Floats) -> tuple[Floats, Floats]:
    """WGS84 (latitude, longitude) in degrees of SWEREF 99 TM (northing, easting)."""
    xi = np.asarray(north, dtype=np.float64) / (_K0 * _AR)
    eta = (np.asarray(east, dtype=np.float64) - _FE) / (_K0 * _AR)
    xp, ep = xi.copy(), eta.copy()
    for j, delta in enumerate(_DELTA, 1):
        xp -= delta * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        ep -= delta * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    conf = np.arcsin(np.sin(xp) / np.cosh(ep))
    dlon = np.arctan(np.sinh(ep) / np.cos(xp))
    s = np.sin(conf)
    e2 = _E2
    a = e2 + e2**2 + e2**3 + e2**4
    b = -(7 * e2**2 + 17 * e2**3 + 30 * e2**4) / 6
    c = (224 * e2**3 + 889 * e2**4) / 120
    d = -4279 * e2**4 / 1260
    phi = conf + s * np.cos(conf) * (a + b * s**2 + c * s**4 + d * s**6)
    return np.degrees(phi), _LON0 + np.degrees(dlon)


# --- the pack's grid ------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """The pack's 1 m cells: ``cols`` east by ``rows`` north, centred on ``origin``, row
    0 north, in the plane tangent at the origin (``sd_schemas.geo.local_xy``)."""

    origin: LatLon
    cols: int
    rows: int

    def _radii(self) -> tuple[float, float]:
        return tangent_radii(self.origin.lat)

    def latlon(self, east: Floats, north: Floats) -> tuple[Floats, Floats]:
        east_r, north_r = self._radii()
        lat = self.origin.lat + np.degrees(np.asarray(north) / north_r)
        lon = self.origin.lon + np.degrees(np.asarray(east) / east_r)
        return lat, lon

    def plane(self, lat: Floats, lon: Floats) -> tuple[Floats, Floats]:
        east_r, north_r = self._radii()
        east = np.radians(np.asarray(lon) - self.origin.lon) * east_r
        north = np.radians(np.asarray(lat) - self.origin.lat) * north_r
        return east, north

    def cell_of(
        self, east: Floats, north: Floats
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
        """(row, column, inside) of the cells under points, as ``sd_safety.floor`` finds them."""
        row = np.floor(self.rows / 2 - north).astype(np.int64)
        col = np.floor(east + self.cols / 2).astype(np.int64)
        inside = (row >= 0) & (row < self.rows) & (col >= 0) & (col < self.cols)
        return row, col, inside

    def centres(self, row0: int, row1: int) -> tuple[Floats, Floats]:
        """(latitude, longitude) of the centres of rows ``row0 .. row1 - 1``."""
        north = self.rows / 2 - (np.arange(row0, row1, dtype=np.float64) + 0.5)
        east = np.arange(self.cols, dtype=np.float64) + 0.5 - self.cols / 2
        ee, nn = np.meshgrid(east, north)
        return self.latlon(ee, nn)

    def lonlat_bbox(self, extra_m: float = 0.0) -> tuple[float, float, float, float]:
        """(west, south, east, north) in degrees around the pack plus ``extra_m``."""
        lat, lon = self.latlon(
            np.array([-self.cols / 2 - extra_m, self.cols / 2 + extra_m]),
            np.array([-self.rows / 2 - extra_m, self.rows / 2 + extra_m]),
        )
        return float(lon[0]), float(lat[0]), float(lon[1]), float(lat[1])

    def sweref_bbox(self, extra_m: float = 2.0) -> tuple[float, float, float, float]:
        """(west, south, east, north) in SWEREF 99 TM that cover the pack plus ``extra_m``:
        its edges are sampled, since they bend a little in SWEREF."""
        t = np.linspace(-0.5, 0.5, 33)
        edge = np.ones_like(t) / 2
        east = np.concatenate([t, t, -edge, edge]) * self.cols
        north = np.concatenate([-edge, edge, t, t]) * self.rows
        n, e = to_sweref(*self.latlon(east, north))
        return (
            float(e.min()) - extra_m,
            float(n.min()) - extra_m,
            float(e.max()) + extra_m,
            float(n.max()) + extra_m,
        )


def read_area(path: Path) -> Grid:
    """The pack's grid around the polygon in a GeoJSON file (WGS84)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FetchError(f"området {path} går inte att läsa: {exc}", 2) from exc
    points = [pt for geom in _geometries(data) for pt in _polygon_points(geom)]
    if not points:
        raise FetchError(f"{path}: ingen polygon i filen", 2)
    lons, lats = [p[0] for p in points], [p[1] for p in points]
    if not (
        LAT_RANGE[0] <= min(lats)
        and max(lats) <= LAT_RANGE[1]
        and LON_RANGE[0] <= min(lons)
        and max(lons) <= LON_RANGE[1]
    ):
        raise FetchError(f"{path}: området ligger utanför Sverige (eller lon/lat i fel ordning)", 2)
    origin = LatLon(lat=(min(lats) + max(lats)) / 2, lon=(min(lons) + max(lons)) / 2)
    east_r, north_r = tangent_radii(origin.lat)
    width = math.radians(max(lons) - min(lons)) * east_r
    height = math.radians(max(lats) - min(lats)) * north_r
    if width > MAX_SIDE_M or height > MAX_SIDE_M:
        raise FetchError(
            f"{path}: området är {width / 1000:.1f} x {height / 1000:.1f} km, högst 10 x 10 km", 2
        )
    # Even sizes, so the origin is a cell corner.
    cols = 2 * math.ceil(width / 2 + MARGIN_M)
    rows = 2 * math.ceil(height / 2 + MARGIN_M)
    return Grid(origin=origin, cols=cols, rows=rows)


def _geometries(data: Any) -> list[Any]:
    if not isinstance(data, dict):
        return []
    kind = data.get("type")
    if kind == "FeatureCollection":
        return [g for f in data.get("features") or [] for g in _geometries(f)]
    if kind == "Feature":
        return _geometries(data.get("geometry"))
    return [data]


def _polygon_points(geom: Mapping[str, Any]) -> list[tuple[float, float]]:
    kind = geom.get("type")
    coords: Any = geom.get("coordinates")
    if kind not in ("Polygon", "MultiPolygon"):
        raise FetchError(f"området ska vara en Polygon eller MultiPolygon, inte {kind}", 2)
    try:
        rings: Any = coords if kind == "Polygon" else [ring for poly in coords for ring in poly]
        return [(float(pt[0]), float(pt[1])) for ring in rings for pt in ring]
    except (TypeError, ValueError, IndexError) as exc:
        raise FetchError(f"områdets koordinater går inte att läsa: {exc}", 2) from exc


# --- network ----------------------------------------------------------------------------


class _HttpsRedirectsOnly(urllib.request.HTTPRedirectHandler):
    """Redirects are followed only to https. The login is an unredirected header, so it
    never goes along a redirect at all."""

    def redirect_request(  # type: ignore[override]
        self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> urllib.request.Request | None:
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise FetchError(f"{req.full_url} skickade vidare till en adress utan https")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HttpsRedirectsOnly())


def _urlopen(req: urllib.request.Request) -> Any:
    """The one place that reaches the network (the tests replace it)."""
    return _OPENER.open(req, timeout=TIMEOUT_S)


def _request(
    url: str,
    *,
    body: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    login: Mapping[str, str] | None = None,
) -> urllib.request.Request:
    req = urllib.request.Request(  # noqa: S310 - https hosts checked by the callers
        url, data=body, headers={"User-Agent": USER_AGENT, **(headers or {})}
    )
    if login:
        token = base64.b64encode(f"{login['user']}:{login['password']}".encode()).decode()
        req.add_unredirected_header("Authorization", f"Basic {token}")
    return req


def _fetch(req: urllib.request.Request) -> bytes:
    """The body, with a few tries for a network that drops; a refused login is final."""
    last: Exception | None = None
    for attempt in range(TRIES):
        try:
            with _urlopen(req) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                with_login = req.has_header("Authorization")
                raise FetchError(LOGIN_REFUSED if with_login else str(exc)) from None
            if exc.code < 500 and exc.code != 429:
                raise FetchError(f"kunde inte hämta {req.full_url}: {exc}") from exc
            last = exc
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
            last = exc
        time.sleep(2**attempt)
    raise FetchError(f"kunde inte hämta {req.full_url}: {last}")


def _json(req: urllib.request.Request) -> Any:
    raw = _fetch(req)
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise FetchError(f"{req.full_url} svarade inte med JSON") from exc


# --- the catalogue ------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    """One of Lantmäteriet's files: its tile in SWEREF 99 TM and when it was made."""

    item_id: str
    href: str
    bbox: tuple[float, float, float, float]  # west, south, east, north
    when: str  # the scan's or the model's date, ISO
    size: int | None
    sha256: str | None

    @property
    def file_name(self) -> str:
        return self.href.rsplit("/", 1)[-1]

    @property
    def year(self) -> int:
        return int(self.when[:4])


def stac_search(collection: str, bbox: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    """Every catalogue item of ``collection`` over ``bbox`` (degrees), page by page."""
    url = f"{STAC}/search"
    body: Any = {"bbox": list(bbox), "collections": [collection], "limit": 100}
    items: list[dict[str, Any]] = []
    for _ in range(STAC_PAGES_MAX):
        page = _json(
            _request(
                url, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
            )
        )
        items += [i for i in page.get("features", []) if isinstance(i, dict)]
        nxt = next((ln for ln in page.get("links", []) if ln.get("rel") == "next"), None)
        if nxt is None:
            return items
        url = str(nxt.get("href", ""))
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != STAC_HOST:
            raise FetchError(f"oväntad adress i Lantmäteriets katalog: {url!r}")
        if str(nxt.get("method", "POST")).upper() != "POST" or not isinstance(
            nxt.get("body"), dict
        ):
            raise FetchError("Lantmäteriets katalog bläddrar på ett oväntat sätt")
        body = nxt["body"]
    raise FetchError(f"Lantmäteriets katalog gav fler än {STAC_PAGES_MAX} sidor")


def tile_of(item: Mapping[str, Any]) -> Tile:
    href = fetch_site._lm_href(item)  # https on Lantmäteriet's download host only
    data = item.get("assets", {}).get("data", {})
    props = item.get("properties", {})
    bbox = data.get("proj:bbox") or props.get("proj:bbox")
    code = data.get("proj:code") or props.get("proj:code")
    if code not in ("EPSG:5845", "EPSG:3006") or not isinstance(bbox, list) or len(bbox) != 4:
        raise FetchError(f"{item.get('id')}: oväntat koordinatsystem eller utbredning i katalogen")
    when = str(props.get("datetime") or props.get("start_datetime") or "")
    if not re.match(r"^\d{4}-\d\d-\d\d", when):
        raise FetchError(f"{item.get('id')}: katalogen saknar datum")
    checksum = str(data.get("file:checksum", ""))
    # A multihash: 0x12 (SHA-256), 0x20 (32 bytes), then the digest.
    sha = checksum[4:] if checksum.startswith("1220") and len(checksum) == 68 else None
    size = data.get("file:size")
    return Tile(
        item_id=str(item.get("id", "")),
        href=href,
        bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
        when=when,
        size=int(size) if isinstance(size, int) else None,
        sha256=sha,
    )


def choose_tiles(items: Sequence[Mapping[str, Any]], within: Sequence[float]) -> list[Tile]:
    """The newest file of each tile that overlaps ``within`` (SWEREF west, south, east,
    north), in a fixed order."""
    newest: dict[tuple[float, ...], Tile] = {}
    for item in items:
        tile = tile_of(item)
        w, s, e, n = tile.bbox
        if e <= within[0] or w >= within[2] or n <= within[1] or s >= within[3]:
            continue
        old = newest.get(tile.bbox)
        if old is None or (tile.when, tile.item_id) > (old.when, old.item_id):
            newest[tile.bbox] = tile
    return [newest[k] for k in sorted(newest)]


# --- downloads ------------------------------------------------------------------------------


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(tile: Tile, cache: Path, login: Mapping[str, str]) -> Path:
    """The tile's file in ``cache``: downloaded with the login unless an intact copy is
    there already."""
    dest = cache / tile.file_name
    if dest.is_file() and (tile.sha256 is None or _sha256_of(dest) == tile.sha256):
        print(f"    {tile.file_name} (finns redan)")
        return dest
    cache.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    size = f"{tile.size / 1e6:.0f} MB" if tile.size else "okänd storlek"
    print(f"    {tile.file_name} ({size})")
    req = _request(tile.href, login=login)
    last: Exception | None = None
    digest, got = hashlib.sha256(), 0
    for attempt in range(TRIES):
        try:
            digest, got = hashlib.sha256(), 0
            with _urlopen(req) as resp, part.open("wb") as out:
                for block in iter(lambda: resp.read(CHUNK), b""):
                    out.write(block)
                    digest.update(block)
                    got += len(block)
            break
        except urllib.error.HTTPError as exc:
            part.unlink(missing_ok=True)
            if exc.code in (401, 403):
                raise FetchError(LOGIN_REFUSED) from None
            if exc.code < 500 and exc.code != 429:
                raise FetchError(f"kunde inte hämta {tile.file_name}: {exc}") from None
            last = exc
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
            part.unlink(missing_ok=True)
            last = exc
        time.sleep(2**attempt)
    else:
        raise FetchError(f"kunde inte hämta {tile.file_name}: {last}")
    if (tile.size is not None and got != tile.size) or (
        tile.sha256 is not None and digest.hexdigest() != tile.sha256
    ):
        part.unlink(missing_ok=True)
        raise FetchError(f"{tile.file_name}: filen stämmer inte med katalogens storlek och SHA-256")
    part.replace(dest)
    return dest


# --- reading the files (laspy and tifffile only while the tool runs) ------------------------


def read_points(path: Path) -> Iterator[tuple[Floats, Floats, Floats, NDArray[Any]]]:
    """(easting, northing, height, class) of a point cloud's points, in chunks."""
    try:
        import laspy  # type: ignore[import-not-found]  # BSD-2-Clause; LAZ via lazrs (MIT)
    except ImportError as exc:
        raise FetchError("laspy saknas: kör verktyget med just floor-fetch", 2) from exc
    with laspy.open(path) as reader:
        for chunk in reader.chunk_iterator(POINTS_PER_CHUNK):
            yield (
                np.asarray(chunk.x, dtype=np.float64),
                np.asarray(chunk.y, dtype=np.float64),
                np.asarray(chunk.z, dtype=np.float64),
                np.asarray(chunk.classification),
            )


@dataclass(frozen=True)
class Raster:
    """A north-up grid in SWEREF 99 TM: ``values[0, 0]`` is the north-west cell."""

    values: NDArray[np.float32]
    west: float
    north: float
    cell: float
    nodata: float | None


def read_raster(path: Path) -> Raster:
    """The ground model tile (a GeoTIFF/COG with one band)."""
    try:
        import tifffile  # type: ignore[import-not-found]  # BSD-3-Clause
    except ImportError as exc:
        raise FetchError("tifffile saknas: kör verktyget med just floor-fetch", 2) from exc
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        scale = page.tags[33550].value  # ModelPixelScale
        tie = page.tags[33922].value  # ModelTiepoint
        nodata_tag = page.tags.get(42113)  # GDAL_NODATA
        values = np.asarray(page.asarray(), dtype=np.float32)
    if values.ndim != 2 or abs(scale[0] - scale[1]) > 1e-9 or tie[0] != 0 or tie[1] != 0:
        raise FetchError(f"{path.name}: markhöjdmodellen har ett oväntat rutnät")
    nodata = float(str(nodata_tag.value).strip("\x00 ")) if nodata_tag is not None else None
    return Raster(values, float(tie[3]), float(tie[4]), float(scale[0]), nodata)


# --- the grids ------------------------------------------------------------------------------


def sample_ground(grid: Grid, rasters: Sequence[Raster]) -> NDArray[np.int32]:
    """The ground height (cm) at every cell's centre, nearest ground model cell;
    ``DTM_NODATA_CM`` where no raster has a value."""
    dtm = np.full((grid.rows, grid.cols), DTM_NODATA_CM, dtype=np.int32)
    for row0 in range(0, grid.rows, ROWS_PER_BLOCK):
        row1 = min(row0 + ROWS_PER_BLOCK, grid.rows)
        north, east = to_sweref(*grid.centres(row0, row1))
        block = dtm[row0:row1]
        for r in rasters:
            row = np.floor((r.north - north) / r.cell).astype(np.int64)
            col = np.floor((east - r.west) / r.cell).astype(np.int64)
            h, w = r.values.shape
            inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
            value = np.full(row.shape, np.nan, dtype=np.float64)
            value[inside] = r.values[row[inside], col[inside]]
            if r.nodata is not None:
                value[value == r.nodata] = np.nan
            cm = np.round(value * 100.0)
            ok = (
                np.isfinite(cm) & (cm >= DTM_MIN_CM) & (cm <= DTM_MAX_CM) & (block == DTM_NODATA_CM)
            )
            block[ok] = cm[ok].astype(np.int32)
    return dtm


class Canopy:
    """The highest laser point and the oldest scan year in every cell, filled tile by tile."""

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        # float32 (0.1 mm at 1000 m): a 10 x 10 km pack is 100 million cells.
        self.top = np.full((grid.rows, grid.cols), -np.inf, dtype=np.float32)
        self.year = np.zeros((grid.rows, grid.cols), dtype=np.uint16)
        self.points = 0

    def add(self, east: Floats, north: Floats, z: Floats, cls: NDArray[Any], year: int) -> None:
        keep = ~np.isin(cls, NOISE_CLASSES) & np.isfinite(z)
        lat, lon = from_sweref(north[keep], east[keep])
        row, col, inside = self.grid.cell_of(*self.grid.plane(lat, lon))
        flat = row[inside] * self.grid.cols + col[inside]
        np.maximum.at(self.top.reshape(-1), flat, z[keep][inside].astype(np.float32))
        years = self.year.reshape(-1)
        cur = years[flat]
        years[flat] = np.where((cur == 0) | (cur > year), year, cur)
        self.points += int(flat.size)


def tree_heights(top: NDArray[np.floating[Any]], dtm_cm: NDArray[np.int32]) -> NDArray[np.int16]:
    """Tree height (dm, rounded up) of the highest point above the ground; -1 where a
    cell has no point or no ground."""
    known = np.isfinite(top) & (dtm_cm != DTM_NODATA_CM)
    height_dm = np.where(known, top.astype(np.float64) * 10.0 - dtm_cm / 10.0, 0.0)
    # Rounded up, after the float noise of the subtraction is rounded away.
    dm = np.ceil(np.round(np.clip(height_dm, 0.0, np.iinfo(np.int16).max), 3))
    return np.where(known, dm, CHM_UNKNOWN_DM).astype(np.int16)


def scan_squares(year: NDArray[np.uint16], cell_m: int) -> NDArray[np.uint16]:
    """The oldest scan year of each ``cell_m`` square from the north-west corner, among
    its cells with points; 0 (unknown) where none has."""
    rows, cols = year.shape
    tr, tc = -(-rows // cell_m), -(-cols // cell_m)
    big = np.iinfo(np.uint16).max
    padded = np.full((tr * cell_m, tc * cell_m), big, dtype=np.uint16)
    padded[:rows, :cols] = np.where(year == 0, big, year)
    oldest = padded.reshape(tr, cell_m, tc, cell_m).min(axis=(1, 3))
    return np.where(oldest == big, 0, oldest).astype(np.uint16)


# --- power lines ------------------------------------------------------------------------------


def overpass_query(bbox: tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox
    box = f"({south:.6f},{west:.6f},{north:.6f},{east:.6f})"
    return (
        "[out:json][timeout:90];"
        f'(way["power"~"^(line|minor_line)$"]{box};node["power"~"^(tower|pole)$"]{box};);'
        "out geom;"
    )


def parse_overpass(raw: bytes) -> list[PackLine]:
    """Power lines (ways) and towers and poles (nodes) from Overpass's JSON answer."""
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise FetchError("Overpass svarade inte med JSON") from exc
    if not isinstance(data, dict):
        raise FetchError("Overpass svarade på ett oväntat sätt")
    remark = str(data.get("remark", ""))
    if "error" in remark.lower() or "timed out" in remark.lower():
        raise FetchError(f"Overpass kunde inte svara: {remark[:200]}")
    lines: list[PackLine] = []
    for el in data.get("elements", []):
        tags = el.get("tags") or {}
        props = {"source": "OpenStreetMap", "osm": f"{el.get('type')}/{el.get('id')}"}
        props |= {k: str(tags[k]) for k in ("power", "voltage", "name") if k in tags}
        if el.get("type") == "way" and el.get("geometry"):
            pts = tuple(LatLon(lat=g["lat"], lon=g["lon"]) for g in el["geometry"] if g)
        elif el.get("type") == "node" and "lat" in el:
            pts = (LatLon(lat=el["lat"], lon=el["lon"]),)
        else:
            continue
        if pts:
            lines.append(PackLine(pts, props))
    return lines


def fetch_lines(grid: Grid) -> list[PackLine]:
    query = overpass_query(grid.lonlat_bbox(LINE_MARGIN_M))
    body = urllib.parse.urlencode({"data": query}).encode()
    return parse_overpass(_fetch(_request(OVERPASS, body=body)))


# --- settings -------------------------------------------------------------------------------


def parse_args(argv: Sequence[str], keys: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for arg in argv:
        key, sep, value = arg.partition("=")
        if not sep or key not in keys:
            raise FetchError(f"okänd inställning {arg!r} (tillåtna: {', '.join(keys)})", 2)
        out[key] = value
    if not re.match(FLOOR_PACK_NAME_PATTERN, out.get("name", "")):
        raise FetchError("ange name=<namn>: a-z, 0-9, _ och -, högst 64 tecken", 2)
    for flag in ("keep_cache", "lines"):
        if out.get(flag, "1" if flag == "lines" else "0") not in ("0", "1"):
            raise FetchError(f"{flag} ska vara 0 eller 1", 2)
    return out


def floor_dir(args: Mapping[str, str], env: Mapping[str, str], root: Path = ROOT) -> Path:
    """``out=``, else ``${SD_DATA_DIR}/floor`` (environment or ``.env``)."""
    if args.get("out"):
        return Path(args["out"]).expanduser().resolve()
    data = env.get("SD_DATA_DIR", "")
    dotenv = root / ".env"
    if not data and dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if line.startswith("SD_DATA_DIR="):
                data = line.partition("=")[2].strip()
    if not data:
        raise FetchError("Datamappen är okänd. Kör just setup-config, eller ange out=<mapp>.", 2)
    return Path(data).expanduser().resolve() / "floor"


def get_login(root: Path = ROOT) -> dict[str, str]:
    """The Geotorget login (environment, the local file, or asked for in a terminal)."""
    login = fetch_site.lm_login(root)
    if login:
        return login
    if not sys.stdin.isatty():
        raise FetchError(fetch_site.LOGIN_MISSING, 2)
    user = input("Geotorget-användare: ").strip()
    password = getpass.getpass("Geotorget-lösenord: ")
    if not user or not password:
        raise FetchError(fetch_site.LOGIN_MISSING, 2)
    return {"user": user, "password": password}


# --- main -------------------------------------------------------------------------------------


def fetch(argv: Sequence[str], env: Mapping[str, str], root: Path = ROOT) -> int:
    args = parse_args(argv, FETCH_KEYS)
    if "area" not in args:
        raise FetchError("ange area=<fil.geojson> med området", 2)
    grid = read_area(Path(args["area"]).expanduser())
    base = floor_dir(args, env, root)
    keep_cache, with_lines = args.get("keep_cache") == "1", args.get("lines", "1") == "1"
    login = get_login(root)
    cache = base / ".cache"
    print(f"==> område {grid.cols} x {grid.rows} m, kartpaket {base / args['name']}")

    print("==> Lantmäteriets katalog")
    bbox, within = grid.lonlat_bbox(50.0), grid.sweref_bbox()
    copc = choose_tiles(stac_search(COPC_COLLECTION, bbox), within)
    dtm_tiles = choose_tiles(stac_search(DTM_COLLECTION, bbox), within)
    if not copc or not dtm_tiles:
        raise FetchError("Lantmäteriet har ingen laserdata skog eller markhöjdmodell för området")

    lines: list[PackLine] = []
    if with_lines:
        print("==> kraftledningar (OpenStreetMap, Overpass)")
        lines = fetch_lines(grid)
        print(f"    {len(lines)} ledningar, stolpar och torn")
    else:
        print(NO_LINES_WARNING, file=sys.stderr)

    print("==> markhöjdmodell")
    rasters = [read_raster(download(t, cache, login)) for t in dtm_tiles]
    dtm = sample_ground(grid, rasters)
    del rasters

    print("==> laserdata skog (punktmoln)")
    canopy = Canopy(grid)
    for tile in copc:
        path = download(tile, cache, login)
        try:
            for east, north, z, cls in read_points(path):
                canopy.add(east, north, z, cls, tile.year)
        finally:
            if not keep_cache:
                path.unlink(missing_ok=True)
    if keep_cache:
        print(KEEP_CACHE_WARNING.format(cache=cache), file=sys.stderr)

    chm = tree_heights(canopy.top, dtm)
    years = scan_squares(canopy.year, SCAN_CELL_M)
    today = dt.datetime.now(dt.UTC).date().isoformat()
    try:
        pack = write_floor_pack(
            base / args["name"],
            name=args["name"],
            origin=grid.origin,
            dtm_cm=dtm,
            chm_dm=chm,
            scan_year=years,
            scan_cell_m=SCAN_CELL_M,
            attribution=ATTRIBUTION + (OSM_ATTRIBUTION if with_lines else ""),
            lines=lines,
            about={
                "script": "tools/terrain/fetch_floor_map.py",
                "script_version": SCRIPT_VERSION,
                "retrieved": today,
                "margin_m": MARGIN_M,
                "pointclouds": [f"{t.item_id} ({t.when[:10]})" for t in copc],
                "ground_models": [f"{t.item_id} ({t.when[:10]})" for t in dtm_tiles],
                "lines_source": "OpenStreetMap (Overpass)" if with_lines else "none",
                "noise_classes_left_out": [str(c) for c in NOISE_CLASSES],
            },
        )
    except FloorPackError as exc:
        raise FetchError(f"kartpaketet blev inte rätt: {exc}") from exc
    print(summary(pack, base / args["name"]))
    return 0


def summary(pack: FloorPack, folder: Path) -> str:
    unknown = float((pack.chm_dm < 0).mean())
    known_years = pack.scan_year[pack.scan_year > 0]
    oldest = int(known_years.min()) if known_years.size else 0
    return (
        f"Klart: {folder}\n"
        f"  {pack.dtm_cm.shape[1]} x {pack.dtm_cm.shape[0]} m, okänd trädhöjd i "
        f"{unknown:.1%} av rutorna, äldsta skanning {oldest or 'okänd'}, "
        f"{len(pack.lines)} ledningar\n"
        f"  floor_pack_id = {pack.name}\n  floor_pack_sha256 = {pack.sha256}"
    )


_HOST = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.@-]{0,127}$")
_REMOTE_DIR = re.compile(r"^[A-Za-z0-9_./~][A-Za-z0-9_./~-]{0,255}$")


def push(argv: Sequence[str], env: Mapping[str, str], root: Path = ROOT) -> int:
    """Copy a checked pack to ``<dir>/floor/<name>/`` on the onboard computer."""
    args = parse_args(argv, PUSH_KEYS)
    host, remote = args.get("host", ""), args.get("dir", "")
    if not _HOST.match(host) or not _REMOTE_DIR.match(remote):
        raise FetchError("ange host=<användare@dator> och dir=<state_dir på drönaren>", 2)
    folder = floor_dir(args, env, root) / args["name"]
    try:
        pack = load_floor_pack(folder)
    except FloorPackError as exc:
        raise FetchError(f"kartpaketet går inte att läsa, skickas inte: {exc}") from exc
    target = f"{remote.rstrip('/')}/floor"
    for argv_ in (
        ["ssh", host, "mkdir", "-p", target],
        ["scp", "-q", "-r", str(folder), f"{host}:{target}/"],
    ):
        res = subprocess.run(argv_, check=False)  # noqa: S603 - fixed programs, checked arguments
        if res.returncode != 0:
            raise FetchError(f"{argv_[0]} misslyckades (kod {res.returncode})")
    print(f"Klart: {pack.name} finns nu i {host}:{target}/{pack.name}")
    print(f"  floor_pack_sha256 = {pack.sha256}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args[:1] == ["push"]:
            return push(args[1:], os.environ)
        return fetch(args, os.environ)
    except FetchError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
