"""Fetch height data and map layers for a simulator site: ``just world-fetch site=<name>``.

A *public* site has its definition in ``simulation/gazebo/sites/<name>/site.toml``. This
tool writes its data next to that file, ready to commit after review:

- ``dem.asc``: heights in metres above sea level on a square grid in the site's local
  east/north plane (azimuthal equidistant around home), from Copernicus DEM GLO-30
  (no account needed), or from Lantmäteriet's Markhöjdmodell 1 m (bare ground, CC BY 4.0,
  needs a Geotorget login) when the site sets ``[dem] source = "lantmateriet-mhm"`` and
  ``surface = "dtm"``. Cropped to the site and resampled to ``[dem] cell_m``.
- ``osm.geojson``: forest, water, open land, roads, buildings and parking from
  OpenStreetMap (ODbL), clipped to the site, in WGS84.
- ``data.toml``: where and when the data came from.

A *local* site is defined in the gitignored ``config/site.local.toml`` (see
``config/site.example.toml``). It uses Lantmäteriet's Markhöjdmodell (1 m) and orthophoto
through their STAC API, with the owner's own Geotorget login, and writes everything to
``build/sites/<name>/``, which is never committed or uploaded. The login is read from
``SD_LM_USER`` / ``SD_LM_PASSWORD`` in the environment or, when they are unset, from the
gitignored ``config/geotorget.local.toml`` (``lm_login``; see
``config/geotorget.example.toml``), and handed to GDAL in its environment, never on a
command line. GDAL's error text is shown with the login cut out.

OpenStreetMap is read through its main API (``/api/0.6/map`` for the site's box, plus the
site's multipolygons whole): a handful of requests for a rare, manual refresh of one
small site, well within the API's usage policy.

Needs internet and GDAL's command-line tools (``gdalwarp``, ``gdal_translate``;
``sudo apt-get install gdal-bin``) only while it runs. The simulator itself reads the
written files offline. Exit codes: 0 done, 1 a download or a check failed, 2 bad
arguments.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon, box, mapping

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_SITES = ROOT / "simulation" / "gazebo" / "sites"
LOCAL_SITES = ROOT / "config" / "site.local.toml"
KEYS = ("site",)
SITE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
TIMEOUT_S = 120
USER_AGENT = "searchdrone-world-fetch (+https://github.com/YXHFR/Searchdrone)"

COPERNICUS = "https://copernicus-dem-30m.s3.amazonaws.com"
OSM_API = "https://api.openstreetmap.org/api/0.6"
LM_STAC_DEM = "https://api.lantmateriet.se/stac-hojd/v1"
LM_STAC_ORTHO = "https://api.lantmateriet.se/stac-bild/v1"
LM_DEM_COLLECTION = "dtm-cog"
LM_HOST = "dl1.lantmateriet.se"
ORTHO_CELL_M = 0.16
# Colour orthophotos only: "rgbi" is red, green, blue and near infrared (the first three
# bands are kept); "cir" (false colour) and "pan" (grey) are not.
ORTHO_SPECTRA = ("rgb", "rgbi")
LOGIN_FILE = Path("config") / "geotorget.local.toml"
# Map data is fetched this far outside the site so that ways crossing its edge are whole.
MARGIN_M = 150.0
COORD_DIGITS = 7  # about 1 cm

# OSM tags -> the surface kinds the world generator knows.
AREA_KINDS: dict[tuple[str, str], str] = {
    ("natural", "water"): "water",
    ("water", "lake"): "water",
    ("water", "pond"): "water",
    ("landuse", "reservoir"): "water",
    ("natural", "wood"): "forest",
    ("landuse", "forest"): "forest",
    ("natural", "scrub"): "scrub",
    ("natural", "heath"): "scrub",
    ("natural", "wetland"): "wetland",
    ("natural", "bare_rock"): "rock",
    ("natural", "beach"): "sand",
    ("natural", "sand"): "sand",
    ("natural", "grassland"): "meadow",
    ("landuse", "meadow"): "meadow",
    ("landuse", "grass"): "grass",
    ("landuse", "farmland"): "farmland",
    ("landuse", "residential"): "residential",
    ("landuse", "commercial"): "residential",
    ("leisure", "park"): "grass",
    ("leisure", "pitch"): "pitch",
    ("amenity", "parking"): "parking",
}
# Highway classes kept as roads, with a width when the way has none (metres).
ROAD_WIDTH_M: dict[str, float] = {
    "motorway": 12.0,
    "trunk": 9.0,
    "primary": 8.0,
    "secondary": 7.0,
    "tertiary": 6.5,
    "unclassified": 5.0,
    "residential": 5.5,
    "service": 4.0,
    "living_street": 4.5,
    "track": 3.0,
    "cycleway": 2.5,
    "footway": 1.8,
    "path": 1.2,
    "bridleway": 1.5,
    "pedestrian": 4.0,
    "steps": 1.5,
}


class FetchError(Exception):
    """A Swedish message for the owner and the exit code."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SiteDef:
    name: str
    lat: float
    lon: float
    heading_deg: float
    half_size_m: float
    dem_source: str  # "copernicus-glo30" | "lantmateriet-mhm"
    cell_m: float
    ortho: bool
    out: Path
    public: bool

    @property
    def proj(self) -> str:
        """The site's local plane: azimuthal equidistant around home, metres east/north."""
        return f"+proj=aeqd +lat_0={self.lat} +lon_0={self.lon} +datum=WGS84 +units=m +no_defs"


# --- settings -----------------------------------------------------------------------


def parse_args(argv: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for arg in argv:
        key, sep, value = arg.partition("=")
        if not sep or key not in KEYS:
            raise FetchError(f"okänd inställning {arg!r} (använd site=<namn>)", 2)
        out[key] = value
    if "site" not in out:
        raise FetchError("ange site=<namn>", 2)
    if not SITE_NAME.match(out["site"]):
        raise FetchError("site: a-z, 0-9 och _, högst 32 tecken, först en bokstav", 2)
    return out


def _num(table: Mapping[str, Any], key: str, where: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FetchError(f"{where}: {key} ska vara ett tal", 2)
    return float(value)


def load_site(name: str, root: Path = ROOT) -> tuple[SiteDef, dict[str, str]]:
    """The site's definition, and the Lantmäteriet login for a local site."""
    public = root / "simulation" / "gazebo" / "sites" / name / "site.toml"
    if public.is_file():
        data = tomllib.loads(public.read_text(encoding="utf-8"))
        where = str(public.relative_to(root))
        out, is_public = public.parent, True
    else:
        local = root / "config" / "site.local.toml"
        if not local.is_file():
            raise FetchError(f"platsen {name} finns inte (varken offentlig eller i {local})", 2)
        sites = tomllib.loads(local.read_text(encoding="utf-8")).get("sites", {})
        if name not in sites:
            raise FetchError(f"platsen {name} saknas i config/site.local.toml", 2)
        data, where = sites[name], f"config/site.local.toml [sites.{name}]"
        out, is_public = root / "build" / "sites" / name, False
        if "lm_user" in data or "lm_password" in data:
            raise FetchError(
                "inloggningen hör inte hemma i filen: använd SD_LM_USER och SD_LM_PASSWORD", 2
            )
    home = data.get("home", {})
    dem = data.get("dem", {})
    area = data.get("area", {})
    source = str(dem.get("source", "copernicus-glo30" if is_public else "lantmateriet-mhm"))
    if source not in ("copernicus-glo30", "lantmateriet-mhm"):
        raise FetchError(f"{where}: dem.source ska vara copernicus-glo30 eller lantmateriet-mhm", 2)
    # A public site may use Lantmäteriet's Markhöjdmodell (CC BY 4.0, with the owner's
    # login) instead of Copernicus; it is bare ground, so the site must say so.
    if is_public and source == "lantmateriet-mhm" and dem.get("surface") != "dtm":
        raise FetchError(f'{where}: lantmateriet-mhm är markhöjd: sätt dem.surface = "dtm"', 2)
    ortho = bool(data.get("ortho", False))
    if is_public and ortho:
        raise FetchError(f"{where}: flygfoton hämtas bara för lokala platser", 2)
    site = SiteDef(
        name=name,
        lat=_num(home, "lat", where),
        lon=_num(home, "lon", where),
        heading_deg=float(home.get("heading_deg", 0.0)),
        half_size_m=_num(area, "half_size_m", where),
        dem_source=source,
        cell_m=_num(dem, "cell_m", where),
        ortho=ortho,
        out=out,
        public=is_public,
    )
    if not 100.0 <= site.half_size_m <= 2500.0:
        raise FetchError(f"{where}: area.half_size_m ska vara 100-2500", 2)
    if not 1.0 <= site.cell_m <= 50.0:
        raise FetchError(f"{where}: dem.cell_m ska vara 1-50", 2)
    login: dict[str, str] = {}
    if source == "lantmateriet-mhm" or ortho:
        login = lm_login(root)
    if source == "lantmateriet-mhm" and not login:
        raise FetchError(LOGIN_MISSING, 2)
    return site, login


LOGIN_MISSING = (
    "Lantmäteriets inloggning saknas. Kopiera config/geotorget.example.toml till "
    "config/geotorget.local.toml och skriv in ditt Geotorget-konto där, eller sätt "
    "SD_LM_USER och SD_LM_PASSWORD."
)


def lm_login(root: Path = ROOT) -> dict[str, str]:
    """The owner's Geotorget login: ``SD_LM_USER``/``SD_LM_PASSWORD``, else the gitignored
    ``config/geotorget.local.toml``; empty when neither has one. The file must be readable
    by its owner only."""
    user, password = os.environ.get("SD_LM_USER", ""), os.environ.get("SD_LM_PASSWORD", "")
    if user and password:
        return {"user": user, "password": password}
    path = root / LOGIN_FILE
    if not path.is_file():
        return {}
    if os.name == "posix" and path.stat().st_mode & 0o077:
        raise FetchError(f"{LOGIN_FILE} kan läsas av andra. Kör: chmod 600 {LOGIN_FILE}", 2)
    try:
        table = tomllib.loads(path.read_text(encoding="utf-8")).get("geotorget", {})
    except tomllib.TOMLDecodeError as exc:
        raise FetchError(f"{LOGIN_FILE} går inte att läsa: {exc}", 2) from exc
    user, password = str(table.get("user", "")).strip(), str(table.get("password", ""))
    if not user or not password:
        raise FetchError(f"{LOGIN_FILE}: fyll i user och password under [geotorget]", 2)
    return {"user": user, "password": password}


# --- geometry -----------------------------------------------------------------------


def lonlat_bbox(site: SiteDef, extra_m: float) -> tuple[float, float, float, float]:
    """A WGS84 box (min_lon, min_lat, max_lon, max_lat) that covers the site plus extra_m."""
    half = site.half_size_m + extra_m
    dlat = math.degrees(half / 6_356_752.0) * 1.01
    dlon = math.degrees(half / (6_378_137.0 * math.cos(math.radians(site.lat)))) * 1.01
    return (site.lon - dlon, site.lat - dlat, site.lon + dlon, site.lat + dlat)


def copernicus_tiles(bbox: tuple[float, float, float, float]) -> list[str]:
    min_lon, min_lat, max_lon, max_lat = bbox
    urls = []
    for lat in range(math.floor(min_lat), math.floor(max_lat) + 1):
        for lon in range(math.floor(min_lon), math.floor(max_lon) + 1):
            ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00"
            ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00"
            name = f"Copernicus_DSM_COG_10_{ns}_{ew}_DEM"
            urls.append(f"{COPERNICUS}/{name}/{name}.tif")
    return urls


# --- network ------------------------------------------------------------------------


def _get(url: str, *, data: bytes | None = None, headers: Mapping[str, str] = {}) -> bytes:
    request = urllib.request.Request(  # noqa: S310 - fixed https hosts above
        url, data=data, headers={"User-Agent": USER_AGENT, **headers}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as resp:  # noqa: S310
            return resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise FetchError(f"kunde inte hämta {url}: {exc}") from exc


def _gdal(argv: Sequence[str], env: Mapping[str, str] | None = None) -> None:
    if shutil.which(argv[0]) is None:
        raise FetchError(f"{argv[0]} saknas. Installera GDAL: sudo apt-get install gdal-bin", 2)
    # A minimal environment: the program path, a home for GDAL's cache and the proxy and
    # certificate settings a network may need; nothing else from the caller's environment.
    keep = (
        "PATH",
        "HOME",
        "LANG",
        "TMPDIR",
        "HTTPS_PROXY",
        "https_proxy",
        "NO_PROXY",
        "no_proxy",
        "SSL_CERT_FILE",
        "CURL_CA_BUNDLE",
        "GDAL_DATA",
        "PROJ_DATA",
    )
    base = {k: os.environ[k] for k in keep if k in os.environ}
    full_env = {**base, "GDAL_HTTP_USERAGENT": USER_AGENT, **(env or {})}
    res = subprocess.run(argv, env=full_env, capture_output=True, text=True, check=False)  # noqa: S603
    if res.returncode != 0:
        raise FetchError(f"{argv[0]} misslyckades: {redact(res.stderr, env)[-500:]}")


def redact(text: str, env: Mapping[str, str] | None) -> str:
    """``text`` with the login from GDAL's environment (and its parts) cut out."""
    out = text.strip()
    userpwd = (env or {}).get("GDAL_HTTP_USERPWD", "")
    parts = {userpwd, *userpwd.split(":", 1)} - {""}
    forms = set(parts)
    for secret in parts:
        forms.add(urllib.parse.quote(secret, safe=""))
    if userpwd:
        forms.add(base64.b64encode(userpwd.encode()).decode())  # Basic auth
    for secret in sorted(forms, key=len, reverse=True):
        out = out.replace(secret, "***")
    return out


def stac_items(
    api: str, bbox: tuple[float, float, float, float], collections: Sequence[str]
) -> list[dict[str, Any]]:
    body: dict[str, Any] = {"bbox": list(bbox), "limit": 100}
    if collections:
        body["collections"] = list(collections)
    raw = _get(
        f"{api}/search",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    items = json.loads(raw).get("features", [])
    return [i for i in items if isinstance(i, dict)]


def _lm_href(item: Mapping[str, Any]) -> str:
    href = str(item.get("assets", {}).get("data", {}).get("href", ""))
    if urllib.parse.urlparse(href).hostname != LM_HOST or not href.startswith("https://"):
        raise FetchError(f"oväntad adress i Lantmäteriets katalog: {href!r}")
    return href


def lm_env(login: Mapping[str, str]) -> dict[str, str]:
    return {"GDAL_HTTP_AUTH": "BASIC", "GDAL_HTTP_USERPWD": f"{login['user']}:{login['password']}"}


# --- height data --------------------------------------------------------------------


def fetch_dem(site: SiteDef, login: Mapping[str, str], work: Path) -> tuple[Path, list[str]]:
    """Warp the source onto the site grid; returns the ASCII grid and the source names."""
    half = site.half_size_m + MARGIN_M
    # Grid nodes on multiples of cell_m, so the grid is centred on home.
    half = math.ceil(half / site.cell_m) * site.cell_m
    bbox = lonlat_bbox(site, MARGIN_M + 2 * site.cell_m)
    if site.dem_source == "copernicus-glo30":
        urls, env = copernicus_tiles(bbox), {}
    else:
        items = stac_items(LM_STAC_DEM, bbox, [LM_DEM_COLLECTION])
        if not items:
            raise FetchError("Lantmäteriet har ingen höjdmodell för platsen")
        urls, env = sorted(_lm_href(i) for i in items), lm_env(login)
    out = work / "dem.asc"
    # -te names cell edges; a node grid centred on home needs half a cell more each way.
    edge = half + site.cell_m / 2
    _gdal(
        [
            "gdalwarp",
            "-q",
            "-overwrite",
            "-t_srs",
            site.proj,
            "-te",
            str(-edge),
            str(-edge),
            str(edge),
            str(edge),
            "-tr",
            str(site.cell_m),
            str(site.cell_m),
            "-r",
            "bilinear",
            "-dstnodata",
            "-9999",
            *[f"/vsicurl/{u}" for u in urls],
            str(work / "dem.tif"),
        ],
        env,
    )
    _gdal(
        [
            "gdal_translate",
            "-q",
            "-of",
            "AAIGrid",
            "-co",
            "DECIMAL_PRECISION=2",
            str(work / "dem.tif"),
            str(out),
        ]
    )
    return out, [u.rsplit("/", 1)[-1] for u in urls]


def write_dem(src: Path, dest: Path, site: SiteDef) -> float:
    """Copy the grid with a short header we control; returns the height at home."""
    lines = src.read_text(encoding="ascii").splitlines()
    header = {}
    for line in lines[:6]:
        key, value = line.split()
        header[key.lower()] = float(value)
    grid = np.loadtxt(lines[6:], dtype=np.float64)
    if (grid <= -9999).any():
        raise FetchError("höjdmodellen har hål inom platsen")
    n = grid.shape[0]
    if grid.shape != (n, n) or n % 2 == 0:
        raise FetchError(f"oväntad rutnätsstorlek {grid.shape}")
    centre = float(grid[n // 2, n // 2])
    rows = [" ".join(f"{v:.2f}" for v in row) for row in grid]
    dest.write_text(
        "\n".join(
            [
                f"# Heights in metres above sea level, {n} x {n} nodes every {site.cell_m:g} m,",
                "# centred on home; the first row is the northernmost.",
                f"cell_m {site.cell_m:g}",
                f"nodes {n}",
                *rows,
            ]
        )
        + "\n",
        encoding="ascii",
    )
    return centre


# --- orthophoto (local sites only) ------------------------------------------------------


ORTHO_IN_CI = "Ortofoton hämtas aldrig i CI: de får bara finnas på din egen dator."


def in_ci() -> bool:
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def choose_ortho(items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The newest colour orthophoto over the site: every tile of the newest flight year,
    at its finest resolution. Tiles of one year are flown at different times of day, so
    the year (``flygar``) picks them, not the timestamp."""

    def key(item: Mapping[str, Any]) -> tuple[str, int, float]:
        props = item.get("properties", {})
        year = str(props.get("flygar") or str(props.get("datetime", ""))[:4])
        try:
            return str(item.get("collection", "")), int(year or 0), float(props["upplosning"])
        except ValueError as exc:
            raise FetchError(f"Lantmäteriet gav ett oväntat flygår: {year!r}") from exc

    usable = [
        i
        for i in items
        if str(i.get("collection", "")).startswith("orto-")
        and i.get("properties", {}).get("spektraltyp") in ORTHO_SPECTRA
        and isinstance(i.get("properties", {}).get("upplosning"), int | float)
    ]
    if not usable:
        raise FetchError("Lantmäteriet har inget färgortofoto för platsen")
    year = max(key(i)[1] for i in usable)
    cell = min(key(i)[2] for i in usable if key(i)[1] == year)
    collection = min(key(i)[0] for i in usable if key(i)[1:] == (year, cell))
    return [i for i in usable if key(i) == (collection, year, cell)]


def fetch_ortho(site: SiteDef, login: Mapping[str, str], work: Path, dest: Path) -> dict[str, Any]:
    """The orthophoto on the site's plane, ``ORTHO_CELL_M`` per pixel, row 0 north, as an
    (n, n, 3) uint8 array in ``dest``; returns where it came from."""
    if in_ci():
        raise FetchError(ORTHO_IN_CI, 2)
    chosen = choose_ortho(stac_items(LM_STAC_ORTHO, lonlat_bbox(site, 0.0), []))
    urls = sorted(_lm_href(i) for i in chosen)
    edge = site.half_size_m
    raw = work / "ortho.raw"
    _gdal(
        [
            "gdalwarp",
            "-q",
            "-overwrite",
            "-t_srs",
            site.proj,
            "-te",
            str(-edge),
            str(-edge),
            str(edge),
            str(edge),
            "-tr",
            str(ORTHO_CELL_M),
            str(ORTHO_CELL_M),
            "-r",
            "bilinear",
            "-ot",
            "Byte",
            "-of",
            "ENVI",
            "-co",
            "INTERLEAVE=BIP",
            *[f"/vsicurl/{u}" for u in urls],
            str(raw),
        ],
        lm_env(login),
    )
    size = round(2 * edge / ORTHO_CELL_M)
    data = np.fromfile(raw, dtype=np.uint8)
    bands = data.size // (size * size)
    if data.size != bands * size * size:
        raise FetchError(f"ortofotot har inte {size} x {size} pixlar")
    if bands < 3:
        raise FetchError("ortofotot har färre än tre färgband")
    np.save(dest, data.reshape(size, size, bands)[:, :, :3])
    props = chosen[0].get("properties", {})
    return {
        "collection": str(chosen[0].get("collection")),
        "year": int(props.get("flygar") or str(props.get("datetime", ""))[:4]),
        "source_cell_m": float(props["upplosning"]),
        "cell_m": ORTHO_CELL_M,
        "tiles": len(urls),
    }


# --- OpenStreetMap --------------------------------------------------------------------


@dataclass
class OsmData:
    nodes: dict[int, tuple[float, float]]  # id -> (lon, lat)
    ways: dict[int, tuple[list[int], dict[str, str]]]
    relations: dict[int, tuple[list[tuple[str, int, str]], dict[str, str]]]


def parse_osm(raw: bytes, into: OsmData | None = None) -> OsmData:
    data = into or OsmData({}, {}, {})
    root = ET.fromstring(raw)  # noqa: S314 - OSM API response, no entities
    for el in root:
        tags = {t.get("k", ""): t.get("v", "") for t in el.iter("tag")}
        oid = int(el.get("id", "0"))
        if el.tag == "node":
            data.nodes[oid] = (float(el.get("lon", "0")), float(el.get("lat", "0")))
        elif el.tag == "way":
            data.ways[oid] = ([int(n.get("ref", "0")) for n in el.iter("nd")], tags)
        elif el.tag == "relation":
            members = [
                (m.get("type", ""), int(m.get("ref", "0")), m.get("role", ""))
                for m in el.iter("member")
            ]
            data.relations[oid] = (members, tags)
    return data


def fetch_osm(site: SiteDef) -> OsmData:
    min_lon, min_lat, max_lon, max_lat = lonlat_bbox(site, MARGIN_M)
    data = parse_osm(
        _get(f"{OSM_API}/map?bbox={min_lon:.6f},{min_lat:.6f},{max_lon:.6f},{max_lat:.6f}")
    )
    # Multipolygons (big forests and lakes) reach outside the box: fetch them whole.
    for rid, (_, tags) in sorted(data.relations.items()):
        if tags.get("type") == "multipolygon" and area_kind(tags):
            parse_osm(_get(f"{OSM_API}/relation/{rid}/full"), data)
    return data


def area_kind(tags: Mapping[str, str]) -> str | None:
    for (key, value), kind in AREA_KINDS.items():
        if tags.get(key) == value:
            return kind
    return None


def _coords(data: OsmData, refs: Iterable[int]) -> list[tuple[float, float]] | None:
    out = []
    for ref in refs:
        if ref not in data.nodes:
            return None
        out.append(data.nodes[ref])
    return out


def _rings(data: OsmData, way_ids: Sequence[int]) -> list[list[tuple[float, float]]]:
    """Join a relation's member ways into closed rings (ways that cannot close are dropped)."""
    open_ways = []
    rings = []
    for wid in way_ids:
        if wid not in data.ways:
            continue
        coords = _coords(data, data.ways[wid][0])
        if not coords or len(coords) < 2:
            continue
        open_ways.append(coords)
    while open_ways:
        ring = open_ways.pop(0)
        changed = True
        while ring[0] != ring[-1] and changed:
            changed = False
            for i, way in enumerate(open_ways):
                if way[0] == ring[-1]:
                    ring += way[1:]
                elif way[-1] == ring[-1]:
                    ring += way[::-1][1:]
                elif way[-1] == ring[0]:
                    ring = way[:-1] + ring
                elif way[0] == ring[0]:
                    ring = way[::-1][:-1] + ring
                else:
                    continue
                open_ways.pop(i)
                changed = True
                break
        if ring[0] == ring[-1] and len(ring) >= 4:
            rings.append(ring)
    return rings


def _polygon_from_rings(
    outers: list[list[tuple[float, float]]], inners: list[list[tuple[float, float]]]
) -> MultiPolygon | Polygon:
    polys = []
    holes = [Polygon(r) for r in inners]
    for ring in outers:
        poly = Polygon(ring)
        inside = [h.exterior.coords for h in holes if poly.contains(h.representative_point())]
        polys.append(Polygon(ring, inside))
    valid = shapely.make_valid(MultiPolygon(polys) if len(polys) > 1 else polys[0])
    return cast(MultiPolygon | Polygon, valid)


def _round_geom(geom: Any) -> Any:
    return shapely.set_precision(geom, 10**-COORD_DIGITS)


def osm_features(data: OsmData, clip: Polygon) -> list[dict[str, Any]]:
    """GeoJSON features for the generator, clipped to the site, sorted for stable output."""
    out: list[dict[str, Any]] = []

    def add(kind: str, osm_id: str, geom: Any, props: dict[str, Any]) -> None:
        geom = _round_geom(geom.intersection(clip))
        if geom.is_empty or geom.geom_type not in (
            "Polygon",
            "MultiPolygon",
            "LineString",
            "MultiLineString",
        ):
            if geom.is_empty or geom.geom_type != "GeometryCollection":
                return
            parts = [g for g in geom.geoms if g.geom_type in ("Polygon", "LineString")]
            if not parts:
                return
            geom = shapely.union_all(parts)
        out.append(
            {
                "type": "Feature",
                "properties": {"kind": kind, "osm": osm_id, **props},
                "geometry": mapping(geom),
            }
        )

    for wid, (refs, tags) in sorted(data.ways.items()):
        coords = _coords(data, refs)
        if not coords or len(coords) < 2:
            continue
        closed = refs[0] == refs[-1] and len(refs) >= 4
        if "building" in tags and closed:
            add("building", f"w{wid}", shapely.make_valid(Polygon(coords)), building_props(tags))
        elif tags.get("highway") in ROAD_WIDTH_M and tags.get("area") != "yes":
            add("road", f"w{wid}", LineString(coords), road_props(tags))
        elif closed and (kind := area_kind(tags)):
            add(kind, f"w{wid}", shapely.make_valid(Polygon(coords)), {})
    for rid, (members, tags) in sorted(data.relations.items()):
        if tags.get("type") != "multipolygon":
            continue
        kind = "building" if "building" in tags else area_kind(tags)
        if kind is None:
            continue
        outers = _rings(
            data, [ref for t, ref, role in members if t == "way" and role in ("outer", "")]
        )
        inners = _rings(data, [ref for t, ref, role in members if t == "way" and role == "inner"])
        if outers:
            props = building_props(tags) if kind == "building" else {}
            add(kind, f"r{rid}", _polygon_from_rings(outers, inners), props)
    return sorted(out, key=lambda f: (f["properties"]["kind"], f["properties"]["osm"]))


def _float_tag(tags: Mapping[str, str], key: str) -> float | None:
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(m)?\s*$", tags.get(key, ""))
    return float(m.group(1)) if m else None


def building_props(tags: Mapping[str, str]) -> dict[str, Any]:
    props: dict[str, Any] = {"building": tags.get("building", "yes")}
    if (h := _float_tag(tags, "height")) is not None:
        props["height_m"] = h
    if (lv := _float_tag(tags, "building:levels")) is not None:
        props["levels"] = lv
    if tags.get("roof:shape"):
        props["roof"] = tags["roof:shape"]
    return props


def road_props(tags: Mapping[str, str]) -> dict[str, Any]:
    highway = tags["highway"]
    width = _float_tag(tags, "width")
    props: dict[str, Any] = {"highway": highway, "width_m": width or ROAD_WIDTH_M[highway]}
    if width is not None:
        props["width_tagged"] = True
    if tags.get("surface"):
        props["surface"] = tags["surface"]
    if tags.get("bridge") == "yes" or tags.get("tunnel") == "yes":
        props["layer"] = "bridge" if tags.get("bridge") == "yes" else "tunnel"
    return props


def site_clip(site: SiteDef) -> Polygon:
    min_lon, min_lat, max_lon, max_lat = lonlat_bbox(site, 0.0)
    return box(min_lon, min_lat, max_lon, max_lat)


def write_local_site_toml(site: SiteDef) -> None:
    """The definition the world generator reads, next to the local data; no login."""
    surface = "dtm" if site.dem_source == "lantmateriet-mhm" else "dsm"
    (site.out / "site.toml").write_text(
        "# Written by tools/terrain/fetch_site.py from config/site.local.toml; local only.\n"
        f'name = "{site.name}"\nworld = "sd_{site.name}"\ntitle = "{site.name} (local)"\n'
        f"[home]\nlat = {site.lat}\nlon = {site.lon}\nheading_deg = {site.heading_deg}\n"
        f"[area]\nhalf_size_m = {site.half_size_m}\n"
        f'[dem]\nsource = "{site.dem_source}"\ncell_m = {site.cell_m}\n'
        f'surface = "{surface}"\ncanopy_m = {0 if surface == "dtm" else 12}\n',
        encoding="utf-8",
    )


# --- main -----------------------------------------------------------------------------


def run(argv: Sequence[str], root: Path = ROOT) -> int:
    args = parse_args(argv)
    site, login = load_site(args["site"], root)
    site.out.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.now(dt.UTC).date().isoformat()
    with tempfile.TemporaryDirectory(prefix="sd-world-") as tmp:
        work = Path(tmp)
        print(f"==> höjdmodell ({site.dem_source})")
        asc, sources = fetch_dem(site, login, work)
        home_alt = write_dem(asc, site.out / "dem.asc", site)
        ortho = ""
        if site.ortho:
            print("==> ortofoto (Lantmäteriet)")
            ortho = str(fetch_ortho(site, login, work, site.out / "ortho.npy")["collection"])
    if not site.public:
        write_local_site_toml(site)
    print("==> OpenStreetMap")
    features = osm_features(fetch_osm(site), site_clip(site))
    geojson = {"type": "FeatureCollection", "features": features}
    (site.out / "osm.geojson").write_text(
        json.dumps(geojson, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8"
    )
    meta = [
        "# Written by tools/terrain/fetch_site.py (just world-fetch). Where the data came from.",
        f'retrieved = "{today}"',
        f'dem_source = "{site.dem_source}"',
        f"dem_tiles = {json.dumps(sources)}",
        f"dem_at_home_m = {home_alt:.2f}",
        f"osm_features = {len(features)}",
    ]
    if ortho:
        meta.append(f'ortho_collection = "{ortho}"')
    (site.out / "data.toml").write_text("\n".join(meta) + "\n", encoding="utf-8")
    print(f"Klart: {site.out} ({len(features)} kartobjekt, höjd vid hem {home_alt:.1f} m)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(list(sys.argv[1:] if argv is None else argv))
    except FetchError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
