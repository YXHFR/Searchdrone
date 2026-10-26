"""Pick real places for the search policy simulator: ``just rl-maps [key=value ...]``.

Keys (all optional):

- ``osm=<file>[,<file>...]``: OpenStreetMap extracts (``.osm.pbf``). Default:
  ``build/rl/osm/sweden-latest.osm.pbf``, downloaded from Geofabrik with ``yes=1``.
- ``n=600``: how many places; ``seed=0``: which ones; ``workers=8``: parallel reads.
- ``out=build/rl/sites``: where the places go (relative paths are from the repository).

How a place is picked: a random point in the extracts' area (even over the area, not over
degrees), a 2 x 2 km grid around it, ESA WorldCover 2021 (10 m) for the land cover, read
straight from ESA's cloud copy, only the part over the place. A place is kept when it is
mostly forest and open land, not a lake or a town (``grid.acceptable``); then one pass
over the OpenStreetMap file draws every road and track on the kept places, and a place
with no road or track near its middle is dropped too.

Every place is one small ``<out>/<nnnn>.npz`` (``grid.site_arrays``) and ``sites.json``
says where they are and where the data came from. Nothing is written inside the
repository's tracked files: ``build/`` is ignored. The data: OpenStreetMap
(© OpenStreetMap contributors, ODbL 1.0) and ESA WorldCover 2021 v200 (© ESA, CC BY 4.0).

Needs internet while it runs. Exit codes: 0 done, 1 a download failed or too few places,
2 bad arguments.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import shutil
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NoReturn

import numpy as np
import osmium
import rasterio
from rasterio.windows import from_bounds

from rlmaps import grid

ROOT = Path(__file__).resolve().parents[2]
KEYS = ("osm", "n", "seed", "workers", "out", "yes")
GEOFABRIK_SWEDEN = "https://download.geofabrik.de/europe/sweden-latest.osm.pbf"
DEFAULT_OSM = ROOT / "build/rl/osm/sweden-latest.osm.pbf"
WORLDCOVER = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)
# Not a track for someone on foot: planned, being built, or inside a building.
NOT_TRACKS = {"proposed", "construction", "platform", "corridor", "elevator", "abandoned",
              "razed", "bus_stop", "rest_area", "services"}  # fmt: skip
BUCKET_DEG = 0.05
USER_AGENT = "searchdrone-rl-maps (+https://github.com/YXHFR/Searchdrone)"
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


def _fail(message: str, code: int = 2) -> NoReturn:
    print(f"rl-maps: {message}", file=sys.stderr)
    raise SystemExit(code)


def _parse(argv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in argv:
        key, sep, value = raw.partition("=")
        if not sep or key not in KEYS:
            _fail(f"'{raw}' känns inte igen. Tillåtna nycklar: {', '.join(KEYS)}.")
        out[key] = value
    return out


def _int(args: dict[str, str], key: str, default: int, lowest: int) -> int:
    try:
        value = int(args.get(key, default))
    except ValueError:
        _fail(f"{key} ska vara ett heltal.")
    if value < lowest:
        _fail(f"{key} ska vara minst {lowest}.")
    return value


def _path(raw: str) -> Path:
    p = Path(raw).expanduser()
    return p if p.is_absolute() else ROOT / p


def worldcover_tile(lat: float, lon: float) -> str:
    """The 3 x 3 degree tile name, by its south-west corner (N57E015)."""
    la, lo = math.floor(lat / 3) * 3, math.floor(lon / 3) * 3
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


def read_classes(place: grid.Place) -> np.ndarray | None:
    """WorldCover classes at the place's sample points, or None (sea, or two tiles)."""
    west, south, east, north = place.bounds(margin_m=50.0)
    tiles = {worldcover_tile(lat, lon) for lat in (south, north) for lon in (west, east)}
    if len(tiles) != 1:
        return None
    url = "/vsicurl/" + WORLDCOVER.format(tile=tiles.pop())
    try:
        with rasterio.Env(**GDAL_ENV), rasterio.open(url) as ds:
            win = (
                from_bounds(west, south, east, north, ds.transform).round_offsets().round_lengths()
            )
            data = ds.read(1, window=win)
            transform = ds.window_transform(win)
    except rasterio.errors.RasterioIOError:
        return None  # no tile: open sea
    lon, lat = place.sample_points()
    inv = ~transform
    col, row = inv * (lon, lat)
    col = np.clip(np.floor(col).astype(int), 0, data.shape[1] - 1)
    row = np.clip(np.floor(row).astype(int), 0, data.shape[0] - 1)
    return data[row, col]


def osm_area(paths: list[Path]) -> list[tuple[float, float, float, float]]:
    """west, south, east, north of each extract, from its header."""
    out = []
    for p in paths:
        box = osmium.io.Reader(str(p), osmium.osm.osm_entity_bits.NOTHING).header().box()
        if not box.valid():
            _fail(f"{p} säger inte vilket område den täcker.", 1)
        out.append((box.bottom_left.lon, box.bottom_left.lat, box.top_right.lon, box.top_right.lat))
    return out


def candidates(areas: list[tuple[float, float, float, float]], rng: np.random.Generator):
    """Endless random points, even over the ground (not over degrees)."""
    weights = np.array(
        [(e - w) * (math.sin(math.radians(n)) - math.sin(math.radians(s))) for w, s, e, n in areas]
    )
    weights = weights / weights.sum()
    while True:
        w, s, e, n = areas[int(rng.choice(len(areas), p=weights))]
        # Uniform on the sphere: uniform in sin(latitude).
        lat = math.degrees(
            math.asin(rng.uniform(math.sin(math.radians(s)), math.sin(math.radians(n))))
        )
        lon = rng.uniform(w, e)
        margin_lat = (grid.SIZE_M / 2) / grid.M_PER_DEG_LAT
        margin_lon = (grid.SIZE_M / 2) / grid.Place(lat, lon).m_per_deg_lon()
        if s + margin_lat < lat < n - margin_lat and w + margin_lon < lon < e - margin_lon:
            yield grid.Place(round(lat, 5), round(lon, 5))


def tracks(paths: list[Path], places: list[grid.Place]) -> list[list[np.ndarray]]:
    """Every road and track over each place, as polylines in the place's grid metres."""
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, pl in enumerate(places):
        w, s, e, n = pl.bounds(margin_m=100.0)
        for bx in range(math.floor(w / BUCKET_DEG), math.floor(e / BUCKET_DEG) + 1):
            for by in range(math.floor(s / BUCKET_DEG), math.floor(n / BUCKET_DEG) + 1):
                buckets[(bx, by)].append(i)
    lines: list[list[np.ndarray]] = [[] for _ in places]
    for p in paths:
        print(f"==> stigar och vägar ur {p.name}", flush=True)
        fp = (
            osmium.FileProcessor(str(p), osmium.osm.NODE | osmium.osm.WAY)
            .with_locations()
            .with_filter(osmium.filter.KeyFilter("highway"))
        )
        for obj in fp:
            if not obj.is_way() or obj.tags.get("highway") in NOT_TRACKS:
                continue
            coords = [(n.lon, n.lat) for n in obj.nodes if n.location.valid()]
            if len(coords) < 2:
                continue
            hit = {
                i
                for lon, lat in coords
                for i in buckets.get(
                    (math.floor(lon / BUCKET_DEG), math.floor(lat / BUCKET_DEG)), ()
                )
            }
            if not hit:
                continue
            arr = np.array(coords)
            for i in hit:
                lines[i].append(places[i].to_grid_m(arr[:, 0], arr[:, 1]))
    return lines


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310 (fixed https URL)
    print(f"==> hämtar {url}", flush=True)
    with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as fh:  # noqa: S310
        shutil.copyfileobj(resp, fh, length=1 << 20)
    part.replace(dest)


def main(argv: list[str] | None = None) -> int:
    args = _parse(list(sys.argv[1:] if argv is None else argv))
    n = _int(args, "n", 600, 10)
    seed = _int(args, "seed", 0, 0)
    workers = _int(args, "workers", 8, 1)
    out = _path(args.get("out", "build/rl/sites"))
    if "osm" in args:
        paths = [_path(p) for p in args["osm"].split(",") if p]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            _fail(f"filen finns inte: {', '.join(missing)}.")
    else:
        paths = [DEFAULT_OSM]
        if not DEFAULT_OSM.exists():
            if args.get("yes") != "1":
                print(
                    f"Sverigefilen från OpenStreetMap saknas ({DEFAULT_OSM}). Kör med yes=1 "
                    f"för att hämta den (cirka 700 MB) från {GEOFABRIK_SWEDEN}."
                )
                return 0
            _download(GEOFABRIK_SWEDEN, DEFAULT_OSM)

    rng = np.random.default_rng(seed)
    gen = candidates(osm_area(paths), rng)
    kept: list[tuple[grid.Place, dict[int, np.ndarray]]] = []
    tried = 0
    reasons: dict[str, int] = defaultdict(int)
    print(f"==> marktäcke (ESA WorldCover) för {n} platser", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        while len(kept) < int(n * 1.3) and tried < n * 40:
            batch = [next(gen) for _ in range(workers * 4)]
            tried += len(batch)
            for place, classes in zip(batch, pool.map(read_classes, batch), strict=True):
                if classes is None:
                    reasons["havet eller kanten mellan två kartblad"] += 1
                    continue
                shares = grid.class_shares(classes)
                why = grid.acceptable(shares, np.ones((grid.GRID_CELLS,) * 2, dtype=bool))
                if why:
                    reasons[why] += 1
                    continue
                kept.append((place, shares))
            print(f"    {len(kept)} godkända av {tried} provade", flush=True)
    lines = tracks(paths, [p for p, _ in kept])
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.npz"):
        old.unlink()
    sites = []
    for (place, shares), pl_lines in zip(kept, lines, strict=True):
        track = grid.rasterise(pl_lines)
        why = grid.acceptable(shares, track)
        if why:
            reasons[why] += 1
            continue
        name = f"{len(sites):04d}"
        np.savez_compressed(out / f"{name}.npz", **grid.site_arrays(shares, track))
        sites.append({"name": name, "lat": place.lat, "lon": place.lon})
        if len(sites) == n:
            break
    doc = {
        "schema": "rlmaps.sites/v1",
        "made": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "cell_m": grid.CELL_M,
        "grid_cells": grid.GRID_CELLS,
        "seed": seed,
        "osm_files": [p.name for p in paths],
        "sources": [
            "OpenStreetMap: © OpenStreetMap contributors, ODbL 1.0 (openstreetmap.org/copyright)",
            "ESA WorldCover 2021 v200: © ESA WorldCover project / Contains modified Copernicus "
            "Sentinel data (2021) processed by ESA WorldCover consortium, CC BY 4.0",
        ],
        "rejected": dict(reasons),
        "sites": sites,
    }
    (out / "sites.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"klart: {len(sites)} platser i {out}. Bortvalda: {dict(reasons)}")
    if len(sites) < n:
        print(f"rl-maps: bara {len(sites)} av {n} platser hittades.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    os.environ.setdefault("GDAL_HTTP_USERAGENT", USER_AGENT)
    raise SystemExit(main())
