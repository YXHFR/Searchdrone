"""A tiny synthetic floor pack for the floor tests: flat ground, trees where a test puts
them, written to disk in the pack format when a test needs the loader. Not real map data.

Points are given in the pack's plane (east, north metres from its centre); ``at`` turns
them into latitude/longitude.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from pathlib import Path

import numpy as np

from sd_safety.floor import FloorPack
from sd_safety.targets import Plane
from sd_schemas.base import LatLon

ORIGIN = LatLon(lat=59.3, lon=18.2)  # a round point in Sweden, nothing there in particular
PLANE = Plane(ORIGIN.lat, ORIGIN.lon)
GROUND_M = 40.0
FLIGHT = date(2026, 6, 1)
START_AMSL = GROUND_M + 60.0


def at(east: float, north: float) -> LatLon:
    lat, lon = PLANE.latlon(east, north)
    return LatLon(lat=lat, lon=lon)


def cell(size: int, east: float, north: float) -> tuple[int, int]:
    """The (row, column) of the 1 m cell under a point, in a grid of ``size`` x ``size``."""
    return math.floor(size / 2 - north), math.floor(east + size / 2)


def make_pack(
    size: int = 60,
    *,
    tree_m: float = 0.0,
    scan_year: int = 2019,
    scan_cell_m: int = 2500,
    trees: dict[tuple[float, float], float] | None = None,
    ground: dict[tuple[float, float], float] | None = None,
    lines: tuple[tuple[tuple[float, float], ...], ...] = (),
) -> FloorPack:
    """Flat ground at ``GROUND_M``, trees ``tree_m`` tall everywhere, plus ``trees`` (m;
    -1 = unknown) and ``ground`` (m above sea level; nan = no ground height) at points.
    ``lines``: corners in the plane."""
    dtm = np.full((size, size), round(GROUND_M * 100), np.int32)
    chm = np.full((size, size), round(tree_m * 10), np.int16)
    for (east, north), height in (trees or {}).items():
        chm[cell(size, east, north)] = -1 if height < 0 else round(height * 10)
    for (east, north), level in (ground or {}).items():
        dtm[cell(size, east, north)] = (
            np.iinfo(np.int32).min if math.isnan(level) else round(level * 100)
        )
    tiles = -(-size // scan_cell_m)
    return FloorPack(
        name="synthetic",
        sha256="0" * 64,
        origin=ORIGIN,
        dtm_cm=dtm,
        chm_dm=chm,
        scan_year=np.full((tiles, tiles), scan_year, np.uint16),
        scan_cell_m=scan_cell_m,
        lines=lines,
        attribution="synthetic test data",
    )


def write_pack(folder: Path, pack: FloorPack, lines: list[list[LatLon]] | None = None) -> Path:
    """Write ``pack`` in the pack format (lines as LineStrings in latitude/longitude)."""
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "dtm_cm.npy", pack.dtm_cm)
    np.save(folder / "chm_dm.npy", pack.chm_dm)
    np.save(folder / "scan_year.npy", pack.scan_year)
    features = [
        {
            "type": "Feature",
            "properties": {"power": "line"},
            "geometry": {"type": "LineString", "coordinates": [[p.lon, p.lat] for p in line]},
        }
        for line in lines or []
    ]
    (folder / "lines.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )
    rows, cols = pack.dtm_cm.shape
    hashes = "\n".join(
        f'"{name}" = "{hashlib.sha256((folder / name).read_bytes()).hexdigest()}"'
        for name in ("dtm_cm.npy", "chm_dm.npy", "scan_year.npy", "lines.geojson")
    )
    (folder / "floor_pack.toml").write_text(
        f"""format = 1
name = "{pack.name}"
attribution = "{pack.attribution}"
script_version = "0.0.0"

[origin]
lat = {pack.origin.lat}
lon = {pack.origin.lon}

[grid]
size_m = [{cols}, {rows}]
scan_cell_m = {pack.scan_cell_m}

[sha256]
{hashes}
""",
        encoding="utf-8",
    )
    return folder
