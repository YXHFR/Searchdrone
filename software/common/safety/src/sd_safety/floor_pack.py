"""The map pack the height floor (:mod:`sd_safety.floor`) counts from, and its loader.

The only module in ``sd_safety`` that reads files: a pack is loaded once, before the
flight, and the floor is worked out from it in memory. A pack lives in
``${SD_DATA_DIR}/floor/<name>/``, outside the repository (map data is never committed)::

    floor_pack.toml   format, name, attribution, [origin] lat/lon, [grid] size_m
                      (east, north) and scan_cell_m, [sha256] of every file below
    dtm_cm.npy        ground height above sea level (RH 2000), cm, int32;
                      DTM_NODATA_CM = no ground height
    chm_dm.npy        tree height above the ground, dm, int16; -1 = unknown
    scan_year.npy     the scan's year per scan_cell_m square, uint16; 0 = unknown
    lines.geojson     known power lines (LineString/MultiLineString, WGS84), may be empty

The grids are 1 m cells in the east/north plane tangent at the origin
(``sd_schemas.geo.local_xy``), centred on it, row 0 north (like a site's ``chm_dm.npy``
from ``tools/terrain/fetch_trees.py``). ``scan_year`` covers the same area in squares of
``scan_cell_m`` from the north-west corner; where a square holds several scans, the
oldest year belongs there. ``FloorPack.sha256`` is the hash of ``floor_pack.toml``, which
holds every file's hash: it names the whole pack (``MissionCharter.floor_pack_sha256``).
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from sd_safety.targets import Plane, Xy
from sd_schemas.base import LatLon

__all__ = [
    "CHM_UNKNOWN_DM",
    "DTM_NODATA_CM",
    "FLOOR_PACK_FILES",
    "FLOOR_PACK_FORMAT",
    "SCAN_YEAR_UNKNOWN",
    "FloorPack",
    "FloorPackError",
    "Line",
    "load_floor_pack",
]

FLOOR_PACK_FORMAT: Final = 1
FLOOR_PACK_FILES: Final = ("dtm_cm.npy", "chm_dm.npy", "scan_year.npy", "lines.geojson")
DTM_NODATA_CM: Final = int(np.iinfo(np.int32).min)
CHM_UNKNOWN_DM: Final = -1
SCAN_YEAR_UNKNOWN: Final = 0

Line = tuple[Xy, ...]  # a power line's corners in the pack's plane


class FloorPackError(ValueError):
    """A map pack that is incomplete, altered or not in this format."""


@dataclass(frozen=True, eq=False)
class FloorPack:
    """A map pack in memory. Build it with :func:`load_floor_pack`, or directly in tests."""

    name: str
    sha256: str
    origin: LatLon
    dtm_cm: NDArray[np.int32]
    chm_dm: NDArray[np.int16]
    scan_year: NDArray[np.uint16]
    scan_cell_m: int
    lines: tuple[Line, ...] = ()
    attribution: str = ""
    plane: Plane = field(init=False)

    def __post_init__(self) -> None:
        if self.dtm_cm.dtype != np.int32 or self.dtm_cm.ndim != 2 or self.dtm_cm.size == 0:
            raise FloorPackError("dtm_cm is a non-empty 2-D int32 grid")
        if self.chm_dm.dtype != np.int16 or self.chm_dm.shape != self.dtm_cm.shape:
            raise FloorPackError("chm_dm is an int16 grid of dtm_cm's shape")
        if self.scan_cell_m < 1:
            raise FloorPackError("scan_cell_m is at least 1")
        rows, cols = self.dtm_cm.shape
        tiles = (-(-rows // self.scan_cell_m), -(-cols // self.scan_cell_m))
        if self.scan_year.dtype != np.uint16 or self.scan_year.shape != tiles:
            raise FloorPackError(f"scan_year is a uint16 grid of shape {tiles}")
        if any(len(line) < 2 for line in self.lines):
            raise FloorPackError("a power line has at least two points")
        object.__setattr__(self, "plane", Plane(self.origin.lat, self.origin.lon))

    def line_xy(self, points: Sequence[LatLon]) -> Line:
        """A line given in latitude/longitude (one the pilot drew), in the pack's plane."""
        return tuple(self.plane.point(p) for p in points)


def load_floor_pack(folder: Path) -> FloorPack:
    """Read and check a pack: every file's hash must match ``floor_pack.toml``. The grids
    are memory-mapped, not read whole."""
    try:
        raw = (folder / "floor_pack.toml").read_bytes()
        meta = tomllib.loads(raw.decode("utf-8"))
        if meta.get("format") != FLOOR_PACK_FORMAT:
            raise FloorPackError(f"floor_pack.toml: format is not {FLOOR_PACK_FORMAT}")
        hashes = meta["sha256"]
        for name in FLOOR_PACK_FILES:
            if _file_sha256(folder / name) != hashes[name]:
                raise FloorPackError(f"{name} does not match its sha256")
        dtm = np.load(folder / "dtm_cm.npy", mmap_mode="r", allow_pickle=False)
        cols, rows = meta["grid"]["size_m"]
        if dtm.shape != (rows, cols):
            raise FloorPackError("dtm_cm.npy does not have the size in floor_pack.toml")
        origin = LatLon(lat=meta["origin"]["lat"], lon=meta["origin"]["lon"])
        lines = _read_lines(folder / "lines.geojson", Plane(origin.lat, origin.lon))
        return FloorPack(
            name=str(meta["name"]),
            sha256=hashlib.sha256(raw).hexdigest(),
            origin=origin,
            dtm_cm=dtm,
            chm_dm=np.load(folder / "chm_dm.npy", mmap_mode="r", allow_pickle=False),
            scan_year=np.load(folder / "scan_year.npy", mmap_mode="r", allow_pickle=False),
            scan_cell_m=int(meta["grid"]["scan_cell_m"]),
            lines=lines,
            attribution=str(meta["attribution"]),
        )
    except FloorPackError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise FloorPackError(f"{folder}: not a readable floor pack ({exc})") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _read_lines(path: Path, plane: Plane) -> tuple[Line, ...]:
    """The power lines in a GeoJSON FeatureCollection, in the pack's plane."""
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise FloorPackError("lines.geojson is a FeatureCollection")
    lines: list[Line] = []
    for feature in data["features"]:
        geom = feature["geometry"]
        if geom["type"] == "LineString":
            parts = [geom["coordinates"]]
        elif geom["type"] == "MultiLineString":
            parts = geom["coordinates"]
        else:
            raise FloorPackError(f"lines.geojson: a {geom['type']} is not a line")
        for part in parts:
            pts = [LatLon(lat=float(pt[1]), lon=float(pt[0])) for pt in part]
            lines.append(tuple(plane.point(q) for q in pts))
    return tuple(lines)
