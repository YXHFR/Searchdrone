"""Build ``fixture.pmtiles``, the small synthetic offline map for the panel tests.

The input is only the committed ``fixture-area.geojson`` next to this file (made-up shapes
around the public ArduPilot SITL site; no OpenStreetMap data). Every feature names its
vector layer in ``properties.layer`` with the Protomaps basemap layer names (``earth``,
``water``, ``landcover``, ``roads``), so the panel's real map style draws it.

Vector tiles for zoom 12-15, gzip-compressed, written byte for byte the same on every run
(``tools/tests/test_map_fixture.py`` checks that the committed file is up to date).

    uv run python software/panel/e2e/fixtures/make_fixture_pmtiles.py
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

import mapbox_vector_tile
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer
from shapely.geometry import box, mapping, shape
from shapely.ops import transform

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "fixture-area.geojson"
TARGET = HERE / "fixture.pmtiles"
MIN_ZOOM, MAX_ZOOM = 12, 15
EXTENT = 4096
BUFFER = 64  # tile units, so lines and polygon edges do not show seams
R = 6378137.0


def _merc(lon: float, lat: float, z: float | None = None) -> tuple[float, float]:
    x = math.radians(lon) * R
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R
    return x, y


def _tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    size = 2 * math.pi * R / 2**z
    minx = -math.pi * R + x * size
    maxy = math.pi * R - y * size
    return minx, maxy - size, minx + size, maxy


def _tile_range(z: int, lon0: float, lon1: float) -> range:
    def tx(lon: float) -> int:
        return int((lon + 180) / 360 * 2**z)

    return range(tx(lon0), tx(lon1) + 1)


def _tile_y(z: int, lat: float) -> int:
    rad = math.radians(lat)
    return int((1 - math.asinh(math.tan(rad)) / math.pi) / 2 * 2**z)


def build(source: Path = SOURCE) -> bytes:
    data = json.loads(source.read_text(encoding="utf-8"))
    features = [(f["properties"], shape(f["geometry"])) for f in data["features"]]
    lon0, lat0, lon1, lat1 = (
        min(g.bounds[0] for _, g in features),
        min(g.bounds[1] for _, g in features),
        max(g.bounds[2] for _, g in features),
        max(g.bounds[3] for _, g in features),
    )
    merc = [(p, transform(_merc, g)) for p, g in features]

    tiles: list[tuple[int, bytes]] = []
    for z in range(MIN_ZOOM, MAX_ZOOM + 1):
        for x in _tile_range(z, lon0, lon1):
            for y in range(_tile_y(z, lat1), _tile_y(z, lat0) + 1):
                minx, miny, maxx, maxy = _tile_bounds(z, x, y)
                pad = (maxx - minx) * BUFFER / EXTENT
                clip = box(minx - pad, miny - pad, maxx + pad, maxy + pad)
                layers: dict[str, list[dict[str, Any]]] = {}
                for props, geom in merc:
                    part = geom.intersection(clip)
                    if part.is_empty:
                        continue
                    attrs = {k: v for k, v in props.items() if k != "layer"}
                    layers.setdefault(props["layer"], []).append(
                        {"geometry": mapping(part), "properties": attrs}
                    )
                if not layers:
                    continue
                mvt = mapbox_vector_tile.encode(
                    [{"name": n, "features": f} for n, f in sorted(layers.items())],
                    default_options={
                        "quantize_bounds": (minx, miny, maxx, maxy),
                        "extents": EXTENT,
                        "y_coord_down": False,
                    },
                )
                tiles.append((zxy_to_tileid(z, x, y), gzip.compress(mvt, mtime=0)))

    out = _Buffer()
    writer = Writer(out)
    for tile_id, blob in sorted(tiles):
        writer.write_tile(tile_id, blob)
    e7 = 10_000_000
    writer.finalize(
        {
            "tile_type": TileType.MVT,
            "tile_compression": Compression.GZIP,
            "min_zoom": MIN_ZOOM,
            "max_zoom": MAX_ZOOM,
            "min_lon_e7": round(lon0 * e7),
            "min_lat_e7": round(lat0 * e7),
            "max_lon_e7": round(lon1 * e7),
            "max_lat_e7": round(lat1 * e7),
            "center_zoom": MIN_ZOOM + 2,
            "center_lon_e7": round((lon0 + lon1) / 2 * e7),
            "center_lat_e7": round((lat0 + lat1) / 2 * e7),
        },
        {
            "name": "searchdrone-test-area",
            "attribution": "Searchdrone test data (synthetic)",
            "vector_layers": [
                {"id": n, "fields": {"kind": "String"}}
                for n in sorted({p["layer"] for p, _ in features})
            ],
        },
    )
    return out.getvalue()


class _Buffer:
    """A write-only byte buffer (the writer only calls ``write``)."""

    def __init__(self) -> None:
        self._parts: list[bytes] = []

    def write(self, b: bytes) -> int:
        self._parts.append(bytes(b))
        return len(b)

    def getvalue(self) -> bytes:
        return b"".join(self._parts)


def main() -> int:
    blob = build()
    TARGET.write_bytes(blob)
    sys.stdout.write(f"{TARGET.name}: {len(blob)} bytes\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
