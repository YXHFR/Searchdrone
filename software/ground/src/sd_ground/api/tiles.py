"""The offline map: ``GET /tiles/{file}`` (outside ``/api/v1``) and ``PanelConfig.map``.

``just maps-fetch`` writes the map pack ``<data_dir>/tiles/<pack>.pmtiles`` and the glyphs
and sprites under ``<data_dir>/tiles/assets/``. An aerial photo pack
``<data_dir>/tiles/<imagery>.pmtiles`` (raster tiles, attribution in its metadata) is optional;
the panel draws it under the map's roads, paths and names. Nothing here reaches the internet: the
panel reads the pack with HTTP Range requests (PMTiles) and the server only hands out
files whose names match a fixed pattern and that lie inside the tiles directory.
"""

from __future__ import annotations

import gzip
import json
import re
import struct
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from sd_ground.api.deps import ApiStatusError, ground, require
from sd_ground.auth import Principal
from sd_ground.context import Ground
from sd_ground.logging import get_logger
from sd_schemas.api import MapConfig, MapImagery
from sd_schemas.base import LatLon

__all__ = [
    "PACK_NAME",
    "imagery_config",
    "map_config",
    "read_attribution",
    "read_center",
    "read_maxzoom",
    "router",
    "tiles_dir",
]

log = get_logger("sd_ground.tiles")
router = APIRouter()

Observer = Annotated[Principal, Depends(require("observer"))]
G = Annotated[Ground, Depends(ground)]

PACK_NAME = r"[a-z0-9][a-z0-9_-]{0,63}"  # also checked by `just maps-fetch` and the config
_PACK = re.compile(rf"^{PACK_NAME}\.pmtiles$")
_FONT = re.compile(r"^assets/fonts/[A-Za-z0-9 ]{1,64}/[0-9]{1,5}-[0-9]{1,5}\.pbf$")
_SPRITE = re.compile(r"^assets/sprites/v4/[a-z]{1,16}(@2x)?\.(json|png)$")
_TYPES = {
    ".pmtiles": "application/vnd.pmtiles",
    ".pbf": "application/x-protobuf",
    ".json": "application/json",
    ".png": "image/png",
}
# PMTiles v3 header: metadata offset and length at bytes 24-39 (uint64), directory
# compression at 97, tile type at 99, max zoom at 101, bounds at 102-117, centre at 119-126
# (int32 little-endian, 1e-7°).
_HEADER_LEN = 127
_MAGIC = b"PMTiles\x03"
_RASTER_TYPES = {2, 3, 4, 5}  # png, jpeg, webp, avif
_COMPRESSION_NONE, _COMPRESSION_GZIP = 1, 2
_MAX_METADATA = 64 * 1024
_MAX_ATTRIBUTION = 300


def tiles_dir(g: Ground) -> Path:
    return g.loaded.data_dir / "tiles"


def _header(path: Path) -> bytes | None:
    try:
        with path.open("rb") as f:
            head = f.read(_HEADER_LEN)
    except OSError:
        return None
    return head if len(head) == _HEADER_LEN and head.startswith(_MAGIC) else None


def read_maxzoom(path: Path) -> int | None:
    """The highest zoom level with tiles in a PMTiles v3 pack."""
    head = _header(path)
    return None if head is None else min(head[101], 22)


def read_center(path: Path) -> LatLon | None:
    """The centre stored in a PMTiles v3 header, or the middle of its bounds."""
    head = _header(path)
    if head is None:
        return None
    min_lon, min_lat, max_lon, max_lat = (v / 1e7 for v in struct.unpack_from("<4i", head, 102))
    lon, lat = (v / 1e7 for v in struct.unpack_from("<2i", head, 119))
    if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
        lon, lat = (min_lon + max_lon) / 2, (min_lat + max_lat) / 2
    try:
        return LatLon(lat=lat, lon=lon)
    except ValueError:
        return None


def read_attribution(path: Path) -> str | None:
    """The ``attribution`` text in a PMTiles v3 pack's metadata, if it has one."""
    head = _header(path)
    if head is None:
        return None
    offset, length = struct.unpack_from("<2Q", head, 24)
    if not 0 < length <= _MAX_METADATA:
        return None
    try:
        with path.open("rb") as f:
            f.seek(offset)
            raw = f.read(length)
        if head[97] == _COMPRESSION_GZIP:
            raw = gzip.decompress(raw)
        elif head[97] != _COMPRESSION_NONE:
            return None
        meta = json.loads(raw)
    except (OSError, EOFError, ValueError):
        return None
    text = meta.get("attribution") if isinstance(meta, dict) else None
    if not isinstance(text, str) or not 0 < len(text.strip()) <= _MAX_ATTRIBUTION:
        return None
    return text.strip()


def _contained(g: Ground, name: str) -> Path | None:
    """``<tiles>/<name>`` if it is a regular file inside the tiles directory (no symlink out)."""
    root = tiles_dir(g).resolve()
    path = (root / name).resolve()
    return path if path.is_relative_to(root) and path.is_file() else None


def map_config(g: Ground) -> MapConfig | None:
    """``PanelConfig.map``: set only when the configured map pack exists and is readable."""
    name = f"{g.config.map.pack}.pmtiles"
    pack = _contained(g, name)
    if pack is None:
        return None
    center, maxzoom = read_center(pack), read_maxzoom(pack)
    if center is None or maxzoom is None:
        log.warning("tiles.bad_pack", pack=name)
        return None
    return MapConfig(
        pack_url=f"/tiles/{name}",
        center=center,
        zoom=g.config.map.zoom,
        maxzoom=maxzoom,
        imagery=imagery_config(g),
    )


def imagery_config(g: Ground) -> MapImagery | None:
    """The aerial photo pack, set only when it is a raster pack that names its attribution."""
    name = f"{g.config.map.imagery}.pmtiles"
    pack = _contained(g, name)
    if pack is None:
        return None
    head = _header(pack)
    attribution = read_attribution(pack)
    if head is None or head[99] not in _RASTER_TYPES or attribution is None:
        log.warning("tiles.bad_imagery", pack=name)
        return None
    return MapImagery(
        pack_url=f"/tiles/{name}", maxzoom=min(head[101], 22), attribution=attribution
    )


@router.get("/tiles/{file:path}", response_class=FileResponse)
async def get_tile_file(file: str, g: G, _: Observer) -> FileResponse:
    """A map pack (HTTP Range supported), a glyph range or a sprite file."""
    missing = ApiStatusError(404, "NOT_FOUND", "Kartfilen finns inte.")
    if not (_PACK.match(file) or _FONT.match(file) or _SPRITE.match(file)):
        raise missing
    path = _contained(g, file)
    if path is None:
        raise missing
    # The type comes from the requested name, which the patterns above fixed.
    media_type = _TYPES[Path(file).suffix]
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-cache"})
