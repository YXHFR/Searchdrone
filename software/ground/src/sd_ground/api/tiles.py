"""The offline map: ``GET /tiles/{file}`` (outside ``/api/v1``) and ``PanelConfig.map``.

``just maps-fetch`` writes the map pack ``<data_dir>/tiles/<pack>.pmtiles`` and the glyphs
and sprites under ``<data_dir>/tiles/assets/``. Nothing here reaches the internet: the
panel reads the pack with HTTP Range requests (PMTiles) and the server only hands out
files whose names match a fixed pattern and that lie inside the tiles directory.
"""

from __future__ import annotations

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
from sd_schemas.api import MapConfig
from sd_schemas.base import LatLon

__all__ = ["map_config", "read_center", "router", "tiles_dir"]

log = get_logger("sd_ground.tiles")
router = APIRouter()

Observer = Annotated[Principal, Depends(require("observer"))]
G = Annotated[Ground, Depends(ground)]

_PACK = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}\.pmtiles$")
_FONT = re.compile(r"^assets/fonts/[A-Za-z0-9 ]{1,64}/[0-9]{1,5}-[0-9]{1,5}\.pbf$")
_SPRITE = re.compile(r"^assets/sprites/v4/[a-z]{1,16}(@2x)?\.(json|png)$")
_TYPES = {
    ".pmtiles": "application/vnd.pmtiles",
    ".pbf": "application/x-protobuf",
    ".json": "application/json",
    ".png": "image/png",
}
# PMTiles v3 header: bounds at bytes 102-117, centre at 119-126 (int32 little-endian, 1e-7°).
_HEADER_LEN = 127
_MAGIC = b"PMTiles\x03"


def tiles_dir(g: Ground) -> Path:
    return g.loaded.data_dir / "tiles"


def read_center(path: Path) -> LatLon | None:
    """The centre stored in a PMTiles v3 header, or the middle of its bounds."""
    try:
        with path.open("rb") as f:
            head = f.read(_HEADER_LEN)
    except OSError:
        return None
    if len(head) < _HEADER_LEN or not head.startswith(_MAGIC):
        return None
    min_lon, min_lat, max_lon, max_lat = (v / 1e7 for v in struct.unpack_from("<4i", head, 102))
    lon, lat = (v / 1e7 for v in struct.unpack_from("<2i", head, 119))
    if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
        lon, lat = (min_lon + max_lon) / 2, (min_lat + max_lat) / 2
    try:
        return LatLon(lat=lat, lon=lon)
    except ValueError:
        return None


def map_config(g: Ground) -> MapConfig | None:
    """``PanelConfig.map``: set only when the configured map pack exists and is readable."""
    pack = tiles_dir(g) / f"{g.config.map.pack}.pmtiles"
    if not pack.is_file():
        return None
    center = read_center(pack)
    if center is None:
        log.warning("tiles.bad_pack", pack=pack.name)
        return None
    return MapConfig(pack_url=f"/tiles/{pack.name}", center=center, zoom=g.config.map.zoom)


@router.get("/tiles/{file:path}", response_class=FileResponse)
async def get_tile_file(file: str, g: G, _: Observer) -> FileResponse:
    """A map pack (HTTP Range supported), a glyph range or a sprite file."""
    missing = ApiStatusError(404, "NOT_FOUND", "Kartfilen finns inte.")
    if not (_PACK.match(file) or _FONT.match(file) or _SPRITE.match(file)):
        raise missing
    root = tiles_dir(g).resolve()
    path = (root / file).resolve()  # a symlink out of the tiles directory is refused
    if not path.is_relative_to(root) or not path.is_file():
        raise missing
    return FileResponse(path, media_type=_TYPES[path.suffix], headers={"Cache-Control": "no-cache"})
