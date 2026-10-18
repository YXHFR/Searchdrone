"""The offline map: ``/tiles/`` files with HTTP Range, and ``PanelConfig.map`` (B1-11)."""

from __future__ import annotations

import shutil
import struct
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer

from sd_ground.api.tiles import read_attribution, read_center
from sd_ground.context import Ground

REPO = Path(__file__).resolve().parents[4]
FIXTURE = REPO / "software" / "panel" / "e2e" / "fixtures" / "fixture.pmtiles"
CMAC_LAT, CMAC_LON = -35.363261, 149.165230


@pytest.fixture
def tiles(ground: Ground) -> Path:
    root = ground.loaded.data_dir / "tiles"
    (root / "assets" / "fonts" / "Noto Sans Regular").mkdir(parents=True)
    (root / "assets" / "sprites" / "v4").mkdir(parents=True)
    shutil.copyfile(FIXTURE, root / "demo.pmtiles")
    (root / "assets" / "fonts" / "Noto Sans Regular" / "0-255.pbf").write_bytes(b"glyphs")
    (root / "assets" / "sprites" / "v4" / "dark.json").write_text("{}")
    (root / "assets" / "sprites" / "v4" / "dark@2x.png").write_bytes(b"\x89PNG")
    (root / "secret.txt").write_text("not a map file")
    return root


def test_no_pack_means_no_map(logged_in: TestClient) -> None:
    assert logged_in.get("/api/v1/panel/config").json()["map"] is None


def test_panel_config_points_at_the_pack(logged_in: TestClient, tiles: Path) -> None:
    body = logged_in.get("/api/v1/panel/config").json()["map"]
    assert body["pack_url"] == "/tiles/demo.pmtiles"
    assert body["assets_url"] == "/tiles/assets"
    assert body["zoom"] == 15
    assert body["maxzoom"] == 15  # from the pack header
    assert "OpenStreetMap" in body["attribution"]
    assert abs(body["center"]["lat"] - CMAC_LAT) < 0.01
    assert abs(body["center"]["lon"] - CMAC_LON) < 0.01


def test_pack_is_served_with_range(logged_in: TestClient, tiles: Path) -> None:
    whole = logged_in.get("/tiles/demo.pmtiles")
    assert whole.status_code == 200
    assert whole.content == FIXTURE.read_bytes()
    part = logged_in.get("/tiles/demo.pmtiles", headers={"Range": "bytes=0-126"})
    assert part.status_code == 206
    assert part.content == FIXTURE.read_bytes()[:127]
    assert part.headers["content-range"].startswith("bytes 0-126/")
    assert part.headers["content-type"] == "application/vnd.pmtiles"
    # The panel listener's security headers are on these responses too.
    assert "default-src 'self'" in part.headers["content-security-policy"]


def test_glyphs_and_sprites(logged_in: TestClient, tiles: Path) -> None:
    glyph = logged_in.get("/tiles/assets/fonts/Noto%20Sans%20Regular/0-255.pbf")
    assert glyph.status_code == 200
    assert glyph.headers["content-type"] == "application/x-protobuf"
    assert logged_in.get("/tiles/assets/sprites/v4/dark.json").json() == {}
    assert logged_in.get("/tiles/assets/sprites/v4/dark@2x.png").status_code == 200


@pytest.mark.parametrize(
    "path",
    [
        "/tiles/secret.txt",
        "/tiles/missing.pmtiles",
        "/tiles/../config/ground.toml",
        "/tiles/%2e%2e/%2e%2e/etc/passwd",
        "/tiles/assets/fonts/..%2F..%2Fdemo/0-255.pbf",
        "/tiles/assets/sprites/v4/dark.svg",
        "/tiles/assets/sprites/v3/dark.json",
        "/tiles/Demo.pmtiles",
    ],
)
def test_only_map_files_inside_the_tiles_dir(logged_in: TestClient, tiles: Path, path: str) -> None:
    assert logged_in.get(path).status_code == 404


def test_symlink_out_of_the_tiles_dir_is_refused(
    logged_in: TestClient, tiles: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.pmtiles"
    outside.write_bytes(FIXTURE.read_bytes())
    (tiles / "link.pmtiles").symlink_to(outside)
    assert logged_in.get("/tiles/link.pmtiles").status_code == 404


def test_symlinked_pack_out_of_the_tiles_dir_gives_no_map(make_ground, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from sd_ground.api.tiles import map_config

    g: Ground = make_ground()
    root = g.loaded.data_dir / "tiles"
    root.mkdir(parents=True)
    outside = tmp_path / "outside.pmtiles"
    shutil.copyfile(FIXTURE, outside)
    (root / "demo.pmtiles").symlink_to(outside)
    assert map_config(g) is None


def test_type_comes_from_the_requested_name(logged_in: TestClient, tiles: Path) -> None:
    (tiles / "notes.txt").write_text("x")
    (tiles / "odd.pmtiles").symlink_to(tiles / "notes.txt")
    res = logged_in.get("/tiles/odd.pmtiles")
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/vnd.pmtiles"


def test_login_required(panel: TestClient, tiles: Path) -> None:
    res = panel.get("/tiles/demo.pmtiles")
    assert res.status_code == 401
    assert res.json()["code"] == "UNAUTHENTICATED"


def test_configured_pack_name(make_ground, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from sd_ground.api.tiles import map_config

    g: Ground = make_ground(map={"pack": "sim-area", "zoom": 14})
    root = g.loaded.data_dir / "tiles"
    root.mkdir(parents=True)
    assert map_config(g) is None
    shutil.copyfile(FIXTURE, root / "sim-area.pmtiles")
    cfg = map_config(g)
    assert cfg is not None
    assert cfg.pack_url == "/tiles/sim-area.pmtiles"
    assert cfg.zoom == 14


def test_broken_pack_gives_no_map(make_ground) -> None:  # type: ignore[no-untyped-def]
    from sd_ground.api.tiles import map_config

    g: Ground = make_ground()
    root = g.loaded.data_dir / "tiles"
    root.mkdir(parents=True)
    (root / "demo.pmtiles").write_bytes(b"not a pmtiles file at all")
    assert map_config(g) is None


def _header(center: tuple[int, int], bounds: tuple[int, int, int, int]) -> bytes:
    head = bytearray(127)
    head[:8] = b"PMTiles\x03"
    struct.pack_into("<4i", head, 102, *bounds)
    struct.pack_into("<2i", head, 119, *center)
    return bytes(head)


def test_read_center(tmp_path: Path) -> None:
    pack = tmp_path / "p.pmtiles"
    bounds = (1490000000, -353800000, 1492000000, -353400000)
    pack.write_bytes(_header((1491652300, -353632610), bounds))
    c = read_center(pack)
    assert c is not None
    assert (round(c.lat, 6), round(c.lon, 6)) == (CMAC_LAT, CMAC_LON)
    # A centre outside the bounds (or unset) falls back to the middle of the bounds.
    pack.write_bytes(_header((0, 0), bounds))
    c = read_center(pack)
    assert c is not None
    assert (round(c.lat, 4), round(c.lon, 4)) == (-35.36, 149.1)
    assert read_center(tmp_path / "missing.pmtiles") is None
    (tmp_path / "short.pmtiles").write_bytes(b"PMTiles\x03")
    assert read_center(tmp_path / "short.pmtiles") is None


def _photo_pack(
    path: Path,
    *,
    tile_type: TileType = TileType.JPEG,
    metadata: dict[str, object] | None = None,
    compression: Compression = Compression.GZIP,
) -> None:
    """A tiny raster pack: one tile of stand-in bytes at zoom 16 and the given metadata."""
    with path.open("wb") as f:
        writer = Writer(f)
        writer.write_tile(zxy_to_tileid(16, 60_000, 39_000), b"tile")
        e7 = 10_000_000
        writer.finalize(
            {
                "tile_type": tile_type,
                "tile_compression": Compression.NONE,
                "internal_compression": compression,
                "min_zoom": 12,
                "max_zoom": 16,
                "min_lon_e7": round((CMAC_LON - 0.01) * e7),
                "min_lat_e7": round((CMAC_LAT - 0.01) * e7),
                "max_lon_e7": round((CMAC_LON + 0.01) * e7),
                "max_lat_e7": round((CMAC_LAT + 0.01) * e7),
                "center_zoom": 14,
                "center_lon_e7": round(CMAC_LON * e7),
                "center_lat_e7": round(CMAC_LAT * e7),
            },
            {"attribution": "Synthetic test imagery"} if metadata is None else metadata,
        )


def test_no_photo_pack_means_no_imagery(logged_in: TestClient, tiles: Path) -> None:
    assert logged_in.get("/api/v1/panel/config").json()["map"]["imagery"] is None


@pytest.mark.parametrize("compression", [Compression.GZIP, Compression.NONE])
def test_photo_pack_with_attribution(
    logged_in: TestClient, tiles: Path, compression: Compression
) -> None:
    _photo_pack(tiles / "imagery.pmtiles", compression=compression)
    body = logged_in.get("/api/v1/panel/config").json()["map"]["imagery"]
    assert body == {
        "pack_url": "/tiles/imagery.pmtiles",
        "maxzoom": 16,
        "attribution": "Synthetic test imagery",
    }
    assert logged_in.get("/tiles/imagery.pmtiles").status_code == 200


@pytest.mark.parametrize(
    ("tile_type", "metadata"),
    [
        (TileType.MVT, None),  # a vector pack is not a photo
        (TileType.JPEG, {}),  # the licence needs an attribution
        (TileType.JPEG, {"attribution": "   "}),
        (TileType.JPEG, {"attribution": "x" * 301}),
        (TileType.JPEG, {"attribution": 7}),
    ],
)
def test_photo_pack_without_what_it_needs_is_not_shown(
    logged_in: TestClient, tiles: Path, tile_type: TileType, metadata: dict[str, object] | None
) -> None:
    _photo_pack(tiles / "imagery.pmtiles", tile_type=tile_type, metadata=metadata)
    assert logged_in.get("/api/v1/panel/config").json()["map"]["imagery"] is None


def test_read_attribution_of_a_broken_pack(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pmtiles"
    bad.write_bytes(b"not a pack")
    assert read_attribution(bad) is None
    _photo_pack(bad)
    data = bytearray(bad.read_bytes())
    data[97] = 9  # unknown directory compression
    bad.write_bytes(bytes(data))
    assert read_attribution(bad) is None
