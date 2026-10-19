"""tools/terrain/fetch_trees.py without the network: the TIFF and zip readers, tree tops
and species."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "terrain" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("fetch_site")
trees: Any = _load("fetch_trees")


def _info(width: int, height: int, rows: list[bytes]) -> Any:
    """A TiffInfo like Skogsstyrelsen's: one 16-bit row per strip, 1 m cells."""
    counts = np.array([len(r) for r in rows], dtype=np.uint64)
    offsets = (100 + np.concatenate([[0], np.cumsum(counts)[:-1]])).astype(np.uint64)
    tags = {
        258: (3, 1, struct.pack("<H", 16)),
        259: (3, 1, struct.pack("<H", 1)),
        277: (3, 1, struct.pack("<H", 1)),
        278: (16, 1, struct.pack("<Q", 1)),
        33550: (12, 3, struct.pack("<3d", 1.0, 1.0, 0.0)),
    }
    return trees.TiffInfo(width, height, offsets, counts, (620000.0, 6700000.0), 1.0, tags)


def test_bigtiff_rows_round_trip() -> None:
    rows = [(np.arange(4, dtype="<i2") + r).tobytes() for r in range(3)]
    info = _info(4, 10, rows)
    out = trees.write_bigtiff(info, 5, rows)
    back = trees.parse_bigtiff(out)
    assert (back.width, back.height, back.cell) == (4, 3, 1.0)
    # The written file starts 5 rows further south.
    assert back.origin == (620000.0, 6700000.0 - 5.0)
    for i, row in enumerate(rows):
        a = int(back.offsets[i])
        assert out[a : a + int(back.counts[i])] == row
    assert struct.unpack("<H", back.tags[259][2][:2])[0] == 1


def test_bigtiff_refuses_other_layouts() -> None:
    rows = [b"\0" * 8]
    info = _info(4, 1, rows)
    info.tags[278] = (16, 1, struct.pack("<Q", 2))
    with pytest.raises(trees.FetchError, match="rad per strip"):
        trees.parse_bigtiff(trees.write_bigtiff(info, 0, rows))
    with pytest.raises(trees.FetchError, match="BigTIFF"):
        trees.parse_bigtiff(b"MM\x00\x2a" + b"\0" * 32)


def _local(name: str, size: int, method: int = 9, flag: int = 0) -> bytes:
    raw = name.encode()
    return (
        struct.pack("<IHHHHHIIIHH", 0x04034B50, 45, flag, method, 0, 0, 0, size, size, len(raw), 0)
        + raw
    )


def test_zip_entries_find_the_tif_after_the_world_file() -> None:
    head = _local("a.tfw", 3) + b"xyz" + _local("a.tif", 0xFFFFFFFF)
    entries = trees.zip_entries(head + b"data")
    assert entries == [("a.tfw", 35, 3), ("a.tif", 35 + 3 + 35, -1)]
    with pytest.raises(trees.FetchError, match="komprimering"):
        trees.zip_entries(_local("b.tif", 5, method=12))


def _bump(n: int, r: float, c: float, h: float, w: float) -> np.ndarray:
    rr, cc = np.mgrid[0:n, 0:n]
    return h * np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * w * w))


def test_find_tops_one_per_crown_at_its_height() -> None:
    n = 60
    chm = np.maximum.reduce(
        [_bump(n, 15, 15, 22.0, 2.5), _bump(n, 40, 42, 12.0, 1.8), _bump(n, 45, 10, 2.5, 1.0)]
    )
    tops = trees.find_tops(chm, 1.0, n / 2)
    assert len(tops) == 2  # the 2.5 m bush is below MIN_TREE_M
    (e1, n1, h1), (e2, n2, h2) = sorted(tops, key=lambda t: -t[2])
    assert (e1, n1) == (15 - 30 + 0.5, 30 - 15 - 0.5) and h1 == pytest.approx(22.0)
    assert (e2, n2) == (42 - 30 + 0.5, 30 - 40 - 0.5) and h2 == pytest.approx(12.0)


def test_close_tops_of_one_crown_are_one_tree() -> None:
    chm = np.zeros((20, 20))
    chm[10, 10] = chm[10, 11] = 15.0  # a flat top two cells wide
    assert len(trees.find_tops(chm, 1.0, 10.0)) == 1


def test_crown_radius_grows_with_height_within_limits() -> None:
    r = trees.crown_radius(np.array([2.0, 10.0, 25.0, 60.0]))
    assert r[0] == 1.5 and r[-1] == 4.0 and r[1] < r[2]


def test_species_follow_the_land_cover_class() -> None:
    assert trees.species_for(111, "a") == "pine"
    assert trees.species_for(122, "a") == "spruce"
    assert trees.species_for(116, "a") == "deciduous"
    assert trees.species_for(3, "a") == "deciduous"  # open land: park and garden trees
    mixed = {trees.species_for(113, f"k{i}") for i in range(200)}
    assert mixed == {"spruce", "pine"}
    assert trees.species_for(113, "k7") == trees.species_for(113, "k7")


def test_county_is_read_from_the_site(tmp_path: Path) -> None:
    assert trees.county_of("hellasgarden", ROOT) == 1
    folder = tmp_path / "simulation" / "gazebo" / "sites" / "x"
    folder.mkdir(parents=True)
    (folder / "site.toml").write_text("[trees]\ncounty = 40\n")
    with pytest.raises(trees.FetchError, match="county"):
        trees.county_of("x", tmp_path)


def test_arguments() -> None:
    with pytest.raises(trees.FetchError, match="site="):
        trees.parse_args([])
    with pytest.raises(trees.FetchError, match="okänd"):
        trees.parse_args(["site=a", "x=1"])
    assert trees.parse_args(["site=abc"]) == {"site": "abc"}


# --- the site's rows streamed out of the county archive ----------------------------------


def _tiff_tags_first(width: int, rows: list[bytes]) -> bytes:
    """A BigTIFF laid out like Skogsstyrelsen's: tags first, then one strip per row."""
    height = len(rows)
    entries = {
        256: (16, 1, struct.pack("<Q", width)),
        257: (16, 1, struct.pack("<Q", height)),
        258: (3, 1, struct.pack("<H", 16)),
        259: (3, 1, struct.pack("<H", 1)),
        277: (3, 1, struct.pack("<H", 1)),
        278: (16, 1, struct.pack("<Q", 1)),
        33550: (12, 3, struct.pack("<3d", 1.0, 1.0, 0.0)),
        33922: (12, 6, struct.pack("<6d", 0, 0, 0, 620000.0, 6700000.0, 0)),
        273: (16, height, b"\0" * 8 * height),
        279: (16, height, b"\0" * 8 * height),
    }
    extra_at = 16 + 8 + 20 * len(entries) + 8
    data_at = extra_at + sum(len(raw) for _, _, raw in entries.values() if len(raw) > 8)
    counts = np.array([len(r) for r in rows], dtype="<u8")
    offsets = (data_at + np.concatenate([[0], np.cumsum(counts)[:-1]])).astype("<u8")
    entries[273] = (16, height, offsets.tobytes())
    entries[279] = (16, height, counts.tobytes())
    head = bytearray()
    extra = bytearray()
    for tag in sorted(entries):
        typ, count, raw = entries[tag]
        value = raw.ljust(8, b"\0") if len(raw) <= 8 else struct.pack("<Q", extra_at + len(extra))
        if len(raw) > 8:
            extra += raw
        head += struct.pack("<HHQ", tag, typ, count) + value
    out = b"II\x2b\x00\x08\x00\x00\x00" + struct.pack("<Q", 16)
    out += struct.pack("<Q", len(entries)) + bytes(head) + struct.pack("<Q", 0) + bytes(extra)
    assert len(out) == data_at
    return out + b"".join(rows)


@pytest.fixture
def county_zip(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    """A county archive served in 1 kB pieces: a world file, then the Deflate64 TIFF."""
    import inflate64  # type: ignore[import-not-found]

    width = 50
    rng = np.random.default_rng(7)
    rows = [rng.integers(0, 400, width, dtype="<u2").tobytes() for _ in range(400)]
    deflater = inflate64.Deflater()
    packed = deflater.deflate(_tiff_tags_first(width, rows)) + deflater.flush()
    archive = _local("a.tfw", 3) + b"xyz" + _local("a.tif", 0xFFFFFFFF) + packed

    def stream(url: str, size: int, start: int = 0) -> Any:
        assert size == len(archive)
        for at in range(start, size, 1024):
            yield archive[at : at + 1024]

    monkeypatch.setattr(trees, "TIFF_HEAD", 8192)
    monkeypatch.setattr(trees, "file_size", lambda url: len(archive))
    monkeypatch.setattr(trees, "_request", lambda url, headers={}, length=None: archive[:65536])
    monkeypatch.setattr(trees, "stream", stream)
    return rows


def test_fetch_chm_keeps_only_the_sites_rows(
    county_zip: list[bytes], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Rows 300-349 and columns 10-40 of a grid whose north-west corner is 620000, 6700000.
    bounds = (620010.0, 6699650.0, 620040.0, 6699700.0)
    monkeypatch.setattr(trees, "site_bounds_sweref", lambda site, extra: bounds)
    out, meta = trees.fetch_chm(None, 1, tmp_path)
    assert meta["chm_rows"] == "300-349" and meta["chm_entry"] == "a.tif"
    data = out.read_bytes()
    back = trees.parse_bigtiff(data)
    assert (back.width, back.height) == (50, 50)
    assert back.origin == (620000.0, 6699700.0)
    for i in range(50):
        a = int(back.offsets[i])
        assert data[a : a + int(back.counts[i])] == county_zip[300 + i]


def test_fetch_chm_refuses_a_site_outside_the_file(
    county_zip: list[bytes], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    del county_zip
    bounds = (620010.0, 6699500.0, 620040.0, 6699700.0)  # 500 rows down, the file has 400
    monkeypatch.setattr(trees, "site_bounds_sweref", lambda site, extra: bounds)
    with pytest.raises(trees.FetchError, match="utanför"):
        trees.fetch_chm(None, 1, tmp_path)


# --- land cover colours ------------------------------------------------------------------


def _legend_json(entries: list[tuple[str, str]]) -> bytes:
    colormap = {"entries": [{"color": c, "quantity": q} for c, q in entries]}
    rule = {"symbolizers": [{"Raster": {"colormap": colormap}}]}
    return json.dumps({"Legend": [{"rules": [rule]}]}).encode()


def test_legend_maps_colours_to_classes() -> None:
    legend = trees.parse_legend(_legend_json([("#00FF00", "111"), ("#0a0b0c", "3.0")]))
    assert legend == {(0, 255, 0): 111, (10, 11, 12): 3}
    with pytest.raises(trees.FetchError, match="skogsklasserna"):
        trees.parse_legend(_legend_json([("#000000", "3")]))


def test_colours_become_classes_and_too_many_unknown_is_an_error() -> None:
    legend = {(0, 255, 0): 111, (10, 11, 12): 3}
    img = np.zeros((10, 10, 3), np.uint8)
    img[:, :5] = (0, 255, 0)
    img[:, 5:] = (10, 11, 12)
    img[0, 0] = (1, 2, 3)  # one unknown cell of 100: under 2 %
    classes = trees.classes_from_colours(img, legend)
    assert classes[0, 0] == 0 and classes[5, 2] == 111 and classes[5, 7] == 3
    assert classes.dtype == np.int32 and classes.shape == (10, 10)
    img[:3, :] = (1, 2, 3)
    with pytest.raises(trees.FetchError, match="okänd färg"):
        trees.classes_from_colours(img, legend)
