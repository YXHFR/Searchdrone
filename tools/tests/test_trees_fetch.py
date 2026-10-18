"""tools/terrain/fetch_trees.py without the network: the TIFF and zip readers, tree tops
and species."""

from __future__ import annotations

import importlib.util
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
