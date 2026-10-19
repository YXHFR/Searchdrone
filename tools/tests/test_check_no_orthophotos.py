"""tools/check_no_orthophotos.py: names, picture-shaped arrays, GeoTIFFs, built worlds and
history. Every file here is synthetic and made at run time."""

from __future__ import annotations

import importlib.util
import io
import json
import struct
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "check_no_orthophotos", ROOT / "tools" / "check_no_orthophotos.py"
)
assert _spec is not None and _spec.loader is not None
chk: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chk)


def npy(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def tiff(tags: list[int]) -> bytes:
    """A little-endian TIFF header and one directory with these tags (no pixels)."""
    entries = b"".join(struct.pack("<HHII", t, 3, 1, 0) for t in sorted(tags))
    return b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", len(tags)) + entries + b"\0" * 4


@pytest.mark.parametrize(
    ("name", "data", "flagged"),
    [
        ("build/ortho/x/ortho.npy", None, True),
        ("a/ortho.toml", None, True),
        ("x/Orto_2025.TIF", None, True),
        ("x/tile.jp2", None, True),
        ("x/frame.bin", npy(np.zeros((300, 300, 3), np.uint8)), True),  # whatever it is called
        ("x/rgba.npy", npy(np.zeros((256, 512, 4), np.uint8)), True),
        ("x/chm_dm.npy", npy(np.zeros((300, 300), np.uint16)), False),  # tree heights
        ("x/small.npy", npy(np.zeros((32, 32, 3), np.uint8)), False),
        ("x/float.npy", npy(np.zeros((300, 300, 3), np.float32)), False),
        ("x/map.tif", tiff([256, 257, 33550, 33922]), True),
        ("x/plain.tif", tiff([256, 257]), False),
        (
            "runs/r/snapshots/metrics.json",
            json.dumps({"ground_texture": "orthophoto"}).encode(),
            True,
        ),
        ("runs/r/snapshots/metrics.json", json.dumps({"tree_source": "raster"}).encode(), False),
        ("runs/r/snapshots/ATTRIBUTION.txt", "Lantmäteriet's orthophoto (c)".encode(), True),
        ("simulation/sd_sim/src/sd_sim/snapshot.py", "Lantmäteriet's orthophoto".encode(), False),
    ],
)
def test_rules(name: str, data: bytes | None, flagged: bool) -> None:
    assert bool(chk.check(name, data)) is flagged


def test_folders_are_walked_and_missing_ones_are_fine(tmp_path: Path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / "a" / "b" / "readme.txt").write_text("hello")
    assert chk.main([str(tmp_path / "a"), str(tmp_path / "missing")]) == 0
    np.save(tmp_path / "a" / "b" / "x.npy", np.zeros((400, 400, 3), np.uint8))
    assert chk.main([str(tmp_path / "a")]) == 1


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_git_history_finds_a_photo_deleted_again(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "keep.txt").write_text("ok")
    _git(tmp_path, "add", "keep.txt")
    _git(tmp_path, "commit", "-qm", "one")
    assert chk.main(["--git", "--root", str(tmp_path)]) == 0
    np.save(tmp_path / "pic.npy", np.zeros((300, 300, 3), np.uint8))
    _git(tmp_path, "add", "pic.npy")
    _git(tmp_path, "commit", "-qm", "two")
    _git(tmp_path, "rm", "-q", "pic.npy")
    _git(tmp_path, "commit", "-qm", "three")
    assert chk.check_git(tmp_path) and "pic.npy" in chk.check_git(tmp_path)[0]


def test_this_repository_is_clean() -> None:
    assert chk.check_git(ROOT) == []
