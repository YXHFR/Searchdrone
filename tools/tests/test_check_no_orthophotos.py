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


def tiff(tags: list[int], pad: int = 0) -> bytes:
    """A little-endian TIFF header and one directory with these tags (no pixels), the
    directory ``pad`` bytes after the header."""
    entries = b"".join(struct.pack("<HHII", t, 3, 1, 0) for t in sorted(tags))
    ifd = struct.pack("<H", len(tags)) + entries + b"\0" * 4
    return b"II*\x00" + struct.pack("<I", 8 + pad) + b"\0" * pad + ifd


def bigtiff(tags: list[int]) -> bytes:
    """A big-endian BigTIFF header and one directory with these tags."""
    entries = b"".join(struct.pack(">HHQQ", t, 3, 1, 0) for t in sorted(tags))
    return b"MM\x00+" + struct.pack(">HHQ", 8, 0, 16) + struct.pack(">Q", len(tags)) + entries


def png_with(text: bytes) -> bytes:
    chunk = struct.pack(">I", len(text)) + b"tEXt" + text + b"\0\0\0\0"
    return chk.PNG_SIGNATURE + chunk


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
        ("x/far.tif", tiff([256, 257, 33922], pad=200_000), True),  # directory at the end
        ("x/big.tif", bigtiff([256, 257, 34735]), True),
        ("x/bigplain.tif", bigtiff([256, 257]), False),
        ("x/terrain_0_0.png", png_with(b"Source\0Searchdrone ground_texture=orthophoto"), True),
        ("x/terrain_0_0.png", png_with(b"Source\0Searchdrone"), False),
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


def test_findings_go_to_the_step_summary_in_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    (tmp_path / "ortho.toml").write_text("")
    assert chk.main([str(tmp_path / "ortho.toml")]) == 1
    assert "ortho.toml" in summary.read_text()


def test_symbolic_links_are_findings_not_followed(tmp_path: Path) -> None:
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    np.save(hidden / "x.npy", np.zeros((300, 300, 3), np.uint8))
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "file").symlink_to(hidden / "x.npy")
    (runs / "folder").symlink_to(hidden)
    found = chk.check_paths([runs])
    assert len(found) == 2 and all("symbolic link" in f for f in found)
    assert chk.main([str(runs / "folder")]) == 1


def test_a_tiff_directory_far_into_the_file_is_read(tmp_path: Path) -> None:
    (tmp_path / "map.tif").write_bytes(tiff([256, 257, 33550], pad=chk.HEAD * 2))
    assert chk.main([str(tmp_path)]) == 1


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


def test_git_history_reads_every_blob_whatever_its_name(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    (tmp_path / "blob.dat").write_bytes(npy(np.zeros((300, 300, 3), np.uint8)))
    (tmp_path / "geo.bin").write_bytes(bigtiff([256, 34735]))
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "one")
    found = " ".join(chk.check_git(tmp_path))
    assert "blob.dat" in found and "geo.bin" in found


def test_this_repository_is_clean() -> None:
    assert chk.check_git(ROOT) == []
