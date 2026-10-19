"""The sd_fpv7 drift gate (``sim_model_from_cad.py --check``): what counts as a change."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import sim_model_drift as drift

CAD = {"mass_kg": 1.606, "cog_m": [0.001, 0.0, 0.05], "name": "sd_fpv7"}


def folder(path: Path) -> Path:
    (path / "meshes").mkdir(parents=True)
    (path / "cad.json").write_text(json.dumps(CAD), encoding="utf-8")
    (path / "meshes" / "frame.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    return path


@pytest.fixture
def pair(tmp_path: Path) -> tuple[Path, Path]:
    fresh = folder(tmp_path / "fresh")
    committed = tmp_path / "committed"
    shutil.copytree(fresh, committed)
    (committed / "model.sdf.jinja").write_text("<sdf/>\n")  # hand-written, not compared
    return fresh, committed


def test_equal_folders_pass(pair: tuple[Path, Path]) -> None:
    assert drift.differences(*pair) == []


def test_json_within_tolerance_passes(pair: tuple[Path, Path]) -> None:
    fresh, committed = pair
    near = {**CAD, "mass_kg": CAD["mass_kg"] * (1 + drift.JSON_TOLERANCE / 10)}
    (committed / "cad.json").write_text(json.dumps(near), encoding="utf-8")
    assert drift.differences(fresh, committed) == []


def test_json_beyond_tolerance_fails(pair: tuple[Path, Path]) -> None:
    fresh, committed = pair
    off = {**CAD, "mass_kg": CAD["mass_kg"] * (1 + drift.JSON_TOLERANCE * 10)}
    (committed / "cad.json").write_text(json.dumps(off), encoding="utf-8")
    assert drift.differences(fresh, committed) == ["skiljer sig från CAD-modellen: cad.json"]


def test_changed_mesh_byte_fails(pair: tuple[Path, Path]) -> None:
    fresh, committed = pair
    (committed / "meshes" / "frame.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 2 0\nf 1 2 3\n")
    assert drift.differences(fresh, committed) == [
        "skiljer sig från CAD-modellen: meshes/frame.obj"
    ]


def test_missing_file_fails(pair: tuple[Path, Path]) -> None:
    fresh, committed = pair
    (committed / "meshes" / "frame.obj").unlink()
    assert drift.differences(fresh, committed) == ["saknas: meshes/frame.obj"]


def test_extra_mesh_fails(pair: tuple[Path, Path]) -> None:
    fresh, committed = pair
    (committed / "meshes" / "old.obj").write_text("v 0 0 0\n")
    assert drift.differences(fresh, committed) == ["finns inte i CAD-modellen: meshes/old.obj"]


def test_close_compares_structure() -> None:
    assert drift.close({"a": [1, 2.0]}, {"a": [1.0, 2]})
    assert not drift.close({"a": [1, 2]}, {"a": [1, 2, 3]})
    assert not drift.close({"a": 1}, {"b": 1})
    assert not drift.close("sd_fpv7", "sd_quad14")
    assert drift.close(True, True)
    assert not drift.close(True, 1)
    assert not drift.close(1, True)
