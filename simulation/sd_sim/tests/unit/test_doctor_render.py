"""sim-doctor's render-node row: Gazebo's camera needs the GPU's render node."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sd_sim import doctor


def test_missing_node_is_a_warning(tmp_path: Path) -> None:
    row = doctor.check_render_node(tmp_path / "renderD128")
    assert row.status == doctor.WARN


def test_node_without_access_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    node = tmp_path / "renderD128"
    node.touch()
    monkeypatch.setattr(os, "access", lambda _path, _mode: False)
    row = doctor.check_render_node(node)
    assert row.status == doctor.FAIL
    assert "render" in row.detail


def test_node_with_access_is_ok(tmp_path: Path) -> None:
    node = tmp_path / "renderD128"
    node.touch()
    assert doctor.check_render_node(node).status == doctor.OK
