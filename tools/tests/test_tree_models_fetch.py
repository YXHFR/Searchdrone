"""tools/terrain/fetch_tree_models.py without the network: cutting sprites out of a sheet
and drawing them."""

from __future__ import annotations

import importlib.util
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
models: Any = _load("fetch_tree_models")


def test_components_are_found_with_their_boxes() -> None:
    mask = np.zeros((10, 10), bool)
    mask[1:3, 1:4] = True
    mask[6:9, 7] = True
    assert sorted(models.components(mask)) == [(1, 1, 3, 4, 6), (6, 7, 9, 8, 3)]


def test_sprites_keep_green_parts_and_drop_bark() -> None:
    diff = np.zeros((256, 256, 3), np.float32)
    alpha = np.zeros((256, 256), np.float32)
    diff[16:112, 16:64] = (0.2, 0.5, 0.1)  # a twig
    alpha[16:112, 16:64] = 1.0
    diff[150:250, 150:250] = (0.4, 0.3, 0.2)  # bark: opaque but not green
    alpha[150:250, 150:250] = 1.0
    out = models.sprites(diff, alpha, 0.004)
    assert len(out) == 1
    assert out[0].shape[2] == 4 and out[0][..., 3].max() == 1.0
    with pytest.raises(models.FetchError):
        models.sprites(diff, alpha * 0, 0.004)


def test_stamp_draws_a_turned_sprite_at_its_base() -> None:
    sprite = np.zeros((40, 10, 4), np.float32)
    sprite[..., 1] = sprite[..., 3] = 1.0
    canvas = np.zeros((100, 100, 4), np.float32)
    models.stamp(canvas, sprite, 50, 50, 40, 90)  # pointing right from (50, 50)
    drawn = canvas[..., 3] > 0.5
    rows, cols = np.nonzero(drawn)
    assert cols.min() >= 48 and cols.max() >= 85 and abs(rows.mean() - 50) < 3


def test_flat_picture_has_a_flat_normal_map() -> None:
    rgba = np.ones((8, 8, 4), np.float32)
    assert np.allclose(models.normal_from_colour(rgba), [0.5, 0.5, 1.0])


def test_fit_fills_the_cell_with_the_trunk_at_the_bottom_centre() -> None:
    canvas = np.zeros((200, 200, 4), np.float32)
    canvas[40:190, 90:110] = 1.0
    out, occl = models.fit(canvas, np.ones((200, 200), np.float32), 100.0, 190.0)
    assert out.shape == (models.INNER, models.INNER, 4) and occl.shape == out.shape[:2]
    assert out[0, models.INNER // 2, 3] > 0.5 and out[-1, models.INNER // 2, 3] > 0.5
