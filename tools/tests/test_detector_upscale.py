"""The training tiles (``tools/train/tiles.py``) and the drone's tiles
(``sd_detect.tiling``) are enlarged the same way: the training environment has no
``sd_detect``, so ``tiles.py`` writes the resize out, and this keeps the two equal."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from sd_detect.tiling import MAX_UPSCALE, Tile, check_upscale, enlarge, tile_to_picture

TRAIN = Path(__file__).resolve().parents[1] / "train"


def _tiles() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_tiles", TRAIN / "tiles.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclass looks itself up there
    spec.loader.exec_module(module)
    return module


tiles = _tiles()


@pytest.mark.parametrize("upscale", [1, 2, 3, 4])
def test_training_and_the_drone_enlarge_a_tile_to_the_same_pixels(upscale: int) -> None:
    rgb = np.random.default_rng(upscale).integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
    assert np.array_equal(tiles.enlarge(rgb, upscale), enlarge(rgb, upscale))


def test_training_and_the_drone_agree_on_which_factors_are_allowed() -> None:
    assert tiles.MAX_UPSCALE == MAX_UPSCALE
    for tile in (64, 128, 640):
        for upscale in range(0, MAX_UPSCALE + 2):
            try:
                want = check_upscale(tile, upscale)
            except ValueError:
                with pytest.raises(ValueError):
                    tiles.check_upscale(tile, upscale)
            else:
                assert tiles.check_upscale(tile, upscale) == want


def test_a_training_box_maps_back_to_the_picture_as_the_drones_does() -> None:
    person = (1000.0, 700.0, 1014.0, 728.0)  # 14 x 28 px
    for (x0, y0, x1, y1), kept in tiles.cut(4608, 2592, [person], 640, upscale=2):
        for box in kept:
            back = tile_to_picture(np.array(box), Tile(x0, y0, x1, y1), 2)
            assert back == pytest.approx(person)
            assert box[2] - box[0] == pytest.approx(28.0)  # the model sees it twice as wide
