from itertools import pairwise

import numpy as np
import pytest

from sd_motion import DIRS, Params, Terrain, for_category
from sd_motion.params import load_all


def test_every_row_loads_and_aliases_resolve() -> None:
    table = load_all()
    assert set(table) == {"dementia", "despondent", "hiker", "gatherer"}
    assert for_category("hunter") == table["hiker"]
    assert for_category("berry_picker") == table["gatherer"]
    assert for_category("child_4_6") == table["despondent"]


@pytest.mark.parametrize("category", ["dementia", "despondent", "hiker", "gatherer"])
def test_mobility_curve_passes_its_points_and_falls(category: str) -> None:
    p = for_category(category)
    for hours, share in p.mobility:
        assert p.still_moving(hours) == pytest.approx(share)
    times = np.linspace(0, 48, 400)
    values = [p.still_moving(t) for t in times]
    assert all(b <= a + 1e-12 for a, b in pairwise(values))
    assert p.still_moving(48) < p.mobility[-1][1]


def test_bad_mobility_is_refused() -> None:
    with pytest.raises(ValueError, match="first at 0 h"):
        Params(1.0, 0.1, 0.1, 0.1, 1.0, ((1.0, 0.5), (2.0, 0.2)))
    with pytest.raises(ValueError, match="never rise"):
        Params(1.0, 0.1, 0.1, 0.1, 1.0, ((0.0, 0.5), (2.0, 0.7)))


def test_terrain_from_worldcover() -> None:
    classes = np.array([[10, 20, 80], [90, 50, 30]])
    track = np.array([[False, False, True], [False, False, False]])
    t = Terrain.from_classes(classes, track, 25.0)
    assert t.speed_factor[0, 0] == pytest.approx(0.45)
    assert t.speed_factor[0, 2] == 0.0  # a track mapped onto water stays water
    assert t.trap[0, 1] and t.trap[1, 0] and not t.trap[0, 0]
    assert t.wall[0, 2] == 1.0


def test_track_axis_and_the_way_to_it() -> None:
    shape = (9, 9)
    track = np.zeros(shape, bool)
    track[4, :] = True  # east-west
    t = Terrain.from_cover(np.zeros(shape, bool), track, np.zeros(shape, bool), 25.0)
    assert (t.track_dir[4] == 2).all()
    assert t.near_track_dir[2, 4] == DIRS.index((1, 0))  # south of it: go north
    assert t.near_track_dir[6, 4] == DIRS.index((-1, 0))
    assert t.near_track_dir[0, 4] == -1  # too far


def test_terrain_shapes_must_match() -> None:
    with pytest.raises(ValueError, match="does not match"):
        Terrain(np.ones((2, 2)), np.zeros((2, 3), np.int8), np.zeros((2, 2), np.int8),
                np.zeros((2, 2), bool), 25.0)  # fmt: skip
