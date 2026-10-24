"""The search worlds' persons as Gazebo models at the places the oracle has them."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from sd_sim import config, launch, objects, search_world, world
from sd_sim.errors import InfraError
from sd_sim.farm import people
from sd_sim.people_scene import POSE_MODELS, PeopleScene, person, resource_path


def _centre(p: people.Person) -> tuple[float, float]:
    corners = p.corners()
    return (
        sum(c[0] for c in corners) / len(corners),
        sum(c[1] for c in corners) / len(corners),
    )


@pytest.mark.parametrize("scenario", ["SK01", "SK04", "SK06", "SK09"])
@pytest.mark.parametrize("seed", [1, 2, 7])
def test_every_person_stands_where_the_oracle_has_it(scenario: str, seed: int) -> None:
    w = search_world.load(scenario, seed)
    placed = PeopleScene(w).objects()
    assert len(placed) == len(w.people) > 0
    for thing, model in zip(w.people, placed, strict=True):
        east, north = _centre(model)
        assert math.hypot(east - thing.east, north - thing.north) < 0.01, thing.thing_id
        assert model.label == people.PERSON_LABEL
        assert model.model == POSE_MODELS[thing.pose] and model.pose == thing.pose
        assert model.is_target == thing.target
        assert model.truth["role"] == ("target" if thing.target else "person")
        world.checked_fragment(model)  # local model:// files and the label only
    objects.check(placed, world.GROUND_SIZE_M / 2)


def test_sk03_has_nobody() -> None:
    assert PeopleScene(search_world.load("SK03", 1)).objects() == []


def test_a_lying_person_lies_along_its_heading() -> None:
    w = search_world.load("SK01", 1)
    thing = next(t for t in w.people if t.pose == "lying")
    model = person(thing)
    h = math.radians(thing.heading_deg)
    along = (math.sin(h), math.cos(h))  # compass heading as east, north
    across = (along[1], -along[0])

    def span(axis: tuple[float, float]) -> float:
        dots = [c[0] * axis[0] + c[1] * axis[1] for c in model.corners()]
        return max(dots) - min(dots)

    assert span(along) > 1.7 > span(across)
    assert max(c[2] for c in model.corners()) < 0.5  # on the ground, not standing


def test_the_upper_garment_colours_the_model() -> None:
    w = search_world.load("SK01", 1)
    target = person(next(t for t in w.people if t.target))
    green = person(next(t for t in w.people if t.upper == "green"))
    assert target.tint is not None and target.tint[0] > 0.5 > target.tint[1]  # red
    assert green.tint is not None and green.tint[1] > green.tint[0]


def test_gazebo_finds_the_fetched_models_first(tmp_path: Path) -> None:
    got = resource_path({"GZ_SIM_RESOURCE_PATH": "/a:/b"}, tmp_path)
    assert got.split(":") == [str(people.assets_dir(tmp_path)), "/a", "/b"]
    assert resource_path({}, tmp_path) == str(people.assets_dir(tmp_path))


def test_the_stack_asks_for_the_models_before_it_starts(tmp_path: Path) -> None:
    cfg = config.SimConfig(physics="gz", world=world.WORLD_NAME, vehicle=world.MODEL_NAME)
    scene = PeopleScene(search_world.load("SK01", 1))
    st = launch.Stack(cfg, tmp_path / "run", repo=tmp_path, scene=scene)
    st.placed = list(scene.objects())
    assert st.places_people()
    with pytest.raises(InfraError, match="just sim-farm-setup"):
        st.check_people_fetched()
    plain = launch.Stack(cfg, tmp_path / "run2", repo=tmp_path)
    plain.check_people_fetched()  # nothing placed, nothing to fetch
