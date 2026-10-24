"""A search scenario's persons as 3D models in the Gazebo world, where the oracle has them.

``PeopleScene(world)`` is a launcher scene (``sd_sim.launch.SearchScene``): every person
of a ``search_world.World`` (the SKxx scenarios the oracle detector and the scripted
operator go by) becomes a ``sd_sim.farm.people.Person``, a pinned Gazebo Fuel model
with the person label, in the person's pose and turned to its heading, in one solid
colour: the upper garment's (``sd_schemas.colors``), so the red jacket stays red. Each
model's box is centred on the person's place in the world, so the camera's picture and
the scenario's truth agree. Decoys are not placed (the trained detector is for persons).

The models are not in the repository: ``just sim-farm-setup`` fetches them into
``build/simfarm/assets/people/`` (credits and licences in
``simulation/farm/people/ATTRIBUTION.md``). Gazebo finds them through
``GZ_SIM_RESOURCE_PATH`` (``resource_path``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from sd_schemas.colors import COLOR_TABLE
from sd_sim import paths
from sd_sim.farm import people
from sd_sim.oracle_detector import lab_to_rgb
from sd_sim.search_world import Thing, World

__all__ = ["POSE_MODELS", "PeopleScene", "person", "resource_path"]

# The catalog model for each pose: Rescue Randy stands and is laid down for "lying".
POSE_MODELS = {
    "lying": "sd_person_rescue_randy",
    "sitting": "sd_person_sitting",
    "standing": "sd_person_standing",
}
# A lying model's head is along its -y axis; turned so the head points along the heading.
_LYING_TURN_DEG = -90.0


def _tint(name: str | None) -> tuple[float, float, float] | None:
    if name is None:
        return None
    rgb = lab_to_rgb(tuple(float(v) for v in COLOR_TABLE[name].lab))  # type: ignore[index, arg-type]
    return (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)


def _name(thing: Thing) -> str:
    return "person_" + re.sub(r"[^A-Za-z0-9_]", "_", thing.thing_id)[:56]


def person(thing: Thing) -> people.Person:
    """One person of the world as a model whose box is centred on the person's place."""
    yaw = thing.heading_deg + (_LYING_TURN_DEG if thing.pose == "lying" else 0.0)
    placed = people.Person(
        name=_name(thing),
        model=POSE_MODELS[thing.pose],
        pose=thing.pose,
        east_m=thing.east,
        north_m=thing.north,
        yaw_deg=yaw % 360.0,
        tint=_tint(thing.upper),
        is_target=thing.target,
    )
    corners = placed.corners()
    east = sum(c[0] for c in corners) / len(corners)
    north = sum(c[1] for c in corners) / len(corners)
    return replace(
        placed,
        east_m=round(placed.east_m - (east - thing.east), 3),
        north_m=round(placed.north_m - (north - thing.north), 3),
    )


@dataclass(frozen=True)
class PeopleScene:
    """The persons of one search world (``search_world.load(scenario, seed)``)."""

    world: World

    def objects(self) -> list[people.Person]:
        return [person(t) for t in self.world.people]


def resource_path(env: Mapping[str, str], repo: Path = paths.REPO) -> str:
    """``GZ_SIM_RESOURCE_PATH`` with the fetched person models first."""
    parts = [str(people.assets_dir(repo)), *env.get("GZ_SIM_RESOURCE_PATH", "").split(":")]
    return ":".join(p for p in dict.fromkeys(parts) if p)
