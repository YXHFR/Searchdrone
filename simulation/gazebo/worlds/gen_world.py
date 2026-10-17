"""Render the Gazebo world for one simulator instance (deterministic, seed 42).

    uv run --package sd-sim python simulation/gazebo/worlds/gen_world.py [instance]

Writes build/sim/worlds/sd_lake_forest_i<N>.sdf, the GeoJSON layer next to it and the
instance's vehicle model under build/sim/models/. The launcher does the same before it
starts Gazebo, so running this by hand is only needed to inspect the files.
"""

from __future__ import annotations

import sys

from sd_sim import world


def main(argv: list[str]) -> int:
    instance = int(argv[0]) if argv else 0
    if not 0 <= instance <= 7:
        sys.stderr.write("instansen ska vara 0-7\n")
        return 2
    path = world.render_world(world.WORLD_NAME, instance)
    sys.stdout.write(f"{path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
