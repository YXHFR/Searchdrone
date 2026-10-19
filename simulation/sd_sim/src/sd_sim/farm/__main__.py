"""``python -m sd_sim.farm <command> [key=value ...]``.

run      the farm loop (``just sim-farm``)
status   what the farm is doing (``just sim-farm-status``)
setup    fetch the person models into build/ (``just sim-farm-setup``)
round    one round from ``spec=<folder>/spec.json`` (the loop runs this in its worktree)
"""

from __future__ import annotations

import sys
from pathlib import Path

from sd_sim import _args
from sd_sim.farm import loop, people, status
from sd_sim.farm import round as farm_round
from sd_sim.farm.gazebo import GazeboRunner

USAGE = "användning: python -m sd_sim.farm run|status|setup|round [nyckel=värde ...]"


def main(argv: list[str]) -> int:
    if not argv:
        _args.fail(USAGE)
    cmd, rest = argv[0], argv[1:]
    if cmd == "run":
        return loop.main(rest)
    if cmd == "status":
        _args.parse(rest, (), "sd_sim.farm status")
        return status.main()
    if cmd == "setup":
        _args.parse(rest, (), "sd_sim.farm setup")
        fetched = people.fetch()
        print(f"personmodeller: {len(people.CATALOG)} på plats ({len(fetched)} nyhämtade)")  # noqa: T201 - command-line output
        return 0
    if cmd == "round":
        values = _args.parse(rest, ("spec",), "sd_sim.farm round")
        if "spec" not in values:
            _args.fail("sd_sim.farm round: spec= behövs")
        return farm_round.execute(Path(values["spec"]), GazeboRunner())
    _args.fail(USAGE)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
