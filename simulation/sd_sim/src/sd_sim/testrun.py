"""Scenario test runner: ``python -m sd_sim.testrun [key=value ...]``.

Only the command-line interface exists so far; running scenarios is not built yet.
"""

from __future__ import annotations

import json
import sys

from sd_sim import _args

ALLOWED_KEYS = (
    "physics",
    "video",
    "brain",
    "instance",
    "sc",
    "variant",
    "link_profile",
    "qgc",
    "private_db",
    "shard",
    "shards",
    "workers",
    "dry",
)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    values = _args.parse(argv, ALLOWED_KEYS, "sd_sim.testrun")
    if values.get("qgc", "false").lower() in ("1", "true"):
        _args.fail("sd_sim.testrun: qgc=true är inte tillåtet i tester (en styrväg förbi grinden).")
    if values.get("dry") == "1":
        print(json.dumps({"args": argv}))
        return 0
    return _args.not_built("B0-08")


if __name__ == "__main__":
    sys.exit(main())
