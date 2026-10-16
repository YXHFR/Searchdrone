"""Print every parameter name in an ArduPilot ``apm.pdef.xml``, one per line, sorted.

Used by sitl.Dockerfile, and to regenerate the committed list after a version change:

    python3 Tools/autotest/param_metadata/param_parse.py --vehicle ArduCopter --format xml
    python3 simulation/docker/known_params.py apm.pdef.xml --header <tag> \
        > simulation/sitl/params/known_params_<tag>.txt

Standard library only (the image's build stage has nothing else).
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET

HEADER = (
    "# Every parameter name of ArduCopter {tag}, one per line.\n"
    "# Generated with simulation/docker/known_params.py from apm.pdef.xml\n"
    "# (Tools/autotest/param_metadata/param_parse.py --vehicle ArduCopter).\n"
)


def names(path: str) -> list[str]:
    tree = ET.parse(path)  # noqa: S314 - ArduPilot's own generated file
    return sorted({p.get("name", "").split(":")[-1] for p in tree.iter("param") if p.get("name")})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="known_params.py")
    parser.add_argument("pdef")
    parser.add_argument("--header", metavar="TAG", help="write the comment header for TAG")
    args = parser.parse_args(argv)
    out = sys.stdout
    if args.header:
        out.write(HEADER.format(tag=args.header))
    out.writelines(name + "\n" for name in names(args.pdef))
    return 0


if __name__ == "__main__":
    sys.exit(main())
