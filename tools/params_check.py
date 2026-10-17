"""Rules for the simulator's parameter layers.

* The sim-only layer (``sd_quad14-sim.parm``) may only set ``SIM_*`` and the names in
  ``tools/params_allowlist.toml``.
* A per-test override under ``simulation/sitl/params/tests/`` may only set names that the
  safety layer also sets, and must name its test in a comment line (``# test: <name>``).
* ArduPilot's default layers (``copter.parm``, ``gazebo-iris.parm``) must not set a
  safety-relevant parameter that the safety layer leaves unset; the safety layer then
  decides the value, never a default file.

``uv run tools/params_check.py`` runs the first two rules; with ``--ardupilot-home DIR``
also the third. Exit code 1 lists every finding.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PARAMS = REPO / "simulation" / "sitl" / "params"
SAFETY = PARAMS / "searchdrone-safety.parm"
SIM = PARAMS / "sd_quad14-sim.parm"
OVERRIDES = PARAMS / "tests"
ALLOWLIST = REPO / "tools" / "params_allowlist.toml"
SHADOW_PREFIXES = (
    "FS_",
    "FENCE_",
    "BATT_",
    "RTL_",
    "SRTL_",
    "ARMING_",
    "MAV_",
    "DISARM_",
    "LAND_",
    "WP_",
    "LOIT_",
    "ANGLE_MAX",
    "AVOID_",
    "FLTMODE",
    "SERIAL",
)
TEST_COMMENT = re.compile(r"^#\s*test:\s*\S+")


def names(text: str) -> list[str]:
    """Parameter names of a ``NAME value`` / ``NAME,value`` file, in order."""
    out = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            out.append(re.split(r"[\s,]+", line)[0])
    return out


def allowlist(path: Path = ALLOWLIST) -> set[str]:
    return set(tomllib.loads(path.read_text(encoding="utf-8"))["allowed"])


def sim_layer_problems(sim_text: str, allowed: set[str]) -> list[str]:
    return [
        f"{SIM.name}: {name} är varken SIM_* eller med i {ALLOWLIST.name}"
        for name in names(sim_text)
        if not name.startswith("SIM_") and name not in allowed
    ]


def override_problems(files: Iterable[Path], safety_names: set[str]) -> list[str]:
    out = []
    for path in sorted(files):
        text = path.read_text(encoding="utf-8")
        if not any(TEST_COMMENT.match(line) for line in text.splitlines()):
            out.append(f"{path.name}: saknar kommentarraden '# test: <testets namn>'")
        out += [
            f"{path.name}: {name} finns inte i {SAFETY.name}"
            for name in names(text)
            if name not in safety_names
        ]
    return out


def shadow_problems(default_layers: Iterable[Path], safety_names: set[str]) -> list[str]:
    out = []
    for path in default_layers:
        for name in names(path.read_text(encoding="utf-8")):
            if name.startswith(SHADOW_PREFIXES) and name not in safety_names:
                out.append(f"{path.name} sätter {name}, som {SAFETY.name} inte sätter")
    return out


def default_layers(ardupilot_home: Path) -> list[Path]:
    base = ardupilot_home / "Tools" / "autotest" / "default_params"
    return [base / "copter.parm", base / "gazebo-iris.parm"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rules for the simulator's parameter layers.")
    parser.add_argument("--ardupilot-home", type=Path)
    args = parser.parse_args(argv)
    safety_names = set(names(SAFETY.read_text(encoding="utf-8")))
    problems = sim_layer_problems(SIM.read_text(encoding="utf-8"), allowlist())
    problems += override_problems(OVERRIDES.glob("*.parm"), safety_names)
    if args.ardupilot_home is not None:
        layers = default_layers(args.ardupilot_home)
        missing = [str(p) for p in layers if not p.is_file()]
        if missing:
            problems.append("ArduPilots standardlager saknas: " + ", ".join(missing))
        else:
            problems += shadow_problems(layers, safety_names)
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
