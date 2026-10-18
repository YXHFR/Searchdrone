"""Rules for the simulator's parameter layers.

* A model's sim-only layer (``sd_quad14-sim.parm``, ``sd_fpv7-sim.parm``) may only set
  ``SIM_*``, the names in ``tools/params_allowlist.toml`` and that model's own
  ``[vehicles.<model>]`` names there. A layer that sets its own pack size sets
  ``BATT_CAPACITY``, ``BATT_LOW_MAH`` and ``BATT_CRT_MAH`` together, with the reserves at
  the safety layer's shares of the capacity.
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
SIM_LAYERS = {"sd_quad14": SIM, "sd_fpv7": PARAMS / "sd_fpv7-sim.parm"}
BATTERY = ("BATT_CAPACITY", "BATT_LOW_MAH", "BATT_CRT_MAH")
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


def values(text: str) -> dict[str, float]:
    out = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            name, value = re.split(r"[\s,]+", line)[:2]
            out[name] = float(value)
    return out


def allowlist(path: Path = ALLOWLIST, vehicle: str = "sd_quad14") -> set[str]:
    """The names every sim-only layer may set, plus ``vehicle``'s own."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    own = data.get("vehicles", {}).get(vehicle, {}).get("allowed", [])
    return set(data["allowed"]) | set(own)


def sim_layer_problems(sim_text: str, allowed: set[str], layer: str = SIM.name) -> list[str]:
    return [
        f"{layer}: {name} är varken SIM_* eller med i {ALLOWLIST.name}"
        for name in names(sim_text)
        if not name.startswith("SIM_") and name not in allowed
    ]


def battery_problems(sim_text: str, safety_text: str, layer: str) -> list[str]:
    """A layer's own pack keeps the safety layer's reserves as shares of the capacity."""
    own, safety = values(sim_text), values(safety_text)
    have = [name for name in BATTERY if name in own]
    if not have:
        return []
    if len(have) != len(BATTERY):
        return [f"{layer}: sätter {', '.join(have)} men inte alla av {', '.join(BATTERY)}"]
    out = []
    for name in BATTERY[1:]:
        share = own[name] / own["BATT_CAPACITY"]
        wanted = safety[name] / safety["BATT_CAPACITY"]
        if abs(share - wanted) > 1e-9:
            out.append(
                f"{layer}: {name} är {share:.1%} av BATT_CAPACITY, {SAFETY.name} har {wanted:.1%}"
            )
    return out


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
    safety_text = SAFETY.read_text(encoding="utf-8")
    safety_names = set(names(safety_text))
    problems = []
    for vehicle, path in SIM_LAYERS.items():
        text = path.read_text(encoding="utf-8")
        problems += sim_layer_problems(text, allowlist(vehicle=vehicle), path.name)
        problems += battery_problems(text, safety_text, path.name)
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
