"""The safety manifest (``tools/safety_manifest.toml``) matches what is really loaded.

The manifest names the flight controller's safety parameter layers per profile, the
files whose content is pinned (the safety layers, the Lua script, the deny list and the
parameter allowlist) and their digests. A changed safety file therefore fails here until
the manifest is updated in the same pull request, where the reviewer sees both.

Checks (each finding is printed as ``rule: message``, exit code 1):

* ``pinned``: every pinned file exists and has the manifest's digest. Parameter files and
  TOML files are compared by content (names and values; comments and order do not count),
  everything else byte for byte. The files in ``REQUIRED_PINS`` must be pinned, so taking
  a pin out of the manifest does not let its file change unseen.
* ``sim`` / ``field``: the profile's layers, merged in order, meet the simulator's safety
  rules (``sd_sim.params.SAFETY_RULES`` / ``FIELD_RULES``), and the simulator profile has
  exactly the values the ground station reads back
  (``sd_safety.constants.FC_SAFETY_EXPECTED``).
* ``ruled``: every failsafe, fence, battery, arming, RTL and scripting line in a safety
  layer has a rule (``sd_sim.params.RULED_PREFIXES``): the simulator's layers in
  ``SAFETY_RULES``, the real-flight layers on top of them in ``FIELD_RULES``.
* ``loaded``: SITL loads the simulator profile's first layer as its safety layer for every
  vehicle and stages every pinned Lua script; the onboard computer's default and example
  config (``expected_param_files``) read back exactly the simulator profile's layers.
* ``shadow``: no sim-only layer sets a name a safety layer sets or any safety-relevant
  name (``sd_sim.params.RULED_PREFIXES`` such as ``FS_``, ``FENCE_``, ``BATT_``,
  ``ARMING_``, and ``SHADOW_PREFIXES``: the second battery, flight modes, EKF sources,
  GPS, motor and board safety, terrain, landing, ``RCn_OPTION`` ...), and
  ``tools/params_allowlist.toml`` lets no sim-only layer do so either. Names are
  compared in upper case.

``uv run tools/check_safety_manifest.py`` checks; ``--update`` writes the current digests
into the manifest and prints the pinned files whose digest changed (do that only for a
change you mean, and say so in the pull request).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "tools" / "safety_manifest.toml"
ONBOARD_EXAMPLE = "config/onboard.example.toml"
PARAMS_ALLOWLIST = "tools/params_allowlist.toml"
DIGEST_LINE = re.compile(r'^(sha256\s*=\s*)"[0-9a-f]*"', re.M)
# Files that must stay pinned whatever the manifest says: the rules, the deny list, the
# allowlist, the safety layers and the Lua script. Any other layer a profile names, and
# any other Lua script SITL stages, must be pinned too (``pinned_problems``,
# ``loaded_problems``).
REQUIRED_PINS = (
    "software/common/mav/src/sd_mav/denylist.py",
    "tools/params_allowlist.toml",
    "simulation/sd_sim/src/sd_sim/params.py",
    "software/common/safety/src/sd_safety/constants.py",
    "simulation/sitl/params/searchdrone-safety.parm",
    "simulation/sitl/params/searchdrone-safety.field.parm",
    "simulation/sitl/scripts/sd_gcs_loiter_cap.lua",
    # What enforces them: the gate's run-time allowlist, SITL's layer order, the check
    # that holds sim layers to the allowlist, and these checks themselves.
    "software/common/mav/src/sd_mav/allowlist.py",
    "simulation/sd_sim/src/sd_sim/sitl.py",
    "tools/params_check.py",
    "tools/check_forbidden_mavlink.py",
    "tools/check_safety_manifest.py",
)
# Safety-relevant parameter families on top of RULED_PREFIXES that no sim-only layer may
# set (the safety layers may, without a rule).
SHADOW_PREFIXES = (
    "BATT2_",
    "FLTMODE",
    "EK3_SRC",
    "EK3_ENABLE",
    "AHRS_EKF_TYPE",
    "AHRS_GPS_",
    "GPS_",
    "MOT_SAFE_",
    "BRD_SAFETY",
    "TERRAIN_",
    "LAND_",
    "PILOT_THR_",
    "RCMAP_",
)
RC_OPTION = re.compile(r"RC\d+_OPTION$")


@dataclass(frozen=True)
class Manifest:
    profiles: dict[str, list[str]]  # profile -> layer paths, in load order
    pinned: dict[str, str]  # path -> sha256


def load_manifest(text: str) -> Manifest:
    data = tomllib.loads(text)
    profiles = {name: list(p["layers"]) for name, p in data.get("profile", {}).items()}
    pinned = {str(p["path"]): str(p["sha256"]) for p in data.get("pinned", [])}
    return Manifest(profiles, pinned)


def parse_parm(text: str) -> dict[str, float]:
    """``NAME value`` / ``NAME,value`` lines; ``#`` starts a comment."""
    out: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            parts = [p for p in re.split(r"[\s,]+", line) if p]
            out[parts[0]] = float(parts[1]) if len(parts) > 1 else float("nan")
    return out


def digest(path: Path) -> str:
    """sha256 of the file's meaning: parameters and TOML by content, the rest by bytes."""
    if path.suffix == ".parm":
        data = json.dumps(sorted(parse_parm(path.read_text(encoding="utf-8")).items()))
        return hashlib.sha256(data.encode()).hexdigest()
    if path.suffix == ".toml":
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
        return hashlib.sha256(json.dumps(parsed, sort_keys=True).encode()).hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merged(root: Path, layers: Iterable[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for rel in layers:
        out.update(parse_parm((root / rel).read_text(encoding="utf-8")))
    return out


def pinned_problems(root: Path, manifest: Manifest) -> list[str]:
    out = []
    for rel, want in sorted(manifest.pinned.items()):
        path = root / rel
        if not path.is_file():
            out.append(f"pinned: {rel} saknas")
        elif digest(path) != want:
            out.append(
                f"pinned: {rel} har ändrats sedan manifestet skrevs; granska ändringen och kör "
                "'uv run tools/check_safety_manifest.py --update'"
            )
    for layers in manifest.profiles.values():
        for rel in layers:
            if rel not in manifest.pinned:
                out.append(f"pinned: säkerhetslagret {rel} är inte låst i manifestet")
    for rel in REQUIRED_PINS:
        if rel not in manifest.pinned:
            out.append(f"pinned: {rel} måste vara låst i manifestet (REQUIRED_PINS)")
    return out


def rule_problems(
    profile: str,
    values: dict[str, float],
    rules: Iterable[tuple[str, str, float]],
    expected: dict[str, float] | None = None,
) -> list[str]:
    from sd_sim.params import mismatches, rule_violations

    out = [f"{profile}: {line}" for line in rule_violations(values, rules)]
    if expected is not None:
        out += [f"{profile}: FC_SAFETY_EXPECTED {line}" for line in mismatches(expected, values)]
    return out


def ruled_problems(
    root: Path, layers: Iterable[str], rules: Iterable[tuple[str, str, float]], rules_name: str
):
    from sd_sim.params import RULED_PREFIXES

    ruled = {name for name, _, _ in rules}
    out = []
    for rel in layers:
        for name in parse_parm((root / rel).read_text(encoding="utf-8")):
            if name.startswith(RULED_PREFIXES) and name not in ruled:
                out.append(
                    f"ruled: {Path(rel).name} sätter {name}, som saknar regel i {rules_name}"
                )
    return out


def loaded_problems(root: Path, manifest: Manifest) -> list[str]:
    """What SITL and the onboard computer really load, against the manifest."""
    from sd_onboard.config import OnboardConfig
    from sd_sim.sitl import LUA_SCRIPTS, SIM_PARAM_FILES, SitlLayout

    out = []
    sim = manifest.profiles.get("sim", [])
    layout = SitlLayout(arducopter=root / "-", ardupilot_home=root / "-", repo=root)
    for vehicle in sorted(SIM_PARAM_FILES):
        try:
            layers = layout.project_layers(vehicle)
        except Exception as exc:
            out.append(f"loaded: SITL ({vehicle}) startar inte: {exc}")
            continue
        safety = [str(p.relative_to(root)) for p in layers[: len(sim)]]
        if safety != sim:
            out.append(f"loaded: SITL ({vehicle}) laddar {safety}, manifestet säger {sim}")
    for name in LUA_SCRIPTS:
        rel = f"simulation/sitl/scripts/{name}"
        if rel not in manifest.pinned:
            out.append(f"loaded: SITL kör {rel}, som inte är låst i manifestet")
    default = list(OnboardConfig().expected_param_files)
    if default != sim:
        out.append(f"loaded: sd-onboard läser tillbaka {default} som standard, manifestet {sim}")
    example = root / ONBOARD_EXAMPLE
    if example.is_file():
        files = tomllib.loads(example.read_text(encoding="utf-8")).get("expected_param_files")
        if files != sim:
            out.append(f"loaded: {ONBOARD_EXAMPLE} läser tillbaka {files}, manifestet {sim}")
    return out


def shadow_problems(root: Path, manifest: Manifest) -> list[str]:
    from sd_sim.params import RULED_PREFIXES
    from sd_sim.sitl import SIM_PARAM_FILES

    safety: set[str] = set()
    for layers in manifest.profiles.values():
        safety |= set(merged(root, layers))

    prefixes = RULED_PREFIXES + SHADOW_PREFIXES
    upper = {n.upper() for n in safety}

    def unsafe(names: Iterable[str]) -> list[str]:
        return sorted(
            n
            for n in names
            if n.upper() in upper or n.upper().startswith(prefixes) or RC_OPTION.match(n.upper())
        )

    out = []
    params = root / "simulation" / "sitl" / "params"
    for name in sorted(set(SIM_PARAM_FILES.values())):
        found = unsafe(parse_parm((params / name).read_text(encoding="utf-8")))
        out += [f"shadow: {name} sätter säkerhetsparametern {p}" for p in found]
    allow = tomllib.loads((root / PARAMS_ALLOWLIST).read_text(encoding="utf-8"))
    names = set(allow.get("allowed", []))
    for vehicle in allow.get("vehicles", {}).values():
        names |= set(vehicle.get("allowed", []))
    out += [f"shadow: {PARAMS_ALLOWLIST} tillåter säkerhetsparametern {p}" for p in unsafe(names)]
    return out


def check(root: Path = REPO, manifest_path: Path | None = None) -> list[str]:
    from sd_safety.constants import FC_SAFETY_EXPECTED
    from sd_sim.params import FIELD_RULES, SAFETY_RULES

    manifest = load_manifest(
        (manifest_path or root / MANIFEST.relative_to(REPO)).read_text("utf-8")
    )
    out = pinned_problems(root, manifest)
    for profile in ("sim", "field"):
        if profile not in manifest.profiles:
            out.append(f"{profile}: profilen saknas i manifestet")
    missing = [
        rel for layers in manifest.profiles.values() for rel in layers if not (root / rel).is_file()
    ]
    if out or missing:
        return out + [f"{rel}: saknas" for rel in sorted(set(missing))]
    sim, field = manifest.profiles["sim"], manifest.profiles["field"]
    expected = {k: float(v) for k, v in FC_SAFETY_EXPECTED.items()}
    out += rule_problems("sim", merged(root, sim), SAFETY_RULES, expected)
    out += rule_problems("field", merged(root, field), FIELD_RULES)
    out += ruled_problems(root, sim, SAFETY_RULES, "SAFETY_RULES")
    # The real-flight layers on top of the simulator's answer to FIELD_RULES.
    out += ruled_problems(root, [r for r in field if r not in sim], FIELD_RULES, "FIELD_RULES")
    out += loaded_problems(root, manifest)
    out += shadow_problems(root, manifest)
    return out


def update(manifest_path: Path, root: Path = REPO) -> list[str]:
    """Write each pinned file's current digest into the manifest, comments kept.

    Returns the pinned files whose digest changed."""
    text = manifest_path.read_text(encoding="utf-8")
    old = load_manifest(text).pinned
    blocks = re.split(r"(?=^\[\[pinned\]\])", text, flags=re.M)
    changed = []
    for i, block in enumerate(blocks):
        found = re.search(r'^path\s*=\s*"([^"]+)"', block, re.M)
        if block.startswith("[[pinned]]") and found:
            rel = found.group(1)
            sha = digest(root / rel)
            if old.get(rel) != sha:
                changed.append(rel)
            blocks[i] = DIGEST_LINE.sub(lambda m, s=sha: f'{m.group(1)}"{s}"', block, count=1)
    manifest_path.write_text("".join(blocks), encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--update", action="store_true", help="write the current digests")
    args = parser.parse_args(argv)
    # REPO and MANIFEST are passed, not left to the defaults, so a test can point them
    # at a copy.
    if args.update:
        changed = update(MANIFEST, REPO)
        for rel in changed:
            print(f"--update: ny digest för {rel}")
        if not changed:
            print("--update: inga låsta filer har ändrats")
    problems = check(REPO, MANIFEST)
    for line in problems:
        print(line)
    if problems:
        return 1
    print("check_safety_manifest: manifestet stämmer med det som laddas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
