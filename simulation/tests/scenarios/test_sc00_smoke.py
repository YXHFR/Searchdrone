"""SC00: smoke test. Start the stack, no flight.

Checks what the stack has today: SITL with the test rig, mavtap and linksim become
healthy, the flight controller stays disarmed in the same mode, its whole parameter
list matches the project's parameter files, secrets stay out of logs and command lines
and every simulator port listens on loopback only. Criteria that need the ground
station are recorded as not built yet and are turned on when it lands.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from sd_sim import config, netcheck, paths
from sd_sim.errors import InfraError
from sd_sim.harness import Harness
from sd_sim.launch import Stack
from sd_sim.report import Report
from sd_sim.sitl import CMAC_HOME
from sd_sim.util import new_run_dir

pytestmark = [
    pytest.mark.sitl,
    pytest.mark.slow,
    pytest.mark.budget_s(120),
    pytest.mark.skipif(not paths.arducopter_bin().is_file(), reason="ArduCopter SITL is not built"),
]

READY_LIMIT_S = 90.0
STABLE_S = 30.0
PARAM_TOLERANCE = 1e-4
MIN_PARAMS = 500  # ArduCopter 4.7 has well over a thousand
SECRET_ENV_KEY = re.compile(r"(PASSWORD|TOKEN|KEY)$")
GROUND = "inte byggt än (B1-05)"


def parse_parm(path: Path) -> dict[str, float]:
    """ArduPilot parameter file: ``NAME value`` or ``NAME,value``; ``#`` starts a comment."""
    out: dict[str, float] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = re.split(r"[\s,]+", line)
        out[parts[0]] = float(parts[1])
    return out


def expected_params(argv: list[str]) -> tuple[dict[str, float], list[str]]:
    """Parameters from the project's own layers on SITL's ``--defaults`` (later wins)."""
    layers = argv[argv.index("--defaults") + 1].split(",")
    own = paths.SIMULATION / "sitl" / "params"
    files = [Path(p) for p in layers if Path(p).parent == own]
    merged: dict[str, float] = {}
    for path in files:
        merged.update(parse_parm(path))
    return merged, [p.name for p in files]


def mismatches(expected: dict[str, float], actual: dict[str, float]) -> list[str]:
    out = []
    for name, value in sorted(expected.items()):
        if name not in actual:
            out.append(f"{name}: saknas i flygkontrollen (väntat {value:g})")
        elif abs(actual[name] - value) > PARAM_TOLERANCE:
            out.append(f"{name}: {actual[name]:g} (väntat {value:g})")
    return out


def secret_values(secrets_dir: Path) -> list[str]:
    values = [p.read_text(encoding="utf-8").strip() for p in secrets_dir.iterdir() if p.is_file()]
    env_file = paths.REPO / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and SECRET_ENV_KEY.search(key.strip()) and value.strip():
                values.append(value.strip().strip("'\""))
    return [v for v in values if len(v) >= 8]


def files_containing(root: Path, values: list[str]) -> list[str]:
    needles = [v.encode() for v in values]
    hits = []
    for path in root.rglob("*"):
        if path.is_file() and path.name != "result.json":
            data = path.read_bytes()
            if any(n in data for n in needles):
                hits.append(str(path.relative_to(root)))
    return hits


def argv_containing(pids: set[int], values: list[str]) -> list[int]:
    hits = []
    for pid in pids:
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            continue
        if any(v.encode() in cmdline for v in values):
            hits.append(pid)
    return hits


def test_sc00_smoke(sim_instance: int, sim_physics: str, sim_brain: str) -> None:
    cfg = config.load(
        {
            "physics": sim_physics,
            "gui": "false",
            "video": "off",
            "qgc": "false",
            "instance": str(sim_instance),
        }
    )
    cfg.home, cfg.personal_home = CMAC_HOME, False
    run_dir = new_run_dir(sim_instance, "SC00")
    rep = Report("SC00", run_dir, sim_instance, cfg.physics, sim_brain)
    stack = Stack(cfg, run_dir, observer=False)
    harness: Harness | None = None
    values: list[str] = []
    in_argv: list[int] = []
    modes_ok = False
    try:
        harness = Harness(stack.ports["harness_mavlink"])
        t0 = time.monotonic()
        stack.up()
        harness.wait(lambda s: s.heartbeat, 60, "HEARTBEAT från flygkontrollen")
        harness.wait(lambda s: s.ready, READY_LIMIT_S, "position från EKF och GPS")
        ready_s = round(time.monotonic() - t0, 1)
        rep.check(
            "SC00.1",
            "Stacken frisk (HEARTBEAT, EKF och GPS)",
            ready_s <= READY_LIMIT_S,
            measured=ready_s,
            limit=READY_LIMIT_S,
            unit="s",
            clock="host",
        )
        rep.not_built("SC00.2", "Fordonet anslutet i markserverns API", GROUND)

        mode, modes, armed = harness.state.mode, set(), False
        end = time.monotonic() + STABLE_S
        while time.monotonic() < end:
            harness.pump(0.5)
            modes.add(harness.state.mode)
            armed = armed or harness.state.armed
        rep.check(
            "SC00.3",
            f"Läget oförändrat och desarmerad i {STABLE_S:.0f} s",
            modes == {mode} and not armed,
            measured=sorted(modes),
            limit=[mode],
        )

        actual = harness.read_all_params()
        expected, files = expected_params(stack.sitl_argv())
        rep.param_mismatches = mismatches(expected, actual)
        rep.check(
            "SC00.4",
            "Hela parameterlistan lästes och stämmer med projektets filer",
            len(actual) >= MIN_PARAMS and not rep.param_mismatches,
            measured={
                "params": len(actual),
                "files": files,
                "mismatches": len(rep.param_mismatches),
            },
            limit={"min_params": MIN_PARAMS, "mismatches": 0},
        )
        rep.not_built("SC00.5", "Profil sim och ingen fjärråtkomst i markservern", GROUND)

        values = secret_values(stack.secrets_dir)
        modes_ok = all(p.stat().st_mode & 0o777 == 0o600 for p in stack.secrets_dir.iterdir())
        in_argv = argv_containing(stack.group.owned_pids(), values)
        bad = [
            str(item)
            for item in netcheck.violations(
                netcheck.parse_ss(netcheck.run_ss()),
                stack.ports,
                netcheck.load_allowlist(),
                owned_pids=stack.group.owned_pids(),
            )
        ]
        rep.check(
            "SC00.7", "Simulatorns portar lyssnar bara på loopback", not bad, measured=bad, limit=[]
        )
        rep.not_built("SC00.8", "Ingen localhost-pilot i markservern", GROUND)
        rep.not_built("SC00.9", "Markserverns databasroll har minsta privilegium", GROUND)
    except InfraError as exc:
        rep.infra_failure = stack.mask(str(exc))
        raise
    finally:
        if harness is not None:
            harness.close()
        stack.down()
        if rep.infra_failure is None and "SC00.7" in {c.id for c in rep.criteria}:
            in_logs = files_containing(run_dir, values)
            rep.check(
                "SC00.6",
                "Inga hemligheter i loggar eller kommandorader, filerna 0600",
                modes_ok and not in_logs and not in_argv,
                measured={"files": in_logs, "pids": in_argv, "modes_0600": modes_ok},
                limit={"files": [], "pids": [], "modes_0600": True},
            )
        rep.write()
    assert rep.passed, "\n".join(rep.failures())
