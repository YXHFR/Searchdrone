"""Scenario test runner: ``python -m sd_sim.testrun [key=value ...]``.

Runs the simulator tests with pytest and the ``sd_sim.pytest_plugin`` plugin:

    pytest -p sd_sim.pytest_plugin -m "<selection>" [-n <workers> --dist load]
        --shard-id <shard> --num-shards <shards> --reruns 1 --only-rerun InfraError
        --physics <physics> --brain <brain> simulation/tests/ simulation/sd_sim/tests/sitl/

``mode=pr`` (default) selects ``sitl and not gz and not llm and not nightly and not e2e``
and none of the search sets (with ``physics=gz`` the Gazebo tests are included);
``mode=e2e`` selects the end-to-end set of ``just e2e`` (``sitl and e2e``: the demo and
the emergency buttons), which CI runs in a job of its own; ``mode=nightly`` selects
``sitl and nightly``; ``mode=panel`` selects the panel's browser test (``sitl and
panel``, started by ``just test-panel-sitl``). The search chain has three sets, each in
a CI job of its own and in no other mode: ``mode=search`` (``sitl and search``, the pull
request set), ``mode=search-nightly`` (``sitl and search_nightly``, the nightly oracle
campaign) and ``mode=e2e-search`` (``sitl and e2e_search``, the search demo of ``just
e2e-search``). A search test that also carries ``e2e`` runs in ``mode=e2e`` only.
``brain=`` picks the AI commander the scenarios fly
with: ``scripted`` (default, the script per scenario), ``grammar`` (the fixed command
mode), ``cloud`` (the cloud model with the owner's key, the command mode without it; adds the
``llm`` tests) or
``none`` (no commander: only the tests that need none). ``sc=`` and ``variant=`` narrow
the run by test name. ``seeds=`` (``1-50`` or ``1,2,3``, at most 200 seeds) reaches the
search tests as ``SD_SEARCH_SEEDS``, which they read when they are collected;
``sk06_seeds=`` (same form) gives the nightly SK06 comparison its own seeds as
``SD_SK06_SEEDS``. Only
test-rig failures (``InfraError``) are retried, once.
``detector=model`` (with ``model=<manifest .json>`` or ``SD_DETECTOR_MODEL``) flies
``mode=e2e-search`` with the owner's trained person detector on Gazebo's stills instead
of the oracle (``sd_sim.detector_choice``); the manifest and the model's SHA-256 are
checked before pytest starts, and a bad one stops the run (exit 2).
Afterwards ``runs/summary.md`` lists every result written during the run.

In ``mode=e2e`` the demo (``test_demo_flight``) gets simulator instance 0, the only one
with video, in a pytest run of its own; the other tests run beside it on instances 1 and
up. ``runs=N`` repeats the whole set N times and passes only if every round passes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from sd_sim import _args, detector_choice, paths, search_world

ALLOWED_KEYS = (
    "mode",
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
    "runs",
    "seeds",
    "sk06_seeds",
    "detector",
    "model",
    "dry",
)
TEST_PATHS = ("simulation/tests", "simulation/sd_sim/tests/sitl")
PYTEST_NO_TESTS = 5
# Settings that later parts of the stack read; accepted now so CI and local calls can
# pass them already.
FUTURE_KEYS = ("video", "link_profile", "private_db", "instance")
BRAINS = ("scripted", "grammar", "cloud", "none")
MODES = ("pr", "e2e", "nightly", "panel", "search", "search-nightly", "e2e-search")
# The search chain's sets: mode -> marker. Each runs in its own mode only.
SEARCH_SETS = {"search": "search", "search-nightly": "search_nightly", "e2e-search": "e2e_search"}
SEEDS_ENV = "SD_SEARCH_SEEDS"
SK06_SEEDS_ENV = "SD_SK06_SEEDS"


DEMO_TEST = "test_demo_flight"
FIRST_INSTANCE_ENV = "SD_SIM__FIRST_INSTANCE"


def selection(mode: str, physics: str, brain: str = "scripted") -> str:
    # The panel's browser test needs Playwright beside the stack (`just test-panel-sitl`),
    # so it is never part of the pull-request or the nightly SITL suite.
    if mode == "panel":
        return "sitl and panel"
    gz = "" if physics == "gz" else " and not gz"
    no_search = " and not search_nightly and not e2e_search"
    if mode in SEARCH_SETS:
        base = f"sitl and {SEARCH_SETS[mode]}{gz} and not llm and not nightly and not e2e"
    elif mode == "nightly":
        base = f"sitl and nightly and not search{no_search}"
    elif mode == "e2e":
        # SK08's first seed carries both e2e and search: it runs here, not in mode=search.
        base = f"sitl and e2e{gz} and not llm and not nightly{no_search}"
    else:
        base = f"sitl{gz} and not llm and not nightly and not e2e and not search{no_search}"
    if brain == "cloud":
        base = base.replace(" and not llm", "")
    elif brain == "none":
        base += " and not brain"
    return base + " and not panel"


def _int(values: dict[str, str], key: str, default: int, low: int, high: int) -> int:
    raw = values.get(key, str(default))
    try:
        number = int(raw)
    except ValueError:
        _args.fail(f"sd_sim.testrun: {key}= ska vara ett heltal, inte {raw!r}")
    if not low <= number <= high:
        _args.fail(f"sd_sim.testrun: {key}= ska vara {low}-{high}")
    return number


def seeds(values: dict[str, str]) -> dict[str, str]:
    """``seeds=`` and ``sk06_seeds=`` as the environment the search tests read, checked as
    they read it (``sd_sim.search_world.seeds``)."""
    env: dict[str, str] = {}
    for key, name in (("seeds", SEEDS_ENV), ("sk06_seeds", SK06_SEEDS_ENV)):
        raw = values.get(key)
        if raw is None:
            continue
        try:
            search_world.seeds(raw)
        except search_world.WorldError as e:
            _args.fail(f"sd_sim.testrun: {key}= {e}")
        env[name] = raw
    return env


def detector_env(values: dict[str, str]) -> dict[str, str]:
    """``detector=`` and ``model=`` as the environment the search demo reads (only the
    key decides, never an inherited ``SD_SIM__DETECTOR``). A bad model fails here,
    before anything starts."""
    try:
        choice = detector_choice.choose(values.get("detector", "oracle"), values.get("model"))
    except detector_choice.DetectorError as e:
        _args.fail(f"sd_sim.testrun: {e}")
    if not choice.model:
        return {}
    if values.get("mode") != "e2e-search":
        _args.fail("sd_sim.testrun: detector=model fungerar bara med mode=e2e-search")
    return {detector_choice.MODE_ENV: choice.mode, **choice.env()}


def pytest_argv(values: dict[str, str], name_filter: str = "") -> list[str]:
    mode = values.get("mode", "pr")
    if mode not in MODES:
        _args.fail(f"sd_sim.testrun: mode= ska vara en av {', '.join(MODES)}")
    physics = values.get("physics", "builtin")
    if physics not in ("builtin", "gz"):
        _args.fail("sd_sim.testrun: physics= ska vara builtin eller gz")
    shards = _int(values, "shards", 1, 1, 64)
    shard = _int(values, "shard", 1, 1, shards)
    workers = _int(values, "workers", 1, 1, 8)
    brain = values.get("brain", "scripted")
    if brain not in BRAINS:
        _args.fail(f"sd_sim.testrun: brain= ska vara en av {', '.join(BRAINS)}")
    argv = [sys.executable, "-m", "pytest", "-p", "sd_sim.pytest_plugin"]
    argv += ["-m", selection(mode, physics, brain)]
    names = [values[k] for k in ("sc", "variant") if values.get(k)]
    if name_filter:
        names.append(name_filter)
    if names:
        argv += ["-k", " and ".join(names)]
    if workers > 1:
        argv += ["-n", str(workers), "--dist", "load"]
    argv += ["--shard-id", str(shard), "--num-shards", str(shards)]
    argv += ["--reruns", "1", "--only-rerun", "InfraError"]
    argv += ["--physics", physics, "--brain", brain]
    argv += list(TEST_PATHS)
    return argv


def results_since(start: float, runs: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(runs.glob("*/result.json")):
        if path.stat().st_mtime < start:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            data = {"passed": False, "error": "result.json är inte giltig JSON"}
        data["_run"] = path.parent.name
        out.append(data)
    return out


def summary_markdown(results: list[dict[str, Any]], exit_code: int) -> str:
    lines = [
        "# SITL-tester",
        "",
        f"pytest avslutades med kod {exit_code}.",
        "",
        "| Scenario | Variant | Godkänt | Ej byggda kriterier | Tid (s) | Körning |",
        "|---|---|---|---|---|---|",
    ]
    for res in results:
        name = res.get("scenario") or res.get("name") or "?"
        ok = "ja" if res.get("passed") else "nej"
        lines.append(
            f"| {name} | {res.get('variant', '-')} | {ok} | {res.get('not_built', '-')} "
            f"| {res.get('duration_s', '-')} "
            f"| {res['_run']} |"
        )
    if not results:
        lines.append("| (inga resultat) | | | | | |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    values = _args.parse(argv, ALLOWED_KEYS, "sd_sim.testrun")
    if values.get("qgc", "").strip().lower() not in ("", "0", "false"):
        _args.fail("sd_sim.testrun: qgc=true är inte tillåtet i tester (en styrväg förbi grinden).")
    if values.get("dry") == "1":
        print(json.dumps({"args": argv}))
        return 0
    for key in FUTURE_KEYS:
        if key in values:
            print(f"Obs: {key}= används inte av de nuvarande testerna.")
    rounds = _int(values, "runs", 1, 1, 99)
    start = time.time()
    passed = 0
    code = 0
    for i in range(rounds):
        if rounds > 1:
            print(f"=== omgång {i + 1} av {rounds}", flush=True)
        got = run_round(values)
        passed += got == 0
        code = code or got
    if rounds > 1:
        print(f"{passed} av {rounds} omgångar gröna.", flush=True)
    runs = paths.runs_dir()
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "summary.md").write_text(
        summary_markdown(results_since(start, runs), code), encoding="utf-8"
    )
    return code


def commands(values: dict[str, str]) -> list[tuple[list[str], dict[str, str]]]:
    """The pytest runs of one round, each with its extra environment. ``mode=e2e`` with
    more than one worker: the demo alone on instance 0, the rest on instances 1 and up."""
    env = seeds(values) | detector_env(values)
    if values.get("mode") != "e2e":
        return [(pytest_argv(values), env)]
    workers = _int(values, "workers", 1, 1, 8)
    if workers == 1:
        return [(pytest_argv(values), env)]
    demo = pytest_argv({**values, "workers": "1"}, DEMO_TEST)
    rest_workers = str(min(workers - 1, 7))
    rest = pytest_argv({**values, "workers": rest_workers}, f"not {DEMO_TEST}")
    return [(demo, env), (rest, {**env, FIRST_INSTANCE_ENV: "1"})]


def run_round(values: dict[str, str]) -> int:
    cmds = commands(values)
    for cmd, _extra in cmds:
        print("+ " + " ".join(cmd[1:]), flush=True)
    # Only detector= picks the detector, never a value inherited from the shell.
    base = {k: v for k, v in os.environ.items() if k != detector_choice.MODE_ENV}
    if len(cmds) == 1:
        cmd, extra = cmds[0]
        env = {**base, **extra}
        codes = [subprocess.run(cmd, cwd=paths.REPO, env=env, check=False).returncode]  # noqa: S603
    else:
        procs = [
            subprocess.Popen(cmd, cwd=paths.REPO, env={**base, **extra})  # noqa: S603
            for cmd, extra in cmds
        ]
        codes = [p.wait() for p in procs]
    code = 0
    for got in codes:
        if got == PYTEST_NO_TESTS:
            if values.get("mode") == "nightly":
                print("Inga nattester finns än; inget att köra.")
                got = 0
            else:
                print("Inga SITL-tester valdes. Kontrollera sc=, variant= och markeringarna.")
                got = 1
        code = code or got
    return code


if __name__ == "__main__":
    sys.exit(main())
