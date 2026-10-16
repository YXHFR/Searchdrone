"""Scenario test runner: ``python -m sd_sim.testrun [key=value ...]``.

Runs the simulator tests with pytest and the ``sd_sim.pytest_plugin`` plugin:

    pytest -p sd_sim.pytest_plugin -m "<selection>" [-n <workers> --dist load]
        --shard-id <shard> --num-shards <shards> --reruns 1 --only-rerun InfraError
        --physics <physics> --brain <brain> simulation/tests/ simulation/sd_sim/tests/sitl/

``mode=pr`` (default) selects ``sitl and not gz and not llm and not nightly`` (with
``physics=gz`` the Gazebo tests are included); ``mode=nightly`` selects
``sitl and nightly``. ``sc=`` and ``variant=`` narrow the run by test name. Only
test-rig failures (``InfraError``) are retried, once. Afterwards ``runs/summary.md``
lists every result written during the run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from sd_sim import _args, paths

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
    "dry",
)
TEST_PATHS = ("simulation/tests", "simulation/sd_sim/tests/sitl")
PYTEST_NO_TESTS = 5
# Settings that later parts of the stack read; accepted now so CI and local calls can
# pass them already.
FUTURE_KEYS = ("video", "link_profile", "private_db", "instance")


def selection(mode: str, physics: str) -> str:
    if mode == "nightly":
        return "sitl and nightly"
    if physics == "gz":
        return "sitl and not llm and not nightly"
    return "sitl and not gz and not llm and not nightly"


def _int(values: dict[str, str], key: str, default: int, low: int, high: int) -> int:
    raw = values.get(key, str(default))
    try:
        number = int(raw)
    except ValueError:
        _args.fail(f"sd_sim.testrun: {key}= ska vara ett heltal, inte {raw!r}")
    if not low <= number <= high:
        _args.fail(f"sd_sim.testrun: {key}= ska vara {low}-{high}")
    return number


def pytest_argv(values: dict[str, str]) -> list[str]:
    mode = values.get("mode", "pr")
    if mode not in ("pr", "nightly"):
        _args.fail("sd_sim.testrun: mode= ska vara pr eller nightly")
    physics = values.get("physics", "builtin")
    if physics not in ("builtin", "gz"):
        _args.fail("sd_sim.testrun: physics= ska vara builtin eller gz")
    shards = _int(values, "shards", 1, 1, 64)
    shard = _int(values, "shard", 1, 1, shards)
    workers = _int(values, "workers", 1, 1, 8)
    argv = [sys.executable, "-m", "pytest", "-p", "sd_sim.pytest_plugin"]
    argv += ["-m", selection(mode, physics)]
    names = [values[k] for k in ("sc", "variant") if values.get(k)]
    if names:
        argv += ["-k", " and ".join(names)]
    if workers > 1:
        argv += ["-n", str(workers), "--dist", "load"]
    argv += ["--shard-id", str(shard), "--num-shards", str(shards)]
    argv += ["--reruns", "1", "--only-rerun", "InfraError"]
    argv += ["--physics", physics, "--brain", values.get("brain", "scripted")]
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
    if "brain" in values:
        print("Varning: brain= ignoreras, befälhavaren är inte byggd än: inte byggt än (B1-10)")
    for key in FUTURE_KEYS:
        if key in values:
            print(f"Obs: {key}= används inte av de nuvarande testerna.")
    cmd = pytest_argv(values)
    print("+ " + " ".join(cmd[1:]), flush=True)
    start = time.time()
    code = subprocess.run(cmd, cwd=paths.REPO, env=os.environ.copy(), check=False).returncode  # noqa: S603
    if code == PYTEST_NO_TESTS:
        if values.get("mode") == "nightly":
            print("Inga nattester finns än; inget att köra.")
            code = 0
        else:
            print("Inga SITL-tester valdes. Kontrollera sc=, variant= och markeringarna.")
            code = 1
    runs = paths.runs_dir()
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "summary.md").write_text(
        summary_markdown(results_since(start, runs), code), encoding="utf-8"
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
