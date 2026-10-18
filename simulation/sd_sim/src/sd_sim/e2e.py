"""The end-to-end suite: ``python -m sd_sim.e2e [runs=N] [offline=1]`` (``just e2e``).

Each run is two pytest passes over the tests marked ``e2e`` (headless: builtin physics,
the test picture as video, the ``ScriptedBackend``):

1. the demo SC01 (``test_sc01_demo_end_to_end``) on instance 0, the whole stack as
   separate programs as ``just demo`` starts it (instance 0, so its video reaches
   MediaMTX when it runs);
2. the emergency buttons N1-N7 (SC02, SC03, SC15, SC08) on four instances in parallel.

``runs=N`` repeats both N times; every run must pass ("N av N"). ``offline=1`` first
shows that this process cannot reach the internet (the CI job blocks the stack's
traffic, ``tools/offline/isolate.sh``) and fails when it can. Only test-rig failures
(``InfraError``) are retried, once. The result is in ``runs/e2e-summary.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

from sd_sim import _args, paths

ALLOWED_KEYS = ("runs", "offline", "workers", "dry")
DEMO = "test_sc01_demo_end_to_end"
PROBE_URL = "https://example.com"
PROBE_TIMEOUT_S = 5.0
MAX_RUNS = 50


def pytest_argv(*, demo: bool, workers: int) -> list[str]:
    argv = [sys.executable, "-m", "pytest", "-p", "sd_sim.pytest_plugin", "-m", "sitl and e2e"]
    argv += ["-k", DEMO if demo else f"not {DEMO}"]
    if not demo and workers > 1:
        argv += ["-n", str(workers), "--dist", "load"]
    argv += ["--reruns", "1", "--only-rerun", "InfraError"]
    argv += ["--physics", "builtin", "--brain", "scripted", "simulation/tests"]
    return argv


def internet_reachable(url: str = PROBE_URL, timeout: float = PROBE_TIMEOUT_S) -> bool:
    """Can this process open ``url`` (directly or through a proxy in the environment)?"""
    try:
        with urllib.request.urlopen(url, timeout=timeout):  # noqa: S310 - fixed https URL
            return True
    except urllib.error.HTTPError:
        return True  # an answer is an answer
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _int(values: dict[str, str], key: str, default: int, high: int) -> int:
    raw = values.get(key, str(default))
    if not raw.isdigit() or not 1 <= int(raw) <= high:
        _args.fail(f"just e2e: {key}= ska vara ett heltal 1-{high}, inte {raw!r}")
    return int(raw)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    values = _args.parse(argv, ALLOWED_KEYS, "just e2e")
    runs = _int(values, "runs", 1, MAX_RUNS)
    workers = _int(values, "workers", 4, 7)
    offline = values.get("offline", "0")
    if offline not in ("0", "1"):
        _args.fail("just e2e: offline= ska vara 0 eller 1")
    if values.get("dry") == "1":
        print(json.dumps({"args": argv}))
        return 0
    if offline == "1":
        if internet_reachable():
            print(f"offline=1: {PROBE_URL} svarar; nätet är inte spärrat.", file=sys.stderr)
            return 1
        print(f"offline=1: {PROBE_URL} svarar inte (som väntat).")
    results: list[dict[str, object]] = []
    for n in range(1, runs + 1):
        start = time.monotonic()
        codes = []
        for demo in (True, False):
            cmd = pytest_argv(demo=demo, workers=workers)
            print("+ " + " ".join(cmd[1:]), flush=True)
            env = os.environ.copy()
            codes.append(subprocess.run(cmd, cwd=paths.REPO, env=env, check=False).returncode)  # noqa: S603
        passed = all(c == 0 for c in codes)
        results.append(
            {"run": n, "passed": passed, "codes": codes, "s": round(time.monotonic() - start)}
        )
        print(f"Körning {n} av {runs}: {'godkänd' if passed else 'INTE godkänd'}", flush=True)
    good = sum(1 for r in results if r["passed"])
    runs_dir = paths.runs_dir()
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary = {"runs": runs, "passed": good, "offline": offline == "1", "results": results}
    (runs_dir / "e2e-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"just e2e: {good} av {runs} godkända.")
    return 0 if good == runs else 1


if __name__ == "__main__":
    sys.exit(main())
