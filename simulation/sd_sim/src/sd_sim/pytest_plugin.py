"""pytest plugin for the simulator tests (loaded with ``-p sd_sim.pytest_plugin``).

* Options ``--physics``, ``--brain``, ``--shard-id`` and ``--num-shards``.
* Markers for the simulator tests (``sitl``, ``gz``, ``llm``, ``nightly``, ``budget_s``, ...).
* Sharding: every ``sitl`` test carries ``@pytest.mark.budget_s(<seconds>)`` (estimated
  wall time including stack start). The tests are spread over the shards by longest
  processing time first: sort by budget, largest first, and put each test in the shard
  with the smallest sum so far. The order is deterministic, so every pytest-xdist worker
  computes the same split.
* With ``SD_SIM__REQUIRE_SITL=1`` (CI) a skipped ``sitl`` test counts as failed, so the
  required check can never go green without flying.
* Fixtures ``sim_instance`` (one simulator instance per xdist worker) and
  ``sim_physics`` / ``sim_brain``.
* Every ``sitl`` and ``video`` test that runs writes ``runs/<run>/result.json`` with its
  outcome and the values it measured (``sd_sim.report.measure``), passed or not
  (``SD_SIM__RUNS_DIR`` moves ``runs/``, for the plugin's own tests). A value measured
  while a shared (module or session) fixture starts counts for the first test using it.
"""

from __future__ import annotations

import math
import os
import re
import time
import warnings
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sd_sim import paths
from sd_sim.report import Report, take_measured

MARKERS = (
    "sitl: needs ArduPilot SITL",
    "gz: needs Gazebo (local only)",
    "llm: needs the cloud model (the owner's key; never in CI)",
    "nightly: runs only in the scheduled nightly suite",
    "script(name): script for the scripted commander",
    "budget_s(seconds): estimated wall time of a SITL test, stack start included",
    "private_db: the instance gets its own Postgres container",
    "param_override(file): extra SITL parameter layer for one test",
    "lan: needs lan=1 (local only)",
    "panel: drives the real panel in a browser (just test-panel-sitl only)",
    "brain: flies with an AI commander (skipped with brain=none)",
    "video: needs MediaMTX (just video-up) and GStreamer; run with just test-video",
    "e2e: the end-to-end set of just e2e (the demo and the emergency buttons)",
    "search: the search chain set of a pull request (just ci-sitl mode=search)",
    "search_nightly: the nightly search campaign over SD_SEARCH_SEEDS (mode=search-nightly)",
    "e2e_search: the end-to-end search demo of just e2e-search",
)
MAX_INSTANCE = 7
# A shard's estimated work per worker must fit in this many seconds, and no single
# test may be longer than MAX_NODE_S (the job timeout leaves room for the stack).
SHARD_LIMIT_S = 25 * 60
MAX_NODE_S = 900
_WORKER = re.compile(r"^gw(\d+)$")
_SCENARIO = re.compile(r"(?:^|_)sc(\d+[a-z]?)(?=_|$)")
_RESULT_MARKERS = ("sitl", "video")
_STARTED = pytest.StashKey[tuple[str, float]]()
# Reports of tests that ran in this process, until their teardown is logged, and each
# test's logged outcomes per phase (setup, call, teardown) of its latest attempt.
_pending: dict[str, Report] = {}
_logged: dict[str, dict[str, str]] = {}


@dataclass(frozen=True)
class ShardPlan:
    shards: list[list[str]]
    sums: list[float]


def plan_shards(budgets: Iterable[tuple[str, float]], num_shards: int) -> ShardPlan:
    """Longest processing time first: deterministic for equal budgets (node id order)."""
    if num_shards < 1:
        raise ValueError("num_shards ska vara minst 1")
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    sums = [0.0] * num_shards
    for node, budget in sorted(budgets, key=lambda item: (-item[1], item[0])):
        idx = min(range(num_shards), key=lambda i: (sums[i], i))
        shards[idx].append(node)
        sums[idx] += budget
    return ShardPlan(shards, sums)


def budget_problems(
    budgets: Sequence[tuple[str, float]], num_shards: int, workers: int
) -> list[str]:
    """Why the suite does not fit in the CI job; empty when it does."""
    problems = [
        f"{node}: budget_s {budget:.0f} > {MAX_NODE_S}"
        for node, budget in budgets
        if budget > MAX_NODE_S
    ]
    plan = plan_shards(budgets, num_shards)
    if any(total / workers > SHARD_LIMIT_S for total in plan.sums):
        total = sum(budget for _, budget in budgets)
        need = math.ceil(total / (workers * SHARD_LIMIT_S))
        worst = max(plan.sums) / workers / 60
        problems.append(
            f"en skärva tar {worst:.1f} min per arbetare (gräns {SHARD_LIMIT_S // 60}); "
            f"använd minst {need} skärvor"
        )
    return problems


def budget_of(item: pytest.Item) -> float | None:
    marker = item.get_closest_marker("budget_s")
    if marker is None or not marker.args:
        return None
    return float(marker.args[0])


def require_sitl() -> bool:
    """CI sets ``SD_SIM__REQUIRE_SITL=1``: a SITL test may then never be skipped."""
    return os.environ.get("SD_SIM__REQUIRE_SITL") == "1"


def worker_instance(environ: dict[str, str] | None = None) -> int:
    """Simulator instance for this pytest-xdist worker (gw0 -> 0); 0 without xdist.
    ``SD_SIM__FIRST_INSTANCE`` moves them all up, for a second pytest run beside a first
    (``just e2e``: the demo keeps instance 0, the only one with video)."""
    env = os.environ if environ is None else environ
    name = env.get("PYTEST_XDIST_WORKER", "")
    match = _WORKER.match(name)
    first = env.get("SD_SIM__FIRST_INSTANCE", "")
    instance = (int(match.group(1)) if match else 0) + (int(first) if first.isdigit() else 0)
    if instance > MAX_INSTANCE:
        raise pytest.UsageError(f"högst {MAX_INSTANCE + 1} arbetare (simulatorinstanser 0-7)")
    return instance


def _int_option(config: pytest.Config, name: str) -> int:
    value = config.getoption(name)
    return 1 if value is None else int(value)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("searchdrone simulator")
    group.addoption("--physics", choices=("builtin", "gz"), default="builtin")
    group.addoption("--brain", choices=("scripted", "grammar", "cloud", "none"), default="scripted")
    group.addoption("--shard-id", type=int, default=1, help="1-based shard number")
    group.addoption("--num-shards", type=int, default=1)


def pytest_configure(config: pytest.Config) -> None:
    for line in MARKERS:
        config.addinivalue_line("markers", line)
    shard, shards = _int_option(config, "--shard-id"), _int_option(config, "--num-shards")
    if shards < 1 or not 1 <= shard <= shards:
        raise pytest.UsageError(f"--shard-id {shard} ligger utanför 1..{shards}")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    missing = [i.nodeid for i in items if i.get_closest_marker("sitl") and budget_of(i) is None]
    if missing:
        raise pytest.UsageError("SITL-tester utan @pytest.mark.budget_s: " + ", ".join(missing))
    needs_sitl = require_sitl() and any(i.get_closest_marker("sitl") for i in items)
    if needs_sitl and not paths.arducopter_bin().is_file():
        raise pytest.UsageError(
            f"SD_SIM__REQUIRE_SITL=1 men {paths.arducopter_bin()} finns inte: "
            "SITL-testerna skulle hoppas över."
        )
    shards = _int_option(config, "--num-shards")
    if shards == 1:
        return
    shard = _int_option(config, "--shard-id")
    plan = plan_shards(((i.nodeid, budget_of(i) or 0.0) for i in items), shards)
    keep = set(plan.shards[shard - 1])
    selected = [i for i in items if i.nodeid in keep]
    deselected = [i for i in items if i.nodeid not in keep]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def pytest_runtest_setup(item: pytest.Item) -> None:
    take_measured()  # nothing a previous test left behind
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    item.stash[_STARTED] = (stamp, time.monotonic())


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report: pytest.TestReport = yield
    if report.skipped and require_sitl() and item.get_closest_marker("sitl"):
        report.outcome = "failed"
        report.longrepr = f"SITL-testet hoppades över ({call.when}) men SD_SIM__REQUIRE_SITL=1"
    if report.when == "teardown" and any(item.get_closest_marker(m) for m in _RESULT_MARKERS):
        # Written once the reports are logged: pytest-rerunfailures marks an attempt it
        # repeats as "rerun" only after this hook, when it logs them.
        _pending[item.nodeid] = result_report(item)
    return report


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    phases = _logged.setdefault(report.nodeid, {})
    if report.when == "setup":
        phases.clear()  # a new attempt
    phases[report.when or ""] = report.outcome
    if report.when != "teardown":
        return
    _logged.pop(report.nodeid, None)
    rep = _pending.pop(report.nodeid, None)  # only in the process that ran the test
    if rep is not None and wants_result(phases):
        try:
            write_test_result(rep, phases)
        except OSError as exc:
            warnings.warn(f"result.json för {report.nodeid} skrevs inte: {exc}", stacklevel=1)


def scenario_id(test_name: str, module: str = "") -> str:
    """``test_sc02_stop_and_resume`` -> ``SC02``; else the module's scenario
    (``test_demo_flight`` in ``test_sc01.py`` -> ``SC01``); else the name stays."""
    m = _SCENARIO.search(test_name.split("[", 1)[0]) or _SCENARIO.search(module)
    return f"SC{m[1].upper()}" if m else test_name


def wants_result(phases: dict[str, str]) -> bool:
    """A test that ran to its end: a test skipped on purpose writes nothing, and an
    attempt pytest-rerunfailures repeats leaves the result to the repeat."""
    if "rerun" in phases.values():
        return False
    return phases.get("setup") != "skipped" and phases.get("call") != "skipped"


def result_report(item: pytest.Item) -> Report:
    """The test's report with what it measured; its run folder is made when written."""
    instance = worker_instance()
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", item.name).strip("-")[:80]
    runs = Path(os.environ.get("SD_SIM__RUNS_DIR") or paths.runs_dir())
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rep = Report(
        scenario_id(item.name, item.path.stem),
        runs / f"{stamp}-{label}-i{instance}",
        instance,
        str(item.config.getoption("--physics", "builtin")),
        str(item.config.getoption("--brain", "scripted")),
        variant=item.name,
    )
    if _STARTED in item.stash:
        rep.started_utc, rep._t0 = item.stash[_STARTED]
    rep.criteria = take_measured()
    return rep


def write_test_result(rep: Report, phases: dict[str, str]) -> Path:
    """``runs/<run>/result.json`` for one test: what it measured, then its outcome."""
    failed = [f"{when}: {outcome}" for when, outcome in phases.items() if outcome != "passed"]
    rep.check("T", "Testet godkänt", not failed, measured="; ".join(failed) or "passed")
    rep.run_dir.mkdir(parents=True, exist_ok=True)
    return rep.write()


@pytest.fixture
def sim_instance() -> int:
    return worker_instance()


@pytest.fixture
def sim_physics(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--physics"))


@pytest.fixture
def sim_brain(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--brain"))
