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
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass

import pytest

from sd_sim import paths

MARKERS = (
    "sitl: needs ArduPilot SITL",
    "gz: needs Gazebo (local only)",
    "llm: needs the local language model (local only, never in CI)",
    "nightly: runs only in the scheduled nightly suite",
    "script(name): script for the scripted commander",
    "budget_s(seconds): estimated wall time of a SITL test, stack start included",
    "private_db: the instance gets its own Postgres container",
    "param_override(file): extra SITL parameter layer for one test",
    "lan: needs lan=1 (local only)",
    "panel: drives the real panel in a browser (just test-panel-sitl only)",
    "brain: flies with an AI commander (skipped with brain=none)",
    "video: needs MediaMTX (just video-up) and GStreamer; run with just test-video",
    "e2e: the end-to-end suite (the demo and the emergency buttons); run with just e2e",
)
MAX_INSTANCE = 7
# A shard's estimated work per worker must fit in this many seconds, and no single
# test may be longer than MAX_NODE_S (the job timeout leaves room for the stack).
SHARD_LIMIT_S = 25 * 60
MAX_NODE_S = 900
_WORKER = re.compile(r"^gw(\d+)$")


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
    """Simulator instance for this pytest-xdist worker (gw0 -> 0); 0 without xdist."""
    env = os.environ if environ is None else environ
    name = env.get("PYTEST_XDIST_WORKER", "")
    match = _WORKER.match(name)
    instance = int(match.group(1)) if match else 0
    if instance > MAX_INSTANCE:
        raise pytest.UsageError(f"högst {MAX_INSTANCE + 1} arbetare (simulatorinstanser 0-7)")
    return instance


def _int_option(config: pytest.Config, name: str) -> int:
    value = config.getoption(name)
    return 1 if value is None else int(value)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("searchdrone simulator")
    group.addoption("--physics", choices=("builtin", "gz"), default="builtin")
    group.addoption(
        "--brain", choices=("scripted", "grammar", "llm_local", "none"), default="scripted"
    )
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


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report: pytest.TestReport = yield
    if report.skipped and require_sitl() and item.get_closest_marker("sitl"):
        report.outcome = "failed"
        report.longrepr = f"SITL-testet hoppades över ({call.when}) men SD_SIM__REQUIRE_SITL=1"
    return report


@pytest.fixture
def sim_instance() -> int:
    return worker_instance()


@pytest.fixture
def sim_physics(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--physics"))


@pytest.fixture
def sim_brain(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--brain"))
