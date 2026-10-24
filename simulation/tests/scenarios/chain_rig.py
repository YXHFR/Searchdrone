"""The search chain end to end on ArduCopter SITL: the rig for the SK scenarios.

Everything but the camera, the detector, the image model and the person at the panel is
the product's own:

* the onboard side runs the executor and the perception task (``PerceptionLoop``) as
  ``sd-onboard`` wires them (target detections to the executor's keep-out cylinders, the
  executor's stills to ``capture``); the stills are ``SyntheticStillSource``'s pictures
  of the scenario's world from the flight controller's position, and the detector is
  ``OracleDetector`` (``det_oracle_v0``);
* ``sd-search`` runs as its CLI builds it (fusion, ``ColorScorer``, the probability map,
  the cascade) with ``OracleMatcher`` (``vlm_oracle_v0``) as the image model;
* ``sd-brain`` runs with the ``ScriptedBackend`` and the scenario's search script
  (``simulation/scenarios/scripts/SKxx.yaml``), whose fixture is the scenario file;
* ``ScriptedOperator`` approves the search profile and works the candidate queue through
  the panel API (the test rig's pilot token), the only one who confirms or rejects.

``search_run`` flies one scenario and seed: the operator names the charter, approves the
AI's proposal and "clear around the drone", asks for the start; the AI searches; the
operator confirms the missing person (or rejects everyone in SK03) and asks the drone
home. It checks the scenario invariants (``Scenario.check_invariants``) and the search
invariants (E1, E2, E8 in ``sd_sim.search_invariants``), and writes
``search_metrics.json`` and ``search_report.md`` in the run's folder.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import statistics
import sys
import time
import zlib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flow_rig  # pyright: ignore[reportMissingImports]
from scenario_rig import H, Scenario  # pyright: ignore[reportMissingImports]
from test_mission_flow_sitl import SitlOnboard  # pyright: ignore[reportMissingImports]
from test_onboard_sitl import Sim  # pyright: ignore[reportMissingImports]

from sd_geo import PinholeCamera
from sd_onboard.perception.pipeline import Perception
from sd_safety.targets import Plane, distance_m
from sd_schemas.probmap import ProbabilityGrid
from sd_search.app import SearchService
from sd_search.client import GroundClient
from sd_search.mapper import MapService
from sd_search.match.cascade import Cascade
from sd_search.probmap.terrain import NoTerrain, SimTerrainSource, TerrainSource
from sd_search.profile import GroundProfileSource
from sd_sim import world as sim_world
from sd_sim.eval_search import RunMetrics, percentile, write_run
from sd_sim.operator import ScriptedOperator
from sd_sim.oracle_detector import OracleDetector, OraclePerception, SyntheticStillSource
from sd_sim.report import measure
from sd_sim.search_invariants import (
    CandidateWatch,
    SeenKeepout,
    keepout_distances,
    keepout_never_entered,
    no_confirm_without_human,
)
from sd_sim.search_world import World
from sd_sim.vlm_oracle import OracleMatcher, Truth

__all__ = ["ChainOnboard", "SearchRun", "search_run", "search_service"]

SIM_ENV = {"SD_SIM": "1"}
PERIOD_S = 0.2  # how often perception asks for a still (the source gives one a second)
TRUTH_M = 12.0  # a card or a detection this near a thing is that thing
LANDED_LIMIT_S = 480.0  # SmartRTL retraces the flown path at 5 m/s
WATCH_S = 0.25


class ChainOnboard(SitlOnboard):
    """The onboard executor plus the perception task on the world's synthetic stills,
    wired as ``sd-onboard`` wires them."""

    def __init__(self, sim: Sim, url: str, token: str, world: World, keep_dir: Path | None) -> None:
        super().__init__(sim, url, token)
        cam = sim_world.STILLS
        camera = PinholeCamera(int(cam["width"]), int(cam["height"]), float(cam["hfov_deg"]))

        def telemetry() -> Any:
            return self.vehicle.telemetry

        stills = SyntheticStillSource(world, telemetry, camera, env=SIM_ENV)
        self.detector = OracleDetector(stills, seed=world.seed, env=SIM_ENV)
        self.positions: dict[str, tuple[float, float]] = {}  # detection id -> where
        self.perception = OraclePerception(
            self.detector,
            telemetry,
            self._send,
            self.clock,
            camera,
            period_s=PERIOD_S,
            keep_dir=keep_dir,
            env=SIM_ENV,
            on_targets=self.ex.on_detections,
        )
        self.ex.camera = self.perception.capture

    def _send(self, out: Perception) -> None:
        for d in out.report.detections:
            self.positions[str(d.detection_id)] = (d.lat, d.lon)
        self.link.perception(out)

    async def start(self, home: tuple[float, float]) -> None:
        await super().start(home)
        self._tasks.append(asyncio.ensure_future(self.perception.run(self._stop)))


@dataclass
class SearchRun:
    """What a run saw, for its checks and its metrics."""

    world: World
    sc: Scenario
    onboard: ChainOnboard
    operator: ScriptedOperator
    plane: Plane
    matcher: OracleMatcher
    lift_t: float | None = None
    end_t: float | None = None
    battery_start: float | None = None
    keepouts: dict[str, SeenKeepout] = field(default_factory=dict[str, SeenKeepout])
    candidates: CandidateWatch = field(default_factory=CandidateWatch)

    def thing_at(self, lat: float, lon: float) -> tuple[str | None, bool]:
        """The simulator's truth at a point: the nearest thing within ``TRUTH_M`` and
        whether it is the missing person."""
        east, north = self.plane.xy(lat, lon)
        t = self.world.nearest(east, north, TRUTH_M)
        if t is None:
            return None, False
        return t.thing_id, t.target

    def vlm_truth(self, req: Any) -> Truth:
        """The image model's truth for a request: the thing under its crops."""
        for c in list(req.crops):
            where = self.onboard.positions.get(str(c.detection_id))
            if where is None:
                continue
            east, north = self.plane.xy(*where)
            t = self.world.nearest(east, north, TRUTH_M)
            if t is not None:
                return Truth(person=t.person, target=t.target)
        return Truth(person=False, target=False)


@contextlib.asynccontextmanager
async def search_service(
    sc: Scenario, world: World, truth: Callable[[Any], Truth]
) -> AsyncIterator[OracleMatcher]:
    """``sd-search`` as its CLI builds it, with the oracle as the image model."""
    client = GroundClient(f"http://127.0.0.1:{sc.port}", flow_rig.SEARCH)
    profiles = GroundProfileSource(client)
    terrain: TerrainSource = SimTerrainSource(world.terrain()) if world.trail else NoTerrain()
    matcher = OracleMatcher(truth, seed=world.seed, sleep=True, environ=SIM_ENV)
    service = SearchService(
        client,
        profiles,
        mapper=MapService(client, profiles, terrain),
        # The oracle has the engine's shape; its engine name and request type are its own.
        cascade=Cascade(matcher),  # pyright: ignore[reportArgumentType]
    )
    task = asyncio.ensure_future(service.run(asyncio.Event(), poll_s=0.25))
    try:
        yield matcher
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await client.aclose()


async def rows(sc: Scenario, sql: str, **params: Any) -> list[Any]:
    async with sc.r.flow.repo.engine.connect() as conn:
        return list((await conn.execute(text(sql), {"m": sc.mid, **params})).all())


async def _watch(run: SearchRun, stop: asyncio.Event) -> None:
    """Sample the cylinders the executor holds and the candidates the ground holds."""
    ex = run.onboard.ex
    looked = 0.0
    while not stop.is_set():
        now = time.monotonic()
        held = {str(k.keepout_id): k for k in ex.keepouts_in_force()}
        for kid, k in held.items():
            if kid not in run.keepouts:
                run.keepouts[kid] = SeenKeepout(kid, k.lat, k.lon, k.radius_m, now)
        for kid, seen in list(run.keepouts.items()):
            if kid not in held and seen.until == math.inf:
                run.keepouts[kid] = SeenKeepout(
                    kid, seen.lat, seen.lon, seen.radius_m, seen.since, now
                )
        if now - looked >= 2.0 and run.sc.mid:
            looked = now
            got = await rows(
                run.sc,
                "SELECT candidate_id::text, max(rev) FROM candidates "
                "WHERE mission_id = CAST(:m AS uuid) GROUP BY candidate_id",
            )
            run.candidates.look({str(r[0]): int(r[1]) for r in got})
        await asyncio.sleep(WATCH_S)


def _lifted(sc: Scenario) -> float | None:
    return next((s.t for s in sc.track if s.armed and s.alt_m > 1.0), None)


async def _search_invariants(run: SearchRun) -> list[str]:
    sc = run.sc
    revisions = await rows(
        sc,
        "SELECT candidate_id::text AS candidate_id, rev, state, created_at FROM candidates "
        "WHERE mission_id = CAST(:m AS uuid)",
    )
    ids = sorted({r.candidate_id for r in revisions})
    decisions = await rows(
        sc,
        "SELECT candidate_id::text AS candidate_id, decision, operator_ref, session_id, "
        "created_at FROM candidate_decisions WHERE candidate_id::text = ANY(:ids)",
        ids=ids,
    )
    views = await rows(
        sc,
        "SELECT detection_id::text AS detection_id, session_id, created_at "
        "FROM candidate_views WHERE mission_id = CAST(:m AS uuid)",
    )
    obs = await rows(
        sc,
        "SELECT candidate_id::text, detection_id::text FROM candidate_observations "
        "WHERE candidate_id::text = ANY(:ids)",
        ids=ids,
    )
    observed: dict[str, list[str]] = {}
    for cid, det in obs:
        observed.setdefault(cid, []).append(det)
    broken = no_confirm_without_human(
        [r._asdict() for r in revisions],
        [d._asdict() for d in decisions],
        [v._asdict() for v in views],
        observed,
    )
    run.candidates.look({cid: max(r.rev for r in revisions if r.candidate_id == cid)
                         for cid in ids})  # fmt: skip
    broken += run.candidates.broken
    broken += keepout_never_entered(run.keepouts.values(), run.sc.track)
    return broken


async def _map_measures(run: SearchRun) -> tuple[float | None, float | None]:
    """POS at the end (the first map's probabilities times the last map's cumulative
    POD) and the missing person's cell's POA percentile in the first map."""
    maps = await rows(
        run.sc,
        "SELECT version, grid, p, pod_cum FROM probability_maps "
        "WHERE mission_id = CAST(:m AS uuid) ORDER BY version",
    )
    if not maps:
        return None, None
    first, last = maps[0], maps[-1]
    g0 = ProbabilityGrid.model_validate(first.grid["grid"])
    g1 = ProbabilityGrid.model_validate(last.grid["grid"])
    p0 = np.frombuffer(zlib.decompress(bytes(first.p)), dtype="<f4").astype(np.float64)
    pod = np.frombuffer(zlib.decompress(bytes(last.pod_cum)), dtype="<f4").astype(np.float64)
    same = (g0.nx, g0.ny, g0.x0_m, g0.y0_m, g0.cell_m) == (g1.nx, g1.ny, g1.x0_m, g1.y0_m,
                                                           g1.cell_m)  # fmt: skip
    pos = float((p0 * pod).sum()) if same and p0.size == pod.size else None
    rank = None
    target = run.world.target
    if target is not None:
        lat, lon = run.world.where(target)
        x, y = Plane(g0.ipp_lat, g0.ipp_lon).xy(lat, lon)
        i, j = math.floor((x - g0.x0_m) / g0.cell_m), math.floor((y - g0.y0_m) / g0.cell_m)
        if 0 <= i < g0.nx and 0 <= j < g0.ny:
            v = p0[j * g0.nx + i]
            rank = round(100.0 * float((p0 < v).sum()) / p0.size, 1)
    return (None if pos is None else round(pos, 4)), rank


async def _metrics(run: SearchRun, broken: list[str]) -> RunMetrics:
    sc, world, op = run.sc, run.world, run.operator
    target = world.target
    m = RunMetrics(
        scenario=world.scenario,
        seed=world.seed,
        detector=run.onboard.detector.profile.profile_id,
        matcher=run.matcher.profile.profile_id,
        limit_s=world.limit_s,
        has_target=target is not None,
        invariants_broken=broken,
    )
    lift = run.lift_t
    for d in op.decisions:
        if d.decision == "confirm" and not d.target:
            m.false_confirms += 1
    confirmed = op.confirmed
    if confirmed is not None:
        m.confirmed_thing = confirmed.thing
        if confirmed.target and lift is not None:
            m.time_to_confirmed_s = round(confirmed.t - lift, 1)
    m.queue_rank_target = op.target_first_rank
    m.target_ever_low = op.target_ever_low
    m.decisions = [
        {
            "t_s": None if lift is None else round(d.t - lift, 1),
            "seq": d.seq,
            "decision": d.decision,
            "truth": d.thing or "inget",
            "rank": d.rank,
            "priority_class": d.priority_class,
        }
        for d in op.decisions
    ]
    queue = await op.queue()
    m.candidates = len(queue)
    m.queue_end = [
        {
            "seq": c["seq"],
            "state": c["state"],
            "priority": c["priority"],
            "priority_class": c["priority_class"],
            "engine": c.get("engine"),
            "truth": run.thing_at(c["lat"], c["lon"])[0],
        }
        for c in queue
    ]
    for c in queue:
        thing = run.thing_at(c["lat"], c["lon"])[0]
        if thing is not None and thing not in m.fields_by_thing and c.get("fields"):
            m.fields_by_thing[thing] = dict(c["fields"])
        if c["state"] == "confirmed" and target is not None and thing == target.thing_id:
            m.geo_error_m = round(distance_m(c["lat"], c["lon"], *world.where(target)), 2)
    flight = [s for s in sc.track if s.armed]
    m.flight_s = round(flight[-1].t - flight[0].t, 1) if len(flight) > 1 else None
    hours = (m.flight_s or 0.0) / 3600.0
    if hours > 0:
        by_class: dict[str, int] = {}
        for c in queue:
            if c["state"] != "merged" and not run.thing_at(c["lat"], c["lon"])[1]:
                by_class[c["priority_class"]] = by_class.get(c["priority_class"], 0) + 1
        m.false_candidates_per_hour = {k: round(v / hours, 1) for k, v in by_class.items()}
    lags = await rows(
        sc,
        "SELECT EXTRACT(EPOCH FROM min(o.created_at) - s.captured_utc) * 1000 "
        "FROM candidate_observations o JOIN detections d ON d.detection_id = o.detection_id "
        "JOIN perception_stills s ON s.still_id = d.still_id "
        "WHERE d.mission_id = CAST(:m AS uuid) GROUP BY d.detection_id, s.captured_utc",
    )
    ms = [float(r[0]) for r in lags]
    m.latency_still_to_queue_ms = {
        "p50": None if not ms else round(statistics.median(ms)),
        "p95": None if not ms else round(percentile(ms, 95) or 0.0),
    }
    engines = await rows(
        sc,
        "SELECT DISTINCT a.engine FROM candidate_assessments a JOIN candidates c "
        "ON c.candidate_id = a.candidate_id WHERE c.mission_id = CAST(:m AS uuid)",
    )
    m.engines = sorted(str(r[0]) for r in engines)
    m.keepouts = len(run.keepouts)
    m.keepout_min_distance_m = keepout_distances(run.keepouts.values(), sc.track)
    if m.keepout_min_distance_m is not None:
        m.keepout_min_distance_m = round(m.keepout_min_distance_m, 1)
    m.pos_end, _rank = await _map_measures(run)
    battery_end = sc.tel.battery_pct
    if run.battery_start is not None and battery_end is not None:
        m.battery_used_pct = round(run.battery_start - battery_end, 1)
    return m.finish()


async def search_run(
    sc: Scenario,
    world: World,
    onboard: ChainOnboard,
    run_dir: Path,
    *,
    confirm_limit_s: float | None = None,
    fly: Callable[[SearchRun], Any] | None = None,
    require_found: bool = True,
    with_map: bool | None = None,
) -> RunMetrics:
    """One scenario run (see the module's docstring). ``fly`` replaces the AI's search
    once airborne (SK06 orders the sectors itself); ``confirm_limit_s`` is how long the
    operator may take to the confirmed find after take-off (the scenario's limit when
    left out). With ``require_found`` the test fails without a confirmed find."""
    plane = Plane(*sc.r.home)
    limit = confirm_limit_s or world.limit_s
    holder: list[SearchRun] = []
    script = world.scenario

    async with search_service(sc, world, lambda req: holder[0].vlm_truth(req)) as matcher:
        await sc.open()
        operator = ScriptedOperator(
            sc.r.panel,
            sc.mid,
            truth=lambda lat, lon: holder[0].thing_at(lat, lon),
            headers=H,
            seed=world.seed,
            env=SIM_ENV,
        )
        run = SearchRun(world, sc, onboard, operator, plane, matcher)
        holder.append(run)
        await operator.approve_profile(world.profile_draft(sc.r.clock.utc_now().isoformat()))
        await sc.start_brain(script)
        charter = next(iter(_charters(world)))
        await sc.say(f"Använd ramen {charter}.")
        await sc.approve_proposed()
        await sc.clearance()
        run.battery_start = sc.tel.battery_pct
        await sc.say("Starta.")
        await sc.wait_command("takeoff", "done", 90, issuer_kind="brain")
        run.lift_t = _lifted(sc)

        stop = asyncio.Event()
        watch = asyncio.ensure_future(_watch(run, stop))
        reviewing = asyncio.ensure_future(operator.run(stop))
        flying = None if fly is None else asyncio.ensure_future(fly(run))
        try:
            if world.target is not None:
                with contextlib.suppress(AssertionError):
                    await sc.until(
                        lambda: (
                            operator.confirmed is not None
                            or reviewing.done()
                            or (flying is not None and flying.done())
                            or not sc.tel.armed
                        ),  # the AI flew home without a find
                        limit,
                        f"{world.scenario} seed {world.seed}: confirmed find",
                    )
                if reviewing.done() and reviewing.exception() is not None:
                    raise reviewing.exception()  # type: ignore[misc]
                if operator.confirmed is not None:
                    await sc.say("Flyg hem.")
                elif sc.tel.armed:
                    assert (await sc.press("return_home")).status_code == 202
            elif fly is None:

                async def going_home() -> bool:
                    return any(c["skill"] == "return_home" for c in await sc.commands())

                await sc.until(going_home, limit, f"{world.scenario}: the AI flies home")
            if flying is not None and not flying.done():
                flying.cancel()
            await sc.until(lambda: not sc.tel.armed, LANDED_LIMIT_S, "landed")
            run.end_t = time.monotonic()
            # The last cards the operator has not reached yet are left as they are.
        finally:
            stop.set()
            for task in (watch, reviewing, flying):
                if task is not None:
                    task.cancel()
            await asyncio.gather(
                *(t for t in (watch, reviewing, flying) if t is not None),
                return_exceptions=True,
            )
        await sc.check_invariants()
        broken = await _search_invariants(run)
        metrics = await _metrics(run, broken)
        metrics.with_map = with_map
    write_run(run_dir, metrics)
    if metrics.time_to_confirmed_s is not None:
        measure(f"{world.scenario} take-off -> confirmed", metrics.time_to_confirmed_s, limit)
    assert not broken, broken
    assert metrics.false_confirms == 0, metrics.decisions
    if (
        require_found
        and world.target is not None
        and metrics.confirmed_thing != world.target.thing_id
    ):
        orders = [(c["skill"], c["status"], c.get("status_reason")) for c in await sc.commands()]
        det = onboard.detector
        seen = det.in_view.get(world.target.thing_id, 0)
        drawn = sum(d.thing_id == world.target.thing_id for d in det.drawn)
        where = world.where(world.target)
        near = [round(distance_m(c["lat"], c["lon"], *where)) for c in await run.operator.queue()]
        raise AssertionError(
            f"no confirmed find: {metrics.decisions}; queue {metrics.queue_end}; "
            f"AI turns {sc.brain_turns()}; orders {orders}; "
            f"the target in {seen} stills, detected in {drawn}, cards from it at {sorted(near)} m"
        )
    return metrics


def _charters(world: World) -> list[str]:
    data = yaml.safe_load(world.fixture_path.read_text(encoding="utf-8"))
    return list(data["charters"])
