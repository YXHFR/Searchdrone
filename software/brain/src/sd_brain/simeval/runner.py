"""One simulated search flight: the real ``CommanderLoop`` against ``SimGround``.

The pilot's opening line starts the flight. Then the world runs until something the AI
should hear happens (an order done, a sighting, a STOP), the event goes to the loop, and
the loop's turn answers through the decision maker and the gate. When the drone hovers
with nothing to do, the loop gets an idle tick; three in a row without an order end the
flight, as do landing and the time limit. At ``max_turns`` the AI gets no more turns and
the harness presses FLY HOME, as the pilot's button does, past the AI: the world runs on
until the drone has landed or the time is up. The score records that the cap was hit and
gives neither the landing nor the coverage of that pass home to the AI.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from typing import Any

from sd_brain.backends import GrammarBackend
from sd_brain.config import BrainConfig
from sd_brain.decide.service import DecisionService
from sd_brain.evals.runner import EvalClock
from sd_brain.health import HealthState
from sd_brain.loop import CommanderLoop, TurnRecord
from sd_brain.simeval.decider import DecisionMaker
from sd_brain.simeval.scenario import SimScenario
from sd_brain.simeval.score import FlightLog, FlightScore, LoggedMark, LoggedOrder, score_flight
from sd_brain.simeval.world import SimGround, SimWorld
from sd_schemas.skills import SKILL_REGISTRY

__all__ = ["FlightResult", "fly"]

IDLE_TICKS_TO_END = 3


@dataclass
class FlightResult:
    score: FlightScore
    log: FlightLog
    records: list[TurnRecord]
    switch_reason: str | None
    replies: list[dict[str, Any]]  # the cloud model's answers in order (for replay fixtures)


async def fly(
    scenario: SimScenario,
    decider: DecisionMaker,
    *,
    model: str,
    ai_config_hash: str,
    cost_before_sek: float | None = None,
    meter: Any = None,
    max_turns: int = 40,
) -> FlightResult:
    if cost_before_sek is None:
        cost_before_sek = meter.cost_sek if meter is not None else 0.0
    calls0 = meter.calls if meter is not None else 0
    unmeasured0 = meter.unmeasured_calls if meter is not None else 0
    replies0 = len(meter.replies) if meter is not None else 0
    clock = EvalClock()
    world = SimWorld(scenario)
    ground = SimGround(world, now=clock.utc_now)
    cfg = BrainConfig()
    loop = CommanderLoop(
        ground,
        decider,
        GrammarBackend(),
        SKILL_REGISTRY,
        cfg,
        ai_config_hash,
        health=HealthState(clock.monotonic),
        decide=DecisionService([]),
        clock=clock,  # type: ignore[arg-type]
    )
    await loop.set_mission(ground.mission_id)
    loop.ready_since = clock.utc_now() - dt.timedelta(seconds=1)
    loop.handle_event(ground.operator_message(scenario.opening, created_at=clock.utc_now()))
    await loop.settle()
    idle = 0
    while not world.finished() and len(loop.records) < max_turns:
        before = len(ground.orders)
        events = world.advance()
        clock.t = max(clock.t, world.t)
        if events:
            for evt in events:
                evt["mission_id"] = ground.mission_id
                ground.sync()
                ground.apply_event(evt)
                loop.handle_event(evt)
            await loop.settle()
            idle = 0
            continue
        if world.finished():
            break
        if not loop.idle_tick(force=True):
            break
        await loop.settle()
        idle = idle + 1 if len(ground.orders) == before else 0
        if idle >= IDLE_TICKS_TO_END:
            break
    turn_cap_reached = len(loop.records) >= max_turns and not world.finished()
    coverage = world.coverage()  # the AI's search: not the pass home after the cap
    if turn_cap_reached:
        _fly_home(world)
    ground.sync()
    questions = sum(1 for m in ground.posted if m.kind == "question")
    target = scenario.target_latlon
    cost = (meter.cost_sek - cost_before_sek) if meter is not None else 0.0
    calls = (meter.calls - calls0) if meter is not None else 0
    unmeasured = (meter.unmeasured_calls - unmeasured0) if meter is not None else 0
    log = FlightLog(
        flight_id=str(uuid.uuid4()),
        seed=scenario.seed,
        decider=decider.started_as,
        exam=decider.exam,
        model=model if decider.started_as == "cloud" else "script",
        ai_config_hash=ai_config_hash,
        target_lat=target[0],
        target_lon=target[1],
        duration_s=world.t,
        max_time_s=scenario.max_time_s,
        orders=[
            LoggedOrder(
                t_s=o.t,
                skill=o.skill,
                decision=o.decision,  # type: ignore[arg-type]
                reason_codes=list(o.reason_codes),
                under_stop=o.under_stop,
            )
            for o in ground.orders
        ],
        marks=[LoggedMark(t_s=m.t, lat=m.lat, lon=m.lon, label=m.label) for m in world.marks],
        forbidden_calls=list(decider.forbidden_calls),
        geofence_breaches=world.geofence_breaches,
        coverage=coverage,
        pilot_questions=questions,
        landed_home=world.landed and abs(world.east) < 5 and abs(world.north) < 5,
        cost_sek=cost,
        # Measured only when the provider reported the price of every call in this flight;
        # a flight without a priced call (no key, a refused key) has no measured cost.
        cost_measured=calls > 0 and unmeasured == 0,
        cloud_calls=sum(
            r.output.diagnostics.rounds
            for r in loop.records
            if r.output is not None and r.output.diagnostics.backend in ("cloud", "llm")
        ),
        turns=len(loop.records),
        turn_cap_reached=turn_cap_reached,
        fallback_reason=decider.switch_reason,
    )
    replies = list(meter.replies[replies0:]) if meter is not None else []
    return FlightResult(score_flight(log), log, list(loop.records), decider.switch_reason, replies)


def _fly_home(world: SimWorld) -> None:
    """FLY HOME without the AI: the world alone, until landed or out of time. A STOP the
    scenario sets on the way holds the drone; after its release the button holds again."""
    while not world.finished():
        if world.plan is None and not world.latched:
            world.fly_home()
        world.advance()
