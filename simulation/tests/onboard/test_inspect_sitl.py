"""A closer look straight over a candidate (TN-6) against ArduCopter SITL, builtin physics.

The onboard rig of ``test_onboard_sitl``: the executor in this process, SD-link to a test
ground. The cameras and the detector are faked (``capture_round``): a person 40 m east of
home in every still, small enough never to be a clear picture, so the drone steps all the
way down. The height floor comes from a map pack of flat open meadow at home's height.

* without the laser: steps down to 30 m above ground, asks "Gå lägre?", gets no answer,
  waits over the find, and flies back to where it broke off on the pilot's "resume";
* with a (fake) laser: the pilot's no ("Nej, stanna här") ends the look at once at 30 m;
* with a (fake) laser: the pilot's yes takes it below 30 m, and "confirmed" leaves it
  hovering over the find.
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_onboard_sitl import (  # pyright: ignore[reportMissingImports]
    Flight,
    Sim,
    _install,
    _takeoff,
    flight,
    sim,
)

from sd_safety.floor import FloorPack
from sd_schemas.base import LatLon, new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.inspect import LOWER_OFFER_S
from sd_schemas.link import BBox, GeoDetectionMsg, InspectLower, InspectOutcome, InspectStay
from sd_schemas.rangefinder import LaserState

__all__ = ["sim"]  # the fixture, imported from the onboard SITL rig

pytestmark = [pytest.mark.sitl, pytest.mark.slow]

TARGET = (40.0, 0.0)  # east, north of home (m)
START_ALT = 50.0
SKILLS = ["takeoff", "goto", "hold", "resume", "return_home", "capture_stills", "inspect_detection"]


class Events:
    """The look's own events (``inspect.*``) as the executor sends them up."""

    def __init__(self, f: Flight) -> None:
        self.seen: list[tuple[str, dict[str, Any]]] = []
        send = f.ex.emit_inspect

        def recording(kind: str, data: dict[str, Any]) -> None:
            self.seen.append((kind, data))
            send(kind, data)

        f.ex.emit_inspect = recording  # type: ignore[method-assign]

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [d for k, d in self.seen if k == kind]


class Person:
    """``Executor.capture_round``: a main and a tele still per round, both seeing the
    person at ``TARGET`` (40 px: never a clear picture)."""

    def __init__(self, f: Flight) -> None:
        self.lat, self.lon = f.at(*TARGET)
        self.asked: list[tuple[UUID, int]] = []

    def det(self) -> GeoDetectionMsg:
        return GeoDetectionMsg(
            detection_id=new_id(),
            bbox=BBox(x=10, y=10, w=20, h=40),
            score=0.9,
            lat=self.lat,
            lon=self.lon,
            sigma95_m=2.0,
            px_size=40,
            crop_inline=False,
            label="person",
        )

    async def __call__(
        self, command_id: UUID, step: int
    ) -> list[tuple[str, UUID, list[GeoDetectionMsg]]]:
        self.asked.append((command_id, step))
        await asyncio.sleep(0.2)  # the stills take a moment
        return [("main", new_id(), [self.det()]), ("tele", new_id(), [self.det()])]


class Laser:
    """A laser that always sees the flat ground (``LaserReader.state``)."""

    def __init__(self, surface_amsl_m: float) -> None:
        self.surface = surface_amsl_m

    def state(self, now_ms: int) -> LaserState:
        return LaserState(confirmed=True, surface_amsl_m=self.surface, min_range_m=0.0)


def meadow(f: Flight) -> FloorPack:
    """600 x 600 m of open meadow (40 cm grass) at home's height, centred on home."""
    home = f.ground.telemetry.home
    assert home is not None
    size = 600
    return FloorPack(
        name="meadow",
        sha256="0" * 64,
        origin=LatLon(lat=home.lat, lon=home.lon),
        dtm_cm=np.full((size, size), round(home.alt_amsl_m * 100), dtype=np.int32),
        chm_dm=np.full((size, size), 4, dtype=np.int16),
        scan_year=np.full((1, 1), 2019, dtype=np.uint16),
        scan_cell_m=size,
    )


async def _look(f: Flight, *, laser: bool) -> tuple[Any, Events, Person, list[float]]:
    """Charter (sim, overflight allowed), takeoff, the order; returns the heights flown
    during the look (above home) as they are sampled."""
    await _install(f, allowed_skills=SKILLS, operation_mode="sim", overflight_allowed=True)
    await _takeoff(f, START_ALT)
    await f.until(lambda: f.ex.state == S.HOLD, 60, "takeoff done")
    pack = meadow(f)
    f.ex.floor_pack = pack
    if laser:
        home = f.ground.telemetry.home
        assert home is not None
        f.ex.laser = Laser(home.alt_amsl_m)  # type: ignore[assignment]
    ev, person = Events(f), Person(f)
    f.ex.capture_round = person
    heights: list[float] = []
    f.onboard.subscribe(lambda t: heights.append(t.alt_rel_home_m) if t.armed else None)
    lat, lon = f.at(*TARGET)
    cmd = await f.order(
        "inspect_detection",
        {
            "candidate_id": str(new_id()),
            "lat": lat,
            "lon": lon,
            "sigma95_m": 2.0,
            "target_radius_m": 32.0,
        },
    )
    ack = await f.stage(cmd, {"running"}, 10)
    assert ack.stage == "running", ack
    await f.until(lambda: f.ex.state == S.INSPECT, 5, "INSPECT")
    return cmd, ev, person, heights


def _xy(f: Flight) -> tuple[float, float]:
    t = f.onboard.telemetry
    return f.plane.xy(t.lat, t.lon)


@pytest.mark.budget_s(420)
def test_steps_down_to_30_m_waits_and_flies_back(sim: Sim) -> None:
    async def run() -> None:
        async with flight(sim) as f:
            cmd, ev, person, heights = await _look(f, laser=False)
            await f.until(lambda: bool(ev.of("inspect.lower_offer")), 180, "Gå lägre?")
            offer = ev.of("inspect.lower_offer")[0]
            assert offer["command_id"] == str(cmd.command_id)
            assert math.dist(_xy(f), TARGET) <= 5
            assert abs(f.onboard.telemetry.alt_rel_home_m - 30) <= 2
            assert {c for c, _ in person.asked} == {cmd.command_id}
            assert max(s for _, s in person.asked) >= 2  # 50 -> 40 -> 30 m
            # No answer: it stays at 30 m and waits over the find.
            await f.until(
                lambda: f.ex.hold_reason == "inspect_await", LOWER_OFFER_S + 20, "the wait"
            )
            done = ev.of("inspect.done")
            assert len(done) == 1 and done[0]["end"] == "lower_declined", done
            assert min(heights) >= 30 - 2  # never below the automatic floor
            assert f.ex.state == S.HOLD
            # The pilot: resume. Straight up to 50 m, then back to where it broke off.
            assert f.session is not None
            await f.session.send(
                InspectOutcome(command_id=cmd.command_id, outcome="resume"), ttl_ms=15_000
            )
            await f.until(lambda: f.ex.snapshot().inspect is None, 60, "the return")
            assert f.ex.state in (S.INSPECT, S.TRANSIT, S.HOLD), f.ex.state
            await f.until(
                lambda: f.ex.state == S.HOLD and math.dist(_xy(f), (0.0, 0.0)) <= 5,
                120,
                "back at the breakpoint",
            )
            assert abs(f.onboard.telemetry.alt_rel_home_m - START_ALT) <= 3
            assert not ev.of("inspect.aborted")

    asyncio.run(run())


@pytest.mark.budget_s(300)
def test_the_pilots_no_stops_the_question_at_once_at_30_m(sim: Sim) -> None:
    async def run() -> None:
        async with flight(sim) as f:
            cmd, ev, _, heights = await _look(f, laser=True)
            await f.until(lambda: bool(ev.of("inspect.lower_offer")), 180, "Gå lägre?")
            hover = f.onboard.telemetry.alt_rel_home_m
            assert abs(hover - 30) <= 2
            assert f.session is not None
            loop = asyncio.get_running_loop()
            said_no = loop.time()
            await f.session.send(InspectStay(command_id=cmd.command_id), ttl_ms=15_000)
            # The drone stops asking at once, well before the 60 s of no answer.
            await f.until(lambda: f.ex.hold_reason == "inspect_await", 20, "the wait")
            assert loop.time() - said_no < 20 < LOWER_OFFER_S
            done = ev.of("inspect.done")
            assert len(done) == 1 and done[0]["end"] == "lower_declined", done
            await asyncio.sleep(5)
            assert min(heights) >= 30 - 2  # the laser could have taken it lower: a no keeps it
            assert abs(f.onboard.telemetry.alt_rel_home_m - hover) <= 2
            assert math.dist(_xy(f), TARGET) <= 5
            assert len(ev.of("inspect.lower_offer")) == 1
            assert not ev.of("inspect.aborted")

    asyncio.run(run())


@pytest.mark.budget_s(360)
def test_with_the_laser_the_pilots_yes_goes_lower_and_confirmed_hovers(sim: Sim) -> None:
    async def run() -> None:
        async with flight(sim) as f:
            cmd, ev, _, heights = await _look(f, laser=True)
            await f.until(lambda: bool(ev.of("inspect.lower_offer")), 180, "Gå lägre?")
            assert f.session is not None
            await f.session.send(InspectLower(command_id=cmd.command_id), ttl_ms=15_000)
            await f.until(lambda: f.ex.hold_reason == "inspect_await", 120, "the wait")
            done = ev.of("inspect.done")
            assert len(done) == 1 and done[0]["end"] == "floor", done
            low = f.onboard.telemetry.alt_rel_home_m
            assert low < 30 - 5, low  # below the automatic floor with the pilot's yes
            assert min(heights) >= 15 - 2  # never below the charter's lowest height
            assert math.dist(_xy(f), TARGET) <= 5
            await f.session.send(
                InspectOutcome(command_id=cmd.command_id, outcome="confirmed"), ttl_ms=15_000
            )
            await f.until(lambda: f.ex.snapshot().inspect is None, 30, "confirmed")
            await asyncio.sleep(5)
            assert f.ex.state == S.HOLD
            assert math.dist(_xy(f), TARGET) <= 5  # hovers over the find
            assert abs(f.onboard.telemetry.alt_rel_home_m - low) <= 2
            assert not ev.of("inspect.aborted")

    asyncio.run(run())
