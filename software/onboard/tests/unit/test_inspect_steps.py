"""TN-6: the closer look straight over a candidate (``skills.inspect_steps``) and the
executor's part of it (``executor.inspect``): the wait, the decision and the return."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any
from uuid import UUID

import numpy as np
import pytest
from onboard_helpers import ONBOARD_SKILLS, PLANE, Rig, at, charter, rig

from sd_mav.modes import CopterMode
from sd_onboard.executor.inspect import INTERNAL_INSPECT_CLIMB, abort_reason
from sd_onboard.executor.limits import LimitAction
from sd_onboard.executor.link_loss import LinkLossInput, LinkLossRule
from sd_onboard.skills import inspect_steps
from sd_onboard.skills.base import InspectShot
from sd_onboard.skills.inspect_steps import estimate, is_clear, look_radius_m, pick
from sd_safety.constants import LINK_LOSS_DETECT_S
from sd_safety.floor import FloorPack, floor_at
from sd_schemas.base import LatLon, new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.executor import LinkLossPhase
from sd_schemas.inspect import AWAIT_DECISION_S, DESCENT_MAX_MS, LOWER_OFFER_S
from sd_schemas.keepout import Keepout
from sd_schemas.link import (
    BBox,
    GeoDetectionMsg,
    InspectLower,
    InspectOutcome,
    InspectStay,
    KeepoutInstall,
)
from sd_schemas.rangefinder import LaserState

SKILLS = [*ONBOARD_SKILLS, "capture_stills", "inspect_detection"]
TARGET = (60.0, 0.0)  # east, north of home


# --- the pure parts ---------------------------------------------------------------------


def det(
    east: float, north: float, score: float = 0.9, px: float = 40.0, sigma: float = 2.0
) -> GeoDetectionMsg:
    lat, lon = at(east, north)
    return GeoDetectionMsg(
        detection_id=new_id(),
        bbox=BBox(x=10, y=10, w=px / 2, h=px),
        score=score,
        lat=lat,
        lon=lon,
        sigma95_m=sigma,
        px_size=px,
        crop_inline=False,
        label="person",
    )


def _anchor() -> LatLon:
    lat, lon = at(*TARGET)
    return LatLon(lat=lat, lon=lon)


def _dist(lat: float, lon: float) -> float:
    return math.dist(PLANE.xy(lat, lon), PLANE.point(_anchor()))


def test_the_look_radius_is_15_to_30_m() -> None:
    assert look_radius_m(None) == 15.0
    assert look_radius_m(10.0) == 20.0
    assert look_radius_m(40.0) == 30.0


def test_the_target_in_a_still_is_the_detection_nearest_the_anchor() -> None:
    near, nearer, far = det(70, 0, score=0.99), det(62, 0, score=0.6), det(100, 0)
    shot = InspectShot(new_id(), "main", (near, far, nearer))
    m = pick(shot, _anchor(), 15.0, _dist)
    assert m is not None and m.score == 0.6  # the nearest, not the surest
    assert pick(InspectShot(new_id(), "tele", (far,)), _anchor(), 15.0, _dist) is None


def test_the_estimate_weighs_the_anchor_and_the_sure_matches() -> None:
    a = _anchor()
    assert estimate(a, 5.0, []) == a
    lat, lon = at(66, 0)
    sure = inspect_steps.Match(new_id(), "tele", lat, lon, 0.9, 0.5, 100)
    unsure = inspect_steps.Match(new_id(), "tele", lat, lon, 0.3, 0.5, 100)
    assert estimate(a, 5.0, [unsure]) == a
    moved = PLANE.point(estimate(a, 5.0, [sure]))
    assert 65 < moved[0] < 66  # the sharp match counts far more than the anchor


def test_a_clear_picture_needs_two_sharp_big_stills() -> None:
    def m(score: float, px: float) -> inspect_steps.Match:
        return inspect_steps.Match(new_id(), "main", 0, 0, score, 1, px)

    assert is_clear([m(0.9, 100), m(0.5, 96)])
    assert not is_clear([m(0.9, 100), m(0.9, 95)])
    assert not is_clear([m(0.9, 100), m(0.49, 200)])
    one = m(0.9, 200)
    assert not is_clear([one, one])  # the same still twice


def test_abort_reasons() -> None:
    assert abort_reason(S.HELD, "operator_stop", None, "human") == "stop"
    assert abort_reason(S.HELD, "operator_home", None, "human") == "home"
    assert abort_reason(S.HELD, "system_battery", None, "human") == "battery"
    assert abort_reason(S.HELD, "external", None, "human") == "override"
    assert abort_reason(S.FAILSAFE, None, None, "gnss") == "gnss"
    assert abort_reason(S.RTL, None, None, "battery") == "battery"
    assert abort_reason(S.RTL, None, None, None) == "home"
    assert abort_reason(S.HOLD, None, "link_loss", None) == "link_loss"
    assert abort_reason(S.HOLD, None, "limit", None) == "override"
    assert abort_reason(S.FAILSAFE, None, None, "fc_failsafe") == "failed"


async def test_the_link_rule_climbs_first_during_a_look() -> None:
    rule = LinkLossRule()
    async with rig() as r:
        await r.install(over())
        v = r.ex.vehicle_state()
        c = r.ex.charter
    assert c is not None
    x = LinkLossInput(
        now_ms=10_000,
        state=S.INSPECT,
        vehicle=v.model_copy(update={"alt_rel_home_m": 25.0}),
        charter=c,
        link_up=False,
        link_last_inbound_ms=0,
        nav_ok=True,
        keepouts=(),
        gcs_continue=False,
        active_skill="inspect_detection",
        inspect_start_alt_m=60.0,
    )
    actions = rule.check(x)
    climb = [a for a in actions if a.kind == "inspect_climb"]
    assert climb and climb[0].data["alt_m"] == 60.0
    assert rule.phase == LinkLossPhase.GRACE


# --- the rig ----------------------------------------------------------------------------


class Rounds:
    """``Executor.capture_round``: a main and a tele still per round, the detections from
    ``seen(step)`` in both."""

    def __init__(self, seen: Callable[[int], list[GeoDetectionMsg]] | None = None) -> None:
        self.seen = seen or (lambda _step: [])
        self.asked: list[tuple[UUID, int]] = []
        self.works = True

    async def __call__(
        self, command_id: UUID, step: int
    ) -> list[tuple[str, UUID, list[GeoDetectionMsg]]]:
        self.asked.append((command_id, step))
        if not self.works:
            return []
        return [("main", new_id(), self.seen(step)), ("tele", new_id(), self.seen(step))]


def meadow(chm_dm: int = 4) -> FloorPack:
    """600 x 600 m of open meadow at home's height, scanned 2019."""
    from onboard_helpers import HOME_ALT

    size = 600
    return FloorPack(
        name="meadow",
        sha256="0" * 64,
        origin=LatLon(lat=PLANE.lat0, lon=PLANE.lon0),
        dtm_cm=np.full((size, size), round(HOME_ALT * 100), dtype=np.int32),
        chm_dm=np.full((size, size), chm_dm, dtype=np.int16),
        scan_year=np.full((1, 1), 2019, dtype=np.uint16),
        scan_cell_m=size,
    )


def over(**kw: Any) -> Any:
    return charter(allowed_skills=SKILLS, operation_mode="sim", overflight_allowed=True, **kw)


def order(r: Rig, **kw: Any) -> Any:
    lat, lon = at(*TARGET)
    params: dict[str, Any] = {
        "candidate_id": str(kw.pop("candidate_id", new_id())),
        "lat": lat,
        "lon": lon,
        "sigma95_m": 2.0,
        "target_radius_m": 32.0,
    }
    params.update(kw)
    return r.order("inspect_detection", params)


def events(r: Rig, kind: str) -> list[dict[str, Any]]:
    return [d for k, _, d in r.up.events if k == kind]


async def start(
    r: Rig,
    *,
    alt: float = 60.0,
    pack: FloorPack | None = None,
    rounds: Rounds | None = None,
    **kw: Any,
) -> tuple[Any, Rounds]:
    await r.install(over(**kw))
    await r.takeoff(alt)
    r.ex.floor_pack = pack
    rd = rounds or Rounds()
    r.ex.capture_round = rd
    cmd = await r.send(order(r), 0.5)
    assert r.ex.state == S.INSPECT, r.up.acks[-2:]
    return cmd, rd


def alt(r: Rig) -> float:
    return r.ex.vehicle_state().alt_rel_home_m


def east_north(r: Rig) -> tuple[float, float]:
    v = r.ex.vehicle_state()
    return PLANE.xy(v.lat, v.lon)


async def awaiting(r: Rig, limit: float = 240) -> dict[str, Any]:
    await r.until(lambda: r.ex.hold_reason == "inspect_await", limit)
    done = events(r, "inspect.done")
    assert len(done) == 1
    return done[0]


def ground_floor(r: Rig, pack: FloorPack) -> tuple[float, float]:
    """(auto, lowest without the laser) over the target, above home."""
    c = r.ex.charter
    assert c is not None
    lat, lon = at(*TARGET)
    f = floor_at(
        pack,
        lat,
        lon,
        flight_date=r.clock.utc_now().date(),
        laser=None,
        start_alt_amsl_m=c.home.alt_amsl_m + 60,
    )
    return f.floor_auto_amsl_m - c.home.alt_amsl_m, f.floor_lowest_amsl_m - c.home.alt_amsl_m


# --- the steps ------------------------------------------------------------------------------


async def test_without_a_map_it_looks_from_its_height_then_waits_and_flies_back() -> None:
    async with rig() as r:
        cmd, rounds = await start(r, alt=40)
        done = await awaiting(r)
        assert done["end"] == "floor" and done["steps"] == 0 and not done["clear"]
        assert len(done["still_ids"]) == 4  # 2 rounds x (main + tele)
        assert [s for _, s in rounds.asked] == [0, 0]
        assert {c for c, _ in rounds.asked} == {cmd.command_id}
        assert math.dist(east_north(r), TARGET) < 3 and abs(alt(r) - 40) < 2
        assert "done" in r.up.stages(cmd.command_id)
        snap = r.ex.snapshot()
        assert snap.inspect is not None and snap.inspect.phase == "await"
        assert snap.inspect.floor_reason == "no_map"
        r.ex.on_inspect_outcome(
            InspectOutcome(command_id=cmd.command_id, outcome="resume"), new_id()
        )
        await r.until(lambda: r.ex.state == S.TRANSIT, 5)
        assert r.ex.snapshot().inspect is None
        await r.until(lambda: r.ex.state == S.HOLD, 120)
        assert math.dist(east_north(r), (0, 0)) < 3  # back where it broke off
        assert r.ex.resume_point is None
        ended = [d["cause"] for d in events(r, "executor.inspect_ended")]
        assert ended == ["returned"]
        assert not events(r, "inspect.aborted")


async def test_the_candidates_own_cylinder_is_suspended_until_the_return() -> None:
    async with rig() as r:
        await r.install(over())
        await r.takeoff(40)
        r.ex.capture_round = Rounds()
        cid = new_id()
        lat, lon = at(*TARGET)
        own = Keepout(
            keepout_id=new_id(), lat=lat, lon=lon, radius_m=32, source="candidate", source_id=cid
        )
        r.ex.on_keepouts(KeepoutInstall(install_seq=1, keepouts=[own]), new_id())
        cmd = await r.send(order(r, candidate_id=cid), 0.5)
        assert r.ex.state == S.INSPECT, r.up.acks[-2:]
        assert own not in r.ex.keepouts_in_force()
        # The target's own detections make no cylinder; another person's does.
        r.ex.on_detections([det(62, 0), det(150, 100)])
        assert len(r.ex.local_keepouts) == 1
        await awaiting(r)
        r.ex.on_inspect_outcome(
            InspectOutcome(command_id=cmd.command_id, outcome="rejected"), new_id()
        )
        assert own in r.ex.keepouts_in_force()


async def test_meadow_steps_down_to_30_m_and_asks_then_stays_when_no_answer() -> None:
    pack = meadow()
    async with rig() as r:
        cmd, rounds = await start(r, pack=pack)
        auto, lowest = ground_floor(r, pack)
        assert auto == pytest.approx(30.0) and lowest < 28
        lows: list[float] = []
        rates: list[float] = []
        last = (r.clock.monotonic(), alt(r))

        def watch() -> bool:
            nonlocal last
            now, a = r.clock.monotonic(), alt(r)
            if now - last[0] >= 1.0:
                rates.append((last[1] - a) / (now - last[0]))
                last = (now, a)
            lows.append(a)
            return bool(events(r, "inspect.lower_offer"))

        await r.until(watch, 200)
        offer = events(r, "inspect.lower_offer")[0]
        assert offer["command_id"] == str(cmd.command_id)
        assert offer["now_above_ground_m"] == pytest.approx(30, abs=1.6)
        assert offer["lowest_above_ground_m"] == pytest.approx(lowest, abs=0.2)
        assert r.ex.snapshot().inspect is not None
        assert r.ex.snapshot().inspect.phase == "lower_offer"  # type: ignore[union-attr]
        assert min(lows) > auto - 1.6  # never below the automatic floor without the yes
        assert max(rates) <= DESCENT_MAX_MS + 0.3
        await r.run(LOWER_OFFER_S - 5)
        assert r.ex.state == S.INSPECT  # still asking
        done = await awaiting(r, 20)
        assert done["end"] == "lower_declined" and done["steps"] in (3, 4)
        assert done["lowest_above_ground_m"] == pytest.approx(30, abs=1.6)
        assert sorted({s for _, s in rounds.asked}) == list(range(done["steps"] + 1))


async def test_the_pilots_yes_goes_down_to_the_lowest_floor() -> None:
    pack = meadow()
    async with rig() as r:
        cmd, _ = await start(r, pack=pack)
        _, lowest = ground_floor(r, pack)
        await r.until(lambda: bool(events(r, "inspect.lower_offer")), 200)
        r.ex.on_inspect_lower(InspectLower(command_id=cmd.command_id), new_id())
        lows: list[float] = []

        def watch() -> bool:
            lows.append(alt(r))
            return r.ex.hold_reason == "inspect_await"

        await r.until(watch, 120)
        done = events(r, "inspect.done")[0]
        assert done["end"] == "floor"
        assert min(lows) > lowest - 1.6  # never below the lowest floor
        assert alt(r) < 30 - 5
        assert len(events(r, "inspect.lower_offer")) == 1


async def test_the_pilots_no_ends_the_look_at_once_where_it_hovers() -> None:
    pack = meadow()
    async with rig() as r:
        cmd, _ = await start(r, pack=pack)
        auto, _ = ground_floor(r, pack)
        await r.until(lambda: bool(events(r, "inspect.lower_offer")), 200)
        asked_at, hover = r.clock.monotonic(), alt(r)
        r.ex.on_inspect_stay(InspectStay(command_id=cmd.command_id), no := new_id())
        assert "done" in r.up.stages(no)
        assert [e["command_id"] for e in events(r, "executor.inspect_stay")] == [
            str(cmd.command_id)
        ]
        lows: list[float] = []

        def watch() -> bool:
            lows.append(alt(r))
            return r.ex.hold_reason == "inspect_await"

        await r.until(watch, 10)  # not the 60 s of no answer
        assert r.clock.monotonic() - asked_at < 5
        done = events(r, "inspect.done")
        assert len(done) == 1 and done[0]["end"] == "lower_declined"
        assert done[0]["lowest_above_ground_m"] == pytest.approx(30, abs=1.6)
        assert min(lows) > min(hover, auto) - 1.6  # never lower for a no
        await r.run(LOWER_OFFER_S + 5)  # the old answer time runs out: nothing more
        assert r.ex.hold_reason == "inspect_await"
        assert alt(r) > min(hover, auto) - 1.6
        assert len(events(r, "inspect.lower_offer")) == 1
        assert len(events(r, "inspect.done")) == 1


async def test_a_no_before_the_question_or_after_a_yes_is_stale() -> None:
    async with rig() as r:
        cmd, _ = await start(r, pack=meadow())
        await r.until(lambda: alt(r) < 55, 120)
        r.ex.on_inspect_stay(InspectStay(command_id=cmd.command_id), early := new_id())
        assert r.up.codes(early) == ("ONB_STALE_ORDER",)
        await r.until(lambda: bool(events(r, "inspect.lower_offer")), 200)  # still asked
        r.ex.on_inspect_stay(InspectStay(command_id=new_id()), other := new_id())
        assert r.up.codes(other) == ("ONB_STALE_ORDER",)  # another look's
        r.ex.on_inspect_lower(InspectLower(command_id=cmd.command_id), yes := new_id())
        assert "done" in r.up.stages(yes)
        await r.run(1)
        r.ex.on_inspect_stay(InspectStay(command_id=cmd.command_id), late := new_id())
        assert r.up.codes(late) == ("ONB_STALE_ORDER",)  # the yes stands
        done = await awaiting(r, 120)
        assert done["end"] == "floor" and alt(r) < 30 - 5


async def test_a_no_with_a_yes_at_the_same_time_keeps_the_height() -> None:
    pack = meadow()
    async with rig() as r:
        await start(r, pack=pack)
        auto, _ = ground_floor(r, pack)
        await r.until(lambda: bool(events(r, "inspect.lower_offer")), 200)
        hover = alt(r)
        a = r.ex.active
        assert a is not None
        handler: Any = a.handler
        handler.steps.lower()  # both before the step wakes up: the no wins
        handler.steps.stay()
        lows: list[float] = []

        def watch() -> bool:
            lows.append(alt(r))
            return r.ex.hold_reason == "inspect_await"

        await r.until(watch, 10)
        assert events(r, "inspect.done")[0]["end"] == "lower_declined"
        assert min(lows) > min(hover, auto) - 1.6


async def test_a_lower_without_a_look_flying_is_stale() -> None:
    async with rig() as r:
        await r.install(over())
        r.ex.on_inspect_lower(InspectLower(command_id=new_id()), ref := new_id())
        assert r.up.codes(ref) == ("ONB_STALE_ORDER",)
        r.ex.on_inspect_outcome(
            InspectOutcome(command_id=new_id(), outcome="resume"), ref2 := new_id()
        )
        assert r.up.codes(ref2) == ("ONB_STALE_ORDER",)


async def test_a_clear_picture_ends_the_steps_at_once() -> None:
    pack = meadow()
    async with rig() as r:
        rounds = Rounds(lambda _s: [det(61, 1, score=0.9, px=120)])
        await start(r, pack=pack, rounds=rounds)
        done = await awaiting(r)
        assert done["end"] == "clear" and done["clear"] and done["steps"] == 0
        assert abs(alt(r) - 60) < 2


async def test_the_drone_moves_over_the_refined_estimate() -> None:
    pack = meadow()
    async with rig() as r:
        rounds = Rounds(lambda _s: [det(68, 0, score=0.9, px=40, sigma=0.5)])
        await start(r, pack=pack, rounds=rounds)
        await r.until(lambda: alt(r) < 52, 120)
        assert east_north(r)[0] > 65  # over the target as seen, not the order's point


async def test_at_most_steps_max_steps() -> None:
    pack = meadow()
    async with rig() as r:
        await r.install(over().model_copy(update={"alt_max_agl_m": 120.0}))
        await r.takeoff(115)
        r.ex.floor_pack = pack
        r.ex.capture_round = Rounds()
        a0 = alt(r)
        await r.send(order(r), 0.5)
        done = await awaiting(r, 400)
        assert done["end"] == "steps_max" and done["steps"] == 8
        # 8 steps of 10 m, each ending up to 0.7 m above its height (never below)
        assert a0 - 80 - 0.5 < alt(r) < a0 - 80 + 8 * 0.7


async def test_the_time_limit_ends_the_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inspect_steps, "INSPECT_MAX_S", 25.0)
    async with rig() as r:
        await start(r, pack=meadow())
        done = await awaiting(r, 60)
        assert done["end"] == "timeout"


@pytest.mark.parametrize(
    ("spare_s", "end"), [(inspect_steps.DETECT_SPARE_MAX_S, "steps_max"), (0.0, "timeout")]
)
async def test_the_rounds_of_stills_do_not_count_in_the_time_limit(
    monkeypatch: pytest.MonkeyPatch, spare_s: float, end: str
) -> None:
    """A detector on a CPU takes seconds a still (a full-resolution main and tele still
    every round): the rounds' time does not count in INSPECT_MAX_S, so a slow detector
    does not end the look early; only up to DETECT_SPARE_MAX_S of it, though."""
    monkeypatch.setattr(inspect_steps, "DETECT_SPARE_MAX_S", spare_s)

    async with rig() as r:

        class Slow(Rounds):
            async def __call__(
                self, command_id: UUID, step: int
            ) -> list[tuple[str, UUID, list[GeoDetectionMsg]]]:
                await r.clock.sleep(8)  # the detector on a main and a tele still
                return await super().__call__(command_id, step)

        await r.install(over().model_copy(update={"alt_max_agl_m": 120.0}))
        await r.takeoff(115)
        r.ex.floor_pack = meadow()
        r.ex.capture_round = Slow()
        await r.send(order(r), 0.5)
        done = await awaiting(r, 600)
        assert done["end"] == end


async def test_no_stills_at_all_fails_the_look() -> None:
    async with rig() as r:
        rounds = Rounds()
        rounds.works = False
        cmd, _ = await start(r, rounds=rounds)
        await r.until(lambda: "failed" in r.up.stages(cmd.command_id), 60)
        assert r.up.codes(cmd.command_id) == ("ONB_CAMERA_UNAVAILABLE",)
        assert r.ex.state == S.HOLD and r.ex.hold_reason == "skill_done"
        assert [d["reason"] for d in events(r, "inspect.aborted")] == ["failed"]
        assert r.ex.snapshot().inspect is None


async def test_a_new_cylinder_around_the_drone_stops_the_steps() -> None:
    pack = meadow()
    async with rig() as r:
        await start(r, pack=pack)
        await r.until(lambda: alt(r) < 55, 120)
        # Someone else under the drone (outside the target's look radius from the
        # target, but the drone stands over them): no deeper into their cylinder.
        e, n = east_north(r)
        lat, lon = at(e, n + 20)
        k = Keepout(
            keepout_id=new_id(),
            lat=lat,
            lon=lon,
            radius_m=35,
            source="candidate",
            source_id=new_id(),
        )
        r.ex.on_keepouts(KeepoutInstall(install_seq=5, keepouts=[k]), new_id())
        done = await awaiting(r, 120)
        assert done["end"] == "keepout"


async def test_the_wait_ends_after_two_minutes_and_flies_back_climbing_first() -> None:
    pack = meadow()
    async with rig() as r:
        await start(r, pack=pack)
        await r.until(lambda: bool(events(r, "inspect.lower_offer")), 200)
        await awaiting(r, 80)
        assert alt(r) < 35
        await r.run(AWAIT_DECISION_S - 2)
        assert r.ex.hold_reason == "inspect_await"
        await r.until(lambda: r.ex.state == S.INSPECT, 5)  # the return's climb
        assert r.ex.active is not None and r.ex.active.skill == INTERNAL_INSPECT_CLIMB
        start_xy = east_north(r)
        await r.until(lambda: r.ex.state == S.TRANSIT, 60)
        assert alt(r) > 58 and math.dist(east_north(r), start_xy) < 3  # straight up
        await r.until(lambda: r.ex.state == S.HOLD, 120)
        assert math.dist(east_north(r), (0, 0)) < 3
        assert events(r, "executor.inspect_await_timeout")


async def test_confirmed_hovers_over_the_find() -> None:
    async with rig() as r:
        cmd, _ = await start(r, alt=40)
        await awaiting(r)
        r.ex.on_inspect_outcome(
            InspectOutcome(command_id=cmd.command_id, outcome="confirmed"), new_id()
        )
        await r.run(5)
        assert r.ex.state == S.HOLD and r.ex.hold_reason == "skill_done"
        assert math.dist(east_north(r), TARGET) < 3
        assert r.ex.snapshot().inspect is None
        assert [d["cause"] for d in events(r, "executor.inspect_ended")] == ["confirmed"]


async def test_a_decision_while_still_flying_ends_the_steps() -> None:
    async with rig() as r:
        cmd, _ = await start(r, pack=meadow())
        await r.until(lambda: alt(r) < 55, 120)
        r.ex.on_inspect_outcome(
            InspectOutcome(command_id=cmd.command_id, outcome="rejected"), new_id()
        )
        assert "done" in r.up.stages(cmd.command_id)
        await r.until(lambda: r.ex.state == S.TRANSIT, 60)
        await r.until(lambda: r.ex.state == S.HOLD, 120)
        assert math.dist(east_north(r), (0, 0)) < 3 and alt(r) > 58


async def test_one_look_at_a_time() -> None:
    async with rig() as r:
        await start(r, alt=40)
        await awaiting(r)
        second = await r.send(order(r), 0.2)
        assert r.up.codes(second.command_id) == ("INSPECT_BUSY",)


async def test_another_order_while_waiting_ends_the_look() -> None:
    async with rig() as r:
        await start(r, alt=40)
        await awaiting(r)
        lat, lon = at(0, 50)
        goto = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}), 0.5)
        assert "running" in r.up.stages(goto.command_id)
        assert [d["reason"] for d in events(r, "inspect.aborted")] == ["override"]
        assert r.ex.snapshot().inspect is None


async def test_stills_while_waiting_keep_the_wait() -> None:
    async with rig() as r:
        await start(r, alt=40)
        await awaiting(r)
        r.ex.camera = _camera()
        stills = await r.send(r.order("capture_stills", {"count": 1}), 2)
        assert "done" in r.up.stages(stills.command_id)
        assert r.ex.hold_reason == "inspect_await" and r.ex.snapshot().inspect is not None


def _camera() -> Any:
    async def cam(command_id: UUID) -> UUID | None:
        return new_id()

    return cam


async def test_stop_during_the_look_ends_it_without_the_wait() -> None:
    async with rig() as r:
        await start(r, pack=meadow())
        await r.run(3)
        r.ex.on_override(r.notice("stop"))
        await r.run(0.5)
        assert r.ex.state == S.HELD
        assert [d["reason"] for d in events(r, "inspect.aborted")] == ["stop"]
        assert r.ex.snapshot().inspect is None


async def test_fly_home_is_allowed_from_inspect() -> None:
    async with rig() as r:
        await start(r, pack=meadow())
        await r.run(2)
        home = await r.send(r.order("return_home", {}), 0.5)
        assert "running" in r.up.stages(home.command_id) or "accepted" in r.up.stages(
            home.command_id
        )
        assert r.ex.state == S.RTL
        assert [d["reason"] for d in events(r, "inspect.aborted")] == ["override"]


async def test_other_orders_are_refused_while_the_look_flies() -> None:
    async with rig() as r:
        await start(r, pack=meadow())
        lat, lon = at(0, 50)
        goto = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}), 0.2)
        assert r.up.codes(goto.command_id) == ("ONB_INVALID_TRANSITION",)
        assert r.ex.state == S.INSPECT


async def test_link_loss_ends_the_look_and_climbs_straight_up() -> None:
    async with rig() as r:
        await start(r, pack=meadow())
        # Lost exactly as a step ends in LOITER: the flight controller has taken it, no
        # HEARTBEAT shows it yet. The climb cuts that switch short; the LOITER that shows
        # then is ours (``Authority.late``), never a takeover, and the climb asks for GUIDED.
        await r.until(lambda: alt(r) < 45, 120)
        cut: list[CopterMode] = []
        record = r.ex.auth.sent_and_cancelled

        def cancelled(mode: CopterMode, *args: Any) -> None:
            cut.append(mode)
            record(mode, *args)

        r.ex.auth.sent_and_cancelled = cancelled  # type: ignore[method-assign]
        send = r.ex.vehicle.set_mode_ordered

        async def answered(mode: CopterMode, **kw: Any) -> Any:
            out = await send(mode, **kw)
            if mode == CopterMode.LOITER and r.link:  # accepted: the link is lost now
                r.link = False
                r.ex.on_link(False, None)
                lost = r.clock.monotonic_ms() - round(LINK_LOSS_DETECT_S * 1000) - 1
                r.ex.link_last_inbound_ms = lost
            return out

        r.ex.vehicle.set_mode_ordered = answered  # type: ignore[method-assign]
        await r.until(lambda: bool(events(r, "inspect.aborted")), 30)
        assert cut == [CopterMode.LOITER]  # the race: the step's LOITER cut short
        assert events(r, "inspect.aborted")[0]["reason"] == "link_loss"
        await r.run(0.5)
        assert r.ex.state == S.INSPECT
        assert r.ex.active is not None and r.ex.active.skill == INTERNAL_INSPECT_CLIMB
        here = east_north(r)
        await r.until(lambda: alt(r) > 58, 30)
        assert math.dist(east_north(r), here) < 3
        await r.until(lambda: r.ex.state == S.HOLD, 30)
        assert r.ex.hold_reason == "link_loss"
        await r.until(lambda: r.ex.state == S.RTL, 60)


async def test_the_ai_going_quiet_does_not_stop_the_look() -> None:
    async with rig() as r:
        await r.install(
            over().model_copy(update={"on_ai_silent": "hold", "ai_silent_timeout_s": 5})
        )
        await r.takeoff(40)
        r.ex.capture_round = Rounds()
        cmd = await r.send(order(r), 0.5)
        r.commander = False
        await awaiting(r)
        await r.run(10)
        assert r.ex.hold_reason == "inspect_await"
        r.ex.on_inspect_outcome(
            InspectOutcome(command_id=cmd.command_id, outcome="resume"), new_id()
        )
        await r.run(1)
        assert r.ex.snapshot().inspect is None


class FakeLaser:
    def __init__(self, surface: float) -> None:
        self.surface = surface
        self.on = True

    def state(self, now_ms: int) -> LaserState:
        if not self.on:
            return LaserState(confirmed=False, surface_amsl_m=None, min_range_m=0.0)
        return LaserState(confirmed=True, surface_amsl_m=self.surface, min_range_m=0.0)


async def test_the_laser_lost_below_the_maps_floor_climbs_back() -> None:
    pack = meadow()
    async with rig() as r:
        from onboard_helpers import HOME_ALT

        laser = FakeLaser(HOME_ALT)
        r.ex.laser = laser  # type: ignore[assignment]
        cmd, _ = await start(r, pack=pack, alt_min_agl_m=6, default_alt_agl_m=40)
        _, no_laser = ground_floor(r, pack)
        await r.until(lambda: bool(events(r, "inspect.lower_offer")), 200)
        r.ex.on_inspect_lower(InspectLower(command_id=cmd.command_id), new_id())
        await r.until(lambda: alt(r) < no_laser - 3, 120)
        laser.on = False
        await r.until(lambda: bool(events(r, "inspect.laser_lost")), 5)
        done = await awaiting(r, 60)
        assert done["end"] == "laser_lost"
        assert alt(r) > no_laser - 1.6


async def test_the_laser_lost_while_waiting_climbs_and_waits_again() -> None:
    pack = meadow()
    async with rig() as r:
        from onboard_helpers import HOME_ALT

        laser = FakeLaser(HOME_ALT)
        r.ex.laser = laser  # type: ignore[assignment]
        cmd, _ = await start(r, pack=pack, alt_min_agl_m=6, default_alt_agl_m=40)
        _, no_laser = ground_floor(r, pack)
        await r.until(lambda: bool(events(r, "inspect.lower_offer")), 200)
        r.ex.on_inspect_lower(InspectLower(command_id=cmd.command_id), new_id())
        await awaiting(r, 120)
        assert alt(r) < no_laser - 3
        laser.on = False
        await r.until(lambda: r.ex.state == S.INSPECT, 5)
        assert events(r, "inspect.laser_lost")
        await r.until(lambda: r.ex.hold_reason == "inspect_await", 30)
        assert alt(r) > no_laser - 1.6
        assert r.ex.snapshot().inspect is not None


async def test_the_side_look_in_live_without_the_permit_also_waits() -> None:
    async with rig() as r:
        await r.install(charter(allowed_skills=SKILLS, operation_mode="live"))
        await r.takeoff(40)
        r.ex.camera = _camera()
        lat, lon = at(150, 0)
        cmd = await r.send(
            r.order(
                "inspect_detection",
                {"candidate_id": str(new_id()), "lat": lat, "lon": lon, "target_radius_m": 35},
            ),
            0.5,
        )
        assert r.ex.state == S.INSPECT
        await r.until(lambda: r.ex.hold_reason == "inspect_await", 120)
        assert events(r, "inspect.done")[0]["end"] == "side_look"
        assert "done" in r.up.stages(cmd.command_id)


# --- the charter: operation mode and map pack ---------------------------------------------


async def test_a_charter_that_does_not_fit_the_profile_is_refused() -> None:
    async with rig(options={"profile": "sim"}) as r:
        await r.ex.on_charter(_install(charter(operation_mode="live")))
        assert r.up.charter_acks[-1].reason_codes == ["ONB_OPERATION_MODE_MISMATCH"]
        await r.ex.on_charter(_install(charter(operation_mode="sim")))
        assert r.up.charter_acks[-1].accepted
    async with rig(options={"profile": "field"}) as r:
        await r.ex.on_charter(_install(charter(operation_mode="sim")))
        assert r.up.charter_acks[-1].reason_codes == ["ONB_OPERATION_MODE_MISMATCH"]


async def test_a_charter_naming_a_pack_that_is_not_onboard_is_refused(tmp_path: Any) -> None:
    named = {"floor_pack_id": "meadow", "floor_pack_sha256": "a" * 64}
    async with rig(options={"floor_dir": tmp_path}) as r:
        await r.ex.on_charter(_install(charter(**named)))
        assert r.up.charter_acks[-1].reason_codes == ["ONB_FLOOR_PACK_MISMATCH"]
        assert r.ex.floor_pack is None
        pack = meadow()
        r.ex.floor_pack = pack  # already read: kept when it is the one named
        await r.ex.on_charter(
            _install(charter(floor_pack_id="meadow", floor_pack_sha256=pack.sha256))
        )
        assert r.up.charter_acks[-1].accepted and r.ex.floor_pack is pack
        await r.ex.on_charter(_install(charter()))
        assert r.up.charter_acks[-1].accepted and r.ex.floor_pack is None
    async with rig() as r:  # no folder for packs at all
        await r.ex.on_charter(_install(charter(**named)))
        assert r.up.charter_acks[-1].reason_codes == ["ONB_FLOOR_PACK_MISMATCH"]


def _install(c: Any) -> Any:
    from sd_schemas.link import CharterInstall

    return CharterInstall(charter=c)


def test_limit_action_kind_has_inspect_climb() -> None:
    LimitAction("L-LINK", "inspect_climb", "link_loss", "x", {"alt_m": 1.0})


# --- the review of TN-6: climbs that are never cut short, the watches that stay -------------


def _cylinder(r: Rig, east: float, north: float, radius: float, seq: int) -> Keepout:
    """Someone else's cylinder (a ground team): not the target's, never suspended."""
    lat, lon = at(east, north)
    k = Keepout(
        keepout_id=new_id(),
        lat=lat,
        lon=lon,
        radius_m=radius,
        source="candidate",
        source_id=new_id(),
    )
    r.ex.on_keepouts(KeepoutInstall(install_seq=seq, keepouts=[k]), new_id())
    return k


def _climbs(r: Rig) -> list[float]:
    return [d["climb_to_m"] for d in events(r, "executor.keepout_hold") if "climb_to_m" in d]


async def test_a_cylinder_around_the_drone_on_a_leg_climbs_over_it() -> None:
    # The reflex on the approach leg: no way out (inside), so straight up over the
    # cylinder through ``InspectHandler.climb`` -> ``StepLook.climb``, and the steps end.
    async with rig() as r:
        await start(r, pack=meadow(), alt_max_agl_m=120)
        await r.run(3)
        assert r.ex.state == S.INSPECT
        e, n = east_north(r)
        _cylinder(r, e, n, 80, seq=7)  # top 80 + 10 m above home: higher than the drone
        await r.until(lambda: bool(_climbs(r)), 5)
        assert _climbs(r) == [90.0]
        here = (e, n)
        done = await awaiting(r, 120)
        assert done["end"] == "keepout"
        assert alt(r) > 88 and math.dist(east_north(r), here) < 15  # up, not on
        assert r.ex.hold_reason == "inspect_await"  # the look's own wait, not a limit


async def test_a_cylinder_around_the_drone_going_down_stops_and_climbs() -> None:
    async with rig() as r:
        await start(r, pack=meadow())
        await r.until(lambda: alt(r) < 45, 120)  # down from 50 to 40
        e, n = east_north(r)
        _cylinder(r, e, n + 5, 40, seq=7)  # top 40 + 10 m: above the drone
        lows: list[float] = []

        def watch() -> bool:
            lows.append(alt(r))
            return r.ex.hold_reason == "inspect_await"

        await r.until(watch, 60)
        done = events(r, "inspect.done")[0]
        assert done["end"] == "keepout"
        assert _climbs(r) == [50.0]
        assert min(lows) > 45 - 2.0  # stopped where it was, no deeper into the cylinder
        assert alt(r) > 49


async def test_a_cylinder_around_the_drone_while_it_hovers_at_a_step_climbs() -> None:
    async with rig() as r:
        await start(r, pack=meadow())

        def hovering_low() -> bool:
            p = r.ex.snapshot().inspect
            return p is not None and p.step == 2 and p.phase == "step" and alt(r) < 41

        await r.until(hovering_low, 120)  # the settle and the stills at 40 m
        e, n = east_north(r)
        _cylinder(r, e + 10, n, 45, seq=7)  # top 45 + 10 m
        done = await awaiting(r, 60)
        assert done["end"] == "keepout"
        assert _climbs(r) == [55.0] and alt(r) > 54


async def test_a_decision_during_the_laser_climb_while_waiting_lets_it_finish() -> None:
    pack = meadow()
    async with rig() as r:
        from onboard_helpers import HOME_ALT

        laser = FakeLaser(HOME_ALT)
        r.ex.laser = laser  # type: ignore[assignment]
        cmd, _ = await start(r, pack=pack, alt_min_agl_m=6, default_alt_agl_m=40)
        _, no_laser = ground_floor(r, pack)
        await r.until(lambda: bool(events(r, "inspect.lower_offer")), 200)
        r.ex.on_inspect_lower(InspectLower(command_id=cmd.command_id), new_id())
        await awaiting(r, 120)
        assert alt(r) < no_laser - 3
        laser.on = False
        await r.until(lambda: r.ex.state == S.INSPECT, 5)  # the climb to the map's floor
        r.ex.on_inspect_outcome(
            InspectOutcome(command_id=cmd.command_id, outcome="confirmed"), ref := new_id()
        )
        assert "done" in r.up.stages(ref)
        assert events(r, "executor.inspect_outcome_after_climb")
        await r.run(0.5)
        assert r.ex.active is not None and r.ex.active.skill == INTERNAL_INSPECT_CLIMB
        await r.until(lambda: r.ex.state == S.HOLD, 30)
        assert alt(r) > no_laser - 1.6  # up at the floor, not left below it
        assert r.ex.hold_reason == "skill_done" and r.ex.snapshot().inspect is None
        assert [d["cause"] for d in events(r, "executor.inspect_ended")] == ["confirmed"]


async def test_a_decision_during_the_steps_laser_climb_lets_it_finish() -> None:
    pack = meadow()
    async with rig() as r:
        from onboard_helpers import HOME_ALT

        laser = FakeLaser(HOME_ALT)
        r.ex.laser = laser  # type: ignore[assignment]
        cmd, _ = await start(r, pack=pack, alt_min_agl_m=6, default_alt_agl_m=40)
        _, no_laser = ground_floor(r, pack)
        await r.until(lambda: bool(events(r, "inspect.lower_offer")), 200)
        r.ex.on_inspect_lower(InspectLower(command_id=cmd.command_id), new_id())
        await r.until(lambda: alt(r) < no_laser - 3, 120)
        laser.on = False
        await r.until(lambda: bool(events(r, "inspect.laser_lost")), 5)
        r.ex.on_inspect_outcome(
            InspectOutcome(command_id=cmd.command_id, outcome="confirmed"), new_id()
        )
        assert r.ex.state == S.INSPECT  # the look's climb goes on
        assert r.ex.active is not None and r.ex.active.command_id == cmd.command_id
        await r.until(lambda: bool(events(r, "inspect.done")), 60)  # then the decision
        assert events(r, "inspect.done")[0]["end"] == "laser_lost"
        assert alt(r) > no_laser - 1.6
        await r.run(1)
        assert r.ex.hold_reason == "skill_done" and r.ex.snapshot().inspect is None
        assert [d["cause"] for d in events(r, "executor.inspect_ended")] == ["confirmed"]


async def test_the_laser_lost_over_a_confirmed_find_climbs_to_the_floor() -> None:
    pack = meadow()
    async with rig() as r:
        from onboard_helpers import HOME_ALT

        laser = FakeLaser(HOME_ALT)
        r.ex.laser = laser  # type: ignore[assignment]
        cmd, _ = await start(r, pack=pack, alt_min_agl_m=6, default_alt_agl_m=40)
        _, no_laser = ground_floor(r, pack)
        await r.until(lambda: bool(events(r, "inspect.lower_offer")), 200)
        r.ex.on_inspect_lower(InspectLower(command_id=cmd.command_id), new_id())
        await awaiting(r, 120)
        r.ex.on_inspect_outcome(
            InspectOutcome(command_id=cmd.command_id, outcome="confirmed"), new_id()
        )
        await r.run(5)
        assert r.ex.hold_reason == "skill_done" and r.ex.snapshot().inspect is None
        assert alt(r) < no_laser - 3  # hovering low over the find, with the laser
        here = east_north(r)
        laser.on = False
        await r.until(lambda: r.ex.state == S.INSPECT, 5)
        assert events(r, "executor.find_laser_lost")
        await r.until(lambda: r.ex.state == S.HOLD, 30)
        assert r.ex.hold_reason == "skill_done"
        assert alt(r) > no_laser - 1.6 and math.dist(east_north(r), here) < 3


async def test_another_order_over_a_confirmed_find_ends_the_laser_watch() -> None:
    async with rig() as r:
        cmd, _ = await start(r, alt=40)
        await awaiting(r)
        r.ex.on_inspect_outcome(
            InspectOutcome(command_id=cmd.command_id, outcome="confirmed"), new_id()
        )
        await r.run(1)
        assert r.ex._find_watch is not None
        lat, lon = at(0, 50)
        await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}), 0.5)
        assert r.ex._find_watch is None


class _FailsFrom(Rounds):
    """The cameras stop working from step ``step``."""

    def __init__(self, step: int) -> None:
        super().__init__()
        self.fail_from = step

    async def __call__(
        self, command_id: UUID, step: int
    ) -> list[tuple[str, UUID, list[GeoDetectionMsg]]]:
        self.works = step < self.fail_from
        return await super().__call__(command_id, step)


async def test_a_failed_look_low_over_the_person_climbs_back_first() -> None:
    async with rig() as r:
        cmd, _ = await start(r, pack=meadow(), rounds=_FailsFrom(2))
        await r.until(lambda: "failed" in r.up.stages(cmd.command_id), 120)
        assert r.up.codes(cmd.command_id) == ("ONB_CAMERA_UNAVAILABLE",)
        assert [d["reason"] for d in events(r, "inspect.aborted")] == ["failed"]
        assert alt(r) < 42  # at the third step, 40 m
        here = east_north(r)
        await r.run(0.5)
        assert r.ex.state == S.INSPECT
        assert r.ex.active is not None and r.ex.active.skill == INTERNAL_INSPECT_CLIMB
        await r.until(lambda: r.ex.state == S.HOLD, 60)
        assert r.ex.hold_reason == "skill_done"
        assert alt(r) > 58 and math.dist(east_north(r), here) < 3  # straight up to 60 m
        assert r.ex.snapshot().inspect is None


async def test_the_laser_lost_mid_descent_stops_at_the_maps_floor() -> None:
    pack = meadow(chm_dm=40)  # 4 m bushes: the map's floor without the laser at ~24.5 m
    async with rig() as r:
        from onboard_helpers import HOME_ALT

        laser = FakeLaser(HOME_ALT)
        r.ex.laser = laser  # type: ignore[assignment]
        cmd, _ = await start(r, pack=pack, alt_min_agl_m=6, default_alt_agl_m=40)
        _, no_laser = ground_floor(r, pack)
        await r.until(lambda: bool(events(r, "inspect.lower_offer")), 200)
        r.ex.on_inspect_lower(InspectLower(command_id=cmd.command_id), new_id())
        # Going down, still above the map's floor, with a goal below it.
        await r.until(lambda: alt(r) < no_laser + 1.0, 120)
        laser.on = False
        lows: list[float] = []

        def watch() -> bool:
            lows.append(alt(r))
            return r.ex.hold_reason == "inspect_await"

        await r.until(watch, 60)
        assert events(r, "inspect.done")[0]["end"] == "laser_lost"
        assert min(lows) > no_laser - 0.4  # stopped at once, not first 1 m below


async def test_the_laser_lost_while_the_cameras_work_climbs_at_once() -> None:
    pack = meadow()
    async with rig() as r:
        from onboard_helpers import HOME_ALT

        laser = FakeLaser(HOME_ALT)
        r.ex.laser = laser  # type: ignore[assignment]
        _, no_laser = 0.0, 0.0
        busy: list[bool] = [False]

        class Slow(Rounds):
            async def __call__(
                self, command_id: UUID, step: int
            ) -> list[tuple[str, UUID, list[GeoDetectionMsg]]]:
                if alt(r) < no_laser - 3:
                    busy[0] = True
                    await r.clock.sleep(15)  # the cameras take their time
                    busy[0] = False
                return await super().__call__(command_id, step)

        cmd, _ = await start(r, pack=pack, rounds=Slow(), alt_min_agl_m=6, default_alt_agl_m=40)
        _, no_laser = ground_floor(r, pack)
        await r.until(lambda: bool(events(r, "inspect.lower_offer")), 200)
        r.ex.on_inspect_lower(InspectLower(command_id=cmd.command_id), new_id())
        await r.until(lambda: busy[0], 120)
        laser.on = False
        await r.until(lambda: bool(events(r, "inspect.laser_lost")), 2)  # not after 15 s
        done = await awaiting(r, 60)
        assert done["end"] == "laser_lost" and alt(r) > no_laser - 1.6


async def test_a_yes_before_the_question_is_stale() -> None:
    async with rig() as r:
        cmd, _ = await start(r, pack=meadow())
        await r.until(lambda: alt(r) < 55, 120)
        r.ex.on_inspect_lower(InspectLower(command_id=cmd.command_id), early := new_id())
        assert r.up.codes(early) == ("ONB_STALE_ORDER",)
        await r.until(lambda: bool(events(r, "inspect.lower_offer")), 200)  # still asked
        r.ex.on_inspect_lower(InspectLower(command_id=cmd.command_id), yes := new_id())
        assert "done" in r.up.stages(yes)


async def test_a_missing_pack_is_told_without_the_error_text(tmp_path: Any) -> None:
    async with rig(options={"floor_dir": tmp_path}) as r:
        named = {"floor_pack_id": "meadow", "floor_pack_sha256": "a" * 64}
        await r.ex.on_charter(_install(charter(**named)))
        missing = events(r, "executor.floor_pack_missing")
        assert missing and missing[-1]["error"] == "FloorPackError"  # the kind only
        assert str(tmp_path) not in str(missing[-1])
