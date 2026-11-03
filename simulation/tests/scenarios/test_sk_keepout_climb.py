"""B1 on ArduCopter SITL: over a person with no safe way out.

An inspection (pilot-supervised, spec titta-narmare) still climbs; every other order
leaves by the shortest way out at its height (owner decision 2026-11-03: never climb and
hover over a possible person).

* ``test_no_way_out_over_a_person_climbs_straight_up_and_hovers``: on the leg to an
  inspection point a person's cylinder comes around the drone, its leg heading deeper,
  and the gate refuses every way out (here as if it ate the return reserve). The drone
  climbs straight up to the cylinder's top (its radius above home) plus
  ``KEEPOUT_CLIMB_MARGIN_M``, within the charter's height, and hovers there in LOITER:
  the order fails ``KEEPOUT_VIOLATION`` into the keep-out hover (``hold_reason``
  ``limit``), and the operator is told at once (``executor.keepout_hold`` with
  ``inside`` and ``climb_to_m``). It never goes deeper than braking takes it, nor
  sideways. STOP then brakes and holds as always, and FLY HOME brings it home.
* ``test_a_goto_held_over_a_person_leaves_by_the_shortest_way_and_hovers``: any other
  order. A goto flies on inside a person's cylinder that came around it, and a second
  cylinder lies just ahead, closer than it can turn away from. It stops short of that one
  and leaves the one it is in by the shortest straight way out at its height (an internal
  goto that replaces the order, ``superseded`` with ``KEEPOUT_VIOLATION``; past the
  other person, not over them), hovers in LOITER outside, and the operator is told once
  (``keepout_hold`` outside, ``keepout_exit``). STOP and FLY HOME work as always.
* ``test_a_link_loss_during_the_way_out_goes_on_and_flies_home``: the same way out of a
  confirmed find's own cylinder, and SD-link lost just as it begins. The loss is noticed
  on the way out: it does not stop it (no ``internal_find_goto``). It ends in the
  keep-out hold (``limit``) outside, where the find hover starts, and flies home when
  that ends, as the link-loss rule says.
* ``test_a_link_loss_with_no_find_during_the_way_out_goes_on_and_flies_home``: the same
  with no confirmed find near, the way out slowed so it is still running at 23 s. The
  hold due then does not cut it short: it goes on out, hovers in the keep-out hold
  (``limit``), and flies home at 53 s.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Literal, NamedTuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_rig import (  # pyright: ignore[reportMissingImports]
    FIXTURE,
    fresh_db,
    migrated_db,
    passwords,
    scenario,
    sim,
)
from test_sc_emergency import stopped_and_holding  # pyright: ignore[reportMissingImports]
from test_sc_linkloss import (  # pyright: ignore[reportMissingImports]
    FIND_HOME_AT,
    HOLD_AT,
    HOME_AT,
    HOME_MODES,
    SLACK_S,
    back_and_land,
    home_m,
    north_m,
    onboard_events,
    trace,
    until_event,
)
from test_sk_inspect import (  # pyright: ignore[reportMissingImports]
    Camera,
    a_candidate,
    airborne_for_inspection,
    inspecting,
)

from sd_onboard.executor import executor as executor_module
from sd_onboard.executor.keepout_reflex import (
    INTERNAL_KEEPOUT_EXIT,
    KEEPOUT_CLIMB_MARGIN_M,
    stop_m,
)
from sd_onboard.executor.link_loss import INTERNAL_FIND_GOTO
from sd_safety import evaluate
from sd_safety.keepout import KEEPOUT_ARC_MARGIN_M
from sd_safety.targets import Plane, distance_m
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState, LinkLossPhase
from sd_schemas.gate import GateContext, GateDecision, GateRequest
from sd_schemas.keepout import INSPECT_SPEED_MS, Keepout
from sd_schemas.link import KeepoutInstall
from sd_sim.search_invariants import stop_distance_m

pytestmark = [pytest.mark.sitl, pytest.mark.slow]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

# Wide enough that its top and the margin lie above the inspection's height (about 97 m
# for a person's cylinder) and under the charter's 120 m less the altitude limit's 5 m.
PERSON_R_M = 100.0
CLIMB_TO_M = PERSON_R_M + KEEPOUT_CLIMB_MARGIN_M
ALT_TOL_M = 2.0
# ArduCopter's default climb speed in GUIDED (``WP_SPD_UP``, not set in the repo's params).
SITL_MAX_CLIMB_MS = 2.5
LOW_GOTO_ALT_M = 22.0
SIDEWAYS_M = 15.0  # braking from the leg's speed, then straight up
# The ground numbers its installs from the wall clock in ms: one hour past the
# executor's, so the ground's own resends in this test do not replace this one.
INSTALL_AHEAD = 3_600_000


def _refused(req: GateRequest, ctx: GateContext) -> GateDecision:
    return evaluate(req, ctx).model_copy(
        update={"decision": "REJECT", "reason_codes": ["ROUTE_BATTERY"]}
    )


async def _status(sc: Any, command_id: str, *want: str) -> bool:
    return (await sc.command(command_id))["status"] in want


@pytest.mark.budget_s(480)
async def test_no_way_out_over_a_person_climbs_straight_up_and_hovers(
    sim: Any, migrated_db: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await airborne_for_inspection(sc)
        p1 = next(p for p in FIXTURE["pins"] if p["id"] == "P1")
        cid = await a_candidate(sc, sc.at(float(p1["east_m"]), float(p1["north_m"])))
        Camera(sc)
        order = await inspecting(sc, cid)
        ex = sc.r.onboard.ex
        st = sc.sim.harness.state
        await sc.until(
            lambda: north_m(sc) >= 50.0 and st.ground_speed_ms > 3.0, 120, "on the leg north"
        )

        # The gate refuses every way out; a person's cylinder comes around the drone, its
        # centre just ahead on the leg.
        monkeypatch.setattr(executor_module, "evaluate", _refused)
        plane = Plane(*sc.r.home)
        e, n = plane.xy(st.lat, st.lon)
        lat, lon = plane.latlon(e, n + 10.0)
        person = Keepout(
            keepout_id=new_id(),
            lat=lat,
            lon=lon,
            radius_m=PERSON_R_M,
            source="candidate",
            source_id=new_id(),
        )
        assert sc.track[-1].alt_m < CLIMB_TO_M - ALT_TOL_M
        since = time.monotonic()
        here = (st.lat, st.lon)
        install = KeepoutInstall(
            install_seq=ex.keepout_seq + INSTALL_AHEAD, keepouts=[*ex.keepouts, person]
        )
        ex.on_keepouts(install, new_id())
        assert person in ex.keepouts  # taken, not refused as stale

        holds = await until_event(sc, "keepout_hold")
        told = holds[-1]["payload"]
        assert told.get("inside") is True, told
        assert told.get("climb_to_m") == pytest.approx(CLIMB_TO_M, abs=0.1), told
        assert not await _events(sc, "keepout_replan")
        await sc.until(lambda: _status(sc, order["id"], "failed"), 120, "the order ended")
        after = await sc.command(order["id"])
        assert "KEEPOUT_VIOLATION" in str(after), after
        await sc.until(
            lambda: abs(sc.track[-1].alt_m - CLIMB_TO_M) <= ALT_TOL_M and st.ground_speed_ms < 0.5,
            120,
            "up there and still",
        )

        async def kept_out() -> bool:  # the snapshot may still be the one before the end
            live = await sc.onboard_state()
            return live["executor_state"] == "HOLD" and live["hold_reason"] == "limit"

        await sc.until(kept_out, 5, "the keep-out hover")
        await sc.hold_for(10.0, lambda: sc.fc_mode() == "LOITER", "hovering up there")
        assert len(await _events(sc, "keepout_hold")) == 1  # told once

        track = [s for s in sc.track if s.t >= since and s.armed]
        dists = [distance_m(s.lat, s.lon, lat, lon) for s in track]
        assert min(dists) >= dists[0] - stop_distance_m(INSPECT_SPEED_MS), (dists[0], min(dists))
        sideways = max(distance_m(s.lat, s.lon, *here) for s in track)
        assert sideways <= SIDEWAYS_M, sideways
        assert max(s.alt_m for s in track) <= CLIMB_TO_M + ALT_TOL_M

        # STOP and FLY HOME go past the AI as always.
        monkeypatch.undo()
        t0 = time.monotonic()
        assert (await sc.press("stop")).status_code == 202
        await stopped_and_holding(sc, t0)
        assert (await sc.press("resume")).status_code == 202
        assert (await sc.press("return_home")).status_code == 202
        await sc.until(lambda: not sc.tel.armed, 300, "landed")
        assert sc.from_home_m() <= 3.0
        await sc.check_invariants()


AHEAD_R_M = 30.0  # the second cylinder, just ahead on the goto's leg
NOISE_M = 1.0  # the flight controller's wobble and the track's sampling
# The way out: the cylinder's radius from about its centre, the margin, braking from the
# leg's speed first, and the wobble.
OUT_SLACK_M = 10.0


class Exit(NamedTuple):
    """A goto superseded by the way out of ``around``: where it was, and when."""

    goto: dict[str, Any]
    around: Keepout
    ahead: tuple[float, float]
    here: tuple[float, float]
    since: float
    speed: float
    alt_m: float


async def _a_goto_leaves(
    sc: Any, source: Literal["candidate", "confirmed"] = "candidate", alt_m: float = 40.0
) -> Exit:
    """A goto at ``alt_m`` flies on inside a person's cylinder (``source``) that came around
    it, a second one just ahead, closer than it can turn away from: it stops short of that
    one and leaves the one it is in; the operator is told, the goto superseded."""
    p1 = next(p for p in FIXTURE["pins"] if p["id"] == "P1")
    target = sc.at(float(p1["east_m"]), float(p1["north_m"]))
    res = await sc.operator_order("goto", {"lat": target[0], "lon": target[1], "alt_m": alt_m})
    assert res["gate"]["decision"] == "ALLOW", res
    goto = await sc.wait_command("goto", "running", 30)
    ex = sc.r.onboard.ex
    st = sc.sim.harness.state
    plane = Plane(*sc.r.home)
    line: list[tuple[float, float, float, float, float]] = []

    def on_the_leg() -> bool:
        """Straight on the leg at speed: where the drone is, which way, how fast."""
        now = sc.track[-1]
        before = next((s for s in reversed(sc.track) if s.t <= now.t - 1.0), None)
        if before is None or now.groundspeed_ms < 4.0 or sc.from_home_m() < 30.0:
            return False
        if abs(now.alt_m - alt_m) > ALT_TOL_M:
            return False  # still changing height
        (e, n), (e0, n0) = plane.xy(now.lat, now.lon), plane.xy(before.lat, before.lon)
        step = math.dist((e, n), (e0, n0))
        if step < 0.9 * now.groundspeed_ms * (now.t - before.t):
            return False  # still speeding up
        line[:] = [(e, n, (e - e0) / step, (n - n0) / step, now.groundspeed_ms)]
        return True

    await sc.until(on_the_leg, 90, "on the goto's leg at speed")
    ((e, n, de, dn, speed),) = line
    # One person's cylinder around the drone (it may fly on out of it), and another's
    # edge closer ahead than the drone can turn away from, far enough to stop outside.
    edge = (stop_distance_m(speed) + stop_m(speed)) / 2
    assert stop_distance_m(speed) + 3.0 < edge < stop_m(speed), (speed, edge)
    around = Keepout(
        keepout_id=new_id(),
        lat=plane.latlon(e, n)[0],
        lon=plane.latlon(e, n)[1],
        radius_m=PERSON_R_M,
        source=source,
        source_id=new_id(),
    )
    a_lat, a_lon = plane.latlon(e + (edge + AHEAD_R_M) * de, n + (edge + AHEAD_R_M) * dn)
    ahead = Keepout(
        keepout_id=new_id(),
        lat=a_lat,
        lon=a_lon,
        radius_m=AHEAD_R_M,
        source="candidate",
        source_id=new_id(),
    )
    since = time.monotonic()
    here = (st.lat, st.lon)
    install = KeepoutInstall(
        install_seq=ex.keepout_seq + INSTALL_AHEAD, keepouts=[*ex.keepouts, around, ahead]
    )
    ex.on_keepouts(install, new_id())
    assert around in ex.keepouts and ahead in ex.keepouts  # taken, not refused as stale

    holds = await until_event(sc, "keepout_hold", 30.0)
    told = holds[-1]["payload"]
    assert told.get("inside") is False, told  # it leaves: not over the person
    assert "climb_to_m" not in told, told
    assert await _events(sc, "keepout_exit")
    after = await sc.command(goto["id"])
    assert after["status"] == "superseded", after
    assert "KEEPOUT_VIOLATION" in str(after), after
    return Exit(goto, around, (a_lat, a_lon), here, since, speed, alt_m)


def _outside(sc: Any, k: Keepout) -> bool:
    s = sc.track[-1]
    return distance_m(s.lat, s.lon, k.lat, k.lon) >= k.radius_m


def _the_way_out(sc: Any, x: Exit, until: float | None = None) -> None:
    """Straight out at its height: never above it, never farther than the shortest way
    (and braking), never into the other person's cylinder."""
    track = [s for s in sc.track if s.t >= x.since and s.armed and (until is None or s.t < until)]
    far = max(distance_m(s.lat, s.lon, *x.here) for s in track)
    assert far <= PERSON_R_M + KEEPOUT_ARC_MARGIN_M + stop_distance_m(x.speed) + OUT_SLACK_M, far
    assert max(s.alt_m for s in track) <= x.alt_m + ALT_TOL_M  # no climb
    nearest = min(distance_m(s.lat, s.lon, *x.ahead) for s in track)
    assert nearest >= AHEAD_R_M - NOISE_M, (nearest, AHEAD_R_M)


@pytest.mark.budget_s(480)
async def test_a_goto_held_over_a_person_leaves_by_the_shortest_way_and_hovers(
    sim: Any, migrated_db: Any, tmp_path: Path
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()
        x = await _a_goto_leaves(sc)
        st = sc.sim.harness.state
        await sc.until(
            lambda: _outside(sc, x.around) and st.ground_speed_ms < 0.5, 120, "out and still"
        )
        seen: list[dict[str, Any]] = []

        async def kept_out() -> bool:
            seen[:] = [await sc.onboard_state()]
            return seen[0]["executor_state"] == "HOLD" and seen[0]["hold_reason"] == "limit"

        try:
            await sc.until(kept_out, 5, "the keep-out hover")
        except AssertionError as e:
            live = {k: seen[0].get(k) for k in ("executor_state", "hold_reason", "active")}
            kinds = [ev["type"] for ev in await sc.events()][-30:]
            raise AssertionError(f"{e}; onboard {live}; events {kinds}") from e
        await sc.hold_for(10.0, lambda: sc.fc_mode() == "LOITER", "hovering outside")
        assert len(await _events(sc, "keepout_hold")) == 1  # told once
        assert _outside(sc, x.around)
        _the_way_out(sc, x)

        t0 = time.monotonic()
        assert (await sc.press("stop")).status_code == 202
        await stopped_and_holding(sc, t0)
        assert (await sc.press("resume")).status_code == 202
        assert (await sc.press("return_home")).status_code == 202
        await sc.until(lambda: not sc.tel.armed, 300, "landed")
        assert sc.from_home_m() <= 3.0
        await sc.check_invariants()


@pytest.mark.budget_s(540)
async def test_a_link_loss_during_the_way_out_goes_on_and_flies_home(
    sim: Any, migrated_db: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Slowed, so that the loss is noticed on the way out.
    monkeypatch.setattr(executor_module, "_EXIT_SPEED_MS", 3.0)
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        # The cylinder left is a confirmed find's own.
        x = await _a_goto_leaves(sc, source="confirmed")
        ex = sc.r.onboard.ex
        sc.radio(sdlink=False)
        down = time.monotonic()
        skills: set[str | None] = set()
        noticed: list[str | None] = []  # what ran when the loss was noticed

        def watch() -> None:
            active = ex.active.skill if ex.active is not None else None
            skills.add(active)
            if ex.link_rule.phase != LinkLossPhase.NONE and not noticed:
                noticed.append(active)

        def find_hover() -> bool:
            watch()
            return ex.state == ExecutorState.HOLD and ex.link_rule.phase == LinkLossPhase.FIND_HOVER

        try:
            await sc.until(find_hover, 90, "the find hover in the keep-out hold")
            assert noticed == [INTERNAL_KEEPOUT_EXIT], noticed  # noticed on the way out
            assert ex.hold_reason == "limit", ex.hold_reason
            assert sc.fc_mode() == "LOITER"
            assert _outside(sc, x.around)

            def kept_out() -> bool:
                watch()
                return ex.hold_reason == "limit" and sc.fc_mode() == "LOITER"

            await sc.hold_for(
                max(0.0, FIND_HOME_AT - SLACK_S - (time.monotonic() - down)),
                kept_out,
                "the find hover outside",
            )
            await sc.until(lambda: sc.fc_mode() in HOME_MODES, 2 * SLACK_S + 1, "home")
            took = time.monotonic() - down
            assert FIND_HOME_AT - SLACK_S <= took <= FIND_HOME_AT + SLACK_S, took
            homing = time.monotonic()
            assert INTERNAL_FIND_GOTO not in skills, skills
            _the_way_out(sc, x, until=homing)
            await sc.until(lambda: not sim.harness.state.armed, 300, "landed")
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n{trace(sc, down)}") from None
        finally:
            await back_and_land(sc)
        assert home_m(sc) <= 3.0
        ended = "executor.find_hold_ended"
        await sc.until(lambda: bool(sc.published(ended)), 15, "the queued events after the link")
        started = await onboard_events(sc, "find_hold_started")
        assert started and started[-1]["payload"]["keepout_id"] == str(x.around.keepout_id), started
        assert not await onboard_events(sc, "find_move_keepout_exit")
        await sc.check_invariants()


@pytest.mark.budget_s(420)
async def test_a_link_loss_with_no_find_during_the_way_out_goes_on_and_flies_home(
    sim: Any, migrated_db: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Slowed, so that the way out (about 105 m) is still running well past the hold's
    # time (23 s) and ends before home is due (53 s).
    monkeypatch.setattr(executor_module, "_EXIT_SPEED_MS", 3.0)
    async with scenario(sim, migrated_db, tmp_path, proxied_link=True) as sc:
        await sc.airborne()
        x = await _a_goto_leaves(sc)
        ex = sc.r.onboard.ex
        sc.radio(sdlink=False)
        down = time.monotonic()
        past_hold: list[str | None] = []  # what ran once the hold was due

        def watch() -> None:
            active = ex.active.skill if ex.active is not None else None
            if time.monotonic() - down >= HOLD_AT + 0.5 and not past_hold:
                past_hold.append(active)

        def kept_out() -> bool:
            watch()
            return ex.state == ExecutorState.HOLD and ex.hold_reason == "limit"

        def homing() -> bool:
            watch()
            return sc.fc_mode() in HOME_MODES

        try:
            await sc.until(kept_out, HOME_AT - SLACK_S, "the keep-out hold outside")
            assert _outside(sc, x.around)
            await sc.until(homing, HOME_AT + SLACK_S - (time.monotonic() - down), "home")
            took = time.monotonic() - down
            # The hold was due on the way out: it went on, and home is on time.
            assert past_hold == [INTERNAL_KEEPOUT_EXIT], past_hold
            assert HOME_AT - SLACK_S <= took <= HOME_AT + SLACK_S, took
            _the_way_out(sc, x, until=time.monotonic())
            await sc.until(lambda: not sim.harness.state.armed, 300, "landed")
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n{trace(sc, down)}") from None
        finally:
            await back_and_land(sc)
        assert home_m(sc) <= 3.0
        await until_event(sc, "link_loss_rtl")  # queued while the link was down
        assert await onboard_events(sc, "link_loss_hold")
        assert not await onboard_events(sc, "find_hold_started")
        await sc.check_invariants()


async def _events(sc: Any, kind: str) -> list[dict[str, Any]]:
    return [e for e in await sc.events(f"executor.{kind}") if e["type"] == f"executor.{kind}"]
