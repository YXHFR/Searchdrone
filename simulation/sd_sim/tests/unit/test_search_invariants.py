"""The search scenarios' invariants E1, E2 and E8 catch what they name."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import replace

import pytest

from sd_safety.constants import KEEPOUT_PASS_GAP_S
from sd_safety.targets import Plane
from sd_sim import params, paths
from sd_sim import search_invariants as si
from sd_sim.invariants import Sample
from sd_sim.search_invariants import (
    CandidateWatch,
    SeenKeepout,
    keepout_distances,
    keepout_never_entered,
    no_confirm_without_human,
    stop_distance_m,
)

T0 = dt.datetime(2026, 10, 23, 10, 0, tzinfo=dt.UTC)
HOME = (-35.363261, 149.165230)  # ArduPilot's SITL site CMAC


def at(s: float) -> dt.datetime:
    return T0 + dt.timedelta(seconds=s)


def rev(state: str, s: float, cid: str = "c1", n: int = 2) -> dict[str, object]:
    return {"candidate_id": cid, "rev": n, "state": state, "created_at": at(s)}


def dec(decision: str, s: float, cid: str = "c1", ref: str = "operator:ci", session: str = "s1"):
    return {"candidate_id": cid, "decision": decision, "operator_ref": ref,
            "session_id": session, "created_at": at(s)}  # fmt: skip


VIEW = {"detection_id": "d1", "session_id": "s1", "created_at": at(1)}
OBSERVED = {"c1": ["d1"]}


def test_a_person_confirming_after_a_view_is_fine() -> None:
    assert no_confirm_without_human([rev("ready", 0, n=1), rev("confirmed", 3)],
                                    [dec("confirm", 2)], [VIEW], OBSERVED) == []  # fmt: skip


def test_confirmed_without_a_decision_breaks_e1() -> None:
    broken = no_confirm_without_human([rev("confirmed", 3)], [], [VIEW], OBSERVED)
    assert broken and "utan en persons beslut" in broken[0]


def test_a_decision_after_the_state_does_not_count() -> None:
    assert no_confirm_without_human([rev("rejected", 3)], [dec("reject", 5)], [], OBSERVED)


def test_confirming_without_a_large_view_breaks_e1() -> None:
    late = {**VIEW, "created_at": at(9)}
    other = {**VIEW, "session_id": "s2"}
    for views in ([], [late], [other]):
        broken = no_confirm_without_human([rev("confirmed", 3)], [dec("confirm", 2)], views,
                                          OBSERVED)  # fmt: skip
        assert any("stor vy" in b for b in broken), views


def test_a_decision_without_an_operator_breaks_e1() -> None:
    broken = no_confirm_without_human([], [dec("reject", 1, ref="service:search")], [], {})
    assert broken and "utan en person" in broken[0]


def test_a_vanished_or_older_candidate_breaks_e2() -> None:
    w = CandidateWatch()
    w.look({"a": 1, "b": 2})
    w.look({"a": 2, "b": 2, "c": 1})
    assert w.broken == []
    w.look({"a": 1, "c": 1})
    assert len(w.broken) == 2


def _track(
    points: list[tuple[float, float, float]], mode: str = "GUIDED", speed: float = 5.0
) -> list[Sample]:
    plane = Plane(*HOME)
    return [Sample(t, *plane.latlon(e, n), 40.0, speed, mode, True) for t, e, n in points]


def _flown(points: list[tuple[float, float, float, float]]) -> list[Sample]:
    """A track with each sample's own ground speed: (t, east, north, speed)."""
    plane = Plane(*HOME)
    return [Sample(t, *plane.latlon(e, n), 40.0, v, "GUIDED", True) for t, e, n, v in points]


def _cyl(since: float = 0.0, until: float = float("inf"), came: float | None = None) -> SeenKeepout:
    lat, lon = Plane(*HOME).latlon(100.0, 0.0)
    return SeenKeepout("k1", lat, lon, 30.0, since, until, came=came)


def test_flying_around_a_cylinder_is_fine() -> None:
    track = _track([(1, 0, 0), (2, 60, 0), (3, 100, 40), (4, 140, 0)])
    assert keepout_never_entered([_cyl()], track) == []
    assert keepout_distances([_cyl()], track) is not None


def test_leaving_a_cylinder_made_around_the_drone_is_fine_but_coming_back_is_not() -> None:
    leave = _track([(1, 95, 0), (2, 80, 0), (3, 40, 0)])
    assert keepout_never_entered([_cyl()], leave) == []
    back = _track([(1, 95, 0), (2, 40, 0), (3, 90, 0)])
    broken = keepout_never_entered([_cyl()], back)
    assert broken and broken[0].startswith("E8")
    assert (keepout_distances([_cyl()], back) or 0.0) < 0


def test_inside_since_it_came_the_drone_may_not_go_deeper() -> None:
    """Made around the drone 5 m inside the edge: the way out never goes deeper, so 5 m
    plus the track's metre is the deepest it may be until it is out."""
    wobble = _track([(1, 75, 0), (2, 75.8, 0), (3, 70, 0), (4, 60, 0)])
    assert keepout_never_entered([_cyl()], wobble) == []
    deeper = _track([(1, 75, 0), (2, 85, 0), (3, 60, 0)])
    broken = keepout_never_entered([_cyl()], deeper)
    assert broken and broken[0].startswith("E8") and "djupare" in broken[0]
    across = _track([(1, 75, 0), (2, 100, 0), (3, 140, 0)])  # over the person and out
    assert keepout_never_entered([_cyl()], across)


def test_inside_since_it_came_braking_may_take_it_its_stop_deeper() -> None:
    """Made around a drone heading deeper at 5 m/s: braking takes it on by its stop
    (7.5 m), not more; the same drone flying across is still caught."""
    braked = _track([(0, 75, 0), (0.5, 77.5, 0), (1, 80, 0), (2, 82, 0), (5, 82, 0), (8, 60, 0)])
    assert keepout_never_entered([_cyl()], braked) == []
    on = _track([(0, 75, 0), (0.5, 77.5, 0), (1, 80, 0), (2, 85, 0), (3, 90, 0)])
    broken = keepout_never_entered([_cyl()], on)
    assert broken and "inbromsningen tar den 7.5 m till" in broken[0]
    # Heading out when it came: no stop to allow for.
    turned = _track([(0, 75, 0), (1, 70, 0), (2, 77, 0)])
    assert keepout_never_entered([_cyl()], turned)


def _turning_in(brake_after_s: float | None, decel_ms2: float = 2.7) -> list[Sample]:
    """A cylinder comes 10 m around a drone at a survey corner: it flies north-east at
    3 m/s, 1 m/s of it towards the middle, and turns east (2.6 m/s² east), as in SK03 seed
    2 (CI run 42166655395: 1.35 m/s deeper when the cylinder came, 2.86 m/s 0.65 s later).
    With ``brake_after_s`` it then brakes at ``decel_ms2`` to a stop. Flown in 0.05 s steps
    and sampled at 5 Hz, as the rig streams."""
    e, n, ve, vn = 80.0, 0.0, 1.0, 2.8
    out: list[tuple[float, float, float, float]] = []
    dt_s = 0.05
    for i in range(121):
        t = i * dt_s
        if i % 4 == 0:
            out.append((round(t, 2), e, n, math.hypot(ve, vn)))
        if brake_after_s is None or t < brake_after_s - 1e-9:
            ve += 2.6 * dt_s
            vn = max(0.0, vn - 4.3 * dt_s)  # the turn takes the northward speed
            speed = math.hypot(ve, vn)
            if speed > 3.4:
                ve, vn = ve * 3.4 / speed, vn * 3.4 / speed
        else:
            speed = math.hypot(ve, vn)
            if speed > 0.0:
                slower = max(0.0, speed - decel_ms2 * dt_s) / speed
                ve, vn = ve * slower, vn * slower
        e, n = e + ve * dt_s, n + vn * dt_s
    return _flown(out)


def _bunched(track: list[Sample]) -> list[Sample]:
    """The same track as the rig may receive it: every second sample a few ms after the
    one before (``scenario_rig._observe`` pumps under the harness lock)."""
    return [replace(s, t=track[i - 1].t + 0.004) if i % 2 else s for i, s in enumerate(track)]


def test_a_cylinder_that_comes_in_a_turn_towards_it_allows_the_turns_speed() -> None:
    """The speed deeper at the first sample (about 0) is not what braking starts from: the
    flight controller finishes the turn until the way out takes, and the drone brakes
    from the fastest it went deeper before the braking is in force. Without braking it is
    still caught."""
    braked = _turning_in(brake_after_s=0.65)
    assert keepout_never_entered([_cyl()], braked) == []
    assert keepout_never_entered([_cyl()], _bunched(braked)) == []
    on = _turning_in(brake_after_s=None)
    broken = keepout_never_entered([_cyl()], on)
    assert broken and broken[0].startswith("E8") and "djupare" in broken[0]


@pytest.mark.parametrize(
    ("brake_after_s", "decel_ms2"),
    [(1.15, 2.7), (1.65, 2.7), (0.65, 1.35), (None, 2.7)],
    ids=["half-a-second-late", "a-second-late", "half-the-braking", "no-braking"],
)
@pytest.mark.parametrize("bunch", [False, True], ids=["even", "bunched"])
def test_a_turn_towards_a_cylinder_braking_late_or_weakly_breaks_e8(
    brake_after_s: float | None, decel_ms2: float, bunch: bool
) -> None:
    """The corner's allowance is the stop from the turn's speed, braking on time: braking
    later or weaker goes deeper and is caught, with samples evenly spaced or bunched."""
    track = _turning_in(brake_after_s, decel_ms2)
    broken = keepout_never_entered([_cyl()], _bunched(track) if bunch else track)
    assert broken and broken[0].startswith("E8") and "djupare" in broken[0]


@pytest.mark.parametrize("bunch", [False, True], ids=["even", "bunched"])
def test_bunched_samples_do_not_make_a_sideways_drone_fast_inwards(bunch: bool) -> None:
    """10 m inside, flying sideways at 5 m/s when the cylinder comes, then 2 m/s straight
    in for 2.4 s without braking (14.8 m deep): caught however the samples bunch, since
    0.4 m between two samples a few ms apart is no 5 m/s inwards."""
    pts: list[tuple[float, float, float, float]] = []
    for i in range(3):  # sideways (north), 5 Hz
        pts.append((i * 0.2, 80.0, i * 1.0, 5.0))
    for i in range(1, 13):  # then 2 m/s east, deeper, from 0.4 s on
        pts.append((0.4 + i * 0.2, 80.0 + i * 0.4, 2.0, 2.0))
    track = _flown(pts)
    broken = keepout_never_entered([_cyl()], _bunched(track) if bunch else track)
    assert broken and broken[0].startswith("E8") and "djupare" in broken[0]


def test_a_new_shape_of_a_cylinder_is_no_new_grace() -> None:
    """The ground moved a cylinder (same id) 5 m ahead of a drone at 5 m/s, 10 s after the
    id came: a new cylinder there would allow braking in, a new shape of an old one not."""
    braked = _track([(10, 65, 0), (11, 69, 0), (12, 71.5, 0), (16, 71.5, 0)])
    assert keepout_never_entered([_cyl(since=10.0)], braked) == []
    broken = keepout_never_entered([_cyl(since=10.0, came=0.0)], braked)
    assert broken and broken[0].startswith("E8")
    # Moved onto the drone: as one made around it, out and never deeper.
    leave = _track([(10, 75, 0), (11, 70, 0), (12, 60, 0)])
    assert keepout_never_entered([_cyl(since=10.0, came=0.0)], leave) == []
    deeper = _track([(10, 75, 0), (11, 90, 0)])
    assert keepout_never_entered([_cyl(since=10.0, came=0.0)], deeper)


def test_the_way_home_goes_past_cylinders() -> None:
    track = _track([(1, 95, 0), (2, 40, 0)]) + _track([(3, 90, 0), (4, 140, 0)], "SMART_RTL")
    assert keepout_never_entered([_cyl()], track) == []
    searching = _track([(1, 95, 0), (2, 40, 0), (3, 90, 0), (4, 140, 0)])
    assert keepout_never_entered([_cyl()], searching)


def test_searching_again_after_the_way_home_is_checked() -> None:
    track = (
        _track([(1, 95, 0), (2, 40, 0)]) + _track([(3, 40, 10)], "SMART_RTL") + _track([(4, 90, 0)])
    )
    broken = keepout_never_entered([_cyl()], track)
    assert broken and broken[0].startswith("E8")


def test_a_landing_after_stop_is_checked() -> None:
    home = _track([(1, 95, 0), (2, 40, 0)]) + _track([(3, 60, 0)], "SMART_RTL")
    assert keepout_never_entered([_cyl()], home + _track([(4, 90, 0)], "LAND")) == []
    stop = _track([(1, 95, 0), (2, 40, 0)]) + _track([(3, 90, 0)], "LAND")
    broken = keepout_never_entered([_cyl()], stop)
    assert broken and broken[0].startswith("E8")


def test_the_distance_counts_from_clearly_outside() -> None:
    leave = _track([(1, 95, 0), (2, 70, 0), (3, 60, 0), (4, 20, 0)])
    assert keepout_distances([_cyl()], leave) == pytest.approx(10.0, abs=0.5)


def test_a_cylinder_counts_only_while_it_is_in_force() -> None:
    track = _track([(1, 40, 0), (2, 100, 0), (3, 40, 0)])
    assert keepout_never_entered([_cyl(since=2.5)], track) == []
    assert keepout_never_entered([_cyl(until=0.5)], track) == []
    assert keepout_never_entered([_cyl()], track)


def test_a_cylinder_closer_than_the_stop_may_be_braked_into_by_the_stop() -> None:
    """The edge 5 m ahead at 5 m/s, closer than the 7.5 m the drone needs to stop: braking
    in by what the stop leaves (2.5 m) is no entry, deeper is."""
    assert stop_distance_m(5.0) == pytest.approx(7.5)
    braked = _track([(0, 65, 0), (1, 69, 0), (2, 71.5, 0), (3, 72.2, 0), (6, 72.2, 0)])
    assert keepout_never_entered([_cyl()], braked) == []
    deeper = _track([(0, 65, 0), (1, 69, 0), (2, 74, 0)])
    broken = keepout_never_entered([_cyl()], deeper)
    assert broken and broken[0].startswith("E8") and "inbromsningen" in broken[0]


def test_after_braking_into_a_cylinder_the_drone_may_leave_but_not_come_back() -> None:
    leave = _track([(0, 65, 0), (1, 69, 0), (2, 71.5, 0), (5, 71.5, 0), (7, 60, 0)])
    assert keepout_never_entered([_cyl()], leave) == []
    back = [*leave, *_track([(9, 72, 0)])]
    broken = keepout_never_entered([_cyl()], back)
    assert broken and broken[0].startswith("E8") and "inbromsningen" not in broken[0]


def test_a_cylinder_farther_than_the_stop_gets_no_braking() -> None:
    """The edge 20 m ahead at 5 m/s: room to stop, so 2 m in is an entry."""
    track = _track([(0, 50, 0), (1, 65, 0), (2, 72, 0)])
    broken = keepout_never_entered([_cyl()], track)
    assert broken and broken[0].startswith("E8")
    slow = _track([(0, 68, 0), (1, 72, 0)], speed=0.5)  # 2 m ahead, still: no excuse either
    assert keepout_never_entered([_cyl()], slow)


def test_braked_into_a_cylinder_the_drone_may_not_go_deeper_after_the_stop() -> None:
    stays = _track([(0, 65, 0), (1, 69, 0), (2, 71.5, 0), (5, 71.5, 0), (20, 71.8, 0)])
    assert keepout_never_entered([_cyl()], stays) == []  # stopped in it, hovers there
    deeper = _track([(0, 65, 0), (1, 69, 0), (2, 71.5, 0), (5, 71.5, 0), (9, 90, 0)])
    broken = keepout_never_entered([_cyl()], deeper)
    assert broken and broken[0].startswith("E8") and "inbromsningen" in broken[0]


def test_only_the_speed_towards_the_cylinder_is_braked() -> None:
    """5 m outside the edge at 5 m/s, but flying along it: no braking takes it in."""
    track = _track([(0, 65, 0), (1, 65, 4), (2, 72, 4)])
    broken = keepout_never_entered([_cyl()], track)
    assert broken and broken[0].startswith("E8")


def test_braking_counts_only_from_when_the_cylinder_came() -> None:
    braked = _track([(0, 65, 0), (1, 69, 0), (2, 71.5, 0), (6, 71.5, 0)])
    assert keepout_never_entered([_cyl()], braked) == []
    broken = keepout_never_entered([_cyl(since=-5.0)], braked)  # came long before this
    assert broken and broken[0].startswith("E8")


def test_a_fast_drone_that_never_brakes_breaks_e8() -> None:
    """At 14 m/s with the edge 5 m ahead, the stop may take it past the middle of the
    cylinder (41 m in, the radius is 30), so the depth alone allows the whole crossing:
    flying straight through without slowing down is still an entry."""
    assert stop_distance_m(14.0) - 5.0 > 30.0
    through = _flown([(0, 65, 0, 14), (1, 79, 0, 14), (2, 93, 0, 14), (3, 107, 0, 14),
                      (4, 121, 0, 14)])  # fmt: skip
    broken = keepout_never_entered([_cyl()], through)
    assert broken and broken[0].startswith("E8") and "m/s" in broken[0]
    braking = _flown([(0, 65, 0, 14), (1, 78, 0, 12.5), (2, 88, 0, 10.5), (3, 96, 0, 8),
                      (4, 102, 0, 5.5), (5, 105, 0, 3), (6, 106, 0, 0.5),
                      (9, 106, 0, 0.2)])  # fmt: skip
    assert keepout_never_entered([_cyl()], braking) == []


def test_stop_model_matches_the_flight_controllers_braking() -> None:
    """E8's stop model is the flight controller's LOITER braking: the safety file pins
    LOIT_BRK_* to the model's values, and the ramp and delay fit the model's times."""
    path = paths.REPO / "simulation" / "sitl" / "params" / "searchdrone-safety.parm"
    safety = params.parse_parm(path.read_text(encoding="utf-8"), path.name)
    for name, value in si.FC_BRAKE.items():
        assert safety.get(name) == value, f"{name} i {path.name} ska vara {value:g}"
    rules = {name: value for name, _, value in params.SAFETY_RULES}
    assert {name: rules.get(name) for name in si.FC_BRAKE} == si.FC_BRAKE
    # The jerk-limited ramp to full braking fits inside the settling time.
    assert si.STOP_DECEL_MS2 / si.STOP_JERK_MS3 <= si.STOP_SETTLE_S
    # A stick released just before still brakes within the reaction and settling.
    assert si.STOP_BRAKE_DELAY_S <= si.STOP_REACT_S + si.STOP_SETTLE_S
    # Copter-4.7.1 allows these values (AC_Loiter.cpp ranges).
    assert 0.25 <= si.STOP_DECEL_MS2 <= 2.5 and 5.0 <= si.STOP_JERK_MS3 <= 50.0
    assert 0.0 <= si.STOP_BRAKE_DELAY_S <= 2.0


# --- a passage through an unconfirmed person (owner decision 2026-11-03) ------------------


def _soft(since: float = 0.0) -> SeenKeepout:
    return replace(_cyl(since), passable=True)


def _line(t0: float, e0: float, e1: float, speed: float, n: float = 0.0, alt: float = 40.0,
          dt: float = 0.5) -> list[Sample]:  # fmt: skip
    """A straight leg east along ``n`` at ``speed``, sampled every ``dt``."""
    plane = Plane(*HOME)
    out: list[Sample] = []
    steps = round(abs(e1 - e0) / (speed * dt))
    for i in range(steps + 1):
        e = e0 + (e1 - e0) * i / steps
        out.append(Sample(t0 + i * dt, *plane.latlon(e, n), alt, speed, "AUTO", True))
    return out


def test_straight_through_a_passable_cylinder_is_a_passage() -> None:
    through = _line(0.0, 0.0, 200.0, 7.0)  # 60 m inside at 7 m/s: under 9 s
    assert keepout_never_entered([_soft()], through) == []
    broken = keepout_never_entered([_cyl()], through)  # the same cylinder, confirmed
    assert broken and "E8" in broken[0]


def test_a_passage_may_not_linger_descend_or_turn() -> None:
    slow = _line(0.0, 0.0, 200.0, 3.0)  # 60 m at 3 m/s: 20 s inside
    broken = keepout_never_entered([_soft()], slow)
    assert broken and "obekräftade" in broken[0] and " s " in broken[0], broken
    plane = Plane(*HOME)
    down = [
        Sample(x.t, x.lat, x.lon, 40.0 - (5.0 if 80 < plane.xy(x.lat, x.lon)[0] < 120 else 0.0),
               x.groundspeed_ms, x.mode, True)
        for x in _line(0.0, 0.0, 200.0, 7.0)
    ]  # fmt: skip
    broken = keepout_never_entered([_soft()], down)
    assert broken and "sjönk" in broken[0], broken
    # In, a turn north at the centre and out: a corner inside.
    turn = _line(0.0, 0.0, 100.0, 7.0) + [
        Sample(15.0 + i, *plane.latlon(100.0, 7.0 * (i + 1)), 40.0, 7.0, "AUTO", True)
        for i in range(6)
    ]
    broken = keepout_never_entered([_soft()], turn)
    assert broken and "svängde" in broken[0], broken


def test_a_hover_inside_a_passable_cylinder_breaks_e8() -> None:
    plane = Plane(*HOME)
    hover = _line(0.0, 0.0, 100.0, 7.0) + [
        Sample(15.0 + i, *plane.latlon(100.0, 0.0), 40.0, 0.0, "LOITER", True) for i in range(20)
    ]
    broken = keepout_never_entered([_soft()], hover)
    assert broken and "obekräftade" in broken[0], broken


def test_a_way_out_of_a_passable_cylinder_keeps_the_full_rules() -> None:
    """Came around the drone: leaving it slowly along a straight line is the way out, under
    the full rules, even when it takes longer than a passage."""
    out = _line(0.0, 100.0, 140.0, 2.0)
    assert keepout_never_entered([_soft()], out) == []


def test_stays_with_a_short_break_outside_are_one_passage() -> None:
    """Owner decision 2026-11-04: out of a passable cylinder for less than
    KEEPOUT_PASS_GAP_S and back in is one pass; the stays add up (8.6 s + 8.1 s here, over
    the 12 s and the slack together). Out for longer: two passes, each in time."""
    there = _line(0.0, 0.0, 140.0, 7.0)  # in at 70 m, out at 130 m
    end = there[-1].t

    def back(after: float) -> list[Sample]:
        return _line(end + after, 140.0, 0.0, 7.0, n=10.0)  # in again at 128 m

    broken = keepout_never_entered([_soft()], there + back(2.0))
    assert broken and "korta avbrott" in broken[0], broken
    assert keepout_never_entered([_soft()], there + back(KEEPOUT_PASS_GAP_S + 2.0)) == []
