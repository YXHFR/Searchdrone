"""Owner decisions 2026-11-03 on the cylinders of UNCONFIRMED persons (research 36).

A search route of a charter that allows overflight may cross such a cylinder straight
through, briefly, at its search height; never hover, turn, descend or end a leg in it.
Confirmed finds keep the full cylinder, and so does everything in live without the
permit. A drone boxed in leaves by the shortest way out (``escape``). Going home and
hovering are never refused for an old picture.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
from gate_fixtures import CHARTER, IMPLEMENTED, PLANE, goto, onboard_ctx, point, req
from test_keepout import cylinder

from sd_safety import evaluate, keepout
from sd_safety.constants import KEEPOUT_PASS_MAX_S
from sd_safety.keepout import (
    KEEPOUT_ARC_MARGIN_M,
    UNCONFIRMED_SOURCES,
    Passage,
    chord_m,
    escape,
    passage_for,
    passes,
    reroute,
    route_violation,
    split,
    xy_route_violation,
)
from sd_safety.policy import BASED_ON_EXEMPT, KEEPOUT_PASS_SKILLS
from sd_safety.targets import Polygon, Xy
from sd_schemas.base import LatLon, LatLonAlt
from sd_schemas.gate import GateRequest, StateRef
from sd_schemas.keepout import leg_enters

ALLOWED = [*IMPLEMENTED, "orbit", "inspect_detection", "capture_stills"]
SIM = CHARTER.model_copy(
    update={"allowed_skills": ALLOWED, "operation_mode": "sim", "overflight_allowed": True}
)
LIVE = CHARTER.model_copy(
    update={"allowed_skills": ALLOWED, "operation_mode": "live", "overflight_allowed": False}
)
BOX = Polygon(((-400.0, -400.0), (400.0, -400.0), (400.0, 400.0), (-400.0, 400.0)))
PASS = Passage.at(7.0)  # 84 m of a leg inside a cylinder at 7 m/s


# --- which moves get a passage ----------------------------------------------------------


def test_only_search_routes_with_overflight_and_a_speed_get_a_passage() -> None:
    assert {"survey_sector", "follow_path", "resume"} == KEEPOUT_PASS_SKILLS
    for skill in KEEPOUT_PASS_SKILLS:
        p = passage_for(skill, SIM, 7.0)
        assert p is not None and p.max_chord_m == pytest.approx(7.0 * KEEPOUT_PASS_MAX_S)
        assert passage_for(skill, LIVE, 7.0) is None  # live without the permit
        assert passage_for(skill, None, 7.0) is None
        for speed in (None, 0.0, -1.0, math.nan, math.inf):
            assert passage_for(skill, SIM, speed) is None
    for skill in ("goto", "orbit", "inspect_detection", "hold", "return_home", "track_target"):
        assert passage_for(skill, SIM, 7.0) is None


def test_only_detections_and_candidates_are_unconfirmed() -> None:
    assert {"detection", "candidate"} == UNCONFIRMED_SOURCES
    d = cylinder(0, 0, source="detection")
    c = cylinder(0, 0, source="candidate")
    f = cylinder(0, 0, source="confirmed")
    assert split([d, c, f]) == ([f], [d, c])


# --- one leg past one cylinder ----------------------------------------------------------


def test_chord_m() -> None:
    c, r = (0.0, 0.0), 30.0
    assert chord_m((-100, 0), (100, 0), c, r) == pytest.approx(60.0)  # through the centre
    assert chord_m((-100, 18), (100, 18), c, r) == pytest.approx(48.0)  # 2*sqrt(30^2-18^2)
    assert chord_m((-100, 40), (100, 40), c, r) == 0.0  # beside it
    assert chord_m((0, 0), (100, 0), c, r) == pytest.approx(30.0)  # starts at the centre
    assert chord_m((-100, 0), (-50, 0), c, r) == 0.0  # stops short
    assert chord_m((5, 5), (5, 5), c, r) == 0.0


def test_a_leg_passes_straight_through_but_never_stops_turns_or_lingers_inside() -> None:
    c, r = (0.0, 0.0), 30.0
    assert passes([(-100, 0), (100, 0)], c, r, PASS)  # 60 m <= 84 m
    assert not passes([(-100, 0), (0, 0)], c, r, PASS)  # ends inside: a hover or a corner
    assert not passes([(-200, 0), (-100, 0), (-10, 0), (100, 0)], c, r, PASS)  # a corner inside
    assert not passes([(-100, 0), (100, 0)], c, 50.0, PASS)  # 100 m > 84 m
    assert not passes([(-100, 0), (100, 0)], c, r, Passage(max_chord_m=59.0))
    assert passes([(-100, 40), (100, 40)], c, r, Passage(max_chord_m=0.0))  # never inside


def test_no_corner_just_outside_the_edge_of_a_cylinder_it_crosses() -> None:
    """Review of #202: a corner just outside the edge is a turn the flight controller cuts
    into the cylinder. Each corner next to a stretch inside keeps KEEPOUT_ARC_MARGIN_M."""
    c, r = (0.0, 0.0), 30.0
    near = r + KEEPOUT_ARC_MARGIN_M - 0.5
    far = r + KEEPOUT_ARC_MARGIN_M + 0.5
    assert not passes([(-100, 0), (near, 0)], c, r, PASS)  # ends just outside
    assert passes([(-100, 0), (far, 0)], c, r, PASS)
    assert not passes([(-100, -100), (-near, 0), (100, 0)], c, r, PASS)  # starts just outside
    assert passes([(-100, -100), (-far, 0), (100, 0)], c, r, PASS)
    # A corner just outside a cylinder that no leg next to it comes into is no turn over it.
    assert passes([(-100, 100), (0, near), (100, 100)], c, r, PASS)


def test_a_lawnmower_turn_just_outside_the_edge_is_no_passage() -> None:
    """Review of #202: legs along y=0 to (0, 0), north to (0, 30), then west along y=30,
    past r=42 at (-40, 15). Each chord alone is under 84 m (78.4, 25.6, 78.4), but the
    drone turns twice by 90 degrees about 0.7 m outside the edge and is over the person
    for about 26 s."""
    c, r = (-40.0, 15.0), 42.0
    route: list[Xy] = [(-200, 0), (0, 0), (0, 30), (-200, 30)]
    chords = [chord_m(a, b, c, r) for a, b in pairwise(route)]
    assert chords == pytest.approx([78.4, 25.6, 78.4], abs=0.1)
    assert all(x <= PASS.max_chord_m for x in chords)  # the old per-leg check let it pass
    assert not passes(route, c, r, PASS)
    assert xy_route_violation(route, [], [(c, r)], PASS) == 0
    # Planned again it goes around the cylinder like a confirmed one, never into it.
    out = reroute(route, [], area=BOX, soft=[(c, r)], passage=PASS)
    assert out is not None and xy_route_violation(out, [], [(c, r)], PASS) is None
    assert all(not leg_enters(a, b, c, r) for a, b in pairwise(out))


def test_legs_one_after_another_inside_one_cylinder_are_one_pass() -> None:
    """Out by more than the margin and straight back in: the time over the person adds up."""
    c, r = (0.0, 0.0), 30.0
    route: list[Xy] = [(-100, -10), (40, -10), (-100, 10)]
    first, second = (chord_m(a, b, c, r) for a, b in pairwise(route))
    assert first <= PASS.max_chord_m and second <= PASS.max_chord_m
    assert first + second > PASS.max_chord_m
    assert not passes(route, c, r, PASS)
    # Far enough apart that a leg clear of it lies between: two passes, each in time.
    lanes: list[Xy] = [(-100, -10), (100, -10), (100, 200), (-100, 200), (-100, 10), (100, 10)]
    assert passes(lanes, c, r, PASS)


def test_the_first_leg_from_inside_counts_the_time_spent_in_it_already() -> None:
    """Review of #202: a route planned again mid-passage gets no fresh budget. A first leg
    that starts inside continues the pass the drone is on, its time inside counted; with
    that time unknown it may only leave without going deeper."""
    c, r = (0.0, 0.0), 30.0
    on: list[Xy] = [(-10, 0), (100, 0)]  # 40 m inside, through the centre: deeper first
    assert not passes(on, c, r, PASS)  # time inside unknown
    assert passes(on, c, r, PASS, spent_m=0.0)  # it came around the drone just now
    assert passes(on, c, r, PASS, spent_m=PASS.max_chord_m - 40.0)
    assert not passes(on, c, r, PASS, spent_m=PASS.max_chord_m - 39.0)
    out: list[Xy] = [(10, 0), (100, 0)]  # straight out, no deeper: always let go
    assert passes(out, c, r, PASS)
    assert passes(out, c, r, PASS, spent_m=PASS.max_chord_m + 50.0)
    # ... but a leg straight back in after it adds up with it: one pass.
    back: list[Xy] = [(10, 0), (100, 0), (100, 100), (-100, 20)]
    assert passes(back, c, r, PASS, spent_m=0.0)
    assert not passes([(10, 0), (40, 0), (-100, 5)], c, r, PASS, spent_m=30.0)
    # The same through xy_route_violation, by the cylinder's index.
    soft = [(c, r)]
    assert xy_route_violation(on, [], soft, PASS) == 0
    assert xy_route_violation(on, [], soft, PASS, spent_m=[0.0]) is None
    assert xy_route_violation(on, [], soft, PASS, spent_m=[60.0]) == 0
    # Seconds to metres at the passage's speed.
    assert PASS.spent_m(KEEPOUT_PASS_MAX_S) == pytest.approx(PASS.max_chord_m)
    assert PASS.spent_m(-1.0) == 0.0


def test_reroute_mid_passage_never_gives_a_fresh_budget() -> None:
    c, r = (0.0, 0.0), 30.0
    soft = [(c, r)]
    route: list[Xy] = [(-10, 0), (150, 0)]
    assert reroute(route, [], area=BOX, soft=soft, passage=PASS, spent_m=[0.0]) == route
    # Most of the budget spent: no way on through the centre; it leaves without going
    # deeper, and the result is clear by the same check.
    out = reroute(route, [], area=BOX, soft=soft, passage=PASS, spent_m=[70.0])
    assert out is not None and out != route
    assert xy_route_violation(out, [], soft, PASS, spent_m=[70.0]) is None
    assert not leg_enters(out[0], out[1], c, r)


# --- a whole route ----------------------------------------------------------------------


def test_route_through_an_unconfirmed_cylinder_only_with_a_passage() -> None:
    soft = [((0.0, 0.0), 30.0)]
    route: list[Xy] = [(-100, -200), (-100, 0), (100, 0), (100, 200)]
    assert xy_route_violation(route, [], soft) == 0  # no passage: the full cylinder
    assert xy_route_violation(route, [], soft, PASS) is None
    assert xy_route_violation(route, soft, [], PASS) == 0  # a confirmed one: never
    # A corner inside it (a turn in the cylinder) is refused even with a passage.
    turn: list[Xy] = [(-100, 0), (0, 0), (0, 200)]
    assert xy_route_violation(turn, [], soft, PASS) == 0
    # A descending leg gets no passage.
    assert xy_route_violation(route, [], soft, PASS, descends=[False, True, False]) == 0
    assert xy_route_violation(route, [], soft, PASS, descends=[True, False, False]) is None


def test_route_violation_with_heights_and_sources() -> None:
    soft = cylinder(0, 150, radius=35.0, source="detection")
    hard = cylinder(0, 150, radius=35.0, source="confirmed")
    alt = CHARTER.home.alt_amsl_m + 40

    def path(*pts: tuple[float, float, float]) -> list[LatLonAlt]:
        return [
            LatLonAlt(lat=point(e, n).lat, lon=point(e, n).lon, alt_amsl_m=alt + dz)
            for e, n, dz in pts
        ]

    level = path((0, 50, 0), (0, 250, 0))
    assert route_violation(level, [soft], PLANE) == soft
    assert route_violation(level, [soft], PLANE, PASS) is None
    assert route_violation(level, [hard], PLANE, PASS) == hard
    down = path((0, 50, 0), (0, 250, -10))
    assert route_violation(down, [soft], PLANE, PASS) == soft


# --- the gate ---------------------------------------------------------------------------


def _survey(path: list[LatLonAlt], speed: float | None) -> GateRequest:
    return req(
        "survey_sector",
        {"sector_id": "A"},
        planned_path=path,
        planned_speed_ms=speed,
        phase="onboard",
    )


def _level(*en: tuple[float, float]) -> list[LatLonAlt]:
    alt = CHARTER.home.alt_amsl_m + 40
    return [LatLonAlt(lat=point(e, n).lat, lon=point(e, n).lon, alt_amsl_m=alt) for e, n in en]


def test_gate_lets_a_search_route_pass_an_unconfirmed_cylinder_in_sim_only() -> None:
    k = cylinder(0, 150, source="detection")
    path = _level((0, 50), (0, 250))
    sim = onboard_ctx(charter=SIM, keepouts=(k,))
    live = onboard_ctx(charter=LIVE, keepouts=(k,))
    assert "KEEPOUT_VIOLATION" not in evaluate(_survey(path, 7.0), sim).reason_codes
    assert "KEEPOUT_VIOLATION" in evaluate(_survey(path, 7.0), live).reason_codes
    assert "KEEPOUT_VIOLATION" in evaluate(_survey(path, None), sim).reason_codes  # no speed
    # Too slow to be through in KEEPOUT_PASS_MAX_S: 70 m at 5 m/s is 14 s.
    assert "KEEPOUT_VIOLATION" in evaluate(_survey(path, 5.0), sim).reason_codes
    # A confirmed find is never passed.
    hard = onboard_ctx(charter=SIM, keepouts=(cylinder(0, 150),))
    assert "KEEPOUT_VIOLATION" in evaluate(_survey(path, 7.0), hard).reason_codes
    # A goto through it is never a passage.
    through = req("goto", goto(0, 250), phase="onboard", planned_path=path, planned_speed_ms=7.0)
    assert "KEEPOUT_VIOLATION" in evaluate(through, sim).reason_codes


def test_gate_counts_the_time_the_drone_spent_inside_already() -> None:
    """A search route planned from inside an unconfirmed cylinder continues the pass the
    drone is on: the gate counts the time it spent inside (``keepout_inside_s``)."""
    k = cylinder(0, 150, source="detection")  # r=35 around (0, 150)
    path = _level((0, 140), (0, 250))  # from inside, through the centre: 45 m inside
    unknown = onboard_ctx(charter=SIM, keepouts=(k,))
    assert "KEEPOUT_VIOLATION" in evaluate(_survey(path, 7.0), unknown).reason_codes
    came = onboard_ctx(charter=SIM, keepouts=(k,), keepout_inside_s={k.keepout_id: 0.0})
    assert "KEEPOUT_VIOLATION" not in evaluate(_survey(path, 7.0), came).reason_codes
    # 45 m at 7 m/s is 6.4 s: with 6 s spent already the pass is over 12 s.
    late = onboard_ctx(charter=SIM, keepouts=(k,), keepout_inside_s={k.keepout_id: 6.0})
    assert "KEEPOUT_VIOLATION" in evaluate(_survey(path, 7.0), late).reason_codes


def test_return_home_and_hold_are_never_refused_for_an_old_picture() -> None:
    assert {"return_home", "hold"} == BASED_ON_EXEMPT
    old = StateRef(state_epoch=6, state_seq=1)
    c = onboard_ctx(charter=SIM)
    for skill in BASED_ON_EXEMPT:
        assert "STATE_CHANGED" not in evaluate(req(skill, {}, based_on=old), c).reason_codes
    assert "STATE_CHANGED" in evaluate(req(based_on=old), c).reason_codes


# --- reroute with a passage -------------------------------------------------------------


def test_reroute_passes_an_unconfirmed_cylinder_straight_and_goes_around_a_long_one() -> None:
    soft = [((0.0, 0.0), 30.0)]
    route: list[Xy] = [(-150, 0), (0, 0), (150, 0)]  # a corner in the middle of it
    out = reroute(route, [], area=BOX, soft=soft, passage=PASS)
    assert out == [(-150, 0), (150, 0)]  # the corner is dropped, the leg passes
    assert out is not None and xy_route_violation(out, [], soft, PASS) is None
    # Too wide to pass in time: gone around like a confirmed one.
    wide = [((0.0, 0.0), 50.0)]
    out = reroute(route, [], area=BOX, soft=wide, passage=PASS)
    assert out is not None and len(out) > 2
    assert all(not leg_enters(a, b, *wide[0]) for a, b in pairwise(out))
    # Without a passage the unconfirmed cylinder is a full one.
    out = reroute(route, [], area=BOX, soft=soft)
    assert out is not None and all(not leg_enters(a, b, *soft[0]) for a, b in pairwise(out))


# --- boxed in ---------------------------------------------------------------------------


def test_escape_takes_the_shortest_way_out_and_may_cross_an_unconfirmed_cylinder() -> None:
    here: Xy = (0.0, 0.0)
    soft = [((0.0, 0.0), 30.0)]
    p = escape(here, [], soft, BOX)
    assert p is not None
    assert math.dist(p, here) == pytest.approx(30.0 + KEEPOUT_ARC_MARGIN_M, abs=0.5)
    # Inside one unconfirmed cylinder with others close around: through the gap or across one.
    ring = [((60.0 * math.cos(a), 60.0 * math.sin(a)), 35.0) for a in (0, 2.1, 4.2)]
    p = escape(here, [], [*soft, *ring], BOX)
    assert p is not None
    assert all(math.dist(p, c) >= r + KEEPOUT_ARC_MARGIN_M - 1e-6 for c, r in [*soft, *ring])


def test_escape_never_enters_a_confirmed_cylinder_and_gives_none_when_boxed_in() -> None:
    here: Xy = (0.0, 0.0)
    soft = [((0.0, 0.0), 30.0)]
    east = [((70.0, 0.0), 40.0)]  # confirmed, on the near side
    p = escape(here, east, soft, BOX)
    assert p is not None and not leg_enters(here, p, *east[0])
    # Confirmed all around: no way out (the caller flies home).
    walls = [
        ((70.0 * math.cos(math.radians(a)), 70.0 * math.sin(math.radians(a))), 45.0)
        for a in range(0, 360, 30)
    ]
    assert escape(here, walls, soft, BOX) is None
    # Outside every cylinder: nothing to escape.
    assert escape((200.0, 200.0), east, soft, BOX) is None
    # The way out stays inside the area.
    tight = Polygon(((-35.0, -400.0), (400.0, -400.0), (400.0, 400.0), (-35.0, 400.0)))
    p = escape((-25.0, 0.0), [], [((-25.0, 0.0), 30.0)], tight)
    assert p is not None and p[0] > -35.0


def test_escape_prefers_a_way_past_another_unconfirmed_person_to_one_over_them() -> None:
    here: Xy = (10.0, 0.0)  # inside A, nearest its east side, where B lies
    a, b = ((0.0, 0.0), 50.0), ((45.0, 0.0), 30.0)
    p = escape(here, [], [a, b], BOX)
    assert p is not None and not leg_enters(here, p, *b)
    assert all(math.dist(p, c) >= r + KEEPOUT_ARC_MARGIN_M - 1e-6 for c, r in (a, b))


# --- the edges of the passage code -----------------------------------------------------


def test_a_passage_with_no_time_has_no_budget_left_once_inside() -> None:
    assert Passage(max_chord_m=84.0, max_s=0.0).spent_m(0.1) == math.inf


def test_a_route_without_heights_never_counts_as_going_down() -> None:
    soft = cylinder(0, 150, radius=35.0, source="detection")
    flat = [LatLon(lat=point(0, 50).lat, lon=point(0, 50).lon), point(0, 250)]
    assert route_violation(flat, [soft], PLANE, PASS) is None


def test_reroute_with_a_passage_and_nothing_to_fly() -> None:
    assert reroute([], [], soft=[((0.0, 0.0), 30.0)], passage=PASS) == []


def test_reroute_with_a_passage_gives_none_when_a_confirmed_cylinder_has_no_way_around() -> None:
    hard = [((0.0, 0.0), 30.0)]
    soft = [((0.0, 150.0), 20.0)]
    narrow = Polygon(((-20.0, -400.0), (20.0, -400.0), (20.0, 400.0), (-20.0, 400.0)))
    route: list[Xy] = [(0.0, -200.0), (0.0, 200.0)]
    assert reroute(route, hard, area=narrow, soft=soft, passage=PASS) is None


def test_reroute_with_a_passage_gives_up_after_its_tries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cylinder too wide to pass is gone around on a second try; with one try only
    there is no way (the drone holds instead of planning on and on)."""
    wide = [((0.0, 0.0), 50.0)]
    route: list[Xy] = [(-150, 0), (150, 0)]
    assert reroute(route, [], area=BOX, soft=wide, passage=PASS) is not None
    monkeypatch.setattr(keepout, "_PROMOTE_TRIES", 1)
    assert reroute(route, [], area=BOX, soft=wide, passage=PASS) is None


def test_reroute_from_inside_lets_a_leg_straight_out_be() -> None:
    cyl = [((0.0, 0.0), 30.0)]
    route: list[Xy] = [(10.0, 0.0), (200.0, 0.0)]
    assert reroute(route, cyl, area=BOX) == route


def test_escape_skips_a_shorter_way_that_leaves_the_area() -> None:
    """West is the shortest way out, but it leaves the area: the next best way inside."""
    here: Xy = (-20.0, 0.0)
    area = Polygon(((-30.0, -400.0), (400.0, -400.0), (400.0, 400.0), (-30.0, 400.0)))
    p = escape(here, [], [((0.0, 0.0), 30.0)], area)
    assert p is not None and p[0] > -30.0
    assert math.dist(p, (0.0, 0.0)) >= 30.0 + KEEPOUT_ARC_MARGIN_M - 1e-6


def test_reroute_from_inside_straight_out_with_no_way_on_gives_none() -> None:
    """The leg straight out is the only way out; when the rest has no way around after it
    (a second cylinder fills the strip it flies in) there is no way at all."""
    cyl = [((0.0, 0.0), 30.0), ((100.0, 0.0), 20.0)]
    strip = Polygon(((-50.0, -10.0), (300.0, -10.0), (300.0, 10.0), (-50.0, 10.0)))
    assert reroute([(10.0, 0.0), (200.0, 0.0)], cyl, area=strip) is None
