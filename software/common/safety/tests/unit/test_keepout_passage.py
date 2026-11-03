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

from sd_safety import evaluate
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
from sd_schemas.base import LatLonAlt
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
    assert passes((-100, 0), (100, 0), c, r, PASS, first=False)  # 60 m <= 84 m
    assert not passes((-100, 0), (0, 0), c, r, PASS, first=False)  # ends inside: a corner
    assert not passes((-10, 0), (100, 0), c, r, PASS, first=False)  # starts inside: a corner
    assert passes((-10, 0), (100, 0), c, r, PASS, first=True)  # the drone is there already
    assert not passes((-100, 0), (100, 0), c, 50.0, PASS, first=False)  # 100 m > 84 m
    assert not passes((-100, 0), (100, 0), c, r, Passage(max_chord_m=59.0), first=False)
    assert passes((-100, 40), (100, 40), c, r, Passage(max_chord_m=0.0), first=False)


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
