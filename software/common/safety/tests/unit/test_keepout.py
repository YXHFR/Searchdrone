"""B2-09: keep-out cylinders in the gate (R-KEEPOUT), the orbit's radius and the reroute.

The vehicle hovers 50 m north of home (``gate_fixtures.vehicle``); a cylinder is placed
where each test needs it, in metres east/north of home.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import pytest
from gate_fixtures import (
    CHARTER,
    IMPLEMENTED,
    PLANE,
    ctx,
    goto,
    latlon,
    onboard_ctx,
    point,
    polygon,
    req,
)
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sd_safety import evaluate
from sd_safety.keepout import (
    KEEPOUT_ARC_MARGIN_M,
    ORBIT_KEEPOUT_MARGIN_M,
    reroute,
    ring_violation,
    route_violation,
    xy_route_violation,
)
from sd_safety.targets import Polygon, Xy, polygon_from_geojson
from sd_schemas.base import LatLonAlt, new_id
from sd_schemas.gate import GateContext, GateRequest
from sd_schemas.keepout import Keepout, leg_enters
from sd_schemas.skills import OrbitRing

ALLOWED = [*IMPLEMENTED, "orbit", "inspect_detection", "capture_stills"]
CHARTER_B2 = CHARTER.model_copy(update={"allowed_skills": ALLOWED})


def cylinder(east: float, north: float, radius: float = 35.0, source: str = "confirmed") -> Keepout:
    lat, lon = latlon(east, north)
    return Keepout(
        keepout_id=new_id(),
        lat=lat,
        lon=lon,
        radius_m=radius,
        source=source,  # type: ignore[arg-type]
        source_id=new_id(),
    )


def with_keepouts(*keepouts: Keepout, onboard: bool = False, **update: Any) -> GateContext:
    build = onboard_ctx if onboard else ctx
    return build(charter=CHARTER_B2, keepouts=keepouts, **update)


def codes(r: GateRequest, c: GateContext) -> list[str]:
    return evaluate(r, c).reason_codes


def orbit(east: float, north: float, radius: float, alt_m: float = 40.0) -> dict[str, Any]:
    lat, lon = latlon(east, north)
    return {"lat": lat, "lon": lon, "radius_m": radius, "alt_m": alt_m}


# --- the leg test ---------------------------------------------------------------------------


def test_leg_enters_basics() -> None:
    c, r = (0.0, 0.0), 10.0
    assert leg_enters((-20.0, 0.0), (20.0, 0.0), c, r)  # straight through
    assert not leg_enters((-20.0, 10.0), (20.0, 10.0), c, r)  # touching the edge is allowed
    assert not leg_enters((-20.0, 11.0), (20.0, 11.0), c, r)
    assert leg_enters((-20.0, 0.0), (-5.0, 0.0), c, r)  # ends inside
    assert not leg_enters((-20.0, 0.0), (-15.0, 0.0), c, r)  # stops short
    assert not leg_enters((5.0, 0.0), (30.0, 0.0), c, r)  # from inside, straight out
    assert not leg_enters((5.0, 0.0), (5.0, 30.0), c, r)  # from inside, sideways out
    assert leg_enters((5.0, 0.0), (-30.0, 0.0), c, r)  # from inside, deeper first
    assert not leg_enters((5.0, 0.0), (5.0, 0.0), c, r)  # standing still inside
    assert not leg_enters((-20.0, 0.0), (-20.0, 0.0), c, r)  # a point outside


# --- test_gate_keepout_violation -------------------------------------------------------------


def test_gate_keepout_violation() -> None:
    """A goto whose straight leg crosses a cylinder is refused on both sides; one that
    passes beside it, or goes there and stops short, is allowed."""
    k = cylinder(0, 150)
    through = req("goto", goto(0, 250))
    for onboard in (False, True):
        c = with_keepouts(k, onboard=onboard)
        d = evaluate(through, c)
        assert d.decision == "REJECT"
        assert "KEEPOUT_VIOLATION" in d.reason_codes
        assert d.details["keepout_id"] == str(k.keepout_id)
        assert d.message_sv.startswith("Vägen går in i en förbudscylinder")
    c = with_keepouts(k)
    assert codes(req("goto", goto(-120, 250)), c) == []  # passes 51 m west of the centre
    assert codes(req("goto", goto(0, 100)), c) == []  # stops 15 m short of the edge
    assert codes(req("goto", goto(0, 130)), c) == ["KEEPOUT_VIOLATION"]  # ends inside


def test_gate_keepout_lets_a_drone_inside_leave_but_not_go_deeper() -> None:
    k = cylinder(0, 40)  # the drone, 50 m north of home, is 10 m inside its north side
    c = with_keepouts(k)
    assert codes(req("goto", goto(0, 120)), c) == []  # straight out
    assert codes(req("goto", goto(80, 50)), c) == []  # sideways out
    assert codes(req("goto", goto(0, -60)), c) == ["KEEPOUT_VIOLATION"]  # through the centre


def test_gate_keepout_every_cylinder_counts() -> None:
    near, far = cylinder(200, 200, source="candidate"), cylinder(0, 150, source="detection")
    c = with_keepouts(near, far)
    d = evaluate(req("goto", goto(0, 250)), c)
    assert d.details["keepout_id"] == str(far.keepout_id)


def test_gate_keepout_exemptions() -> None:
    k = cylinder(0, 150)
    c = with_keepouts(k)
    assert codes(req("return_home", {}), c) == []  # home is always allowed
    assert codes(req("hold", {}), c) == []  # does not move
    # Without cylinders, or for a skill whose route the ground cannot see, nothing to say.
    assert codes(req("goto", goto(0, 250)), with_keepouts()) == []
    assert codes(req("survey_sector", {"sector_id": "A"}), c) == []


def test_gate_keepout_checks_the_onboard_path_of_every_move() -> None:
    """Onboard the handler's path is the route: a survey whose mission would cross a
    cylinder is refused (the handler reroutes it first, so this is the last line)."""
    k = cylinder(-150, -150)
    here = point(0, 50)
    alt = CHARTER.home.alt_amsl_m + 40
    path = [
        LatLonAlt(lat=p.lat, lon=p.lon, alt_amsl_m=alt)
        for p in (here, point(-150, 50), point(-150, -250))
    ]
    r = req("survey_sector", {"sector_id": "A"}, planned_path=path, phase="onboard")
    assert "KEEPOUT_VIOLATION" in codes(r, with_keepouts(k, onboard=True))
    home = req("return_home", {}, planned_path=path, phase="onboard")
    assert codes(home, with_keepouts(k, onboard=True)) == []


def test_route_violation_edges() -> None:
    k = cylinder(0, 150)
    assert route_violation([], [k], PLANE) is None
    assert route_violation([point(0, 0), point(0, 300)], [], PLANE) is None
    assert route_violation([point(0, 150)], [k], PLANE) == k  # a single point inside
    assert route_violation([point(0, 0)], [k], PLANE) is None
    assert xy_route_violation([(0.0, 0.0)], []) is None


# --- test_orbit_radius_gate ------------------------------------------------------------------


def test_orbit_radius_gate() -> None:
    """Around a find: the radius must be at least its cylinder's + 10 m."""
    k = cylinder(0, 150, radius=35.0)
    c = with_keepouts(k)
    d = evaluate(req("orbit", orbit(0, 150, 40)), c)
    assert d.decision == "REJECT"
    assert d.reason_codes == ["KEEPOUT_VIOLATION"]
    assert codes(req("orbit", orbit(0, 150, 44.9)), c) == ["KEEPOUT_VIOLATION"]
    assert codes(req("orbit", orbit(0, 150, 45.0)), c) == []
    assert codes(req("orbit", orbit(0, 150, 60.0)), c) == []


def test_orbit_radius_gate_other_cylinders_and_the_charter() -> None:
    """The ring keeps 10 m from every cylinder, and the whole ring stays in the charter."""
    k = cylinder(0, 150, radius=35.0)
    other = cylinder(120, 150, radius=31.0)  # 120 m east of the find
    c = with_keepouts(k, other)
    # Ring at 60..60.5 m: the other cylinder's edge is 89 m out, + 10 = 79 > 60.5: fine.
    assert codes(req("orbit", orbit(0, 150, 60.0)), c) == []
    # Ring at 80 m: within 10 m of the other cylinder's edge (89 m).
    assert codes(req("orbit", orbit(0, 150, 80.0)), c) == ["KEEPOUT_VIOLATION"]
    # The ring reaches past the western edge (-300): outside the charter, not a cylinder.
    assert codes(req("orbit", orbit(-200, 150, 120.0)), with_keepouts()) == ["OUTSIDE_CHARTER"]
    # The way to the ring passes through a cylinder.
    blocker = cylinder(0, 100, radius=31.0)
    d = evaluate(req("orbit", orbit(0, 200, 50.0)), with_keepouts(blocker))
    assert d.reason_codes == ["KEEPOUT_VIOLATION"]
    assert d.details["keepout_at"] == "route"


def test_ring_violation_band() -> None:
    k = cylinder(0, 0, radius=30.0)
    lat, lon = latlon(0, 0)
    for inner, outer, bad in (
        (39.0, 39.5, True),
        (40.0, 40.4, False),
        (100.0, 101.0, False),
    ):
        ring = OrbitRing(lat=lat, lon=lon, inner_m=inner, outer_m=outer)
        assert (ring_violation(ring, [k], PLANE) is not None) is bad
    assert ORBIT_KEEPOUT_MARGIN_M == 10.0


# --- inspect_detection through the gate -------------------------------------------------------


def inspect(east: float, north: float, **extra: Any) -> dict[str, Any]:
    lat, lon = latlon(east, north)
    base = {
        "detection_id": str(new_id()),
        "standoff_m": 15,
        "lat": lat,
        "lon": lon,
        "target_radius_m": 35.0,
    }
    return {**base, **extra}


def test_inspect_goes_to_the_cylinder_edge_and_passes() -> None:
    k = cylinder(0, 200, radius=35.0)
    c = with_keepouts(k)
    assert codes(req("inspect_detection", inspect(0, 200)), c) == []
    # Unresolved without the target's position (the ground fills it in).
    bare = {"detection_id": str(new_id()), "standoff_m": 20}
    assert "SCHEMA_INVALID" in codes(req("inspect_detection", bare), c)


def test_inspect_refused_when_the_way_there_crosses_a_cylinder() -> None:
    """Every direction around the target is blocked by a ring of other cylinders."""
    target = (0.0, 250.0)
    ring = [
        cylinder(target[0] + 60 * math.sin(a), target[1] + 60 * math.cos(a), radius=31.0)
        for a in (i * math.pi / 6 for i in range(12))
    ]
    c = with_keepouts(cylinder(*target), *ring)
    assert "SCHEMA_INVALID" in codes(req("inspect_detection", inspect(*target)), c)


# --- test_keepout_reroute --------------------------------------------------------------------

BOX = Polygon(((-400.0, -400.0), (400.0, -400.0), (400.0, 400.0), (-400.0, 400.0)))


def _check_reroute(route: list[Xy], cyl: list[tuple[Xy, float]], area: Polygon | None) -> None:
    out = reroute(route, cyl, area=area)
    if out is None:
        return  # no way around inside the area: the executor holds instead
    assert out[0] == route[0]
    assert xy_route_violation(out, cyl) is None
    if area is not None:
        assert all(area.leg_outside_by(a, b) <= 0.0 for a, b in pairwise(out))
    kept = [p for p in route[1:] if not any(math.dist(p, c) < r for c, r in cyl)]
    it = iter(out[1:])
    assert all(any(q == p for q in it) for p in kept)  # the corners kept, in their order


xy = st.tuples(st.floats(-300, 300), st.floats(-300, 300))


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    route=st.lists(xy, min_size=2, max_size=8),
    cyl=st.lists(st.tuples(xy, st.floats(30.5, 60.0)), min_size=1, max_size=3),
)
def test_keepout_reroute(route: list[Xy], cyl: list[tuple[Xy, float]]) -> None:
    """Property: a rerouted route never enters a cylinder, stays in the area and keeps
    every corner outside the cylinders in order; or there is none (None)."""
    if any(math.dist(route[0], c) < r for c, r in cyl):
        return  # starting inside: test_reroute_from_inside
    _check_reroute(route, cyl, BOX)
    _check_reroute(route, cyl, None)


def test_reroute_goes_around_on_the_short_side_with_the_margin() -> None:
    route: list[Xy] = [(0.0, -200.0), (0.0, 200.0)]
    cyl = [((10.0, 0.0), 40.0)]  # the leg passes 10 m west of the centre
    out = reroute(route, cyl, area=BOX)
    assert out is not None and len(out) > 2
    assert all(p[0] <= 0.0 for p in out)  # around the west side, the shorter one
    closest = min(math.dist(p, cyl[0][0]) for p in out[1:-1])
    assert closest >= 40.0 + KEEPOUT_ARC_MARGIN_M - 1e-6
    assert reroute([], cyl) == []
    assert reroute(route, []) == route


def test_reroute_takes_the_other_side_when_one_leaves_the_area() -> None:
    area = polygon_from_geojson(polygon((-30, -300), (300, -300), (300, 300), (-30, 300)), PLANE)
    route: list[Xy] = [(0.0, -200.0), (0.0, 200.0)]
    cyl = [((10.0, 0.0), 40.0)]
    out = reroute(route, cyl, area=area)
    assert out is not None
    assert all(p[0] >= 0.0 for p in out)  # around the east side: the west leaves the area
    # A leg kept as it was that leaves the area: no route to give.
    assert reroute([(0.0, -200.0), (0.0, 200.0), (500.0, 200.0)], cyl, area=area) is None
    narrow = polygon_from_geojson(polygon((-20, -300), (20, -300), (20, 300), (-20, 300)), PLANE)
    assert reroute(route, cyl, area=narrow) is None  # no way around inside


def test_reroute_from_inside() -> None:
    cyl = [((0.0, 0.0), 40.0)]
    # Inside, the next corner straight out: no extra exit needed.
    out = reroute([(0.0, 10.0), (0.0, 200.0)], cyl)
    assert out == [(0.0, 10.0), (0.0, 200.0)]
    # Inside, the next corner beyond the centre: out along the leg flown, then around.
    out = reroute([(0.0, 10.0), (0.0, -200.0), (100.0, -200.0)], cyl, exit_dir=(0.0, 1.0))
    assert out is not None
    exit_point = out[1]
    assert exit_point[0] == 0.0 and exit_point[1] > 40.0 + KEEPOUT_ARC_MARGIN_M
    assert xy_route_violation(out[1:], cyl) is None
    # Without a direction: straight out from the centre.
    out = reroute([(10.0, 0.0), (-200.0, 0.0)], cyl)
    assert out is not None and out[1][1] == 0.0 and out[1][0] > 45.0
    assert xy_route_violation(out[1:], cyl) is None
    # On the centre itself every way is out: the route stays as it is.
    assert reroute([(0.0, 0.0), (0.0, -200.0)], cyl) == [(0.0, 0.0), (0.0, -200.0)]
    # Every corner left is inside: nothing to fly.
    assert reroute([(0.0, 0.0), (0.0, 5.0)], cyl) == [(0.0, 0.0)]
    # A corner inside the cylinder is dropped; a single point stays as it is.
    out = reroute([(0.0, -200.0), (0.0, 0.0), (0.0, 200.0)], cyl)
    assert out is not None and (0.0, 0.0) not in out
    assert reroute([(0.0, -200.0)], cyl) == [(0.0, -200.0)]


def test_reroute_gives_up_on_cylinders_it_cannot_pass() -> None:
    """Overlapping cylinders that wall off the way, in a narrow area: no reroute."""
    wall = [((float(x), 0.0), 40.0) for x in range(-360, 361, 60)]
    assert reroute([(0.0, -200.0), (0.0, 200.0)], wall, area=BOX) is None


@pytest.mark.parametrize("sign", [1, -1])
def test_reroute_with_ends_inside_the_margin(sign: int) -> None:
    """Both ends between the cylinder and its margin: the way around starts straight out."""
    cyl = [((0.0, 0.0), 40.0)]
    route: list[Xy] = [(0.0, -43.0 * sign), (0.0, 43.0 * sign)]
    out = reroute(route, cyl, area=BOX)
    assert out is not None
    assert xy_route_violation(out, cyl) is None
