"""choose_mode(): when the find hover applies, and where the drone hovers."""

from __future__ import annotations

import math
import random

import pytest
from gate_fixtures import CHARTER, PLANE, latlon

from sd_safety.constants import FIND_ENGAGE_RADIUS_M, FIND_HOVER_EDGE_MARGIN_M
from sd_safety.link_loss import (
    FIND_HOVER_UNTIL_MS,
    HOLD_MS,
    HOME_MS,
    FindPlan,
    NormalPlan,
    choose_mode,
    plan_times,
)
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.link import Keepout


def keepout(east: float, north: float, radius: float = 15.0, **kw: object) -> Keepout:
    lat, lon = latlon(east, north)
    data: dict[str, object] = {
        "keepout_id": new_id(),
        "lat": lat,
        "lon": lon,
        "radius_m": radius,
        "source": "confirmed",
        "source_id": new_id(),
    }
    data.update(kw)
    return Keepout.model_validate(data)


def choose(east: float, north: float, keepouts: list[Keepout], **kw: object) -> object:
    lat, lon = latlon(east, north)
    args: dict[str, object] = {
        "state": S.SURVEY,
        "lat": lat,
        "lon": lon,
        "alt_rel_home_m": 40.0,
        "nav_ok": True,
        "keepouts": keepouts,
        "charter": CHARTER,
    }
    args.update(kw)
    return choose_mode(**args)  # type: ignore[arg-type]


def test_choose_mode_find_within_100_m() -> None:
    k = keepout(0, 60)
    plan = choose(0, 0, [k])
    assert isinstance(plan, FindPlan)
    assert plan.keepout_id == k.keepout_id and plan.candidate_id == k.source_id
    hx, hy = PLANE.xy(plan.hover_lat, plan.hover_lon)
    assert math.isclose(hx, 0, abs_tol=0.01)
    assert math.isclose(hy, 60 - 15 - FIND_HOVER_EDGE_MARGIN_M, abs_tol=0.01)
    assert math.isclose(plan.yaw_deg, 0.0, abs_tol=0.01)  # facing north, at the find
    assert math.isclose(plan.distance_m, 40, abs_tol=0.01)
    assert plan.alt_m == 40.0


@pytest.mark.parametrize(("north", "find"), [(99.0, True), (101.0, False)])
def test_choose_mode_engage_radius(north: float, find: bool) -> None:
    assert isinstance(choose(0, 0, [keepout(0, north)]), FindPlan) is find


@pytest.mark.parametrize("state", [S.TRANSIT, S.SURVEY, S.HOLD])
def test_choose_mode_states_that_may_hover(state: S) -> None:
    assert isinstance(choose(0, 0, [keepout(0, 60)], state=state), FindPlan)


@pytest.mark.parametrize("state", [S.TAKEOFF, S.RTL, S.HELD, S.FAILSAFE, S.LANDED])
def test_choose_mode_other_states_are_normal(state: S) -> None:
    assert isinstance(choose(0, 0, [keepout(0, 60)], state=state), NormalPlan)


def test_choose_mode_needs_good_navigation() -> None:
    assert isinstance(choose(0, 0, [keepout(0, 60)], nav_ok=False), NormalPlan)


@pytest.mark.parametrize("source", ["detection", "candidate"])
def test_choose_mode_only_confirmed_finds(source: str) -> None:
    assert isinstance(choose(0, 0, [keepout(0, 60, source=source)]), NormalPlan)


def test_choose_mode_hover_point_outside_the_charter() -> None:
    # The notch (east 100..300, north 100..300) is outside the area; buffer 20 m. A find
    # deep in the notch puts the hover point there too.
    assert isinstance(choose(200, 140, [keepout(200, 230, radius=10)]), NormalPlan)
    assert not isinstance(choose(0, 0, []), FindPlan)


def test_choose_mode_nearest_then_lowest_id() -> None:
    near, far = keepout(0, 50), keepout(0, -80)
    assert choose(0, 0, [far, near]).keepout_id == near.keepout_id  # type: ignore[attr-defined]
    assert choose(0, 0, [near, far]).keepout_id == near.keepout_id  # type: ignore[attr-defined]
    first, second = sorted((new_id(), new_id()), key=str)
    a = keepout(0, 50, keepout_id=second)
    b = keepout(0, -50, keepout_id=first)
    assert choose(0, 0, [a, b]).keepout_id == b.keepout_id  # type: ignore[attr-defined]


def test_choose_mode_on_the_centre_hovers_towards_home() -> None:
    plan = choose(0, 80, [keepout(0, 80, radius=10)])
    assert isinstance(plan, FindPlan)
    hx, hy = PLANE.xy(plan.hover_lat, plan.hover_lon)
    assert math.isclose(hx, 0, abs_tol=0.01) and math.isclose(hy, 80 - 15, abs_tol=0.01)
    assert math.isclose(plan.yaw_deg, 0.0, abs_tol=0.01)


def test_choose_mode_on_the_centre_of_a_find_on_home_hovers_north() -> None:
    plan = choose(0, 0, [keepout(0, 0, radius=10)])
    assert isinstance(plan, FindPlan)
    hx, hy = PLANE.xy(plan.hover_lat, plan.hover_lon)
    assert math.isclose(hx, 0, abs_tol=0.01) and math.isclose(hy, 15, abs_tol=0.01)
    assert math.isclose(plan.yaw_deg, 180.0, abs_tol=0.01)


def test_choose_mode_clamps_the_height() -> None:
    low = choose(0, 0, [keepout(0, 60)], alt_rel_home_m=5.0)
    high = choose(0, 0, [keepout(0, 60)], alt_rel_home_m=500.0)
    assert low.alt_m == CHARTER.alt_min_agl_m  # type: ignore[attr-defined]
    assert high.alt_m == CHARTER.alt_max_agl_m  # type: ignore[attr-defined]


def _segment_point_distance(
    a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]
) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2))
    return math.dist((ax + t * dx, ay + t * dy), p)


def test_hover_point_outside_keepout() -> None:
    """On r + 5 m from the find, and the straight way there never enters the cylinder
    when the drone starts outside it (1 000 random cases, fixed seed)."""
    rnd = random.Random(1019)  # noqa: S311 - test data
    checked = 0
    for _ in range(1000):
        ce, cn = rnd.uniform(-250, 50), rnd.uniform(-250, 50)
        r = rnd.uniform(2, 40)
        angle = rnd.uniform(0, 2 * math.pi)
        dist = rnd.uniform(0, FIND_ENGAGE_RADIUS_M)
        me = (ce + dist * math.cos(angle), cn + dist * math.sin(angle))
        k = keepout(ce, cn, radius=r)
        plan = choose(me[0], me[1], [k])
        if not isinstance(plan, FindPlan):
            continue
        checked += 1
        c = PLANE.xy(k.lat, k.lon)
        p = PLANE.xy(plan.hover_lat, plan.hover_lon)
        assert math.isclose(math.dist(c, p), r + FIND_HOVER_EDGE_MARGIN_M, abs_tol=0.05)
        start = PLANE.xy(*latlon(*me))
        if math.dist(start, c) >= r:
            assert _segment_point_distance(start, p, c) >= r - 0.05
        bearing = math.degrees(math.atan2(c[0] - p[0], c[1] - p[1])) % 360
        assert abs((plan.yaw_deg - bearing + 180) % 360 - 180) < 0.1
    assert checked > 900


def test_plan_times() -> None:
    normal = plan_times(NormalPlan(), 1000)
    assert (normal.hold_ms, normal.home_ms, normal.hover_until_ms) == (
        1000 + HOLD_MS,
        1000 + HOME_MS,
        None,
    )
    assert (HOLD_MS, HOME_MS, FIND_HOVER_UNTIL_MS) == (23_000, 53_000, 123_000)
    find = choose(0, 0, [keepout(0, 60)])
    assert isinstance(find, FindPlan)
    t = plan_times(find, 1000)
    assert t.hold_ms is None and t.home_ms == t.hover_until_ms == 124_000
