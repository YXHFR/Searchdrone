"""Property tests: random orders against random charters, concave ones included.

No ALLOW may break a limit. The check here is independent of the gate's geometry: it
samples each route densely and tests every sample with a winding-number test.
"""

from __future__ import annotations

import math
from typing import Any

from gate_fixtures import CHARTER, ctx, latlon, polygon, req, vehicle
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sd_safety import evaluate
from sd_safety.constants import BRAIN_BATTERY_MARGIN_PCT
from sd_schemas.gate import GateDecision
from sd_schemas.safety import SafetyLimits

Xy = tuple[float, float]


@st.composite
def star_polygons(draw: st.DrawFn) -> list[Xy]:
    """A simple polygon around home: corners at increasing angles, random radii (often concave)."""
    n = draw(st.integers(3, 10))
    cuts = sorted(draw(st.lists(st.floats(0, 1), min_size=n, max_size=n, unique=True)))
    corners: list[Xy] = []
    for i, cut in enumerate(cuts):
        angle = 2 * math.pi * (i + cut) / n
        r = draw(st.floats(30, 480))
        corners.append((r * math.cos(angle), r * math.sin(angle)))
    return corners


def _battery(i: int) -> float | None:
    """3 % unknown, 7 % near or below the reserve, the rest comfortably above it."""
    return None if i < 3 else (i * 2.5 if i < 10 else 21 + i * 0.8)


def _along(ring: list[Xy], corner: int, s: float) -> Xy:
    x, y = ring[corner % len(ring)]
    return x * s, y * s


def _inside(p: Xy, ring: list[Xy]) -> bool:
    """Winding number, with points on an edge counted inside (within 1 cm)."""
    winding = 0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1], strict=True):
        dx, dy = x2 - x1, y2 - y1
        t = max(0.0, min(1.0, ((p[0] - x1) * dx + (p[1] - y1) * dy) / (dx * dx + dy * dy)))
        if math.hypot(p[0] - x1 - t * dx, p[1] - y1 - t * dy) < 0.01:
            return True
        cross = dx * (p[1] - y1) - dy * (p[0] - x1)
        if y1 <= p[1] < y2 and cross > 0:
            winding += 1
        elif y2 <= p[1] < y1 and cross < 0:
            winding -= 1
    return winding != 0


def _leg_inside(a: Xy, b: Xy, ring: list[Xy]) -> bool:
    steps = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 2))
    return all(
        _inside((a[0] + (b[0] - a[0]) * i / steps, a[1] + (b[1] - a[1]) * i / steps), ring)
        for i in range(steps + 1)
    )


scale = st.floats(0, 1.15)  # along home -> a corner; > 1 leaves the (star-shaped) area


@settings(max_examples=10_000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    ring=star_polygons(),
    start_at=st.tuples(st.integers(0, 9), scale),
    target_at=st.tuples(st.integers(0, 9), scale),
    alt=st.floats(12, 140),
    speed=st.one_of(st.none(), st.floats(0.5, 30)),
    battery=st.integers(0, 99).map(_battery),
    max_alt=st.floats(40, 200),
    charter_alt=st.floats(40, 200),
    max_speed=st.floats(1, 25),
    max_range=st.floats(300, 1000),
    survey=st.booleans(),
)
def test_no_allow_breaks_a_limit(
    ring: list[Xy],
    start_at: tuple[int, float],
    target_at: tuple[int, float],
    alt: float,
    speed: float | None,
    battery: float | None,
    max_alt: float,
    charter_alt: float,
    max_speed: float,
    max_range: float,
    survey: bool,
) -> None:
    start, target = _along(ring, *start_at), _along(ring, *target_at)
    limits = SafetyLimits(
        max_alt_agl_m=max_alt, max_speed_ms=max_speed, max_range_from_pilot_m=max_range
    )
    charter = CHARTER.model_copy(
        update={
            "area": polygon(*ring),
            "alt_max_agl_m": charter_alt,
            "default_alt_agl_m": min(40.0, charter_alt),
            "max_speed_ms": max_speed * 0.8,
            "max_range_from_pilot_m": max_range * 0.9,
        }
    )
    lat, lon = latlon(*start)
    c = ctx(limits=limits, charter=charter, vehicle=vehicle(lat=lat, lon=lon, battery_pct=battery))
    if survey:
        e, n = target
        box = [(e, n), (e + 30, n), (e + 30, n + 30), (e, n + 30)]
        params: dict[str, Any] = {"polygon": polygon(*box), "alt_m": alt, "speed_ms": speed}
        decision = evaluate(req("survey_sector", params), c)
        legs = [(start, box[0]), *zip(box, box[1:] + box[:1], strict=True)]
    else:
        t_lat, t_lon = latlon(*target)
        params = {"lat": t_lat, "lon": t_lon, "alt_m": alt, "speed_ms": speed}
        decision = evaluate(req(params=params), c)
        legs = [(start, target)]
    _check(
        decision,
        legs,
        ring,
        alt,
        speed,
        battery,
        limits,
        charter_alt,
        max_speed * 0.8,
        max_range * 0.9,
    )


def _check(
    d: GateDecision,
    legs: list[tuple[Xy, Xy]],
    ring: list[Xy],
    alt: float,
    speed: float | None,
    battery: float | None,
    limits: SafetyLimits,
    charter_alt: float,
    charter_speed: float,
    charter_range: float,
) -> None:
    assert d.decision in ("ALLOW", "REJECT", "PENDING_APPROVAL")
    if d.decision != "ALLOW":
        assert d.reason_codes
        return
    assert alt <= limits.max_alt_agl_m and alt <= charter_alt and alt >= CHARTER.alt_min_agl_m
    assert speed is None or (speed <= limits.max_speed_ms and speed <= charter_speed)
    assert battery is not None and battery > CHARTER.battery_rtl_pct + BRAIN_BATTERY_MARGIN_PCT
    reach = min(limits.max_range_from_pilot_m, charter_range)
    for a, b in legs:
        assert _leg_inside(a, b, ring), (a, b)
        for p in (a, b):
            assert math.hypot(*p) <= reach * 1.001 + 0.1  # the plane is within 0.1 % here


@settings(max_examples=500, deadline=None)
@given(
    params=st.dictionaries(
        st.text(max_size=8),
        st.one_of(
            st.none(),
            st.integers(),
            st.floats(allow_nan=False),
            st.text(max_size=8),
            st.lists(st.integers(), max_size=3),
        ),
        max_size=6,
    ),
    skill=st.sampled_from(
        [
            "goto",
            "survey_sector",
            "takeoff",
            "mark_location",
            "report",
            "orbit",
            "propose_charter_change",
            "x",
        ]
    ),
)
def test_never_raises_and_is_deterministic(params: dict[str, Any], skill: str) -> None:
    r, c = req(skill, params), ctx()
    first = evaluate(r, c)
    assert evaluate(r, c) == first
    assert first.decision != "ALLOW" or skill in ("report",)
