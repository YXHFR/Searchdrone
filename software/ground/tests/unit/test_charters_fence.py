"""Charter checks, sectors, the fence polygon and parameters, and limit changes (pure)."""

from __future__ import annotations

from typing import Any

import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from sd_ground.services import charters, fence, limits
from sd_safety.targets import Plane
from sd_schemas.base import LatLonAlt, new_id
from sd_schemas.charter import SECTOR_MAX_AREA_M2, AutonomyMode, CharterProposal, MissionCharter
from sd_schemas.safety import LimitCaps, SafetyLimits
from sd_schemas.skills import SKILL_REGISTRY

CMAC = LatLonAlt(lat=-35.363261, lon=149.165230, alt_amsl_m=584.0)
PLANE = Plane(CMAC.lat, CMAC.lon)
REG = SKILL_REGISTRY.as_mapping()


def square(half: float, east: float = 0.0, north: float = 0.0) -> dict[str, Any]:
    ring = []
    for x, y in [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)]:
        lat, lon = PLANE.latlon(x + east, y + north)
        ring.append([lon, lat])
    return {"type": "Polygon", "coordinates": [ring]}


def make(area: dict[str, Any], lim: SafetyLimits | None = None, **prop: Any) -> MissionCharter:
    proposal = CharterProposal(area=area, reason_sv="test", **prop)
    return charters.build(
        proposal,
        area=area,
        charter_id=str(new_id()),
        mission_id=str(new_id()),
        vehicle_id="sim-01",
        version=1,
        home=CMAC,
        limits=lim or SafetyLimits(),
        registry=SKILL_REGISTRY,
        autonomy_mode=AutonomyMode.WITHIN_CHARTER,
    )


def codes(c: MissionCharter, lim: SafetyLimits | None = None, **kw: Any) -> set[str]:
    args: dict[str, Any] = {
        "limits": lim or SafetyLimits(),
        "registry": REG,
        "max_area_km2": 4.0,
        "active": None,
        "base_version": None,
        "airborne": False,
    } | kw
    return {e.code for e in charters.validate(c, **args)}


def test_demo_charter_is_valid() -> None:
    c = make(square(200))
    assert codes(c) == set()
    assert [s.sector_id for s in c.sectors] == ["A"]
    assert c.pilot_position.lat == CMAC.lat
    assert c.alt_max_agl_m == 120


def test_home_outside_area() -> None:
    assert "HOME_OUTSIDE_AREA" in codes(make(square(50, east=200)))


def test_area_beyond_range_and_fence() -> None:
    got = codes(make(square(450)))
    assert {"AREA_BEYOND_RANGE", "AREA_BEYOND_FC_FENCE", "BUFFER_INVALID"} <= got


def test_raised_limits_allow_a_larger_area() -> None:
    lim = SafetyLimits(max_range_from_pilot_m=1500)
    c = make(square(450), lim)
    assert codes(c, lim) == set()


def test_charter_looser_than_limits() -> None:
    c = make(square(100), max_speed_ms=14, alt_max_agl_m=100)
    strict = SafetyLimits(max_speed_ms=10, max_alt_agl_m=90)
    assert "CHARTER_EXCEEDS_LIMITS" in codes(c, strict)


def test_alt_band_invalid() -> None:
    with pytest.raises(charters.CharterInvalidError) as err:
        make(square(100), alt_min_agl_m=50, default_alt_agl_m=40, alt_max_agl_m=60)
    assert err.value.errors[0].code == "ALT_BAND_INVALID"


def test_stub_skill_refused() -> None:
    c = make(square(100), allowed_skills=["goto", "orbit"])
    assert "SKILL_NOT_IMPLEMENTED" in codes(c)
    c = make(square(100), allowed_skills=["goto"])
    c2 = c.model_copy(update={"allowed_skills": ["teleport"]})
    assert "SKILL_UNKNOWN" in codes(c2)


def test_bad_polygon_and_too_big() -> None:
    bow = square(100)
    ring = bow["coordinates"][0]
    ring[1], ring[2] = ring[2], ring[1]  # self-crossing
    c = make(square(100)).model_copy(update={"area": bow})
    assert "AREA_INVALID" in codes(c)
    assert "AREA_INVALID" in codes(make(square(200)), max_area_km2=0.1)


def test_base_version_and_in_air_growth() -> None:
    old = make(square(100))
    bigger = make(square(150))
    assert "BASE_VERSION_STALE" in codes(bigger, active=old, base_version=None)
    assert "AREA_CHANGE_REQUIRES_LANDING" in codes(
        bigger, active=old, base_version=1, airborne=True
    )
    smaller = make(square(80))
    assert codes(smaller, active=old, base_version=1, airborne=True) == set()


def test_sectors_split_large_area() -> None:
    lim = SafetyLimits(max_range_from_pilot_m=2000)
    c = make(square(600), lim)
    ids = [s.sector_id for s in c.sectors]
    assert ids[0] == "A" and len(ids) == len(set(ids)) >= 4
    total = 0.0
    for s in c.sectors:
        a = charters.sector_area_m2(s.area, PLANE)
        assert a <= SECTOR_MAX_AREA_M2
        total += a
    assert total == pytest.approx(1200 * 1200, rel=1e-3)
    assert codes(c, lim) == set()


def test_sector_outside_area() -> None:
    c = make(square(100))
    bad = c.model_copy(
        update={"sectors": [c.sectors[0].model_copy(update={"area": square(50, east=300)})]}
    )
    assert "SECTOR_OUTSIDE_AREA" in codes(bad)


def test_resolve_autonomy() -> None:
    a = make(square(100)).model_copy(update={"autonomy_mode": AutonomyMode.APPROVE_EACH})
    assert charters.resolve_autonomy(None, None) is AutonomyMode.WITHIN_CHARTER
    assert charters.resolve_autonomy(a, None) is AutonomyMode.APPROVE_EACH
    assert charters.resolve_autonomy(a, AutonomyMode.WITHIN_CHARTER) is AutonomyMode.WITHIN_CHARTER


# --- fence ----------------------------------------------------------------------------


def _xy(corners: list[tuple[float, float]]) -> ShapelyPolygon:
    return ShapelyPolygon([PLANE.xy(lat, lon) for lat, lon in corners])


def test_fence_polygon_covers_area_plus_buffer() -> None:
    c = make(square(200))
    corners = fence.fence_polygon(c)
    assert 3 <= len(corners) <= fence.FENCE_MAX_VERTICES
    poly = _xy(corners)
    area = charters.polygon_xy(c.area, PLANE)
    assert poly.covers(area.buffer(c.geofence_buffer_m))
    assert not poly.covers(area.buffer(c.geofence_buffer_m + 11))


def test_fence_polygon_simplified_for_many_corners() -> None:
    ring = []
    for i in range(150):
        import math

        a = 2 * math.pi * i / 150
        r = 150 + (20 if i % 2 else 0)
        lat, lon = PLANE.latlon(r * math.cos(a), r * math.sin(a))
        ring.append([lon, lat])
    ring.append(ring[0])
    c = make(square(100)).model_copy(update={"area": {"type": "Polygon", "coordinates": [ring]}})
    corners = fence.fence_polygon(c)
    assert len(corners) <= fence.FENCE_MAX_VERTICES
    area = charters.polygon_xy(c.area, PLANE)
    assert _xy(corners).covers(area.buffer(c.geofence_buffer_m))


GOOD = {
    "FENCE_ENABLE": 1,
    "FENCE_TYPE": 7,
    "FENCE_ACTION": 3,
    "FENCE_ALT_MAX": 125,
    "FENCE_RADIUS": 530,
    "LOIT_SPEED_MS": 15,
    "FS_GCS_ENABLE": 3,
    "FS_GCS_TIMEOUT": 20,
    "MAV_GCS_SYSID": 250,
    "BATT_FS_LOW_ACT": 3,
    "BATT_FS_CRT_ACT": 1,
}


def test_params_standard_ok() -> None:
    assert fence.check_fc_safety_params(GOOD, SafetyLimits()) == []
    assert fence.check_fc_safety_params(GOOD, SafetyLimits(max_alt_agl_m=60)) == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("FENCE_ENABLE", 0),
        ("FENCE_TYPE", 3),
        ("FENCE_ACTION", 0),
        ("FENCE_ALT_MAX", 1000),
        ("FENCE_ALT_MAX", 100),
        ("FENCE_RADIUS", 5000),
        ("FS_GCS_ENABLE", 0),
        ("FS_GCS_TIMEOUT", 5),
        ("MAV_GCS_SYSID", 255),
        ("BATT_FS_LOW_ACT", 0),
        ("BATT_FS_CRT_ACT", 0),
        ("LOIT_SPEED_MS", None),
    ],
)
def test_params_violations(name: str, value: float | None) -> None:
    bad = GOOD | {name: value}
    got = fence.check_fc_safety_params(bad, SafetyLimits())
    assert [v.param for v in got] == [name]


def test_raised_limits_move_the_fence() -> None:
    raised = SafetyLimits(max_alt_agl_m=200, max_speed_ms=20, max_range_from_pilot_m=1000)
    t = fence.fence_param_targets(raised)
    assert t == {"FENCE_ALT_MAX": 205, "FENCE_RADIUS": 1030, "LOIT_SPEED_MS": 20}
    assert fence.fence_param_targets(SafetyLimits(max_alt_agl_m=50))["FENCE_ALT_MAX"] == 125
    params = GOOD | t
    assert fence.check_fc_safety_params(params, raised) == []
    assert fence.check_fc_safety_params(GOOD, raised) != []


# --- limits ---------------------------------------------------------------------------


def test_limits_change_rules() -> None:
    caps = LimitCaps()
    std = SafetyLimits()
    assert (
        limits.check_change(
            SafetyLimits(max_alt_agl_m=80), old=std, caps=caps, active=None, in_flight=True
        )
        is None
    )
    above = limits.check_change(
        SafetyLimits(max_alt_agl_m=400), old=std, caps=caps, active=None, in_flight=False
    )
    assert above is not None and above.status == 422 and above.code == "LIMIT_ABOVE_ABSOLUTE"
    raised = SafetyLimits(max_alt_agl_m=200)
    assert limits.check_change(raised, old=std, caps=caps, active=None, in_flight=False) is None
    air = limits.check_change(raised, old=std, caps=caps, active=None, in_flight=True)
    assert air is not None and air.code == "LIMITS_CHANGE_REQUIRES_LANDING"
    active = make(square(100))
    below = limits.check_change(
        SafetyLimits(max_alt_agl_m=60), old=std, caps=caps, active=active, in_flight=False
    )
    assert below is not None and below.status == 409 and below.code == "LIMITS_BELOW_ACTIVE_CHARTER"
