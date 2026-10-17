"""Charter drafts become proposals with sectors (05 §9.2.1)."""

from __future__ import annotations

import math

import pytest
from brain_helpers import situation
from shapely.geometry import Point, Polygon

from sd_brain.charter_draft import AreaRef, CharterDraft, DraftError, to_proposal
from sd_brain.geo import bearing_word, from_enu, to_enu
from sd_schemas.base import LatLon

H = (-35.363261, 149.165230)


def _poly(geo: dict) -> Polygon:
    return Polygon([to_enu(lat, lon, *H) for lon, lat in geo["coordinates"][0]])


def _draft(*areas: AreaRef, **kw) -> CharterDraft:
    return CharterDraft(search_areas=list(areas), reason_sv="test", **kw)


def test_rect_around_pin() -> None:
    p = to_proposal(
        _draft(AreaRef(kind="rect", center_pin_id="P1", width_m=100, height_m=100)),
        situation("draft_p1.json"),
    )
    assert p.sectors is not None and [s.sector_id for s in p.sectors] == ["A"]
    sector = _poly(p.sectors[0].area)
    assert sector.area == pytest.approx(10_000, abs=1)
    assert p.area is not None
    area = _poly(p.area)
    assert area.contains(Point(0, 0)) and area.contains(sector)


def test_circle() -> None:
    p = to_proposal(
        _draft(AreaRef(kind="circle", center_pin_id="P1", radius_m=50)), situation("draft_p1.json")
    )
    assert p.sectors is not None
    assert _poly(p.sectors[0].area).area == pytest.approx(math.pi * 50**2, rel=0.01)


def test_large_rect_is_split() -> None:
    p = to_proposal(
        _draft(AreaRef(kind="rect", center_pin_id="P1", width_m=600, height_m=600)),
        situation("draft_p1.json"),
    )
    assert p.sectors is not None
    assert [s.sector_id for s in p.sectors] == ["A", "B"]
    areas = [_poly(s.area).area for s in p.sectors]
    assert all(a <= 250_000 for a in areas)
    assert sum(areas) == pytest.approx(360_000, abs=1)


def _sector(area: AreaRef) -> Polygon:
    sectors = to_proposal(_draft(area), situation("draft_p1.json")).sectors
    assert sectors is not None
    return _poly(sectors[0].area)


def test_heading_turns_the_rect() -> None:
    a = _sector(AreaRef(kind="rect", center_pin_id="P1", width_m=100, height_m=300))
    b = _sector(AreaRef(kind="rect", center_pin_id="P1", width_m=100, height_m=300, heading_deg=90))
    assert a.area == pytest.approx(b.area, rel=1e-5)
    ax0, _, ax1, _ = a.bounds
    _, by0, _, by1 = b.bounds
    assert ax1 - ax0 == pytest.approx(by1 - by0, abs=0.5)


def test_drawn_operating_area_and_none_fields() -> None:
    s = situation("draft_p1.json")
    d = _draft(
        AreaRef(kind="drawn", area_id="ar-1"), operating_area=AreaRef(kind="drawn", area_id="ar-1")
    )
    p = to_proposal(d, s)
    assert p.area is None and p.area_annotation_id is not None
    body = p.model_dump(mode="json", exclude_none=True)
    assert "alt_max_agl_m" not in body and "battery_rtl_pct" not in body
    assert body["alt_min_agl_m"] == 15


def test_errors() -> None:
    s = situation("draft_p1.json")
    with pytest.raises(DraftError, match="okänd nål"):
        to_proposal(_draft(AreaRef(kind="rect", center_pin_id="P7", width_m=100, height_m=100)), s)
    with pytest.raises(DraftError, match="okänt område"):
        to_proposal(_draft(AreaRef(kind="drawn", area_id="ar-9")), s)
    with pytest.raises(DraftError):
        to_proposal(_draft(AreaRef(kind="rect", center_pin_id="P1")), s)
    with pytest.raises(DraftError):
        to_proposal(_draft(AreaRef(kind="circle", center_pin_id="P1")), s)
    with pytest.raises(DraftError, match="battery_rtl_pct"):
        to_proposal(
            _draft(
                AreaRef(kind="rect", center_pin_id="P1", width_m=100, height_m=100),
                battery_rtl_pct=5,
            ),
            s,
        )
    with pytest.raises(ValueError):
        CharterDraft.model_validate(
            {
                "search_areas": [{"kind": "drawn", "area_id": "ar-1"}],
                "reason_sv": "x",
                "autonomy_mode": "WITHIN_CHARTER",
            }
        )


def test_rect_from_coordinates() -> None:
    lat, lon = from_enu(0, 100, *H)
    p = to_proposal(
        _draft(AreaRef(kind="rect", center=LatLon(lat=lat, lon=lon), width_m=50, height_m=50)),
        situation("draft_p1.json"),
    )  # type: ignore[arg-type]
    assert p.sectors is not None and len(p.sectors) == 1


def test_geo_helpers() -> None:
    lat, lon = from_enu(100, 200, *H)
    e, n = to_enu(lat, lon, *H)
    assert e == pytest.approx(100, abs=0.01) and n == pytest.approx(200, abs=0.01)
    assert bearing_word(0, 10) == "norr" and bearing_word(10, 10) == "nordost"
