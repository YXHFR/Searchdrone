"""Charter drafts become proposals with sectors (05 §9.2.1)."""

from __future__ import annotations

import math

import pytest
from brain_helpers import situation
from pydantic import ValidationError
from shapely.geometry import Point, Polygon

from sd_brain.charter_draft import (
    CharterDraft,
    CircleArea,
    DraftError,
    DrawnArea,
    RectArea,
    to_proposal,
)
from sd_brain.geo import bearing_word, from_enu, to_enu
from sd_schemas.base import LatLon
from sd_schemas.commander import Situation

H = (-35.363261, 149.165230)


def _poly(geo: dict) -> Polygon:
    return Polygon([to_enu(lat, lon, *H) for lon, lat in geo["coordinates"][0]])


def _draft(*areas: DrawnArea | RectArea | CircleArea, **kw) -> CharterDraft:
    return CharterDraft(search_areas=list(areas), reason_sv="test", **kw)


def _wide() -> Situation:
    """draft_p1 with a range that fits a large search area."""
    return situation("draft_p1.json", limits={"max_range_from_pilot_m": 2000.0})


def _with_big_area(half_m: float = 300.0) -> Situation:
    """draft_p1 plus a drawn area "ar-2", a square of 2 x ``half_m`` around home (ar-1 is
    a 100 m square around P1, without home)."""
    s = situation("draft_p1.json")
    corners = ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))
    coords = []
    for e, n in corners:
        lat, lon = from_enu(e * half_m, n * half_m, *H)
        coords.append([lon, lat])
    big = s.areas[0].model_copy(
        update={
            "id": "0192f49e-2000-7000-8000-0000000000a2",
            "label": "ar-2",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        }
    )
    return s.model_copy(update={"areas": [*s.areas, big]})


def test_rect_around_pin() -> None:
    p = to_proposal(
        _draft(RectArea(kind="rect", center_pin_id="P1", width_m=100, height_m=100)),
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
        _draft(CircleArea(kind="circle", center_pin_id="P1", radius_m=50)),
        situation("draft_p1.json"),
    )
    assert p.sectors is not None
    assert _poly(p.sectors[0].area).area == pytest.approx(math.pi * 50**2, rel=0.01)


def test_large_rect_is_split() -> None:
    p = to_proposal(
        _draft(RectArea(kind="rect", center_pin_id="P1", width_m=600, height_m=600)),
        _wide(),
    )
    assert p.sectors is not None
    assert [s.sector_id for s in p.sectors] == ["A", "B"]
    areas = [_poly(s.area).area for s in p.sectors]
    assert all(a <= 250_000 for a in areas)
    assert sum(areas) == pytest.approx(360_000, abs=1)


def _sector(area: RectArea) -> Polygon:
    sectors = to_proposal(_draft(area), situation("draft_p1.json")).sectors
    assert sectors is not None
    return _poly(sectors[0].area)


def test_heading_turns_the_rect() -> None:
    a = _sector(RectArea(kind="rect", center_pin_id="P1", width_m=100, height_m=300))
    b = _sector(
        RectArea(kind="rect", center_pin_id="P1", width_m=100, height_m=300, heading_deg=90)
    )
    assert a.area == pytest.approx(b.area, rel=1e-5)
    ax0, _, ax1, _ = a.bounds
    _, by0, _, by1 = b.bounds
    assert ax1 - ax0 == pytest.approx(by1 - by0, abs=0.5)


def test_drawn_operating_area_and_none_fields() -> None:
    s = _with_big_area()
    d = _draft(
        DrawnArea(kind="drawn", area_id="ar-1"),
        operating_area=DrawnArea(kind="drawn", area_id="ar-2"),
    )
    p = to_proposal(d, s)
    assert str(p.area_annotation_id) == "0192f49e-2000-7000-8000-0000000000a2"
    assert p.area is None and p.area_annotation_id is not None
    body = p.model_dump(mode="json", exclude_none=True)
    assert "alt_max_agl_m" not in body and "battery_rtl_pct" not in body
    assert body["alt_min_agl_m"] == 15


def test_errors_name_the_field_and_end_with_a_full_stop() -> None:
    s = situation("draft_p1.json")
    with pytest.raises(DraftError, match="Okänd nål P7") as exc:
        to_proposal(_draft(RectArea(kind="rect", center_pin_id="P7", width_m=100, height_m=100)), s)
    assert exc.value.field == "search_areas.0" and str(exc.value).endswith(".")
    with pytest.raises(DraftError, match="Okänt område") as exc:
        to_proposal(_draft(DrawnArea(kind="drawn", area_id="ar-9")), s)
    assert str(exc.value).endswith(".")
    with pytest.raises(DraftError, match="battery_rtl_pct") as exc:
        to_proposal(
            _draft(
                RectArea(kind="rect", center_pin_id="P1", width_m=100, height_m=100),
                battery_rtl_pct=5,
            ),
            s,
        )
    assert exc.value.field == "battery_rtl_pct" and str(exc.value).endswith(".")


def test_kind_selects_the_shape() -> None:
    """A rect without its size, a circle without its radius or a drawn area without its
    label never validates (the five recorded drafts: two had no width_m/height_m)."""
    for bad in (
        {"kind": "rect", "center_pin_id": "P1"},
        {"kind": "rect", "center_pin_id": "P1", "radius_m": 50},
        {"kind": "circle", "center_pin_id": "P1"},
        {"kind": "drawn"},
        {"kind": "rect", "width_m": 100, "height_m": 100},  # no centre
        {"kind": "rect", "area_id": "ar-1", "width_m": 100, "height_m": 100},
    ):
        with pytest.raises(ValidationError):
            CharterDraft.model_validate({"search_areas": [bad], "reason_sv": "x"})
    with pytest.raises(ValidationError):
        CharterDraft.model_validate(
            {
                "search_areas": [{"kind": "drawn", "area_id": "ar-1"}],
                "reason_sv": "x",
                "autonomy_mode": "WITHIN_CHARTER",
            }
        )


def test_the_ai_cannot_draw_the_flight_area() -> None:
    """D1: operating_area is only an area the operator drew."""
    for op in (
        {"kind": "rect", "center_pin_id": "P1", "width_m": 200, "height_m": 200},
        {"kind": "rect", "center": {"lat": 0, "lon": 0}, "width_m": 200, "height_m": 200},
        {"kind": "circle", "center_pin_id": "P1", "radius_m": 300},
    ):
        with pytest.raises(ValidationError, match="operating_area"):
            CharterDraft.model_validate(
                {
                    "search_areas": [
                        {"kind": "rect", "center_pin_id": "P1", "width_m": 100, "height_m": 100}
                    ],
                    "operating_area": op,
                    "reason_sv": "x",
                }
            )


def test_drawn_flight_area_without_home_is_refused() -> None:
    s = situation("draft_p1.json")  # ar-1: a 100 m square around P1, 200 m north of home
    with pytest.raises(DraftError, match="Hemplatsen ligger utanför flygområdet") as exc:
        to_proposal(
            _draft(
                DrawnArea(kind="drawn", area_id="ar-1"),
                operating_area=DrawnArea(kind="drawn", area_id="ar-1"),
            ),
            s,
        )
    assert exc.value.field == "operating_area"


def test_search_area_outside_a_drawn_flight_area_is_refused() -> None:
    with pytest.raises(DraftError, match="Ett sökområde ligger utanför flygområdet"):
        to_proposal(
            _draft(
                RectArea(kind="rect", center_pin_id="P1", width_m=400, height_m=400),
                operating_area=DrawnArea(kind="drawn", area_id="ar-2"),
            ),
            _with_big_area(),
        )


def test_area_beyond_the_range_is_refused() -> None:
    s = situation("draft_p1.json")  # 500 m from the pilot
    with pytest.raises(DraftError, match="högst 500 m") as exc:
        to_proposal(_draft(RectArea(kind="rect", center_pin_id="P1", width_m=600, height_m=600)), s)
    assert exc.value.field == "search_areas"
    lat0 = LatLon(lat=0, lon=0)
    with pytest.raises(DraftError, match="från piloten"):
        to_proposal(_draft(RectArea(kind="rect", center=lat0, width_m=100, height_m=100)), s)
    with pytest.raises(DraftError, match="högst 250 m"):
        to_proposal(
            _draft(
                RectArea(kind="rect", center_pin_id="P1", width_m=100, height_m=100),
                max_range_from_pilot_m=250,
            ),
            s,
        )


def test_rect_from_coordinates() -> None:
    lat, lon = from_enu(0, 100, *H)
    p = to_proposal(
        _draft(RectArea(kind="rect", center=LatLon(lat=lat, lon=lon), width_m=50, height_m=50)),
        situation("draft_p1.json"),
    )
    assert p.sectors is not None and len(p.sectors) == 1


def test_geo_helpers() -> None:
    lat, lon = from_enu(100, 200, *H)
    e, n = to_enu(lat, lon, *H)
    assert e == pytest.approx(100, abs=0.01) and n == pytest.approx(200, abs=0.01)
    assert bearing_word(0, 10) == "norr" and bearing_word(10, 10) == "nordost"
