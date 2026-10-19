"""The charter draft the AI fills in, and turning it into the ground station's proposal.

The model never writes polygons: it names a drawn area, or a rectangle or circle around
a pin. ``kind`` selects the shape and the fields it needs (a rectangle needs ``width_m``
and ``height_m``, a circle ``radius_m``, a drawn area ``area_id``). The flight area
(``operating_area``) is computed here around home, the pilot and the search areas; the
model may only name an area the operator drew for it (owner decision). :func:`to_proposal`
turns the draft into a ``CharterProposal`` with sectors A, B, … and refuses what the
ground station would refuse for certain (home or a search area outside the flight area,
an area beyond the range), with a Swedish text for the operator, the ground's reason code
and a hint for the model. It has no field for the autonomy mode or the mission limits.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError, model_validator
from shapely.geometry import MultiPoint, Polygon, box
from shapely.geometry import Point as ShPoint
from shapely.ops import unary_union

from sd_brain.geo import from_enu, to_enu
from sd_schemas.base import Id, LatLon, SdModel
from sd_schemas.charter import SECTOR_MAX_AREA_M2, CharterProposal, Sector
from sd_schemas.commander import Situation

__all__ = [
    "AreaRef",
    "CharterDraft",
    "CharterError",
    "CharterProposalOutcome",
    "CircleArea",
    "DraftError",
    "DrawnArea",
    "RectArea",
    "error_path",
    "to_proposal",
]

SPLIT_TARGET_M2 = 240_000.0
MERGE_BELOW_M2 = 100.0
OPERATING_BUFFER_M = 50.0
MAX_HULL_CORNERS = 200
# The ground measures geodesic distances, this module a flat plane: a metre of slack so a
# draft at the very edge of the range is left to the ground station's own check.
RANGE_SLACK_M = 1.0


class DraftError(ValueError):
    """The draft cannot be turned into a proposal (sent back to the model as a tool error).

    The text is plain Swedish for the operator. ``field`` names the draft field at fault
    when there is one (e.g. ``search_areas.0``), ``code`` the ground station's reason code
    for the same fault, ``hint`` how the model fixes it (English, for the model only), and
    ``repairable = False`` a fault the model cannot fix (e.g. home not known yet).
    """

    def __init__(
        self,
        text: str,
        field: str | None = None,
        *,
        code: str = "DRAFT_INVALID",
        hint: str = "",
        repairable: bool = True,
    ) -> None:
        super().__init__(text)
        self.field = field
        self.code = code
        self.hint = hint
        self.repairable = repairable


def _pin_id() -> Any:
    return Field(
        default=None,
        pattern=r"^P[0-9]{1,3}$",
        description="The map pin at the centre (Kartnålar in LÄGE), e.g. P1.",
    )


def _center_field() -> Any:
    return Field(
        default=None,
        description="Centre as coordinates; only when the operator wrote coordinates, "
        "otherwise use center_pin_id.",
    )


class DrawnArea(SdModel):
    kind: Literal["drawn"] = Field(description="An area the operator drew on the map.")
    area_id: str = Field(
        max_length=40, description="The drawn area's label (Ritade områden in LÄGE), e.g. ar-1."
    )


class _Centered(SdModel):
    @model_validator(mode="after")
    def _one_center(self) -> Any:
        if (getattr(self, "center_pin_id", None) is None) == (
            getattr(self, "center", None) is None
        ):
            raise ValueError("give center_pin_id (or center when the operator wrote coordinates)")
        return self


class RectArea(_Centered):
    kind: Literal["rect"] = Field(description="A rectangle around a pin.")
    center_pin_id: str | None = _pin_id()
    center: LatLon | None = _center_field()
    width_m: float = Field(
        ge=20, le=2000, description="Side along heading_deg in metres (required)."
    )
    height_m: float = Field(
        ge=20, le=2000, description="Side across heading_deg in metres (required)."
    )
    heading_deg: float = Field(
        default=0,
        ge=0,
        le=180,
        description="Direction of width_m, degrees clockwise from north (0 = north-south).",
    )


class CircleArea(_Centered):
    kind: Literal["circle"] = Field(description="A circle around a pin.")
    center_pin_id: str | None = _pin_id()
    center: LatLon | None = _center_field()
    radius_m: float = Field(ge=20, le=1000, description="Radius in metres (required).")


AreaRef = Annotated[DrawnArea | RectArea | CircleArea, Field(discriminator="kind")]


class CharterDraft(SdModel):
    base_version: int | None = Field(
        default=None,
        description="Version of the charter in force that this changes; leave out for the "
        "first charter.",
    )
    operating_area: DrawnArea | None = Field(
        default=None,
        description="Leave out: the system computes the flight area around home, the pilot "
        "and the search areas. Only when the operator drew a flight area and asks for it: "
        "that drawn area.",
    )
    search_areas: list[AreaRef] = Field(
        min_length=1,
        max_length=4,
        description="The areas to search: a drawn area, or a rect or circle around a pin.",
    )
    alt_min_agl_m: float | None = Field(
        default=None, ge=0, le=1000, description="Lowest height in metres above home."
    )
    alt_max_agl_m: float | None = Field(
        default=None, ge=0, le=1000, description="Highest height in metres above home."
    )
    default_alt_agl_m: float | None = Field(
        default=None, ge=0, le=1000, description="Search height in metres above home."
    )
    max_speed_ms: float | None = Field(
        default=None, ge=0, le=100, description="Highest speed in m/s."
    )
    max_range_from_pilot_m: float | None = Field(
        default=None, ge=0, le=10_000, description="Farthest distance from the pilot in metres."
    )
    battery_rtl_pct: float | None = Field(
        default=None, ge=0, le=100, description="Battery percent at which the drone flies home."
    )
    allowed_skills: list[str] | None = Field(
        default=None,
        max_length=32,
        description="Leave out for the default skills. Otherwise a subset of the skill "
        "names listed.",
    )
    reason_sv: str = Field(
        min_length=1, max_length=200, description="One Swedish sentence on what and why."
    )


class CharterError(SdModel):
    code: str
    detail_sv: str
    field: str | None = None
    limit: float | None = None


class CharterProposalOutcome(SdModel):
    """What ``GroundClient.propose_charter`` always returns (``ok = false`` = no card)."""

    ok: bool
    proposal_id: Id | None = None
    approval_id: Id | None = None
    errors: list[CharterError] = Field(default_factory=list[CharterError])


# --- geometry ---------------------------------------------------------------------------


class _Frame:
    def __init__(self, s: Situation) -> None:
        if s.home is None:
            raise DraftError("Hempositionen är okänd ännu.", repairable=False)
        self.lat0, self.lon0 = s.home.lat, s.home.lon

    def enu(self, lat: float, lon: float) -> tuple[float, float]:
        return to_enu(lat, lon, self.lat0, self.lon0)

    def poly_from_geojson(self, geometry: dict[str, Any], field: str) -> Polygon:
        if geometry.get("type") != "Polygon":
            raise DraftError("Ytan är ingen polygon.", field)
        ring = geometry["coordinates"][0]
        return Polygon([self.enu(float(lat), float(lon)) for lon, lat in ring])

    def geojson(self, poly: Polygon) -> dict[str, Any]:
        ring = []
        for east, north in poly.exterior.coords:
            lat, lon = from_enu(east, north, self.lat0, self.lon0)
            ring.append([round(lon, 9), round(lat, 9)])
        ring[-1] = ring[0]
        return {"type": "Polygon", "coordinates": [ring]}


def _center(a: RectArea | CircleArea, s: Situation, f: _Frame, field: str) -> tuple[float, float]:
    if a.center_pin_id is not None:
        pin = next((p for p in s.pins if p.id == a.center_pin_id), None)
        if pin is None:
            known = ", ".join(p.id for p in s.pins) or "inga"
            raise DraftError(
                f"Okänd nål {a.center_pin_id}; kända: {known}.",
                field,
                hint="Use a pin listed under Kartnålar in LÄGE.",
            )
        return f.enu(pin.lat, pin.lon)
    if a.center is not None:
        return f.enu(a.center.lat, a.center.lon)
    raise DraftError(
        "Sökområdet saknar mittpunkt.",
        field,
        hint="Give center_pin_id (or center when the operator wrote coordinates).",
    )


def _drawn(a: DrawnArea, s: Situation, field: str) -> Any:
    area = next((x for x in s.areas if x.kind == "area" and x.label == a.area_id), None)
    if area is None:
        known = ", ".join(x.label for x in s.areas if x.kind == "area") or "inga"
        raise DraftError(
            f"Okänt område {a.area_id}; kända: {known}.",
            field,
            hint="Use an area listed under Ritade områden in LÄGE.",
        )
    return area


def _to_polygon(
    a: DrawnArea | RectArea | CircleArea, s: Situation, f: _Frame, field: str
) -> Polygon:
    if isinstance(a, DrawnArea):
        return f.poly_from_geojson(_drawn(a, s, field).geometry, field)
    cx, cy = _center(a, s, f, field)
    if isinstance(a, CircleArea):
        pts = [
            (cx + a.radius_m * math.sin(2 * math.pi * i / 32),
             cy + a.radius_m * math.cos(2 * math.pi * i / 32))
            for i in range(32)
        ]  # fmt: skip
        return Polygon(pts)
    # width_m runs along heading_deg (clockwise from north), height_m across it.
    h = math.radians(a.heading_deg)
    ux, uy = math.sin(h), math.cos(h)  # along the width
    vx, vy = math.cos(h), -math.sin(h)  # across (90° clockwise)
    hw, hh = a.width_m / 2, a.height_m / 2
    corners = [
        (cx + sx * hw * ux + sy * hh * vx, cy + sx * hw * uy + sy * hh * vy)
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    return Polygon(corners)


def _split(poly: Polygon) -> list[Polygon]:
    """Cut a too large polygon into west-to-east strips of equal width."""
    if poly.area <= SECTOR_MAX_AREA_M2:
        return [poly]
    n = math.ceil(poly.area / SPLIT_TARGET_M2)
    minx, miny, maxx, maxy = poly.bounds
    while n <= 64:
        width = (maxx - minx) / n
        pieces = []
        for i in range(n):
            cut = poly.intersection(
                box(minx + i * width, miny - 1, minx + (i + 1) * width, maxy + 1)
            )
            pieces.extend(
                g for g in getattr(cut, "geoms", [cut]) if isinstance(g, Polygon) and not g.is_empty
            )
        merged: list[Polygon] = []
        for piece in pieces:
            if piece.area < MERGE_BELOW_M2 and merged:
                joined = unary_union([merged[-1], piece])
                merged[-1] = joined if isinstance(joined, Polygon) else merged[-1]
            else:
                merged.append(piece)
        if all(p.area <= SECTOR_MAX_AREA_M2 for p in merged):
            return merged
        n += 1
    raise DraftError("Området går inte att dela i sektorer; välj ett mindre område.")


def _sector_id(i: int) -> str:
    return chr(ord("A") + i)


def _simplify(poly: Polygon) -> Polygon:
    tol = 0.5
    out = poly
    while len(out.exterior.coords) - 1 > MAX_HULL_CORNERS:
        simpler = poly.simplify(tol)
        if not isinstance(simpler, Polygon):
            break
        out = simpler
        tol *= 2
    return out


def _check_area(area: Polygon, polys: list[Polygon], draft: CharterDraft, s: Situation) -> None:
    """What the ground station would refuse for certain, said to the model in the turn:
    home outside the flight area (HOME_OUTSIDE_AREA), a search area outside it
    (SECTOR_OUTSIDE_AREA), a corner beyond the range from the pilot (AREA_BEYOND_RANGE;
    the pilot stands at home in v1)."""
    if not area.covers(ShPoint(0.0, 0.0)):  # home is the plane's origin
        raise DraftError(
            "Hemplatsen ligger utanför flygområdet.",
            "operating_area",
            code="HOME_OUTSIDE_AREA",
            hint="Leave operating_area out: the system computes the flight area around "
            "home and the search areas.",
        )
    grown = area.buffer(0.5)
    if not all(grown.covers(p) for p in polys):
        raise DraftError(
            "Ett sökområde ligger utanför flygområdet.",
            "operating_area",
            code="SECTOR_OUTSIDE_AREA",
            hint="Leave operating_area out, or search inside the drawn flight area.",
        )
    reach = s.limits.max_range_from_pilot_m
    if draft.max_range_from_pilot_m:  # 0 = unset, as the ground station reads it
        reach = min(reach, draft.max_range_from_pilot_m)
    far = max(math.hypot(x, y) for x, y in area.exterior.coords)
    if far > reach + RANGE_SLACK_M:
        raise DraftError(
            f"Flygområdet når {far:.0f} m från piloten men högst {reach:.0f} m är tillåtet. "
            "Välj ett mindre sökområde eller ett som ligger närmare hemplatsen.",
            "search_areas",
            code="AREA_BEYOND_RANGE",
            hint="Choose a smaller search area, or one closer to home.",
        )


def to_proposal(draft: CharterDraft, s: Situation) -> CharterProposal:
    """The ground station's ``CharterProposal`` for a draft (pure; no I/O)."""
    f = _Frame(s)
    polys: list[Polygon] = []
    for i, a in enumerate(draft.search_areas):
        field = f"search_areas.{i}"
        poly = _to_polygon(a, s, f, field)
        if not poly.is_valid or poly.area <= 0:
            raise DraftError("Sökområdet är ingen giltig yta.", field)
        polys.extend(_split(poly))
    if len(polys) > 26:
        raise DraftError("För många sektorer; välj ett mindre område.", "search_areas")
    sectors = [
        Sector(sector_id=_sector_id(i), name=_sector_id(i), area=f.geojson(p), order_idx=i)
        for i, p in enumerate(polys)
    ]
    body: dict[str, Any] = {"sectors": [x.model_dump(mode="json") for x in sectors]}
    op = draft.operating_area
    if op is not None:
        drawn = _drawn(op, s, "operating_area")
        area = f.poly_from_geojson(drawn.geometry, "operating_area")
        body["area_annotation_id"] = str(drawn.id)
    else:
        points = [ShPoint(0.0, 0.0)]
        if s.pilot is not None:
            points.append(ShPoint(*f.enu(s.pilot.lat, s.pilot.lon)))
        hull = unary_union([*polys, MultiPoint(points)]).convex_hull
        buffered = hull.buffer(OPERATING_BUFFER_M, quad_segs=4)
        if not isinstance(buffered, Polygon):
            raise DraftError("Flygområdet gick inte att räkna ut.", "search_areas")
        area = _simplify(buffered)
        body["area"] = f.geojson(area)
    _check_area(area, polys, draft, s)
    for name in (
        "alt_min_agl_m",
        "alt_max_agl_m",
        "default_alt_agl_m",
        "max_speed_ms",
        "max_range_from_pilot_m",
        "battery_rtl_pct",
        "allowed_skills",
    ):
        value = getattr(draft, name)
        if value is not None:
            body[name] = value
    body["base_version"] = draft.base_version
    body["reason_sv"] = draft.reason_sv
    try:
        return CharterProposal.model_validate(body)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = error_path(first.get("loc", ()))
        raise DraftError(
            f"Ogiltigt värde i {field}.", field, hint=f"{field}: {first.get('msg')}."
        ) from None


_AREA_TAGS = frozenset({"drawn", "rect", "circle"})


def error_path(loc: Any) -> str:
    """A validation error's path as the draft names it: ``search_areas.0.width_m``
    (pydantic puts the union's tag in between, ``search_areas.0.rect.width_m``)."""
    parts = [str(p) for p in loc]
    kept = [p for i, p in enumerate(parts) if not (p in _AREA_TAGS and i > 0)]
    return ".".join(kept) or "arguments"
