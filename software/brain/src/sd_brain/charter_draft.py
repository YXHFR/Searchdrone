"""The charter draft the AI fills in, and turning it into the ground station's proposal.

The model never writes polygons: it names a drawn area, or a rectangle or circle around
a pin. :func:`to_proposal` turns that into a ``CharterProposal`` with sectors A, B, …
It has no field for the autonomy mode or the mission limits.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, ValidationError
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
    "DraftError",
    "to_proposal",
]

SPLIT_TARGET_M2 = 240_000.0
MERGE_BELOW_M2 = 100.0
OPERATING_BUFFER_M = 50.0
MAX_HULL_CORNERS = 200


class DraftError(ValueError):
    """The draft cannot be turned into a proposal (sent back to the model as a tool error)."""


class AreaRef(SdModel):
    kind: Literal["drawn", "rect", "circle"]
    area_id: str | None = Field(default=None, max_length=40)  # drawn: the area's label
    center_pin_id: str | None = Field(default=None, pattern=r"^P[0-9]{1,3}$")
    center: LatLon | None = None  # only when the operator wrote coordinates
    width_m: float | None = Field(default=None, ge=20, le=2000)
    height_m: float | None = Field(default=None, ge=20, le=2000)
    heading_deg: float = Field(default=0, ge=0, le=180)
    radius_m: float | None = Field(default=None, ge=20, le=1000)


class CharterDraft(SdModel):
    base_version: int | None = None
    operating_area: AreaRef | None = None
    search_areas: list[AreaRef] = Field(min_length=1, max_length=4)
    alt_min_agl_m: float | None = Field(default=None, ge=0, le=1000)
    alt_max_agl_m: float | None = Field(default=None, ge=0, le=1000)
    default_alt_agl_m: float | None = Field(default=None, ge=0, le=1000)
    max_speed_ms: float | None = Field(default=None, ge=0, le=100)
    max_range_from_pilot_m: float | None = Field(default=None, ge=0, le=10_000)
    battery_rtl_pct: float | None = Field(default=None, ge=0, le=100)
    allowed_skills: list[str] | None = Field(default=None, max_length=32)
    reason_sv: str = Field(min_length=1, max_length=200)


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
            raise DraftError("hempositionen är okänd ännu")
        self.lat0, self.lon0 = s.home.lat, s.home.lon

    def enu(self, lat: float, lon: float) -> tuple[float, float]:
        return to_enu(lat, lon, self.lat0, self.lon0)

    def poly_from_geojson(self, geometry: dict[str, Any]) -> Polygon:
        if geometry.get("type") != "Polygon":
            raise DraftError("ytan är ingen polygon")
        ring = geometry["coordinates"][0]
        return Polygon([self.enu(float(lat), float(lon)) for lon, lat in ring])

    def geojson(self, poly: Polygon) -> dict[str, Any]:
        ring = []
        for east, north in poly.exterior.coords:
            lat, lon = from_enu(east, north, self.lat0, self.lon0)
            ring.append([round(lon, 9), round(lat, 9)])
        ring[-1] = ring[0]
        return {"type": "Polygon", "coordinates": [ring]}


def _center(a: AreaRef, s: Situation, f: _Frame) -> tuple[float, float]:
    if a.center_pin_id is not None:
        pin = next((p for p in s.pins if p.id == a.center_pin_id), None)
        if pin is None:
            known = ", ".join(p.id for p in s.pins) or "inga"
            raise DraftError(f"okänd nål {a.center_pin_id}; kända: {known}")
        return f.enu(pin.lat, pin.lon)
    if a.center is not None:
        return f.enu(a.center.lat, a.center.lon)
    raise DraftError("ange center_pin_id (eller center om operatören skrev koordinater)")


def _drawn(a: AreaRef, s: Situation) -> Any:
    area = next((x for x in s.areas if x.kind == "area" and x.label == a.area_id), None)
    if area is None:
        known = ", ".join(x.label for x in s.areas if x.kind == "area") or "inga"
        raise DraftError(f"okänt område {a.area_id}; kända: {known}")
    return area


def _to_polygon(a: AreaRef, s: Situation, f: _Frame) -> Polygon:
    if a.kind == "drawn":
        return f.poly_from_geojson(_drawn(a, s).geometry)
    cx, cy = _center(a, s, f)
    if a.kind == "circle":
        if a.radius_m is None:
            raise DraftError("en cirkel behöver radius_m")
        pts = [
            (cx + a.radius_m * math.sin(2 * math.pi * i / 32),
             cy + a.radius_m * math.cos(2 * math.pi * i / 32))
            for i in range(32)
        ]  # fmt: skip
        return Polygon(pts)
    if a.width_m is None or a.height_m is None:
        raise DraftError("en rektangel behöver width_m och height_m")
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
    raise DraftError("området går inte att dela i sektorer, välj ett mindre område")


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


def to_proposal(draft: CharterDraft, s: Situation) -> CharterProposal:
    """The ground station's ``CharterProposal`` for a draft (pure; no I/O)."""
    f = _Frame(s)
    polys: list[Polygon] = []
    for a in draft.search_areas:
        poly = _to_polygon(a, s, f)
        if not poly.is_valid or poly.area <= 0:
            raise DraftError("sökområdet är ingen giltig yta")
        polys.extend(_split(poly))
    if len(polys) > 26:
        raise DraftError("för många sektorer, välj ett mindre område")
    sectors = [
        Sector(sector_id=_sector_id(i), name=_sector_id(i), area=f.geojson(p), order_idx=i)
        for i, p in enumerate(polys)
    ]
    body: dict[str, Any] = {"sectors": [x.model_dump(mode="json") for x in sectors]}
    op = draft.operating_area
    if op is not None and op.kind == "drawn":
        body["area_annotation_id"] = str(_drawn(op, s).id)
    elif op is not None:
        body["area"] = f.geojson(_to_polygon(op, s, f))
    else:
        points = [ShPoint(0.0, 0.0)]
        if s.pilot is not None:
            points.append(ShPoint(*f.enu(s.pilot.lat, s.pilot.lon)))
        hull = unary_union([*polys, MultiPoint(points)]).convex_hull
        buffered = hull.buffer(OPERATING_BUFFER_M, quad_segs=4)
        if not isinstance(buffered, Polygon):
            raise DraftError("flygområdet gick inte att räkna ut")
        body["area"] = f.geojson(_simplify(buffered))
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
        field = ".".join(str(p) for p in first.get("loc", ()))
        raise DraftError(f"ogiltigt värde i {field}: {first.get('msg')}") from None
