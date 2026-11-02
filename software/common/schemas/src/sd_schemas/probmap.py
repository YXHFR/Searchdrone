"""The probability map: where the missing person may be, and how well each cell was seen.

The search service (``sd-search``) computes the map from the approved search profile, the
operator's areas and the stills the drone has taken, and writes each new version to the
ground station, which stores it append-only and serves it to the panel and the AI. The
map only ranks where to look: it never starts a flight, and a human still approves every
order through the safety gate.

Cells are ``cell_m`` squares in the local east/north plane with its origin at the place
last seen (the IPP); ``p`` and ``pod`` are row-major, row 0 the southernmost. ``row`` is
the rest of the world: the chance the person is outside the grid. ``sum(p) + row == 1``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from sd_schemas.base import GeoJsonPolygon, Id, LatLon, SdModel

__all__ = [
    "PROBMAP_B64_MAX",
    "PROBMAP_CELLS_MAX",
    "PROBMAP_SIDE_MAX",
    "PSR_MAX",
    "MapNote",
    "MapReason",
    "PoaHint",
    "ProbabilityGrid",
    "ProbabilityMapInputs",
    "ProbabilityMapLatest",
    "ProbabilityMapView",
    "ProbabilityMapWrite",
    "ProbabilityMapWritten",
    "SearchArea",
    "SearchStill",
    "SectorScore",
    "situation_line",
    "suggest_sector_order",
]

PROBMAP_SIDE_MAX: Final = 160  # 4 km of 25 m cells
PROBMAP_CELLS_MAX: Final = PROBMAP_SIDE_MAX * PROBMAP_SIDE_MAX
# base64 of at most 25 600 float32 values, compressed or not (102 400 bytes, plus zlib's
# worst-case overhead), with room to spare.
PROBMAP_B64_MAX: Final = 140_000
PSR_MAX: Final = 100.0  # a sector's pos_plan per planned minute, est_min >= 0.01

PoaHint = Literal["high", "low", "excluded"]
# Why a version was written.
MapReason = Literal["prior", "search", "candidate", "profile", "areas", "charter", "confirmed"]
# What the panel should say about a map: no terrain data (rings only), no statistics for
# children (the general rings are used), uncertain statistics, no charter yet.
MapNote = Literal["no_terrain", "child_statistics", "uncertain_statistics", "no_charter"]


class ProbabilityGrid(SdModel):
    ipp_lat: float = Field(ge=-90, le=90)  # the origin of the plane
    ipp_lon: float = Field(ge=-180, le=180)
    x0_m: float = Field(ge=-50_000, le=50_000)  # the south-west corner, east of the IPP
    y0_m: float = Field(ge=-50_000, le=50_000)  # north of the IPP
    nx: int = Field(ge=1, le=PROBMAP_SIDE_MAX)
    ny: int = Field(ge=1, le=PROBMAP_SIDE_MAX)
    cell_m: float = Field(ge=5, le=200)
    # The grid's bounds, for a map image source.
    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)

    @property
    def cells(self) -> int:
        return self.nx * self.ny


class SectorScore(SdModel):
    """One sector of the charter on the current map.

    ``poa``: the chance the person is in the sector now (cells whose centre lies in it).
    ``covered_pct``: the share of its cells seen with a POD of at least 0.1.
    ``mean_pod``: the mean cumulative POD over its cells.
    ``pos_plan``: the chance of finding the person if the sector is surveyed once more as
    planned. ``psr``: ``pos_plan`` per minute of planned flight (higher first); with
    ``est_min`` in hundredths of a minute it is at most ``PSR_MAX`` (1 / 0.01).
    """

    sector_id: str = Field(pattern=r"^[A-Z]{1,2}$")
    poa: float = Field(ge=0, le=1)
    covered_pct: float = Field(ge=0, le=100)
    mean_pod: float = Field(ge=0, le=1)
    pos_plan: float = Field(ge=0, le=1)
    est_min: float = Field(ge=0, le=10_000)
    psr: float = Field(ge=0, le=PSR_MAX)
    spacing_m: float = Field(gt=0, le=500)  # the suggested line spacing for a survey


class _MapBody(SdModel):
    version: int = Field(ge=1)
    reason: MapReason
    grid: ProbabilityGrid
    row: float = Field(ge=0, le=1)
    frozen: bool = False
    event_seq: int = Field(ge=0)  # the newest still the map has taken in
    profile_version: int = Field(ge=0)  # 0 = no approved profile (no map is written then)
    sectors: list[SectorScore] = Field(default_factory=list[SectorScore], max_length=64)
    notes: list[MapNote] = Field(default_factory=list[MapNote], max_length=4)


class ProbabilityMapWrite(_MapBody):
    """``sd-search`` -> ground: a new version. ``p_z`` and ``pod_z`` are base64 of
    zlib-compressed little-endian float32 arrays of ``grid.nx * grid.ny`` values; the
    ground checks the sizes, that every value is finite and in [0, 1], and the sum."""

    p_z: str = Field(min_length=4, max_length=PROBMAP_B64_MAX)
    pod_z: str = Field(min_length=4, max_length=PROBMAP_B64_MAX)


class ProbabilityMapWritten(SdModel):
    version: int
    frozen: bool


class ProbabilityMapView(_MapBody):
    """The panel's view of the latest version. ``p_b64`` and ``pod_b64`` are base64 of
    little-endian float16 arrays (row-major, row 0 the southernmost)."""

    mission_id: Id
    p_b64: str
    pod_b64: str
    created_at: AwareDatetime


class SearchStill(SdModel):
    """A still the drone took, as the map needs it: where it looked and how well."""

    still_id: Id
    seq: int = Field(ge=1)  # the feed's cursor (after_seq)
    command_id: Id | None  # the order it was taken during: one pass of the search
    captured_utc: AwareDatetime
    footprint: list[LatLon] = Field(min_length=4, max_length=4)
    px_per_m: float = Field(gt=0)
    detector_ok: bool  # False: the detector failed on it, so it saw nothing


class SearchArea(SdModel):
    """An area the operator drew with a hint for the map."""

    annotation_id: Id
    poa_hint: PoaHint
    polygon: GeoJsonPolygon


class ProbabilityMapLatest(SdModel):
    version: int = Field(ge=1)
    frozen: bool
    event_seq: int = Field(ge=0)


class ProbabilityMapInputs(SdModel):
    """What the map is computed from besides the profile and the stills: the charter's
    area and sectors, the operator's areas, and the latest stored version."""

    area: GeoJsonPolygon | None  # the approved charter's area; None before one
    sectors: list[tuple[str, GeoJsonPolygon]] = Field(
        default_factory=list[tuple[str, GeoJsonPolygon]], max_length=64
    )
    alt_m: float = Field(default=40, gt=0, le=500)  # the charter's default search height
    speed_ms: float = Field(default=5, gt=0, le=50)  # the survey speed
    areas: list[SearchArea] = Field(default_factory=list[SearchArea], max_length=200)
    latest: ProbabilityMapLatest | None = None

    @model_validator(mode="after")
    def _unique_sectors(self) -> Self:
        ids = [s for s, _ in self.sectors]
        if len(ids) != len(set(ids)):
            raise ValueError("sector ids are unique")
        return self


def suggest_sector_order(scores: Sequence[SectorScore]) -> list[SectorScore]:
    """Greedy by PSR, highest first (ties by sector id). Only a suggestion: the AI
    proposes, and the gate and a person approve as before."""
    return sorted(scores, key=lambda s: (-s.psr, s.sector_id))


def situation_line(scores: Sequence[SectorScore]) -> str:
    """The AI's situation line, in the suggested order (Swedish decimal commas). The last
    field is the planned minutes of one pass (``est_min``) and the lane spacing they are
    planned for (``spacing_m``), so the AI can scale them to a closer search and weigh
    another sector against the battery's time to the return limit."""
    parts = [
        f"{s.sector_id} {_sv(s.poa, 2)}/{s.covered_pct:.0f}%/{_sv(s.mean_pod, 2)}/{_sv(s.psr, 3)}"
        f"/{_sv(s.est_min, 1)}@{s.spacing_m:.0f}"
        for s in suggest_sector_order(scores)
    ]
    return "Sektorer (POA kvar / täckt / medel-POD / PSR / min@stråk m): " + "  ".join(parts)


def _sv(x: float, n: int) -> str:
    return f"{x:.{n}f}".replace(".", ",")
