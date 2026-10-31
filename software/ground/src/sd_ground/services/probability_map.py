"""The probability map: what ``sd-search`` needs to compute it, and its versions.

The search service reads the map's inputs (``inputs``: the approved charter's area and
sectors, the search height and speed, the operator's areas with a POA hint and the newest
version) and the stills (``stills``), and writes each new version (``write``). The ground
checks a version before it stores it: the next number, nothing after a frozen version
while a find is still confirmed (the pilot's reopen unlocks the map), the approved
profile's place last seen as the grid's origin, the array sizes, finite values in [0, 1]
and ``sum(p) + row == 1``. The grid's bounds are computed here, not
taken from the writer. The panel reads the newest version (``view``), an image of it
(``png``) and the sector scores (``sector_scores``); the AI gets the situation line.

The map only ranks where to look. Nothing here starts a flight or changes an order: the
AI proposes, and the gate and a person approve as before.

Events on the bus (topic ``probability_map``): ``probability_map.updated {version}``.
Audit events: ``probability_map.frozen`` when a confirmed find locks the map and
``probability_map.unfrozen`` when the first version after a reopen unlocks it.
"""

from __future__ import annotations

import base64
import binascii
import struct
import zlib
from typing import TYPE_CHECKING, Any, Final, Literal
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from sd_ground.api.deps import ApiStatusError
from sd_ground.db.probmap import MapWriteError, ProbabilityMapRepo
from sd_ground.logging import get_logger
from sd_ground.services import search_profile as profile_service
from sd_schemas.geo import plane_distance_m, plane_offset
from sd_schemas.probmap import (
    ProbabilityGrid,
    ProbabilityMapInputs,
    ProbabilityMapLatest,
    ProbabilityMapView,
    ProbabilityMapWrite,
    ProbabilityMapWritten,
    SearchArea,
    SearchStill,
    SectorScore,
    situation_line,
)

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow

__all__ = [
    "STILLS_MAX",
    "SURVEY_SPEED_MS",
    "inputs",
    "png",
    "probability_line",
    "repo",
    "sector_scores",
    "stills",
    "view",
    "write",
]

log = get_logger("sd_ground.probability_map")
AREAS_MAX: Final = 200  # ProbabilityMapInputs.areas

TOPIC: Final = "probability_map"
STILLS_MAX: Final = 500
SURVEY_SPEED_MS: Final = 7.0  # survey_sector's default speed (capped by the charter's)
SUM_TOLERANCE: Final = 1e-3  # float32 cells summed
IPP_TOLERANCE_M: Final = 1.0
# The heat layer: transparent to red, the sought layer: transparent to blue.
_RGB: Final = {"poa": (220, 38, 38), "pod": (37, 99, 235)}
_ALPHA_MAX: Final = 200

Floats = NDArray[np.float64]


def repo(flow: MissionFlow) -> ProbabilityMapRepo:
    return ProbabilityMapRepo(flow.repo.engine)


def _latest(head: dict[str, Any] | None) -> ProbabilityMapLatest | None:
    if head is None:
        return None
    return ProbabilityMapLatest(
        version=int(head["version"]), frozen=bool(head["frozen"]), event_seq=int(head["event_seq"])
    )


# --- for sd-search --------------------------------------------------------------------


async def inputs(flow: MissionFlow, mission_id: str) -> ProbabilityMapInputs:
    mission = await flow.open_mission_of(mission_id)  # search reads open missions only (409)
    charter = await flow.active_charter(mission)
    annotations = [] if flow.g.repo is None else await flow.g.repo.annotations(mission_id)
    areas = [
        SearchArea(annotation_id=a.id, poa_hint=a.poa_hint, polygon=a.geometry)
        for a in annotations
        if a.kind == "area" and a.poa_hint is not None
    ][-AREAS_MAX:]  # the newest, within what the inputs carry
    head = await repo(flow).head(mission_id)
    if charter is None:
        return ProbabilityMapInputs(area=None, areas=areas, latest=_latest(head))
    return ProbabilityMapInputs(
        area=charter.area,
        sectors=[(s.sector_id, s.area) for s in charter.sectors],
        alt_m=charter.default_alt_agl_m,
        speed_ms=min(SURVEY_SPEED_MS, charter.max_speed_ms),
        areas=areas,
        latest=_latest(head),
    )


async def stills(
    flow: MissionFlow, mission_id: str, after_seq: int, limit: int
) -> list[SearchStill]:
    await flow.open_mission_of(mission_id)
    rows = await repo(flow).stills(mission_id, after_seq, min(limit, STILLS_MAX))
    return [SearchStill.model_validate(r) for r in rows]


def _bounded_grid(grid: ProbabilityGrid) -> ProbabilityGrid:
    """The grid with its bounds computed from its origin and size."""
    south, west = plane_offset(grid.ipp_lat, grid.ipp_lon, grid.x0_m, grid.y0_m)
    north, east = plane_offset(
        grid.ipp_lat,
        grid.ipp_lon,
        grid.x0_m + grid.nx * grid.cell_m,
        grid.y0_m + grid.ny * grid.cell_m,
    )
    return grid.model_copy(update={"west": west, "south": south, "east": east, "north": north})


def _invalid(detail_sv: str) -> ApiStatusError:
    return ApiStatusError(422, "MAP_INVALID", detail_sv)


def decode_f32(z: str, cells: int) -> Floats:
    """base64 zlib float32 -> ``cells`` values; never inflates past the expected size."""
    try:
        raw = base64.b64decode(z, validate=True)
        d = zlib.decompressobj()
        out = d.decompress(raw, cells * 4 + 1)
        if not d.eof or d.unconsumed_tail or d.unused_data:
            raise _invalid("Kartans data har fel storlek.")
    except (binascii.Error, zlib.error):
        raise _invalid("Kartans data kan inte läsas.") from None
    if len(out) != cells * 4:
        raise _invalid("Kartans data har fel storlek.")
    a = np.frombuffer(out, dtype="<f4").astype(np.float64)
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise _invalid("Kartans värden måste ligga mellan 0 och 1.")
    return a


async def write(
    flow: MissionFlow, mission_id: str, body: ProbabilityMapWrite
) -> ProbabilityMapWritten:
    mission = await flow.open_mission_of(mission_id)
    profile = await profile_service.approved(flow, mission_id)
    if profile is None or profile.profile_version != body.profile_version:
        raise ApiStatusError(
            409, "MAP_PROFILE_STALE", "Kartan bygger inte på sökprofilen som gäller nu."
        )
    charter = await flow.active_charter(mission)
    known = set() if charter is None else {s.sector_id for s in charter.sectors}
    if any(s.sector_id not in known for s in body.sectors):
        raise _invalid("Kartan nämner en sektor som inte finns i uppdragsramen.")
    if len({s.sector_id for s in body.sectors}) != len(body.sectors):
        raise _invalid("Kartan nämner samma sektor två gånger.")
    grid = _bounded_grid(body.grid)
    ls = profile.last_seen
    if plane_distance_m(grid.ipp_lat, grid.ipp_lon, ls.lat, ls.lon) > IPP_TOLERANCE_M:
        raise _invalid("Kartans origo är inte platsen där personen sågs senast.")
    p = decode_f32(body.p_z, grid.cells)
    pod = decode_f32(body.pod_z, grid.cells)
    if abs(float(p.sum()) + body.row - 1.0) > SUM_TOLERANCE:
        raise _invalid("Kartans sannolikheter summerar inte till 1.")
    stored = body.model_dump(mode="json", exclude={"p_z", "pod_z", "version", "reason", "row"})
    stored["grid"] = grid.model_dump(mode="json")
    try:
        unlocked = await repo(flow).insert(
            mission_id,
            version=body.version,
            reason=body.reason,
            grid=stored,
            p=zlib.compress(p.astype("<f4").tobytes(), 6),
            pod_cum=zlib.compress(pod.astype("<f4").tobytes(), 6),
            row=body.row,
            frozen=body.frozen,
            event_seq=body.event_seq,
            # Only a person confirms a find; the search service cannot fake its effect. The
            # repo checks it under the map's write lock, in the same step as the insert.
            needs_confirmed=body.frozen or body.reason == "confirmed",
        )
    except MapWriteError as exc:
        raise ApiStatusError(409, exc.code, exc.detail_sv) from None
    data = {"version": body.version, "reason": body.reason, "frozen": body.frozen}
    log.info("probability_map.updated", mission_id=mission_id, **data)
    flow.publish(
        TOPIC,
        "probability_map.updated",
        {"version": body.version},
        vehicle_id=mission.vehicle_id,
        mission_id=mission_id,
    )
    if unlocked:
        flow.g.add_event(
            "probability_map.unfrozen",
            severity="info",
            mission_id=mission_id,
            vehicle_id=mission.vehicle_id,
            payload=data,
        )
    if body.frozen:
        flow.g.add_event(
            "probability_map.frozen",
            severity="info",
            mission_id=mission_id,
            vehicle_id=mission.vehicle_id,
            payload=data,
        )
    return ProbabilityMapWritten(version=body.version, frozen=body.frozen)


# --- for the panel and the AI -----------------------------------------------------------


def _arrays(row: dict[str, Any]) -> tuple[ProbabilityGrid, Floats, Floats]:
    grid = ProbabilityGrid.model_validate(row["grid"]["grid"])
    n = grid.cells
    p = np.frombuffer(zlib.decompress(bytes(row["p"])), dtype="<f4").astype(np.float64)
    pod = np.frombuffer(zlib.decompress(bytes(row["pod_cum"])), dtype="<f4").astype(np.float64)
    if p.size != n or pod.size != n:  # stored by ``write`` after the same check
        raise ApiStatusError(500, "MAP_CORRUPT", "Kartan i databasen har fel storlek.")
    return grid, p, pod


async def _newest_row(flow: MissionFlow, mission_id: str) -> dict[str, Any] | None:
    await flow.mission(mission_id)
    return await repo(flow).latest(mission_id)


async def _row(flow: MissionFlow, mission_id: str) -> dict[str, Any]:
    row = await _newest_row(flow, mission_id)
    if row is None:
        raise ApiStatusError(404, "MAP_NOT_FOUND", "Ingen sannolikhetskarta ännu.")
    return row


def _f16(a: Floats) -> str:
    return base64.b64encode(a.astype("<f2").tobytes()).decode("ascii")


async def view(flow: MissionFlow, mission_id: str) -> ProbabilityMapView | None:
    """The newest version, or None before the first."""
    row = await _newest_row(flow, mission_id)
    if row is None:
        return None
    grid, p, pod = _arrays(row)
    meta = row["grid"]
    return ProbabilityMapView(
        mission_id=UUID(mission_id),
        version=int(row["version"]),
        reason=row["reason"],
        grid=grid,
        row=round(float(row["row"]), 4),
        frozen=bool(row["frozen"]),
        event_seq=int(row["event_seq"]),
        profile_version=int(meta.get("profile_version", 0)),
        sectors=[SectorScore.model_validate(s) for s in meta.get("sectors", [])],
        notes=list(meta.get("notes", [])),
        p_b64=_f16(p),
        pod_b64=_f16(pod),
        created_at=row["created_at"],
    )


async def sector_scores(flow: MissionFlow, mission_id: str) -> list[SectorScore]:
    await flow.mission(mission_id)
    head = await repo(flow).head(mission_id)
    if head is None:
        return []
    return [SectorScore.model_validate(s) for s in head["grid"].get("sectors", [])]


async def probability_line(flow: MissionFlow, mission_id: str) -> str | None:
    """The AI's situation line, or None before the first map with sectors. A fault here
    never takes the rest of the situation with it."""
    try:
        head = await repo(flow).head(mission_id)
        scores = [] if head is None else head["grid"].get("sectors", [])
        if not scores:
            return None
        return situation_line([SectorScore.model_validate(s) for s in scores])
    except Exception:
        log.exception("probability_map.situation_line_failed", mission_id=mission_id)
        return None


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def encode_png(rgba: NDArray[np.uint8]) -> bytes:
    """An RGBA image (rows top to bottom) as PNG."""
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw, 6))
        + _chunk(b"IEND", b"")
    )


def layer_image(values: Floats, grid: ProbabilityGrid, layer: Literal["poa", "pod"]) -> bytes:
    """One pixel per cell, north up. POA is scaled to the map's highest cell; POD is
    absolute (0 transparent, 1 the strongest colour)."""
    v = values.reshape(grid.ny, grid.nx)[::-1]  # row 0 is the southernmost
    top = float(v.max()) if layer == "poa" else 1.0
    scaled = np.zeros_like(v) if top <= 0 else np.clip(v / top, 0.0, 1.0)
    if layer == "poa":
        scaled = np.sqrt(scaled)  # low cells stay visible
    rgba = np.zeros((grid.ny, grid.nx, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = _RGB[layer]
    rgba[..., 3] = np.round(scaled * _ALPHA_MAX).astype(np.uint8)
    return encode_png(rgba)


async def png(
    flow: MissionFlow, mission_id: str, layer: Literal["poa", "pod"]
) -> tuple[bytes, ProbabilityGrid, int]:
    row = await _row(flow, mission_id)
    grid, p, pod = _arrays(row)
    return layer_image(p if layer == "poa" else pod, grid, layer), grid, int(row["version"])
