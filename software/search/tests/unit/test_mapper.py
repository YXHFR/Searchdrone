"""``MapService`` against an in-memory ground: when versions are written, and freezing."""

from __future__ import annotations

import base64
import zlib

import numpy as np
from fake_ground import FakeGround, utc  # pyright: ignore[reportMissingImports]
from search_helpers import MISSION, at, profile  # pyright: ignore[reportMissingImports]

from sd_schemas.base import new_id
from sd_schemas.probmap import (
    ProbabilityMapInputs,
    ProbabilityMapWrite,
    ProbabilityMapWritten,
    SearchArea,
    SearchStill,
)
from sd_schemas.search import CandidateState
from sd_search.app import SearchService
from sd_search.client import GroundError
from sd_search.mapper import SAVE_EVERY_S, MapService
from sd_search.probmap.terrain import NoTerrain
from sd_search.profile import GroundProfileSource


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def poly(east: float, north: float, half: float) -> dict[str, object]:
    pts = [
        at(east - half, north - half),
        at(east + half, north - half),
        at(east + half, north + half),
        at(east - half, north + half),
    ]
    ring = [[lon, lat] for lat, lon in pts]
    return {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}


def ground(**kw: object) -> FakeGround:
    g = FakeGround(MISSION, profile_body=profile(subject_category="dementia"))
    g.inputs = ProbabilityMapInputs.model_validate(
        {"area": poly(0, 0, 600), "sectors": [("A", poly(-300, 0, 290))]} | kw
    )
    return g


def add_still(g: FakeGround, east: float, north: float, command_id: str) -> SearchStill:
    corners = [at(east - 15, north - 15), at(east + 15, north - 15)]
    corners += [at(east + 15, north + 15), at(east - 15, north + 15)]
    s = SearchStill.model_validate(
        {
            "still_id": str(new_id()),
            "seq": len(g.still_rows) + 1,
            "command_id": command_id,
            "captured_utc": utc(len(g.still_rows)),
            "footprint": [{"lat": a, "lon": b} for a, b in corners],
            "px_per_m": 100.0,
            "detector_ok": True,
        }
    )
    g.still_rows.append(s)
    return s


def mapper(g: FakeGround, clock: Clock, *, motion: bool = True) -> MapService:
    # Wall time follows the monotonic clock, ten minutes after last seen at the start.
    return MapService(
        g,
        GroundProfileSource(g),
        NoTerrain(),
        motion=motion,
        monotonic=clock,
        clock=lambda: utc(600 + clock.t),
    )


def p_of(g: FakeGround) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(g.maps[-1].p_z))
    return np.frombuffer(raw, dtype="<f4")


async def test_the_first_version_is_the_prior_and_sums_to_one() -> None:
    g = ground()
    m = mapper(g, Clock())
    assert await m.step([MISSION]) == 1
    (v,) = g.maps
    assert v.version == 1 and v.reason == "prior" and not v.frozen
    assert "no_terrain" in v.notes
    assert abs(float(p_of(g).sum()) + v.row - 1) < 1e-4
    assert [s.sector_id for s in v.sectors] == ["A"]
    assert await m.step([MISSION], force=True) == 0  # nothing changed


async def test_no_profile_no_map() -> None:
    g = ground()
    g.profile_body = None
    assert await mapper(g, Clock()).step([MISSION]) == 0
    assert not g.maps


async def test_stills_are_saved_every_30_s_and_at_a_new_pass() -> None:
    g = ground()
    clock = Clock()
    m = mapper(g, clock)
    await m.step([MISSION])
    pass1 = str(new_id())
    add_still(g, -300, 0, pass1)
    clock.t += 5
    assert await m.step([MISSION]) == 0  # within the 30 s: waits
    clock.t += SAVE_EVERY_S
    assert await m.step([MISSION]) == 1
    assert g.maps[-1].reason == "search" and g.maps[-1].event_seq == 1
    add_still(g, -300, 25, pass1)
    clock.t += 3
    assert await m.step([MISSION]) == 0
    add_still(g, -250, 0, str(new_id()))  # a new order began: the pass before has ended
    clock.t += 3
    assert await m.step([MISSION]) == 1
    assert [v.version for v in g.maps] == [1, 2, 3]


async def test_a_new_area_writes_at_once() -> None:
    g = ground()
    clock = Clock()
    m = mapper(g, clock)
    await m.step([MISSION])
    area = SearchArea.model_validate(
        {"annotation_id": str(new_id()), "poa_hint": "excluded", "polygon": poly(200, 200, 100)}
    )
    g.inputs = g.inputs.model_copy(update={"areas": [area]})
    clock.t += 3
    assert await m.step([MISSION]) == 1
    assert g.maps[-1].reason == "areas"


async def test_a_failed_write_is_tried_again() -> None:
    g = ground()
    clock = Clock()
    m = mapper(g, clock)
    await m.step([MISSION])
    area = SearchArea.model_validate(
        {"annotation_id": str(new_id()), "poa_hint": "excluded", "polygon": poly(200, 200, 100)}
    )
    g.inputs = g.inputs.model_copy(update={"areas": [area]})
    write = g.write_map
    calls = 0

    async def down_once(mission_id: str, body: ProbabilityMapWrite) -> ProbabilityMapWritten:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise GroundError("ground down", 503)
        return await write(mission_id, body)

    g.write_map = down_once
    clock.t += 3
    assert await m.step([MISSION]) == 0  # logged, not raised
    clock.t += 3
    assert await m.step([MISSION]) == 1
    assert [v.reason for v in g.maps] == ["prior", "areas"]


async def test_a_fault_in_the_map_never_stops_the_round() -> None:
    g = ground()

    async def broken(mission_id: str) -> ProbabilityMapInputs:
        raise ValueError("bug")

    g.map_inputs = broken
    assert await mapper(g, Clock()).step([MISSION, MISSION]) == 0


async def test_confirm_freezes_map() -> None:
    g = ground()
    clock = Clock()
    m = mapper(g, clock)
    s = SearchService(g, GroundProfileSource(g), mapper=m)
    lat, lon = at(-300, 0)
    still = add_still(g, -300, 0, str(new_id()))
    g.add_detection(
        detection_id=str(new_id()),
        still_id=str(still.still_id),
        lat=lat,
        lon=lon,
        captured_utc=utc(0),
    )
    await s.step()
    assert len(g.maps) == 1 and not g.maps[-1].frozen
    (cid,) = g.revs
    g.decide(cid, CandidateState.CONFIRMED)
    clock.t += 3
    await s.step()
    last = g.maps[-1]
    assert last.frozen and last.reason == "confirmed"
    add_still(g, 0, 0, str(new_id()))
    for _ in range(3):
        clock.t += SAVE_EVERY_S + 1
        await s.step()
    assert g.maps[-1].version == last.version  # the version stands still


async def test_reopen_unlocks_map() -> None:
    g = ground()
    clock = Clock()
    m = mapper(g, clock)
    s = SearchService(g, GroundProfileSource(g), mapper=m)
    lat, lon = at(-300, 0)
    still = add_still(g, -300, 0, str(new_id()))
    g.add_detection(
        detection_id=str(new_id()),
        still_id=str(still.still_id),
        lat=lat,
        lon=lon,
        captured_utc=utc(0),
    )
    await s.step()
    (cid,) = g.revs
    g.decide(cid, CandidateState.CONFIRMED)
    clock.t += 3
    await s.step()
    frozen = g.maps[-1]
    assert frozen.frozen
    g.decide(cid, CandidateState.READY)  # the pilot reopens the find
    clock.t += 3
    await s.step()
    after = g.maps[-1]
    assert after.version == frozen.version + 1 and not after.frozen
    assert after.reason == "candidate"
    add_still(g, 0, 0, str(new_id()))
    clock.t += SAVE_EVERY_S + 1
    await s.step()
    assert g.maps[-1].version == after.version + 1  # versions come in again


async def test_a_pending_candidate_keeps_its_cell_until_rejected() -> None:
    g = ground()
    clock = Clock()
    m = mapper(g, clock)
    s = SearchService(g, GroundProfileSource(g), mapper=m)
    lat, lon = at(-300, 0)
    still = add_still(g, -300, 0, str(new_id()))
    g.add_detection(
        detection_id=str(new_id()),
        still_id=str(still.still_id),
        lat=lat,
        lon=lon,
        captured_utc=utc(0),
    )
    await s.step()
    first = p_of(g).copy()
    raw = zlib.decompress(base64.b64decode(g.maps[-1].pod_z))
    grid = g.maps[-1].grid
    pod = np.frombuffer(raw, dtype="<f4").reshape(grid.ny, grid.nx)
    assert pod.max() == 0.0  # the only still saw the candidate: nothing counted down
    (cid,) = g.revs
    g.decide(cid, CandidateState.REJECTED)
    clock.t += 3
    await s.step()
    assert g.maps[-1].reason == "candidate"
    assert p_of(g).min() >= 0 and (p_of(g) < first).any()


async def first_map(*, motion: bool, start_s: float) -> np.ndarray:
    g = ground()
    clock = Clock()
    clock.t = start_s
    await mapper(g, clock, motion=motion).step([MISSION])
    return p_of(g)


async def test_the_map_walks_to_the_clock_when_movement_is_on() -> None:
    early = await first_map(motion=True, start_s=0)
    late = await first_map(motion=True, start_s=3 * 3600)
    assert not np.allclose(early, late)


async def test_with_movement_off_the_clock_does_not_matter() -> None:
    early = await first_map(motion=False, start_s=0)
    late = await first_map(motion=False, start_s=3 * 3600)
    assert np.array_equal(early, late)
