"""The service loop against an in-memory ground (``fake_ground.FakeGround``)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fake_ground import FakeGround, utc  # pyright: ignore[reportMissingImports]
from search_helpers import (  # pyright: ignore[reportMissingImports]
    BLUE,
    MISSION,
    RED,
    at,
    crop_jpeg,
    profile,
)

from sd_schemas.base import new_id
from sd_schemas.link import BBox
from sd_schemas.search import CandidateState
from sd_search.app import SearchService
from sd_search.client import GroundError
from sd_search.profile import FileProfileSource, GroundProfileSource


def detect(
    g: FakeGround, east: float, north: float, t: float, *, crop: bytes | None = None, **kw: object
):
    lat, lon = at(east, north)
    did = str(new_id())
    if crop is not None:
        g.crops[did] = crop
    return g.add_detection(
        detection_id=did,
        still_id=str(new_id()),
        lat=lat,
        lon=lon,
        captured_utc=utc(t),
        has_crop=crop is not None,
        # crop_jpeg's 96 px crop of a 96 px square: the box is its middle third
        bbox=BBox(x=132, y=132, w=32, h=32),
        crop_box=BBox(x=100, y=100, w=96, h=96),
        **kw,
    )


def service(g: FakeGround) -> SearchService:
    return SearchService(g, GroundProfileSource(g))


async def test_detections_become_a_scored_multi_frame_candidate() -> None:
    g = FakeGround(MISSION, profile_body=profile())
    s = service(g)
    detect(g, 0, 0, 0, crop=crop_jpeg(RED))
    assert await s.step() == 1
    (cid,) = g.revs
    first = g.latest(cid)
    assert first.state is CandidateState.READY and first.seq == 1 and first.n_seen == 1
    detect(g, 1, 0, 1, crop=crop_jpeg(RED))
    assert await s.step() == 1
    c = g.latest(cid)
    assert c.rev == 2 and c.n_seen == 2 and c.priority > first.priority  # multi_frame
    assert g.assessments[-1].fields["garment:0"] == "match"
    assert c.latest_assessment_id is None or c.latest_assessment_id  # the ground sets it
    assert len(g.observed) == 2 and set(g.observed.values()) == {cid}


async def test_a_low_candidate_is_kept() -> None:
    g = FakeGround(MISSION, profile_body=profile())
    s = service(g)
    detect(g, 0, 0, 0, score=0.01, crop=crop_jpeg(BLUE))  # weak and the wrong colour
    await s.step()
    (cid,) = g.revs
    c = g.latest(cid)
    assert g.assessments[-1].fields["garment:0"] == "conflict"
    assert 0 < c.priority < 0.30 and c.state is CandidateState.READY


async def test_restart_resumes_without_duplicates_or_losses() -> None:
    g = FakeGround(MISSION, profile_body=profile())
    detect(g, 0, 0, 0, crop=crop_jpeg(RED))
    detect(g, 40, 0, 1)
    await service(g).step()
    before = {k: len(v) for k, v in g.revs.items()}
    observed = dict(g.observed)
    # The service dies; two detections arrive meanwhile; a new one starts.
    detect(g, 0.5, 0, 2)
    detect(g, 80, 0, 3)
    again = service(g)
    assert await again.step() == 2
    assert await again.step() == 0  # nothing twice
    assert observed.items() <= g.observed.items() and len(g.observed) == 4
    assert len(g.revs) == 3
    first = next(iter(before))
    assert len(g.revs[first]) == before[first] + 1


async def test_a_persons_decision_makes_the_service_read_back() -> None:
    g = FakeGround(MISSION)
    s = service(g)
    detect(g, 0, 0, 0)
    await s.step()
    (cid,) = g.revs
    g.decide(cid, CandidateState.REJECTED)
    detect(g, 1, 0, 1)  # near the rejected one: the stale fusion would join it
    await s.step()  # 409, read back
    await s.step()
    assert len(g.revs) == 2
    new = next(k for k in g.revs if k != cid)
    assert g.latest(cid).state is CandidateState.REJECTED and g.latest(new).seq == 2


async def test_a_held_candidate_keeps_its_state() -> None:
    g = FakeGround(MISSION)
    s = service(g)
    detect(g, 0, 0, 0)
    await s.step()
    (cid,) = g.revs
    c = g.latest(cid)
    g.revs[cid].append(c.model_copy(update={"rev": c.rev + 1, "state": CandidateState.INSPECTING}))
    detect(g, 1, 0, 1)
    await s.step()
    await s.step()
    assert g.latest(cid).state is CandidateState.INSPECTING and g.latest(cid).n_seen == 2


async def test_only_an_approved_profile_is_used() -> None:
    g = FakeGround(MISSION, profile_body=profile(approved_by=None, approved_at=None))
    s = service(g)
    detect(g, 0, 0, 0, crop=crop_jpeg(RED))
    await s.step()
    assert g.assessments[-1].profile_version == 0 and g.assessments[-1].fields == {}
    other = profile(mission_id=str(new_id()))  # another mission's profile
    g.profile_body = other
    assert await GroundProfileSource(g).current(MISSION) is None


async def test_a_new_profile_version_scores_again() -> None:
    g = FakeGround(MISSION, profile_body=profile())
    s = service(g)
    detect(g, 0, 0, 0, crop=crop_jpeg(RED))
    await s.step()
    assert len(g.assessments) == 1
    await s.step()
    assert len(g.assessments) == 1  # nothing new
    g.profile_body = profile(profile_version=2)
    await s.step()
    assert [a.profile_version for a in g.assessments] == [1, 2]


async def test_heartbeat_says_cloud_off() -> None:
    g = FakeGround(MISSION)
    await service(g).beat(force=True)
    assert g.beats[-1].cloud == "off"


async def test_heartbeat_every_five_seconds() -> None:
    g = FakeGround(MISSION)
    now = [100.0]
    s = SearchService(g, GroundProfileSource(g), monotonic=lambda: now[0])
    await s.beat()
    now[0] += 4.9
    await s.beat()
    assert len(g.beats) == 1
    now[0] += 0.1
    await s.beat()
    assert len(g.beats) == 2


def test_file_profile_only_in_the_simulator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "profile.json"
    path.write_text(profile().model_dump_json(), encoding="utf-8")
    monkeypatch.delenv("SD_SIM", raising=False)
    with pytest.raises(RuntimeError, match="simulatorn"):
        FileProfileSource(path)
    monkeypatch.setenv("SD_SIM", "1")
    assert FileProfileSource(path).profile.approved
    path.write_text(profile(approved_by=None, approved_at=None).model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="godkänd"):
        FileProfileSource(path)


async def test_one_failing_mission_does_not_stop_the_others() -> None:
    class Broken(FakeGround):
        async def missions(self) -> list[str]:
            return ["broken", self.mission_id]

        async def detections(self, mission_id: str, after_seq: int, limit: int):  # pyright: ignore[reportIncompatibleMethodOverride]
            if mission_id == "broken":
                raise GroundError("broken: 500", 500)
            return await super().detections(mission_id, after_seq, limit)

    g = Broken(MISSION, profile_body=profile())
    s = service(g)
    detect(g, 0, 0, 0, crop=crop_jpeg(RED))
    assert await s.step() == 1
    assert len(g.revs) == 1 and "broken" not in s.missions
