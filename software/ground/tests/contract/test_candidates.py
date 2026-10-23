"""B2-04: candidates end to end on the ground. Perception reports come in over the real
SD-link server (``FakeOnboard``), ``sd-search`` (its service loop and client, in process)
fuses them through the search API and the panel reads the queue.

E1: the search token can never confirm or reject. E2: a candidate with priority 0.05 is
in the queue. E7 (no command path) is checked here on the API and in sd_search's tests.
"""

from __future__ import annotations

import io
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from PIL import Image, ImageDraw
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from flow_rig import SEARCH, VID, H, Rig, rig  # pyright: ignore[reportMissingImports]

from sd_ground.app import commander_asgi, create_commander_app
from sd_schemas.base import LatLon, new_id
from sd_schemas.link import BBox, CropData, DetectorInfo, GeoDetectionMsg, PerceptionReport, Pose
from sd_schemas.search import SearchProfile
from sd_search.app import SearchService
from sd_search.client import GroundClient
from sd_search.profile import GroundProfileSource

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]

RED, GRASS = (200, 30, 30), (70, 120, 40)
SQUARE = BBox(x=60, y=60, w=96, h=96)  # the crop's square in the still
BOX = BBox(x=92, y=92, w=32, h=32)  # the person in the still: the square's middle third


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


def _crop(color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (96, 96), GRASS)
    ImageDraw.Draw(img).rectangle((32, 32, 63, 63), fill=color)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=92)
    return out.getvalue()


def person(r: Rig, east: float, north: float, **kw: Any) -> GeoDetectionMsg:
    lat, lon = r.at(east, north)
    base: dict[str, Any] = {
        "detection_id": new_id(),
        "label": "person",
        "source": "model",
        "score": 0.6,
        "bbox": BOX,
        "lat": lat,
        "lon": lon,
        "sigma95_m": 3.0,
        "px_size": 32.0,  # 1.33 m at 24 px/m: a person
        "crop_inline": True,
    }
    return GeoDetectionMsg.model_validate(base | kw)


def report(r: Rig, *dets: GeoDetectionMsg) -> PerceptionReport:
    lat, lon = r.home
    corners = [r.at(e, n) for e, n in ((-60, 60), (60, 60), (60, -60), (-60, -60))]
    return PerceptionReport(
        still_id=new_id(),
        command_id=None,
        captured_utc=r.clock.utc_now(),
        sensor="rgb",
        pose=Pose(
            lat=lat,
            lon=lon,
            alt_amsl_m=624,
            agl_m=40,
            roll_deg=0,
            pitch_deg=0,
            yaw_deg=0,
            source_ts_ms=0,
        ),
        footprint=[LatLon(lat=la, lon=lo) for la, lo in corners],
        px_per_m=24.0,
        synthetic=True,
        detector=DetectorInfo(
            name="sd_detect:test",
            license="Apache-2.0",
            version="0",
            placement="onboard",
            ok=True,
            tiles=1,
            latency_ms=30,
            rate_hz=1,
        ),
        detections=list(dets),
    )


async def send(r: Rig, *dets: GeoDetectionMsg, color: tuple[int, int, int] = RED) -> None:
    rep = report(r, *dets)
    await r.onboard.session.send(rep, ttl_ms=30_000)
    for d in rep.detections:
        if d.crop_inline:
            crop = CropData(
                ref_id=d.detection_id,
                purpose="detection",
                still_id=rep.still_id,
                bbox_in_still=SQUARE,
                width=96,
                height=96,
                jpeg=_crop(color),
            )
            await r.onboard.session.send(crop, ttl_ms=30_000)
    q = r.flow.reports.get(VID)
    before = 0 if q is None else q.handled

    def done() -> bool:
        q = r.flow.reports.get(VID)
        return q is not None and q.handled > before and not q.running

    await r.until(done, 10, "report handled")


class StaticProfile:
    def __init__(self, profile: SearchProfile | None) -> None:
        self.profile = profile

    async def current(self, mission_id: str) -> SearchProfile | None:
        if self.profile is None:
            return None
        return SearchProfile.model_validate(self.profile.model_dump() | {"mission_id": mission_id})


def red_profile(r: Rig) -> SearchProfile:
    lat, lon = r.home
    now = r.clock.utc_now().isoformat()
    return SearchProfile.model_validate(
        {
            "profile_id": str(new_id()),
            "profile_version": 1,
            "mission_id": str(new_id()),
            "subject_category": "hiker",
            "garments": [{"part": "upper", "colors": ["red"], "certainty": "certain"}],
            "last_seen": {"lat": lat, "lon": lon, "at_utc": now},
            "approved_by": "operator:demo",
            "approved_at": now,
        }
    )


def search_service(
    r: Rig, profile: SearchProfile | None = None
) -> tuple[SearchService, GroundClient]:
    transport = httpx.ASGITransport(app=commander_asgi(r.g, create_commander_app(r.g)))  # pyright: ignore[reportArgumentType]
    client = GroundClient("http://127.0.0.1:7700", SEARCH, transport=transport)
    return SearchService(client, StaticProfile(profile)), client


async def queue(r: Rig, mid: str, **params: Any) -> dict[str, Any]:
    res = await r.panel.get(f"/api/v1/missions/{mid}/candidates", params=params)
    assert res.status_code == 200, res.text
    assert "Sd-Not-Built" not in res.headers
    return res.json()


def revision(r: Rig, **kw: Any) -> dict[str, Any]:
    lat, lon = r.at(0, 100)
    now = r.clock.utc_now().isoformat()
    return {
        "candidate_id": str(new_id()),
        "base_rev": 0,
        "state": "new",
        "lat": lat,
        "lon": lon,
        "sigma95_m": 3.0,
        "n_seen": 1,
        "best_detector_score": 0.3,
        "priority": 0.4,
        "first_seen_utc": now,
        "last_seen_utc": now,
    } | kw


async def post(r: Rig, mid: str, *revisions: dict[str, Any]) -> httpx.Response:
    assert r.search is not None
    return await r.search.post(
        f"/search/v1/missions/{mid}/candidates", json={"revisions": list(revisions)}
    )


async def test_detections_become_a_candidate_in_the_queue(r: Rig) -> None:
    mid = await r.open_mission()
    service, client = search_service(r, red_profile(r))
    try:
        first = person(r, 0, 100)
        await send(r, first, person(r, 0, 130, crop_inline=False))
        assert await service.step() == 2
        assert r.search is not None
        feed = (await r.search.get(f"/search/v1/missions/{mid}/detections")).json()
        assert [d["has_crop"] for d in feed] == [True, False]
        assert feed[0]["crop_box"] == SQUARE.model_dump() and feed[0]["seq"] < feed[1]["seq"]
        crop = await r.search.get(f"/search/v1/detections/{first.detection_id}/crop")
        assert crop.status_code == 200 and crop.headers["content-type"] == "image/jpeg"

        await send(r, person(r, 1, 100))  # the same person in the next still
        assert await service.step() == 1
        body = await queue(r, mid)
        assert body["counts"] == {"review": 2, "low": 0, "confirmed": 0, "rejected": 0}
        top = body["items"][0]
        assert top["seq"] == 1 and top["state"] == "ready" and top["n_seen"] == 2
        assert top["rev"] == 2 and top["n_in_view"] >= 2
        assert top["engine"] == "code" and top["fields"]["garment:0"] == "match"
        assert top["thumb_detection_id"] == str(first.detection_id)
        assert top["assessed_profile_version"] == 1 and top["current_profile_version"] is None
        assert top["priority_class"] == "high" and top["priority"] > body["items"][1]["priority"]
        assert body["items"][1]["fields"] == {"garment:0": "not_visible", "state": "not_visible"}

        detail = await r.panel.get(f"/api/v1/candidates/{top['candidate_id']}")
        assert detail.status_code == 200, detail.text
        d = detail.json()
        assert len(d["observations"]) == 2 and d["decisions"] == []
        assert [a["engine"] for a in d["assessments"]] == ["code", "code"]
        assert d["card"]["candidate_id"] == top["candidate_id"] and d["mission_id"] == mid

        types = [
            e.type for e in r.g.bus.replay(r.g.bus.server_epoch, 0) or [] if e.topic == "candidates"
        ]
        assert types.count("candidate.new") == 2 and types.count("candidate.updated") == 1
        assert types.count("candidate.assessed") == 3
    finally:
        await client.aclose()


async def test_a_restart_gives_no_duplicates_and_loses_nothing(r: Rig) -> None:
    mid = await r.open_mission()
    service, client = search_service(r)
    await send(r, person(r, 0, 100), person(r, 30, 100))
    await service.step()
    await client.aclose()
    await send(r, person(r, 1, 100), person(r, 60, 100))  # while sd-search is down
    again, client = search_service(r)
    try:
        assert await again.step() == 2
        assert await again.step() == 0
    finally:
        await client.aclose()
    async with r.flow.repo.engine.connect() as conn:
        rows = (await conn.execute(text(
            "SELECT COUNT(*) AS n, COUNT(DISTINCT detection_id) AS d FROM candidate_observations"
        ))).one()  # fmt: skip
    assert tuple(rows) == (4, 4)
    body = await queue(r, mid)
    assert sorted(c["n_seen"] for c in body["items"]) == [1, 1, 2]
    assert sorted(c["seq"] for c in body["items"]) == [1, 2, 3]


async def test_e2_a_low_priority_candidate_is_in_the_queue(r: Rig) -> None:
    mid = await r.open_mission()
    res = await post(r, mid, revision(r, priority=0.05))
    assert res.status_code == 200, res.text
    body = await queue(r, mid)
    assert [(c["priority"], c["priority_class"]) for c in body["items"]] == [(0.05, "low")]
    assert body["counts"]["low"] == 1 and body["counts"]["review"] == 0
    assert (await queue(r, mid, min_priority=0.3))["items"] == []


async def test_e1_the_search_token_never_decides(r: Rig) -> None:
    mid = await r.open_mission()
    for state in ("confirmed", "rejected", "inspecting"):
        res = await post(r, mid, revision(r, state=state))
        assert res.status_code == 422, state
    rev = revision(r)
    assert (await post(r, mid, rev)).status_code == 200
    stale = await post(r, mid, rev | {"base_rev": 0})
    assert stale.status_code == 409 and stale.json()["code"] == "CANDIDATE_EXISTS"
    wrong = await post(r, mid, rev | {"base_rev": 5, "state": "ready"})
    assert wrong.status_code == 409 and wrong.json()["code"] == "REV_CONFLICT"

    # A person rejects it (a later ticket's route; written here as that route would).
    async with r.flow.repo.engine.begin() as conn:
        await conn.execute(text(
            "INSERT INTO candidates (candidate_id, rev, mission_id, seq, state, geom, sigma95_m, "
            "n_seen, n_in_view, best_detector_score, priority, first_seen_utc, last_seen_utc) "
            "SELECT candidate_id, 2, mission_id, seq, 'rejected', geom, sigma95_m, n_seen, "
            "n_in_view, best_detector_score, priority, first_seen_utc, last_seen_utc "
            "FROM candidates WHERE candidate_id = :c"
        ), {"c": rev["candidate_id"]})  # fmt: skip
        await conn.execute(text(
            "INSERT INTO candidate_decisions (candidate_id, decision, subject_rev, operator_ref, "
            "session_id) VALUES (:c, 'reject', 1, 'operator:demo', 's')"
        ), {"c": rev["candidate_id"]})  # fmt: skip
    closed = await post(r, mid, rev | {"base_rev": 2, "state": "ready"})
    assert closed.status_code == 409 and closed.json()["code"] == "CANDIDATE_CLOSED"
    young = revision(r)
    merge = await post(r, mid, young | {"state": "merged", "merged_into": rev["candidate_id"]})
    assert merge.status_code == 409 and merge.json()["code"] == "CANDIDATE_HELD"
    body = await queue(r, mid)
    assert body["counts"]["rejected"] == 1 and body["items"][0]["state"] == "rejected"

    # A new candidate next to the rejected one says so.
    assert (await post(r, mid, young)).status_code == 200
    near = next(c for c in (await queue(r, mid))["items"] if c["state"] == "new")
    assert (near["near_seq"], near["near_state"]) == (1, "rejected")


async def test_one_observation_per_detection_and_batches_are_atomic(r: Rig) -> None:
    mid = await r.open_mission()
    det = person(r, 0, 100)
    await send(r, det)
    obs = [{"detection_id": str(det.detection_id), "distance_m": 0.0}]
    a, b = revision(r, observations=obs), revision(r, observations=obs)
    res = await post(r, mid, a, b)  # the second observes the same detection: none written
    assert res.status_code == 409
    assert (await queue(r, mid))["items"] == []
    other = await post(
        r, mid, revision(r, observations=[{"detection_id": str(new_id()), "distance_m": 0}])
    )
    assert other.status_code == 409 and other.json()["code"] == "DETECTION_NOT_FOUND"
    assert (await post(r, mid, a)).status_code == 200
    assert (await post(r, mid, b)).status_code == 409


async def test_merged_candidates_only_on_request(r: Rig) -> None:
    mid = await r.open_mission()
    old, young = revision(r), revision(r)
    assert (await post(r, mid, old, young)).status_code == 200
    merged = young | {"base_rev": 1, "state": "merged", "merged_into": old["candidate_id"]}
    res = await post(r, mid, merged)
    assert res.status_code == 200, res.text
    assert [c["candidate_id"] for c in (await queue(r, mid))["items"]] == [old["candidate_id"]]
    only = (await queue(r, mid, state="merged"))["items"]
    assert [(c["candidate_id"], c["merged_into"]) for c in only] == [
        (young["candidate_id"], old["candidate_id"])
    ]
    again = await post(r, mid, merged | {"base_rev": 2})
    assert again.status_code == 409 and again.json()["code"] == "CANDIDATE_CLOSED"


async def test_closed_missions_and_unknown_candidates(r: Rig) -> None:
    assert r.search is not None
    unknown = str(new_id())
    assert (await r.search.get(f"/search/v1/missions/{unknown}/detections")).status_code == 404
    assert (await r.panel.get(f"/api/v1/candidates/{unknown}")).status_code == 404
    assert (await r.search.get(f"/search/v1/detections/{unknown}/crop")).status_code == 404
    mid = await r.open_mission()
    missions = (await r.search.get("/search/v1/missions")).json()
    assert [m["mission_id"] for m in missions] == [mid]
    res = await r.panel.post(f"/api/v1/missions/{mid}/end", json={"reason": "test"}, headers=H)
    assert res.status_code == 200, res.text
    late = await post(r, mid, revision(r))
    assert late.status_code == 409 and late.json()["code"] == "MISSION_CLOSED"
    assert (await r.search.get("/search/v1/missions")).json() == []
    assert (await queue(r, mid))["items"] == []  # a closed mission's queue still answers


async def test_a_held_candidate_keeps_its_state(r: Rig) -> None:
    mid = await r.open_mission()
    rev = revision(r)
    assert (await post(r, mid, rev)).status_code == 200
    # A person opens it (a later ticket's route; written here as that route would).
    async with r.flow.repo.engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO candidates (candidate_id, rev, mission_id, seq, state, geom, "
                "sigma95_m, n_seen, n_in_view, best_detector_score, priority, first_seen_utc, "
                "last_seen_utc) SELECT candidate_id, 2, mission_id, seq, 'inspecting', geom, "
                "sigma95_m, n_seen, n_in_view, best_detector_score, priority, first_seen_utc, "
                "last_seen_utc FROM candidates WHERE candidate_id = :c"
            ),
            {"c": rev["candidate_id"]},
        )
    moved = await post(r, mid, rev | {"base_rev": 2, "state": "ready"})
    assert moved.status_code == 409 and moved.json()["code"] == "CANDIDATE_HELD"
    seen_again = await post(r, mid, rev | {"base_rev": 2, "state": None, "n_seen": 2})
    assert seen_again.status_code == 200, seen_again.text
    (card,) = (await queue(r, mid))["items"]
    assert (card["state"], card["n_seen"], card["rev"]) == ("inspecting", 2, 3)


async def _store_profile(r: Rig, mid: str, version: int, *, approved: bool) -> None:
    body = red_profile(r).model_dump(mode="json") | {"profile_version": version}
    if not approved:
        body |= {"approved_by": None, "approved_at": None}
    async with r.flow.repo.engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO search_profiles (mission_id, profile_id, profile_version, body, "
                "proposed_by, approved_by, approved_at) VALUES (CAST(:m AS uuid), "
                "CAST(:p AS uuid), :v, CAST(:b AS jsonb), 'brain', :by, :at)"
            ),
            {
                "m": mid,
                "p": body["profile_id"],
                "v": version,
                "b": json.dumps(body),
                "by": "operator:demo" if approved else None,
                "at": r.clock.utc_now() if approved else None,
            },
        )


async def test_the_search_profile_is_the_approved_one(r: Rig) -> None:
    assert r.search is not None
    mid = await r.open_mission()
    url = f"/search/v1/missions/{mid}/search-profile"
    assert (await r.search.get(url)).status_code == 404
    await _store_profile(r, mid, 1, approved=False)  # an AI proposal is never used
    res = await r.search.get(url)
    assert res.status_code == 404 and res.json()["code"] == "PROFILE_NOT_FOUND"
    await _store_profile(r, mid, 2, approved=True)
    await _store_profile(r, mid, 3, approved=False)
    res = await r.search.get(url)
    assert res.status_code == 200, res.text
    got = SearchProfile.model_validate(res.json())
    assert (str(got.mission_id), got.profile_version, got.approved) == (mid, 2, True)

    # sd-search reads it from the ground and matches the red jacket with it.
    transport = httpx.ASGITransport(app=commander_asgi(r.g, create_commander_app(r.g)))  # pyright: ignore[reportArgumentType]
    client = GroundClient("http://127.0.0.1:7700", SEARCH, transport=transport)
    try:
        service = SearchService(client, GroundProfileSource(client))
        await send(r, person(r, 0, 100))
        assert await service.step() == 1
    finally:
        await client.aclose()
    (card,) = (await queue(r, mid))["items"]
    assert card["fields"]["garment:0"] == "match" and card["assessed_profile_version"] == 2


async def test_a_merge_needs_another_known_candidate(r: Rig) -> None:
    mid = await r.open_mission()
    a, b = revision(r), revision(r)
    assert (await post(r, mid, a, b)).status_code == 200
    for target in (str(new_id()), a["candidate_id"]):
        merged = a | {"base_rev": 1, "state": "merged", "merged_into": target}
        res = await post(r, mid, merged)
        assert res.status_code == 409 and res.json()["code"] == "MERGE_TARGET_INVALID", target
