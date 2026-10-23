"""B2-07: a person's decision on a candidate, end to end on the ground.

A perception report with a crop comes in over the real SD-link server, the search API
writes a candidate that observed it, and people decide through the panel API.

E1: only a person decides; service tokens, and CLI tokens outside the test rig, get 403;
an observer confirms but neither inspects nor reopens. A confirmation needs a large view
from the same session (409 ``NOT_VIEWED``). E2: the tables only grow.
"""

from __future__ import annotations

import io
import sys
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx
import pytest
from PIL import Image
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from flow_rig import BRAIN, SEARCH, VID, H, Rig, rig  # pyright: ignore[reportMissingImports]

from sd_ground.app import create_panel_app, panel_asgi
from sd_ground.auth import AuthMode
from sd_ground.services.candidates import CANDIDATE_TRANSITIONS, PILOT_ONLY
from sd_schemas.base import LatLon, new_id
from sd_schemas.link import BBox, CropData, DetectorInfo, GeoDetectionMsg, PerceptionReport, Pose
from sd_schemas.search import CandidateState

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]

GPX_NS = "{http://www.topografix.com/GPX/1/1}"


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


def _jpeg() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (96, 96), (200, 30, 30)).save(out, format="JPEG")
    return out.getvalue()


async def detection(r: Rig) -> str:
    """One target detection with a crop, sent by the drone; its id."""
    lat, lon = r.at(0, 100)
    det = GeoDetectionMsg.model_validate(
        {
            "detection_id": new_id(),
            "label": "person",
            "source": "model",
            "score": 0.6,
            "bbox": BBox(x=92, y=92, w=32, h=32),
            "lat": lat,
            "lon": lon,
            "sigma95_m": 3.0,
            "px_size": 32.0,
            "crop_inline": True,
        }
    )
    hlat, hlon = r.home
    corners = [r.at(e, n) for e, n in ((-60, 160), (60, 160), (60, 40), (-60, 40))]
    rep = PerceptionReport(
        still_id=new_id(),
        command_id=None,
        captured_utc=r.clock.utc_now(),
        sensor="rgb",
        pose=Pose(
            lat=hlat,
            lon=hlon,
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
        detections=[det],
    )
    q = r.flow.reports.get(VID)
    before = 0 if q is None else q.handled
    await r.onboard.session.send(rep, ttl_ms=30_000)
    crop = CropData(
        ref_id=det.detection_id,
        purpose="detection",
        still_id=rep.still_id,
        bbox_in_still=BBox(x=60, y=60, w=96, h=96),
        width=96,
        height=96,
        jpeg=_jpeg(),
    )
    await r.onboard.session.send(crop, ttl_ms=30_000)

    def done() -> bool:
        q = r.flow.reports.get(VID)
        return q is not None and q.handled > before and not q.running

    await r.until(done, 10, "report handled")
    return str(det.detection_id)


async def candidate(r: Rig, mid: str, state: str = "ready") -> tuple[str, str]:
    """A candidate in ``state`` that observed a fresh detection: (candidate id, detection)."""
    det = await detection(r)
    lat, lon = r.at(0, 100)
    now = r.clock.utc_now().isoformat()
    cid = str(new_id())
    assert r.search is not None
    res = await r.search.post(
        f"/search/v1/missions/{mid}/candidates",
        json={
            "revisions": [
                {
                    "candidate_id": cid,
                    "base_rev": 0,
                    "state": state,
                    "lat": lat,
                    "lon": lon,
                    "sigma95_m": 3.0,
                    "n_seen": 1,
                    "best_detector_score": 0.6,
                    "priority": 0.7,
                    "first_seen_utc": now,
                    "last_seen_utc": now,
                    "observations": [{"detection_id": det, "distance_m": 0.0}],
                }
            ]
        },
    )
    assert res.status_code == 200, res.text
    return cid, det


@asynccontextmanager
async def person(
    r: Rig, role: Literal["pilot", "observer"], name: str, auth_mode: AuthMode = "local"
) -> AsyncIterator[httpx.AsyncClient]:
    """A logged-in person's panel session (the cookie, as the browser sends it)."""
    token, _ = r.g.sessions.create_session(
        role=role, issuer_ref=f"operator:{name}", auth_mode=auth_mode
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=panel_asgi(r.g, create_panel_app(r.g))),  # pyright: ignore[reportArgumentType]
        base_url="http://127.0.0.1:8080",
        cookies={"sd_session": token},
        headers=H,
    ) as client:
        yield client


async def rev_of(c: httpx.AsyncClient, cid: str) -> int:
    res = await c.get(f"/api/v1/candidates/{cid}")
    assert res.status_code == 200, res.text
    return int(res.json()["card"]["rev"])


async def decide(
    c: httpx.AsyncClient, cid: str, decision: str, note: str = "", rev: int | None = None
) -> httpx.Response:
    subject = await rev_of(c, cid) if rev is None else rev
    return await c.post(
        f"/api/v1/candidates/{cid}/decision",
        json={"decision": decision, "note": note, "subject_rev": subject},
        headers=H,
    )


async def view(c: httpx.AsyncClient, det: str) -> str:
    res = await c.post(f"/api/v1/detections/{det}/view", headers=H)
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["detection_id"] == det
    pic = await c.get(f"/api/v1/views/{body['view_id']}")
    assert pic.status_code == 200 and pic.headers["content-type"] == "image/jpeg"
    return body["view_id"]


def test_candidate_transitions_table() -> None:
    """Every state a person can act on, and nothing that only the search service does."""
    s = CandidateState
    assert set(CANDIDATE_TRANSITIONS) == {"confirm", "reject", "inspect", "unsure", "reopen"}
    for d in ("confirm", "reject"):
        assert set(CANDIDATE_TRANSITIONS[d]) == {s.READY, s.ASSESSING, s.INSPECTING}
    assert CANDIDATE_TRANSITIONS["inspect"] == {s.READY: s.INSPECTING, s.ASSESSING: s.INSPECTING}
    assert CANDIDATE_TRANSITIONS["reopen"] == {s.REJECTED: s.READY, s.CONFIRMED: s.READY}
    assert set(CANDIDATE_TRANSITIONS["unsure"].values()) == {None}
    for table in CANDIDATE_TRANSITIONS.values():
        assert s.MERGED not in table and s.MERGED not in table.values()
        assert s.NEW not in table.values()
    assert frozenset({"inspect", "reopen"}) == PILOT_ONLY


async def test_candidate_transitions(r: Rig) -> None:
    mid = await r.open_mission()
    cid, det = await candidate(r, mid)
    async with person(r, "pilot", "kalle") as pilot:
        # Allowed: unsure keeps the state, reject, reopen, then confirm after a view.
        res = await decide(pilot, cid, "unsure", "oklart")
        assert res.status_code == 200, res.text
        assert res.json()["card"]["state"] == "ready"
        assert (await decide(pilot, cid, "reject")).json()["card"]["state"] == "rejected"
        # Not allowed: from rejected only reopen.
        for d in ("confirm", "reject", "unsure"):
            res = await decide(pilot, cid, d)
            assert res.status_code == 409 and res.json()["code"] in (
                "TRANSITION_NOT_ALLOWED",
                "NOT_VIEWED",
            ), d
        assert (await decide(pilot, cid, "reopen")).json()["card"]["state"] == "ready"
        await view(pilot, det)
        res = await decide(pilot, cid, "confirm")
        assert res.status_code == 200, res.text
        card = res.json()["card"]
        assert card["state"] == "confirmed"
        for d in ("confirm", "reject", "unsure"):
            res = await decide(pilot, cid, d)
            assert res.status_code == 409 and res.json()["code"] == "TRANSITION_NOT_ALLOWED", d
        # Undoing a confirmation needs a reason, and is its own event.
        res = await decide(pilot, cid, "reopen")
        assert res.status_code == 409 and res.json()["code"] == "NOTE_REQUIRED"
        res = await decide(pilot, cid, "reopen", "fel person, jackan är orange")
        assert res.status_code == 200 and res.json()["card"]["state"] == "ready"

        # The rev the person saw must be in force.
        stale = await decide(pilot, cid, "reject", rev=1)
        assert stale.status_code == 409 and stale.json()["code"] == "SUPERSEDED"
        # Inspect is the drone's closer look: not built yet, and nothing is written.
        res = await decide(pilot, cid, "inspect")
        assert res.status_code == 501 and res.json()["code"] == "NOT_BUILT"

        d = (await pilot.get(f"/api/v1/candidates/{cid}")).json()
        assert [x["decision"] for x in d["decisions"]] == [
            "unsure",
            "reject",
            "reopen",
            "confirm",
            "reopen",
        ]
        assert {x["operator_ref"] for x in d["decisions"]} == {"operator:kalle"}
    # A merged or new candidate is not a person's to confirm or reject.
    new_cid, new_det = await candidate(r, mid, state="new")
    async with person(r, "pilot", "kalle") as pilot:
        await view(pilot, new_det)
        res = await decide(pilot, new_cid, "confirm")
        assert res.status_code == 409 and res.json()["code"] == "TRANSITION_NOT_ALLOWED"
        assert (await decide(pilot, new_cid, "unsure")).status_code == 200

    async with r.flow.repo.engine.connect() as conn:
        found = (
            await conn.execute(
                text("SELECT found_candidate_id::text FROM missions WHERE id = :m"), {"m": mid}
            )
        ).scalar_one()
    assert found is None  # the confirmation was undone
    events = [e.type for e in r.g.bus.replay(r.g.bus.server_epoch, 0) or []]
    assert events.count("candidate.decided") == 6


async def test_only_humans_decide(migrated_db: Db, tmp_path: Path) -> None:
    async with rig(
        migrated_db.name, migrated_db.app_password, tmp_path, auth={"cli_may_confirm": False}
    ) as r:
        mid = await r.open_mission()
        cid, det = await candidate(r, mid)
        body = {"decision": "confirm", "note": "", "subject_rev": 1}
        # Service tokens: the AI, the search service.
        for token in (BRAIN, SEARCH):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=panel_asgi(r.g, create_panel_app(r.g))),  # pyright: ignore[reportArgumentType]
                base_url="http://127.0.0.1:8080",
                headers={**H, "Authorization": f"Bearer {token}"},
            ) as svc:
                for d in ("confirm", "reject", "unsure"):
                    res = await svc.post(
                        f"/api/v1/candidates/{cid}/decision", json=body | {"decision": d}
                    )
                    assert res.status_code in (401, 403), (token[:5], d, res.status_code)
                res = await svc.post(f"/api/v1/detections/{det}/view")
                assert res.status_code in (401, 403)
        for d in ("confirm", "reject", "unsure"):
            res = await r.search.post(  # pyright: ignore[reportOptionalMemberAccess]
                f"/api/v1/candidates/{cid}/decision", json=body | {"decision": d}
            )
            assert res.status_code in (403, 404), d  # no such route on the service listener
        # A CLI token (ci-*) outside the test rig.
        for d in ("confirm", "reject", "unsure"):
            res = await r.panel.post(
                f"/api/v1/candidates/{cid}/decision", json=body | {"decision": d}, headers=H
            )
            assert res.status_code == 403, d
            assert res.json()["code"] == "INTERACTIVE_SESSION_REQUIRED"
        res = await r.panel.post(f"/api/v1/detections/{det}/view", headers=H)
        assert res.status_code == 403
        # An observer confirms (after a view), but never inspects or reopens.
        async with person(r, "observer", "lisa") as observer:
            for d in ("inspect", "reopen"):
                res = await decide(observer, cid, d)
                assert res.status_code == 403 and res.json()["code"] == "FORBIDDEN_ROLE", d
            await view(observer, det)
            res = await decide(observer, cid, "confirm")
            assert res.status_code == 200, res.text
            res = await decide(observer, cid, "reopen", "fel")
            assert res.status_code == 403
        async with r.flow.repo.engine.connect() as conn:
            rows = (
                await conn.execute(text("SELECT decision, operator_ref FROM candidate_decisions"))
            ).all()
        assert [tuple(x) for x in rows] == [("confirm", "operator:lisa")]


async def test_confirm_requires_view(r: Rig) -> None:
    mid = await r.open_mission()
    cid, det = await candidate(r, mid)
    other_cid, other_det = await candidate(r, mid)
    async with person(r, "pilot", "kalle") as pilot, person(r, "pilot", "kalle") as tab:
        res = await decide(pilot, cid, "confirm")
        assert res.status_code == 409 and res.json()["code"] == "NOT_VIEWED"
        await view(pilot, other_det)  # another candidate's picture does not count
        res = await decide(pilot, cid, "confirm")
        assert res.status_code == 409 and res.json()["code"] == "NOT_VIEWED"
        await view(tab, det)  # nor does another session's view
        res = await decide(pilot, cid, "confirm")
        assert res.status_code == 409 and res.json()["code"] == "NOT_VIEWED"
        await view(pilot, det)
        res = await decide(pilot, cid, "confirm", "röd jacka, ligger still")
        assert res.status_code == 200, res.text
        assert other_cid != cid
    unknown = await r.panel.post(f"/api/v1/detections/{new_id()}/view", headers=H)
    assert unknown.status_code == 404 and unknown.json()["code"] == "CROP_NOT_FOUND"
    assert (await r.panel.get(f"/api/v1/views/{new_id()}")).status_code == 404


async def test_confirm_marks_the_find(r: Rig) -> None:
    """The find on the mission, a system line in the chat and ``candidate.decided``."""
    mid = await r.open_mission()
    cid, det = await candidate(r, mid)
    async with person(r, "pilot", "kalle") as pilot:
        await view(pilot, det)
        assert (await decide(pilot, cid, "confirm")).status_code == 200
        gpx = await pilot.get(f"/api/v1/candidates/{cid}/export.gpx")
    async with r.flow.repo.engine.connect() as conn:
        found = (
            await conn.execute(
                text("SELECT found_candidate_id::text FROM missions WHERE id = :m"), {"m": mid}
            )
        ).scalar_one()
    assert found == cid
    chat = (await r.panel.get(f"/api/v1/missions/{mid}/chat")).json()["items"]
    assert any(
        m["role"] == "system" and m["text"] == "Fynd bekräftat: K1 (av kalle)." for m in chat
    )
    decided = [
        e for e in r.g.bus.replay(r.g.bus.server_epoch, 0) or [] if e.type == "candidate.decided"
    ]
    assert len(decided) == 1 and decided[0].topic == "candidates"
    data: dict[str, Any] = decided[0].data
    assert data["state"] == "confirmed" and data["decision"]["decision"] == "confirm"
    assert gpx.status_code == 200, gpx.text
    assert gpx.headers["content-type"].startswith("application/gpx+xml")


async def test_gpx_export(r: Rig) -> None:
    mid = await r.open_mission()
    cid, det = await candidate(r, mid)
    res = await r.panel.get(f"/api/v1/candidates/{cid}/export.gpx")
    assert res.status_code == 409 and res.json()["code"] == "NOT_CONFIRMED"
    await view(r.panel, det)  # the test rig's harness, as operator:ci-harness
    assert (await decide(r.panel, cid, "confirm")).status_code == 200
    res = await r.panel.get(f"/api/v1/candidates/{cid}/export.gpx")
    assert res.status_code == 200
    root = ET.fromstring(res.content)  # noqa: S314 - our own server's answer in a test
    assert root.tag == f"{GPX_NS}gpx" and root.attrib["version"] == "1.1"
    assert root.attrib["creator"] == "Searchdrone"
    (wpt,) = root.findall(f"{GPX_NS}wpt")
    lat, lon = r.at(0, 100)
    assert abs(float(wpt.attrib["lat"]) - lat) < 1e-6
    assert abs(float(wpt.attrib["lon"]) - lon) < 1e-6
    assert wpt.findtext(f"{GPX_NS}name") == "K1"
    assert (wpt.findtext(f"{GPX_NS}time") or "").endswith("Z")
    # GPX 1.1 order inside wpt: time before name before desc.
    assert [c.tag.removeprefix(GPX_NS) for c in wpt] == ["time", "name", "desc"]


async def test_candidate_views_append_only(r: Rig) -> None:
    mid = await r.open_mission()
    cid, det = await candidate(r, mid)
    async with person(r, "pilot", "kalle") as pilot:
        await view(pilot, det)
        await decide(pilot, cid, "unsure")
    for sql in (
        "UPDATE candidate_views SET session_id = 'x'",
        "DELETE FROM candidate_views",
        "UPDATE candidate_decisions SET note = 'x'",
        "DELETE FROM candidate_decisions",
    ):
        with pytest.raises(DBAPIError):
            async with r.flow.repo.engine.begin() as conn:
                await conn.execute(text(sql))
