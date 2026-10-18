"""Perception reports from the vehicle, end to end on the ground: stored detections,
finds, thumbnails, the panel's routes, the live events and the AI commander's event.

A fake onboard computer sends ``PerceptionReport`` over the real SD-link server into a
migrated database (``helpers/flow_rig.py``).
"""

from __future__ import annotations

import base64
import io
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from flow_rig import VID, Rig, rig  # pyright: ignore[reportMissingImports]

from sd_ground.api.commander import to_commander
from sd_ground.api.live import TOPICS as LIVE_TOPICS
from sd_schemas.base import new_id
from sd_schemas.link import PerceptionDetection, PerceptionReport, StillPose

if TYPE_CHECKING:
    from conftest import Db

pytestmark = [pytest.mark.contract, pytest.mark.db]


@pytest.fixture
async def r(migrated_db: Db, tmp_path: Path) -> AsyncIterator[Rig]:
    async with rig(migrated_db.name, migrated_db.app_password, tmp_path) as built:
        yield built


def _jpeg() -> str:
    out = io.BytesIO()
    Image.new("RGB", (32, 32), (200, 30, 30)).save(out, format="JPEG")
    return base64.b64encode(out.getvalue()).decode("ascii")


def detection(r: Rig, label: str, east: float, north: float, **kw: Any) -> PerceptionDetection:
    lat, lon = r.at(east, north)
    base: dict[str, Any] = {
        "detection_id": new_id(),
        "label": label,
        "source": "color" if label == "red_block" else "oracle",
        "score": 0.7,
        "bbox": (100.0, 100.0, 20.0, 20.0),
        "lat": lat,
        "lon": lon,
        "sigma95_m": 5.0,
        "size_m": 0.6,
    }
    return PerceptionDetection.model_validate(base | kw)


def report(r: Rig, *dets: PerceptionDetection) -> PerceptionReport:
    lat, lon = r.home
    return PerceptionReport(
        still_id=new_id(),
        captured_utc=r.clock.utc_now(),
        pose=StillPose(
            lat=lat, lon=lon, alt_amsl_m=624, agl_m=40, roll_deg=0, pitch_deg=0, yaw_deg=0
        ),
        footprint=[],
        px_per_m=24.0,
        synthetic=True,
        sources=["color", "oracle"],
        latency_ms=30,
        detections=list(dets),
    )


async def send(r: Rig, rep: PerceptionReport) -> None:
    await r.onboard.session.send(rep)


async def handled(r: Rig, n: int) -> None:
    """Wait until the ground has taken ``n`` reports off its queue."""

    def done() -> bool:
        q = r.flow.reports.get(VID)
        return q is not None and q.handled >= n and not q.running

    await r.until(done, 10, f"{n} reports handled")


async def detections(r: Rig, mid: str) -> list[dict[str, Any]]:
    res = await r.panel.get(f"/api/v1/missions/{mid}/detections")
    assert res.status_code == 200, res.text
    assert "Sd-Not-Built" not in res.headers
    return res.json()


async def test_report_becomes_detections_finds_and_events(r: Rig) -> None:
    mid = await r.open_mission()
    block = detection(r, "red_block", 10.0, 200.0, crop_jpeg_b64=_jpeg())
    await send(
        r, report(r, block, detection(r, "car", 40.0, 170.0), detection(r, "road", 0.0, 180.0))
    )
    await r.until(lambda: mid in r.flow.finds and len(r.flow.finds[mid].items) == 3, 10, "3 finds")
    rows = await detections(r, mid)
    assert {(d["label"], d["kind"], d["source"]) for d in rows} == {
        ("red_block", "target", "color"),
        ("car", "context", "oracle"),
        ("road", "context", "oracle"),
    }
    stored = next(d for d in rows if d["label"] == "red_block")
    assert stored["has_crop"] and stored["find_id"] == stored["id"]

    crop = await r.panel.get(f"/api/v1/missions/{mid}/detections/{stored['id']}/crop")
    assert crop.status_code == 200 and crop.headers["content-type"] == "image/jpeg"
    assert crop.content == base64.b64decode(block.crop_jpeg_b64 or "")
    files = (await r.panel.get(f"/api/v1/missions/{mid}/files")).json()
    assert [f["kind"] for f in files if f["rel_path"].startswith("crops/")] == ["crop"]
    no_crop = next(d for d in rows if d["label"] == "car")
    res = await r.panel.get(f"/api/v1/missions/{mid}/detections/{no_crop['id']}/crop")
    assert res.status_code == 404

    # The same block again 2 m away joins its find; a second block 20 m away is new.
    await send(
        r,
        report(
            r,
            detection(r, "red_block", 11.5, 201.0, score=0.9),
            detection(r, "red_block", 30.0, 200.0),
        ),
    )
    await r.until(lambda: len(r.flow.finds[mid].items) == 4, 10, "four finds")
    finds = (await r.panel.get(f"/api/v1/missions/{mid}/finds")).json()
    blocks = [f for f in finds if f["label"] == "red_block"]
    assert sorted(f["count"] for f in blocks) == [1, 2]
    first = next(f for f in blocks if f["count"] == 2)
    assert first["name_sv"] == "rött block" and first["score"] == 0.9
    assert first["thumb_detection_id"] == stored["id"]  # the only one with a thumbnail

    events = [e for e in r.g.bus.replay(r.g.bus.server_epoch, 0) or [] if e.topic == "detections"]
    assert "detections" in LIVE_TOPICS
    types = [e.type for e in events]
    assert types.count("detection.new") == 5
    assert types.count("find.new") == 4 and types.count("find.updated") == 1
    to_ai = [to_commander(e) for e in events]
    ai = [e for e in to_ai if e is not None]
    assert [e["type"] for e in ai] == ["event.detection"] * 4
    assert {e["body"]["kind"] for e in ai} == {"target", "context"}


async def test_finds_survive_a_restart_of_the_ground(r: Rig) -> None:
    mid = await r.open_mission()
    first = detection(r, "red_block", 10.0, 200.0, score=0.6)
    seen_again = detection(r, "red_block", 13.0, 200.0, score=0.8, crop_jpeg_b64=_jpeg())
    car = detection(r, "car", 40.0, 170.0, source="oracle")
    car_seen = detection(r, "car", 41.0, 170.0, source="model", score=0.5)
    await send(r, report(r, first, car))
    await send(r, report(r, seen_again, car_seen))
    await handled(r, 2)
    before = (await r.panel.get(f"/api/v1/missions/{mid}/finds")).json()

    r.flow.finds.clear()  # what a restart loses
    after = (await r.panel.get(f"/api/v1/missions/{mid}/finds")).json()
    assert after == before
    block = next(f for f in after if f["label"] == "red_block")
    assert block["find_id"] == str(first.detection_id) and block["count"] == 2
    assert block["thumb_detection_id"] == str(seen_again.detection_id)
    assert block["score"] == 0.8
    car_find = next(f for f in after if f["label"] == "car")
    assert car_find["source"] == "model"  # a real detector saw it too

    # A resend after the restart is neither stored nor counted again.
    await send(r, report(r, seen_again))
    await handled(r, 3)
    again = (await r.panel.get(f"/api/v1/missions/{mid}/finds")).json()
    assert again == before
    assert len(await detections(r, mid)) == 4


async def test_finds_cache_is_bounded(r: Rig) -> None:
    from sd_ground.services.perception import MAX_CACHED_MISSIONS, finds_for

    mid = await r.open_mission()
    for _ in range(MAX_CACHED_MISSIONS + 2):
        await finds_for(r.flow, str(new_id()))  # e.g. the panel reading old missions
    await finds_for(r.flow, mid)
    assert len(r.flow.finds) == MAX_CACHED_MISSIONS
    assert mid in r.flow.finds


async def test_unknown_classes_bad_crops_and_resends(r: Rig) -> None:
    mid = await r.open_mission()
    bad_crop = base64.b64encode(b"not a jpeg").decode("ascii")
    block = detection(r, "red_block", 0.0, 150.0, crop_jpeg_b64=bad_crop)
    rep = report(r, block, detection(r, "submarine", 0.0, 160.0))
    await send(r, rep)
    await send(r, rep)  # a resend after a reconnect: stored once
    await handled(r, 2)
    await r.g.flush_events()
    rows = await detections(r, mid)
    assert [(d["label"], d["has_crop"]) for d in rows] == [("red_block", False)]
    assert [f.count for f in r.flow.finds[mid].items] == [1]
    assert VID in {d["vehicle_id"] for d in rows}


async def test_no_open_mission_stores_nothing(r: Rig) -> None:
    await send(r, report(r, detection(r, "red_block", 0.0, 150.0)))
    mid = await r.open_mission()
    assert await detections(r, mid) == []
