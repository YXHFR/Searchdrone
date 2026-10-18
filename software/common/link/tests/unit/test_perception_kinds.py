"""B2-01: the stage 2 messages in the SD-link catalogue (perception, crops, keep-outs).

The same rules as the base catalogue: who may send each kind, ``ttl_ms``, reliability,
the size limit, the golden files, and a wrong ``vehicle_id`` closes the link.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from link_helpers import T0, VEHICLE, envelope_from

from sd_link.close_codes import CloseCode
from sd_link.codes import SCHEMA_INVALID
from sd_link.constants import MAX_MESSAGE_BYTES
from sd_link.envelope import (
    GROUND_SENDER,
    DecodeError,
    build_envelope,
    check_ttl,
    decode,
    encode,
    may_send,
    vehicle_sender,
)
from sd_link.session import LinkState
from sd_link_testing import FakeLink, next_inbound
from sd_schemas.base import LatLon, SdModel, new_id
from sd_schemas.link import (
    CROP_JPEG_MAX_BYTES,
    CROP_MAX_PX,
    KEEPOUTS_MAX,
    PERCEPTION_DETECTIONS_MAX,
    TTL_REQUIRED_KINDS,
    BBox,
    CropData,
    CropRequest,
    DetectorInfo,
    GeoDetectionMsg,
    Keepout,
    KeepoutInstall,
    PerceptionReport,
    Pose,
)
from sd_testing import FakeClock

GOLDEN = Path(__file__).parents[3] / "schemas" / "tests" / "golden" / "sdlink"
LAT, LON = -35.36146, 149.16523  # near CMAC


def detection() -> GeoDetectionMsg:
    return GeoDetectionMsg(
        detection_id=new_id(),
        bbox=BBox(x=1200.5, y=800.25, w=40.75, h=90.5),
        score=0.987654321,
        lat=LAT,
        lon=LON,
        sigma95_m=4.56789,
        px_size=90.5,
        crop_inline=True,
    )


def report(n: int = 1) -> PerceptionReport:
    return PerceptionReport(
        still_id=new_id(),
        command_id=new_id(),
        captured_utc=T0,
        sensor="rgb",
        pose=Pose(
            lat=LAT,
            lon=LON,
            alt_amsl_m=624.123,
            agl_m=40.456,
            roll_deg=-0.5,
            pitch_deg=1.25,
            yaw_deg=271.75,
            source_ts_ms=1_234_567,
        ),
        footprint=[
            LatLon(lat=LAT + dy, lon=LON + dx)
            for dy, dx in ((3e-4, -4e-4), (3e-4, 4e-4), (-3e-4, 4e-4), (-3e-4, -4e-4))
        ],
        px_per_m=50.0,
        detector=DetectorInfo(
            name="damo-yolo-s",
            license="Apache-2.0",
            version="0.3.1",
            profile_id="det_damo_yolo_s_v1",
            placement="onboard",
            ok=True,
            tiles=45,
            latency_ms=412.5,
            rate_hz=1.5,
        ),
        detections=[detection() for _ in range(n)],
        detections_dropped=7,
    )


def crop(size: int = 1000) -> CropData:
    return CropData(
        ref_id=new_id(),
        purpose="detection",
        still_id=new_id(),
        bbox_in_still=BBox(x=1000, y=700, w=384, h=384),
        width=CROP_MAX_PX,
        height=CROP_MAX_PX,
        jpeg=b"\xff\xd8\xff" + b"\x7f" * (size - 3),
    )


def keepouts(n: int = 1) -> KeepoutInstall:
    return KeepoutInstall(
        install_seq=n,
        keepouts=[
            Keepout(
                keepout_id=new_id(),
                lat=LAT,
                lon=LON,
                radius_m=34.56789,
                source="candidate",
                source_id=new_id(),
            )
            for _ in range(n)
        ],
    )


def view_request() -> CropRequest:
    return CropRequest(
        request_id=new_id(), still_id=new_id(), center_px=(2304, 1296), side_px=1536, max_px=512
    )


FROM_VEHICLE: list[tuple[str, SdModel]] = [("PerceptionReport", report()), ("CropData", crop())]
FROM_GROUND: list[tuple[str, SdModel]] = [
    ("CropRequest", view_request()),
    ("KeepoutInstall", keepouts()),
]


@pytest.mark.parametrize(("kind", "body"), FROM_VEHICLE, ids=lambda v: str(v)[:20])
def test_vehicle_sends_reports_and_crops(kind: str, body: SdModel) -> None:
    assert may_send("vehicle", kind) and not may_send("ground", kind)
    env = envelope_from(body, side="vehicle", ttl_ms=30_000)
    assert env.kind == kind and env.reliable and env.ttl_ms == 30_000


@pytest.mark.parametrize(("kind", "body"), FROM_GROUND, ids=lambda v: str(v)[:20])
def test_ground_sends_view_requests_and_keepouts(kind: str, body: SdModel) -> None:
    assert may_send("ground", kind) and not may_send("vehicle", kind)
    env = envelope_from(body, side="ground", ttl_ms=10_000)
    assert env.kind == kind and env.reliable


@pytest.mark.parametrize("kind", sorted(TTL_REQUIRED_KINDS))
def test_stage2_kinds_need_ttl(kind: str) -> None:
    with pytest.raises(ValueError, match="ttl"):
        check_ttl(kind, None)
    with pytest.raises(ValueError, match="ttl"):
        check_ttl(kind, 0)
    check_ttl(kind, 1)


@pytest.mark.parametrize(("kind", "body"), FROM_VEHICLE + FROM_GROUND, ids=lambda v: str(v)[:20])
def test_build_refuses_a_missing_ttl(kind: str, body: SdModel) -> None:
    side = "vehicle" if may_send("vehicle", kind) else "ground"
    with pytest.raises(ValueError, match="ttl"):
        envelope_from(body, side=side, ttl_ms=None)


def test_decode_refuses_a_missing_ttl() -> None:
    raw = json.loads(encode(envelope_from(report(), side="vehicle", ttl_ms=30_000)))
    raw["ttl_ms"] = None
    with pytest.raises(DecodeError) as err:
        decode(json.dumps(raw))
    assert err.value.reason == SCHEMA_INVALID and "ttl" in err.value.detail


def test_the_largest_messages_fit_the_link() -> None:
    """A full report, the largest crop and a full keep-out list each fit one message."""
    for body, side in (
        (report(PERCEPTION_DETECTIONS_MAX), "vehicle"),
        (crop(CROP_JPEG_MAX_BYTES), "vehicle"),
        (keepouts(KEEPOUTS_MAX), "ground"),
    ):
        text = encode(envelope_from(body, side=side, ttl_ms=30_000))
        assert len(text.encode()) <= MAX_MESSAGE_BYTES, type(body).__name__
        back, typed = decode(text)
        assert typed == body and back.kind == type(body).__name__


async def test_crops_and_keepouts_cross_the_link() -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        body = crop(CROP_JPEG_MAX_BYTES)
        await link.vehicle.send(body, ttl_ms=30_000)
        assert (await next_inbound(link.ground, CropData)).body == body
        install = keepouts(3)
        await link.ground.send(install, ttl_ms=10_000)
        assert (await next_inbound(link.vehicle, KeepoutInstall)).body == install


@pytest.mark.parametrize(
    ("body", "side"), [(report(), "vehicle"), (crop(), "vehicle"), (keepouts(), "ground")]
)
async def test_wrong_vehicle_id_closes_the_link(body: SdModel, side: str) -> None:
    clock = FakeClock()
    async with FakeLink(clock) as link:
        sender = vehicle_sender("sim-02") if side == "vehicle" else GROUND_SENDER
        other = build_envelope(
            body, sender=sender, vehicle_id="sim-02", seq=1, sent_utc=clock.utc_now(), ttl_ms=5000
        )
        if side == "vehicle":
            await link.vehicle_transport.send(encode(other))
            await clock.run_for(0.1)
            assert link.ground.state is LinkState.CLOSED
            assert link.ground_transport.close_code == CloseCode.FORBIDDEN
        else:
            await link.ground_transport.send(encode(other))
            await clock.run_for(0.1)
            assert link.vehicle.state is LinkState.CLOSED
            assert link.vehicle_transport.close_code == CloseCode.FORBIDDEN


def test_golden_files_for_stage2_kinds() -> None:
    kinds = {json.loads(p.read_text(encoding="utf-8"))["kind"] for p in GOLDEN.glob("*.json")}
    assert kinds >= TTL_REQUIRED_KINDS
    assert VEHICLE  # the golden files use the test vehicle
