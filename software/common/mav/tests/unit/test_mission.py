"""Mission encoding, hashes and the upload/read-back protocol."""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from mav_helpers import HOME, ctx, run_with_clock
from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.allowlist import Profile, check
from sd_mav.connection import Role
from sd_mav.fake import CMAC_HOME, FakeFlightController, FakeVehicle
from sd_mav.mission import (
    FENCE,
    MISSION,
    MissionError,
    MissionItem,
    download,
    fence_hash,
    fence_items,
    last_nav_seq,
    mission_hash,
    survey_mission,
    upload,
)
from sd_mav.vehicle import ForbiddenMavlinkError, SendOrigin
from sd_obs.clock import FakeClock
from sd_safety.constants import MISSION_END_LOITER_S, MISSION_MAX_ITEMS
from sd_schemas.base import LatLonAlt

HOME_LLA = LatLonAlt(lat=CMAC_HOME[0], lon=CMAC_HOME[1], alt_amsl_m=CMAC_HOME[2])
NEAR = st.tuples(
    st.floats(HOME[0] - 0.003, HOME[0] + 0.003, allow_nan=False),
    st.floats(HOME[1] - 0.003, HOME[1] + 0.003, allow_nan=False),
)


@settings(max_examples=1000, deadline=None)
@given(
    points=st.lists(NEAR, min_size=1, max_size=60),
    alt=st.floats(0, 120, allow_nan=False),
    speed=st.floats(0.5, 15, allow_nan=False),
)
def test_mission_encoder(points: list[tuple[float, float]], alt: float, speed: float) -> None:
    items = survey_mission(HOME_LLA, points, alt, speed)
    assert len(items) == len(points) + 4 <= MISSION_MAX_ITEMS
    assert [i.seq for i in items] == list(range(len(items)))
    assert items[0].command == 16 and items[0].frame == mavlink.MAV_FRAME_GLOBAL
    assert (items[1].command, items[1].param1, items[1].param2, items[1].param3) == (
        178,
        1,
        speed,
        -1,
    )
    nav = items[2 : last_nav_seq(items) + 1]
    assert len(nav) == len(points)
    for item, (lat, lon) in zip(nav, points, strict=True):
        assert item.command == 16 and item.frame == mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
        assert abs(item.x - lat * 1e7) <= 1 and abs(item.y - lon * 1e7) <= 1
        assert item.z == alt
    loiter, rtl = items[-2], items[-1]
    assert (loiter.command, loiter.param1, loiter.x, loiter.y) == (
        19,
        MISSION_END_LOITER_S,
        nav[-1].x,
        nav[-1].y,
    )
    assert rtl.command == 20
    for item in items:  # everything the encoder makes passes the allowlist
        assert check(item.to_message(1), ctx()) is None


def test_mission_encoder_refuses_bad_input() -> None:
    with pytest.raises(ValueError):
        survey_mission(HOME_LLA, [], 40, 5)
    with pytest.raises(ValueError):
        survey_mission(HOME_LLA, [HOME] * (MISSION_MAX_ITEMS - 3), 40, 5)
    with pytest.raises(ValueError):
        survey_mission(HOME_LLA, [HOME], float("nan"), 5)
    with pytest.raises(ValueError):
        survey_mission(HOME_LLA, [HOME], 40, 0)
    with pytest.raises(ValueError):
        survey_mission(HOME_LLA, [(95.0, 0.0)], 40, 5)


def test_mission_hash_roundtrip() -> None:
    items = survey_mission(
        HOME_LLA, [(HOME[0] + 0.001, HOME[1]), (HOME[0] + 0.001, HOME[1] + 0.001)], 40.0, 7.0
    )
    # As ArduPilot stores it: item 0 replaced by its home, floats through float32,
    # waypoints in frame 6 instead of 3.
    stored = [
        MissionItem(0, 16, 0, x=1, y=2, z=600.0),
        *[
            MissionItem(
                i.seq,
                i.command,
                6 if i.frame == 3 else i.frame,
                i.param1,
                i.param2 + 1e-6,
                i.param3,
                i.param4,
                i.x,
                i.y,
                i.z + 1e-5,
            )
            for i in items[1:]
        ],
    ]
    assert mission_hash(items) == mission_hash(stored)
    moved = [
        *stored[:2],
        MissionItem(2, 16, 3, x=stored[2].x + 1, y=stored[2].y, z=40.0),
        *stored[3:],
    ]
    assert mission_hash(items) != mission_hash(moved)
    faster = [stored[0], MissionItem(1, 178, 2, 1, 8.0, -1), *stored[2:]]
    assert mission_hash(items) != mission_hash(faster)
    with pytest.raises(ValueError):
        mission_hash([MissionItem(1, 22, 3)])
    corners = [(HOME[0], HOME[1]), (HOME[0] + 0.001, HOME[1]), (HOME[0] + 0.001, HOME[1] + 0.001)]
    fence = fence_items(corners)
    assert fence_hash(fence) == fence_hash(fence_items([*corners, corners[0]]))  # closed ring
    assert fence_hash(fence) != fence_hash(fence_items(corners[::-1]))
    assert all(i.mission_type == FENCE and i.command == 5001 and i.param1 == 3 for i in fence)
    for item in fence:
        assert check(item.to_message(1), ctx(Profile.GROUND, "ground")) is None
    with pytest.raises(ValueError):
        fence_items(corners[:2])


def test_upload_and_download_against_the_fake() -> None:
    async def run() -> None:
        clock = FakeClock()
        fc = FakeFlightController(clock)
        onboard = FakeVehicle(clock, fc=fc)
        ground = FakeVehicle(clock, fc=fc, role=Role.GROUND)
        await onboard.start()
        await ground.start()
        await clock.run_for(2.0)
        items = survey_mission(
            HOME_LLA, [(HOME[0] + 0.001, HOME[1]), (HOME[0], HOME[1] + 0.001)], 40, 7
        )
        await run_with_clock(clock, upload(onboard, items, origin=SendOrigin.SKILL))
        back = await run_with_clock(clock, download(onboard, MISSION, origin=SendOrigin.STARTUP))
        assert mission_hash(back) == mission_hash(items)
        fence = fence_items(
            [
                (HOME[0] - 0.001, HOME[1] - 0.001),
                (HOME[0] + 0.001, HOME[1] - 0.001),
                (HOME[0], HOME[1] + 0.001),
            ]
        )
        await run_with_clock(clock, upload(ground, fence, origin=SendOrigin.GROUND))
        assert fence_hash(
            await run_with_clock(clock, download(ground, FENCE, origin=SendOrigin.GROUND))
        ) == fence_hash(fence)
        # PF7: the onboard computer reads the fence but never uploads one.
        assert fence_hash(
            await run_with_clock(clock, download(onboard, FENCE, origin=SendOrigin.STARTUP))
        ) == fence_hash(fence)
        with pytest.raises(ForbiddenMavlinkError):
            await upload(onboard, fence, origin=SendOrigin.SKILL)
        with pytest.raises(ForbiddenMavlinkError):
            await upload(ground, items, origin=SendOrigin.GROUND)
        with pytest.raises(ForbiddenMavlinkError):  # reading is fine, flying is not
            await upload(onboard, items, origin=SendOrigin.STARTUP)
        with pytest.raises(ValueError):
            await upload(onboard, [], origin=SendOrigin.SKILL)
        with pytest.raises(ValueError):
            await upload(onboard, items[1:], origin=SendOrigin.SKILL)
        fc.silent = True
        with pytest.raises(MissionError):
            await run_with_clock(clock, upload(onboard, items, origin=SendOrigin.SKILL))
        with pytest.raises(MissionError):
            await run_with_clock(clock, download(onboard, MISSION, origin=SendOrigin.STARTUP))
        await onboard.close()
        await ground.close()

    asyncio.run(run())


def test_upload_refused_by_the_flight_controller() -> None:
    async def run() -> None:
        clock = FakeClock()
        vehicle = FakeVehicle(clock)
        await vehicle.start()
        await clock.run_for(2.0)
        original = vehicle.fc._mission

        def refuse(link, msg, kind):  # type: ignore[no-untyped-def]
            if kind == "MISSION_COUNT":
                vehicle.fc._send(
                    mavlink.MAVLink_mission_ack_message(1, 191, mavlink.MAV_MISSION_NO_SPACE, 0),
                    link,
                )
                return
            original(link, msg, kind)

        vehicle.fc._mission = refuse  # type: ignore[method-assign]
        items = survey_mission(HOME_LLA, [HOME], 40, 5)
        with pytest.raises(MissionError, match="refused"):
            await run_with_clock(clock, upload(vehicle, items, origin=SendOrigin.SKILL))
        await vehicle.close()

    asyncio.run(run())
