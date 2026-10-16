"""Missions (AUTO) and the fence: encoding, hashing, upload and read-back.

A search mission has this shape (index ``seq``):

====================  ======================  ==================  ==========================
seq                   command                 frame               values
====================  ======================  ==================  ==========================
0                     NAV_WAYPOINT (16)       GLOBAL              home (ArduPilot overwrites)
1                     DO_CHANGE_SPEED (178)   MISSION             1, speed, -1
2 .. last_nav_seq     NAV_WAYPOINT (16)       GLOBAL_RELATIVE_ALT degrees x 1e7, height
last_nav_seq + 1      NAV_LOITER_TIME (19)    GLOBAL_RELATIVE_ALT last point, 30 s
last_nav_seq + 2      NAV_RETURN_TO_LAUNCH    MISSION             -
====================  ======================  ==================  ==========================

The last two items end the mission safely even if the onboard computer stops.

The hashes are normalised, because ArduPilot packs some fields when it stores them:
``mission_hash`` skips item 0 (home), ``fence_hash`` covers every item. Both sides
(onboard and ground) use these functions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.allowlist import FC_COMPID, FENCE_MAX_VERTICES
from sd_mav.vehicle import SendOrigin, Vehicle
from sd_safety.constants import (
    MAV_ACK_TIMEOUT_S,
    MISSION_END_LOITER_S,
    MISSION_MAX_ITEMS,
    MISSION_UPLOAD_TIMEOUT_S,
)
from sd_schemas.base import LatLonAlt

__all__ = [
    "FENCE",
    "MISSION",
    "MissionError",
    "MissionItem",
    "download",
    "fence_hash",
    "fence_items",
    "last_nav_seq",
    "mission_hash",
    "survey_mission",
    "upload",
]

MISSION: Final = mavlink.MAV_MISSION_TYPE_MISSION  # 0
FENCE: Final = mavlink.MAV_MISSION_TYPE_FENCE  # 1
_ATTEMPTS: Final = 3


class MissionError(Exception):
    """An upload or read-back that did not complete."""


@dataclass(frozen=True, slots=True)
class MissionItem:
    seq: int
    command: int
    frame: int
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    param4: float = 0.0
    x: int = 0  # latitude x 1e7
    y: int = 0  # longitude x 1e7
    z: float = 0.0  # height above home (relative frames) or above sea level (seq 0)
    mission_type: int = MISSION
    current: int = 0
    autocontinue: int = 1

    def to_message(self, target_system: int) -> Any:
        return mavlink.MAVLink_mission_item_int_message(
            target_system,
            FC_COMPID,
            self.seq,
            self.frame,
            self.command,
            self.current,
            self.autocontinue,
            self.param1,
            self.param2,
            self.param3,
            self.param4,
            self.x,
            self.y,
            self.z,
            self.mission_type,
        )

    @classmethod
    def from_message(cls, msg: Any) -> MissionItem:
        return cls(
            seq=int(msg.seq),
            command=int(msg.command),
            frame=int(msg.frame),
            param1=float(msg.param1),
            param2=float(msg.param2),
            param3=float(msg.param3),
            param4=float(msg.param4),
            x=int(msg.x),
            y=int(msg.y),
            z=float(msg.z),
            mission_type=int(msg.mission_type),
            current=int(msg.current),
            autocontinue=int(msg.autocontinue),
        )


def _deg7(value: float, limit: float) -> int:
    if not math.isfinite(value) or not -limit <= value <= limit:
        raise ValueError(f"coordinate {value} out of range")
    return round(value * 1e7)


def survey_mission(
    home: LatLonAlt,
    points: Sequence[tuple[float, float]],
    alt_m: float,
    speed_ms: float,
    loiter_s: int = MISSION_END_LOITER_S,
) -> list[MissionItem]:
    """The AUTO mission that flies ``points`` (lat, lon) at ``alt_m`` above home."""
    if not points:
        raise ValueError("a mission needs at least one point")
    if len(points) + 4 > MISSION_MAX_ITEMS:
        raise ValueError(f"{len(points)} points, at most {MISSION_MAX_ITEMS - 4}")
    if not math.isfinite(alt_m) or alt_m < 0:
        raise ValueError(f"height {alt_m}")
    if not math.isfinite(speed_ms) or speed_ms <= 0:
        raise ValueError(f"speed {speed_ms}")
    rel = mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
    items = [
        MissionItem(
            0,
            mavlink.MAV_CMD_NAV_WAYPOINT,
            mavlink.MAV_FRAME_GLOBAL,
            x=_deg7(home.lat, 90),
            y=_deg7(home.lon, 180),
            z=home.alt_amsl_m,
        ),
        MissionItem(
            1,
            mavlink.MAV_CMD_DO_CHANGE_SPEED,
            mavlink.MAV_FRAME_MISSION,
            param1=1,
            param2=speed_ms,
            param3=-1,
        ),
    ]
    for lat, lon in points:
        items.append(
            MissionItem(
                len(items),
                mavlink.MAV_CMD_NAV_WAYPOINT,
                rel,
                x=_deg7(lat, 90),
                y=_deg7(lon, 180),
                z=alt_m,
            )
        )
    last = items[-1]
    items.append(
        MissionItem(
            len(items),
            mavlink.MAV_CMD_NAV_LOITER_TIME,
            rel,
            param1=loiter_s,
            x=last.x,
            y=last.y,
            z=alt_m,
        )
    )
    items.append(
        MissionItem(len(items), mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH, mavlink.MAV_FRAME_MISSION)
    )
    return items


def last_nav_seq(items: Sequence[MissionItem]) -> int:
    """The index of the last search waypoint (before the loiter and return items)."""
    return len(items) - 3


def fence_items(polygon: Sequence[tuple[float, float]]) -> list[MissionItem]:
    """An inclusion polygon (lat, lon corners, not closed) as fence items."""
    corners = list(polygon)
    if len(corners) > 1 and corners[0] == corners[-1]:
        corners = corners[:-1]
    if not 3 <= len(corners) <= FENCE_MAX_VERTICES:
        raise ValueError(f"a fence polygon has 3..{FENCE_MAX_VERTICES} corners, not {len(corners)}")
    return [
        MissionItem(
            i,
            mavlink.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION,
            mavlink.MAV_FRAME_GLOBAL,
            param1=len(corners),
            x=_deg7(lat, 90),
            y=_deg7(lon, 180),
            mission_type=FENCE,
        )
        for i, (lat, lon) in enumerate(corners)
    ]


def _norm(item: MissionItem) -> tuple[Any, ...]:
    # Frames 3 and 6 count as the same frame, so the frame is not part of the hash.
    match item.command:
        case mavlink.MAV_CMD_NAV_WAYPOINT:
            return (16, item.x, item.y, round(float(item.z), 1))
        case mavlink.MAV_CMD_DO_CHANGE_SPEED:
            return (178, int(item.param1), round(float(item.param2), 1))
        case mavlink.MAV_CMD_NAV_LOITER_TIME:
            return (19, int(item.param1), item.x, item.y, round(float(item.z), 1))
        case mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
            return (20,)
        case _:
            raise ValueError(f"command {item.command} is not in the allowlist")


def _digest(rows: list[Any]) -> str:
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def mission_hash(items: Sequence[MissionItem]) -> str:
    """Hash of a mission without item 0 (home), which ArduPilot overwrites."""
    return _digest([_norm(i) for i in items if i.seq >= 1])


def fence_hash(items: Sequence[MissionItem]) -> str:
    """Hash of every fence item."""
    return _digest([(i.command, int(i.param1), i.x, i.y) for i in items])


# --- protocol ------------------------------------------------------------------------------


def _is(vehicle: Vehicle, kind: str, mission_type: int) -> Any:
    return lambda m: (
        m.get_type() == kind and int(m.mission_type) == mission_type and vehicle.addressed_to_us(m)
    )


async def upload(vehicle: Vehicle, items: Sequence[MissionItem], *, origin: SendOrigin) -> None:
    """Upload ``items`` (all of the same mission type); ``MissionError`` on failure.

    Answers each ``MISSION_REQUEST_INT`` (or legacy ``MISSION_REQUEST``) and waits for
    ``MISSION_ACK`` accepted, all within ``MISSION_UPLOAD_TIMEOUT_S``. Every message
    goes through ``Vehicle.send`` and so through the allowlist.
    """
    if not items:
        raise ValueError("nothing to upload")
    mtype = items[0].mission_type
    if any(i.mission_type != mtype for i in items) or [i.seq for i in items] != list(
        range(len(items))
    ):
        raise ValueError("items must share one mission type and be numbered 0..n-1")
    clock = vehicle.clock
    deadline = clock.monotonic() + MISSION_UPLOAD_TIMEOUT_S
    count = mavlink.MAVLink_mission_count_message(vehicle.fc_sysid, FC_COMPID, len(items), mtype)
    await vehicle.send(count, origin=origin)
    silent = 0

    def wanted(m: Any) -> bool:
        return (
            m.get_type() in ("MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK")
            and int(m.mission_type) == mtype
            and vehicle.addressed_to_us(m)
        )

    while True:
        left = deadline - clock.monotonic()
        if left <= 0:
            raise MissionError("upload timed out")
        reply = await vehicle.wait_message(wanted, min(MAV_ACK_TIMEOUT_S, left))
        if reply is None:
            silent += 1
            if silent >= _ATTEMPTS:
                raise MissionError("the flight controller stopped answering the upload")
            continue
        silent = 0
        if reply.get_type() == "MISSION_ACK":
            if int(reply.type) == mavlink.MAV_MISSION_ACCEPTED:
                return
            raise MissionError(f"upload refused: MAV_MISSION_RESULT {int(reply.type)}")
        seq = int(reply.seq)
        if not 0 <= seq < len(items):
            raise MissionError(f"the flight controller asked for item {seq}")
        await vehicle.send(items[seq].to_message(vehicle.fc_sysid), origin=origin)


async def download(vehicle: Vehicle, mission_type: int, *, origin: SendOrigin) -> list[MissionItem]:
    """Read back the stored mission or fence; ``MissionError`` on failure."""
    sysid = vehicle.fc_sysid
    count_msg = None
    for _ in range(_ATTEMPTS):
        await vehicle.send(
            mavlink.MAVLink_mission_request_list_message(sysid, FC_COMPID, mission_type),
            origin=origin,
        )
        count_msg = await vehicle.wait_message(
            _is(vehicle, "MISSION_COUNT", mission_type), MAV_ACK_TIMEOUT_S
        )
        if count_msg is not None:
            break
    if count_msg is None:
        raise MissionError("no MISSION_COUNT")
    count = int(count_msg.count)
    if count > MISSION_MAX_ITEMS:
        raise MissionError(f"{count} items stored, more than {MISSION_MAX_ITEMS}")
    items: list[MissionItem] = []
    for seq in range(count):
        reply = None
        for _ in range(_ATTEMPTS):
            await vehicle.send(
                mavlink.MAVLink_mission_request_int_message(sysid, FC_COMPID, seq, mission_type),
                origin=origin,
            )
            reply = await vehicle.wait_message(
                lambda m, s=seq: (
                    m.get_type() == "MISSION_ITEM_INT"
                    and int(m.mission_type) == mission_type
                    and int(m.seq) == s
                    and vehicle.addressed_to_us(m)
                ),
                MAV_ACK_TIMEOUT_S,
            )
            if reply is not None:
                break
        if reply is None:
            raise MissionError(f"item {seq} did not arrive")
        items.append(MissionItem.from_message(reply))
    await vehicle.send(
        mavlink.MAVLink_mission_ack_message(
            sysid, FC_COMPID, mavlink.MAV_MISSION_ACCEPTED, mission_type
        ),
        origin=origin,
    )
    return items
