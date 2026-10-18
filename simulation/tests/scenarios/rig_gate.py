"""Helpers for the gate scenarios (SC05, SC09, SC11, SC12) on top of ``scenario_rig``.

* ``fence_tamper``: the harness (253/190) uploads a fence that differs from the charter's
  while the vehicle is disarmed (SC09 g only); the upload is recorded for INV5;
* ``arms`` / ``takeoffs``: what the onboard computer sent to arm or take off;
* ``end_mission``: the pilot's emergency close of a mission on the ground.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from flow_rig import H  # pyright: ignore[reportMissingImports]
from pymavlink import mavutil
from scenario_rig import Scenario  # pyright: ignore[reportMissingImports]

from sd_sim.errors import InfraError
from sd_sim.harness import FC_COMPID, FC_SYSID

__all__ = ["FENCE_TAMPER_TYPES", "arms", "end_mission", "fence_tamper", "takeoffs"]

mavlink: Any = mavutil.mavlink
FENCE = 1
# What the harness may send in SC09 g besides its usual writes (INV5).
FENCE_TAMPER_TYPES = ("MISSION_COUNT", "MISSION_ITEM_INT")


def arms(sc: Scenario, since: int = 0) -> int:
    return sum(
        1
        for m in sc.tl.sent[since:]
        if m.sender == "onboard" and m.command == 400 and m.params and m.params[0] == 1
    )


def takeoffs(sc: Scenario, since: int = 0) -> int:
    return sum(1 for m in sc.tl.sent[since:] if m.sender == "onboard" and m.command == 22)


def _upload_fence(sc: Scenario, corners: list[tuple[float, float]]) -> None:
    conn = sc.sim.harness._conn  # pyright: ignore[reportPrivateUsage]
    conn.mav.mission_count_send(FC_SYSID, FC_COMPID, len(corners), FENCE)
    end = time.monotonic() + 10.0
    while time.monotonic() < end:
        msg = conn.recv_match(
            type=["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"],
            blocking=True,
            timeout=0.5,
        )
        if msg is None or msg.get_srcSystem() != FC_SYSID:
            continue
        if msg.get_type() == "MISSION_ACK":
            if int(msg.mission_type) != FENCE:
                continue
            if int(msg.type) != mavlink.MAV_MISSION_ACCEPTED:
                raise InfraError(f"staketet från harnessen avvisades ({msg.type})")
            return
        if int(getattr(msg, "mission_type", FENCE)) != FENCE:
            continue
        seq = int(msg.seq)
        lat, lon = corners[seq]
        conn.mav.mission_item_int_send(
            FC_SYSID,
            FC_COMPID,
            seq,
            mavlink.MAV_FRAME_GLOBAL,
            mavlink.MAV_CMD_NAV_FENCE_POLYGON_VERTEX_INCLUSION,
            0,
            0,
            len(corners),
            0,
            0,
            0,
            round(lat * 1e7),
            round(lon * 1e7),
            0,
            FENCE,
        )
    raise InfraError("staketet från harnessen blev inte kvitterat")


async def fence_tamper(sc: Scenario, half_m: float = 120.0) -> None:
    """A square fence around home that is not the charter's (only on the ground)."""
    assert not sc.tel.armed, "staketet ändras bara medan drönaren är desarmerad"
    corners = [
        sc.at(x, y)
        for x, y in [(-half_m, -half_m), (half_m, -half_m), (half_m, half_m), (-half_m, half_m)]
    ]
    sc._harness_sent("MISSION_COUNT", mission_type=FENCE)  # pyright: ignore[reportPrivateUsage]
    for _ in corners:
        sc._harness_sent("MISSION_ITEM_INT", mission_type=FENCE)  # pyright: ignore[reportPrivateUsage]
    await asyncio.to_thread(_upload_fence, sc, corners)


async def end_mission(sc: Scenario) -> Any:
    return await sc.r.panel.post(
        f"/api/v1/missions/{sc.mission_id}/end", json={"reason": "Test"}, headers=H
    )


async def commander_charter(
    sc: Scenario, name: str, area_enu: list[tuple[float, float]] | None = None, **values: Any
) -> Any:
    """A charter proposal on the AI's own path (commander API, brain token), built from
    the fixture's charter ``name`` like ``ScriptedBackend`` builds it, against the charter
    in force (``base_version``). ``area_enu`` replaces the area (metres east, north)."""
    from sd_brain.backends.scripted import _Fixture  # pyright: ignore[reportPrivateUsage]

    body = (
        _Fixture(sc.fixture).charter(name, "Testkoden.").model_dump(mode="json", exclude_none=True)
    )
    if area_enu is not None:
        ring = [[sc.at(e, n)[1], sc.at(e, n)[0]] for e, n in area_enu]
        body["area"] = {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}
    situation = await sc.situation()
    charter = situation["charter"]
    body["base_version"] = None if charter is None else charter["version"]
    body.update(values)
    return await sc.r.commander.post(
        f"/commander/v1/missions/{sc.mission_id}/charter-proposals", json=body
    )


async def clearance_status(sc: Scenario) -> str | None:
    """The newest clearance's status for the mission in the database."""
    rows = await sc.r.flow.repo._all(  # pyright: ignore[reportPrivateUsage]
        "SELECT status FROM clearances WHERE mission_id = :m ORDER BY confirmed_at DESC",
        m=sc.mission_id,
    )
    return None if not rows else str(rows[0]["status"])
