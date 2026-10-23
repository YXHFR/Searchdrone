"""A closer look at a detection (``inspect_detection``) on ArduCopter SITL, and SK08.

The simulator has no camera in this physics, so the onboard executor's camera is a fake
that notes where the drone is when a still is taken and sends an empty perception
report for it (``command_id`` set), as the perception task would. The detection itself
is a synthetic person report sent the real way (the onboard link, SD-link, the ground's
perception queue and table).

* ``test_inspect_detection_geometry``: the pilot asks for a closer look at a person
  detection that a confirmed find's keep-out cylinder surrounds. The drone flies to the
  cylinder's edge on its own side, high enough that the target is within 25 degrees of
  straight down in every still, never with a track sample inside the cylinder, and
  hovers there when done.
* SK08: STOP and FLY HOME while the drone flies an inspection and the onboard link's
  perception queue is full of reports and crops: the same steps within the same times
  as without them (BRAKE within a second, still within 5 s, holding for 20 s; SMART_RTL
  within a second and landing within 3 m of home).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import math
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from PIL import Image
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scenario_rig import (  # pyright: ignore[reportMissingImports]
    FIXTURE,
    VID,
    Scenario,
    fresh_db,
    migrated_db,
    passwords,
    scenario,
    sim,
)
from test_sc_emergency import (  # pyright: ignore[reportMissingImports]
    HOME_MODES,
    STOP_ACK_S,
    stopped_and_holding,
)

from sd_onboard.link_client import MAX_PERCEPTION
from sd_onboard.perception.pipeline import Perception
from sd_safety.constants import PERSON_KEEPOUT_M
from sd_safety.targets import Plane, distance_m
from sd_schemas.base import LatLon, new_id
from sd_schemas.keepout import INSPECT_VIEW_DEG
from sd_schemas.link import BBox, CropData, DetectorInfo, GeoDetectionMsg, PerceptionReport, Pose
from sd_schemas.skills import SKILL_REGISTRY
from sd_sim.search_world import seed_offset

pytestmark = [pytest.mark.sitl, pytest.mark.slow]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

SIGMA_M = 3.0
TARGET_ENU = (0.0, 100.0)  # a person 100 m north of home, on the way to P1
STILLS = 5  # the skill's stills at the inspection point
FLOOD_S = 0.05  # how often the flood queues a report while SK08 presses its buttons
FLOOD_CROPS = 8
SQUARE = BBox(x=60, y=60, w=96, h=96)
BOX = BBox(x=92, y=92, w=32, h=32)
INSPECT_SKILLS = ["inspect_detection", "orbit", "capture_stills"]


def _jpeg() -> bytes:
    out = io.BytesIO()
    Image.effect_noise((96, 96), 64).convert("RGB").save(out, format="JPEG", quality=92)
    return out.getvalue()


def report(
    sc: Scenario, dets: list[GeoDetectionMsg], *, command_id: UUID | None = None
) -> PerceptionReport:
    """A still's report with the flight controller's position as the pose."""
    tel = sc.tel
    plane = Plane(tel.lat, tel.lon)
    corners = [plane.latlon(e, n) for e, n in ((-30, 20), (30, 20), (30, -20), (-30, -20))]
    return PerceptionReport(
        still_id=new_id(),
        command_id=command_id,
        captured_utc=sc.r.clock.utc_now(),
        sensor="rgb",
        pose=Pose(
            lat=tel.lat,
            lon=tel.lon,
            alt_amsl_m=float(FIXTURE["home"]["alt_amsl_m"]) + tel.alt_rel_home_m,
            agl_m=tel.alt_rel_home_m,
            roll_deg=0,
            pitch_deg=0,
            yaw_deg=0,
            source_ts_ms=0,
        ),
        footprint=[LatLon(lat=la, lon=lo) for la, lo in corners],
        px_per_m=24.0,
        synthetic=True,
        detector=DetectorInfo(
            name="sd_detect:sitl-synthetic",
            license="Apache-2.0",
            version="0",
            placement="onboard",
            ok=True,
            tiles=1,
            latency_ms=30,
            rate_hz=1,
        ),
        detections=dets,
    )


def person(where: tuple[float, float]) -> GeoDetectionMsg:
    return GeoDetectionMsg(
        detection_id=new_id(),
        bbox=BOX,
        score=0.62,
        lat=where[0],
        lon=where[1],
        sigma95_m=SIGMA_M,
        px_size=32.0,
        crop_inline=True,
        label="person",
        source="model",
    )


def with_crops(rep: PerceptionReport, jpeg: bytes) -> Perception:
    crops = [
        CropData(
            ref_id=d.detection_id,
            purpose="detection",
            still_id=rep.still_id,
            bbox_in_still=SQUARE,
            width=96,
            height=96,
            jpeg=jpeg,
        )
        for d in rep.detections
    ]
    return Perception(rep, crops)


async def stored(sc: Scenario, detection_id: UUID) -> bool:
    async with sc.r.flow.repo.engine.connect() as conn:
        row = await conn.execute(
            text("SELECT 1 FROM detections WHERE detection_id = :d"), {"d": str(detection_id)}
        )
        return row.first() is not None


async def a_detection(sc: Scenario, where: tuple[float, float]) -> GeoDetectionMsg:
    """One person detection at ``where``, stored on the ground."""
    det = person(where)
    sc.r.onboard.link.perception(with_crops(report(sc, [det]), _jpeg()))
    await sc.until(lambda: stored(sc, det.detection_id), 15, "the detection on the ground")
    return det


class Camera:
    """The onboard executor's camera: where the drone was at each still."""

    def __init__(self, sc: Scenario) -> None:
        self.sc = sc
        self.poses: list[tuple[float, float, float]] = []
        sc.r.onboard.ex.camera = self.capture

    async def capture(self, command_id: UUID) -> UUID | None:
        tel = self.sc.r.onboard.vehicle.telemetry
        self.poses.append((tel.lat, tel.lon, tel.alt_rel_home_m))
        rep = report(self.sc, [], command_id=command_id)
        self.sc.r.onboard.link.perception(Perception(rep, []))
        return rep.still_id


async def _status(sc: Scenario, command_id: str, *want: str) -> bool:
    return (await sc.command(command_id))["status"] in want


@contextlib.asynccontextmanager
async def flooded(sc: Scenario) -> AsyncIterator[None]:
    """Reports with crops queued faster than SD-link takes them: every ``FLOOD_S`` more
    than the onboard link's perception queue holds, so it is full and drops reports."""
    link = sc.r.onboard.link
    jpeg = _jpeg()
    far = sc.at(-120.0, 250.0)  # well away from the inspected person

    async def run() -> None:
        while True:
            for _ in range(MAX_PERCEPTION + 1):
                dets = [person(far) for _ in range(FLOOD_CROPS)]
                link.perception(with_crops(report(sc, dets), jpeg))
            await asyncio.sleep(FLOOD_S)

    task = asyncio.ensure_future(run())
    try:
        await sc.until(lambda: link.dropped_perception > 0, 30, "the perception queue full")
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def inspecting(sc: Scenario, det: GeoDetectionMsg) -> dict[str, Any]:
    res = await sc.operator_order(
        "inspect_detection", {"detection_id": str(det.detection_id), "standoff_m": 20}
    )
    assert res["gate"]["decision"] == "ALLOW", res
    return await sc.wait_command("inspect_detection", "running", 30)


async def airborne_for_inspection(sc: Scenario) -> None:
    charter = FIXTURE["charters"]["charter_test"]
    skills = [*charter["allowed_skills"], *INSPECT_SKILLS]
    assert all(SKILL_REGISTRY.get(s) is not None for s in skills)
    await sc.airborne(allowed_skills=skills)


@pytest.mark.budget_s(480)
async def test_inspect_detection_geometry(sim: Any, migrated_db: Any, tmp_path: Path) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await airborne_for_inspection(sc)
        target = sc.at(*TARGET_ENU)
        await sc.confirm_find(*target, sigma95_m=SIGMA_M)
        radius = PERSON_KEEPOUT_M + SIGMA_M
        det = await a_detection(sc, target)
        cam = Camera(sc)

        since = time.monotonic()
        await inspecting(sc, det)
        await sc.wait_command("inspect_detection", "done", 180)

        async def holding() -> bool:  # the snapshot may still be the one before the end
            return (await sc.onboard_state())["executor_state"] == "HOLD"

        await sc.until(holding, 5, "hovers after the inspection")

        assert len(cam.poses) == STILLS, cam.poses
        for lat, lon, alt in cam.poses:
            view = math.degrees(math.atan2(distance_m(lat, lon, *target), alt))
            assert view <= INSPECT_VIEW_DEG, (view, cam.poses)
            # On the drone's side, at the edge of the cylinder plus the margin.
            assert distance_m(lat, lon, *target) > radius
        inside = [
            s for s in sc.track if s.t >= since and distance_m(s.lat, s.lon, *target) < radius
        ]
        assert not inside, inside[:3]

        assert (await sc.press("return_home")).status_code == 202
        await sc.until(lambda: not sc.tel.armed, 240, "landed")
        assert sc.from_home_m() <= 3.0
        await sc.check_invariants()


@pytest.mark.search
@pytest.mark.budget_s(600)
@pytest.mark.parametrize(
    "seed",
    [pytest.param(1, marks=pytest.mark.e2e), 2, 3],  # seed 1 in just e2e, too
)
async def test_sk08_stop_and_fly_home_during_an_inspection(
    sim: Any, migrated_db: Any, tmp_path: Path, seed: int
) -> None:
    de, dn = seed_offset(seed, 15.0, "SK08")  # the person moves with the seed
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await airborne_for_inspection(sc)
        det = await a_detection(sc, sc.at(TARGET_ENU[0] + de, TARGET_ENU[1] + dn))
        Camera(sc)
        link = sc.r.onboard.link

        # STOP mid-inspection with the queue full.
        first = await inspecting(sc, det)
        async with flooded(sc):
            await sc.until(lambda: sc.from_home_m() >= 20.0, 60, "on the way")
            dropped = link.dropped_perception
            t0 = time.monotonic()
            assert (await sc.press("stop")).status_code == 202
            await stopped_and_holding(sc, t0)
            assert link.dropped_perception > dropped  # still full through STOP
        live = await sc.onboard_state()
        assert live["executor_state"] == "HELD" and live["held_by"] == "operator_stop", live
        await sc.until(
            lambda: _status(sc, first["id"], "cancelled", "superseded"), 10, "the inspection ended"
        )
        assert (await sc.press("resume")).status_code == 202
        await sc.until(lambda: sc.r.actions.latches.get(VID) is None, 10, "released")

        # FLY HOME mid-inspection with the queue full.
        second = await inspecting(sc, det)
        async with flooded(sc):
            await asyncio.sleep(3.0)
            dropped = link.dropped_perception
            assert (await sc.press("return_home")).status_code == 202
            await sc.until(lambda: sc.mode() in HOME_MODES, 2.0 * STOP_ACK_S, "flying home")
            assert sc.mode() == "SMART_RTL"
            assert link.dropped_perception > dropped  # still full through FLY HOME
        await sc.until(lambda: not sc.tel.armed, 240, "landed")
        assert sc.from_home_m() <= 3.0
        after = await sc.command(second["id"])
        assert after["status"] in ("cancelled", "superseded"), after
        await sc.check_invariants()
