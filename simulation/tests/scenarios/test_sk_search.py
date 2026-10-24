"""SK01 and SK12: the search service on ArduCopter SITL, with synthetic person reports.

The simulator has no person model yet, so the onboard side sends what its perception
task would send for a person: one report a still (twice a second) with the flight
controller's position as the pose, a ``person`` detection where the person stands (a
metre or two off, as georeferencing is) while it is under the camera, and a JPEG crop of
a person in a red jacket on grass. The reports go the real way: the onboard link's
queue, SD-link, the ground's perception queue and table. ``sd-search`` runs in the test
(its service loop and client over the served commander listener, as sd-brain does in
these tests); it dying is played by cancelling its task without a goodbye, so the ground
only notices that its heartbeat stops.

* SK01: the drone flies over the person in red and over a person in blue; the one in
  red becomes a candidate seen in several stills with ``garment:0 = match`` and the
  higher priority, the one in blue a conflict (another strong colour), and the median
  time from a stored detection to its candidate observation is under a second.
* SK12: ``sd-search`` dies mid-search while reports keep coming and is started again:
  every person detection is observed exactly once, the person in red is one candidate,
  the health part ``search`` goes down and comes back without touching ``ready``, and
  STOP while the search service is down brakes and holds as always.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import statistics
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent))

import flow_rig  # pyright: ignore[reportMissingImports]
from scenario_rig import (  # pyright: ignore[reportMissingImports]
    FIXTURE,
    Scenario,
    fresh_db,
    migrated_db,
    passwords,
    pin,
    scenario,
    sim,
)
from test_sc_emergency import stopped_and_holding  # pyright: ignore[reportMissingImports]

from sd_onboard.perception.pipeline import Perception
from sd_safety.targets import Plane, distance_m
from sd_schemas.base import LatLon, new_id
from sd_schemas.link import BBox, CropData, DetectorInfo, GeoDetectionMsg, PerceptionReport, Pose
from sd_schemas.search import SearchProfile
from sd_search.app import SearchService
from sd_search.client import GroundClient
from sd_sim.report import measure
from sd_sim.search_world import seed_offset

pytestmark = [pytest.mark.sitl, pytest.mark.slow]
__all__ = ["fresh_db", "migrated_db", "passwords", "sim"]

STILL_S = 0.5  # one still twice a second
SEEN_M = 25.0  # the camera's half width on the ground at 40 m (66 degrees)
PX_PER_M = 24.0
SQUARE = BBox(x=60, y=60, w=96, h=96)  # the crop's square in the still
BOX = BBox(x=92, y=92, w=32, h=32)  # the person: the square's middle third, 1.33 m
RED, BLUE, GRASS = (200, 30, 30), (30, 60, 200), (70, 120, 40)
OFFSETS = ((0.8, -0.5), (-1.2, 0.4), (0.3, 1.1), (-0.4, -1.3))  # georeferencing's error
LATENCY_S = 1.0


def _crop(color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (96, 96), GRASS)
    ImageDraw.Draw(img).rectangle((32, 32, 63, 63), fill=color)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=92)
    return out.getvalue()


class People:
    """The onboard side's synthetic person reports while the drone flies."""

    def __init__(self, sc: Scenario, people: dict[str, tuple[tuple[float, float], bytes]]) -> None:
        self.sc = sc
        self.people = people
        self.sent: dict[str, list[str]] = {name: [] for name in people}
        self.stills = 0

    def _report(self) -> Perception | None:
        tel = self.sc.tel
        plane = Plane(tel.lat, tel.lon)
        dets: list[GeoDetectionMsg] = []
        crops: list[CropData] = []
        still_id = new_id()
        for name, ((lat, lon), jpeg) in self.people.items():
            if distance_m(tel.lat, tel.lon, lat, lon) > SEEN_M:
                continue
            dx, dy = OFFSETS[len(self.sent[name]) % len(OFFSETS)]
            seen = Plane(lat, lon).latlon(dx, dy)
            did = new_id()
            dets.append(
                GeoDetectionMsg(
                    detection_id=did,
                    bbox=BOX,
                    score=0.62,
                    lat=seen[0],
                    lon=seen[1],
                    sigma95_m=3.0,
                    px_size=32.0,
                    crop_inline=True,
                    label="person",
                    source="model",
                )
            )
            crops.append(
                CropData(
                    ref_id=did,
                    purpose="detection",
                    still_id=still_id,
                    bbox_in_still=SQUARE,
                    width=96,
                    height=96,
                    jpeg=jpeg,
                )
            )
            self.sent[name].append(str(did))
        if not dets:
            return None
        corners = [plane.latlon(e, n) for e, n in ((-30, 20), (30, 20), (30, -20), (-30, -20))]
        alt = float(FIXTURE["home"]["alt_amsl_m"]) + tel.alt_rel_home_m
        report = PerceptionReport(
            still_id=still_id,
            command_id=None,
            captured_utc=self.sc.r.clock.utc_now(),
            sensor="rgb",
            pose=Pose(
                lat=tel.lat,
                lon=tel.lon,
                alt_amsl_m=alt,
                agl_m=tel.alt_rel_home_m,
                roll_deg=0,
                pitch_deg=0,
                yaw_deg=0,
                source_ts_ms=0,
            ),
            footprint=[LatLon(lat=la, lon=lo) for la, lo in corners],
            px_per_m=PX_PER_M,
            synthetic=True,
            detector=DetectorInfo(
                name="sd_detect:sitl-synthetic",
                license="Apache-2.0",
                version="0",
                placement="onboard",
                ok=True,
                tiles=1,
                latency_ms=30,
                rate_hz=1 / STILL_S,
            ),
            detections=dets,
        )
        return Perception(report, crops)

    async def run(self) -> None:
        while True:
            out = self._report()
            if out is not None:
                self.sc.r.onboard.link.perception(out)
                self.stills += 1
            await asyncio.sleep(STILL_S)

    def total(self) -> int:
        return sum(len(v) for v in self.sent.values())


class Profile:
    """The approved profile (a fixture): a red jacket, as the panel would approve it."""

    def __init__(self, sc: Scenario) -> None:
        now = sc.r.clock.utc_now().isoformat()
        self.profile = {
            "profile_id": str(new_id()),
            "profile_version": 1,
            "subject_category": "hiker",
            "garments": [{"part": "upper", "colors": ["red"], "certainty": "certain"}],
            "last_seen": {"lat": sc.r.home[0], "lon": sc.r.home[1], "at_utc": now},
            "approved_by": "operator:demo",
            "approved_at": now,
        }

    async def current(self, mission_id: str) -> SearchProfile | None:
        return SearchProfile.model_validate(self.profile | {"mission_id": mission_id})


@contextlib.asynccontextmanager
async def search_service(sc: Scenario, profile: Profile) -> AsyncIterator[asyncio.Future[None]]:
    """``sd-search`` on the served commander listener, until the test kills it."""
    client = GroundClient(f"http://127.0.0.1:{sc.port}", flow_rig.SEARCH)
    service = SearchService(client, profile)
    task = asyncio.ensure_future(service.run(asyncio.Event(), poll_s=0.25))
    try:
        yield task
    finally:
        task.cancel()  # no goodbye: the ground only sees the heartbeat stop
        await asyncio.gather(task, return_exceptions=True)
        await client.aclose()


async def _rows(sc: Scenario, sql: str, **params: Any) -> list[Any]:
    async with sc.r.flow.repo.engine.connect() as conn:
        return list((await conn.execute(text(sql), {"m": sc.mid, **params})).all())


async def observed(sc: Scenario) -> tuple[int, int]:
    """Observations and the distinct detections they observe, in this mission."""
    rows = await _rows(
        sc,
        "SELECT COUNT(*), COUNT(DISTINCT o.detection_id) FROM candidate_observations o "
        "JOIN detections d ON d.detection_id = o.detection_id "
        "WHERE d.mission_id = CAST(:m AS uuid)",
    )
    return int(rows[0][0]), int(rows[0][1])


async def stored(sc: Scenario) -> int:
    rows = await _rows(
        sc,
        "SELECT COUNT(*) FROM detections WHERE mission_id = CAST(:m AS uuid) AND label = 'person'",
    )
    return int(rows[0][0])


async def caught_up(sc: Scenario, people: People, limit_s: float) -> None:
    async def done() -> bool:
        n = await stored(sc)
        return n == people.total() and (await observed(sc))[1] == n

    end = time.monotonic() + limit_s
    while not await done():
        if time.monotonic() > end:
            raise AssertionError(
                f"sd-search hann inte ikapp: {await stored(sc)} lagrade, "
                f"{await observed(sc)} observerade, {people.total()} skickade"
            )
        await asyncio.sleep(0.25)


async def queue(sc: Scenario) -> list[dict[str, Any]]:
    res = await sc.r.panel.get(f"/api/v1/missions/{sc.mid}/candidates")
    assert res.status_code == 200, res.text
    return res.json()["items"]


def near(items: list[dict[str, Any]], where: tuple[float, float]) -> list[dict[str, Any]]:
    return [c for c in items if distance_m(c["lat"], c["lon"], *where) <= 12.0]


async def fly_over(sc: Scenario, where: tuple[float, float]) -> None:
    res = await sc.order("goto", {"lat": where[0], "lon": where[1], "alt_m": 40, "speed_ms": 5})
    assert res.get("decision") == "ALLOW", res
    await sc.until(lambda: distance_m(sc.tel.lat, sc.tel.lon, *where) <= 5.0, 90, "over the person")
    await asyncio.sleep(3.1)  # the gate's rate limit: one AI action per 3 s


async def home_and_land(sc: Scenario) -> None:
    assert (await sc.press("return_home")).status_code == 202
    await sc.until(lambda: not sc.tel.armed, 240, "landed")
    await sc.check_invariants()


@pytest.mark.budget_s(480)
async def test_sk01_the_person_in_red_becomes_a_matching_candidate(
    sim: Any, migrated_db: Any, tmp_path: Path
) -> None:
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()
        p1 = pin("P1")
        plane = Plane(*sc.r.home)
        red = plane.latlon(0.0, 80.0)  # on the way to P1
        blue = plane.latlon(0.0, 160.0)
        people = People(sc, {"red": (red, _crop(RED)), "blue": (blue, _crop(BLUE))})
        sender = asyncio.ensure_future(people.run())
        try:
            async with search_service(sc, Profile(sc)):
                await fly_over(sc, p1)
                await caught_up(sc, people, 15)
                items = await queue(sc)
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

        assert len(people.sent["red"]) >= 2 and len(people.sent["blue"]) >= 2, people.sent
        (target,) = near(items, red)
        (other,) = near(items, blue)
        assert target["n_seen"] == len(people.sent["red"]) and target["n_seen"] >= 2
        assert target["fields"]["garment:0"] == "match", target
        assert other["fields"]["garment:0"] == "conflict", other  # blue, never red
        assert target["priority"] > other["priority"]
        assert target["state"] == "ready" and target["engine"] == "code"
        n, distinct = await observed(sc)
        assert n == distinct == people.total()

        lags = await _rows(
            sc,
            "SELECT EXTRACT(EPOCH FROM o.created_at - d.created_at) FROM candidate_observations o "
            "JOIN detections d ON d.detection_id = o.detection_id "
            "WHERE d.mission_id = CAST(:m AS uuid)",
        )
        median = statistics.median(float(row[0]) for row in lags)
        measure("median detection -> candidate", round(median, 3), LATENCY_S)
        assert median < LATENCY_S, f"median detection -> candidate {median:.2f} s"
        await home_and_land(sc)


@pytest.mark.search
@pytest.mark.budget_s(540)
@pytest.mark.parametrize("seed", [1, 2, 3])
async def test_sk12_search_service_restart_mid_search(
    sim: Any, migrated_db: Any, tmp_path: Path, seed: int
) -> None:
    de, dn = seed_offset(seed, 10.0, "SK12")  # the person moves with the seed
    async with scenario(sim, migrated_db, tmp_path) as sc:
        await sc.airborne()
        plane = Plane(*sc.r.home)
        red = plane.latlon(de, 60.0 + dn)
        people = People(sc, {"red": (red, _crop(RED))})
        profile = Profile(sc)
        sender = asyncio.ensure_future(people.run())
        try:
            res = await sc.order("goto", {"lat": red[0], "lon": red[1], "alt_m": 40, "speed_ms": 3})
            assert res.get("decision") == "ALLOW", res
            async with search_service(sc, profile):
                await sc.until(lambda: len(people.sent["red"]) >= 6, 90, "the person in view")
                await sc.until(
                    lambda: sc.r.g.search_seen_mono is not None, 10, "the search heartbeat"
                )
            # sd-search is dead; the drone flies on and the person is still in view.
            await sc.until(lambda: len(people.sent["red"]) >= 16, 30, "reports while down")
            health = (await sc.r.panel.get("/api/v1/health")).json()
            assert health["ready"] is True, health

            # STOP while the search service is down: the emergency path as always.
            t0 = time.monotonic()
            assert (await sc.press("stop")).status_code == 202
            await stopped_and_holding(sc, t0)  # 20 s more: the heartbeat is now stale
            health = (await sc.r.panel.get("/api/v1/health")).json()
            assert health["parts"]["search"] == "down", health["parts"]
            assert health["ready"] is True, health
            assert (await sc.press("resume")).status_code == 202
            await sc.until(lambda: sc.r.actions.latches.get(flow_rig.VID) is None, 5, "released")

            async with search_service(sc, profile):
                await caught_up(sc, people, 20)
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
                await caught_up(sc, people, 10)
                health = (await sc.r.panel.get("/api/v1/health")).json()
                assert health["parts"]["search"] == "ok", health["parts"]
                items = await queue(sc)
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

        n, distinct = await observed(sc)
        assert n == distinct == people.total() == await stored(sc)
        (target,) = near(items, red)
        assert target["n_seen"] == len(people.sent["red"])
        assert target["fields"]["garment:0"] == "match"
        seqs = [c["seq"] for c in items]
        assert len(seqs) == len(set(seqs))
        await home_and_land(sc)
