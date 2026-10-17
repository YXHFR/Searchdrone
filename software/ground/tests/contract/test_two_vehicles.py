"""B1-14: two vehicles write telemetry at the same time without their rows mixing.

Each vehicle has its own mission and its own track. Both write through ``Repo`` as the
app role, in parallel batches on separate connections; afterwards every row must belong to
its own vehicle, its own mission and its own track, with nothing lost.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import Db  # fixtures live in conftest.py (not importable at runtime)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helpers"))

from flow_rig import app_url  # pyright: ignore[reportMissingImports]

from sd_ground.db.engine import make_engine
from sd_ground.db.repo import Repo, TelemetryRow

pytestmark = [pytest.mark.contract, pytest.mark.db]

HOME_LAT, HOME_LON = -35.363261, 149.165230
VEHICLES = {"sim-01": (1, 0.0), "sim-02": (2, 0.002)}  # sysid, east offset in degrees
BATCHES, BATCH = 20, 10
T0 = dt.datetime(2026, 10, 17, 12, 0, tzinfo=dt.UTC)


def _seed(db: Db) -> dict[str, str]:
    missions: dict[str, str] = {}
    for vid, (sysid, _) in VEHICLES.items():
        db.run(
            "INSERT INTO vehicles (id, name, kind, mav_sysid) VALUES ($1, $1, 'sim', $2)",
            vid,
            sysid,
        )
        missions[vid] = db.run(
            "INSERT INTO missions (vehicle_id, title, limits, created_by, data_dir) "
            "VALUES ($1, 'Test', '{}', 'operator:demo', $2) RETURNING id::text",
            vid,
            f"missions/{vid}",
        )[0]["id"]
    return missions


def _rows(vid: str, mission_id: str, batch: int) -> list[TelemetryRow]:
    _, east = VEHICLES[vid]
    rows: list[TelemetryRow] = []
    for i in range(BATCH):
        n = batch * BATCH + i
        rows.append(
            {
                "vehicle_id": vid,
                "mission_id": mission_id,
                "t": T0 + dt.timedelta(milliseconds=500 * n),
                "lat": HOME_LAT + n * 1e-6,
                "lon": HOME_LON + east,
                "alt_amsl_m": 600.0,
                "alt_rel_home_m": 16.0,
                "heading_deg": 0.0,
                "groundspeed_ms": 5.0,
                "climb_ms": 0.0,
                "battery_pct": 90.0,
                "battery_v": 16.0,
                "flight_mode": "GUIDED",
                "armed": True,
                "landed_state": "IN_AIR",
                "gps_fix": 3,
                "sats": 12,
                "executor_state": None,
            }
        )
    return rows


async def _write(repo: Repo, vid: str, mission_id: str, barrier: asyncio.Barrier) -> None:
    for b in range(BATCHES):
        await barrier.wait()  # both vehicles write batch b at the same time
        await repo.add_telemetry(_rows(vid, mission_id, b))


def test_two_vehicles_write_telemetry_at_once(migrated_db: Db) -> None:
    missions = _seed(migrated_db)

    async def run() -> None:
        repo = Repo(make_engine(app_url(migrated_db.name, migrated_db.app_password)))
        barrier = asyncio.Barrier(len(missions))
        try:
            await asyncio.gather(
                *(_write(repo, vid, mid, barrier) for vid, mid in missions.items())
            )
        finally:
            await repo.close()

    asyncio.run(run())

    rows = migrated_db.run(
        "SELECT t.vehicle_id, t.mission_id::text AS mission_id, m.vehicle_id AS owner, "
        "ST_X(t.pos) AS lon, ST_Y(t.pos) AS lat "
        "FROM telemetry t JOIN missions m ON m.id = t.mission_id ORDER BY t.id"
    )
    assert len(rows) == len(VEHICLES) * BATCHES * BATCH
    first_round = [r["vehicle_id"] for r in rows[: len(VEHICLES) * BATCH]]
    assert set(first_round) == set(VEHICLES), "the writers did not run side by side"
    for vid, mission_id in missions.items():
        own = [r for r in rows if r["vehicle_id"] == vid]
        assert len(own) == BATCHES * BATCH
        assert {r["mission_id"] for r in own} == {mission_id}
        assert {r["owner"] for r in own} == {vid}
        expected_lon = HOME_LON + VEHICLES[vid][1]
        assert all(r["lon"] == pytest.approx(expected_lon, abs=1e-9) for r in own)
        lats = [r["lat"] for r in own]
        assert lats == sorted(lats), "a vehicle's own track came back out of order"
