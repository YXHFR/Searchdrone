"""Telemetry: 5 Hz to the panel over WebSocket, 2 Hz into the database.

The vehicle state (``vehicle.state``: executor state, link, latch, airborne) goes to the
panel at 2 Hz as well; the panel's Resume and "clear around the drone" depend on it.

Samples are stored while a vehicle's open mission is flying (``ACTIVE`` up to closing) and
written in batches of 10 rows (one transaction about every 5 s). If the database does not
answer, up to 3 600 rows (30 min) are kept in memory; beyond that the oldest are dropped and
``telemetry.dropped`` is logged once the database is back. Full rate is always in the
flight's ``flight.tlog``.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from sd_ground.context import Ground
from sd_ground.db.repo import TelemetryRow
from sd_ground.logging import get_logger
from sd_ground.services.health import vehicle_state
from sd_schemas.api import Telemetry
from sd_schemas.mission import MissionState

__all__ = ["BATCH", "BUFFER_MAX", "PUBLISH_S", "RECORDING", "SAMPLE_S", "TelemetryService"]

PUBLISH_S = 0.2  # 5 Hz to the panel
SAMPLE_S = 0.5  # 2 Hz to the database
BATCH = 10
BUFFER_MAX = 3600
MISSIONS_REFRESH_S = 1.0
RECORDING = frozenset(
    {
        MissionState.ACTIVE,
        MissionState.PAUSED,
        MissionState.HELD,
        MissionState.FAILSAFE,
        MissionState.RETURNING,
        MissionState.LANDED,
    }
)
log = get_logger("sd_ground.telemetry")


def _row(vehicle_id: str, mission_id: str, t: Telemetry) -> TelemetryRow:
    return {
        "vehicle_id": vehicle_id,
        "mission_id": mission_id,
        "t": t.t,
        "lat": t.lat,
        "lon": t.lon,
        "alt_amsl_m": t.alt_amsl_m,
        "alt_rel_home_m": t.alt_rel_home_m,
        "heading_deg": t.heading_deg,
        "groundspeed_ms": t.groundspeed_ms,
        "climb_ms": t.climb_ms,
        "battery_pct": t.battery_pct,
        "battery_v": t.battery_v,
        "flight_mode": t.flight_mode,
        "armed": t.armed,
        "landed_state": t.landed_state,
        "gps_fix": t.gps_fix,
        "sats": t.gps_sats,
        "executor_state": None,
    }


class TelemetryService:
    def __init__(self, g: Ground) -> None:
        self.g = g
        self.buffer: deque[TelemetryRow] = deque()
        self.dropped = 0
        self._recording: dict[str, str] = {}  # vehicle_id -> mission_id
        self._vehicles: list[str] = [g.config.default_vehicle_id]

    async def refresh_missions(self) -> None:
        repo = self.g.repo
        if repo is None or not self.g.db_ok:
            return
        try:
            missions = await repo.open_missions()
            vehicles = [v["id"] for v in await repo.vehicles()]
        except Exception as exc:
            log.warning("telemetry.missions_failed", error=type(exc).__name__)
            return
        self._recording = {m.vehicle_id: str(m.id) for m in missions if m.status in RECORDING}
        self._vehicles = vehicles or [self.g.config.default_vehicle_id]

    def publish(self) -> None:
        for vid in self._vehicles:
            t = self.g.vehicle_link.telemetry(vid)
            if t is not None:
                self.g.bus.publish(
                    "telemetry",
                    "telemetry",
                    t.model_dump(mode="json"),
                    ts=self.g.now(),
                    vehicle_id=vid,
                    mission_id=self._recording.get(vid),
                )

    def publish_state(self) -> None:
        """``vehicle.state`` for each vehicle. A failure skips one sample and never stops
        the telemetry loop (5 Hz telemetry and the recording go on)."""
        actions = self.g.vehicle_actions
        for vid in self._vehicles:
            try:
                latch = None if actions is None else actions.latches.get(vid)
                live = vehicle_state(self.g, vid, latch)
            except Exception as exc:  # one sample is lost; the loop lives on
                log.error("telemetry.state_failed", vehicle_id=vid, error=type(exc).__name__)
                continue
            self.g.bus.publish(
                "vehicle",
                "vehicle.state",
                live.model_dump(mode="json"),
                ts=self.g.now(),
                vehicle_id=vid,
                mission_id=self._recording.get(vid),
            )

    def sample(self) -> None:
        for vid, mission_id in self._recording.items():
            t = self.g.vehicle_link.telemetry(vid)
            if t is None:
                continue
            self.buffer.append(_row(vid, mission_id, t))
            if len(self.buffer) > BUFFER_MAX:
                self.buffer.popleft()
                self.dropped += 1

    async def flush(self, *, force: bool = False) -> None:
        repo = self.g.repo
        if repo is None or not self.buffer or (len(self.buffer) < BATCH and not force):
            return
        batch: list[dict[str, Any]] = list(self.buffer)
        try:
            await repo.add_telemetry(batch)
        except Exception as exc:
            log.warning("telemetry.write_failed", error=type(exc).__name__, buffered=len(batch))
            return
        for _ in batch:
            self.buffer.popleft()
        if self.dropped:
            self.g.add_event(
                "telemetry.dropped", severity="warning", payload={"rows": self.dropped}
            )
            self.dropped = 0

    async def run(self) -> None:
        clock = self.g.clock
        now = clock.monotonic()
        next_publish, next_sample, next_refresh = now, now, now
        while True:
            now = clock.monotonic()
            if now >= next_refresh:
                await self.refresh_missions()
                next_refresh = now + MISSIONS_REFRESH_S
            if now >= next_publish:
                self.publish()
                next_publish += PUBLISH_S
                if next_publish < now:
                    next_publish = now + PUBLISH_S
            if now >= next_sample:
                self.publish_state()
                self.sample()
                await self.flush()
                next_sample += SAMPLE_S
                if next_sample < now:
                    next_sample = now + SAMPLE_S
            await clock.sleep(
                max(0.0, min(next_publish, next_sample, next_refresh) - clock.monotonic())
            )
