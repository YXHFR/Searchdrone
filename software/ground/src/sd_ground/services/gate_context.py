"""The ground's picture for the gate: ``GateContext`` built from what the ground knows.

Every order from the panel or the AI is checked by ``sd_safety.evaluate`` against this
picture (the ground's gate is authoritative; the onboard computer checks again and can
only refuse). Inputs: the mission and its active charter, the ground's own MAVLink
(armed, in the air, battery, position), the latest onboard ``StateSnapshot`` (executor
state, preflight, the state numbers the AI's ``based_on`` is compared with), the latch,
the "clear around the drone" confirmation, the AI's recent orders (rate limit), its
waiting cards and whether a pilot is present.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

from sd_ground.mav.thread import MavSnapshot
from sd_safety.constants import RATE_WINDOW_MS
from sd_schemas.api import Mission
from sd_schemas.base import LatLon
from sd_schemas.charter import MissionCharter
from sd_schemas.gate import ClearanceState, GateContext
from sd_schemas.link import StateSnapshot
from sd_schemas.mission import OperatorLatch
from sd_schemas.skills import SkillSpec

__all__ = ["BrainRate", "build_gate_context"]

_KEEP: Final = 64


@dataclass
class BrainRate:
    """The AI's orders that got ALLOW or PENDING_APPROVAL, per kind (monotonic ms)."""

    actions: deque[int] = field(default_factory=lambda: deque[int](maxlen=_KEEP))
    reports: deque[int] = field(default_factory=lambda: deque[int](maxlen=_KEEP))

    def note(self, rate_class: str, now_ms: int) -> None:
        if rate_class == "action":
            self.actions.append(now_ms)
        elif rate_class == "report":
            self.reports.append(now_ms)

    def recent(self, now_ms: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        lo = now_ms - RATE_WINDOW_MS
        return (
            tuple(t for t in self.actions if t > lo),
            tuple(t for t in self.reports if t > lo),
        )


def build_gate_context(
    *,
    now_ms: int,
    registry: Mapping[str, SkillSpec],
    mission: Mission,
    charter: MissionCharter | None,
    charter_installed: bool,
    fence_verified: bool,
    mav: MavSnapshot | None,
    snapshot: StateSnapshot | None,
    snapshot_received_ms: int | None,
    sdlink_up: bool,
    latch: OperatorLatch | None,
    clearance: ClearanceState | None,
    rate: BrainRate,
    brain_pending_approvals: int,
    operator_present: bool,
) -> GateContext:
    home = charter.home if charter is not None else None
    if charter is not None:
        pilot = charter.pilot_position
    elif mav is not None and mav.telemetry.home is not None:
        pilot = LatLon(lat=mav.telemetry.home.lat, lon=mav.telemetry.home.lon)
    else:
        pilot = LatLon(lat=0.0, lon=0.0)
    vehicle = None
    vehicle_age = None
    if mav is not None and mav.connected:
        vehicle = mav.telemetry.to_vehicle_state(mission.vehicle_id, now_ms, pilot)
        stamp = mav.telemetry.position_mono_ms or mav.telemetry.fc_heartbeat_mono_ms
        vehicle_age = None if stamp is None else max(0, now_ms - stamp)
        if home is None and vehicle.home is not None:
            home = vehicle.home
    snapshot_age = None if snapshot_received_ms is None else max(0, now_ms - snapshot_received_ms)
    actions, reports = rate.recent(now_ms)
    manual = (mav is not None and mav.manual) or (snapshot is not None and snapshot.manual_control)
    return GateContext(
        side="ground",
        now_mono_ms=now_ms,
        registry=registry,
        limits=mission.limits,
        charter=charter,
        charter_installed=charter_installed,
        fence_verified=fence_verified,
        autonomy_mode=None if charter is None else charter.autonomy_mode,
        mission_state=mission.status,
        paused_by=mission.paused_by,
        returning_cause=mission.returning_cause,
        ai_enabled=mission.ai_enabled,
        vehicle=vehicle,
        vehicle_age_ms=vehicle_age,
        snapshot_age_ms=snapshot_age,
        executor_state=None if snapshot is None else snapshot.executor_state,
        state_epoch=None if snapshot is None else snapshot.state_epoch,
        last_transition_state_seq=None if snapshot is None else snapshot.last_transition_state_seq,
        preflight_ok=None if snapshot is None else snapshot.preflight_ok,
        sdlink_up=sdlink_up,
        stop_latch=latch is not None,
        latch=latch,
        manual_control=manual,
        clearance=clearance,
        recent_brain_actions_mono_ms=actions,
        recent_brain_reports_mono_ms=reports,
        brain_pending_approvals=brain_pending_approvals,
        operator_present=operator_present,
    )
