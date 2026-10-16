"""The stable handler interface: every onboard skill is one ``SkillHandler``.

A handler plans (a pure function the onboard gate checks), then runs its procedure
through ``ExecutorIO``, the only way from a handler to MAVLink. Every write goes through
the executor's compare-and-set (§ "who steers") inside ``sd_mav.Vehicle.send``; when the
executor is not in control the write raises ``ControlNotAllowedError`` and nothing is sent.

Stage 2 and 3 skills build on these types. New methods may be added; existing ones are
not changed without a new minor version.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol

from pydantic import Field

from sd_mav.mission import MissionItem
from sd_mav.modes import CopterMode
from sd_mav.vehicle import MavResult
from sd_obs.clock import Clock
from sd_safety.targets import Plane
from sd_schemas.base import Id, LatLonAlt, SdModel
from sd_schemas.charter import MissionCharter
from sd_schemas.executor import ExecutorState, ResumePoint
from sd_schemas.skills import SkillParams
from sd_schemas.vehicle import VehicleState

__all__ = [
    "ExecutorIO",
    "MavAction",
    "PlanContext",
    "PlanError",
    "SkillFailedError",
    "SkillHandler",
    "SkillPlan",
    "SkillStatus",
    "TakeoffTimeoutError",
    "TickContext",
    "mode_failed",
]


class MavAction(SdModel):
    """One planned MAVLink step, for the log and the tests."""

    kind: str  # "set_mode", "arm", "takeoff", "set_position_target", "change_speed", ...
    params: dict[str, Any] = Field(default_factory=dict[str, Any])


class SkillPlan(SdModel):
    path: list[LatLonAlt]  # the route the gate checks: [current position, ..., end]; [] = no move
    target_state: ExecutorState
    speed_ms: float | None
    expected_duration_s: float
    mav_actions: list[MavAction]


class SkillStatus(SdModel):
    stage: Literal["running", "done", "failed"]
    progress_pct: float | None = None
    detail_sv: str = ""
    reason_code: str | None = None


@dataclass(frozen=True, slots=True)
class PlanContext:
    command_id: Id
    vehicle: VehicleState
    charter: MissionCharter
    plane: Plane  # east/north metres around the charter's home
    resume_point: ResumePoint | None
    clock: Clock

    def alt_amsl(self, alt_rel_home_m: float) -> float:
        return self.charter.home.alt_amsl_m + alt_rel_home_m

    def here(self, alt_rel_home_m: float | None = None) -> LatLonAlt:
        v = self.vehicle
        alt = v.alt_rel_home_m if alt_rel_home_m is None else alt_rel_home_m
        return LatLonAlt(lat=v.lat, lon=v.lon, alt_amsl_m=self.alt_amsl(alt))


@dataclass(frozen=True, slots=True)
class TickContext:
    vehicle: VehicleState
    clock: Clock
    now_mono_ms: int


class PlanError(Exception):
    """The order cannot be planned; ``code`` is the reason code for the Ack."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class SkillFailedError(Exception):
    """The procedure failed; the executor fires SKILL_FAILED with ``code``."""

    def __init__(self, code: str, detail_sv: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.detail_sv = detail_sv


def mode_failed(mode: CopterMode) -> SkillFailedError:
    return SkillFailedError(
        "ONB_MODE_CHANGE_TIMEOUT", f"Flygkontrollern bytte inte till {mode.name}."
    )


class TakeoffTimeoutError(SkillFailedError):
    """The takeoff did not climb in time (TAKEOFF_TIMEOUT)."""

    def __init__(self, detail_sv: str) -> None:
        super().__init__("ONB_TAKEOFF_TIMEOUT", detail_sv)


class ExecutorIO(Protocol):
    """The only way from a handler to the flight controller and the ground.

    Every method goes through the executor's compare-and-set; when the executor does not
    steer, ``ControlNotAllowedError`` is raised and nothing is sent. There is no method
    for BRAKE, LAND, disarming or parameters.
    """

    clock: Clock

    @property
    def vehicle(self) -> VehicleState: ...  # the latest state (read only)

    async def set_mode(self, mode: CopterMode) -> bool: ...  # waits for the HEARTBEAT
    async def arm(self) -> MavResult: ...
    async def takeoff(self, alt_m: float) -> MavResult: ...
    async def set_position_target(self, lat: float, lon: float, alt_m: float) -> None: ...
    async def change_speed(self, speed_ms: float) -> MavResult: ...
    async def upload_mission(self, items: list[MissionItem]) -> str: ...  # read-back hash
    async def set_mission_current(self, seq: int) -> MavResult: ...
    async def wait_message(self, predicate: Callable[[Any], bool], within_s: float) -> Any: ...
    async def sleep(self, seconds: float) -> None: ...
    def last_statustext(self) -> str: ...
    def emit_event(self, type: str, data: dict[str, Any]) -> None: ...


class SkillHandler(Protocol):
    name: ClassVar[str]
    TRANSITIONS_EXTRA: ClassVar[dict[Any, Any]]

    def plan(self, params: SkillParams, ctx: PlanContext) -> SkillPlan: ...  # pure
    async def start(self, plan: SkillPlan, io: ExecutorIO) -> None: ...  # runs until done
    def on_tick(self, ctx: TickContext) -> SkillStatus: ...  # 10 Hz, progress only
    async def cancel(self, io: ExecutorIO) -> None: ...  # leaves the vehicle in LOITER
    def resume_point(self, vehicle: VehicleState) -> ResumePoint | None: ...
