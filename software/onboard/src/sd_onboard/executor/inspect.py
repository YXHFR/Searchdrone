"""A closer look as the executor sees it (``sd_schemas.inspect``, spec "Titta närmare").

The handler (``skills.inspect_detection``) flies the look and takes the stills. The rest
belongs to the executor and lives in one ``InspectSession`` from the order's acceptance
until the drone is back where the search broke off (or the look is ended):

* ``flying``: the handler runs (state ``INSPECT``).
* ``await``: the steps are over; ``HOLD`` with ``hold_reason`` "inspect_await" for at
  most ``AWAIT_DECISION_S`` (the pilot's ``InspectOutcome``).
* ``climb``: the laser lost while waiting, below the map's floor without it: straight up
  to that floor (``INSPECT``), then the wait again.
* ``return_climb``: the return's first part: straight up to the height the look started
  at (``INSPECT``).

The session ends when the return's second part starts: a ``resume`` order the executor
issues itself (``rule:onboard``) to the breakpoint (``TRANSIT``/``SURVEY``), checked by
the whole gate first. It ends earlier on the pilot's "confirmed" (the drone hovers over
the find), another order, or anything that takes the drone away (``abort_reason``).

The candidate's own cylinder is suspended (``suspends``) from the acceptance until the
return starts, only when the charter allows flying over it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal
from uuid import UUID

from sd_onboard.skills.inspect_steps import Floors
from sd_safety.targets import Plane
from sd_schemas.base import LatLon
from sd_schemas.executor import ExecutorState, ResumePoint
from sd_schemas.inspect import InspectAbortReason, InspectOutcomeKind
from sd_schemas.keepout import Keepout
from sd_schemas.link import SkillCommand

__all__ = [
    "INTERNAL_INSPECT_CLIMB",
    "SUSPENDED_PHASES",
    "FindWatch",
    "InspectSession",
    "SessionPhase",
    "abort_reason",
]

S = ExecutorState
# The straight climbs of a closer look: the return's first part, the laser lost while
# waiting, and the SD-link lost during a look (then without a session).
INTERNAL_INSPECT_CLIMB: Final = "internal_inspect_climb"

SessionPhase = Literal["flying", "await", "climb", "return_climb"]
_HELD: Final[dict[str, InspectAbortReason]] = {
    "operator_stop": "stop",
    "operator_home": "home",
    "system_battery": "battery",
}
# The candidate's cylinder is suspended in these phases (the return brings it back).
SUSPENDED_PHASES: Final = frozenset({"flying", "await", "climb"})


@dataclass
class InspectSession:
    command_id: UUID
    candidate_id: UUID
    cmd: SkillCommand
    overflight: bool  # the charter allows flying over the candidate (the step-wise look)
    anchor: LatLon  # the order's position of the candidate
    radius_m: float  # the look radius around ``anchor``
    breakpoint: ResumePoint  # where the search broke off: the return flies here
    saved_resume: ResumePoint | None  # the resume point before the look, put back after
    start_alt_m: float  # above home, when the look started
    phase: SessionPhase = "flying"
    done: bool = False  # ``inspect.done`` sent
    await_until_ms: int | None = None
    laser_seen: bool = False  # the laser confirmed during this look
    steps: int = 0  # steps down taken (for the progress after the handler ended)
    floors: Floors | None = None  # the floors at the last check
    # The pilot's decision while the drone climbed (back to the map's floor without the
    # laser, or over someone else's cylinder): carried out once the climb is done.
    outcome: InspectOutcomeKind | None = None

    def suspends(self, k: Keepout, plane: Plane, *, local: bool) -> bool:
        """Whether ``k`` is the candidate's own cylinder, suspended now: the ground's
        cylinder for the candidate, and the drone's own cylinders centred within the look
        radius of the candidate (its own detections of the target)."""
        if not self.overflight or self.phase not in SUSPENDED_PHASES:
            return False
        if not local:
            return k.source_id == self.candidate_id
        return self.near(k.lat, k.lon, plane)

    def suspends_near(self, lat: float, lon: float, plane: Plane) -> bool:
        """A detection here is the target during its look: it makes no new cylinder."""
        return self.overflight and self.phase in SUSPENDED_PHASES and self.near(lat, lon, plane)

    def near(self, lat: float, lon: float, plane: Plane) -> bool:
        """Within the look radius of the candidate's position."""
        return math.dist(plane.xy(lat, lon), plane.point(self.anchor)) <= self.radius_m


@dataclass
class FindWatch:
    """The laser over a confirmed find, after the look's session ended: the drone may
    hover there below the map's floor without the laser (it came down with it). The
    laser lost there, it climbs straight up to that floor (``Executor._find_watch_tick``);
    the watch ends when the drone leaves the hover."""

    command_id: UUID
    start_alt_m: float  # above home, where the look started
    laser_seen: bool  # the laser confirmed during the look


def abort_reason(
    state: ExecutorState, held_by: str | None, hold_reason: str | None, taken_by: str | None
) -> InspectAbortReason:
    """Why a look ended without the wait, from where the executor went (``inspect.aborted``)."""
    if taken_by == "gnss":
        return "gnss"
    if state == S.HELD:
        return _HELD.get(held_by or "", "override")
    if state == S.RTL:
        return "battery" if taken_by == "battery" else "home"
    if state == S.HOLD:
        return "link_loss" if hold_reason == "link_loss" else "override"
    return "failed"
