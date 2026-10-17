"""The invariants checked after every scenario, on a recorded timeline.

A scenario test records what happened (``Timeline``): the vehicle's track, every MAVLink
message the three senders wrote to the flight controller, the ground station's audit
events, its orders and gate decisions, and the charters in force. ``check_all`` judges
it afterwards; a broken invariant fails the test even when the scenario's own criteria
held. Exceptions are named per scenario (``exempt``), never guessed here.

The senders: ``ground`` (sd-ground, 250/190), ``onboard`` (sd-onboard, 1/191) and
``harness`` (the test harness, 253/190, fault injection only).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from sd_safety.targets import Plane, distance_m, polygon_from_geojson
from sd_schemas.api import CommandStatus

__all__ = [
    "ALL",
    "Charter",
    "Result",
    "Sample",
    "Sent",
    "Timeline",
    "check_all",
    "failed",
]

ALL = tuple(f"INV{i}" for i in range(1, 12))

# Copter modes (custom_mode) the invariants name.
GUIDED, AUTO, LOITER, RTL, LAND, BRAKE, SMART_RTL = 4, 3, 5, 6, 9, 17, 21
# Modes sd-ground may command: STOP's and FLY HOME's steps.
GROUND_MODES = frozenset({BRAKE, LOITER, RTL, LAND, SMART_RTL})
# Modes a failsafe or the operator's STOP/FLY HOME leaves the vehicle in.
HOLDING_MODES = frozenset({BRAKE, LOITER, RTL, LAND, SMART_RTL})
CMD_DO_SET_MODE = 176
CMD_ARM_DISARM = 400
CMD_SET_MESSAGE_INTERVAL = 511
CMD_REQUEST_MESSAGE = 512
CMD_DO_SET_PARAMETER = 180
CMD_DO_FENCE_ENABLE = 207
CMD_BATTERY_RESET = 42651
FORCE_MAGIC = 21196
MISSION_FENCE = 1
# Reading is allowed for everyone.
READ_TYPES = frozenset(
    {
        "HEARTBEAT",
        "PARAM_REQUEST_READ",
        "PARAM_REQUEST_LIST",
        "MISSION_REQUEST_LIST",
        "MISSION_REQUEST_INT",
        "MISSION_REQUEST",
        "MISSION_ACK",
        "REQUEST_DATA_STREAM",
    }
)
# Parameters nobody may write, ever.
NEVER_WRITTEN_PREFIXES = ("FENCE_ENABLE", "FS_", "MAV_GCS_SYSID", "MAV_OPTIONS", "ARMING_")
# The ground moves only the fence's size and LOITER's speed with the pilot's raised
# limits, and only on the ground.
GROUND_PARAMS = frozenset({"FENCE_ALT_MAX", "FENCE_RADIUS", "LOIT_SPEED_MS"})
STEERING_TYPES = frozenset(
    {"SET_POSITION_TARGET_GLOBAL_INT", "SET_POSITION_TARGET_LOCAL_NED", "MISSION_ITEM_INT"}
)
# Operator events that release a STOP or a FLY HOME (INV10).
RELEASING_EVENTS = frozenset({"vehicle.resume", "vehicle.return_home", "vehicle.stop"})
# A fallback mode after a refused mode change is allowed this long (s).
FALLBACK_WINDOW_S = 3.0
# An order already on its way when a hold starts may still arrive this long after (s).
STEER_GRACE_S = 1.0
# The ground logs an operator's action after it has sent its mode change (the audit
# event follows the MAVLink path), at most this much later (s).
EVENT_LAG_S = 2.0
ON_GROUND_ALT_M = 0.5


@dataclass(frozen=True)
class Sample:
    """One telemetry sample of the vehicle (host monotonic time ``t``)."""

    t: float
    lat: float
    lon: float
    alt_m: float  # above home
    groundspeed_ms: float
    mode: int | None
    armed: bool
    landed: bool = False  # EXTENDED_SYS_STATE says on the ground


@dataclass(frozen=True)
class Sent:
    """One MAVLink message a sender wrote to the flight controller."""

    t: float
    sender: str  # "ground", "onboard" or "harness"
    type: str
    command: int | None = None
    params: tuple[float, ...] = ()
    mission_type: int | None = None
    param_id: str | None = None
    alt_m: float = 0.0  # the vehicle's height above home when it was sent
    armed: bool = False
    landed: bool = False

    @property
    def mode(self) -> int | None:
        if self.command == CMD_DO_SET_MODE and len(self.params) >= 2:
            return int(self.params[1])
        return None

    def steers(self) -> bool:
        if self.type in STEERING_TYPES:
            return self.type != "MISSION_ITEM_INT" or self.mission_type in (None, 0)
        return self.type == "MISSION_COUNT" and self.mission_type in (None, 0)


@dataclass(frozen=True)
class Charter:
    """A charter version in force from ``since`` (host monotonic time)."""

    since: float
    area: dict[str, Any]
    pilot: tuple[float, float]
    alt_max_m: float
    speed_max_ms: float
    range_m: float
    buffer_m: float
    autonomy_mode: str
    approved_by: str | None
    status: str = "approved"
    install_status: str = "installed"


@dataclass
class Timeline:
    samples: list[Sample] = field(default_factory=list[Sample])
    sent: list[Sent] = field(default_factory=list[Sent])
    events: list[tuple[float, str, dict[str, Any]]] = field(
        default_factory=list[tuple[float, str, dict[str, Any]]]
    )
    charters: list[Charter] = field(default_factory=list[Charter])
    commands: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    gates: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    clearances: list[float] = field(default_factory=list[float])  # confirmation times
    accepted: dict[str, float] = field(default_factory=dict[str, float])  # command -> time
    inside_m: float | None = None  # INV4 limit when it is stricter than the buffer (SC01: 0)


@dataclass(frozen=True)
class Result:
    id: str
    passed: bool
    measured: float | None = None
    limit: float | None = None
    detail: str = ""


def failed(results: Iterable[Result]) -> list[Result]:
    return [r for r in results if not r.passed]


def _charter_at(tl: Timeline, t: float) -> Charter | None:
    current = None
    for c in sorted(tl.charters, key=lambda c: c.since):
        if c.since <= t and c.status in ("approved", "superseded"):
            current = c
    return current


def _airborne(tl: Timeline) -> list[Sample]:
    return [s for s in tl.samples if s.armed]


def _worst(id_: str, pairs: Iterable[tuple[float, float, float]]) -> Result:
    """``(value, limit, t)`` per sample: passes when no value is over its limit; reports
    the sample with the largest excess."""
    worst: tuple[float, float, float] | None = None
    for value, limit, t in pairs:
        if worst is None or value - limit > worst[0] - worst[1]:
            worst = (value, limit, t)
    if worst is None:
        return Result(id_, True, detail="inga prover i luften")
    value, limit, t = worst
    return Result(id_, value <= limit, round(value, 2), round(limit, 2), f"t={t:.1f}")


def _in_force(tl: Timeline) -> Iterable[tuple[Sample, Charter]]:
    for s in _airborne(tl):
        c = _charter_at(tl, s.t)
        if c is not None:
            yield s, c


def inv1(tl: Timeline) -> Result:
    return _worst("INV1", ((s.alt_m, c.alt_max_m + 2.0, s.t) for s, c in _in_force(tl)))


def inv2(tl: Timeline) -> Result:
    return _worst(
        "INV2",
        (
            (distance_m(c.pilot[0], c.pilot[1], s.lat, s.lon), c.range_m + 5.0, s.t)
            for s, c in _in_force(tl)
        ),
    )


def inv3(tl: Timeline) -> Result:
    return _worst("INV3", ((s.groundspeed_ms, c.speed_max_ms + 1.0, s.t) for s, c in _in_force(tl)))


def _outside_m(c: Charter, s: Sample) -> float:
    plane = Plane(c.pilot[0], c.pilot[1])
    poly = polygon_from_geojson(c.area, plane)
    xy = plane.xy(s.lat, s.lon)
    return 0.0 if poly.contains(xy) else poly.boundary_distance(xy)


def inv4(tl: Timeline) -> Result:
    return _worst(
        "INV4",
        (
            (_outside_m(c, s), c.buffer_m if tl.inside_m is None else tl.inside_m, s.t)
            for s, c in _in_force(tl)
        ),
    )


def _written_param(m: Sent) -> str | None:
    if m.type in ("PARAM_SET", "PARAM_EXT_SET"):
        return m.param_id or "?"
    return None


def inv5(tl: Timeline, harness_extra: Sequence[str] = ()) -> Result:
    """Who may write what to the flight controller (07 INV5)."""
    bad: list[str] = []
    for m in tl.sent:
        if m.type in READ_TYPES:
            continue
        name = _written_param(m)
        if name is not None and name.startswith(NEVER_WRITTEN_PREFIXES):
            bad.append(f"{m.sender} skrev {name}")
            continue
        disarm = m.command == CMD_ARM_DISARM and bool(m.params) and m.params[0] == 0
        if disarm and not m.landed and m.alt_m >= ON_GROUND_ALT_M:
            bad.append(f"{m.sender} desarmerade i luften ({m.alt_m:.1f} m)")
            continue
        if m.sender == "ground":
            bad += _ground_problems(m)
        elif m.sender == "onboard":
            bad += _onboard_problems(m)
        elif m.sender == "harness":
            bad += _harness_problems(m, harness_extra)
        else:
            bad.append(f"okänd avsändare {m.sender}: {m.type}")
    return Result("INV5", not bad, len(bad), 0, "; ".join(bad[:5]))


def _ground_problems(m: Sent) -> list[str]:
    if m.type == "COMMAND_LONG":
        if m.command in (CMD_SET_MESSAGE_INTERVAL, CMD_REQUEST_MESSAGE):
            return []
        if m.command == CMD_DO_SET_MODE and m.mode in GROUND_MODES:
            return []
        disarm = m.command == CMD_ARM_DISARM and bool(m.params) and m.params[0] == 0
        if disarm and (len(m.params) < 2 or m.params[1] != FORCE_MAGIC):
            return []
        return [f"ground: COMMAND_LONG {m.command} {m.params[:2]}"]
    if m.type in ("MISSION_COUNT", "MISSION_ITEM_INT", "MISSION_CLEAR_ALL"):
        if m.mission_type == MISSION_FENCE and not m.armed:
            return []
        return [f"ground: {m.type} typ {m.mission_type} (armerad {m.armed})"]
    if m.type == "PARAM_SET" and m.param_id in GROUND_PARAMS and not m.armed:
        return []
    if m.type == "MANUAL_CONTROL":
        return []  # the pilot's sticks (LOITER only; checked by sd_mav's allowlist)
    return [f"ground: {m.type}"]


def _onboard_problems(m: Sent) -> list[str]:
    if m.type in ("PARAM_SET", "PARAM_EXT_SET"):
        return [f"onboard: {m.type} {m.param_id}"]
    if m.command in (CMD_DO_SET_PARAMETER, CMD_DO_FENCE_ENABLE):
        return [f"onboard: kommando {m.command}"]
    if m.command == CMD_ARM_DISARM and len(m.params) >= 2 and m.params[1] == FORCE_MAGIC:
        return ["onboard: tvingad armering/desarmering"]
    if m.type in ("MISSION_COUNT", "MISSION_ITEM_INT", "MISSION_CLEAR_ALL") and m.mission_type:
        return [f"onboard: {m.type} med mission_type {m.mission_type}"]
    return []


def _harness_problems(m: Sent, extra: Sequence[str]) -> list[str]:
    if m.type == "PARAM_SET" and (m.param_id or "").startswith("SIM_"):
        return []
    if m.command == CMD_BATTERY_RESET:
        return []
    if m.type in extra or (m.command is not None and f"cmd:{m.command}" in extra):
        return []
    return [f"harness: {m.type} {m.command or ''}".strip()]


def inv6(tl: Timeline) -> Result:
    """Every ARM from the onboard computer follows an approved, installed charter and its
    own fresh clearance (used once)."""
    bad: list[str] = []
    used: set[float] = set()
    for m in tl.sent:
        if m.sender != "onboard" or m.command != CMD_ARM_DISARM:
            continue
        if not m.params or m.params[0] != 1:
            continue
        c = _charter_at(tl, m.t)
        if c is None or c.install_status != "installed":
            bad.append(f"armering t={m.t:.1f} utan godkänd och installerad ram")
            continue
        fresh = [t for t in tl.clearances if t <= m.t and m.t - t <= 120.0 and t not in used]
        if not fresh:
            bad.append(f"armering t={m.t:.1f} utan egen bekräftelse ≤ 120 s")
            continue
        used.add(max(fresh))
    return Result("INV6", not bad, len(bad), 0, "; ".join(bad))


GROUND_SKILLS = frozenset({"report", "mark_location", "propose_charter_change"})


def inv7(tl: Timeline, allow_mismatch: bool = False) -> Result:
    """Every order that ran has a ground ALLOW before the onboard computer accepted it,
    and the onboard computer agreed (no ``gate.mismatch``)."""
    bad: list[str] = []
    for cmd in tl.commands:
        if cmd["skill"] in GROUND_SKILLS:
            continue
        accepted = tl.accepted.get(str(cmd["id"]))
        if accepted is None:
            continue
        rows = [g for g in tl.gates if str(g.get("command_id")) == str(cmd["id"])]
        if not any(g["side"] == "ground" and g["decision"] == "ALLOW" for g in rows):
            bad.append(f"{cmd['skill']} {cmd['id']}: inget ALLOW på marken")
    if not allow_mismatch and any(e[1] == "gate.mismatch" for e in tl.events):
        bad.append("gate.mismatch")
    return Result("INV7", not bad, len(bad), 0, "; ".join(bad[:5]))


def inv8(tl: Timeline) -> Result:
    """STOP and FLY HOME never go through the AI."""
    bad = [
        f"{kind} av {payload.get('by')}"
        for _, kind, payload in tl.events
        if kind in ("vehicle.stop", "vehicle.return_home")
        and payload.get("via") != "system"
        and not str(payload.get("by", "")).startswith("operator:")
    ]
    return Result("INV8", not bad, len(bad), 0, "; ".join(bad))


def inv9(tl: Timeline) -> Result:
    """Persistence: every order the API showed was stored (the API reads the database),
    and its status is a known one."""
    known = {s.value for s in CommandStatus}
    bad = [f"{c['id']}: {c['status']}" for c in tl.commands if c["status"] not in known]
    return Result("INV9", not bad, len(bad), 0, "; ".join(bad))


def inv10(tl: Timeline) -> Result:
    """Nothing fights a failsafe or the operator: after the vehicle entered a holding mode
    that none of our senders asked for (a failsafe), or after the operator's STOP or FLY
    HOME, nobody steers or changes to another mode until the operator releases it. The
    same mode may be sent again; a fallback step (BRAKE -> LOITER -> LAND, SMART_RTL ->
    RTL -> LAND) is allowed within ``FALLBACK_WINDOW_S`` after a hold starts."""
    bad: list[str] = []
    for start, mode, cause, index in _holds(tl):
        release = next(
            (
                t - EVENT_LAG_S
                for n, (t, kind, p) in enumerate(tl.events)
                if t >= start and n != index and _releases(kind, p)
            ),
            float("inf"),
        )
        for m in tl.sent:
            if not start + STEER_GRACE_S < m.t < release or m.sender == "harness":
                continue
            if m.steers():
                bad.append(f"{m.sender} styrde t={m.t:.1f} efter {cause}")
            elif m.mode is not None and m.mode != mode:
                fallback = m.mode in _fallbacks(mode) and m.t - start <= FALLBACK_WINDOW_S
                if not fallback:
                    bad.append(f"{m.sender} bytte till läge {m.mode} t={m.t:.1f} efter {cause}")
    return Result("INV10", not bad, len(bad), 0, "; ".join(bad[:5]))


def _fallbacks(mode: int) -> frozenset[int]:
    chains = {BRAKE: {LOITER, LAND}, SMART_RTL: {RTL, LAND}, RTL: {LAND}, LOITER: {LAND}}
    return frozenset(chains.get(mode, set()))


def _releases(kind: str, payload: dict[str, Any]) -> bool:
    return kind in RELEASING_EVENTS and str(payload.get("by", "")).startswith("operator:")


def _holds(tl: Timeline) -> list[tuple[float, int, str, int | None]]:
    """(time, mode, cause, index of the causing event) of every hold nothing may fight."""
    out: list[tuple[float, int, str, int | None]] = []
    asked = [(m.t, m.mode) for m in tl.sent if m.mode is not None and m.sender != "harness"]
    prev: int | None = None
    for s in tl.samples:
        if s.mode != prev and s.mode in HOLDING_MODES and s.armed:
            ours = any(mode == s.mode and -1.0 <= s.t - t <= 5.0 for t, mode in asked)
            if not ours:
                out.append((s.t, s.mode, f"failsafe {s.mode}", None))
        prev = s.mode
    for n, (t, kind, p) in enumerate(tl.events):
        if kind in ("vehicle.stop", "vehicle.return_home") and _releases(kind, p):
            mode = BRAKE if kind == "vehicle.stop" else SMART_RTL
            out.append((t, mode, kind, n))
    return out


def inv11(tl: Timeline) -> Result:
    """Autonomy changes only through a charter the operator approved."""
    bad: list[str] = []
    ordered = sorted(tl.charters, key=lambda c: c.since)
    for before, after in pairwise(ordered):
        if after.autonomy_mode != before.autonomy_mode and not str(after.approved_by).startswith(
            "operator:"
        ):
            bad.append(f"läget ändrat av {after.approved_by}")
    for c in ordered:
        if str(c.approved_by).startswith("brain:"):
            bad.append("en ram godkänd av AI:n")
    return Result("INV11", not bad, len(bad), 0, "; ".join(bad))


def check_all(
    tl: Timeline,
    exempt: Iterable[str] = (),
    *,
    harness_extra: Sequence[str] = (),
    allow_mismatch: bool = False,
) -> list[Result]:
    """Every invariant that is not exempt for the scenario."""
    skip = set(exempt)
    unknown = skip - set(ALL)
    if unknown:
        raise ValueError(f"okända invarianter: {sorted(unknown)}")
    checks = {
        "INV1": lambda: inv1(tl),
        "INV2": lambda: inv2(tl),
        "INV3": lambda: inv3(tl),
        "INV4": lambda: inv4(tl),
        "INV5": lambda: inv5(tl, harness_extra),
        "INV6": lambda: inv6(tl),
        "INV7": lambda: inv7(tl, allow_mismatch),
        "INV8": lambda: inv8(tl),
        "INV9": lambda: inv9(tl),
        "INV10": lambda: inv10(tl),
        "INV11": lambda: inv11(tl),
    }
    return [checks[i]() for i in ALL if i not in skip]
