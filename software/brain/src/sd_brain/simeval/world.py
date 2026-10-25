"""A fast 2D search world and the ground station around it, for simulator exams.

``SimWorld`` flies the orders the gate let through, the way the executor would (straight
legs, a lawnmower over a sector, home and land), drains the battery, keeps a coverage grid
and runs a simple camera: a person under the footprint is seen with a chance per second
that falls with height. Each sighting becomes a map pin and a warning; a pass at 30 m or
lower within 15 m of a pin gives a close-up verdict. False sightings come at scenario
times, and the pilot's STOP (and its release) when the scenario says.

``SimGround`` is ``FakeGround`` with this world behind it and a deliberately reduced copy
of the gate: allowed skills, area, height band, speed and known sectors, with the real
reason codes. It does not check range from the pilot or approval modes (the scenarios fly
``WITHIN_CHARTER`` with skills that need no approval); ``sd_brain`` does not depend on
``sd_safety``. The AI only decides: every movement here happens in the world, as the
executor and ArduPilot would do it.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import Point, Polygon

from sd_brain.evals.fake_ground import FakeGround
from sd_brain.geo import from_enu, to_enu
from sd_brain.simeval.scenario import HOME, SimScenario
from sd_schemas.base import new_id
from sd_schemas.commander import Situation, SkillRequest, SkillSubmitResult

__all__ = ["Mark", "OrderRecord", "SimGround", "SimWorld"]

DEFAULT_SPEED_MS = 8.0
BATTERY_PCT_PER_S = 0.065
FOOTPRINT_PER_M = 0.7  # half-width of the camera footprint per metre of height
COVERAGE_MAX_ALT_M = 80.0  # higher than this, a pass does not count as searched
CELL_M = 10.0
CLOSE_UP_ALT_M = 30.0
CLOSE_UP_RADIUS_M = 15.0
# Plans the AI did not order: no skill event when they end.
BATTERY_ID = "battery"
FLY_HOME_ID = "fly_home"


def _detect_p(alt: float, visibility: float) -> float:
    """The chance per second of seeing a person under the footprint."""
    return 0.25 * visibility * min(1.0, max(0.1, (100.0 - alt) / 70.0))


@dataclass
class Mark:
    t: float
    lat: float
    lon: float
    label: str


@dataclass
class OrderRecord:
    t: float
    skill: str
    decision: str
    reason_codes: list[str]
    under_stop: bool


@dataclass
class _Sighting:
    pin: str
    east: float
    north: float
    true: bool
    confidence: int
    close_up: bool = False


@dataclass
class _Plan:
    command_id: str
    skill: str
    points: list[tuple[float, float]]
    alt: float
    speed: float
    total_m: float
    flown_m: float = 0.0
    sector: str | None = None
    lands: bool = False


@dataclass
class SimWorld:
    scenario: SimScenario
    t: float = 0.0
    east: float = 0.0
    north: float = 0.0
    alt: float = 40.0
    battery: float = 100.0
    landed: bool = False
    latched: bool = False
    battery_rtl: bool = False
    plan: _Plan | None = None
    held_plan: _Plan | None = None
    marks: list[Mark] = field(default_factory=list[Mark])
    sightings: list[_Sighting] = field(default_factory=list[_Sighting])
    geofence_breaches: int = 0
    stop_set_at: float | None = None
    stop_released_at: float | None = None
    sector_progress: dict[str, float] = field(default_factory=dict[str, float])

    def __post_init__(self) -> None:
        sit = self.scenario.situation
        ch = sit["charter"]
        self.rnd = random.Random(self.scenario.seed * 7919 + 1)  # noqa: S311
        self.area = self._poly(ch["area"])
        self.fence = self.area.buffer(float(ch["geofence_buffer_m"]))
        self.sectors = {s["sector_id"]: self._poly(s["area"]) for s in ch["sectors"]}
        self.sector_progress = dict.fromkeys(self.sectors, 0.0)
        self.battery = float(sit["vehicle"]["battery_pct"])
        self.alt = float(sit["vehicle"]["alt_rel_home_m"])
        self.rtl_pct = float(ch["battery_rtl_pct"])
        self.max_speed = float(ch["max_speed_ms"])
        minx, miny, maxx, maxy = self.area.bounds
        self._cells = {
            (i, j)
            for i in range(int((maxx - minx) // CELL_M) + 1)
            for j in range(int((maxy - miny) // CELL_M) + 1)
            if self.area.contains(Point(minx + (i + 0.5) * CELL_M, miny + (j + 0.5) * CELL_M))
        }
        self._origin = (minx, miny)
        self._seen: set[tuple[int, int]] = set()
        self._false_left = list(self.scenario.false_alarms_at_s)

    # --- geometry -----------------------------------------------------------------------

    @staticmethod
    def _poly(geo: dict[str, Any]) -> Polygon:
        return Polygon([to_enu(lat, lon, *HOME) for lon, lat in geo["coordinates"][0]])

    def latlon(self) -> tuple[float, float]:
        return from_enu(self.east, self.north, *HOME)

    def coverage(self) -> float:
        return len(self._seen & self._cells) / len(self._cells) if self._cells else 0.0

    def _lawnmower(self, poly: Polygon, spacing: float) -> list[tuple[float, float]]:
        minx, miny, maxx, maxy = poly.bounds
        pts: list[tuple[float, float]] = []
        x, up = minx + spacing / 2, True
        while x < maxx:
            lane = [(x, miny + 2), (x, maxy - 2)]
            pts.extend(lane if up else lane[::-1])
            x, up = x + spacing, not up
        c = poly.centroid
        return pts or [(c.x, c.y)]

    def _length(self, pts: list[tuple[float, float]]) -> float:
        total, cur = 0.0, (self.east, self.north)
        for p in pts:
            total += math.dist(cur, p)
            cur = p
        return total

    # --- orders -------------------------------------------------------------------------

    def start(self, req: SkillRequest) -> bool:
        """Start an order the gate allowed. False: nothing to fly (done at once)."""
        p = req.params
        speed = min(float(p.get("speed_ms") or DEFAULT_SPEED_MS), self.max_speed)
        cid = str(req.command_id)
        if req.skill == "goto":
            pts = [to_enu(float(p["lat"]), float(p["lon"]), *HOME)]
            self.plan = _Plan(cid, "goto", pts, float(p["alt_m"]), speed, self._length(pts))
        elif req.skill == "survey_sector":
            sector = str(p.get("sector_id"))
            alt = float(p.get("alt_m") or self.alt)
            spacing = float(p.get("spacing_m") or max(10.0, 1.4 * FOOTPRINT_PER_M * alt))
            pts = self._lawnmower(self.sectors[sector], spacing)
            self.plan = _Plan(cid, "survey_sector", pts, alt, speed, self._length(pts), 0, sector)
        elif req.skill == "return_home":
            self.plan = _Plan(cid, "return_home", [(0.0, 0.0)], self.alt, speed, 0, lands=True)
            self.plan.total_m = self._length(self.plan.points)
        elif req.skill == "resume" and self.held_plan is not None:
            self.plan, self.held_plan = self.held_plan, None
            self.plan.command_id = cid
        elif req.skill == "hold":
            self.held_plan, self.plan = self.plan, None
            return False
        else:
            return False
        return True

    def fly_home(self) -> None:
        """The pilot's FLY HOME button: home and land, past the AI (no order, no gate)."""
        if not self.landed:
            self.held_plan, self.plan = None, self._home_plan(FLY_HOME_ID)

    def _home_plan(self, command_id: str) -> _Plan:
        """A return home the AI did not order (battery, the pilot's button)."""
        return _Plan(command_id, "return_home", [(0.0, 0.0)], self.alt, 8.0, 0, lands=True)

    def mark(self, lat: float, lon: float, label: str) -> None:
        self.marks.append(Mark(self.t, lat, lon, label))

    def marked(self, s: _Sighting) -> bool:
        """A mark lies on this sighting's pin. The ground shows a mark as a pin of its own;
        here the sighting's pin says so, so the AI can see what it already marked."""
        return any(
            math.dist(to_enu(m.lat, m.lon, *HOME), (s.east, s.north)) < 1 for m in self.marks
        )

    # --- time ---------------------------------------------------------------------------

    def _event(self, kind: str, body: dict[str, Any]) -> dict[str, Any]:
        return {"type": kind, "body": body}

    def _sight(self, east: float, north: float, true: bool) -> dict[str, Any]:
        pin = f"P{len(self.sightings) + 1}"
        conf = self.rnd.randint(60, 92) if true else self.rnd.randint(35, 75)
        self.sightings.append(_Sighting(pin, east, north, true, conf))
        return self._event(
            "event.alert",
            {
                "level": "warning",
                "code": f"POSSIBLE_PERSON_{pin}",
                "message_sv": f"Detektorn såg en möjlig person ({conf} %), kartnål {pin}.",
            },
        )

    def _close_ups(self) -> list[dict[str, Any]]:
        out = []
        for s in self.sightings:
            near = math.dist((self.east, self.north), (s.east, s.north)) <= CLOSE_UP_RADIUS_M
            if s.close_up or not near or self.alt > CLOSE_UP_ALT_M:
                continue
            s.close_up = True
            verdict = "trolig person" if s.true else "ingen person"
            out.append(
                self._event(
                    "event.alert",
                    {
                        "level": "warning",
                        "code": f"CLOSE_UP_{s.pin}",
                        "message_sv": f"Närbild vid {s.pin}: {verdict}.",
                    },
                )
            )
        return out

    def _camera(self) -> list[dict[str, Any]]:
        if self.landed:
            return []
        r = FOOTPRINT_PER_M * self.alt
        if self.alt <= COVERAGE_MAX_ALT_M:
            ox, oy = self._origin
            ci, cj = int((self.east - ox) // CELL_M), int((self.north - oy) // CELL_M)
            k = int(r // CELL_M) + 1
            for i in range(ci - k, ci + k + 1):
                for j in range(cj - k, cj + k + 1):
                    cx, cy = ox + (i + 0.5) * CELL_M, oy + (j + 0.5) * CELL_M
                    if math.dist((cx, cy), (self.east, self.north)) <= r:
                        self._seen.add((i, j))
        out: list[dict[str, Any]] = []
        te, tn = self.scenario.target_east_north
        seen_target = any(s.true for s in self.sightings)
        if (
            not seen_target
            and math.dist((te, tn), (self.east, self.north)) <= r
            and self.rnd.random() < _detect_p(self.alt, self.scenario.visibility)
        ):
            jitter = self.rnd.uniform(0, 6), self.rnd.uniform(0, 2 * math.pi)
            e = te + jitter[0] * math.cos(jitter[1])
            n = tn + jitter[0] * math.sin(jitter[1])
            out.append(self._sight(e, n, True))
        if self._false_left and self.t >= self._false_left[0] and self.plan is not None:
            self._false_left.pop(0)
            out.append(self._sight(self.east + 5, self.north - 5, False))
        return out

    def _fly(self, plan: _Plan) -> bool:
        """One second along the plan; True when its last point is reached."""
        self.alt += max(-3.0, min(3.0, plan.alt - self.alt))
        left = plan.speed
        while left > 0 and plan.points:
            tx, ty = plan.points[0]
            d = math.dist((self.east, self.north), (tx, ty))
            if d <= left:
                self.east, self.north = tx, ty
                plan.points.pop(0)
                left -= d
                plan.flown_m += d
            else:
                self.east += (tx - self.east) / d * left
                self.north += (ty - self.north) / d * left
                plan.flown_m += left
                left = 0
        if plan.sector is not None and plan.total_m > 0:
            self.sector_progress[plan.sector] = min(1.0, plan.flown_m / plan.total_m)
        if not self.fence.contains(Point(self.east, self.north)) and not plan.lands:
            self.geofence_breaches += 1
        return not plan.points and abs(plan.alt - self.alt) < 0.5

    def advance(self, max_s: float = 600.0) -> list[dict[str, Any]]:
        """Run the world until something the AI should hear happens (or ``max_s``)."""
        end = self.t + max_s
        while self.t < end and not self.landed and self.t < self.scenario.max_time_s:
            self.t += 1.0
            self.battery = max(0.0, self.battery - BATTERY_PCT_PER_S)
            sc = self.scenario
            if sc.stop_at_s is not None and self.stop_set_at is None and self.t >= sc.stop_at_s:
                self.stop_set_at, self.latched = self.t, True
                self.held_plan, self.plan = self.plan, None
                return [
                    self._event(
                        "event.operator_override",
                        {"active": True, "action": "stop", "via": "button"},
                    )
                ]
            if (
                self.latched
                and self.stop_set_at is not None
                and self.t >= self.stop_set_at + sc.stop_hold_s
            ):
                self.latched, self.stop_released_at = False, self.t
                return [self._event("event.operator_override", {"active": False})]
            if self.battery <= self.rtl_pct and not self.battery_rtl:
                self.battery_rtl = True
                self.plan = self._home_plan(BATTERY_ID)
                return [
                    self._event(
                        "event.mission", {"status": "RETURNING", "returning_cause": "battery"}
                    )
                ]
            plan = self.plan
            if plan is None:
                if self.latched:
                    continue
                return []
            done = self._fly(plan)
            events = self._camera() + self._close_ups()
            if done:
                self.plan = None
                if plan.lands:
                    self.landed = True
                    self.alt = 0.0
                if plan.command_id not in (BATTERY_ID, FLY_HOME_ID):
                    events.append(
                        self._event(
                            "event.skill",
                            {"skill": plan.skill, "status": "done", "command_id": plan.command_id},
                        )
                    )
            if events:
                return events
        return []

    def finished(self) -> bool:
        return self.landed or self.t >= self.scenario.max_time_s or self.battery <= 0


class SimGround(FakeGround):
    """The commander API in memory, with the world behind it and the gate's area rules."""

    def __init__(self, world: SimWorld, *, now: Any) -> None:
        super().__init__(world.scenario.situation, now=now)
        self.world = world
        self.orders: list[OrderRecord] = []

    def sync(self) -> None:
        """Copy the world into the situation, as the onboard snapshots would."""
        w, s = self.world, self.sit
        lat, lon = w.latlon()
        v = s["vehicle"]
        v.update(
            lat=round(lat, 7),
            lon=round(lon, 7),
            alt_rel_home_m=round(w.alt, 1),
            battery_pct=round(w.battery, 1),
            battery_to_reserve_s=max(0, int((w.battery - w.rtl_pct) / BATTERY_PCT_PER_S)),
            dist_to_pilot_m=round(math.hypot(w.east, w.north), 1),
            groundspeed_ms=w.plan.speed if w.plan is not None else 0.0,
        )
        if w.landed:
            v.update(airborne=False, landed=True, armed=False, executor_state="PREFLIGHT")
            s["mission_status"] = "LANDED"
            s["active_command"] = None
        for x in s["sectors"]:
            prog = w.sector_progress.get(str(x["sector_id"]), 0.0)
            if x["status"] != "done":
                x["progress"] = round(prog, 2)
                if prog > 0:
                    x["status"] = (
                        "active" if w.plan and w.plan.sector == x["sector_id"] else x["status"]
                    )
        s["pins"] = [
            {
                "id": g.pin,
                "annotation_id": str(new_id())
                if i >= len(s["pins"])
                else s["pins"][i]["annotation_id"],
                "lat": round(from_enu(g.east, g.north, *HOME)[0], 7),
                "lon": round(from_enu(g.east, g.north, *HOME)[1], 7),
                "note": f"möjlig person {g.confidence} %{', markerad' if w.marked(g) else ''}",
            }
            for i, g in enumerate(w.sightings)
        ]
        s["alerts"] = [
            {"code": f"POSSIBLE_PERSON_{g.pin}", "level": "warning"} for g in w.sightings
        ]
        s["state_seq"] += 1
        s["snapshot_state_seq"] = (s.get("snapshot_state_seq") or 0) + 1

    async def situation(self, mission_id: str) -> Situation:
        self.sync()
        return self.situation_model()

    def _gate(self, req: SkillRequest) -> tuple[str, dict[str, Any]] | None:
        ch = self.sit["charter"]
        p = req.params
        if req.skill not in ch["allowed_skills"]:
            return "SKILL_NOT_ALLOWED", {}
        alt = p.get("alt_m")
        if isinstance(alt, int | float) and req.skill in ("goto", "survey_sector"):
            if alt > ch["alt_max_agl_m"]:
                return "ALT_ABOVE_CHARTER", {"param": "alt_m", "limit": ch["alt_max_agl_m"]}
            if alt < ch["alt_min_agl_m"]:
                return "ALT_BELOW_CHARTER", {"param": "alt_m", "limit": ch["alt_min_agl_m"]}
        speed = p.get("speed_ms")
        if isinstance(speed, int | float) and speed > ch["max_speed_ms"]:
            return "SPEED_ABOVE_CHARTER", {"param": "speed_ms", "limit": ch["max_speed_ms"]}
        if req.skill == "goto" and not self.world.area.contains(
            Point(to_enu(float(p["lat"]), float(p["lon"]), *HOME))
        ):
            return "OUTSIDE_CHARTER", {}
        if req.skill == "mark_location" and not self.world.fence.contains(
            Point(to_enu(float(p["lat"]), float(p["lon"]), *HOME))
        ):
            return "OUTSIDE_CHARTER", {}
        if req.skill == "survey_sector" and p.get("sector_id") not in self.world.sectors:
            return "ONB_UNKNOWN_SECTOR", {}
        return None

    async def submit(self, mission_id: str, req: SkillRequest) -> SkillSubmitResult:
        under_stop = self.world.latched
        rejected = self._gate(req)
        if rejected is not None:
            code, details = rejected
            self.step_responses = [
                {
                    "match": {"skill": req.skill},
                    "result": {
                        "decision": "REJECT",
                        "reason_codes": [code],
                        "message_sv": "Grinden sa nej.",
                        "details": details,
                        "status": "rejected_by_gate",
                    },
                }
            ]
        try:
            result = await super().submit(mission_id, req)
        finally:
            self.step_responses = []
        self.orders.append(
            OrderRecord(self.world.t, req.skill, result.decision, result.reason_codes, under_stop)
        )
        if result.decision != "ALLOW":
            return result
        if req.skill == "mark_location":
            p = req.params
            self.world.mark(float(p["lat"]), float(p["lon"]), str(p.get("label", "")))
        elif self.world.start(req):
            cmd = copy.deepcopy(_COMMAND)
            now = self._now().isoformat().replace("+00:00", "Z")
            cmd.update(
                id=str(req.command_id),
                mission_id=self.mission_id,
                skill=req.skill,
                params=req.params,
                created_at=now,
                updated_at=now,
            )
            self.sit["active_command"] = cmd
            self.sit["vehicle"]["executor_state"] = (
                "SURVEY"
                if req.skill == "survey_sector"
                else "RTL"
                if req.skill == "return_home"
                else "TRANSIT"
            )
        return result


_COMMAND: dict[str, Any] = {
    "vehicle_id": "sim-01",
    "issuer_kind": "brain",
    "issuer_ref": "brain:sim",
    "backend": "llm",
    "rationale_sv": None,
    "in_reply_to": None,
    "charter_version": 1,
    "based_on": None,
    "status": "running",
    "status_reason": None,
    "approval_id": None,
    "expires_at": None,
}
