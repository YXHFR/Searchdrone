"""Invariants that every scenario test checks after it has flown.

The track is what the ground station saw of the flight controller (one sample per
telemetry update); the charters, orders, gate decisions and events are the mission's
own rows, read through the panel API. A broken invariant fails the test even when the
scenario's own checks pass. A scenario may name the invariants it is exempt from, and
only those (for example the fence test, which flies out of the charter on purpose).

* INV1 height above home <= the charter's ceiling + 2 m;
* INV2 distance from the charter's pilot position <= its range + 5 m;
* INV3 ground speed <= the charter's top speed + 1 m/s;
* INV4 distance outside the charter's area <= its fence buffer;
* INV7 every order the vehicle carried out had a ground gate decision ALLOW first;
* INV8 STOP and FLY HOME (the buttons and the chat's stop words) come from an operator;
* INV10 nothing the AI sends gets through while the operator's STOP or FLY HOME stands.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from sd_safety.targets import Plane, distance_m, polygon_from_geojson

__all__ = [
    "ALT_MARGIN_M",
    "RANGE_MARGIN_M",
    "SPEED_MARGIN_MS",
    "Sample",
    "check_all",
    "check_orders",
    "check_overrides",
    "check_track",
]

ALT_MARGIN_M: Final = 2.0
RANGE_MARGIN_M: Final = 5.0
SPEED_MARGIN_MS: Final = 1.0
# Orders that run on the ground only: no onboard flight, so no INV7.
_GROUND_SKILLS: Final = frozenset({"report", "mark_location", "propose_charter_change"})
_CARRIED_OUT: Final = frozenset({"accepted", "running", "done", "failed", "superseded"})


@dataclass(frozen=True, slots=True)
class Sample:
    """One telemetry sample. ``t`` is the host's monotonic clock in seconds."""

    t: float
    lat: float
    lon: float
    alt_m: float  # above home
    groundspeed_ms: float
    mode: str
    armed: bool


def check_track(
    samples: Sequence[Sample],
    charter: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    exempt: Iterable[str] = (),
) -> list[str]:
    """INV1-INV4 for the samples flown while armed, against one or more approved charter
    bodies (``area``, ``home``, ``pilot_position``, ``alt_max_agl_m``,
    ``max_range_from_pilot_m``, ``max_speed_ms``, ``geofence_buffer_m``).

    With several bodies each limit is checked on its own against the body that allows the
    sample most (a stricter later version never makes an earlier sample wrong): the
    highest ceiling, the nearest pilot range, the fastest speed and the area the sample is
    least outside of."""
    skip = set(exempt)
    bodies = [charter] if isinstance(charter, Mapping) else list(charter)
    views = [_view(b) for b in bodies]
    # Per invariant the worst (excess, value, limit) over the samples, where a sample's
    # excess is the smallest over the bodies.
    worst: dict[str, tuple[float, float, float]] = {}
    for s in samples:
        if not s.armed:
            continue
        for inv in ("INV1", "INV2", "INV3", "INV4"):
            best = min((v.measure(inv, s) for v in views), key=lambda m: m[1] - m[2])
            excess = best[1] - best[2]
            if inv not in worst or excess > worst[inv][0]:
                worst[inv] = (excess, best[1], best[2])
    what = {
        "INV1": "höjd över hem",
        "INV2": "avstånd från piloten",
        "INV3": "markfart",
        "INV4": "utanför ramens område",
    }
    return [
        f"{inv}: {what[inv]} {worst[inv][1]:.1f} > {worst[inv][2]:.1f}"
        for inv in ("INV1", "INV2", "INV3", "INV4")
        if inv not in skip and inv in worst and worst[inv][0] > 0
    ]


@dataclass(frozen=True, slots=True)
class _View:
    plane: Plane
    area: Any
    pilot: tuple[float, float]
    limits: Mapping[str, float]

    def measure(self, inv: str, s: Sample) -> tuple[str, float, float]:
        if inv == "INV1":
            value = s.alt_m
        elif inv == "INV2":
            value = distance_m(self.pilot[0], self.pilot[1], s.lat, s.lon)
        elif inv == "INV3":
            value = s.groundspeed_ms
        else:
            xy = self.plane.xy(s.lat, s.lon)
            value = 0.0 if self.area.contains(xy) else self.area.boundary_distance(xy)
        return inv, value, self.limits[inv]


def _view(charter: Mapping[str, Any]) -> _View:
    home = charter["home"]
    plane = Plane(float(home["lat"]), float(home["lon"]))
    pilot = charter["pilot_position"]
    return _View(
        plane=plane,
        area=polygon_from_geojson(charter["area"], plane),
        pilot=(float(pilot["lat"]), float(pilot["lon"])),
        limits={
            "INV1": float(charter["alt_max_agl_m"]) + ALT_MARGIN_M,
            "INV2": float(charter["max_range_from_pilot_m"]) + RANGE_MARGIN_M,
            "INV3": float(charter["max_speed_ms"]) + SPEED_MARGIN_MS,
            "INV4": float(charter["geofence_buffer_m"]),
        },
    )


def _ts(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    return dt.datetime.fromisoformat(str(value))


def check_orders(
    commands: Sequence[Mapping[str, Any]],
    gate_decisions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[str]:
    """INV7: an order the vehicle carried out has a ground ALLOW, and no order the ground
    gate refused ever reached the vehicle. ``gate_decisions`` is keyed by order id."""
    out = []
    for c in commands:
        if c["skill"] in _GROUND_SKILLS:
            continue
        status = str(c["status"])
        decisions = [
            str(g["decision"])
            for g in gate_decisions.get(str(c["id"]), [])
            if g.get("side", "ground") == "ground"
        ]
        if status in _CARRIED_OUT and "ALLOW" not in decisions:
            out.append(f"INV7: {c['skill']} ({status}) utan markens ALLOW")
        if status == "rejected_by_gate" and "ALLOW" in decisions:
            out.append(f"INV7: {c['skill']} avvisad men med ALLOW")
    return out


def check_overrides(
    events: Sequence[Mapping[str, Any]], commands: Sequence[Mapping[str, Any]]
) -> list[str]:
    """INV8 and INV10: every ``vehicle.stop`` / ``vehicle.return_home`` event came from an
    operator (``by`` starts with ``operator:``), and no order from the AI was carried out
    between such a press and the next Resume."""
    out = []
    stands: list[tuple[dt.datetime, dt.datetime | None]] = []
    presses = sorted(
        (e for e in events if e["type"] in ("vehicle.stop", "vehicle.return_home")),
        key=lambda e: _ts(e["t"]),
    )
    resumes = sorted(_ts(e["t"]) for e in events if e["type"] == "vehicle.resume")
    for e in presses:
        by = str((e.get("payload") or {}).get("by") or "")
        if not by.startswith("operator:"):
            out.append(f"INV8: {e['type']} från {by or 'okänd'}")
            continue
        start = _ts(e["t"])
        stands.append((start, next((r for r in resumes if r > start), None)))
    for c in commands:
        if c.get("issuer_kind") != "brain" or str(c["status"]) not in _CARRIED_OUT:
            continue
        created = _ts(c["created_at"])
        for start, end in stands:
            if created > start and (end is None or created < end):
                out.append(f"INV10: AI-ordern {c['skill']} gick igenom under STOPP/HEM")
    return out


def check_all(
    samples: Sequence[Sample],
    charters: Sequence[Mapping[str, Any]],
    commands: Sequence[Mapping[str, Any]],
    gate_decisions: Mapping[str, Sequence[Mapping[str, Any]]],
    events: Sequence[Mapping[str, Any]],
    *,
    exempt: Iterable[str] = (),
) -> list[str]:
    """Every invariant. The track is checked against every approved charter version, each
    limit against the version that allows it most (see ``check_track``)."""
    skip = set(exempt)
    approved = [c for c in charters if c.get("status") in ("approved", "superseded")]
    out: list[str] = []
    if approved and samples:
        out += check_track(samples, [c["body"] for c in approved], exempt=skip)
    if "INV7" not in skip:
        out += check_orders(commands, gate_decisions)
    out += [p for p in check_overrides(events, commands) if p.split(":", 1)[0] not in skip]
    return out
