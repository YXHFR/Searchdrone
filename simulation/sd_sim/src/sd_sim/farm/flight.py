"""What a farm round's flight leaves for the grade, and the parts of the flight that need
no simulator: the class register the flight sees people with, the script at the round's
height and ``runs/<run>/flight.json`` (``sd-brain eval grade`` reads it as its
``FarmFlight``). The flight itself is ``simulation/tests/scenarios/farm_flight.py``.
"""

from __future__ import annotations

import datetime as dt
import itertools
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sd_detect.classes import ClassRegistry, default_registry
from sd_safety.targets import distance_m
from sd_schemas.candidate_ai import named_seqs

HOME_RADIUS_M = 3.0  # landed at home: this close to the take-off point
# The search flew on from a raised candidate: this much flown after the raise, not to the
# candidate (legs ending this close to it). The simulator exam's rule (sd_brain.simeval).
FLEW_ON_M = 30.0
CLOSE_UP_RADIUS_M = 15.0
ASK = "?"  # a message from the AI that asks the pilot something
PENDING = frozenset({"pending_approval", "approval_expired", "rejected_by_operator"})
_ALT = re.compile(r"\balt_m: 40\b")
_FIXTURE = re.compile(r"^fixture: (.+)$", re.MULTILINE)


def farm_registry() -> ClassRegistry:
    """The class register with people from the simulator's labels (the oracle source),
    while the simulator has no person model; every other class as registered."""
    return ClassRegistry(
        s.model_copy(update={"source": "oracle"}) if s.id == "person" else s
        for s in default_registry()
    )


def script_at(path: Path, height_m: float) -> str:
    """The farm's script (``SIMFARM.yaml``) at the round's height (every ``alt_m: 40``),
    to be written elsewhere: its fixture, named relative to the script, becomes absolute."""
    text = path.read_text(encoding="utf-8")
    if not _ALT.search(text):
        raise ValueError("manuset har ingen höjd att byta (alt_m: 40)")
    text = _FIXTURE.sub(lambda m: f"fixture: {(path.parent / m.group(1)).resolve()}", text)
    return _ALT.sub(f"alt_m: {height_m:g}", text)


def _utc_s(value: str) -> float:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _order(c: Mapping[str, Any], gate: Sequence[Mapping[str, Any]], t0: float) -> dict[str, Any]:
    decision = "ALLOW"
    if c["status"] == "rejected_by_gate":
        decision = "REJECT"
    elif c["status"] in PENDING:
        decision = "PENDING_APPROVAL"
    codes = sorted({str(code) for g in gate for code in g.get("reason_codes") or ()})
    return {
        "t_s": round(_utc_s(c["created_at"]) - t0, 2),
        "skill": c["skill"],
        "decision": decision,
        "reason_codes": codes,
    }


def _told(
    commands: Sequence[Mapping[str, Any]], chat: Sequence[Mapping[str, Any]], t0: float
) -> list[tuple[float, str]]:
    """What the AI told the operator, oldest first: its chat messages and the text of its
    reports, each with its time from the first order."""
    said = [(m.get("created_at"), m.get("text")) for m in chat if m.get("role") == "brain"]
    said += [
        (c.get("created_at"), (c.get("params") or {}).get("text_sv"))
        for c in commands
        if c.get("issuer_kind") == "brain" and c.get("skill") == "report"
    ]
    return sorted((_utc_s(at) - t0, str(text)) for at, text in said if at and text)


def _flew_on(
    lat: float, lon: float, raised: float | None, track: Sequence[tuple[float, float, float]]
) -> bool:
    """``FLEW_ON_M`` flown after the raise (seconds from the first order), legs that end
    near the candidate not counted. ``track``: (seconds from the first order, lat, lon)."""
    if raised is None:
        return False
    flown = 0.0
    after = [p for p in track if p[0] >= raised]
    for (_, a_lat, a_lon), (_, b_lat, b_lon) in itertools.pairwise(after):
        if distance_m(b_lat, b_lon, lat, lon) > CLOSE_UP_RADIUS_M:
            flown += distance_m(a_lat, a_lon, b_lat, b_lon)
    return flown >= FLEW_ON_M


def _candidates(
    cards: Sequence[Mapping[str, Any]],
    told: Sequence[tuple[float, str]],
    t0: float,
    track: Sequence[tuple[float, float, float]],
) -> list[dict[str, Any]]:
    """The ground's candidates, each raised when the AI first named it (``K3``) to the
    operator after it was seen, as the simulator exam counts a raise, and whether the
    search flew on after that (``flew_on``)."""
    out = []
    for c in cards:
        seen = _utc_s(c["first_seen_utc"]) - t0
        raised = next((t for t, text in told if t >= seen and c["seq"] in named_seqs(text)), None)
        out.append(
            {
                "seq": int(c["seq"]),
                "t_s": round(max(0.0, seen), 2),
                "lat": float(c["lat"]),
                "lon": float(c["lon"]),
                "priority_class": c["priority_class"],
                "raised_t_s": None if raised is None else round(max(0.0, raised), 2),
                "flew_on": _flew_on(float(c["lat"]), float(c["lon"]), raised, track),
            }
        )
    return out


def flight_record(
    *,
    started: float,
    ended: float,
    commands: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Sequence[Mapping[str, Any]]],
    finds: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    chat: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]] = (),
    track: Sequence[tuple[float, float, float]] = (),
    breaches: int,
    coverage: float,
    landed_home: bool,
    turns: int,
    max_time_s: float,
) -> dict[str, Any]:
    """``flight.json`` from what the ground station recorded: the AI's orders with the
    gate's answers, the people found (a find joins the detections of one person; its time
    is when its best picture was taken), the ground's candidate queue (``candidates``, its
    cards) with when the AI raised each and whether the search flew on after that (from
    ``track``: UTC seconds, lat, lon of the armed flight), the pilot questions and the
    landing. Times are seconds from the first order."""
    t0 = min((_utc_s(c["created_at"]) for c in commands), default=0.0)
    taken = {str(d.get("detection_id", d.get("id"))): d.get("captured_at") for d in detections}
    orders = [
        _order(c, gates.get(c["id"], ()), t0) for c in commands if c.get("issuer_kind") == "brain"
    ]
    marks = []
    for f in finds:
        if f.get("label") != "person" or f.get("lat") is None:
            continue
        at = taken.get(str(f.get("thumb_detection_id")))
        marks.append(
            {
                "t_s": round(max(0.0, _utc_s(at) - t0), 2) if at else 0.0,
                "lat": float(f["lat"]),
                "lon": float(f["lon"]),
                "label": "person",
            }
        )
    asked = [m for m in chat if m.get("role") == "brain" and ASK in str(m.get("text", ""))]
    return {
        "duration_s": round(max(0.0, ended - started), 1),
        "max_time_s": max_time_s,
        "orders": orders,
        "marks": marks,
        "candidates": _candidates(
            candidates, _told(commands, chat, t0), t0, [(t - t0, la, lo) for t, la, lo in track]
        ),
        "geofence_breaches": breaches,
        "coverage": round(max(0.0, min(1.0, coverage)), 4),
        "pilot_questions": len(asked),
        "landed_home": landed_home,
        "turns": turns,
    }
