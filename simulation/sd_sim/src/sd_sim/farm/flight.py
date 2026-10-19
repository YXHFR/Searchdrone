"""What a farm round's flight leaves for the grade, and the parts of the flight that need
no simulator: the class register the flight sees people with, the script at the round's
height and ``runs/<run>/flight.json`` (``sd-brain eval grade`` reads it as its
``FarmFlight``). The flight itself is ``simulation/tests/scenarios/farm_flight.py``.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sd_detect.classes import ClassRegistry, default_registry

HOME_RADIUS_M = 3.0  # landed at home: this close to the take-off point
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


def flight_record(
    *,
    started: float,
    ended: float,
    commands: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Sequence[Mapping[str, Any]]],
    finds: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    chat: Sequence[Mapping[str, Any]],
    breaches: int,
    coverage: float,
    landed_home: bool,
    turns: int,
    max_time_s: float,
) -> dict[str, Any]:
    """``flight.json`` from what the ground station recorded: the AI's orders with the
    gate's answers, the people found (a find joins the detections of one person; its time
    is when its best picture was taken), the pilot questions and the landing. Times are
    seconds from the first order."""
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
        "geofence_breaches": breaches,
        "coverage": round(max(0.0, min(1.0, coverage)), 4),
        "pilot_questions": len(asked),
        "landed_home": landed_home,
        "turns": turns,
    }
