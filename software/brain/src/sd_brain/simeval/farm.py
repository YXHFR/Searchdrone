"""``sd-brain eval grade --round <dir>``: the grade of one simulator-farm round.

The farm flies a round in SITL and writes the round folder; this reads it and writes
``grade.json`` next to ``round.json``. What it reads:

- ``round.json`` (``simfarm.round/v1``): ``round_id``, ``seed``, ``commander_requested``
  (``script`` or ``sd-brain``), optional ``truth`` (default ``truth.json``) and ``runs``
  (folder paths relative to the round, default every folder under ``runs/``).
- ``truth.json``: the answer key, ``targets`` (a list of ``{"lat", "lon"}``) or ``target``.
- per run, ``flight.json``: what happened, in the form of :class:`FarmFlight` (the orders
  with the gate's answers, the people the ground found as ``marks``, the ground's
  candidates with when the AI raised each, geofence breaches, coverage, pilot questions,
  landing). The farm builds it from its own logs (tlog, commands, chat).
- per run, ``brain.json`` when sd-brain decided (:class:`BrainRecord`, written with
  :func:`brain_record`): which decision maker flew, why it switched and what it cost.

The grade is :func:`score_flight`, the same as for ``sd-brain eval --backend sim``, best
over the targets. A candidate is raised when the AI first named it (``K3``) to the
operator in a chat message or a report, as the exam counts it. The farm does not measure
yet whether the search flew on after a raise: a raised candidate counts as flown on. A run
whose ground had no candidate queue (no candidates in ``flight.json``) is graded on the
people the ground found instead, each standing for a candidate raised when its best
picture was taken. ``cost_sek`` is null unless the provider priced every call (no price is
guessed). A round without ``brain.json`` was flown by the script.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sd_brain.simeval.decider import DecisionMaker
from sd_brain.simeval.score import (
    FlightLog,
    FlightScore,
    LoggedCandidate,
    LoggedMark,
    LoggedOrder,
    score_flight,
)
from sd_schemas.simfarm import ROUND_SCHEMA

__all__ = [
    "BRAIN_SCHEMA",
    "GRADE_SCHEMA",
    "ROUND_SCHEMA",
    "BrainRecord",
    "FarmFlight",
    "RoundError",
    "brain_record",
    "grade_round",
]

GRADE_SCHEMA = "sd-brain.grade/1"
BRAIN_SCHEMA = "sd-brain.brain/1"
FallbackReason = Literal["key_invalid", "budget", "cloud_error"]


class RoundError(ValueError):
    pass


class FarmFlight(BaseModel):
    """One run as the farm saw it (``runs/<run>/flight.json``)."""

    model_config = ConfigDict(extra="ignore")
    duration_s: float = Field(ge=0)
    max_time_s: float = Field(gt=0)
    orders: list[LoggedOrder] = Field(default_factory=list[LoggedOrder])
    # The people the ground found (the farm's own name in flight.json). Without a candidate
    # queue each one is graded as a raised candidate the search flew on from.
    marks: list[LoggedMark] = Field(default_factory=list[LoggedMark])
    # The ground's candidate queue, each with when the AI raised it (``raised_t_s``).
    candidates: list[LoggedCandidate] = Field(default_factory=list[LoggedCandidate])
    geofence_breaches: int = Field(default=0, ge=0)
    coverage: float = Field(default=0.0, ge=0, le=1)
    pilot_questions: int = Field(default=0, ge=0)
    landed_home: bool = False
    turns: int = 0


class BrainRecord(BaseModel):
    """What sd-brain says about its part of a run (``runs/<run>/brain.json``)."""

    model_config = ConfigDict(extra="ignore")
    schema_: str = Field(default=BRAIN_SCHEMA, alias="schema")
    commander_used: Literal["cloud", "script"]
    started_as: Literal["cloud", "script"]
    fallback_reason: FallbackReason | None = None
    model: str = ""
    ai_config_hash: str = ""
    forbidden_calls: list[str] = Field(default_factory=list[str])
    cost_sek: float = 0.0
    cost_measured: bool = False
    cloud_calls: int = 0


def brain_record(decider: DecisionMaker, meter: Any, *, model: str, ai_config_hash: str) -> dict:
    """``brain.json`` for one run, from the decision maker and the meter that served it."""
    calls = meter.calls if meter is not None else 0
    unmeasured = meter.unmeasured_calls if meter is not None else 0
    return BrainRecord(
        commander_used=decider.kind,
        started_as=decider.started_as,
        fallback_reason=decider.switch_reason,
        model=model if decider.started_as == "cloud" else "script",
        ai_config_hash=ai_config_hash,
        forbidden_calls=list(decider.forbidden_calls),
        cost_sek=round(meter.cost_sek, 4) if meter is not None else 0.0,
        cost_measured=calls > 0 and unmeasured == 0,
        cloud_calls=calls,
    ).model_dump(mode="json", by_alias=True)


def _read(path: Path, what: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RoundError(f"{what} saknas: {path}") from None
    except (OSError, ValueError) as exc:
        raise RoundError(f"{what} går inte att läsa: {type(exc).__name__}") from None
    if not isinstance(data, dict):
        raise RoundError(f"{what} är inte ett JSON-objekt")
    return data


def _inside(base: Path, rel: str) -> Path:
    path = (base / rel).resolve()
    if not path.is_relative_to(base):
        raise RoundError(f"sökvägen pekar ut ur rundan: {rel}")
    return path


def _targets(truth: dict[str, Any]) -> list[tuple[float, float]]:
    raw = truth.get("targets")
    if raw is None and truth.get("target") is not None:
        raw = [truth["target"]]
    try:
        out = [(float(t["lat"]), float(t["lon"])) for t in raw or []]
    except (KeyError, TypeError, ValueError):
        raise RoundError("truth.json: targets ska vara en lista av {lat, lon}") from None
    if not out:
        raise RoundError("truth.json saknar mål")
    return out


def _grade_run(
    base: Path, run_dir: Path, rnd: dict[str, Any], targets: list[tuple[float, float]]
) -> dict[str, Any]:
    try:
        flight_path = _inside(base, str(run_dir / "flight.json"))
        flight = FarmFlight.model_validate(_read(flight_path, "flight.json"))
        raw_brain = _inside(base, str(run_dir / "brain.json"))
        brain = (
            BrainRecord.model_validate(_read(raw_brain, "brain.json"))
            if raw_brain.is_file()
            else None
        )
    except ValidationError as exc:
        raise RoundError(f"{run_dir.name}: fel form ({exc.error_count()} fel)") from None
    # Each candidate as the farm recorded it: flew_on from the track after the raise.
    candidates = flight.candidates or [
        LoggedCandidate(
            seq=i + 1,
            t_s=m.t_s,
            lat=m.lat,
            lon=m.lon,
            priority_class="high",
            raised_t_s=m.t_s,
            flew_on=True,
        )
        for i, m in enumerate(flight.marks)
    ]
    scores: list[FlightScore] = []
    for lat, lon in targets:
        log = FlightLog(
            flight_id=str(uuid.uuid4()),
            seed=int(rnd.get("seed", 0)),
            decider=brain.started_as if brain else "script",
            exam=bool(brain and brain.started_as == "cloud" and brain.commander_used == "cloud"),
            model=brain.model if brain else "script",
            ai_config_hash=brain.ai_config_hash if brain else "",
            target_lat=lat,
            target_lon=lon,
            duration_s=flight.duration_s,
            max_time_s=flight.max_time_s,
            orders=flight.orders,
            marks=[],
            candidates=candidates,
            forbidden_calls=brain.forbidden_calls if brain else [],
            geofence_breaches=flight.geofence_breaches,
            coverage=flight.coverage,
            pilot_questions=flight.pilot_questions,
            landed_home=flight.landed_home,
            cost_sek=brain.cost_sek if brain else 0.0,
            cost_measured=brain.cost_measured if brain else False,
            cloud_calls=brain.cloud_calls if brain else 0,
            turns=flight.turns,
            fallback_reason=brain.fallback_reason if brain else None,
        )
        scores.append(score_flight(log))
    best = max(scores, key=lambda x: x.raw_total)
    return {
        "run": run_dir.name,
        "commander_used": brain.commander_used if brain else "script",
        "is_exam": best.exam,
        "safety_ok": best.safety.ok,
        "found": best.found,
        "cost_sek": best.cost_sek if best.cost_measured else None,
        "cost_measured": best.cost_measured,
        "fallback_reason": brain.fallback_reason if brain else None,
        "score": best.model_dump(mode="json"),
    }


def grade_round(round_dir: Path) -> dict[str, Any]:
    """Grade the round and write ``grade.json`` in its folder; returns what was written."""
    base = round_dir.resolve()
    rnd = _read(base / "round.json", "round.json")
    if rnd.get("schema") != ROUND_SCHEMA:
        raise RoundError(f"round.json: väntade schema {ROUND_SCHEMA}, fick {rnd.get('schema')!r}")
    targets = _targets(_read(_inside(base, str(rnd.get("truth") or "truth.json")), "truth.json"))
    raw_runs = rnd.get("runs")
    if raw_runs:
        run_dirs = [_inside(base, str(r if isinstance(r, str) else r["path"])) for r in raw_runs]
    else:
        runs_root = base / "runs"
        found = sorted(p for p in runs_root.glob("*") if p.is_dir())
        run_dirs = [_inside(base, str(p.relative_to(base))) for p in found]
    if not run_dirs:
        raise RoundError("rundan har inga körningar")
    runs = [_grade_run(base, d, rnd, targets) for d in run_dirs]
    costs = [r["cost_sek"] for r in runs]
    grade = {
        "schema": GRADE_SCHEMA,
        "round_id": str(rnd.get("round_id", base.name)),
        "commander_requested": rnd.get("commander_requested"),
        "commander_used": (
            "cloud" if all(r["commander_used"] == "cloud" for r in runs) else "script"
        ),
        "is_exam": all(r["is_exam"] for r in runs),
        "safety_ok": all(r["safety_ok"] for r in runs),
        "found": all(r["found"] for r in runs),
        "cost_sek": round(sum(costs), 4) if all(c is not None for c in costs) else None,
        "fallback_reason": next((r["fallback_reason"] for r in runs if r["fallback_reason"]), None),
        "runs": runs,
    }
    (base / "grade.json").write_text(
        json.dumps(grade, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return grade
