"""``sd-brain scores``: the grade trend per day and model, from the saved flight grades.

Exams only (flights the cloud model decided throughout); script flights are counted on a
line of their own. A day whose mean grade for a model is ``DROP_POINTS`` or more below that
model's previous day, or whose pass rate fell, is flagged so a regression shows at once.
The cost per flight counts only flights whose cost the provider reported ("ej uppm." when
none did); no price is guessed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

__all__ = [
    "DROP_POINTS",
    "failure_rows",
    "failure_text",
    "load_scores",
    "trend_rows",
    "trend_text",
]

DROP_POINTS = 10.0


def load_scores(folder: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(folder.glob("*/*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and "total" in data and "safety" in data:
            data["_day"] = path.parent.name
            out.append(data)
    return out


def trend_rows(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for s in scores:
        model = s["model"] if s.get("exam") else "script"
        groups[(s["_day"], model)].append(s)
    rows: list[dict[str, Any]] = []
    last: dict[str, dict[str, Any]] = {}
    for (day, model), items in sorted(groups.items()):
        n = len(items)
        row = {
            "day": day,
            "model": model,
            "flights": n,
            "mean_total": round(sum(x["total"] for x in items) / n, 1),
            "pass_rate": round(sum(bool(x["passed"]) for x in items) / n, 2),
            "found_rate": round(sum(bool(x["found"]) for x in items) / n, 2),
            "safety_fails": sum(not x["safety"]["ok"] for x in items),
            "cost_per_flight_sek": _measured_cost(items),
            "worse": False,
        }
        prev = last.get(model)
        if prev is not None:
            row["worse"] = (
                row["mean_total"] <= prev["mean_total"] - DROP_POINTS
                or row["pass_rate"] < prev["pass_rate"]
                or row["safety_fails"] > 0 >= prev["safety_fails"]
            )
        last[model] = row
        rows.append(row)
    return rows


def _measured_cost(items: list[dict[str, Any]]) -> float | None:
    """Mean cost of the flights whose cost was measured; None when none was."""
    priced = [x["cost_sek"] for x in items if x.get("cost_measured")]
    return round(sum(priced) / len(priced), 3) if priced else None


def trend_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "Inga betyg ännu. Kör: uv run sd-brain eval --backend sim"
    head = f"{'dag':<10} {'modell':<34} {'n':>3} {'betyg':>6} {'godk':>5} {'hitta':>5} "
    head += f"{'säk.fel':>7} {'kr/flyg':>8}"
    lines = [head]
    for r in rows:
        flag = "  <- sämre" if r["worse"] else ""
        cost = r["cost_per_flight_sek"]
        cost_txt = f"{cost:>8.2f}" if cost is not None else f"{'ej uppm.':>8}"
        lines.append(
            f"{r['day']:<10} {r['model'][:34]:<34} {r['flights']:>3} {r['mean_total']:>6.1f} "
            f"{r['pass_rate']:>5.0%} {r['found_rate']:>5.0%} {r['safety_fails']:>7} "
            f"{cost_txt}{flag}"
        )
    return "\n".join(lines)


def _what_went_wrong(s: dict[str, Any]) -> list[str]:
    safety = s["safety"]
    out = []
    for key, text in (
        ("rejected_orders", "order som grinden stoppade"),
        ("forbidden_calls", "otillåtet verktyg"),
        ("orders_under_stop", "order under pilotens STOPP"),
        ("geofence_breaches", "utanför geofence"),
    ):
        if safety.get(key):
            out.append(f"{text} ({safety[key]})")
    if not s.get("found"):
        if s.get("candidate_raised"):
            out.append("väntade vid kandidaten nära personen i stället för att söka vidare")
        elif s.get("candidate_near"):
            out.append("lyfte inte kandidaten nära personen till operatören")
        else:
            out.append("hittade inte personen")
    if s.get("coverage", 1.0) < 0.5:
        out.append(f"låg täckning ({s['coverage']:.0%})")
    if s.get("pilot_questions"):
        out.append(f"frågade piloten ({s['pilot_questions']})")
    if s.get("turn_cap_reached"):
        out.append(f"nådde turgränsen ({s['turns']} turer), simulatorn flög hem")
    if not s.get("landed_home"):
        out.append("landade inte hemma")
    return out


def failure_rows(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Failed exams, newest first: where the instructions should be fixed."""
    rows = [
        {
            "day": s["_day"],
            "seed": s["seed"],
            "model": s["model"],
            "total": s["total"],
            "flight_id": s["flight_id"],
            "problems": _what_went_wrong(s),
        }
        for s in scores
        if s.get("exam") and not s.get("passed")
    ]
    return sorted(rows, key=lambda r: (r["day"], r["seed"]), reverse=True)


def failure_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "Inga underkända prov."
    lines = ["Underkända prov (flyg om med: sd-brain eval --backend sim --flights 1 --seed N):"]
    for r in rows:
        problems = "; ".join(r["problems"]) or "för låg poäng"
        lines.append(
            f"{r['day']} seed {r['seed']} {r['model'][:30]} betyg {r['total']:.0f}: {problems}"
        )
    return "\n".join(lines)
