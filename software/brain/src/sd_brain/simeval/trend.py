"""``sd-brain scores``: the grade trend per day and model, from the saved flight grades.

Exams only (flights the cloud model decided throughout); script flights are counted on a
line of their own. A day whose mean grade for a model is ``DROP_POINTS`` or more below that
model's previous day, or whose pass rate fell, is flagged so a regression shows at once.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

__all__ = ["DROP_POINTS", "load_scores", "trend_rows", "trend_text"]

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
            "cost_per_flight_sek": round(sum(x.get("cost_sek", 0.0) for x in items) / n, 3),
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


def trend_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "Inga betyg ännu. Kör: uv run sd-brain eval --backend sim"
    head = f"{'dag':<10} {'modell':<34} {'n':>3} {'betyg':>6} {'godk':>5} {'hitta':>5} "
    head += f"{'säk.fel':>7} {'kr/flyg':>8}"
    lines = [head]
    for r in rows:
        flag = "  <- sämre" if r["worse"] else ""
        lines.append(
            f"{r['day']:<10} {r['model'][:34]:<34} {r['flights']:>3} {r['mean_total']:>6.1f} "
            f"{r['pass_rate']:>5.0%} {r['found_rate']:>5.0%} {r['safety_fails']:>7} "
            f"{r['cost_per_flight_sek']:>8.2f}{flag}"
        )
    return "\n".join(lines)
