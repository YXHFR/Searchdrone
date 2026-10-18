"""``just sim-farm-status``: what the farm is doing and has done.

Reads ``build/simfarm/state.json`` (the loop's state, rewritten as it goes) and
``build/simfarm/rounds.jsonl`` (one line per finished round) and prints: running,
paused or stopped and why; pictures and labels kept; rounds per hour over the last six
hours; rounds that are exams for the AI commander and their cost; the upload backlog;
and the latest errors.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from sd_sim import paths
from sd_sim.farm.store import Ledger

WINDOW_H = 6.0


def farm_dir(repo: Path = paths.REPO) -> Path:
    return repo / "build" / "simfarm"


def write_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {**state, "updated_utc": dt.datetime.now(dt.UTC).isoformat()}
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_round(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def read_rounds(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _utc(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text)


def summary(
    base: Path, now: dt.datetime | None = None, alive: Callable[[Any], bool] = _alive
) -> dict[str, Any]:
    now = now or dt.datetime.now(dt.UTC)
    try:
        state = json.loads((base / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    rounds = read_rounds(base / "rounds.jsonl")
    ok = [r for r in rounds if r.get("status") == "ok"]
    failed = [r for r in rounds if r.get("status") != "ok"]
    since = now - dt.timedelta(hours=WINDOW_H)
    recent = [r for r in ok if r.get("finished_utc") and _utc(r["finished_utc"]) >= since]
    exams = [r for r in ok if (r.get("grade") or {}).get("is_exam")]
    cost = sum(float((r.get("grade") or {}).get("cost_sek") or 0.0) for r in exams)
    uploaded = Ledger(base / "uploaded.jsonl").rounds
    running = state.get("state") in ("running", "paused") and alive(state.get("pid"))
    return {
        "state": state.get("state", "never run") if running or not state else "stopped",
        "reason": state.get("reason"),
        "commit": state.get("commit"),
        "rounds_ok": len(ok),
        "rounds_failed": len(failed),
        "images": sum(int(r.get("images", 0)) for r in ok),
        "labels": sum(int(r.get("labels", 0)) for r in ok),
        "rounds_per_hour": round(len(recent) / WINDOW_H, 2),
        "exams": len(exams),
        "exam_cost_sek": round(cost, 2),
        "storage": "R2 (EU)" if state.get("upload") == "r2" else "lokalt",
        "not_uploaded": (
            [r["round_id"] for r in ok if r.get("round_id") not in uploaded]
            if state.get("upload") == "r2"
            else []
        ),
        "streak": state.get("streak", 0),
        "last_errors": [
            {"round_id": r.get("round_id"), "error": r.get("error")} for r in failed[-3:]
        ],
    }


def render(s: Mapping[str, Any]) -> str:
    lines = [f"Simfarmen: {s['state']}" + (f" ({s['reason']})" if s.get("reason") else "")]
    if s.get("commit"):
        lines.append(f"  main: {str(s['commit'])[:12]}")
    lines += [
        f"  lagring: {s['storage']}",
        f"  bilder: {s['images']} med {s['labels']} personrutor",
        f"  varv: {s['rounds_ok']} godkända, {s['rounds_failed']} fel ({s['streak']} fel i rad)",
        f"  varv per timme (senaste {WINDOW_H:g} h): {s['rounds_per_hour']}",
        f"  prov för AI-befälhavaren: {s['exams']}, kostnad {s['exam_cost_sek']} kr",
    ]
    if s["not_uploaded"]:
        lines.append(f"  väntar på uppladdning: {len(s['not_uploaded'])} varv")
    if s["last_errors"]:
        lines.append("  senaste fel:")
        lines += [f"    {e['round_id']}: {e['error']}" for e in s["last_errors"]]
    return "\n".join(lines)


def main(repo: Path = paths.REPO) -> int:
    print(render(summary(farm_dir(repo))))  # noqa: T201 - command-line output
    return 0
