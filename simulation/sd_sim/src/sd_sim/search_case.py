"""A search run whose invariants broke, as numbers only, so it can be run again as a unit
test without the SITL.

``write_case`` is called by the search scenarios when a search invariant breaks. It puts
``search_case.json`` in the run's folder, and CI uploads that file as an artifact. The
file holds the flown track (time, position, height, ground speed, mode, armed), the
cylinders the onboard executor held (``SeenKeepout``), the candidate rows E1 reads,
the scenario, the seed and the broken invariants. It holds no pictures, no map images
and no search profile text: only the scenario's numbers and geometry, rounded so that
a long flight stays well under a megabyte.

``load_case`` reads the file back. ``replay`` runs the invariants that can be checked
from the file alone (E1 and E8) and returns what they find. E2 is checked over repeated
looks during the flight, so it cannot be run again from the file; its messages are kept
in ``broken``.

``python -m sd_sim.search_case <file>...`` prints, for each file, the broken invariants
recorded in CI and those ``replay`` finds now. It exits 1 if any are found now.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from sd_sim.invariants import Sample
from sd_sim.search_invariants import SeenKeepout, keepout_never_entered, no_confirm_without_human

__all__ = ["CASE", "Case", "load_case", "main", "replay", "write_case"]

CASE: Final = "search_case.json"
VERSION: Final = 1
# Columns of a track row, in order.
TRACK_COLUMNS: Final = ("t", "lat", "lon", "alt_m", "groundspeed_ms", "mode", "armed")


@dataclass
class Case:
    """One run's numbers, as ``search_case.json`` holds them."""

    scenario: str
    seed: int
    broken: list[str]
    track: list[Sample]
    keepouts: list[SeenKeepout]
    # E1's inputs: candidate revisions, decisions, views and the detections each
    # candidate observed (``no_confirm_without_human``); None when not recorded.
    e1: dict[str, Any] | None = None
    params: dict[str, Any] = field(default_factory=dict[str, Any])


def _time(value: float) -> float | None:
    """A monotonic time for JSON: an open end (inf) as null."""
    return None if math.isinf(value) else round(value, 3)


def _row(s: Sample) -> list[Any]:
    # 1e-7 degrees is about 1 cm: finer than the track's own noise.
    return [
        round(s.t, 3),
        round(s.lat, 7),
        round(s.lon, 7),
        round(s.alt_m, 2),
        round(s.groundspeed_ms, 2),
        s.mode,
        s.armed,
    ]


def _keepout(k: SeenKeepout) -> dict[str, Any]:
    return {
        "keepout_id": k.keepout_id,
        "lat": round(k.lat, 7),
        "lon": round(k.lon, 7),
        "radius_m": round(k.radius_m, 3),
        "since": _time(k.since),
        "until": _time(k.until),
        "came": None if k.came is None else _time(k.came),
        "passable": k.passable,
    }


def _plain(value: Any) -> Any:
    if isinstance(value, dt.datetime):
        return value.isoformat()
    return str(value)


def write_case(
    run_dir: Path,
    *,
    scenario: str,
    seed: int,
    broken: Sequence[str],
    track: Iterable[Sample],
    keepouts: Iterable[SeenKeepout],
    e1: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
) -> Path:
    """``search_case.json`` in ``run_dir`` (see the module's docstring)."""
    body = {
        "version": VERSION,
        "scenario": scenario,
        "seed": seed,
        "broken": list(broken),
        "params": dict(params or {}),
        "keepouts": [_keepout(k) for k in keepouts],
        "track_columns": list(TRACK_COLUMNS),
        "track": [_row(s) for s in track],
        "e1": None if e1 is None else dict(e1),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / CASE
    path.write_text(
        json.dumps(body, ensure_ascii=False, separators=(",", ":"), default=_plain) + "\n",
        encoding="utf-8",
    )
    return path


def _inf(value: float | None) -> float:
    return math.inf if value is None else float(value)


def load_case(path: Path) -> Case:
    """A ``search_case.json`` read back."""
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("version") != VERSION:
        raise ValueError(f"{path}: version {body.get('version')}, expected {VERSION}")
    cols = body["track_columns"]
    if tuple(cols) != TRACK_COLUMNS:
        raise ValueError(f"{path}: track columns {cols}")
    track = [Sample(*row) for row in body["track"]]
    keepouts = [
        SeenKeepout(
            k["keepout_id"],
            k["lat"],
            k["lon"],
            k["radius_m"],
            _inf(k["since"]),
            _inf(k["until"]),
            came=None if k["came"] is None else float(k["came"]),
            passable=bool(k["passable"]),
        )
        for k in body["keepouts"]
    ]
    return Case(
        scenario=body["scenario"],
        seed=int(body["seed"]),
        broken=list(body["broken"]),
        track=track,
        keepouts=keepouts,
        e1=body.get("e1"),
        params=dict(body.get("params") or {}),
    )


def replay(case: Case) -> list[str]:
    """E1 (when recorded) and E8 over the case's numbers."""
    broken: list[str] = []
    if case.e1 is not None:
        broken += no_confirm_without_human(
            case.e1["revisions"], case.e1["decisions"], case.e1["views"], case.e1["observed"]
        )
    broken += keepout_never_entered(case.keepouts, case.track)
    return broken


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sd_sim.search_case")
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args(argv)
    found = False
    for path in args.files:
        case = load_case(path)
        now = replay(case)
        found = found or bool(now)
        sys.stdout.write(f"{path}: {case.scenario} frö {case.seed}\n")
        for b in case.broken:
            sys.stdout.write(f"  i CI: {b}\n")
        for b in now:
            sys.stdout.write(f"  nu:   {b}\n")
        if not now:
            sys.stdout.write("  nu:   inga brott\n")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
