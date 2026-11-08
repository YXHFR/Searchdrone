"""The numbers every closer-look run reports (spec "Titta närmare" §16.3).

``InspectMeter`` takes the vehicle's true position through the run (the facit, from the
world: ``sd_sim.obstacles.Scene``, never from the map) and the floor that held at each
moment, and gives:

* ``min_clearance_m``: the shortest distance from the drone to any crown, line or the
  ground;
* ``min_target_horizontal_m``: the shortest distance sideways to the target person;
* ``floor_breach_m``: how far below the floor in force the drone was at most (0 when
  never below);
* ``lowest_above_ground_m``: the lowest height above the ground under it;
* ``inspect_duration_s``: from the first to the last sample;
* ``target_px_first_step`` / ``target_px_last_step`` and ``outcome``: from the run's
  reports, passed through.

Heights are the world's z (metres above home on the ground).

``write_report`` puts a run's numbers in its folder: ``inspect_metrics.json`` and a
section with one row per run in ``search_report.md`` (the run's Swedish summary, which a
search run writes too).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sd_sim.obstacles import Scene

__all__ = ["METRICS_FILE", "REPORT_FILE", "InspectMeter", "InspectMetrics", "write_report"]

METRICS_FILE = "inspect_metrics.json"
REPORT_FILE = "search_report.md"  # = sd_sim.eval_search.REPORT
_HEADING = "## Titta närmare"
_COLUMNS = (
    "| Scenario | Frö | Utfall | Minsta frigång (m) | Under golvet (m) | Närmast i sidled (m) "
    "| Lägst över mark (m) | Tid (s) | Pixlar steg 0 | Pixlar sista steg |\n"
    "|---|---|---|---|---|---|---|---|---|---|\n"
)


@dataclass(frozen=True)
class InspectMetrics:
    min_clearance_m: float
    min_target_horizontal_m: float
    floor_breach_m: float
    lowest_above_ground_m: float
    inspect_duration_s: float
    target_px_first_step: dict[str, float] = field(default_factory=dict[str, float])
    target_px_last_step: dict[str, float] = field(default_factory=dict[str, float])
    outcome: str = ""


class InspectMeter:
    """Samples in, ``InspectMetrics`` out."""

    def __init__(self, scene: Scene, target: tuple[float, float]) -> None:
        self.scene, self.target = scene, target
        self._t: list[float] = []
        self._clearance = math.inf
        self._sideways = math.inf
        self._breach = 0.0
        self._lowest = math.inf

    def sample(self, t: float, east: float, north: float, z: float, floor_z: float | None) -> None:
        """The vehicle at ``t`` (s), and the floor in force then (``None``: none)."""
        self._t.append(t)
        self._clearance = min(self._clearance, self.scene.clearance(east, north, z))
        self._sideways = min(
            self._sideways, math.hypot(east - self.target[0], north - self.target[1])
        )
        if floor_z is not None:
            self._breach = max(self._breach, floor_z - z)
        self._lowest = min(self._lowest, z - self.scene.ground.at1(east, north))

    def result(
        self,
        *,
        outcome: str = "",
        px_first: dict[str, float] | None = None,
        px_last: dict[str, float] | None = None,
    ) -> InspectMetrics:
        if not self._t:
            raise ValueError("inga prov: titten har inte flugits")
        return InspectMetrics(
            min_clearance_m=round(self._clearance, 3),
            min_target_horizontal_m=round(self._sideways, 3),
            floor_breach_m=round(self._breach, 3),
            lowest_above_ground_m=round(self._lowest, 3),
            inspect_duration_s=round(max(self._t) - min(self._t), 3),
            target_px_first_step=dict(px_first or {}),
            target_px_last_step=dict(px_last or {}),
            outcome=outcome,
        )


def _px(values: dict[str, float]) -> str:
    return ", ".join(f"{cam} {px:.0f}" for cam, px in sorted(values.items())) or "-"


def _num(value: float) -> str:
    return "-" if math.isinf(value) else f"{value:.1f}".replace(".", ",")


def write_report(run_dir: Path, scenario: str, seed: int, m: InspectMetrics) -> Path:
    """The run's numbers in ``run_dir``: ``inspect_metrics.json`` (every run of the folder,
    in order) and a row in the closer-look section of ``search_report.md``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    data = run_dir / METRICS_FILE
    runs = json.loads(data.read_text(encoding="utf-8")) if data.is_file() else []
    runs.append({"scenario": scenario, "seed": seed, **asdict(m)})
    data.write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = run_dir / REPORT_FILE
    text = report.read_text(encoding="utf-8") if report.is_file() else ""
    if _HEADING not in text:
        text += ("\n" if text else "") + f"{_HEADING}\n\n{_COLUMNS}"
    row = (
        f"| {scenario} | {seed} | {m.outcome or '-'} | {_num(m.min_clearance_m)} "
        f"| {_num(m.floor_breach_m)} | {_num(m.min_target_horizontal_m)} "
        f"| {_num(m.lowest_above_ground_m)} | {_num(m.inspect_duration_s)} "
        f"| {_px(m.target_px_first_step)} | {_px(m.target_px_last_step)} |\n"
    )
    head, _, rest = text.partition(_HEADING)
    # The section's rows end at the next heading, or at the end of the file.
    body, sep, tail = rest.partition("\n## ")
    text = head + _HEADING + body.rstrip("\n") + "\n" + row + (sep + tail if sep else "")
    report.write_text(text, encoding="utf-8")
    return report
