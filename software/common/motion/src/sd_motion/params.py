"""How each kind of missing person moves, read from ``motion.toml``."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from itertools import pairwise
from typing import Any

# Categories without their own row move like the nearest one that has data.
ALIASES = {"hunter": "hiker", "berry_picker": "gatherer"}
DEFAULT = "despondent"


@dataclass(frozen=True)
class Params:
    speed_mps: float  # on a track; the terrain slows it
    turn45: float  # share of the walkers per minute who turn 45 degrees
    turn_any: float  # share per minute who pick any heading
    k_track: float  # share per minute who turn onto a track
    trap: float  # how much faster walkers stop in shrub and wetland
    mobility: tuple[tuple[float, float], ...]  # (hours, share still moving)

    def __post_init__(self) -> None:
        hours = [h for h, _ in self.mobility]
        shares = [s for _, s in self.mobility]
        if len(hours) < 2 or hours[0] != 0.0 or hours != sorted(set(hours)):
            raise ValueError("mobility needs two or more points, the first at 0 h, rising")
        if any(not 0.0 <= s <= 1.0 for s in shares) or shares != sorted(shares, reverse=True):
            raise ValueError("mobility shares must lie in 0..1 and never rise")

    def still_moving(self, hours: float) -> float:
        """S(t): the share still on the move after ``hours``. Linear from 0 h to the second
        point, linear in log time between later points, constant hazard after the last."""
        pts = self.mobility
        if hours <= 0.0:
            return pts[0][1]
        (h0, s0), (h1, s1) = pts[0], pts[1]
        if hours <= h1:
            return s0 + (s1 - s0) * (hours - h0) / (h1 - h0)
        for (ha, sa), (hb, sb) in pairwise(pts[1:]):
            if hours <= hb:
                return sa + (sb - sa) * math.log(hours / ha) / math.log(hb / ha)
        (ha, sa), (hb, sb) = pts[-2], pts[-1]
        if sb <= 0.0 or sa <= sb:
            return sb
        rate = math.log(sa / sb) / (hb - ha)
        return sb * math.exp(-rate * (hours - hb))


def _from_row(row: dict[str, Any]) -> Params:
    return Params(
        speed_mps=float(row["speed_kmh"]) / 3.6,
        turn45=float(row["turn45"]),
        turn_any=float(row["turn_any"]),
        k_track=float(row["k_track"]),
        trap=float(row["trap"]),
        mobility=tuple((float(h), float(s)) for h, s in row["mobility"]),
    )


def load_all(text: str | None = None) -> dict[str, Params]:
    """Every row of ``motion.toml`` (or of ``text`` in the same form)."""
    if text is None:
        text = files("sd_motion").joinpath("motion.toml").read_text(encoding="utf-8")
    return {name: _from_row(row) for name, row in tomllib.loads(text).items()}


def for_category(category: str) -> Params:
    """The parameters for a subject category (``sd_schemas`` or simulator names)."""
    table = load_all()
    name = ALIASES.get(category, category)
    return table[name] if name in table else table[DEFAULT]
