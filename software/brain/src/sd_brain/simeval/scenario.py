"""Random search scenarios with an answer key, around the public SITL site CMAC.

A scenario is a charter (a rectangle around home split into sectors A, B, …), a drone
hovering over home, a missing person somewhere in the area (the answer key, never shown
to the decider), a few false detections, how much the trees hide, and sometimes a STOP
from the pilot partway through. The same seed always gives the same scenario.
"""

from __future__ import annotations

import copy
import json
import random
from importlib import resources
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sd_brain.geo import from_enu

__all__ = ["OPENING_LINES", "SimScenario", "make_scenario"]

HOME = (-35.363261, 149.16523)
OPENING_LINES = (
    "En person saknas i området. Sök igenom sektorerna och säg till om du ser något.",
    "Vi letar efter en försvunnen vandrare. Sök av hela ramen, sektor för sektor.",
    "Sök efter en person i området. Markera på kartan om du hittar något.",
    "Någon har gått vilse här. Börja söka, du väljer ordning på sektorerna.",
)


class SimScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int
    situation: dict[str, Any]
    opening: str
    target_east_north: tuple[float, float]  # the answer key, metres from home
    false_alarms_at_s: list[float] = Field(default_factory=list[float])
    stop_at_s: float | None = None
    stop_hold_s: float = 20.0
    visibility: float = Field(default=1.0, gt=0, le=1)  # 1 = open ground, low = dense forest
    max_time_s: float = 1500.0

    @property
    def target_latlon(self) -> tuple[float, float]:
        return from_enu(*self.target_east_north, *HOME)


def _rect(e0: float, n0: float, e1: float, n1: float) -> dict[str, Any]:
    corners = [(e0, n0), (e1, n0), (e1, n1), (e0, n1), (e0, n0)]
    ring = []
    for e, n in corners:
        lat, lon = from_enu(e, n, *HOME)
        ring.append([round(lon, 7), round(lat, 7)])
    return {"type": "Polygon", "coordinates": [ring]}


def _base() -> dict[str, Any]:
    raw = (
        resources.files("sd_brain.evals")
        .joinpath("fixtures/airborne_survey.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(raw)["situation"]


def make_scenario(seed: int) -> SimScenario:
    rnd = random.Random(seed)  # noqa: S311  # a reproducible scenario, not crypto
    width = rnd.uniform(300, 500)
    height = rnd.uniform(200, 360)
    # Home (the pilot) lies inside the area, as the gate requires (HOME_OUTSIDE_AREA).
    e0 = rnd.uniform(30 - width, -30)
    n0 = rnd.uniform(-80, -20)
    n_sectors = rnd.randint(2, 4)
    sit = copy.deepcopy(_base())
    ch = sit["charter"]
    ch["area"] = _rect(e0, n0, e0 + width, n0 + height)
    step = width / n_sectors
    names = "ABCD"[:n_sectors]
    ch["sectors"] = [
        {
            "sector_id": name,
            "name": name,
            "area": _rect(e0 + i * step, n0, e0 + (i + 1) * step, n0 + height),
            "order_idx": i,
        }
        for i, name in enumerate(names)
    ]
    # Stands for a range the pilot raised for this mission (the default is 500 m); the
    # scenario rectangles reach a little beyond the default.
    ch["max_range_from_pilot_m"] = sit["limits"]["max_range_from_pilot_m"] = 600.0
    sit["sectors"] = [
        {"sector_id": n, "name": n, "status": "pending", "progress": 0} for n in names
    ]
    sit["active_command"] = None
    sit["pins"] = []
    battery = rnd.uniform(85, 100)
    sit["vehicle"].update(
        lat=HOME[0],
        lon=HOME[1],
        alt_rel_home_m=40.0,
        executor_state="HOLD",
        battery_pct=round(battery, 1),
        battery_to_reserve_s=int((battery - 15) / 0.065),
        dist_to_pilot_m=0.0,
    )
    target = (rnd.uniform(e0 + 10, e0 + width - 10), rnd.uniform(n0 + 10, n0 + height - 10))
    false_alarms = sorted(rnd.uniform(60, 900) for _ in range(rnd.randint(0, 2)))
    stop_at = rnd.uniform(60, 400) if rnd.random() < 0.35 else None
    visibility = rnd.uniform(0.3, 1.0)
    opening = rnd.choice(OPENING_LINES)
    if visibility < 0.5:
        opening += " Det är tät skog, så sikten uppifrån är dålig."
    return SimScenario(
        seed=seed,
        situation=sit,
        opening=opening,
        target_east_north=target,
        false_alarms_at_s=false_alarms,
        stop_at_s=stop_at,
        visibility=visibility,
    )
