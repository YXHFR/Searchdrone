"""A small script driver for the mission-flow tests: plays the AI on ``/commander/v1``.

Only for tests. It posts ``CharterProposal`` and ``SkillRequest`` straight to the
commander API with the commander token, the way ``sd-brain`` does, so the ground
station's mission flow can be tested without the AI process. Every order goes through
the ground's gate like the AI's own; ``based_on`` and ``charter_version`` come from the
ground's ``Situation`` just before each order, as ``sd-brain`` takes them.

Script format (YAML)::

    steps:
      - charter: {area: box, alt_max_agl_m: 100}      # an area from the fixture
      - skill: takeoff
        params: {alt_m: 40}
        expect: ALLOW                                  # optional: the gate's answer
      - skill: goto
        params: {at: p1, alt_m: 40}                    # at: a point from the fixture
      - skill: survey_sector
        params: {sector_id: A, alt_m: 40}

The fixture maps names to points (``{"p1": (lat, lon)}``) and areas (GeoJSON polygons).
Approvals, "clear around the drone" and STOP / FLY HOME are never done here: the test
does them as the pilot, through the panel API.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from sd_schemas.base import new_id

__all__ = ["ScriptDriver", "load_steps"]


def load_steps(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list):
        raise ValueError(f"{path.name}: needs a list 'steps'")
    for step in steps:
        if not isinstance(step, dict) or not ({"charter", "skill"} & set(step)):
            raise ValueError(f"{path.name}: every step is a charter or a skill: {step!r}")
    return steps


@dataclass
class ScriptDriver:
    commander: httpx.AsyncClient  # base URL of :7700, with the commander token
    mission_id: str
    fixture: Mapping[str, Any] = field(default_factory=dict[str, Any])
    results: list[httpx.Response] = field(default_factory=list[httpx.Response])

    async def situation(self) -> dict[str, Any]:
        res = await self.commander.get(f"/commander/v1/missions/{self.mission_id}/situation")
        res.raise_for_status()
        return res.json()

    async def charter(self, area: str | dict[str, Any], **values: Any) -> httpx.Response:
        polygon = self.fixture[area] if isinstance(area, str) else area
        situation = await self.situation()
        base = None if situation["charter"] is None else situation["charter"]["version"]
        body = {"base_version": base, "area": polygon, "reason_sv": "Testmanus", **values}
        res = await self.commander.post(
            f"/commander/v1/missions/{self.mission_id}/charter-proposals", json=body
        )
        self.results.append(res)
        return res

    async def order(
        self, skill: str, params: Mapping[str, Any] | None = None, *, command_id: str | None = None
    ) -> httpx.Response:
        p = dict(params or {})
        at = p.pop("at", None)
        if isinstance(at, str):
            p["lat"], p["lon"] = self.fixture[at]
        situation = await self.situation()
        based_on = None
        if situation["snapshot_state_epoch"] is not None:
            based_on = {
                "state_epoch": situation["snapshot_state_epoch"],
                "state_seq": situation["snapshot_state_seq"],
            }
        charter = situation["charter"]
        body = {
            "command_id": command_id or str(new_id()),
            "skill": skill,
            "params": p,
            "rationale_sv": "Testmanus",
            "in_reply_to": None,
            "based_on": based_on,
            "charter_version": None if charter is None else charter["version"],
            "backend": "scripted",
            "ai_config_hash": "script-driver",
        }
        res = await self.commander.post(
            f"/commander/v1/missions/{self.mission_id}/commands", json=body
        )
        self.results.append(res)
        return res

    async def step(self, step: Mapping[str, Any]) -> httpx.Response:
        if "charter" in step:
            spec = dict(step["charter"])
            area = spec.pop("area")
            res = await self.charter(area, **spec)
        else:
            res = await self.order(str(step["skill"]), step.get("params"))
        want = step.get("expect")
        if want is not None:
            got = res.json().get("decision")
            if got != want:
                raise AssertionError(f"{step}: expected {want}, got {got}: {res.text}")
        return res

    async def run(self, path: Path) -> list[httpx.Response]:
        return [await self.step(s) for s in load_steps(path)]
