"""The AI's two candidate lines in ``Situation``: the open candidates
(``sd_schemas.candidate_ai.candidates_line``, shared with the simulator exam) and the
confirmed finds.

Read-only, and built from enum-like values only: never from anything typed by a person.
The AI sees the candidates; only a person confirms a find.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import TYPE_CHECKING

from sd_ground.logging import get_logger
from sd_safety.constants import PERSON_KEEPOUT_M
from sd_schemas.candidate_ai import candidates_line
from sd_schemas.search import CandidateCard, CandidateState, SearchProfile

if TYPE_CHECKING:
    from sd_ground.mav.thread import MavSnapshot
    from sd_ground.services.missions import MissionFlow

__all__ = ["candidate_lines", "drone_position", "finds_line"]

log = get_logger("sd_ground.candidate_lines")

_S = CandidateState


def drone_position(mav: MavSnapshot | None) -> tuple[float, float] | None:
    """The drone's (lat, lon) when it is connected with a 3D fix, else None."""
    if mav is None or not mav.connected or mav.telemetry.gps_fix_type < 3:
        return None
    return mav.telemetry.lat, mav.telemetry.lon


def finds_line(cards: list[CandidateCard], confirmed_at: Mapping[str, dt.datetime]) -> str | None:
    """The confirmed finds: "Fynd: K3 BEKRÄFTAD 14:32, cylinder 34 m" each, "Fynd: inga
    bekräftade" without one, or None for a mission without candidates. The cylinder is
    the keep-out around a person (PERSON_KEEPOUT_M + the position's sigma95_m)."""
    if not cards:
        return None
    found = sorted((c for c in cards if c.state is _S.CONFIRMED), key=lambda c: c.seq)
    if not found:
        return "Fynd: inga bekräftade"
    parts: list[str] = []
    for c in found:
        when = confirmed_at.get(str(c.candidate_id))
        # UTC, like every other clock in the AI's text (the chat lines' times).
        clock = "" if when is None else f" {when.astimezone(dt.UTC):%H:%M}"
        parts.append(
            f"K{c.seq} BEKRÄFTAD{clock}, cylinder {round(PERSON_KEEPOUT_M + c.sigma95_m)} m"
        )
    return "Fynd: " + " | ".join(parts)


async def candidate_lines(
    flow: MissionFlow, mission_id: str, mav: MavSnapshot | None
) -> tuple[str | None, str | None]:
    """(candidates_line, finds_line) for the situation. A fault here never takes the rest
    of the situation with it: it is logged and both lines are None."""
    from sd_ground.services import candidates as candidates_service

    try:
        cards = await candidates_service.card_list(flow, mission_id)
        if not cards:
            return None, None
        profiles = {
            p.profile_version: p
            for p in (
                SearchProfile.model_validate(r["body"])
                for r in await flow.repo.search_profile_versions(mission_id)
            )
            if p.approved
        }
        r = candidates_service.repo(flow)
        confirmed_at: dict[str, dt.datetime] = {}
        for c in cards:
            if c.state is _S.CONFIRMED:
                confirms = [
                    d for d in await r.decisions(str(c.candidate_id)) if d["decision"] == "confirm"
                ]
                if confirms:
                    confirmed_at[str(c.candidate_id)] = confirms[-1]["created_at"]
        drone = drone_position(mav)
        return candidates_line(cards, profiles, drone), finds_line(cards, confirmed_at)
    except Exception:
        log.exception("candidates.situation_lines_failed", mission_id=mission_id)
        return None, None
