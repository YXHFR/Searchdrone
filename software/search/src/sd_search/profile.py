"""Where the approved search profile comes from: ``ProfileSource``.

Only an approved version is ever used for matching (E10): a profile without
``approved_by`` and ``approved_at`` is treated as no profile. ``GroundProfileSource``
asks the ground station; ``FileProfileSource`` reads a fixture and is only for the
simulator (``SD_SIM=1``), never on a real search.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Protocol

from sd_schemas.search import SearchProfile
from sd_search.client import GroundError, SearchApi

__all__ = ["FileProfileSource", "GroundProfileSource", "ProfileSource", "approved_only"]

log = logging.getLogger("sd_search.profile")


class ProfileSource(Protocol):
    async def current(self, mission_id: str) -> SearchProfile | None: ...


def approved_only(profile: SearchProfile | None) -> SearchProfile | None:
    if profile is None or profile.approved:
        return profile
    log.warning("ett sökprofilförslag utan godkännande används inte")
    return None


class GroundProfileSource:
    def __init__(self, api: SearchApi) -> None:
        self.api = api

    async def current(self, mission_id: str) -> SearchProfile | None:
        try:
            profile = await self.api.profile(mission_id)
        except GroundError as exc:
            log.warning("sökprofilen gick inte att läsa: %s", exc)
            return None
        if profile is not None and str(profile.mission_id) != mission_id:
            return None
        return approved_only(profile)


class FileProfileSource:
    """A fixed profile from a JSON file, for every mission (simulator only)."""

    def __init__(self, path: Path) -> None:
        if os.environ.get("SD_SIM") != "1":
            raise RuntimeError("en sökprofil från fil används bara i simulatorn (SD_SIM=1)")
        self.profile = SearchProfile.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if not self.profile.approved:
            raise ValueError("sökprofilen i filen är inte godkänd")

    async def current(self, mission_id: str) -> SearchProfile | None:
        return approved_only(
            SearchProfile.model_validate(self.profile.model_dump() | {"mission_id": mission_id})
        )
