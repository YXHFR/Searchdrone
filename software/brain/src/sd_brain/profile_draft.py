"""The ``propose_search_profile`` tool's arguments as a ``SearchProfileProposal``.

The model never sees the clock or coordinates it was not given: it names the last-seen
point as a map pin (or the coordinates the operator wrote) and the time as hours ago.
This module turns that into the profile's ``last_seen``. An error goes back to the model
as the tool's result, so it can ask the operator or call again.

The arguments may hold health details: they are never logged here, and an error names
the field, never its value.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import ValidationError

from sd_brain.tools import MAX_SEEN_HOURS_AGO, AdaptError, pin_point
from sd_schemas.commander import Situation
from sd_schemas.search import SearchProfileDraft, SearchProfileProposal

__all__ = ["ProfileArgsError", "profile_proposal"]


class ProfileArgsError(ValueError):
    """Arguments that make no profile; the text is for the model and names no value."""


def _last_seen(raw: Any, s: Situation, now: dt.datetime) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ProfileArgsError(
            "last_seen is required: where (a map pin) and how many hours ago. "
            "If the operator did not say where, ask instead of guessing."
        )
    seen: dict[str, Any] = dict(raw)  # pyright: ignore[reportUnknownArgumentType]
    hours = seen.pop("hours_ago", None)
    if isinstance(hours, bool) or not isinstance(hours, int | float):
        raise ProfileArgsError("last_seen.hours_ago is required (a number of hours)")
    if not 0 <= float(hours) <= MAX_SEEN_HOURS_AGO:
        raise ProfileArgsError(f"last_seen.hours_ago must be 0 to {MAX_SEEN_HOURS_AGO:g}")
    pin = seen.pop("pin", None)
    if pin is not None:
        try:
            seen["lat"], seen["lon"] = pin_point(s, str(pin))
        except AdaptError as exc:
            raise ProfileArgsError(str(exc)) from None
    elif "lat" not in seen or "lon" not in seen:
        known = ", ".join(p.id for p in s.pins) or "none"
        raise ProfileArgsError(
            f"last_seen needs a map pin (known: {known}) or the coordinates the operator "
            "wrote. Ask the operator where the person was last seen."
        )
    seen["at_utc"] = (now - dt.timedelta(hours=float(hours))).isoformat()
    return seen


def profile_proposal(args: dict[str, Any], s: Situation, now: dt.datetime) -> SearchProfileProposal:
    """The proposal the ground stores; raises :class:`ProfileArgsError`."""
    body = dict(args)
    body["last_seen"] = _last_seen(body.get("last_seen"), s, now)
    try:
        draft = SearchProfileDraft.model_validate(body)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "profile"
        raise ProfileArgsError(f"invalid {where}: {first['msg']}") from None
    return SearchProfileProposal(profile=draft)
