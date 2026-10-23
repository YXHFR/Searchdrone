"""What the cloud model gets for one candidate: the fixed prompt, the profile as text, a
context line and the crops, and the answer schema it must fill.

Only an approved profile is ever sent: a proposal the pilot has not approved matches
nothing. With a network and the cloud in use, the whole description goes as text, free
text and health details included, because they say where and how to search; nothing
names who the person is, since the profile has no such field. What the image cannot
show stays on the ground: who approved it, the mission's id, where and when the person
was last seen, who said each detail (``source``) and the age and height bands. Without
an approved profile only "is it a person" is asked (``fields`` is empty).

The static part (the system prompt in ``prompts/cloud_v1.txt``) comes first, so the
provider can cache it. The model sees only ``person``, ``person_p``, ``fields``,
``overall`` and ``explanation_sv``; the code fills the rest of the assessment.
"""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files
from typing import Any, Final, cast

from sd_schemas.search import SearchProfile

__all__ = [
    "PROMPT_ID",
    "UnapprovedProfileError",
    "answer_format",
    "build_messages",
    "context_line",
    "profile_text",
    "system_prompt",
]

PROMPT_ID: Final = "cloud_v1"

# Kept on the ground: who approved it, ids, the place last seen (often a home) and the
# bands; none of it is visible in an image crop.
_GROUND_ONLY: Final = frozenset(
    {
        "approved_by",
        "approved_at",
        "mission_id",
        "profile_id",
        "last_seen",
        "age_band",
        "height_band",
    }
)
# Who said a detail may name a relative or a witness: never sent.
_SOURCE: Final = "source"
_VERDICTS: Final = ["match", "conflict", "not_visible"]


class UnapprovedProfileError(ValueError):
    """A profile no pilot approved: it is never sent anywhere."""


@cache
def system_prompt() -> str:
    return files("sd_search.match").joinpath(f"prompts/{PROMPT_ID}.txt").read_text("utf-8").strip()


def _without_sources(value: Any) -> Any:
    if isinstance(value, dict):
        d = cast("dict[str, Any]", value)
        return {k: _without_sources(v) for k, v in d.items() if k != _SOURCE}
    if isinstance(value, list):
        return [_without_sources(v) for v in cast("list[Any]", value)]
    return value


def profile_text(profile: SearchProfile) -> str:
    """The whole profile as JSON text, with the field key of every assessable field."""
    if not profile.approved:
        raise UnapprovedProfileError("only an approved profile is sent to the cloud model")
    body = _without_sources(profile.model_dump(mode="json", exclude=set(_GROUND_ONLY)))
    body["field_keys"] = profile.field_keys
    return json.dumps(body, ensure_ascii=False, sort_keys=True)


def context_line(agl_m: float | None, px_size: float, n: int) -> str:
    """How the crops were taken. The height above ground is left out when unknown."""
    seen = "seen from above" if agl_m is None else f"seen from above at {agl_m:.0f} m"
    return f"{seen}, person about {px_size:.0f} px, {n} crops from different moments"


def answer_format(field_keys: list[str]) -> dict[str, Any]:
    """``response_format`` for the request: the five fields the model may fill, with one
    verdict per field key of the profile (none without a profile)."""
    fields = {
        "type": "object",
        "properties": {k: {"type": "string", "enum": _VERDICTS} for k in field_keys},
        "required": list(field_keys),
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "person": {"type": "string", "enum": ["yes", "no", "unsure"]},
            "person_p": {"type": "number", "minimum": 0, "maximum": 1},
            "fields": fields,
            "overall": {"type": "number", "minimum": 0, "maximum": 1},
            "explanation_sv": {"type": "string", "maxLength": 300},
        },
        "required": ["person", "person_p", "fields", "overall", "explanation_sv"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {"name": "candidate_assessment", "strict": True, "schema": schema},
    }


def build_messages(
    profile: SearchProfile | None,
    images: list[dict[str, Any]],
    *,
    agl_m: float | None,
    px_size: float,
) -> list[dict[str, Any]]:
    """The chat messages: the fixed system prompt, then the context, the profile (or the
    person question alone) and the crops (content parts from ``CloudImagePolicy``)."""
    lines = [context_line(agl_m, px_size, len(images))]
    if profile is None:
        lines.append(
            "No search profile is approved yet: judge only whether this is a person; "
            "fields is an empty object."
        )
    else:
        lines.append("SÖKPROFIL: " + profile_text(profile))
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": [{"type": "text", "text": "\n".join(lines)}, *images]},
    ]
