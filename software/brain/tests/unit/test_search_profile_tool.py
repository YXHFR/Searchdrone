"""``propose_search_profile``: the arguments, the call to the ground and what is logged."""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

import pytest
from brain_helpers import REG, make_loop, situation

from sd_brain.backends import LlmCommander
from sd_brain.config import BehaviourConfig
from sd_brain.context import ContextBuilder
from sd_brain.profile_draft import ProfileArgsError, profile_proposal
from sd_brain.runtime.base import ModelReply, RawToolCall, Sampling
from sd_brain.runtime.replay import ReplayRuntime
from sd_brain.tools import PROFILE_TOOL, profile_tool

NOW = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.UTC)
MARKER = "Fiktivmarkör-blåbärsskog-4711"


def _args(**over: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "subject_category": "dementia",
        "garments": [
            {"part": "outer", "kind": "jacka", "colors": ["red"], "certainty": "certain"},
            {"part": "lower", "colors": ["blue"], "certainty": "unsure"},
        ],
        "last_seen": {"pin": "P1", "hours_ago": 2},
        "free_text": MARKER,
    }
    args.update(over)
    return args


def test_pin_and_hours_become_last_seen() -> None:
    s = situation("draft_p1.json")
    p1 = next(p for p in s.pins if p.id == "P1")
    prop = profile_proposal(_args(), s, NOW)
    seen = prop.profile.last_seen
    assert (seen.lat, seen.lon) == (p1.lat, p1.lon)
    assert seen.at_utc == NOW - dt.timedelta(hours=2)
    assert prop.profile.garments[1].certainty == "unsure"


def test_operator_coordinates_are_kept() -> None:
    s = situation("draft_p1.json")
    prop = profile_proposal(
        _args(last_seen={"lat": -35.36, "lon": 149.16, "hours_ago": 0.5}), s, NOW
    )
    assert (prop.profile.last_seen.lat, prop.profile.last_seen.lon) == (-35.36, 149.16)


@pytest.mark.parametrize(
    ("seen", "needle"),
    [
        (None, "ask instead of guessing"),
        ({"hours_ago": 1}, "Ask the operator where"),
        ({"pin": "P9", "hours_ago": 1}, "unknown pin P9"),
        ({"pin": "P1"}, "hours_ago is required"),
        ({"pin": "P1", "hours_ago": 500}, "hours_ago must be"),
    ],
)
def test_missing_place_or_time_is_refused(seen: Any, needle: str) -> None:
    with pytest.raises(ProfileArgsError, match=needle):
        profile_proposal(_args(last_seen=seen), situation("draft_p1.json"), NOW)


def test_error_names_the_field_not_its_value() -> None:
    with pytest.raises(ProfileArgsError) as exc:
        profile_proposal(_args(visual_notes=MARKER * 30), situation("draft_p1.json"), NOW)
    assert "visual_notes" in str(exc.value) and MARKER not in str(exc.value)


def test_schema_offers_only_listed_pins() -> None:
    params = profile_tool(situation("draft_p1.json")).openai["function"]["parameters"]
    seen = params["properties"]["last_seen"]
    assert seen["properties"]["pin"]["enum"] == ["P1"]
    assert seen["required"] == ["pin", "hours_ago"]
    assert "lat" not in seen["properties"]
    assert "at_utc" not in seen["properties"]
    no_pins = profile_tool(situation("draft_p1.json", pins=[])).openai["function"]["parameters"]
    assert "pin" not in no_pins["properties"]["last_seen"]["properties"]
    assert no_pins["properties"]["last_seen"]["required"] == ["lat", "lon", "hours_ago"]
    assert not {"profile_id", "approved_by", "mission_id"} & set(params["properties"])


def _reply(content: str | None, *calls: tuple[str, dict[str, Any]]) -> ModelReply:
    return ModelReply(
        content=content,
        tool_calls=[
            RawToolCall(id=f"c{i}", name=n, arguments_json=json.dumps(a, ensure_ascii=False))
            for i, (n, a) in enumerate(calls)
        ],
        finish_reason="stop",
        engine="cloud",
    )


async def test_loop_proposes_and_logs_no_profile_text(caplog: pytest.LogCaptureFixture) -> None:
    rt = ReplayRuntime(
        [
            _reply(None, (PROFILE_TOOL, _args())),
            _reply("Jag har föreslagit en sökprofil. Piloten godkänner den på kortet."),
        ]
    )
    ctx = ContextBuilder(REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000)
    backend = LlmCommander(rt, ctx, REG, BehaviourConfig(), Sampling(), context_tokens=8192)
    loop, fg, _ = make_loop(backend, "draft_p1.json")
    caplog.set_level(logging.DEBUG)
    loop.handle_event(fg.operator_message("Hon har röd jacka, kanske blå byxor, sågs vid P1"))
    await loop.settle()
    assert len(fg.profiles) == 1
    assert fg.profiles[0].profile.free_text == MARKER
    assert not fg.commands and not fg.proposals
    tool_result = rt.calls[1]["messages"][-1]
    assert tool_result["role"] == "tool" and "proposal_id" in tool_result["content"]
    assert MARKER not in caplog.text
    # The decision log keeps the profile call's field names, never its values.
    calls = [c for d in fg.decisions for c in d.tool_calls if c["name"] == PROFILE_TOOL]
    assert calls and "free_text" in calls[0]["arguments"]
    assert MARKER not in json.dumps([d.tool_calls for d in fg.decisions])


async def test_one_proposal_per_turn() -> None:
    """A second call in the same turn would replace the first on the card: refused."""
    rt = ReplayRuntime(
        [
            _reply(None, (PROFILE_TOOL, _args()), (PROFILE_TOOL, _args(garments=[]))),
            _reply("Föreslagen."),
        ]
    )
    ctx = ContextBuilder(REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000)
    backend = LlmCommander(rt, ctx, REG, BehaviourConfig(), Sampling(), context_tokens=8192)
    loop, fg, _ = make_loop(backend, "draft_p1.json")
    loop.handle_event(fg.operator_message("röd jacka vid P1 för två timmar sedan"))
    await loop.settle()
    assert len(fg.profiles) == 1 and len(fg.profiles[0].profile.garments) == 2
    results = [m for m in rt.calls[1]["messages"] if m["role"] == "tool"]
    assert "already proposed" in results[-1]["content"]


async def test_not_offered_to_a_non_pilot() -> None:
    rt = ReplayRuntime([_reply("Okej.")])
    ctx = ContextBuilder(REG, takeoff_policy="request_then_clearance", prompt_budget_tokens=6000)
    backend = LlmCommander(rt, ctx, REG, BehaviourConfig(), Sampling(), context_tokens=8192)
    loop, fg, _ = make_loop(backend, "draft_p1.json")
    loop.handle_event(fg.operator_message("röd jacka", author_role="system"))
    await loop.settle()
    names = {t["function"]["name"] for t in rt.calls[0]["tools"]}
    assert PROFILE_TOOL not in names
