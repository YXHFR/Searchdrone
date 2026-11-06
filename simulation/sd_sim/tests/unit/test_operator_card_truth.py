"""The scripted operator judges a card by what its picture was drawn from where the rig
knows it, and by the thing at the card's position otherwise."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import httpx

from sd_sim.operator import ScriptedOperator

SIM = {"SD_SIM": "1"}
MID = "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a21"


def _card(cid: str, seq: int, thumb: str | None) -> dict[str, Any]:
    return {"candidate_id": cid, "seq": seq, "rev": 1, "state": "ready", "lat": -35.0,
            "lon": 149.0, "priority": 0.7, "priority_class": "high",
            "thumb_detection_id": thumb, "fields": {}}  # fmt: skip


def _run(
    cards: list[dict[str, Any]], sources: Mapping[str, tuple[str | None, bool]]
) -> list[tuple[str, str, str | None, str]]:
    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path.removeprefix("/api/v1")
        if path == f"/missions/{MID}/candidates":
            return httpx.Response(200, json={"items": cards, "counts": {}})
        if path.endswith("/approvals") or path.endswith("/commands"):
            return httpx.Response(200, json=[])
        if path.endswith("/view"):
            return httpx.Response(201, json={})
        c = next(c for c in cards if c["candidate_id"] == path.split("/")[2])
        if path.endswith("/decision"):
            body = json.loads(req.content)
            c["state"] = "confirmed" if body["decision"] == "confirm" else "rejected"
        return httpx.Response(200, json={"card": c})

    async def no_wait(_: float) -> None:
        return None

    op = ScriptedOperator(
        httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://g"),
        MID,
        # At every card's position the nearest thing is the backpack 4 m from the person.
        truth=lambda lat, lon: ("backpack-0-0", False),
        card_truth=lambda c: sources.get(str(c.get("thumb_detection_id"))),
        headers={},
        env=SIM,
        sleep=no_wait,
    )

    async def work() -> None:
        while await op.review_once() is not None:
            pass

    asyncio.run(work())
    return [(d.candidate_id, d.decision, d.thing, d.by) for d in op.decisions]


def test_a_card_whose_picture_shows_the_person_is_confirmed_though_nearer_a_decoy() -> None:
    cards = [_card("bag", 1, "d1"), _card("person", 2, "d2")]
    sources = {"d1": ("backpack-0-0", False), "d2": ("target", True)}
    assert _run(cards, sources) == [
        ("bag", "reject", "backpack-0-0", "picture"),
        ("person", "confirm", "target", "picture"),
    ]


def test_a_card_whose_source_is_unknown_goes_by_the_thing_at_its_position() -> None:
    cards = [_card("old", 1, "d9"), _card("nothumb", 2, None)]
    assert _run(cards, {}) == [
        ("old", "reject", "backpack-0-0", "position"),
        ("nothumb", "reject", "backpack-0-0", "position"),
    ]


def test_a_false_positive_on_plain_ground_is_nothing() -> None:
    assert _run([_card("ground", 1, "d1")], {"d1": (None, False)}) == [
        ("ground", "reject", None, "picture")
    ]
