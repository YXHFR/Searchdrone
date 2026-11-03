"""The scripted operator: works the queue in order, views before confirming, decides
by the simulator's truth, never on a card that is not open, only in the simulator."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from sd_sim.operator import OperatorError, ScriptedOperator

SIM = {"SD_SIM": "1"}
MID = "0192f4a1-7c2e-7d10-9a3b-5e1f0c2d8a21"


def card(cid: str, seq: int, state: str, lat: float, cls: str = "high", rev: int = 1) -> dict:
    return {"candidate_id": cid, "seq": seq, "rev": rev, "state": state, "lat": lat, "lon": 149.0,
            "priority": 0.7, "priority_class": cls, "thumb_detection_id": f"d{seq}",
            "fields": {"garment:0": "match"}}  # fmt: skip


def order(oid: str, skill: str, issuer_kind: str = "brain") -> dict[str, Any]:
    return {"id": oid, "skill": skill, "issuer_kind": issuer_kind, "status": "pending_approval"}


def order_card(oid: str, kind: str = "command") -> dict[str, Any]:
    return {"id": f"card-{oid}", "kind": kind, "subject_id": oid, "subject_version": 1,
            "status": "pending", "required_confirmations": []}  # fmt: skip


class Panel:
    def __init__(
        self,
        cards: list[dict[str, Any]],
        orders: list[dict[str, Any]] | None = None,
        order_cards: list[dict[str, Any]] | None = None,
    ) -> None:
        self.cards = cards
        self.orders = orders or []
        self.order_cards = order_cards or []
        self.gone: set[str] = set()  # order cards answered elsewhere: 409 SUPERSEDED
        self.calls: list[tuple[str, str, Any]] = []
        self.superseded_once = False

    def handler(self, req: httpx.Request) -> httpx.Response:
        body: Any = json.loads(req.content) if req.content else None
        path = req.url.path.removeprefix("/api/v1")
        self.calls.append((req.method, path, body))
        if path == f"/missions/{MID}/candidates":
            return httpx.Response(200, json={"items": self.cards, "counts": {}})
        if path == f"/missions/{MID}/approvals":
            return httpx.Response(
                200, json=[a for a in self.order_cards if a["status"] == "pending"]
            )
        if path == f"/missions/{MID}/commands":
            return httpx.Response(200, json=self.orders)
        if path.startswith("/approvals/"):
            a = next(a for a in self.order_cards if a["id"] == path.split("/")[2])
            if a["id"] in self.gone:
                return httpx.Response(409, json={"code": "SUPERSEDED"})
            a["status"] = "approved"
            return httpx.Response(200, json=a)
        if path == f"/missions/{MID}/search-profile" and req.method == "GET":
            return httpx.Response(200, json={"approved": None, "proposal_id": "p1"})
        if path == f"/missions/{MID}/search-profile":
            return httpx.Response(201, json={"profile_version": 1})
        if path.endswith("/view"):
            return httpx.Response(201, json={})
        cid = path.split("/")[2]
        c = next(c for c in self.cards if c["candidate_id"] == cid)
        if path.endswith("/decision"):
            if c.get("supersede") and not self.superseded_once:
                self.superseded_once = True
                c["rev"] += 1
                return httpx.Response(409, json={"code": "SUPERSEDED"})
            c["state"] = "confirmed" if body["decision"] == "confirm" else "rejected"
            return httpx.Response(200, json={"card": c})
        return httpx.Response(200, json={"card": c})


def make(panel: Panel, **kw: Any) -> ScriptedOperator:
    client = httpx.AsyncClient(transport=httpx.MockTransport(panel.handler), base_url="http://g")

    async def no_wait(_: float) -> None:
        return None

    return ScriptedOperator(
        client, MID, truth=lambda lat, lon: ("target", True) if lat > 0 else ("rock-0-1", False),
        headers={}, env=SIM, sleep=no_wait, **kw,
    )  # fmt: skip


def test_only_in_the_simulator() -> None:
    with pytest.raises(RuntimeError, match="SD_SIM"):
        ScriptedOperator(httpx.AsyncClient(), MID, truth=lambda a, b: (None, False), headers={},
                         env={})  # fmt: skip


def test_rejects_the_wrong_card_first_then_confirms_after_a_view() -> None:
    panel = Panel([card("a", 1, "ready", -1.0), card("b", 2, "new", 1.0),
                   card("c", 3, "assessing", 1.0, cls="low")])  # fmt: skip
    op = make(panel)
    asyncio.run(op.run(asyncio.Event()))
    assert [(d.candidate_id, d.decision) for d in op.decisions] == [
        ("a", "reject"),
        ("c", "confirm"),
    ]
    assert op.confirmed is not None and op.confirmed.thing == "target"
    decided = [c for c in panel.calls if c[1].endswith("/decision")]
    assert all(c[2]["subject_rev"] == 1 for c in decided)
    views = [i for i, c in enumerate(panel.calls) if c[1].endswith("/view")]
    confirm = next(
        i for i, c in enumerate(panel.calls) if c[2] and c[2].get("decision") == "confirm"
    )
    assert views and max(v for v in views if v < confirm) < confirm
    assert not any(c[1] == "/candidates/b/decision" for c in panel.calls)  # "new": not open


def test_a_superseded_card_is_read_again() -> None:
    c = card("a", 1, "ready", 1.0)
    c["supersede"] = True
    panel = Panel([c])
    op = make(panel)
    d = asyncio.run(op.review_once())
    assert d is not None and d.decision == "confirm"
    revs = [x[2]["subject_rev"] for x in panel.calls if x[1].endswith("/decision")]
    assert revs == [1, 2]


def test_notes_the_targets_rank_when_ready_and_a_low_class() -> None:
    panel = Panel([card("a", 1, "ready", -1.0), card("b", 2, "ready", 1.0, cls="low")])
    op = make(panel)
    asyncio.run(op.review_once())
    assert op.target_first_rank == 2 and op.target_ever_low


def test_approves_only_the_ais_closer_looks_then_reviews() -> None:
    orders = [order("look", "inspect_detection"), order("path", "follow_path"),
              order("mine", "inspect_detection", issuer_kind="operator"),
              order("late", "inspect_detection")]  # fmt: skip
    panel = Panel([card("a", 1, "ready", 1.0)], orders, [order_card(o["id"]) for o in orders])
    panel.gone.add("card-late")  # withdrawn by FLY HOME while the operator looked
    op = make(panel)
    asyncio.run(op.run(asyncio.Event()))
    assert op.looks_approved == ["look"]
    posted = [c for c in panel.calls if c[0] == "POST" and c[1].startswith("/approvals/")]
    assert [c[1] for c in posted] == [
        "/approvals/card-look/decision",
        "/approvals/card-late/decision",
    ]
    assert posted[0][2] == {"decision": "approve", "subject_version": 1, "confirmations": []}
    first_review = next(i for i, c in enumerate(panel.calls) if c[1].endswith("/candidates"))
    assert panel.calls.index(posted[0]) < first_review  # the look before the queue
    assert op.confirmed is not None


def test_another_refusal_of_a_look_is_an_error() -> None:
    panel = Panel([], [order("look", "inspect_detection")], [order_card("look")])

    def refuse(req: httpx.Request) -> httpx.Response:
        if req.url.path.startswith("/api/v1/approvals/"):
            return httpx.Response(409, json={"code": "CLEARANCE_REQUIRED"})
        return panel.handler(req)

    client = httpx.AsyncClient(transport=httpx.MockTransport(refuse), base_url="http://g")
    op = ScriptedOperator(client, MID, truth=lambda a, b: (None, False), headers={}, env=SIM)
    with pytest.raises(OperatorError, match="CLEARANCE_REQUIRED"):
        asyncio.run(op.approve_looks())


def test_approves_the_profile_against_what_the_card_showed() -> None:
    panel = Panel([])
    op = make(panel)
    asyncio.run(op.approve_profile({"subject_category": "hiker"}))
    post = next(c for c in panel.calls if c[0] == "POST")
    assert post[2] == {"profile": {"subject_category": "hiker"}, "base_version": 0,
                       "proposal_id": "p1"}  # fmt: skip
