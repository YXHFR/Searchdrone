"""``ScriptedOperator``: the person at the panel in the search scenarios, played by a script.

Only a person confirms a find. In the simulator's search scenarios this class stands in
for that person, through the panel API with the test rig's pilot token (the ground only
lets a CLI token decide on the test rig): it approves the search profile as the card
would send it, then works the candidate queue the way the panel shows it, highest
priority first. For each open card it waits 5-20 s (looking at the pictures), opens the
card's picture in large size, and confirms the card if the simulator's truth says it is
the missing person, otherwise rejects it. The truth is what the picture shows (the thing
its detection came from) where the rig knows it, otherwise the thing at the card. It
never decides on a card that is not open (new, merged, confirmed, rejected). The AI only
proposes a closer look (``inspect_detection`` is an approval card); the operator
approves each such card it finds waiting, before it turns to the queue.

It refuses to start outside the simulator (``SD_SIM=1``) and is never part of the
product: the AI has no way to call it, and it uses the same routes a person's panel
does, so the ground's checks (a person's role, a large view before a confirmation, the
card's revision) apply to it as to anyone.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

__all__ = ["APPROVED_LOOKS", "OPEN_STATES", "Decision", "ScriptedOperator"]

OPEN_STATES = frozenset({"ready", "assessing", "inspecting"})
# The AI's order cards the operator approves: a closer look at a candidate.
APPROVED_LOOKS = frozenset({"inspect_detection"})
DELAY_S = (5.0, 20.0)
POLL_S = 1.0
Judged = Literal["picture", "position"]


@dataclass(frozen=True)
class Decision:
    """One decision, for the run's record."""

    t: float  # monotonic seconds
    candidate_id: str
    seq: int
    decision: Literal["confirm", "reject"]
    target: bool
    thing: str | None  # the world's thing at the card (the simulator's truth), if any
    rank: int  # the card's place in the queue when the operator came to it (1 = top)
    priority_class: str
    fields: Mapping[str, Any]
    # How the truth was found: from the card's picture (what its detection was drawn
    # from) or from the thing at the card's position.
    by: Judged = "position"


@dataclass
class ScriptedOperator:
    panel: httpx.AsyncClient
    mission_id: str
    # The simulator's truth at a card's position: the thing there (or None) and whether
    # it is the missing person.
    truth: Callable[[float, float], tuple[str | None, bool]]
    headers: Mapping[str, str]
    # The truth for a card from what its picture shows (the thing its thumbnail's
    # detection came from), ``None`` where that is unknown: then ``truth`` at the card's
    # position decides.
    card_truth: Callable[[Mapping[str, Any]], tuple[str | None, bool] | None] | None = None
    # Whether the world's thing with this id is a person (the simulator's truth). With it, a
    # rejection of nothing or of a thing that is not a person says so ("Ingen person här",
    # ``reason`` "not_a_person": the cylinder is released); a rejected person, or any
    # rejection without it, is a plain one ("Inte hen": the cylinder stays).
    is_person: Callable[[str], bool] | None = None
    seed: int = 0
    delay_s: tuple[float, float] = DELAY_S
    stop_after_confirm: bool = True
    env: Mapping[str, str] | None = None
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep
    monotonic: Callable[[], float] = time.monotonic
    decisions: list[Decision] = field(default_factory=list[Decision])
    looks_approved: list[str] = field(default_factory=list[str])  # order ids
    target_first_rank: int | None = None  # the target's place when its card was first ready
    target_ever_low: bool = False  # the target's card, assessed, was in the low class
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        if (os.environ if self.env is None else self.env).get("SD_SIM") != "1":
            raise RuntimeError("den skriptade operatören finns bara i simulatorn (SD_SIM=1)")
        self._rng = random.Random(f"operator:{self.seed}")  # noqa: S311 - a simulation

    @property
    def confirmed(self) -> Decision | None:
        return next((d for d in self.decisions if d.decision == "confirm"), None)

    async def _json(self, method: str, path: str, **kw: Any) -> Any:
        res = await self.panel.request(method, f"/api/v1{path}", headers=dict(self.headers), **kw)
        if res.status_code >= 400:
            raise OperatorError(res.status_code, res.text)
        return res.json()

    async def approve_profile(self, draft: Mapping[str, Any]) -> dict[str, Any]:
        """The profile card's approval, as corrected by the operator."""
        state = await self._json("GET", f"/missions/{self.mission_id}/search-profile")
        approved = state.get("approved")
        body = {
            "profile": dict(draft),
            "base_version": 0 if approved is None else int(approved["profile_version"]),
            "proposal_id": state.get("proposal_id"),
        }
        return dict(
            await self._json("POST", f"/missions/{self.mission_id}/search-profile", json=body)
        )

    async def approve_looks(self) -> list[str]:
        """Approve the AI's closer-look cards that wait for the pilot; the order ids. A card
        that went away meanwhile (answered, expired, withdrawn) is left."""
        mid = self.mission_id
        cards = await self._json("GET", f"/missions/{mid}/approvals", params={"status": "pending"})
        waiting = {str(a["subject_id"]): a for a in cards if a["kind"] == "command"}
        if not waiting:
            return []
        orders = await self._json(
            "GET", f"/missions/{mid}/commands", params={"status": "pending_approval"}
        )
        approved = []
        for c in orders:
            card = waiting.get(str(c["id"]))
            if card is None or c["skill"] not in APPROVED_LOOKS or c["issuer_kind"] != "brain":
                continue
            body = {
                "decision": "approve",
                "subject_version": card["subject_version"],
                "confirmations": list(card.get("required_confirmations") or []),
            }
            try:
                await self._json("POST", f"/approvals/{card['id']}/decision", json=body)
            except OperatorError as exc:
                if exc.status == 409 and ("SUPERSEDED" in exc.text or "TIMEOUT" in exc.text):
                    continue
                raise
            approved.append(str(c["id"]))
        self.looks_approved += approved
        return approved

    def judge(self, card: Mapping[str, Any]) -> tuple[str | None, bool, Judged]:
        """The thing a card shows, whether it is the missing person, and how that was
        found: what its picture came from when known, otherwise the thing at its
        position."""
        known = None if self.card_truth is None else self.card_truth(card)
        if known is not None:
            return known[0], known[1], "picture"
        thing, target = self.truth(card["lat"], card["lon"])
        return thing, target, "position"

    def truth_of(self, card: Mapping[str, Any]) -> tuple[str | None, bool]:
        """The thing a card shows and whether it is the missing person (:meth:`judge`)."""
        thing, target, _ = self.judge(card)
        return thing, target

    def not_a_person(self, thing: str | None) -> bool:
        """Whether the operator rejects a card showing ``thing`` as "not a person": only
        with ``is_person``, for nothing there or a thing that is not a person."""
        if self.is_person is None:
            return False
        return thing is None or not self.is_person(thing)

    async def queue(self) -> list[dict[str, Any]]:
        data = await self._json("GET", f"/missions/{self.mission_id}/candidates")
        return list(data["items"])

    def _note_target(self, items: list[dict[str, Any]]) -> None:
        for rank, c in enumerate(items, 1):
            if c["state"] != "ready" or not self.truth_of(c)[1]:
                continue
            if c["priority_class"] == "low":
                self.target_ever_low = True
            if self.target_first_rank is None:
                self.target_first_rank = rank

    async def review_once(self) -> Decision | None:
        """Look at the top open card and decide on it; ``None`` when there is none."""
        items = await self.queue()
        self._note_target(items)
        done = {d.candidate_id for d in self.decisions}
        pick = next(
            (
                (rank, c)
                for rank, c in enumerate(items, 1)
                if c["state"] in OPEN_STATES and c["candidate_id"] not in done
            ),
            None,
        )
        if pick is None:
            return None
        rank, card = pick
        await self.sleep(self._rng.uniform(*self.delay_s))  # looking at the pictures
        cid = card["candidate_id"]
        for _ in range(3):  # the card may change while the operator looks
            detail = await self._json("GET", f"/candidates/{cid}")
            now = detail["card"]
            if now["state"] not in OPEN_STATES:
                return None
            thing, target, by = self.judge(now)
            if now.get("thumb_detection_id"):
                await self._json("POST", f"/detections/{now['thumb_detection_id']}/view")
            decision: Literal["confirm", "reject"] = "confirm" if target else "reject"
            body: dict[str, Any] = {"decision": decision, "subject_rev": now["rev"], "note": ""}
            if decision == "reject" and self.not_a_person(thing):
                body["reason"] = "not_a_person"
            try:
                await self._json("POST", f"/candidates/{cid}/decision", json=body)
            except OperatorError as exc:
                if exc.status == 409 and "SUPERSEDED" in exc.text:
                    continue
                raise
            d = Decision(
                t=self.monotonic(),
                candidate_id=cid,
                seq=int(now["seq"]),
                decision=decision,
                target=target,
                thing=thing,
                rank=rank,
                priority_class=str(now.get("priority_class")),
                fields=dict(now.get("fields") or {}),
                by=by,
            )
            self.decisions.append(d)
            return d
        return None

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.approve_looks()
            d = await self.review_once()
            if d is not None and d.decision == "confirm" and self.stop_after_confirm:
                return
            if d is None:
                await self.sleep(POLL_S)


class OperatorError(RuntimeError):
    def __init__(self, status: int, text: str) -> None:
        super().__init__(f"{status}: {text[:300]}")
        self.status = status
        self.text = text
