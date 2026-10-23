"""``ScriptedOperator``: the person at the panel in the search scenarios, played by a script.

Only a person confirms a find. In the simulator's search scenarios this class stands in
for that person, through the panel API with the test rig's pilot token (the ground only
lets a CLI token decide on the test rig): it approves the search profile as the card
would send it, then works the candidate queue the way the panel shows it, highest
priority first. For each open card it waits 5-20 s (looking at the pictures), opens the
card's picture in large size, and confirms the card if the simulator's truth says it is
the missing person, otherwise rejects it. It never decides on a card that is not open
(new, merged, confirmed, rejected).

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

__all__ = ["OPEN_STATES", "Decision", "ScriptedOperator"]

OPEN_STATES = frozenset({"ready", "assessing", "inspecting"})
DELAY_S = (5.0, 20.0)
POLL_S = 1.0


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


@dataclass
class ScriptedOperator:
    panel: httpx.AsyncClient
    mission_id: str
    # The simulator's truth at a card's position: the thing there (or None) and whether
    # it is the missing person.
    truth: Callable[[float, float], tuple[str | None, bool]]
    headers: Mapping[str, str]
    seed: int = 0
    delay_s: tuple[float, float] = DELAY_S
    stop_after_confirm: bool = True
    env: Mapping[str, str] | None = None
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep
    monotonic: Callable[[], float] = time.monotonic
    decisions: list[Decision] = field(default_factory=list[Decision])
    target_first_rank: int | None = None  # the target's place when its card was first ready
    target_first_seen: float | None = None
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

    async def queue(self) -> list[dict[str, Any]]:
        data = await self._json("GET", f"/missions/{self.mission_id}/candidates")
        return list(data["items"])

    def _note_target(self, items: list[dict[str, Any]]) -> None:
        for rank, c in enumerate(items, 1):
            if c["state"] != "ready" or not self.truth(c["lat"], c["lon"])[1]:
                continue
            if c["priority_class"] == "low":
                self.target_ever_low = True
            if self.target_first_rank is None:
                self.target_first_rank, self.target_first_seen = rank, self.monotonic()

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
            thing, target = self.truth(now["lat"], now["lon"])
            if now.get("thumb_detection_id"):
                await self._json("POST", f"/detections/{now['thumb_detection_id']}/view")
            decision: Literal["confirm", "reject"] = "confirm" if target else "reject"
            body = {"decision": decision, "subject_rev": now["rev"], "note": ""}
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
            )
            self.decisions.append(d)
            return d
        return None

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
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
