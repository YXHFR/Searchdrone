"""The daily cost cap for the cloud model in simulator exams, counted on real usage.

Every call's price comes from the provider's own ``usage.cost`` (OpenRouter states it in
each answer, in US dollars). A call without a stated price is charged at a deliberately
high per-token price, so the cap is never undercounted. The ledger is one JSON file with
the kronor spent per day (Swedish date); several processes share it under a file lock.

Before each call :class:`MeteredRuntime` checks that the day has room for it: what is left
must exceed a reserve of three times the dearest call seen so far (at least
``MIN_RESERVE_SEK``). Otherwise it raises :class:`BudgetExhaustedError` and the decider
switches to the script.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sd_brain.runtime.base import ModelError, ModelReply, ModelRuntime, Sampling

__all__ = [
    "DAILY_CAP_SEK",
    "BudgetExhaustedError",
    "CostLedger",
    "MeteredRuntime",
    "reply_cost_sek",
]

# The owner's hard cap: never more than this per day on the cloud model in exams.
DAILY_CAP_SEK = 30.0
# Kronor per US dollar. Set on the high side so the cap holds when the krona weakens.
USD_SEK = 11.5
# Charged per million tokens when the answer states no price (dearer than any model used).
UNKNOWN_USD_PER_MTOK_IN = 15.0
UNKNOWN_USD_PER_MTOK_OUT = 75.0
MIN_RESERVE_SEK = 0.25
_TZ = ZoneInfo("Europe/Stockholm")


class BudgetExhaustedError(ModelError):
    """The day's cap has no room for another cloud call."""


def reply_cost_sek(reply: ModelReply) -> tuple[float, bool]:
    """(kronor, measured): the stated price, or the high estimate when none is stated."""
    if reply.cost_usd is not None:
        return reply.cost_usd * USD_SEK, True
    usd = (
        reply.prompt_tokens * UNKNOWN_USD_PER_MTOK_IN
        + reply.completion_tokens * UNKNOWN_USD_PER_MTOK_OUT
    ) / 1e6
    return usd * USD_SEK, False


class CostLedger:
    """Kronor spent on the cloud model per Swedish day, in a shared JSON file."""

    def __init__(
        self,
        path: Path,
        *,
        cap_sek: float = DAILY_CAP_SEK,
        now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
    ) -> None:
        if not 0 <= cap_sek <= DAILY_CAP_SEK:
            raise ValueError(f"kostnadstaket får vara högst {DAILY_CAP_SEK:g} kr per dag")
        self.path = path
        self.cap_sek = cap_sek
        self._now = now

    def day(self) -> str:
        return self._now().astimezone(_TZ).date().isoformat()

    @contextlib.contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        with os.fdopen(fd, "r+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            raw = f.read()
            data: dict[str, Any] = json.loads(raw) if raw.strip() else {}
            yield data
            f.seek(0)
            f.truncate()
            f.write(json.dumps(data, indent=2, sort_keys=True) + "\n")

    def spent_sek(self) -> float:
        with self._locked() as data:
            return float((data.get(self.day()) or {}).get("sek", 0.0))

    def remaining_sek(self) -> float:
        return max(0.0, self.cap_sek - self.spent_sek())

    def charge(self, sek: float, *, measured: bool) -> float:
        """Add one call's price; return what the day has spent."""
        with self._locked() as data:
            row = data.setdefault(self.day(), {"sek": 0.0, "calls": 0, "estimated_calls": 0})
            row["sek"] = round(float(row["sek"]) + sek, 6)
            row["calls"] = int(row["calls"]) + 1
            if not measured:
                row["estimated_calls"] = int(row.get("estimated_calls", 0)) + 1
            return float(row["sek"])


class MeteredRuntime:
    """A ``ModelRuntime`` that checks the cap before a call and charges it after.

    ``run_cap_sek`` also caps this run (one ``sd-brain eval`` or one farm lap), below the
    day's cap. ``cost_sek`` and ``calls`` count this runtime only.
    """

    def __init__(
        self, inner: ModelRuntime, ledger: CostLedger, *, run_cap_sek: float | None = None
    ) -> None:
        self.inner = inner
        self.ledger = ledger
        self.run_cap_sek = run_cap_sek
        self.cost_sek = 0.0
        self.calls = 0
        self.unmeasured_calls = 0
        self.dearest_sek = 0.0
        self.key_invalid = False  # the provider refused the key (HTTP 401 or 403)
        self.replies: list[dict[str, Any]] = []  # every answer, for replay fixtures

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    @property
    def supports_images(self) -> bool:
        return self.inner.supports_images

    def reserve_sek(self) -> float:
        return max(MIN_RESERVE_SEK, 3 * self.dearest_sek)

    def can_afford(self) -> bool:
        room = self.ledger.remaining_sek()
        if self.run_cap_sek is not None:
            room = min(room, self.run_cap_sek - self.cost_sek)
        return room > self.reserve_sek()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        timeout_s: float,
        sampling: Sampling,
    ) -> ModelReply:
        if not self.can_afford():
            raise BudgetExhaustedError("kostnadstaket är nått; skriptet tar över")
        try:
            reply = await self.inner.chat(messages, tools, timeout_s, sampling)
        except ModelError as exc:
            if exc.status in (401, 403):
                self.key_invalid = True
            raise
        self.replies.append(
            {
                "content": reply.content,
                "tool_calls": [
                    {"name": c.name, "arguments": c.arguments_json} for c in reply.tool_calls
                ],
            }
        )
        sek, measured = reply_cost_sek(reply)
        self.ledger.charge(sek, measured=measured)
        self.cost_sek += sek
        self.calls += 1
        self.unmeasured_calls += 0 if measured else 1
        self.dearest_sek = max(self.dearest_sek, sek)
        return reply
