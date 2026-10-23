"""The cloud budget guard: the books for every cloud call, the commander's and the search's.

A process reserves its estimate before a call (``reserve``), then books the real use
(``commit``) or gives the reservation back when nothing was sent (``release``). A
reservation that is not booked within ``reservation_ttl_s`` counts at its estimate
(``expired``). What is reserved and not yet booked counts against the caps, so two
processes cannot both spend the last krona.

Caps: per mission and per day (Swedish date), from ``[cloud_budget]``, never above the
outer cap in code. A model without a price is refused (``UNKNOWN_PRICE``); so is every
call while the books cannot be written (``NO_LEDGER``). At 80 % the panel is warned; when
a call no longer fits, the state is ``exhausted``: the panel shows it, the commander
drops to command mode and the matching pauses (both see the refusal on their next call).

The books live in ``cloud_usage`` (append-only); this service keeps the sums in memory
and rebuilds them from the table on first use.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from sd_ground.config import ConfigError
from sd_ground.context import Ground
from sd_ground.db.cloud_usage import CloudUsageRepo, UsageRow, UsageStatus
from sd_ground.logging import get_logger
from sd_schemas.base import new_id
from sd_schemas.cloud_budget import (
    CLOUD_BUDGET_CEILING_SEK_PER_DAY,
    BudgetCommit,
    BudgetCommitted,
    BudgetGrant,
    BudgetReserve,
    BudgetState,
    CloudBudgetStatus,
)

__all__ = [
    "EXHAUSTED_TEXT",
    "WARN_AT",
    "CloudBudget",
    "ModelPrice",
    "Prices",
    "Process",
    "load_prices",
]

log = get_logger("sd_ground.cloud_budget")
STOCKHOLM = ZoneInfo("Europe/Stockholm")
WARN_AT = 0.8
EXHAUSTED_TEXT = "Molnbudget slut – matchning pausad, ingen AI-chatt"  # noqa: RUF001 - the text the panel shows
WARNING_TEXT = "Molnbudgeten är till {pct} % använd."
Process = Literal["commander", "search"]


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    in_usd_per_mtok: float = Field(ge=0, le=1000)
    out_usd_per_mtok: float = Field(ge=0, le=1000)
    image_tokens_per_crop: int = Field(default=400, ge=0, le=10_000)


class Prices(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    usd_to_sek: float = Field(gt=0, le=100)
    models: dict[str, ModelPrice] = Field(default_factory=dict[str, ModelPrice])

    def estimate_sek(self, req: BudgetReserve) -> float | None:
        price = self.models.get(req.model_id)
        if price is None:
            return None
        tokens_in = req.est_prompt_tokens + req.image_count * price.image_tokens_per_crop
        usd = (
            tokens_in * price.in_usd_per_mtok + req.est_completion_tokens * price.out_usd_per_mtok
        ) / 1e6
        return usd * self.usd_to_sek

    def actual_sek(self, model_id: str, done: BudgetCommit) -> float:
        if done.cost_usd is not None:  # the provider's own price for the call
            return done.cost_usd * self.usd_to_sek
        price = self.models[model_id]
        usd = (
            done.prompt_tokens * price.in_usd_per_mtok
            + done.completion_tokens * price.out_usd_per_mtok
        ) / 1e6
        return usd * self.usd_to_sek


def load_prices(root: Path) -> Prices:
    """``config/cloud_prices.example.toml`` with ``config/cloud_prices.toml`` merged over
    it (per model). A broken file stops the server rather than pricing calls wrong."""
    data: dict[str, Any] = {"models": {}}
    for name in ("cloud_prices.example.toml", "cloud_prices.toml"):
        path = root / "config" / name
        if not path.is_file():
            continue
        try:
            part = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError([f"config/{name} går inte att läsa: {exc}"]) from exc
        models = part.pop("models", {})
        data.update(part)
        if isinstance(models, dict):
            data["models"].update(models)
    if "usd_to_sek" not in data:
        data["usd_to_sek"] = 10.5
    try:
        return Prices.model_validate(data)
    except ValidationError as exc:
        raise ConfigError([f"config/cloud_prices.toml går inte att använda: {exc}"]) from exc


@dataclass
class _Entry:
    process: Process
    purpose: str
    model_id: str
    mission_id: str | None
    day: dt.date
    est_sek: float
    status: UsageStatus = "reserved"
    actual_sek: float | None = None
    reserved_mono: float = 0.0

    @property
    def spent(self) -> float:
        if self.status == "committed":
            return self.actual_sek or 0.0
        if self.status == "released":
            return 0.0
        return self.est_sek  # reserved (still open) or expired


@dataclass
class CloudBudget:
    g: Ground
    prices: Prices
    store: CloudUsageRepo | None = None
    _entries: dict[str, _Entry] = field(default_factory=dict[str, _Entry])
    _mission_caps: dict[str, float] = field(default_factory=dict[str, float])
    # Scopes a call no longer fits in ("day:<date>" or a mission id): exhausted until the
    # cap is raised or the day ends.
    _full: set[str] = field(default_factory=set[str])
    _states: dict[str, BudgetState] = field(default_factory=dict[str, BudgetState])
    _active_mission: str | None = None  # the mission of the latest reservation
    _loaded: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # --- caps and sums -----------------------------------------------------------------

    @property
    def cfg(self):
        return self.g.config.cloud_budget

    def _today(self) -> dt.date:
        return self.g.clock.utc_now().astimezone(STOCKHOLM).date()

    def day_cap(self) -> float:
        return min(self.cfg.day_cap_sek, CLOUD_BUDGET_CEILING_SEK_PER_DAY)

    def mission_cap(self, mission_id: str) -> float:
        cap = self._mission_caps.get(mission_id, self.cfg.mission_cap_sek)
        return min(cap, CLOUD_BUDGET_CEILING_SEK_PER_DAY)

    def day_sek(self, day: dt.date | None = None) -> float:
        day = day or self._today()
        return sum(e.spent for e in self._entries.values() if e.day == day)

    def mission_sek(self, mission_id: str) -> float:
        return sum(e.spent for e in self._entries.values() if e.mission_id == mission_id)

    def status(self, mission_id: str | None = None) -> CloudBudgetStatus:
        mission_id = mission_id or self._active_mission
        day = self._today()
        day_sek, day_cap = self.day_sek(day), self.day_cap()
        ratios = [_ratio(day_sek, day_cap)]
        full = f"day:{day}" in self._full
        m_sek = m_cap = 0.0
        if mission_id is not None:
            m_sek, m_cap = self.mission_sek(mission_id), self.mission_cap(mission_id)
            ratios.append(_ratio(m_sek, m_cap))
            full = full or mission_id in self._full
        worst = max(ratios)
        state: BudgetState = (
            "exhausted" if full or worst >= 1.0 else "warning" if worst >= WARN_AT else "ok"
        )
        return CloudBudgetStatus.model_validate(
            {
                "mission_id": mission_id,
                "mission_sek": round(m_sek, 4),
                "mission_cap_sek": m_cap if mission_id is not None else self.cfg.mission_cap_sek,
                "day_sek": round(day_sek, 4),
                "day_cap_sek": day_cap,
                "state": state,
            }
        )

    def state(self) -> BudgetState:
        """For ``/system/status``: the day and the mission of the latest call."""
        return self.status().state

    # --- the books ---------------------------------------------------------------------

    def _ready(self) -> bool:
        return self.store is not None and self.g.db_ok

    async def _load(self) -> None:
        if self._loaded or self.store is None:
            return
        day_start = dt.datetime.combine(self._today(), dt.time(), tzinfo=STOCKHOLM)
        rows = await self.store.since(day_start.astimezone(dt.UTC))
        # A cap the pilot set in this run is newer than the stored ones.
        self._mission_caps = await self.store.mission_caps() | self._mission_caps
        for row in rows:
            entry = self._entries.get(row.reservation_id)
            if row.status == "reserved":
                self._entries[row.reservation_id] = _Entry(
                    process=row.process,
                    purpose=row.purpose,
                    model_id=row.model_id,
                    mission_id=row.mission_id,
                    day=row.created_at.astimezone(STOCKHOLM).date(),
                    est_sek=row.est_sek,
                    reserved_mono=float("-inf"),  # from before this start: expire at once
                )
            elif entry is not None:
                entry.status = row.status
                entry.actual_sek = row.actual_sek
        self._loaded = True
        await self._expire()

    async def _write(self, rid: str, entry: _Entry, **extra: Any) -> None:
        if self.store is None:
            raise RuntimeError("no ledger")
        await self.store.add(
            UsageRow(
                reservation_id=rid,
                mission_id=entry.mission_id,
                process=entry.process,
                purpose=entry.purpose,
                model_id=entry.model_id,
                est_sek=entry.est_sek,
                status=entry.status,
                created_at=self.g.clock.utc_now(),
                **extra,
            )
        )

    async def _expire(self) -> None:
        now = self.g.clock.monotonic()
        for rid, entry in list(self._entries.items()):
            if entry.status != "reserved" or now - entry.reserved_mono < self.cfg.reservation_ttl_s:
                continue
            entry.status = "expired"
            entry.actual_sek = entry.est_sek
            log.warning("cloud_budget.expired", reservation_id=rid, process=entry.process)
            try:
                await self._write(rid, entry, actual_sek=entry.est_sek)
            except Exception as exc:  # the sum in memory already counts it
                log.warning("cloud_budget.write_failed", error=type(exc).__name__)

    async def reserve(self, process: Process, req: BudgetReserve) -> BudgetGrant:
        async with self._lock:
            if not self._ready():
                return self._deny("NO_LEDGER", 0.0, 0.0)
            try:
                await self._load()
                await self._expire()
            except Exception as exc:
                log.warning("cloud_budget.load_failed", error=type(exc).__name__)
                return self._deny("NO_LEDGER", 0.0, 0.0)
            mission_id = None if req.mission_id is None else str(req.mission_id)
            est = self.prices.estimate_sek(req)
            if est is None:
                log.warning("cloud_budget.unknown_price", model_id=req.model_id)
                return self._deny("UNKNOWN_PRICE", 0.0, self._remaining(mission_id))
            day = self._today()
            if mission_id is not None:
                self._active_mission = mission_id
            # Once a scope is full nothing more goes, a smaller call neither: "exhausted"
            # on the panel means no cloud calls, until the pilot raises the cap (or the
            # day ends).
            if f"day:{day}" in self._full or self.day_sek(day) + est > self.day_cap():
                self._full.add(f"day:{day}")
                grant = self._deny("DAY_CAP", est, self._remaining(mission_id))
            elif mission_id is not None and (
                mission_id in self._full
                or self.mission_sek(mission_id) + est > self.mission_cap(mission_id)
            ):
                self._full.add(mission_id)
                grant = self._deny("MISSION_CAP", est, self._remaining(mission_id))
            else:
                rid = str(new_id())
                entry = _Entry(
                    process=process,
                    purpose=req.purpose,
                    model_id=req.model_id,
                    mission_id=mission_id,
                    day=day,
                    est_sek=est,
                    reserved_mono=self.g.clock.monotonic(),
                )
                try:
                    await self._write(rid, entry)
                except Exception as exc:
                    log.warning("cloud_budget.write_failed", error=type(exc).__name__)
                    return self._deny("NO_LEDGER", est, 0.0)
                self._entries[rid] = entry
                grant = BudgetGrant.model_validate(
                    {
                        "granted": True,
                        "reservation_id": rid,
                        "est_sek": est,
                        "remaining_sek": self._remaining(mission_id),
                    }
                )
            await self._notify(mission_id)
            return grant

    def _deny(self, reason: Any, est: float, remaining: float) -> BudgetGrant:
        log.info("cloud_budget.denied", reason=reason)
        return BudgetGrant(granted=False, reason=reason, est_sek=est, remaining_sek=remaining)

    def _remaining(self, mission_id: str | None) -> float:
        left = self.day_cap() - self.day_sek()
        if mission_id is not None:
            left = min(left, self.mission_cap(mission_id) - self.mission_sek(mission_id))
        return max(0.0, left)

    def _entry_of(self, process: Process, rid: str) -> _Entry | None:
        """The caller's own open reservation (another process's is not found)."""
        entry = self._entries.get(rid)
        if entry is None or entry.process != process:
            return None
        return entry

    async def commit(
        self, process: Process, done: BudgetCommit
    ) -> BudgetCommitted | Literal["unknown", "closed"]:
        async with self._lock:
            await self._expire()
            rid = str(done.reservation_id)
            entry = self._entry_of(process, rid)
            if entry is None:
                return "unknown"
            if entry.status != "reserved":
                return "closed"
            actual = self.prices.actual_sek(entry.model_id, done)
            entry.status, entry.actual_sek = "committed", actual
            try:
                await self._write(
                    rid,
                    entry,
                    actual_sek=actual,
                    prompt_tokens=done.prompt_tokens,
                    completion_tokens=done.completion_tokens,
                )
            except Exception as exc:  # the sum in memory counts it all the same
                log.warning("cloud_budget.write_failed", error=type(exc).__name__)
            if actual < entry.est_sek:
                self._freed(entry)
            await self._notify(entry.mission_id)
            return BudgetCommitted(actual_sek=actual)

    async def release(self, process: Process, rid: str) -> Literal["ok", "unknown", "closed"]:
        async with self._lock:
            await self._expire()
            entry = self._entry_of(process, rid)
            if entry is None:
                return "unknown"
            if entry.status != "reserved":
                return "closed"
            entry.status = "released"
            try:
                await self._write(rid, entry)
            except Exception as exc:
                log.warning("cloud_budget.write_failed", error=type(exc).__name__)
            self._freed(entry)
            await self._notify(entry.mission_id)
            return "ok"

    def _freed(self, entry: _Entry) -> None:
        """A booking below its estimate (or a release) gave room back: the scopes it was
        reserved in are no longer full; the next call is measured against the sums."""
        self._full.discard(f"day:{entry.day}")
        if entry.mission_id is not None:
            self._full.discard(entry.mission_id)

    async def set_mission_cap(self, mission_id: str, cap_sek: float, by: str) -> CloudBudgetStatus:
        """The pilot's new cap for the mission (logged as an event; never above the outer
        cap). A raised cap lets calls through again."""
        async with self._lock:
            cap = min(cap_sek, CLOUD_BUDGET_CEILING_SEK_PER_DAY)
            old = self.mission_cap(mission_id)
            self._mission_caps[mission_id] = cap
            self._full.discard(mission_id)
            self.g.add_event(
                "cloud_budget.mission_cap",
                mission_id=mission_id,
                payload={"cap_sek": cap, "previous_sek": old, "by": by},
            )
            log.info("cloud_budget.mission_cap", mission_id=mission_id, cap_sek=cap)
            await self._notify(mission_id)
            return self.status(mission_id)

    async def refresh(self) -> None:
        """Read the books once the database is there (at start, so the panel shows what
        was spent before a restart) and expire late reservations; the background task
        calls it, so the sums do not wait for the next call."""
        if not self._ready():
            return
        async with self._lock:
            try:
                await self._load()
                await self._expire()
            except Exception as exc:  # the next call or tick tries again
                log.warning("cloud_budget.load_failed", error=type(exc).__name__)

    async def tick(self) -> None:
        await self.refresh()

    # --- notices -----------------------------------------------------------------------

    async def _notify(self, mission_id: str | None) -> None:
        """Tell the panel when the state changes: a live ``system`` event, an audit event
        and, in a mission, a line in its chat."""
        status = self.status(mission_id)
        key = mission_id or "-"
        before = self._states.get(key, "ok")
        if status.state == before:
            return
        self._states[key] = status.state
        data = status.model_dump(mode="json")
        self.g.bus.publish(
            "system",
            "cloud_budget.state",
            data,
            ts=self.g.clock.utc_now(),
            vehicle_id=None,
            mission_id=mission_id,
        )
        severity = "warning" if status.state != "ok" else "info"
        self.g.add_event(
            f"cloud_budget.{status.state}", severity=severity, mission_id=mission_id, payload=data
        )
        log.info("cloud_budget.state", state=status.state, before=before)
        if status.state == "ok" or mission_id is None:
            return
        text = (
            EXHAUSTED_TEXT
            if status.state == "exhausted"
            else WARNING_TEXT.format(pct=int(WARN_AT * 100))
        )
        await self._chat_line(mission_id, text, status.state)

    async def _chat_line(self, mission_id: str, text: str, state: BudgetState) -> None:
        from sd_ground.services import chat as chat_service

        flow = self.g.flow
        if flow is None:
            return
        try:
            mission = await flow.mission(mission_id)
            await chat_service.add_line(
                flow,
                mission,
                msg_id=str(new_id()),
                role="system",
                author_ref="system:ground",
                kind="status",
                text=text,
                in_reply_to=None,
                refs={"cloud_budget": state},
            )
        except Exception as exc:  # the live event and the audit event still went out
            log.warning("cloud_budget.chat_failed", error=type(exc).__name__)


def _ratio(spent: float, cap: float) -> float:
    if cap <= 0:
        return 1.0
    return spent / cap
