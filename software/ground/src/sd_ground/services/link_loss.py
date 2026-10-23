"""Link loss on the ground: the keep-out cylinders for the onboard computer, and what the
panel should expect while SD-link is down.

* **Keep-outs:** every confirmed find of the open mission (the newest revision of a
  candidate is ``confirmed``) becomes a ``Keepout`` (``source = "confirmed"``, radius
  ``PERSON_KEEPOUT_M + sigma95_m``), and so does every open candidate the detector or the
  image model is at least ``KEEPOUT_SCORE_MIN`` sure of (``source = "candidate"``; gone
  when a person rejects it). Only the confirmed ones count for the link-loss rule. The
  whole list goes out as ``KeepoutInstall`` when it changes and at every ``Hello``.
  ``install_seq`` is at least the wall clock in milliseconds, so a restarted ground
  station still sends a higher one than the onboard computer kept.
* **Forecast:** once nothing has come from the onboard computer for
  ``LINK_LOSS_DETECT_S``, the same ``sd_safety.link_loss.choose_mode`` as onboard runs on
  the last snapshot and the same list (``VehicleLive.link_loss_plan``), and the panel is
  told what to expect. When the link is back, the first fresh snapshot says whether the
  drone hovers or flies home.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from sd_ground.logging import get_logger
from sd_safety.constants import KEEPOUT_SCORE_MIN, PERSON_KEEPOUT_M
from sd_safety.link_loss import DETECT_MS, FindPlan, choose_mode, plan_times
from sd_schemas.api import AlertCode, LinkLossPlan
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState
from sd_schemas.link import KEEPOUTS_MAX, Keepout, KeepoutInstall

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow
    from sd_schemas.api import Mission

log = get_logger("sd_ground.link_loss")

__all__ = ["KEEPOUT_POLL_S", "LinkWatch", "on_connected", "plan_of", "tick"]

KEEPOUT_POLL_S: Final = 2.0  # how often the cylinders' candidates are read from the database
KEEPOUT_TTL_MS: Final = 10_000  # how late a KeepoutInstall may still be delivered
BACK_WAIT_MS: Final = 10_000  # after the link is back: wait this long for a fresh snapshot
_S = ExecutorState
# Where the onboard rule steers (sd_onboard.executor.state.MOVING); elsewhere someone else
# does, and the ground does not forecast.
_MOVING: Final = frozenset({_S.TAKEOFF, _S.TRANSIT, _S.SURVEY, _S.HOLD})
_NAV_LIMITS: Final = frozenset({"L-GPS", "L-SPOOF"})


@dataclass
class LinkWatch:
    """One vehicle's keep-out list and link-loss forecast."""

    install_seq: int = -1
    sent: tuple[Any, ...] | None = None  # (mission id, cylinders) last installed
    keepouts: list[Keepout] = field(default_factory=list[Keepout])
    ids: dict[str, uuid.UUID] = field(default_factory=dict[str, uuid.UUID])
    seqs: dict[str, int] = field(default_factory=dict[str, int])  # candidate -> K<n>
    polled_ms: int | None = None
    last_inbound_ms: int | None = None
    lost: bool = False
    plan: LinkLossPlan | None = None
    alerted_mav_up: bool | None = None  # the loss alarm went out with MAVLink up
    back_since_ms: int | None = None  # the link is back; waiting for a fresh snapshot


def watch(flow: MissionFlow, vehicle_id: str) -> LinkWatch:
    found = flow.link_watch.get(vehicle_id)
    if found is None:
        found = flow.link_watch[vehicle_id] = LinkWatch()
    return found


def plan_of(flow: MissionFlow, vehicle_id: str) -> LinkLossPlan | None:
    found = flow.link_watch.get(vehicle_id)
    return None if found is None else found.plan


async def tick(flow: MissionFlow, mission: Mission) -> None:
    w = watch(flow, mission.vehicle_id)
    now = flow.now_ms()
    if w.polled_ms is None or now - w.polled_ms >= KEEPOUT_POLL_S * 1000:
        w.polled_ms = now
        await _load(flow, mission, w)
        await _install(flow, mission, w, resend=False)
    await _forecast(flow, mission, w)


async def on_connected(flow: MissionFlow, vehicle_id: str) -> None:
    """At every ``Hello``: the list again (an empty one without an open mission, so a
    restarted onboard computer never keeps an earlier mission's finds)."""
    w = watch(flow, vehicle_id)
    mission = await flow.repo.open_mission(vehicle_id)
    if mission is None:
        w.keepouts, w.seqs = [], {}
    else:
        await _load(flow, mission, w)
    await _install(flow, mission, w, resend=True, vehicle_id=vehicle_id)


# --- keep-outs ----------------------------------------------------------------------------


async def _load(flow: MissionFlow, mission: Mission, w: LinkWatch) -> None:
    rows = await flow.repo.keepout_candidates(str(mission.id), KEEPOUT_SCORE_MIN, KEEPOUTS_MAX + 1)
    if len(rows) > KEEPOUTS_MAX:  # confirmed finds come first and are always kept
        log.warning("link_loss.keepouts_truncated", mission_id=str(mission.id), kept=KEEPOUTS_MAX)
    keepouts: list[Keepout] = []
    seqs: dict[str, int] = {}
    for row in rows[:KEEPOUTS_MAX]:
        cid = str(row["candidate_id"])
        kid = w.ids.get(cid)
        if kid is None:
            kid = w.ids[cid] = new_id()
        seqs[cid] = int(row["seq"])
        confirmed = row["state"] == "confirmed"
        keepouts.append(
            Keepout(
                keepout_id=kid,
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                radius_m=min(500.0, PERSON_KEEPOUT_M + float(row["sigma95_m"])),
                source="confirmed" if confirmed else "candidate",
                source_id=uuid.UUID(cid),
            )
        )
    w.keepouts, w.seqs = keepouts, seqs


async def _install(
    flow: MissionFlow,
    mission: Mission | None,
    w: LinkWatch,
    *,
    resend: bool,
    vehicle_id: str | None = None,
) -> None:
    vid = vehicle_id or (None if mission is None else mission.vehicle_id)
    if flow.link is None or vid is None:
        return
    mid = None if mission is None else str(mission.id)
    key = (mid, tuple((str(k.source_id), k.source, k.lat, k.lon, k.radius_m) for k in w.keepouts))
    if key == w.sent and not resend:
        return
    seq = w.install_seq
    if key != w.sent or seq < 0:
        epoch_ms = int(flow.now().timestamp() * 1000)
        seq = max(w.install_seq + 1, epoch_ms)
    install = KeepoutInstall(install_seq=seq, keepouts=list(w.keepouts))
    sent = await flow.link.send_message(vid, install, mission_id=mid, ttl_ms=KEEPOUT_TTL_MS)
    if sent is None:
        return  # the link is down: the next poll or Hello sends it
    w.install_seq, w.sent = seq, key


# --- the forecast -------------------------------------------------------------------------


def _utc_at(flow: MissionFlow, mono_ms: int) -> dt.datetime:
    return flow.now() - dt.timedelta(milliseconds=flow.now_ms() - mono_ms)


async def _forecast(flow: MissionFlow, mission: Mission, w: LinkWatch) -> None:
    vid = mission.vehicle_id
    now = flow.now_ms()
    if flow.link is not None:
        age = flow.link.link_status(vid).onboard_heartbeat_age_ms
        if age is not None:
            w.last_inbound_ms = now - age
    last = w.last_inbound_ms
    lost = last is not None and now - last >= DETECT_MS
    mav = flow.mav_snapshot(vid)
    mav_up = mav is not None and mav.connected
    if lost and not w.lost and last is not None:
        w.lost, w.back_since_ms = True, None
        w.plan = await _plan(flow, mission, w, last)
        w.alerted_mav_up = None
        if w.plan is not None:
            _lost_alarm(flow, mission, w, mav_up=mav_up)
    elif lost and w.plan is not None and w.alerted_mav_up and not mav_up:
        _lost_alarm(flow, mission, w, mav_up=False)  # now MAVLink is gone too
    elif not lost and w.lost:
        w.lost = False
        if w.plan is not None:
            w.back_since_ms = now
        w.plan, w.alerted_mav_up = None, None
    _link_back(flow, mission, w, now)


async def _plan(
    flow: MissionFlow, mission: Mission, w: LinkWatch, lost_ms: int
) -> LinkLossPlan | None:
    snap, _ = flow.snapshot(mission.vehicle_id)
    if snap is None or snap.executor_state not in _MOVING:
        return None
    charter = await flow.active_charter(mission)
    if charter is None:
        return None
    mav = flow.mav_snapshot(mission.vehicle_id)
    fix_ok = mav is None or not mav.connected or mav.telemetry.gps_fix_type >= 3
    nav_ok = fix_ok and not (_NAV_LIMITS & set(snap.limits_active))
    plan = choose_mode(
        state=snap.executor_state,
        lat=snap.position.lat,
        lon=snap.position.lon,
        alt_rel_home_m=snap.alt_rel_home_m,
        nav_ok=nav_ok,
        keepouts=w.keepouts,
        charter=charter,
    )
    times = plan_times(plan, lost_ms)
    lost_at = _utc_at(flow, lost_ms)  # one clock reading: the other times are offsets

    def at(ms: int | None) -> dt.datetime | None:
        return None if ms is None else lost_at + dt.timedelta(milliseconds=ms - lost_ms)

    home = lost_at + dt.timedelta(milliseconds=times.home_ms - lost_ms)
    if isinstance(plan, FindPlan):
        return LinkLossPlan(
            mode="find",
            keepout_id=plan.keepout_id,
            candidate_id=plan.candidate_id,
            candidate_seq=w.seqs.get(str(plan.candidate_id)),
            hover_lat=plan.hover_lat,
            hover_lon=plan.hover_lon,
            lost_at_utc=lost_at,
            hold_at_utc=None,
            home_at_utc=home,
            hover_until_utc=at(times.hover_until_ms),
        )
    return LinkLossPlan(
        mode="normal",
        lost_at_utc=lost_at,
        hold_at_utc=at(times.hold_ms),
        home_at_utc=home,
    )


def _seconds_to(flow: MissionFlow, at: dt.datetime | None) -> int:
    return 0 if at is None else max(0, round((at - flow.now()).total_seconds()))


def _lost_alarm(flow: MissionFlow, mission: Mission, w: LinkWatch, *, mav_up: bool) -> None:
    plan = w.plan
    if plan is None:
        return
    w.alerted_mav_up = mav_up
    gone = max(0, round((flow.now() - plan.lost_at_utc).total_seconds()))
    home = _seconds_to(flow, plan.home_at_utc)
    k = "" if plan.candidate_seq is None else f"K{plan.candidate_seq}"
    if plan.mode == "find" and mav_up:
        code = AlertCode.FIND_HOLD
        text = (
            f"Datalänken är borta vid fynd {k}. Drönaren svävar vid fyndet i ca {home} s "
            "till och flyger sedan hem. Positionen är sparad. STOPP och FLYG HEM fungerar."
        )
    elif plan.mode == "find":
        code = AlertCode.FIND_HOLD_ALL
        text = (
            f"Ingen kontakt med drönaren sedan {gone} s. Väntat: den svävar vid fynd {k} i "
            f"ca {home} s till och flyger sedan hem. Positionen är sparad. Ta över med RC "
            "om det behövs."
        )
    elif not mav_up:
        code = AlertCode.LINK_LOST_ALL
        text = (
            f"Ingen kontakt med drönaren sedan {gone} s. Väntat: den flyger hem senast om "
            f"{home} s. Ta över med RC om det behövs."
        )
    else:
        return  # the usual rule with MAVLink up: SDLINK_LOST says it
    flow.alert(
        str(code),
        "warning",
        text,
        vehicle_id=mission.vehicle_id,
        mission_id=str(mission.id),
        details=_details(plan),
    )


def _details(plan: LinkLossPlan) -> dict[str, Any]:
    def iso(at: dt.datetime | None) -> str | None:
        return None if at is None else at.isoformat()

    return {
        "mode": plan.mode,
        "lost_at_utc": iso(plan.lost_at_utc),
        "hold_at_utc": iso(plan.hold_at_utc),
        "home_at_utc": iso(plan.home_at_utc),
        "hover_until_utc": iso(plan.hover_until_utc),
        "candidate_seq": plan.candidate_seq,
        "keepout_id": None if plan.keepout_id is None else str(plan.keepout_id),
    }


def _link_back(flow: MissionFlow, mission: Mission, w: LinkWatch, now: int) -> None:
    """The first snapshot after the link came back: hovering or flying home."""
    since = w.back_since_ms
    if since is None:
        return
    if now - since > BACK_WAIT_MS:
        w.back_since_ms = None
        return
    seen = None if flow.link is None else flow.link.snapshot_seen(mission.vehicle_id)
    if seen is None or seen.received_mono_ms < since:
        return
    w.back_since_ms = None
    state = seen.snapshot.executor_state
    if state is _S.HOLD:
        code, text = (
            AlertCode.LINK_BACK_HOLD,
            "Länken är tillbaka. Drönaren svävar och väntar på order.",
        )
    elif state in (_S.RTL, _S.FAILSAFE) or seen.snapshot.flight_mode in ("RTL", "SMART_RTL"):
        code, text = (
            AlertCode.LINK_BACK_HOME,
            "Länken är tillbaka. Drönaren flyger hem. Tryck STOPP om den ska stanna.",
        )
    else:
        return
    flow.alert(
        str(code),
        "warning",
        text,
        vehicle_id=mission.vehicle_id,
        mission_id=str(mission.id),
        details={"executor_state": str(state), "flight_mode": seen.snapshot.flight_mode},
    )
