"""The link-loss rule (L-LINK, L-FIND, L-GCS): its table, and in the air on the fake."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from onboard_helpers import PLANE, Rig, at, charter, rig
from test_limits import C, vs

from sd_mav.modes import CopterMode
from sd_mav.vehicle import SendOrigin
from sd_onboard.executor.link_loss import INTERNAL_FIND_GOTO, LinkLossInput, LinkLossRule
from sd_onboard.io.persistence import StateStore
from sd_safety.constants import FIND_HOVER_EDGE_MARGIN_M, FIND_HOVER_S, LINK_LOSS_DETECT_S
from sd_safety.link_loss import hover_point
from sd_safety.targets import distance_m
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.executor import LinkLossPhase as P
from sd_schemas.link import Keepout, KeepoutInstall, ManualTakeover

SPEC = Path(__file__).with_name("link_loss_spec.yaml")
FIND_EAST, FIND_NORTH, FIND_R = 60.0, 0.0, 10.0


def keepout(east: float = FIND_EAST, north: float = FIND_NORTH, **kw: Any) -> Keepout:
    lat, lon = at(east, north)
    data: dict[str, Any] = {
        "keepout_id": new_id(),
        "lat": lat,
        "lon": lon,
        "radius_m": FIND_R,
        "source": "confirmed",
        "source_id": new_id(),
    }
    data.update(kw)
    return Keepout.model_validate(data)


FIND = keepout()
# Where each phase leaves the executor, as (state, active skill).
CONTEXT: dict[P, tuple[S, str | None]] = {
    P.NONE: (S.TRANSIT, "goto"),
    P.GRACE: (S.TRANSIT, "goto"),
    P.HOLD: (S.HOLD, None),
    P.FIND_MOVE: (S.TRANSIT, INTERNAL_FIND_GOTO),
    P.FIND_HOVER: (S.HOLD, None),
    P.HOME: (S.RTL, "return_home"),
    P.YIELDED: (S.HELD, None),
}


def inp(
    t: float,
    phase: P | None = None,
    *,
    state: S | None = None,
    active: str | None = "",
    link_up: bool = False,
    keepouts: Sequence[Keepout] = (FIND,),
    gcs: bool = False,
    taken_by: str | None = None,
    **vehicle: object,
) -> LinkLossInput:
    ctx_state, ctx_active = CONTEXT[phase or P.NONE]
    return LinkLossInput(
        now_ms=round(t * 1000),
        state=state or ctx_state,
        vehicle=vs(**vehicle),
        charter=C,
        link_up=link_up,
        link_last_inbound_ms=0,
        nav_ok=True,
        keepouts=keepouts,
        gcs_continue=gcs,
        active_skill=ctx_active if active == "" else active,
        taken_by=taken_by,
    )


def into(phase: P) -> tuple[LinkLossRule, float]:
    """A rule in ``phase``, and the time it got there (seconds after the loss)."""
    rule = LinkLossRule()
    t = 0.0
    if phase in (P.GRACE, P.HOLD, P.HOME, P.YIELDED):
        rule.check(inp(3.0, keepouts=()))
        t = 3.0
    if phase in (P.HOLD, P.HOME):
        rule.check(inp(23.0, P.GRACE, keepouts=()))
        t = 23.0
    if phase == P.HOME:
        rule.check(inp(53.0, P.HOLD, keepouts=()))
        t = 53.0
    if phase == P.YIELDED:
        rule.check(inp(5.0, state=S.HELD))
        t = 5.0
    if phase in (P.FIND_MOVE, P.FIND_HOVER):
        rule.check(inp(3.0))
        t = 3.0
    if phase == P.FIND_HOVER:
        rule.check(inp(8.0, P.FIND_HOVER))  # the goto is done: HOLD, nothing running
        t = 8.0
    assert rule.phase == phase, (phase, rule.phase)
    return rule, t


def _hover_latlon() -> tuple[float, float]:
    lat, lon = at(0, 0)
    plat, plon, _ = hover_point(
        lat=lat, lon=lon, find=FIND, home_lat=C.home.lat, home_lon=C.home.lon, plane=PLANE
    )
    return plat, plon


EVENTS: dict[str, Callable[[P, float], LinkLossInput | None]] = {
    "detect_normal": lambda p, t: inp(3.0, p, keepouts=()),
    "detect_find": lambda p, t: inp(3.0, p),
    "detect_find_at_point": lambda p, t: inp(
        3.0, p, lat=_hover_latlon()[0], lon=_hover_latlon()[1]
    ),
    "hold_time": lambda p, t: inp(23.0, p, keepouts=()),
    "home_time": lambda p, t: inp(53.0, p, keepouts=()),
    "gcs_lost": lambda p, t: inp(t + 0.1, p, gcs=True),
    "find_goto_done": lambda p, t: inp(t + 1, p, state=S.HOLD, active=None),
    "find_deadline": lambda p, t: inp(15.0, p),
    "find_timeout": lambda p, t: inp(123.0, p),
    "other_rtl": lambda p, t: inp(
        t + 0.1, p, state=S.RTL, active="return_home", taken_by="battery"
    ),
    "other_hold": lambda p, t: inp(t + 0.1, p, state=S.HOLD, active=None),
    "taken_over": lambda p, t: inp(t + 0.1, p, state=S.HELD, active=None, taken_by="human"),
    "anything": lambda p, t: inp(200.0, p, gcs=True),
    "link_back": lambda p, t: inp(t + 0.1, p, link_up=True),
    "landed": lambda p, t: None,
}


def _rows() -> list[dict[str, Any]]:
    return list(yaml.safe_load(SPEC.read_text()))


@pytest.mark.parametrize("row", _rows(), ids=lambda r: r["id"])
def test_link_loss_table_matches_spec(row: dict[str, Any]) -> None:
    for name in row["from"]:
        phase = P(name)
        rule, t = into(phase)
        x = EVENTS[row["event"]](phase, t)
        if x is None:
            rule.reset()
            actions = []
        else:
            actions = rule.check(x)
        to = row["to"]
        want = phase if to == "same" else P(to[name] if isinstance(to, dict) else to)
        assert rule.phase == want, (row["id"], name, rule.phase)
        assert [a.event for a in actions] == row["emits"], (row["id"], name, actions)


def test_every_phase_and_event_is_in_the_table() -> None:
    rows = _rows()
    assert {r["event"] for r in rows} == set(EVENTS)
    assert {P(n) for r in rows for n in r["from"]} == set(P)


def test_find_start_carries_the_hover_point() -> None:
    rule = LinkLossRule()
    (action,) = rule.check(inp(3.0))
    assert action.kind == "find" and action.hold_reason == "find_hold"
    assert action.data["keepout_id"] == str(FIND.keepout_id)
    assert action.data["candidate_id"] == str(FIND.source_id)
    hlat, hlon = action.data["hover_lat"], action.data["hover_lon"]
    assert isinstance(hlat, float) and isinstance(hlon, float)
    reach = distance_m(FIND.lat, FIND.lon, hlat, hlon)
    assert reach == pytest.approx(FIND_R + FIND_HOVER_EDGE_MARGIN_M, abs=0.1)
    assert action.data["yaw_deg"] == pytest.approx(90.0, abs=1.0)  # facing east, the find


def test_the_find_rule_needs_good_navigation() -> None:
    rule = LinkLossRule()
    x = inp(3.0)
    bad = LinkLossInput(**{**{f: getattr(x, f) for f in x.__dataclass_fields__}, "nav_ok": False})
    (action,) = rule.check(bad)
    assert action.event == "link_lost" and rule.phase == P.GRACE


def test_far_or_unconfirmed_finds_are_ignored() -> None:
    for k in (keepout(east=150), keepout(source="candidate")):
        rule = LinkLossRule()
        (action,) = rule.check(inp(3.0, keepouts=(k,)))
        assert action.event == "link_lost"


def test_a_link_never_heard_counts_as_lost() -> None:
    rule = LinkLossRule()
    x = inp(0.0, keepouts=())
    never = LinkLossInput(
        **{**{f: getattr(x, f) for f in x.__dataclass_fields__}, "link_last_inbound_ms": None}
    )
    events = [a.event for a in rule.check(never)]
    assert events == ["link_lost", "link_loss_hold", "link_loss_rtl"]


def test_gcs_lost_with_the_link_up_goes_home() -> None:
    """Case D: only the ground's MAVLink is gone; the flight controller kept LOITER."""
    rule = LinkLossRule()
    (action,) = rule.check(inp(1.0, P.HOLD, link_up=True, gcs=True))
    assert action.kind == "rtl" and action.event == "gcs_lost_rtl"


def test_ended_reasons() -> None:
    for state, taken_by, reason in (
        (S.RTL, "battery", "battery"),
        (S.RTL, "limit", "limit"),
        (S.FAILSAFE, "gnss", "gnss"),
        (S.HELD, None, "human"),
        (S.FAILSAFE, None, "fc_failsafe"),
    ):
        rule, t = into(P.FIND_HOVER)
        (action,) = rule.check(inp(t + 0.1, P.FIND_HOVER, state=state, taken_by=taken_by))
        assert action.data["reason"] == reason


# --- in the air on the fake ------------------------------------------------------------------


async def _searching(r: Rig, tmp_path: Path, find: Keepout) -> None:
    r.ex.store = StateStore(tmp_path, r.clock)
    await r.install(charter())
    await r.takeoff(40)
    r.ex.on_keepouts(KeepoutInstall(install_seq=1, keepouts=[find]), new_id())
    lat, lon = at(0, 150)
    await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 3}))
    await r.run(1)
    assert r.ex.state == S.TRANSIT


def _records(tmp_path: Path) -> list[dict[str, Any]]:
    lines = (tmp_path / "find_holds.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


async def test_find_hover_then_home(tmp_path: Path) -> None:
    """Case E: SD-link lost 50 m from a confirmed find: hover beside it, home at 123 s."""
    find = keepout(east=50, north=10)
    async with rig() as r:
        await _searching(r, tmp_path, find)
        cut = r.clock.monotonic()
        r.link = False
        closest = math.inf
        hover_at = home_at = None
        yaws: set[float] = set()
        while r.clock.monotonic() - cut < LINK_LOSS_DETECT_S + FIND_HOVER_S + 3:
            await r.run(0.1)
            closest = min(closest, distance_m(r.fc.lat, r.fc.lon, find.lat, find.lon))
            if r.ex.link_rule.phase == P.FIND_MOVE and r.fc.target_yaw_deg is not None:
                yaws.add(round(r.fc.target_yaw_deg, 1))
            if hover_at is None and r.ex.state == S.HOLD and r.fc.mode == CopterMode.LOITER:
                hover_at = r.clock.monotonic() - cut
                assert r.ex.hold_reason == "find_hold"
                snap = r.ex.snapshot()
                assert snap.link_loss_phase == P.FIND_HOVER
                assert snap.find_hold_keepout_id == find.keepout_id
                assert "L-FIND" in snap.limits_active
            if home_at is None and r.ex.state == S.RTL:
                home_at = r.clock.monotonic() - cut
        assert hover_at is not None and hover_at <= 15.0
        # the nose toward the find on the way to the hover point (about east here)
        started = next(d for t, _, d in r.up.events if t == "executor.find_hold_started")
        assert yaws and all(abs(y - started["yaw_deg"]) <= 0.2 for y in yaws), yaws
        assert 60.0 < started["yaw_deg"] < 120.0
        assert home_at is not None and abs(home_at - (LINK_LOSS_DETECT_S + FIND_HOVER_S)) <= 1
        assert closest >= find.radius_m
        kinds = r.up.kinds()
        assert "executor.find_hold_started" in kinds and "executor.link_lost" not in kinds
        ended = [d for t, _, d in r.up.events if t == "executor.find_hold_ended"]
        assert [d["reason"] for d in ended] == ["timeout"]
        records = _records(tmp_path)
        assert [x["record"] for x in records] == ["started", "ended"]
        assert records[1]["keepout_id"] == str(find.keepout_id)
        assert records[1]["ended_reason"] == "timeout"
        assert distance_m(
            records[1]["hover_lat"], records[1]["hover_lon"], find.lat, find.lon
        ) == pytest.approx(find.radius_m + FIND_HOVER_EDGE_MARGIN_M, abs=0.5)
        r.link = True
        await r.until(lambda: r.ex.state == S.LANDED, 300, step=0.05)
        assert r.ex.link_rule.phase == P.NONE


async def test_find_hover_ignores_gcs_lost_and_stays_after_link_back(tmp_path: Path) -> None:
    find = keepout(east=30, north=0)
    async with rig() as r:
        await _searching(r, tmp_path, find)
        r.link = False
        await r.until(lambda: r.ex.link_rule.phase == P.FIND_HOVER, 20)
        await r.run(2)
        r.fc.statustext("GCS Failsafe - Continuing Pilot Control")
        await r.run(1)
        assert r.ex.state == S.HOLD and r.ex.hold_reason == "find_hold"
        assert "executor.gcs_lost_ignored_find" in r.up.kinds()
        r.link = True
        await r.run(FIND_HOVER_S + 10)
        assert r.ex.state == S.HOLD and r.ex.hold_reason == "find_hold"
        ended = [d for t, _, d in r.up.events if t == "executor.find_hold_ended"]
        assert [d["reason"] for d in ended] == ["link_restored"]


async def test_battery_wins_over_the_find_hover(tmp_path: Path) -> None:
    """Case F."""
    find = keepout(east=30, north=0)
    async with rig() as r:
        await _searching(r, tmp_path, find)
        r.link = False
        await r.until(lambda: r.ex.link_rule.phase == P.FIND_HOVER, 20)
        r.fc.battery_pct = 15
        await r.run(1)
        assert r.ex.state == S.RTL
        ended = [d for t, _, d in r.up.events if t == "executor.find_hold_ended"]
        assert [d["reason"] for d in ended] == ["battery"]
        assert r.ex.link_rule.phase == P.YIELDED


async def test_gcs_lost_in_loiter_goes_home() -> None:
    """Cases C and D: the flight controller kept LOITER after its GCS failsafe."""
    async with rig() as r:
        await r.install(charter())
        await r.takeoff(40)
        r.ex.on_override(r.notice("stop"))
        await r.run(1)
        assert r.ex.state == S.HELD
        r.fc.statustext("GCS Failsafe - Continuing Pilot Control")
        await r.run(1)
        assert r.ex.state == S.HELD  # the operator's hold, steered home as the FC would
        assert "executor.gcs_lost_rtl" not in r.up.kinds()
        assert r.up.kinds().count("executor.gcs_lost_rtl_held") == 1
        assert r.fc.mode in (CopterMode.SMART_RTL, CopterMode.RTL)
    async with rig() as r:
        await r.install(charter())
        await r.takeoff(40)
        assert r.ex.state == S.HOLD
        r.fc.statustext("GCS Failsafe - Continuing Pilot Control")
        await r.run(1)
        assert r.ex.state == S.RTL
        assert "executor.gcs_lost_rtl" in r.up.kinds()
        assert r.fc.mode in (CopterMode.SMART_RTL, CopterMode.RTL)


async def test_gcs_lost_in_a_takeover_goes_home_but_not_in_an_rc_mode() -> None:
    """Ta över from the ground has no sticks once its link is gone; the RC switch has."""
    async with rig() as r:
        await r.install(charter())
        await r.takeoff(40)
        r.ex.on_takeover(
            ManualTakeover(
                takeover_id=new_id(),
                active=True,
                action_seq=r.next_seq(),
                operator_id="operator:pilot",
            ),
            new_id(),
        )
        r.fc.set_mode_external(CopterMode.LOITER)
        await r.run(1)
        assert r.ex.held_by == "operator_manual"
        r.fc.statustext("GCS Failsafe - Continuing Pilot Control")
        await r.run(1)
        assert r.fc.mode in (CopterMode.SMART_RTL, CopterMode.RTL)
        r.fc.statustext("GCS Failsafe - Continuing Pilot Control")
        await r.run(1)
        assert r.up.kinds().count("executor.gcs_lost_rtl_held") == 1  # once per hold
    async with rig() as r:
        await r.install(charter())
        await r.takeoff(40)
        r.fc.set_mode_external(CopterMode.POSHOLD)  # no notice: the RC transmitter's switch
        await r.run(1)
        assert r.ex.state == S.HELD and r.ex.held_by == "external"
        r.fc.statustext("GCS Failsafe - Continuing Pilot Control")
        await r.run(2)
        assert r.fc.mode == CopterMode.POSHOLD
        assert "executor.gcs_lost_rtl_held" not in r.up.kinds()


async def test_gcs_home_from_a_hold_yields_to_the_rc_switch_and_newer_notices() -> None:
    """The HELD exception covers only the home mode change, never what the pilot did since."""
    async with rig() as r:
        await r.install(charter())
        await r.takeoff(40)
        r.ex.on_override(r.notice("stop"))
        await r.run(1)
        r.ex._gcs_home_notice = r.ex.auth.notice_mono_ms
        r.ex._gcs_home_sending = True
        assert r.ex.send_gate(SendOrigin.LIMIT) is None
        assert r.ex.send_gate(SendOrigin.SKILL) == "control_held"
        r.fc.set_mode_external(CopterMode.POSHOLD)  # the RC switch, mid-way
        await r.run(1)
        assert r.ex.send_gate(SendOrigin.LIMIT) == "mode_POSHOLD"
        r.fc.set_mode_external(CopterMode.LOITER)
        await r.run(1)
        r.ex.on_override(r.notice("stop"))  # a newer STOP or guard notice
        assert r.ex.send_gate(SendOrigin.LIMIT) == "override_notice"


async def test_keepout_install_highest_seq_wins() -> None:
    async with rig() as r:
        a, b = keepout(), keepout(east=10)
        ref = new_id()
        r.ex.on_keepouts(KeepoutInstall(install_seq=5, keepouts=[a]), ref)
        assert r.up.stages(ref) == ["done"]
        old = new_id()
        r.ex.on_keepouts(KeepoutInstall(install_seq=4, keepouts=[b]), old)
        assert r.up.codes(old) == ("ONB_STALE_ORDER",)
        assert r.ex.keepouts == (a,)
        r.ex.on_keepouts(KeepoutInstall(install_seq=6, keepouts=[]), new_id())
        assert r.ex.keepouts == ()
