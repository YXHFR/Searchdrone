"""The executor's own checks of every order (beyond the shared gate)."""

from __future__ import annotations

import pytest
from onboard_helpers import Rig, at, charter, rig, square

from sd_mav.mission import FENCE
from sd_onboard.skills import HANDLERS
from sd_safety.policy import FORBIDDEN_SKILL_NAMES
from sd_schemas.approval import command_hash
from sd_schemas.base import new_id
from sd_schemas.charter import AutonomyMode
from sd_schemas.executor import ExecutorState as S
from sd_schemas.gate import StateRef
from sd_schemas.link import ApprovalRef, CharterInstall
from sd_schemas.skills import SKILL_REGISTRY, GotoParams, SkillRegistry


def _arms(r: Rig) -> int:
    return sum(
        1 for m in r.vehicle.sent if m.get_type() == "COMMAND_LONG" and int(m.command) == 400
    )


def test_registry_handlers_complete() -> None:
    onboard = {
        s.name for s in SKILL_REGISTRY if s.status == "implemented" and s.runs_on == "onboard"
    }
    assert set(HANDLERS) == onboard
    for spec in SKILL_REGISTRY:
        if spec.status == "stub" or spec.runs_on == "ground":
            assert spec.name not in HANDLERS


def test_forbidden_skill_names() -> None:
    assert not FORBIDDEN_SKILL_NAMES & SKILL_REGISTRY.names()
    assert not FORBIDDEN_SKILL_NAMES & set(HANDLERS)


@pytest.mark.parametrize(
    "name", sorted(s.name for s in SKILL_REGISTRY if s.status == "stub" and s.runs_on == "onboard")
)
async def test_stub_skills_rejected(name: str) -> None:
    async with rig() as r:
        await r.takeoff()
        before = len(r.vehicle.sent)
        cmd = await r.send(r.order(name, {}))
        assert r.up.codes(cmd.command_id) == ("SKILL_NOT_IMPLEMENTED",)
        assert r.steering_sent(before) == []


@pytest.mark.parametrize("name", ["report", "mark_location", "propose_charter_change"])
async def test_ground_skills_go_to_the_wrong_executor(name: str) -> None:
    registry = SkillRegistry(
        s.model_copy(update={"status": "implemented"}) if s.name == name else s
        for s in SKILL_REGISTRY
    )
    async with rig(options={"registry": registry}) as r:
        await r.install()
        cmd = await r.send(r.order(name, {"text_sv": "x"}))
        assert r.up.codes(cmd.command_id) == ("ONB_WRONG_EXECUTOR",)


async def test_onboard_rule_issuer_refused_from_the_link() -> None:
    async with rig() as r:
        await r.takeoff()
        before = len(r.vehicle.sent)
        cmd = await r.send(r.order("hold", {}, issuer="rule:onboard"))
        assert r.up.codes(cmd.command_id) == ("ONB_WRONG_EXECUTOR",)
        assert r.steering_sent(before) == []


async def test_unknown_skill_and_bad_params() -> None:
    async with rig() as r:
        await r.takeoff()
        unknown = await r.send(r.order("set_param", {}))
        assert r.up.codes(unknown.command_id) == ("SKILL_UNKNOWN",)
        bad = await r.send(r.order("goto", {"lat": 999}))
        assert r.up.codes(bad.command_id) == ("SCHEMA_INVALID",)


async def test_takeoff_preconditions() -> None:
    async with rig() as r:
        no_charter = await r.send(r.order("takeoff", {"alt_m": 40}))
        assert r.up.codes(no_charter.command_id) == ("ONB_INVALID_TRANSITION",)
        await r.install()
        no_clear = await r.send(r.order("takeoff", {"alt_m": 40}))
        assert r.up.codes(no_clear.command_id) == ("ONB_CLEARANCE_MISSING",)
        r.clearance(charter_version=99)
        mismatch = await r.send(r.order("takeoff", {"alt_m": 40}))
        assert r.up.codes(mismatch.command_id) == ("ONB_CLEARANCE_MISMATCH",)
        r.clearance()
        await r.run(121)
        stale = await r.send(r.order("takeoff", {"alt_m": 40}))
        assert r.up.codes(stale.command_id) == ("ONB_CLEARANCE_STALE",)
        r.fc.battery_pct = 30  # below 15 % + 20 %
        r.clearance()
        await r.run(1.5)
        low = await r.send(r.order("takeoff", {"alt_m": 40}))
        assert r.up.codes(low.command_id) == ("ONB_PREFLIGHT_NOT_OK",)
        assert _arms(r) == 0


async def test_takeoff_needs_the_rc_transmitter() -> None:
    """With ``require_rc`` (the default outside SITL), PF12 needs a physical RC
    transmitter that the flight controller hears; without it the take-off is refused
    with its own reason, and nothing is armed."""
    async with rig(options={"require_rc": True}) as r:
        await r.install()
        r.clearance()
        await r.run(1.5)
        assert r.ex.preflight["PF12"] is False
        cmd = await r.send(r.order("takeoff", {"alt_m": 40}))
        assert r.up.codes(cmd.command_id) == ("ONB_RC_LINK_DOWN",)
        assert _arms(r) == 0
        r.fc.rc_link = True  # the pilot switches the RC transmitter on
        await r.run(1.5)
        assert r.ex.preflight["PF12"] is True and r.ex.preflight_ok()
        r.fc.rc_link = False  # and off again: the link counts as down within 1.5 s
        await r.run(2.0)
        assert r.ex.preflight["PF12"] is False and not r.ex.preflight_ok()


async def test_takeoff_without_require_rc_has_no_pf12() -> None:
    async with rig() as r:
        await r.install()
        await r.run(1.5)
        assert "PF12" not in r.ex.preflight


async def test_takeoff_not_allowed_in_profile() -> None:
    async with rig(options={"allow_companion_arm": False}) as r:
        await r.install()
        r.clearance()
        cmd = await r.send(r.order("takeoff", {"alt_m": 40}))
        assert r.up.codes(cmd.command_id) == ("ONB_ARM_NOT_ALLOWED_IN_PROFILE",)
        assert _arms(r) == 0


async def test_clearance_remote_rejected() -> None:
    async with rig() as r:
        await r.install()
        r.clearance(origin="remote")
        cmd = await r.send(r.order("takeoff", {"alt_m": 40}))
        assert r.up.codes(cmd.command_id) == ("ONB_CLEARANCE_REMOTE",)
        assert _arms(r) == 0


async def test_clearance_consumed_once() -> None:
    async with rig() as r:
        await r.install()
        r.clearance()
        first = await r.send(r.order("takeoff", {"alt_m": 40}))
        assert "running" in r.up.stages(first.command_id)
        r.ex.on_cancel(first.command_id)
        await r.until(lambda: r.ex.state in (S.HOLD, S.PREFLIGHT), 10)
        assert not r.ex.snapshot().clearance_available


async def test_clearance_invalidated() -> None:
    async with rig() as r:
        c = await r.install()
        r.clearance()
        assert r.ex.snapshot().clearance_available
        # A new charter version (for example a changed autonomy mode) needs a new click.
        await r.ex.on_charter(
            CharterInstall(
                charter=c.model_copy(
                    update={"version": 2, "autonomy_mode": AutonomyMode.APPROVE_EACH}
                )
            )
        )
        assert not r.ex.snapshot().clearance_available
        r.clearance(charter_version=2)
        from sd_schemas.link import HelloAck

        r.ex.on_connected(HelloAck(schema_major=1, server_epoch=new_id(), operator_latch=None))
        assert not r.ex.snapshot().clearance_available


async def test_approval_required() -> None:
    async with rig() as r:
        await r.takeoff()
        lat, lon = at(50, 0)
        params = {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 5}
        await r.ex.on_charter(
            CharterInstall(
                charter=r.charter.model_copy(  # type: ignore[union-attr]
                    update={"version": 2, "autonomy_mode": AutonomyMode.APPROVE_EACH}
                )
            )
        )
        missing = await r.send(r.order("goto", params, issuer="brain:ground"))
        assert r.up.codes(missing.command_id) == ("ONB_APPROVAL_MISSING",)
        wrong = ApprovalRef(approval_id=new_id(), command_hash="0" * 64)
        mismatch = await r.send(r.order("goto", params, issuer="brain:ground", approval=wrong))
        assert r.up.codes(mismatch.command_id) == ("ONB_APPROVAL_MISMATCH",)
        good = ApprovalRef(
            approval_id=new_id(), command_hash=command_hash("goto", GotoParams(**params), 2)
        )
        ok = await r.send(r.order("goto", params, issuer="brain:ground", approval=good))
        assert "running" in r.up.stages(ok.command_id)
        # The operator's own orders need no approval card onboard.
        hold = await r.send(r.order("hold", {}))
        assert "running" in r.up.stages(hold.command_id)


async def test_approval_required_for_always_approve_skills_in_mode_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mode B: a skill on the always-approve list still needs the card (test registry)."""
    registry = SkillRegistry(
        s.model_copy(update={"status": "implemented", "allowed_from": frozenset({S.HOLD})})
        if s.name == "land_at"
        else s
        for s in SKILL_REGISTRY
    )
    monkeypatch.setitem(HANDLERS, "land_at", HANDLERS["hold"])
    async with rig(options={"registry": registry}) as r:
        await r.takeoff()
        cmd = await r.send(r.order("land_at", {}, issuer="brain:ground"))
        assert r.up.codes(cmd.command_id) == ("ONB_APPROVAL_MISSING",)


async def test_stale_order() -> None:
    async with rig() as r:
        await r.takeoff()
        lat, lon = at(30, 0)
        params = {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 5}
        old = StateRef(state_epoch=r.ex.boot_id, state_seq=0)
        before = await r.send(r.order("goto", params, issuer="brain:ground", based_on=old))
        assert r.up.codes(before.command_id) == ("STATE_CHANGED",)
        fresh = await r.send(r.order("goto", params, issuer="brain:ground"))
        assert "running" in r.up.stages(fresh.command_id)
        other = StateRef(state_epoch=r.ex.boot_id + 1, state_seq=10_000)
        op = await r.send(r.order("hold", {}, based_on=other))
        assert r.up.codes(op.command_id) == ("ONB_STALE_ORDER",)
        brain = await r.send(r.order("goto", params, issuer="brain:ground", based_on=other))
        assert r.up.codes(brain.command_id) == ("STATE_CHANGED",)


async def test_charter_version_mismatch() -> None:
    async with rig() as r:
        await r.takeoff()
        cmd = await r.send(r.order("hold", {}, charter_version=7))
        assert r.up.codes(cmd.command_id) == ("CHARTER_VERSION_MISMATCH",)


async def test_rtl_rejects_orders() -> None:
    async with rig() as r:
        await r.takeoff()
        await r.send(r.order("return_home"))
        assert r.ex.state == S.RTL
        lat, lon = at(30, 0)
        goto = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}))
        hold = await r.send(r.order("hold", {}))
        assert r.up.codes(goto.command_id) == ("ONB_RTL_ACTIVE",)
        assert r.up.codes(hold.command_id) == ("ONB_RTL_ACTIVE",)
        again = await r.send(r.order("return_home"))
        assert r.up.stages(again.command_id) == ["accepted", "done"]


async def test_onboard_gate_rejects_what_the_ground_let_through() -> None:
    async with rig() as r:
        await r.takeoff()
        lat, lon = at(400, 0)  # outside the 200 m charter area
        cmd = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}))
        assert "OUTSIDE_CHARTER" in r.up.codes(cmd.command_id)
        assert "gate_disagreement" in [k.removeprefix("executor.") for k in r.up.kinds()]
        high = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 121}))
        assert "ALT_ABOVE_LIMIT" in r.up.codes(high.command_id)


async def test_onboard_uses_planned_path() -> None:
    """Both ends inside a U-shaped area, the straight line between them outside."""
    ring = []
    for x, y in [
        (-200, -200),
        (200, -200),
        (200, 200),
        (100, 200),
        (100, -100),
        (-100, -100),
        (-100, 200),
        (-200, 200),
        (-200, -200),
    ]:
        lat, lon = at(x, y)
        ring.append([lon, lat])
    u_area = {"type": "Polygon", "coordinates": [ring]}
    async with rig() as r:
        await r.install(
            charter(
                area=u_area,
                pilot_position={"lat": at(0, -150)[0], "lon": at(0, -150)[1]},
                home={"lat": at(0, -150)[0], "lon": at(0, -150)[1], "alt_amsl_m": 584.0},
            )
        )
        r.fc.home = (*at(0, -150), 584.0)
        r.fc.lat, r.fc.lon = at(0, -150)
        await r.run(2)
        r.clearance()
        await r.send(r.order("takeoff", {"alt_m": 30}))
        await r.until(lambda: r.ex.state == S.HOLD, 60)
        lat, lon = at(-150, 150)
        await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 30}))
        await r.until(lambda: r.ex.state == S.HOLD, 200, step=0.05)
        lat, lon = at(150, 150)
        cmd = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 30}))
        assert "OUTSIDE_CHARTER" in r.up.codes(cmd.command_id)


async def test_charter_exceeds_hard_limits() -> None:
    async with rig() as r:
        await r.ex.on_charter(CharterInstall(charter=charter(alt_max_agl_m=130)))
        assert r.up.charter_acks[-1].reason_codes == ["ONB_CHARTER_EXCEEDS_HARD_LIMITS"]
        assert r.ex.state == S.IDLE


async def test_charter_follows_a_raised_fence() -> None:
    """The ground raised the limits (and the flight controller's fence) by click."""
    async with rig() as r:
        r.fc.params.update({"FENCE_ALT_MAX": 205.0})
        await r.ex.on_charter(CharterInstall(charter=charter(alt_max_agl_m=200)))
        await r.run(3)
        assert r.up.charter_acks[-1].accepted
        assert r.ex.hard_limits.max_alt_agl_m == 200


@pytest.mark.parametrize(
    ("change", "accepted"),
    [
        ({}, True),
        ({"area": square(150)}, True),  # a smaller area is fine in the air
        ({"area": square(250)}, False),  # a larger one needs a landing
        ({"fence_hash": "x" * 64}, False),
        ({"geofence_buffer_m": 30}, False),
        ({"alt_max_agl_m": 110}, False),
        ({"alt_max_agl_m": 90}, True),
        ({"max_speed_ms": 14}, False),
        ({"max_range_from_pilot_m": 400}, True),
        ({"battery_rtl_pct": 15}, False),
        ({"battery_rtl_pct": 25}, True),
        ({"alt_min_agl_m": 10, "default_alt_agl_m": 40}, False),
        ({"alt_min_agl_m": 20}, True),
        ({"autonomy_mode": AutonomyMode.APPROVE_EACH}, True),
        ({"allowed_skills": ["hold", "return_home"]}, True),
        ({"on_ai_silent": "hold"}, True),
    ],
)
async def test_charter_in_flight(change: dict[str, object], accepted: bool) -> None:
    async with rig() as r:
        await r.install(charter(alt_max_agl_m=100, max_speed_ms=12, battery_rtl_pct=20))
        await r.takeoff()
        r.clearance()
        new = r.charter.model_copy(update={"version": 2, **change})  # type: ignore[union-attr]
        await r.ex.on_charter(CharterInstall(charter=new))
        ack = r.up.charter_acks[-1]
        assert ack.accepted is accepted, ack
        if accepted:
            assert r.ex.charter == new
            assert not r.ex.snapshot().clearance_available  # every new version resets it
        else:
            assert ack.reason_codes == ["ONB_CHARTER_IN_FLIGHT"]
            assert r.ex.charter is not None and r.ex.charter.version == 1


async def test_takeoff_rereads_the_fence() -> None:
    """The fence in the flight controller changed after the charter was installed: the
    onboard gate refuses the takeoff (FENCE_NOT_VERIFIED, with its own decision)."""
    async with rig() as r:
        await r.install()
        r.clearance()
        await r.run(1.5)
        assert r.ex.preflight_ok()
        stored = r.fc.missions[FENCE]
        r.fc.missions[FENCE] = stored[:-1]  # someone changed the fence afterwards
        cmd = await r.send(r.order("takeoff", {"alt_m": 40}))
        assert r.up.codes(cmd.command_id) == ("FENCE_NOT_VERIFIED",)
        assert "gate_disagreement" in [k.removeprefix("executor.") for k in r.up.kinds()]
        assert r.ex.preflight["PF7"] is False
        assert _arms(r) == 0
