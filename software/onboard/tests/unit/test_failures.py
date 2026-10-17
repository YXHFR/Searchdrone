"""Failure paths of the state machine: takeoff, cancel, FAILSAFE release, restart charter."""

from __future__ import annotations

import asyncio

from onboard_helpers import at, charter, rig

from sd_mav.mission import FENCE, fence_hash, fence_items
from sd_mav.modes import CopterMode
from sd_onboard.executor.state import TRANSITIONS, skill_key
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.link import CharterInstall, OperatorRelease
from sd_schemas.skills import SKILL_REGISTRY


def test_registry_allowed_from_matches_the_transition_table() -> None:
    for spec in SKILL_REGISTRY:
        rows = {state for (state, key) in TRANSITIONS if key == skill_key(spec.name)}
        assert spec.allowed_from == rows, spec.name


async def test_arm_refused_goes_back_to_preflight() -> None:
    async with rig() as r:
        await r.install()
        r.clearance()
        r.fc.drop_commands = 0
        original = r.fc._command

        def refuse_arm(link, msg):  # type: ignore[no-untyped-def]
            if int(msg.command) == 400:
                r.fc.statustext("Arm: Battery below minimum")
                return 4  # MAV_RESULT_FAILED
            return original(link, msg)

        r.fc._command = refuse_arm  # type: ignore[method-assign]
        cmd = await r.send(r.order("takeoff", {"alt_m": 40}), 3)
        assert r.up.codes(cmd.command_id) == ("ONB_ARM_REFUSED",)
        assert r.ex.state == S.PREFLIGHT
        assert not r.fc.armed


async def test_takeoff_timeout_when_it_does_not_climb() -> None:
    async with rig() as r:
        await r.install()
        r.clearance()
        original = r.fc._command

        def no_climb(link, msg):  # type: ignore[no-untyped-def]
            result = original(link, msg)
            if int(msg.command) == 22:
                r.fc.target = None  # accepted, but it never lifts
            return result

        r.fc._command = no_climb  # type: ignore[method-assign]
        cmd = await r.send(r.order("takeoff", {"alt_m": 40}))
        await r.until(lambda: r.ex.state != S.TAKEOFF, 30)
        assert r.up.codes(cmd.command_id) == ("ONB_TAKEOFF_TIMEOUT",)
        assert r.ex.state == S.PREFLIGHT  # still on the ground: nothing more is sent


async def test_cancel_in_the_air_loiters() -> None:
    async with rig() as r:
        await r.takeoff()
        lat, lon = at(150, 0)
        cmd = await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 5}))
        await r.run(3)
        r.ex.on_cancel(cmd.command_id)
        assert r.ex.state == S.HOLD
        await r.run(3)
        assert r.fc.mode == CopterMode.LOITER
        assert r.up.stages(cmd.command_id)[-1] == "superseded"
        assert r.ex.resume_point is not None


async def test_failsafe_release_needs_the_cause_gone() -> None:
    async with rig() as r:
        await r.takeoff()
        lat, lon = at(100, 0)
        await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 5}))
        await r.run(2)
        r.fc.battery_pct = 14
        r.fc.statustext("Battery 1 is low")
        r.fc.set_mode_external(CopterMode.LOITER)
        await r.run(1.5)
        assert r.ex.state == S.FAILSAFE
        ref = new_id()
        r.ex.on_release(OperatorRelease(action_seq=0, operator_id="operator:pilot"), ref)
        assert r.up.codes(ref) == ("ONB_FAILSAFE_ACTIVE",)
        r.fc.battery_pct = 80  # the cause is gone (and the ground's heartbeat is fresh)
        await r.run(2)
        ref = new_id()
        r.ex.on_release(OperatorRelease(action_seq=0, operator_id="operator:pilot"), ref)
        assert r.ex.state == S.HOLD, r.up.acks[-2:]


async def test_first_charter_after_a_restart_in_the_air() -> None:
    async with rig(start=False) as r:
        c = charter()
        polygon = [(lat, lon) for lon, lat in c.area["coordinates"][0][:-1]]
        r.fc.missions[FENCE] = fence_items(polygon)
        r.fc.armed, r.fc.alt_rel, r.fc.mode = True, 40.0, CopterMode.LOITER
        r.fc.target = (r.fc.lat, r.fc.lon, 40.0)
        r.background.append(asyncio.ensure_future(r.ex.run(r.stop)))
        await r.until(lambda: r.ex._started, 10)
        assert r.ex.state == S.HELD
        wrong = c.model_copy(update={"fence_hash": "0" * 64})
        await r.ex.on_charter(CharterInstall(charter=wrong))
        await r.run(2)
        assert r.up.charter_acks[-1].reason_codes == ["ONB_CHARTER_IN_FLIGHT"]
        right = c.model_copy(update={"fence_hash": fence_hash(r.fc.missions[FENCE])})
        await r.ex.on_charter(CharterInstall(charter=right))
        await r.run(2)
        assert r.up.charter_acks[-1].accepted
        assert r.ex.charter == right
