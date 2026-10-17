"""Compare-and-set and telling a person from a failsafe."""

from __future__ import annotations

import asyncio

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from onboard_helpers import Rig, at, rig

from sd_mav.modes import CopterMode
from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.link import OperatorRelease


async def _in_goto(r: Rig) -> None:
    await r.takeoff()
    lat, lon = at(150, 0)
    await r.send(r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40, "speed_ms": 8}))
    await r.run(3)
    assert r.ex.state == S.TRANSIT


async def test_compare_and_set() -> None:
    """A mode change from outside in the middle of a goto: HELD, the next target never goes."""
    async with rig() as r:
        await _in_goto(r)
        r.fc.set_mode_external(CopterMode.LOITER)
        await r.run(0.3)
        before = len(r.vehicle.sent)
        await r.run(10)
        assert r.ex.state == S.HELD and r.ex.held_by == "external"
        assert r.steering_sent(before) == []
        assert r.ex.over.stop_latch  # an unrequested LOITER means stop
        assert r.ex.resume_point is not None


async def test_classification_window() -> None:
    async with rig() as r:
        await _in_goto(r)
        r.fc.set_mode_external(CopterMode.RTL)
        await r.run(1.5)
        r.fc.statustext("Battery 1 is low 15.1V used 7600 mAh")
        await r.run(1.0)  # t = 2.5 s after the change
        assert r.ex.state == S.FAILSAFE
    async with rig() as r:
        await _in_goto(r)
        r.fc.set_mode_external(CopterMode.RTL)
        await r.run(2.5)
        assert r.ex.state == S.HELD and r.ex.held_by == "external"
        assert not r.ex.control_suspended  # the window is over; HELD stays
        r.fc.statustext("GCS Failsafe")  # too late to count
        await r.run(1.0)
        assert r.ex.state == S.HELD
    async with rig() as r:
        await _in_goto(r)
        r.fc.statustext("Fence breach")
        await r.run(1.0)
        r.fc.set_mode_external(CopterMode.SMART_RTL)  # 1 s after the signal
        await r.run(1.5)
        assert r.ex.state == S.FAILSAFE


async def test_notice_names_who_changed_the_mode() -> None:
    async with rig() as r:
        await _in_goto(r)
        r.fc.set_mode_external(CopterMode.BRAKE)  # the ground's STOP arrives first as MAVLink
        await r.run(0.3)
        r.ex.on_override(r.notice("stop"))
        assert r.ex.state == S.HELD and r.ex.held_by == "operator_stop"


async def test_external_mode_change_held() -> None:
    """Orders after a takeover are all refused, and nothing that steers goes out."""
    async with rig() as r:
        await _in_goto(r)
        r.fc.set_mode_external(CopterMode.LOITER)
        await r.run(0.5)
        before = len(r.vehicle.sent)
        lat, lon = at(50, 50)
        for _ in range(20):
            for skill, params in (
                ("goto", {"lat": lat, "lon": lon, "alt_m": 40}),
                ("resume", {}),
                ("hold", {}),
            ):
                cmd = await r.send(r.order(skill, params), 0.2)
                assert r.up.codes(cmd.command_id) == ("ONB_CONTROL_HELD",)
        assert r.steering_sent(before) == []


async def test_fc_silent_goes_held() -> None:
    async with rig() as r:
        await _in_goto(r)
        r.fc.silent = True
        await r.run(4)
        assert r.ex.state == S.HELD and r.ex.held_by == "fc_silent"
        assert "executor.fc_silent" in r.up.kinds()


async def test_inbound_command_ignored() -> None:
    """A MAVLink command to the onboard computer changes nothing."""
    from pymavlink.dialects.v20 import ardupilotmega as mavlink

    async with rig() as r:
        await _in_goto(r)
        r.fc.set_mode_external(CopterMode.LOITER)
        await r.run(0.5)
        for sysid in (255, 253):
            msg = mavlink.MAVLink_command_long_message(
                1, 191, mavlink.MAV_CMD_USER_1, 0, 0, 0, 0, 0, 0, 0, 0
            )
            enc = mavlink.MAVLink(None, srcSystem=sysid, srcComponent=190)
            r.vehicle.link.receive(bytes(msg.pack(enc)))
        await r.run(0.5)
        assert r.ex.state == S.HELD
        assert r.up.kinds().count("executor.inbound_command_ignored") == 2


_OPS = st.lists(
    st.sampled_from(
        [
            "goto",
            "resume",
            "hold",
            "survey",
            "takeoff",
            "home",
            "mode",
            "failsafe",
            "stop",
            "rtl_notice",
            "watchdog",
            "battery",
            "cancel",
            "wait",
        ]
    ),
    min_size=1,
    max_size=12,
)


async def _restart_rig_case(ops: list[str]) -> None:
    async with rig(start=False) as r:
        # The vehicle flies (armed, 40 m, LOITER) when sd-onboard starts: HELD.
        r.fc.armed, r.fc.alt_rel, r.fc.mode = True, 40.0, CopterMode.LOITER
        r.fc.target = (r.fc.lat, r.fc.lon, 40.0)
        task = asyncio.ensure_future(r.ex.run(r.stop))
        r.background.append(task)
        await r.until(lambda: r.ex._started, 10)
        r.ex.on_link(True, r.clock.monotonic_ms())
        await r.install()
        assert r.ex.state == S.HELD
        before = len(r.vehicle.sent)
        lat, lon = at(60, 0)
        polygon = {
            "type": "Polygon",
            "coordinates": [[[lon, lat], [lon + 1e-4, lat], [lon + 1e-4, lat + 1e-4], [lon, lat]]],
        }
        for op in ops:
            match op:
                case "goto":
                    await r.ex.on_skill_command(
                        r.order("goto", {"lat": lat, "lon": lon, "alt_m": 40}), deadline_ms=None
                    )
                case "resume" | "hold" | "takeoff":
                    await r.ex.on_skill_command(r.order(op, {}), deadline_ms=None)
                case "home":
                    await r.ex.on_skill_command(r.order("return_home", {}), deadline_ms=None)
                case "survey":
                    await r.ex.on_skill_command(
                        r.order("survey_sector", {"polygon": polygon}), deadline_ms=None
                    )
                case "mode":
                    r.fc.set_mode_external(CopterMode.GUIDED)
                case "failsafe":
                    r.fc.statustext("GCS Failsafe")
                case "stop":
                    r.ex.on_override(r.notice("stop"))
                case "rtl_notice":
                    r.ex.on_override(r.notice("return_home"))
                case "watchdog":
                    r.ex.on_override(r.notice("watchdog_stop"))
                case "battery":
                    r.ex.on_override(r.notice("battery_rtl"))
                case "cancel":
                    r.ex.on_cancel(new_id())
                case _:
                    await r.run(0.5)
            await r.run(0.1)
        await r.run(1)
        assert r.ex.state in (S.HELD, S.FAILSAFE)
        # Only the emergency fallback's mode steps (the ground's MAVLink is not part of
        # this rig, so the flight controller never shows the target mode by itself).
        assert r.steering_sent(before, allow_fallback=True) == []
        # Only Resume hands back; then steering is possible again (when the mode allows).
        r.fc.set_mode_external(CopterMode.LOITER)
        await r.run(0.5)
        r.ex.on_release(OperatorRelease(action_seq=r._seq, operator_id="operator:pilot"), new_id())


@settings(max_examples=2000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(ops=_OPS)
def test_no_control_when_held(ops: list[str]) -> None:
    asyncio.run(_restart_rig_case(ops))
