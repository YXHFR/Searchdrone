"""STOP and FLY HOME through the ground station's own MAVLink channel (04 T14, 03 §7.1).

A fake flight controller and a fake clock; the real ``GroundMav``, ``Latches`` and
``VehicleActions``.
"""

from __future__ import annotations

import pytest
from pymavlink.dialects.v20 import ardupilotmega as mavlink
from stop_helpers import OBSERVER, PILOT, VID, Stack, ground_disarms, ground_modes, stack

from sd_ground.emergency import EmergencyRefusedError, check_role
from sd_ground.services.vehicle_actions import ActionRefusedError
from sd_ground.sources import EmergencyUnavailableError
from sd_mav.modes import HOME_MODES, STOP_MODES, CopterMode
from sd_schemas.link import ManualTakeover, OperatorOverrideNotice, OperatorRelease


def _results(s: Stack) -> list[str]:
    return [o["result"] for o in s.outcomes()]


async def test_stop_in_flight_brake_at_once_and_acked() -> None:
    async with stack() as s:
        s.fly()
        await s.run(0.5)
        before = len(s.fc.received)
        reply = s.stop()
        assert reply.paths == {"mavlink": "queued", "sdlink": "queued"}
        await s.run(0.05)  # the first DO_SET_MODE goes out at once
        assert ground_modes(s.fc, before)[:1] == [CopterMode.BRAKE]
        await s.run(1.0)
        assert s.fc.mode == CopterMode.BRAKE
        assert "acked" in _results(s)
        latch = s.latches.get(VID)
        assert latch is not None and latch.action == "stop"
        notice = s.sink.sent[0]
        assert isinstance(notice, OperatorOverrideNotice)
        assert (notice.kind, notice.action_seq) == ("stop", reply.action_seq)
        assert "vehicle.stop" in s.g.types() and "latch.set" in s.g.types()


@pytest.mark.parametrize(
    ("refused", "expected", "alarm"),
    [
        ({CopterMode.BRAKE}, CopterMode.LOITER, False),
        ({CopterMode.BRAKE, CopterMode.LOITER}, CopterMode.LAND, True),
    ],
)
async def test_stop_steps_when_refused(
    refused: set[CopterMode], expected: CopterMode, alarm: bool
) -> None:
    async with stack() as s:
        s.fly()
        s.fc.refuse_modes = {int(m) for m in refused}
        await s.run(0.5)
        before = len(s.fc.received)
        s.stop()
        await s.run(1.5)
        assert s.fc.mode == expected
        modes = ground_modes(s.fc, before)
        assert modes[0] == CopterMode.BRAKE and expected in modes
        assert ("alarm.STOP_PATH_FAILED" in s.g.types()) is alarm


async def test_silent_fc_steps_after_one_second_each() -> None:
    async with stack() as s:
        s.fly()
        await s.run(0.5)
        s.fc.drop_commands = 1000  # nothing arrives: no ack, the mode never changes
        before = len(s.fc.received)
        t0 = s.clock.monotonic()
        s.stop()
        seen: dict[CopterMode, float] = {}
        for _ in range(400):
            await s.clock.run_for(0.01, step=0.01)
            for m in ground_modes(s.fc, before):
                seen.setdefault(m, s.clock.monotonic() - t0)
        assert seen[CopterMode.BRAKE] < 0.05
        assert seen[CopterMode.LOITER] == pytest.approx(1.0, abs=0.1)
        assert seen[CopterMode.LAND] == pytest.approx(2.0, abs=0.15)
        await s.actions.drain(s.g)
        assert "failed" in _results(s) and "alarm.STOP_PATH_FAILED" in s.g.types()
        # Resent every 200 ms within a step.
        brakes = ground_modes(s.fc, before).count(CopterMode.BRAKE)
        assert 4 <= brakes <= 6


async def test_temporarily_rejected_goes_to_the_next_step_at_once() -> None:
    async with stack() as s:
        s.fly()
        await s.run(0.5)
        original = s.fc._command

        def answer(link, msg):  # type: ignore[no-untyped-def]
            if int(msg.command) == mavlink.MAV_CMD_DO_SET_MODE and int(msg.param2) == 17:
                return mavlink.MAV_RESULT_TEMPORARILY_REJECTED
            return original(link, msg)

        s.fc._command = answer  # type: ignore[method-assign]
        t0 = s.clock.monotonic()
        s.stop()
        await s.run(0.3)
        assert s.fc.mode == CopterMode.LOITER
        assert s.clock.monotonic() - t0 <= 0.35


@pytest.mark.parametrize("action", ["stop", "return_home"])
async def test_taking_off_counts_as_airborne(action: str) -> None:
    async with stack() as s:
        s.fly(alt=1.0)
        s.fc.landed_state = lambda: mavlink.MAV_LANDED_STATE_TAKEOFF  # type: ignore[method-assign]
        await s.run(0.5)
        before = len(s.fc.received)
        s.stop() if action == "stop" else s.home()
        await s.run(1.0)
        modes = ground_modes(s.fc, before)
        assert modes[0] == (CopterMode.BRAKE if action == "stop" else CopterMode.SMART_RTL)
        assert ground_disarms(s.fc, before) == 0
        assert "already_active" not in _results(s)


async def test_stop_armed_on_the_ground_disarms() -> None:
    async with stack() as s:
        s.fc.armed = True
        s.fc.mode = CopterMode.GUIDED
        s.fc.alt_rel = 0.2
        await s.run(0.5)
        before = len(s.fc.received)
        s.stop()
        await s.run(1.0)
        assert not s.fc.armed
        assert ground_disarms(s.fc, before) >= 1 and ground_modes(s.fc, before) == []
        assert s.outcomes()[-1]["mode_confirmed"] == "DISARMED"


async def test_refused_disarm_takes_the_stop_steps() -> None:
    async with stack() as s:
        s.fc.armed = True
        s.fc.mode = CopterMode.GUIDED
        s.fc.alt_rel = 0.2
        original = s.fc._command

        def answer(link, msg):  # type: ignore[no-untyped-def]
            if int(msg.command) == mavlink.MAV_CMD_COMPONENT_ARM_DISARM and int(msg.param1) == 0:
                return mavlink.MAV_RESULT_DENIED
            return original(link, msg)

        s.fc._command = answer  # type: ignore[method-assign]
        await s.run(0.5)
        before = len(s.fc.received)
        s.stop()
        await s.run(1.5)
        assert ground_modes(s.fc, before)[:1] == [CopterMode.BRAKE]
        assert s.fc.mode in STOP_MODES


async def test_stop_disarmed_on_the_ground_is_already_active() -> None:
    async with stack() as s:
        before = len(s.fc.received)
        s.stop()
        await s.run(0.5)
        assert ground_modes(s.fc, before) == [] and ground_disarms(s.fc, before) == 0
        assert "already_active" in _results(s)
        assert isinstance(s.sink.sent[0], OperatorOverrideNotice)  # the notice still goes


async def test_home_in_flight_and_its_steps() -> None:
    async with stack() as s:
        s.fly()
        s.fc.refuse_modes = {int(CopterMode.SMART_RTL)}
        await s.run(0.5)
        before = len(s.fc.received)
        s.home()
        await s.run(1.5)
        assert ground_modes(s.fc, before)[:2] == [CopterMode.SMART_RTL, CopterMode.RTL]
        assert s.fc.mode == CopterMode.RTL
        assert "alarm.HOME_PATH_FAILED" not in s.g.types()


async def test_stop_then_home_300ms_later_no_stop_modes_after_home() -> None:
    async with stack() as s:
        s.fly()
        s.fc.drop_acks = 1000  # slow answers make the race visible
        await s.run(0.5)
        before = len(s.fc.received)
        s.stop()
        await s.run(0.3)
        mark = len(s.fc.received)
        s.home()
        await s.run(3.0)
        after = ground_modes(s.fc, mark)
        first_home = after.index(CopterMode.SMART_RTL)
        assert not [m for m in after[first_home:] if m in STOP_MODES]
        assert s.fc.mode in HOME_MODES
        assert ground_modes(s.fc, before)[0] == CopterMode.BRAKE


@pytest.mark.parametrize("late", [CopterMode.GUIDED, CopterMode.SMART_RTL, CopterMode.RTL])
async def test_stop_mode_race(late: CopterMode) -> None:
    """A late order 50 ms after STOP (here: the fake changes mode by itself) is undone."""
    async with stack() as s:
        s.fly()
        await s.run(0.5)
        s.stop()
        await s.run(0.5)
        s.fc.set_mode_external(late)
        await s.run(2.0)
        assert s.fc.mode in STOP_MODES
        await s.run(8.0)
        assert s.fc.mode in STOP_MODES


async def test_stop_lets_a_failsafe_stand() -> None:
    async with stack() as s:
        s.fly()
        await s.run(0.5)
        s.stop()
        await s.run(1.0)
        s.fc.statustext("GCS Failsafe")
        s.fc.set_mode_external(CopterMode.RTL)
        await s.run(3.0)
        assert s.fc.mode == CopterMode.RTL


async def test_home_mode_race() -> None:
    async with stack() as s:
        s.fly()
        await s.run(0.5)
        s.home()
        await s.run(0.5)
        s.fc.set_mode_external(CopterMode.LOITER)
        await s.run(0.05)
        s.fc.set_mode_external(CopterMode.GUIDED)
        await s.run(2.0)
        assert s.fc.mode in HOME_MODES


async def test_mavlink_down_still_sends_the_notice() -> None:
    async with stack() as s:
        s.fly()
        await s.run(0.5)
        s.ground.link.up = False  # only the ground's MAVLink is cut
        await s.run(3.5)
        s.stop()
        await s.run(4.0)
        assert "failed" in _results(s)
        assert isinstance(s.sink.sent[-1], OperatorOverrideNotice)
        assert s.outcomes()[0]["paths"]["mavlink"] == "unavailable"


async def test_sdlink_down_mavlink_still_acked() -> None:
    async with stack(sink_up=False) as s:
        s.fly()
        await s.run(0.5)
        reply = s.stop()
        assert reply.paths == {"mavlink": "queued", "sdlink": "unavailable"}
        await s.run(1.0)
        assert "acked" in _results(s) and s.fc.mode == CopterMode.BRAKE


async def test_no_path_at_all_is_503() -> None:
    async with stack(sink_up=False) as s:
        s.ground.link.up = False
        await s.run(4.0)
        with pytest.raises(EmergencyUnavailableError) as err:
            s.stop()
        assert (err.value.status, err.value.code) == (503, "NO_PATH")
        assert s.latches.get(VID) is None


async def test_same_client_action_id_gives_the_same_answer() -> None:
    async with stack() as s:
        s.fly()
        await s.run(0.5)
        a = s.actions.dispatch("stop", VID, client_action_id="x1", via="button", by=PILOT)
        b = s.actions.dispatch("stop", VID, client_action_id="x1", via="button", by=PILOT)
        assert a == b
        assert s.actions.outcome(VID, str(a.action_id)) is not None


async def test_unknown_vehicle_is_404() -> None:
    async with stack() as s:
        with pytest.raises(EmergencyUnavailableError) as err:
            s.actions.dispatch("stop", "other-9", client_action_id="y", via="button", by=PILOT)
        assert err.value.status == 404


def test_who_may_press_what() -> None:
    check_role("stop", OBSERVER)  # every logged-in user may press STOP (F24)
    with pytest.raises(EmergencyRefusedError):
        check_role("return_home", OBSERVER)


# --- the latch and Resume -----------------------------------------------------------------


async def test_resume_releases_and_tells_onboard() -> None:
    async with stack() as s:
        s.fly()
        await s.run(0.5)
        stop = s.stop()
        await s.run(1.0)
        reply = s.resume()
        assert reply.action_seq > stop.action_seq
        await s.run(0.1)
        assert s.latches.get(VID) is None
        release = s.sink.sent[-1]
        assert isinstance(release, OperatorRelease) and release.action_seq == reply.action_seq
        assert "latch.released" in s.g.types()
        # Resume changes no mode: the guard of the STOP stops at the newer action.
        mark = len(s.fc.received)
        s.fc.set_mode_external(CopterMode.LOITER)
        await s.run(2.0)
        assert ground_modes(s.fc, mark) == []


async def test_resume_only_for_the_pilot() -> None:
    async with stack() as s:
        s.stop(who=OBSERVER)
        with pytest.raises(ActionRefusedError) as err:
            s.resume(who=OBSERVER)
        assert err.value.status == 403
        assert s.latches.get(VID) is not None


async def test_resume_refused_while_flying_home_then_stop_and_resume() -> None:
    async with stack() as s:
        s.fly()
        await s.run(0.5)
        s.home()
        await s.run(1.0)
        with pytest.raises(ActionRefusedError) as err:
            s.resume()
        assert (err.value.status, err.value.code) == (409, "RETURN_HOME_ACTIVE")
        s.stop()  # STOP during FLY HOME: the latch becomes stop
        await s.run(1.0)
        assert s.latches.get(VID).action == "stop"  # type: ignore[union-attr]
        assert s.fc.mode in STOP_MODES
        s.resume()
        assert s.latches.get(VID) is None


async def test_all_latches_go_at_landing() -> None:
    async with stack() as s:
        s.fly()
        await s.run(0.5)
        s.home()
        await s.run(1.0)
        assert s.latches.get(VID) is not None
        s.fc.alt_rel, s.fc.armed = 0.0, False  # landed and disarmed
        await s.run(1.0)
        assert s.latches.get(VID) is None


async def test_action_seq_never_goes_back() -> None:
    async with stack() as s:
        seqs = [s.stop().action_seq for _ in range(3)]
        assert seqs == sorted(set(seqs))
        s.latches.raise_seq(VID, seqs[-1] + 1000)  # an onboard Hello saw a higher number
        assert s.stop().action_seq > seqs[-1] + 1000


# --- steering by hand ---------------------------------------------------------------------


async def test_takeover_loiter_and_sticks_from_the_ground() -> None:
    from sd_schemas.api import ManualControlInput

    async with stack() as s:
        s.fly()
        await s.run(0.5)
        reply = s.actions.takeover(VID, client_action_id="t1", by=PILOT)
        await s.run(1.0)
        assert s.fc.mode == CopterMode.LOITER
        notice = s.sink.sent[-1]
        assert isinstance(notice, ManualTakeover) and notice.active
        value = ManualControlInput.model_validate(
            {
                "takeover_id": str(reply.action_id),
                "input_seq": 1,
                "pitch": 0.5,
                "roll": 0.0,
                "throttle": 0.5,
                "yaw": 0.0,
                "source": "gamepad",
            }
        )
        assert s.actions.manual_input(VID, value, PILOT)
        assert not s.actions.manual_input(VID, value, OBSERVER)  # only the pilot steers
        await s.run(0.3)
        sticks = s.fc.sticks
        assert sticks is not None and (sticks.get_srcSystem(), sticks.get_srcComponent()) == (
            250,
            190,
        )
        assert int(sticks.x) == 500
        # The 1 s watchdog: no new value from the panel, so nothing more is sent.
        await s.run(1.2)
        count = s.core.sticks_sent
        await s.run(2.0)
        assert s.core.sticks_sent == count
        # A newer value starts it again; an older one is dropped.
        s.actions.manual_input(VID, value.model_copy(update={"input_seq": 2}), PILOT)
        await s.run(0.2)
        assert s.core.sticks_sent > count
        s.actions.handback(VID, client_action_id="h1", by=PILOT)
        await s.run(0.2)
        after = s.core.sticks_sent
        s.actions.manual_input(VID, value.model_copy(update={"input_seq": 3}), PILOT)
        await s.run(0.5)
        assert s.core.sticks_sent == after


async def test_takeover_refused_while_flying_home() -> None:
    async with stack() as s:
        s.fly()
        await s.run(0.5)
        s.home()
        with pytest.raises(ActionRefusedError) as err:
            s.actions.takeover(VID, client_action_id="t2", by=PILOT)
        assert err.value.code == "RETURN_HOME_ACTIVE"


async def test_stop_ends_steering_by_hand() -> None:
    async with stack() as s:
        s.fly()
        await s.run(0.5)
        s.actions.takeover(VID, client_action_id="t3", by=PILOT)
        await s.run(1.0)
        s.stop()
        await s.run(1.0)
        assert s.fc.mode in STOP_MODES
        assert s.core.snapshot().manual is False  # type: ignore[union-attr]


# --- the chat's stop path -------------------------------------------------------------------


async def test_chat_stop_check_goes_the_button_way() -> None:
    async with stack() as s:
        s.fly()
        await s.run(0.5)

        async def check(text: str) -> str | None:
            return "stop" if "stanna" in text else None

        none = await s.actions.chat_override(
            "hur långt kvar?",
            vehicle_id=VID,
            message_id="m1",
            by=PILOT,
            stop_check=check,  # type: ignore[arg-type]
        )
        assert none is None
        reply = await s.actions.chat_override(
            "stanna där!",
            vehicle_id=VID,
            message_id="m2",
            by=PILOT,
            stop_check=check,  # type: ignore[arg-type]
        )
        assert reply is not None
        await s.run(1.0)
        assert s.fc.mode == CopterMode.BRAKE
        assert s.outcomes()[-1]["via"] == "chat"
        # The same line acting twice (the AI's report after the ground's own check) is one.
        again = s.actions.chat_intent("stop", vehicle_id=VID, message_id="m2", by=PILOT)
        assert again == reply


async def test_chat_never_resumes() -> None:
    async with stack() as s:
        s.stop()
        with pytest.raises(ValueError):
            s.actions.chat_intent("resume", vehicle_id=VID, message_id="m3", by=PILOT)

        async def says_resume(text: str) -> str:
            return "resume"

        out = await s.actions.chat_override(
            "fortsätt",
            vehicle_id=VID,
            message_id="m4",
            by=PILOT,
            stop_check=says_resume,  # type: ignore[arg-type]
        )
        assert out is None and s.latches.get(VID) is not None


async def test_broken_stop_check_does_not_break_the_chat() -> None:
    async with stack() as s:

        async def broken(text: str) -> str:
            raise RuntimeError("engine down")

        out = await s.actions.chat_override(
            "stopp",
            vehicle_id=VID,
            message_id="m5",
            by=PILOT,
            stop_check=broken,  # type: ignore[arg-type]
        )
        assert out is None


# --- review fixes: what may stop a chain, and what the watch leaves alone ------------------


async def test_handback_during_fly_home_does_not_stop_the_chain() -> None:
    async with stack() as s:
        s.fly()
        s.fc.refuse_modes = {int(CopterMode.SMART_RTL)}
        s.fc.drop_acks = 1000  # slow answers: the hand-back lands mid-chain
        await s.run(0.5)
        s.home()
        await s.run(0.1)
        s.actions.handback(VID, client_action_id="h9", by=PILOT)
        await s.run(2.5)
        assert s.fc.mode == CopterMode.RTL


async def test_takeover_during_stop_does_not_stop_the_chain() -> None:
    async with stack() as s:
        s.fly()
        s.fc.refuse_modes = {int(CopterMode.BRAKE), int(CopterMode.LOITER)}
        await s.run(0.5)
        s.stop()
        await s.run(0.1)
        s.actions.takeover(VID, client_action_id="t9", by=PILOT)
        await s.run(3.0)
        assert s.fc.mode == CopterMode.LAND
        assert "alarm.STOP_PATH_FAILED" in s.g.types()


@pytest.mark.parametrize("person_mode", [CopterMode.ALT_HOLD, CopterMode.STABILIZE])
@pytest.mark.parametrize("action", ["stop", "return_home"])
async def test_watch_leaves_an_rc_mode_change_alone(action: str, person_mode: CopterMode) -> None:
    """The RC transmitter goes before everything: a mode only a person can choose ends
    the watch instead of being undone."""
    async with stack() as s:
        s.fly()
        await s.run(0.5)
        s.stop() if action == "stop" else s.home()
        await s.run(1.0)
        mark = len(s.fc.received)
        s.fc.set_mode_external(person_mode)
        await s.run(5.0)
        assert s.fc.mode == person_mode
        assert ground_modes(s.fc, mark) == []


async def test_a_repeating_low_battery_does_not_switch_the_stop_watch_off() -> None:
    async with stack() as s:
        s.fly()
        s.fc.charge_state = mavlink.MAV_BATTERY_CHARGE_STATE_LOW  # repeated at 2 Hz
        await s.run(3.0)  # the signal began well before the STOP
        s.stop()
        await s.run(1.0)
        s.fc.set_mode_external(CopterMode.GUIDED)  # a late order from onboard
        await s.run(1.0)
        assert s.fc.mode in STOP_MODES


async def test_a_new_failsafe_after_stop_stands() -> None:
    async with stack() as s:
        s.fly()
        await s.run(0.5)
        s.stop()
        await s.run(1.0)
        s.fc.charge_state = mavlink.MAV_BATTERY_CHARGE_STATE_LOW  # begins now
        await s.run(0.6)
        s.fc.set_mode_external(CopterMode.SMART_RTL)
        await s.run(2.0)
        assert s.fc.mode == CopterMode.SMART_RTL


async def test_dispatch_from_many_threads_keeps_latch_and_numbers_in_step() -> None:
    import threading

    async with stack() as s:
        s.fly()
        await s.run(0.5)
        replies: list[int] = []
        lock = threading.Lock()

        def press(kind: str, n: int) -> None:
            for i in range(40):
                r = s.actions.dispatch(
                    kind,  # type: ignore[arg-type]
                    VID,
                    client_action_id=f"{kind}-{n}-{i}",
                    via="button",
                    by=PILOT,
                )
                with lock:
                    replies.append(r.action_seq)

        threads = [
            threading.Thread(target=press, args=(k, n))
            for n, k in enumerate(("stop", "return_home") * 3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(replies)) == len(replies) == 240
        latch = s.latches.get(VID)
        assert latch is not None
        assert latch.action_seq == max(replies) == s.latches.override_seq(VID)


async def test_rc_switch_to_loiter_during_fly_home_stands() -> None:
    """LOITER is a mode the onboard computer could also ask for, but the RC mode switch
    moved: a person chose it, and the RC transmitter goes before everything."""
    async with stack() as s:
        s.fly()
        s.fc.rc_mode_pwm = 1100
        await s.run(0.5)
        s.home()
        await s.run(1.0)
        mark = len(s.fc.received)
        s.fc.rc_mode_switch(1500, CopterMode.LOITER)
        await s.run(5.0)
        assert s.fc.mode == CopterMode.LOITER
        assert ground_modes(s.fc, mark) == []


async def test_rc_switch_to_rtl_during_stop_stands() -> None:
    async with stack() as s:
        s.fly()
        s.fc.rc_mode_pwm = 1100
        await s.run(0.5)
        s.stop()
        await s.run(1.0)
        s.fc.rc_mode_switch(1900, CopterMode.RTL)
        await s.run(3.0)
        assert s.fc.mode == CopterMode.RTL


async def test_late_loiter_without_rc_movement_is_still_undone_during_fly_home() -> None:
    async with stack() as s:
        s.fly()
        s.fc.rc_mode_pwm = 1100  # an RC receiver is there, but nobody touches the switch
        await s.run(0.5)
        s.home()
        await s.run(1.0)
        s.fc.set_mode_external(CopterMode.LOITER)
        await s.run(1.0)
        assert s.fc.mode in HOME_MODES


async def test_failed_takeover_raises_an_alarm() -> None:
    async with stack() as s:
        s.fly()
        s.fc.refuse_modes = {int(CopterMode.LOITER)}
        await s.run(0.5)
        s.actions.takeover(VID, client_action_id="t5", by=PILOT)
        await s.run(2.0)
        assert "alarm.TAKEOVER_FAILED" in s.g.types()


async def test_rc_switch_flipped_during_the_steps_stops_them() -> None:
    """The pilot flips the RC switch while the chain still steps (BRAKE refused, no
    answer): nothing more is sent, and the RC choice stands."""
    async with stack() as s:
        s.fly()
        s.fc.rc_mode_pwm = 1100
        s.fc.refuse_modes = {int(CopterMode.BRAKE)}
        s.fc.drop_acks = 1000
        await s.run(0.5)
        s.stop()
        await s.run(0.3)
        s.fc.rc_mode_switch(1500, CopterMode.ALT_HOLD)
        mark = len(s.fc.received)
        await s.run(4.0)
        assert ground_modes(s.fc, mark) == []
        assert s.fc.mode == CopterMode.ALT_HOLD
