"""L-SPOOF and the GPS health events in the air on the fake."""

from __future__ import annotations

from onboard_helpers import Rig, charter, rig
from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_schemas.base import new_id
from sd_schemas.executor import ExecutorState as S
from sd_schemas.link import OperatorRelease


async def _flying(r: Rig) -> None:
    await r.install(charter())
    await r.takeoff(40)
    await r.run(12)  # past the height baseline after arming


def _source_set_commands(r: Rig, since: int) -> list[int]:
    return [
        int(m.param1)
        for m in r.vehicle.sent[since:]
        if m.get_type() == "COMMAND_LONG" and int(m.command) == mavlink.MAV_CMD_SET_EKF_SOURCE_SET
    ]


def _event(r: Rig, kind: str) -> list[dict[str, object]]:
    return [d for t, _, d in r.up.events if t == f"executor.{kind}"]


async def test_spoofed_height_switches_the_ekf_off_gps() -> None:
    async with rig() as r:
        await _flying(r)
        since = len(r.vehicle.sent)
        r.fc.gps_offset_m = (0.0, 0.0, 40.0)
        await r.run(5)
        assert r.ex.state == S.FAILSAFE and r.ex.control_suspended
        assert r.fc.ekf_source_set == 2
        assert _source_set_commands(r, since) == [2]
        (found,) = _event(r, "gnss_spoof_suspected")
        assert found["kind"] == "altitude" and found["source_set_ack"] == "ACCEPTED"
        assert "L-SPOOF" in r.ex.snapshot().limits_active
        # The pilot flies from here: Resume is refused for the rest of the flight.
        ref = new_id()
        r.ex.on_release(OperatorRelease(action_seq=r.next_seq(), operator_id="operator:pilot"), ref)
        assert r.ex.state == S.FAILSAFE and r.up.stages(ref) == ["rejected"]


async def test_spoofed_jump_switches_the_ekf_off_gps_once() -> None:
    async with rig() as r:
        await _flying(r)
        r.fc.gps_offset_m = (50.0, 0.0, 0.0)
        await r.run(4)
        assert r.ex.state == S.FAILSAFE
        assert [d["kind"] for d in _event(r, "gnss_spoof_suspected")] == ["jump"]
        r.fc.gps_offset_m = (0.0, 0.0, 0.0)
        await r.run(2)
        r.fc.gps_offset_m = (80.0, 0.0, 0.0)
        await r.run(4)
        assert len(_event(r, "gnss_spoof_suspected")) == 1  # once per flight


async def test_spoof_in_held_only_raises_the_alarm() -> None:
    async with rig() as r:
        await _flying(r)
        r.ex.on_override(r.notice("stop"))
        await r.run(1)
        assert r.ex.state == S.HELD
        since = len(r.vehicle.sent)
        r.fc.gps_offset_m = (0.0, 0.0, 40.0)
        await r.run(5)
        assert r.ex.state == S.HELD
        assert _source_set_commands(r, since) == [] and r.fc.ekf_source_set == 1
        (found,) = _event(r, "gnss_spoof_suspected")
        assert found["source_set_ack"] is None


async def test_gps_degraded_and_restored() -> None:
    async with rig() as r:
        await _flying(r)
        r.fc.gps_fix_type = 1
        await r.run(3)
        assert [d["status"] for d in _event(r, "gnss_degraded")] == ["lost"]
        r.fc.gps_fix_type = 3
        await r.run(6)
        assert len(_event(r, "gnss_restored")) == 1
        assert _event(r, "gnss_spoof_suspected") == []
