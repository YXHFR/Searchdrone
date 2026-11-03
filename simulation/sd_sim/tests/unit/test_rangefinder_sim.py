"""The simulated laser: what it measures in the world, its profiles, its faults, and the
MAVLink it hands SITL, read back the way sd-onboard reads the flight controller's."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import pytest
from pymavlink.dialects.v20 import ardupilotmega as mavlink
from shapely.geometry import Polygon

from sd_onboard.config import RangefinderConfig, rangefinder_param_problems
from sd_safety.targets import Plane
from sd_sim import params, paths, rangefinder
from sd_sim.obstacles import Ground, Scene, SimTree
from sd_sim.rangefinder import PROFILES, Faults, Laser, Pose, Reading

HOME = (-35.363261, 149.165230)
HOME_ALT = 584.0
TREE = SimTree(50.0, 50.0, 24.0)
KNOWN = paths.REPO / "simulation" / "sitl" / "params" / "known_params_Copter-4.7.1.txt"


def _scene() -> Scene:
    return Scene(
        HOME,
        HOME_ALT,
        300.0,
        Ground(),
        (TREE,),
        water=(Polygon([(100, -50), (200, -50), (200, 50), (100, 50)]),),
        light=(Polygon([(-200, -200), (-100, -200), (-100, -100), (-200, -100)]),),
    )


def _laser(model: str = "grf250", **faults: Any) -> Laser:
    return Laser(_scene(), PROFILES[model], Faults(**faults), seed=1)


def test_beam_points_down_the_drones_own_axis() -> None:
    assert rangefinder.beam(Pose(0, 0, 0)) == pytest.approx((0.0, 0.0, -1.0))
    east, north, up = rangefinder.beam(Pose(0, 0, 0, pitch=math.radians(10)))
    assert north == pytest.approx(math.sin(math.radians(10))) and east == pytest.approx(0.0)
    east, north, up = rangefinder.beam(Pose(0, 0, 0, roll=math.radians(10), yaw=math.pi / 2))
    assert north == pytest.approx(math.sin(math.radians(10)))  # rolled right, facing east
    assert up == pytest.approx(-math.cos(math.radians(10)))


def test_grf250_sees_treetops_and_ground_from_search_height() -> None:
    laser = _laser()
    over_tree = laser.read(Pose(TREE.east, TREE.north, 70.0), 0.0)
    assert over_tree is not None and over_tree.kind == "tree" and over_tree.quality == 100
    assert over_tree.distance_m == pytest.approx(46.0, abs=0.5)
    meadow = laser.read(Pose(0.0, 0.0, 200.0), 0.0)
    assert meadow is not None and meadow.distance_m == pytest.approx(200.0, abs=0.5)
    # Under its 10 m minimum it reports what it measures; SITL marks it out of range.
    close = laser.read(Pose(0.0, 0.0, 6.0), 0.0)
    assert close is not None and close.distance_m < PROFILES["grf250"].min_m
    beyond = laser.read(Pose(0.0, 0.0, 260.0), 0.0)
    assert beyond == Reading(251.0, rangefinder.QUALITY_NONE, "none")


def test_tf02_reaches_13_5_m_over_dark_ground_and_40_m_over_light() -> None:
    laser = _laser("tf02")
    for z in (3.0, 13.0):
        r = laser.read(Pose(0.0, 0.0, z), 0.0)
        assert r is not None and r.kind == "ground"
        assert abs(r.distance_m - z) <= (0.05 if z < 5 else 0.01 * z) + 1e-9
    assert laser.read(Pose(0.0, 0.0, 21.0), 0.0) == Reading(41.0, 0, "none")
    light = laser.read(Pose(-150.0, -150.0, 21.0), 0.0)
    assert light is not None and light.distance_m == pytest.approx(21.0, abs=0.22)
    assert laser.read(Pose(-150.0, -150.0, 45.0), 0.0) == Reading(41.0, 0, "none")


def test_noise_stays_inside_the_profile() -> None:
    laser = _laser("grf250")
    values = [laser.read(Pose(0.0, 0.0, 50.0), 0.0) for _ in range(200)]
    distances = [v.distance_m for v in values if v is not None]
    assert max(distances) <= 50.5 and min(distances) >= 49.5
    assert max(distances) - min(distances) > 0.5  # noisy, not a constant


def test_dropout_and_out_of_range_from_their_time() -> None:
    laser = _laser(dropout_at_s=5.0)
    assert laser.read(Pose(0, 0, 30), 4.9) is not None
    assert laser.read(Pose(0, 0, 30), 5.0) is None
    laser = _laser(out_of_range_at_s=5.0)
    assert laser.read(Pose(0, 0, 30), 5.0) == Reading(251.0, 0, "none")


def test_stuck_keeps_the_value_it_had_while_the_drone_moves() -> None:
    laser = _laser(stuck_at_s=1.0)
    first = laser.read(Pose(0, 0, 30), 0.5)
    assert first is not None
    for z in (40.0, 60.0, 80.0):
        r = laser.read(Pose(0, 0, z), 2.0)
        assert r is not None and r.kind == "stuck"
        assert round(r.distance_m * 100) == round(first.distance_m * 100)


def test_garbage_is_nonsense_the_reader_must_refuse() -> None:
    laser = _laser(garbage_at_s=0.0)
    readings = [laser.read(Pose(0, 0, 30), 1.0) for _ in range(200)]
    assert all(r is not None and r.kind == "garbage" for r in readings)
    distances = {r.distance_m for r in readings if r is not None}
    assert {0.0, 655.35} <= distances
    assert 1 in {r.quality for r in readings if r is not None}  # "invalid signal"


def test_spikes_water_and_gaps() -> None:
    spiky = _laser(spikes=1.0)
    r = spiky.read(Pose(0, 0, 50), 0.0)
    assert r is not None and r.kind == "spike" and 10.0 <= r.distance_m <= 35.0
    wet = _laser(water=True)
    assert wet.read(Pose(150.0, 0.0, 40.0), 0.0) == Reading(251.0, 0, "none")
    dry = _laser()
    r = dry.read(Pose(150.0, 0.0, 40.0), 0.0)
    assert r is not None and r.kind == "ground"
    gaps = _laser(gap_rate=1.0)
    r = gaps.read(Pose(TREE.east, TREE.north, 60.0), 0.0)
    assert r is not None and r.kind == "ground" and r.distance_m == pytest.approx(60.0, abs=0.5)


def test_a_fault_switched_on_and_off_while_flying() -> None:
    laser = _laser()
    laser.set_fault("dropout", 10.0)
    assert laser.read(Pose(0, 0, 30), 9.0) is not None
    assert laser.read(Pose(0, 0, 30), 10.0) is None
    laser.set_fault(None, 12.0)
    assert laser.read(Pose(0, 0, 30), 12.0) is not None


def test_faults_from_a_scenario() -> None:
    faults = Faults.from_mapping({"rf_dropout_at_s": 30, "rf_spikes": 0.1, "rf_water": True})
    assert faults == Faults(dropout_at_s=30.0, spikes=0.1, water=True)
    with pytest.raises(ValueError):
        Faults.from_mapping({"rf_typo": 1})
    with pytest.raises(ValueError):
        Faults(gap_rate=1.5)


def test_distance_sensor_is_what_ardupilots_mavlink_driver_takes() -> None:
    profile = PROFILES["tf02"]
    msg = rangefinder.distance_sensor(Reading(12.345, 100, "ground"), profile, 1234)
    mav = mavlink.MAVLink(None, srcSystem=1, srcComponent=rangefinder.COMPID)
    back: Any = mavlink.MAVLink(None).parse_char(msg.pack(mav))
    assert back.get_type() == "DISTANCE_SENSOR"
    assert (back.current_distance, back.min_distance, back.max_distance) == (1234, 10, 4000)
    assert back.orientation == 25 and back.signal_quality == 100
    huge = rangefinder.distance_sensor(Reading(1000.0, 0, "garbage"), profile, 0)
    assert huge.current_distance == 65535


def test_param_layer_is_what_sd_onboard_accepts(tmp_path: Path) -> None:
    known = set(KNOWN.read_text(encoding="utf-8").split())
    for model in ("tf02", "grf250"):
        for qgc in (False, True):
            layer = params.parse_parm(rangefinder.param_layer(model, qgc))
            assert set(layer) <= known, set(layer) - known
            assert layer["RNGFND1_TYPE"] == 10 and layer["RNGFND1_ORIENT"] == 25
            assert f"MAV{rangefinder.mav_channel(qgc)}_OPTIONS" in layer
            low, high = PROFILES[model].tested_m
            assert (layer["RNGFND1_MIN"], layer["RNGFND1_MAX"]) == (low, high)
            cfg = RangefinderConfig(model="other", min_tested_m=low, max_tested_m=high)
            assert rangefinder_param_problems(cfg, layer) == []
    path = rangefinder.write_param_layer(tmp_path, "grf250", False)
    assert path.read_text(encoding="utf-8") == rangefinder.param_layer("grf250", False)


# --- the MAVLink side ------------------------------------------------------------------


class FakeSocket:
    def __init__(self) -> None:
        self.sent: list[Any] = []
        self.parser = mavlink.MAVLink(None)

    def sendto(self, data: bytes, addr: tuple[str, int]) -> None:
        self.sent.append(self.parser.parse_char(data))


def _sim_state(east: float, north: float, z: float) -> Any:
    lat, lon = Plane(*HOME).latlon(east, north)
    msg = mavlink.MAVLink_sim_state_message(
        1, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, HOME_ALT + z, 0, 0, 0, 0, 0,
        round(lat * 1e7), round(lon * 1e7),
    )  # fmt: skip
    msg._header = mavlink.MAVLink_header(msg.id, srcSystem=1, srcComponent=1)
    return msg


def _heartbeat() -> Any:
    msg = mavlink.MAVLink_heartbeat_message(2, 3, 0, 0, 0, 3)
    msg._header = mavlink.MAVLink_header(msg.id, srcSystem=1, srcComponent=1)
    return msg


def _service(tmp_path: Path, **faults: Any) -> tuple[rangefinder.Service, FakeSocket]:
    sock = FakeSocket()
    service = rangefinder.Service(
        _laser(**faults),
        Plane(*HOME),
        HOME_ALT,
        sock,  # type: ignore[arg-type]
        control=tmp_path / "control.json",
        log=tmp_path / "log.csv",
    )
    service._peer = ("127.0.0.1", 9)  # SITL's side, as learnt from its first datagram
    return service, sock


def test_the_service_answers_sim_state_with_distance_sensor(tmp_path: Path) -> None:
    service, sock = _service(tmp_path)
    service.handle(_sim_state(0.0, 0.0, 40.0))  # before the flight controller's heartbeat
    assert sock.sent == []
    service.handle(_heartbeat())
    service.handle(_sim_state(TREE.east, TREE.north, 70.0))
    (msg,) = sock.sent
    assert msg.get_type() == "DISTANCE_SENSOR"
    assert msg.get_srcSystem() == 1 and msg.get_srcComponent() == rangefinder.COMPID
    assert msg.current_distance == pytest.approx(4600, abs=60)


def test_the_service_asks_for_sim_state_at_the_lasers_rate(tmp_path: Path) -> None:
    service, sock = _service(tmp_path)
    service._ask()
    assert sock.sent == []  # no flight controller yet
    service.handle(_heartbeat())
    service._ask()
    (ask,) = sock.sent
    assert ask.command == mavlink.MAV_CMD_SET_MESSAGE_INTERVAL
    assert ask.param1 == mavlink.MAVLINK_MSG_ID_SIM_STATE and ask.param2 == pytest.approx(20_000)
    service._ask()
    assert len(sock.sent) == 1  # not again at once


def test_the_control_file_switches_a_fault(tmp_path: Path) -> None:
    service, sock = _service(tmp_path)
    service.handle(_heartbeat())
    service.handle(_sim_state(0.0, 0.0, 40.0))
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"fault": "dropout"}), encoding="utf-8")
    service._poll_control()
    service.handle(_sim_state(0.0, 0.0, 40.0))
    assert len(sock.sent) == 1
    control.write_text(json.dumps({"fault": None}), encoding="utf-8")
    later = time.time() + 5
    os.utime(control, (later, later))
    service._poll_control()
    service.handle(_sim_state(0.0, 0.0, 40.0))
    assert len(sock.sent) == 2
