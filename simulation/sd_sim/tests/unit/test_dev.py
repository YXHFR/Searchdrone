"""`just dev`: the launcher's --dev flag, physics=mock and the fake flight controller."""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path
from typing import Any

import pytest
from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_sim import config, launch, mockfc, ports
from sd_sim.launch import Stack


def test_dev_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert launch.main(["up", "--dev", "dry=1"]) == 0
    assert '"--dev"' in capsys.readouterr().out


def test_dev_and_demo_do_not_mix() -> None:
    with pytest.raises(SystemExit):
        launch.main(["up", "--dev", "--demo", "dry=1"])


def test_dev_defaults_need_no_simulator_and_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []
    real = config.load

    def load(values: dict[str, Any], **kw: Any) -> config.SimConfig:
        seen.append(dict(values))
        return real(values, **kw, environ={})

    monkeypatch.setattr(config, "load", load)
    assert launch.main(["up", "--dev", "dry=1"]) == 0
    assert seen[-1] == {"physics": "mock", "brain": "scripted", "world": "sd_lake_forest"}
    # A key on the command line wins over the default.
    assert launch.main(["up", "--dev", "physics=builtin", "brain=none", "dry=1"]) == 0
    assert seen[-1]["physics"] == "builtin"
    assert seen[-1]["brain"] == "none"


def test_mock_physics_is_only_for_dev() -> None:
    # Without ArduPilot a smoke flight, demo or test could pass against the fake.
    with pytest.raises(config.ConfigError):
        config.load({"physics": "mock"}, environ={})
    with pytest.raises(SystemExit):
        launch.main(["up", "physics=mock", "dry=1"])
    cfg = config.load({"physics": "mock", "brain": "scripted"}, environ={}, allow_mock=True)
    assert cfg.physics == "mock"
    assert cfg.gui is False  # no Gazebo window
    assert cfg.video == "test"  # no camera without Gazebo


def test_panel_dev_link() -> None:
    link = "http://127.0.0.1:8080/#login=abc"
    assert launch.panel_dev_link(link, 5173) == "http://127.0.0.1:5173/#login=abc"


def test_mock_starts_the_fake_instead_of_arducopter(tmp_path: Path) -> None:
    cfg = config.load({"physics": "mock", "brain": "scripted"}, environ={}, allow_mock=True)
    stack = Stack(cfg, tmp_path, panel_dev=True)
    started: list[list[str]] = []
    stack.group.start = lambda name, argv, **kw: started.append([name, *argv])  # type: ignore[method-assign]
    stack.start_sitl()
    (call,) = started
    assert call[0] == "mockfc"
    assert call[2:4] == ["-m", "sd_sim.mockfc"]
    assert call[-1].endswith("simulation/sitl/params/searchdrone-safety.parm")


def test_the_fake_answers_on_sitls_ports() -> None:
    """Heartbeat and position to the harness port, and a safety parameter read back."""
    instance = 6
    port = ports.for_instance(instance)["harness_mavlink"]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", port))
    sock.settimeout(0.1)
    parser = mavlink.MAVLink(None)
    out = mavlink.MAVLink(None, srcSystem=253, srcComponent=190)

    async def scenario() -> dict[str, Any]:
        stop = asyncio.Event()
        task = asyncio.create_task(mockfc.run(instance, list(mockfc.DEFAULT_PARAMS), stop))
        seen: dict[str, Any] = {}
        sender: tuple[str, int] | None = None
        try:
            for _ in range(60):
                await asyncio.sleep(0.05)
                try:
                    data, sender = sock.recvfrom(65535)
                except TimeoutError:
                    continue
                for msg in parser.parse_buffer(data) or []:
                    seen[msg.get_type()] = msg
                if sender is not None and "PARAM_VALUE" not in seen:
                    req = mavlink.MAVLink_param_request_read_message(1, 1, b"FENCE_ENABLE", -1)
                    sock.sendto(bytes(req.pack(out)), sender)
                if {"HEARTBEAT", "EKF_STATUS_REPORT", "PARAM_VALUE"} <= set(seen):
                    break
        finally:
            stop.set()
            await task
        return seen

    try:
        seen = asyncio.run(scenario())
    finally:
        sock.close()
    assert "HEARTBEAT" in seen
    assert seen["EKF_STATUS_REPORT"].flags & mavlink.EKF_PRED_POS_HORIZ_ABS
    assert seen["PARAM_VALUE"].param_id == "FENCE_ENABLE"
    assert seen["PARAM_VALUE"].param_value == 1.0
