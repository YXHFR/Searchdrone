"""Turning on the camera stream: a lost request is sent again until frames arrive."""

from __future__ import annotations

import subprocess

import pytest

from sd_sim import gazebo
from sd_sim.errors import InfraError

TOPIC = "/world/w/model/m/link/camera_link/sensor/camera/image/enable_streaming"


class FakeGz:
    """``gz topic``: the camera's topic is listed, every publish exits 0 (as gz-transport's
    does whether or not anyone got the message). A virtual clock replaces sleeping."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.now = 0.0
        self.publishes: list[float] = []
        monkeypatch.setattr(gazebo, "_gz", self.run)
        monkeypatch.setattr("sd_sim.gazebo.time.monotonic", lambda: self.now)
        monkeypatch.setattr("sd_sim.gazebo.time.sleep", self.sleep)

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def run(
        self, args: list[str], instance: int, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        if "-p" in args:
            self.publishes.append(self.now)
            self.now += 0.8  # gz topic -p waits 0.8 s before it publishes
        return subprocess.CompletedProcess(args, 0, stdout=f"/clock\n{TOPIC}\n", stderr="")


def test_a_lost_request_is_sent_again(monkeypatch: pytest.MonkeyPatch) -> None:
    gz = FakeGz(monkeypatch)
    # The first request is lost; the stream runs from the second on.
    topic = gazebo.enable_camera(0, streaming=lambda: len(gz.publishes) >= 2)
    assert topic == TOPIC
    assert len(gz.publishes) == 2
    assert gz.publishes[1] - gz.publishes[0] >= gazebo.CAMERA_CONFIRM_S


def test_a_running_stream_is_asked_once(monkeypatch: pytest.MonkeyPatch) -> None:
    gz = FakeGz(monkeypatch)
    assert gazebo.enable_camera(0, streaming=lambda: True) == TOPIC
    assert len(gz.publishes) == 1


def test_no_frames_at_all_is_an_infrastructure_error(monkeypatch: pytest.MonkeyPatch) -> None:
    gz = FakeGz(monkeypatch)
    with pytest.raises(InfraError, match="inga bilder"):
        gazebo.enable_camera(0, timeout=20.0, streaming=lambda: False)
    assert 3 <= len(gz.publishes) <= 4
    assert gz.now <= 20.0 + gazebo.CAMERA_CONFIRM_S + 1.0


def test_without_a_check_it_publishes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    gz = FakeGz(monkeypatch)
    assert gazebo.enable_camera(0) == TOPIC
    assert len(gz.publishes) == 1
