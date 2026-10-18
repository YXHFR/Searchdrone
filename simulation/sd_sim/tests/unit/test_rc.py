"""The simulated pilot's transmitter: throttle centred only while flying."""

from __future__ import annotations

import socket
import struct
import time

import pytest

from sd_sim.rc import CENTRE, THROTTLE_DOWN, RcTransmitter, throttle_for


@pytest.mark.parametrize(
    ("armed", "alt", "mode", "want"),
    [
        (False, 0.0, "GUIDED", THROTTLE_DOWN),  # on the ground: arming needs it low
        (True, 1.0, "GUIDED", THROTTLE_DOWN),  # taking off or touching down
        (True, 40.0, "LOITER", CENTRE),  # LOITER holds its height
        (True, 40.0, "GUIDED", CENTRE),
        (True, 40.0, "SMART_RTL", THROTTLE_DOWN),  # landing
        (True, 40.0, "LAND", THROTTLE_DOWN),
    ],
)
def test_throttle_for(armed: bool, alt: float, mode: str, want: int) -> None:
    assert throttle_for(armed, alt, mode) == want


def test_sends_eight_channels_to_sitl() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as rx:
        rx.bind(("127.0.0.1", 0))
        rx.settimeout(2)
        rc = RcTransmitter(rx.getsockname()[1]).start()
        try:
            rc.throttle = CENTRE
            time.sleep(0.1)
            data, _ = rx.recvfrom(64)
        finally:
            rc.stop()
    channels = struct.unpack("<8H", data)
    assert channels[2] in (THROTTLE_DOWN, CENTRE)
    assert [c for i, c in enumerate(channels) if i != 2] == [CENTRE] * 7
