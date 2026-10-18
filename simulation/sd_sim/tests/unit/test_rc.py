"""The simulated pilot's transmitter: throttle centred only while flying."""

from __future__ import annotations

import socket
import struct
import time

import pytest

from sd_sim.rc import CENTRE, THROTTLE_DOWN, RcTransmitter, has_flown, throttle_for


@pytest.mark.parametrize(
    ("armed", "flown", "mode", "want"),
    [
        (False, False, "GUIDED", THROTTLE_DOWN),  # on the ground: arming needs it low
        (True, False, "GUIDED", THROTTLE_DOWN),  # taking off
        (True, True, "LOITER", CENTRE),  # LOITER holds its height
        (True, True, "GUIDED", CENTRE),
        (True, True, "SMART_RTL", THROTTLE_DOWN),  # landing
        (True, True, "LAND", THROTTLE_DOWN),
    ],
)
def test_throttle_for(armed: bool, flown: bool, mode: str, want: int) -> None:
    assert throttle_for(armed, flown, mode) == want


def test_a_low_hold_after_leaving_the_ground_keeps_the_throttle_centred() -> None:
    flown = False
    for alt in (0.0, 1.0, 5.0, 1.5):  # up past 3 m, then a STOP holds it at 1.5 m
        flown = has_flown(flown, True, alt)
    assert flown and throttle_for(True, flown, "LOITER") == CENTRE
    assert not has_flown(flown, False, 0.0)  # disarmed: the next flight starts again


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
