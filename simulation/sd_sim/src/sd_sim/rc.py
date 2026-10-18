"""The pilot's RC transmitter in the simulator: SITL's RC input over UDP.

SITL has no transmitter, and without RC input its throttle stays at the minimum, so
LOITER (the pilot's take-over, STOP's fallback) sinks to the ground. :class:`RcSticks`
plays a pilot who holds the transmitter but does not steer: throttle down on the ground
(arming needs it), centred in the air (1500 us holds the height in LOITER), the other
sticks centred. :func:`pilot_throttle` is that rule.
"""

from __future__ import annotations

import socket
import struct
import threading

__all__ = ["RcSticks", "pilot_throttle"]

THROTTLE_LOW = 1000
THROTTLE_CENTRE = 1500
AIRBORNE_M = 3.0
LANDING_MODES = frozenset({"RTL", "SMART_RTL", "LAND"})


def pilot_throttle(*, armed: bool, alt_m: float, mode: str) -> int:
    """Centred in the air, down on the ground and while landing."""
    airborne = armed and alt_m > AIRBORNE_M
    return THROTTLE_CENTRE if airborne and mode not in LANDING_MODES else THROTTLE_LOW


class RcSticks:
    """Sends SITL's eight RC channels at 50 Hz from a thread."""

    def __init__(self, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = ("127.0.0.1", port)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rc-sticks", daemon=True)
        self.throttle = THROTTLE_LOW
        self.mode_pwm = 1500  # channel 5, ArduCopter's flight-mode switch

    def __enter__(self) -> RcSticks:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(2)
        self._sock.close()

    def _run(self) -> None:
        while not self._stop.wait(0.02):
            packet = struct.pack(
                "<8H", 1500, 1500, self.throttle, 1500, self.mode_pwm, 1500, 1500, 1500
            )
            self._sock.sendto(packet, self._addr)
