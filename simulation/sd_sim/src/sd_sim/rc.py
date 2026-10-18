"""The simulated pilot's RC transmitter: SITL's RC input over UDP.

A real flight needs a physical transmitter that the flight controller hears, with the
sticks centred while the drone flies. SITL without RC input keeps the throttle at its
minimum, so every hold in LOITER (after a skill, a STOP, a take-over) would sink to the
ground. The launcher's observer therefore plays the pilot: throttle down before take-off
and while landing (arming needs it low), centred from leaving the ground on (1500 us
holds the height in LOITER), the other sticks centred. It never moves a stick on its own.
"""

from __future__ import annotations

import socket
import struct
import threading

__all__ = ["LANDING_MODES", "RcTransmitter", "has_flown", "throttle_for"]

LANDING_MODES = frozenset({"RTL", "SMART_RTL", "LAND"})
AIRBORNE_M = 3.0
THROTTLE_DOWN = 1000
CENTRE = 1500
SEND_PERIOD_S = 0.02  # 50 Hz, like a transmitter


def has_flown(flown: bool, armed: bool, rel_alt_m: float) -> bool:
    """Whether this flight has left the ground: set above 3 m, cleared when disarmed."""
    return armed and (flown or rel_alt_m > AIRBORNE_M)


def throttle_for(armed: bool, flown: bool, mode: str) -> int:
    """Centred from leaving the ground until a landing mode or disarming (so a hold in
    LOITER low over the ground holds too), down before take-off and while landing."""
    return CENTRE if armed and flown and mode not in LANDING_MODES else THROTTLE_DOWN


class RcTransmitter:
    """Sends eight channels (roll, pitch, throttle, yaw, mode switch, ...) at 50 Hz."""

    def __init__(self, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = ("127.0.0.1", port)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rc", daemon=True)
        self.throttle = THROTTLE_DOWN

    def start(self) -> RcTransmitter:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(2)
        self._sock.close()

    def _run(self) -> None:
        while not self._stop.wait(SEND_PERIOD_S):
            packet = struct.pack("<8H", CENTRE, CENTRE, self.throttle, *([CENTRE] * 5))
            self._sock.sendto(packet, self._addr)
