"""Holding the fake flight controller's HEARTBEAT back: a mode it took shows later, as
ArduCopter's HEARTBEAT (one a second) does."""

from __future__ import annotations

from typing import Any

from sd_mav.fake import FakeFlightController


class HeartbeatHold:
    """While ``held`` no HEARTBEAT goes out; :meth:`release` sends one at once."""

    def __init__(self, fc: FakeFlightController) -> None:
        self.fc = fc
        self.held = False
        send = fc._send

        def held_send(msg: Any, link: Any = None) -> None:
            if not (self.held and msg.get_type() == "HEARTBEAT"):
                send(msg, link)

        fc._send = held_send  # type: ignore[method-assign]

    def release(self) -> None:
        self.held = False
        self.fc._send(self.fc.heartbeat())
