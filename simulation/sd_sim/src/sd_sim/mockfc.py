"""A fake flight controller on SITL's ports: ``physics=mock`` for ``just dev``.

``python -m sd_sim.mockfc --instance N [--params FILE ...]`` runs
:class:`sd_mav.fake.FakeFlightController` (a point mass that speaks real MAVLink 2) in
place of ArduCopter. Like SITL's ``--serial0..2 udpclient:`` it sends to the harness,
mavtap and the radio link of instance N and answers whatever comes back on each
channel, so the harness, sd-onboard and sd-ground see a vehicle without ArduPilot,
Gazebo or a build of either. Its parameters are the safety layers' values (later files
win), so sd-onboard's read-back finds what it expects.

It is for developing the ground station, the panel and the AI against a vehicle that
starts in a second. It is no flight model: failsafes, the fence and the EKF are not
simulated. Anything about flight safety is tested against SITL (``just test-sitl``).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import socket
from pathlib import Path
from typing import Any

from pymavlink.dialects.v20 import ardupilotmega as mavlink

from sd_mav.fake import CMAC_HOME, FakeFlightController, FakeLink
from sd_obs.clock import MonotonicClock
from sd_sim import paths, ports
from sd_sim.params import merge_layers
from sd_sim.util import say

# SITL's serial ports in the launcher's order: serial0 harness, serial1 mavtap,
# serial2 the radio link (sd_sim.sitl.build_sitl_argv).
CHANNELS = ("harness_mavlink", "mavtap_in", "radio_mavlink_in")
DEFAULT_PARAMS = (paths.REPO / "simulation" / "sitl" / "params" / "searchdrone-safety.parm",)


class MockFlightController(FakeFlightController):
    """The fake with the EKF flag ArduCopter reports once it has a position to fly on
    (``EKF_PRED_POS_HORIZ_ABS``), which the harness waits for."""

    def _ekf(self) -> Any:
        msg = super()._ekf()
        msg.flags |= mavlink.EKF_PRED_POS_HORIZ_ABS
        return msg


class UdpLink(FakeLink):
    """One SITL-style ``udpclient`` channel: the flight controller's bytes go to ``dest``,
    and datagrams that come back to this socket go to the flight controller."""

    def __init__(self, fc: FakeFlightController, dest: tuple[str, int]) -> None:
        super().__init__(fc, fc.sysid, 1)
        self.dest = dest
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.setblocking(False)

    def receive(self, data: bytes) -> None:
        with contextlib.suppress(OSError):  # nobody listening yet, as with SITL
            self.sock.sendto(data, self.dest)

    def on_readable(self) -> None:
        while True:
            try:
                data = self.sock.recv(65535)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return  # ICMP port unreachable from an earlier send
            self._fc.deliver(self, data)


def params_from(files: list[Path]) -> dict[str, float]:
    return merge_layers(files)


async def run(instance: int, param_files: list[Path], stop: asyncio.Event) -> None:
    fc = MockFlightController(MonotonicClock(), home=CMAC_HOME)
    fc.params.update(params_from(param_files))
    port_map = ports.for_instance(instance)
    loop = asyncio.get_running_loop()
    links = []
    for name in CHANNELS:
        link = UdpLink(fc, ("127.0.0.1", port_map[name]))
        fc.links.append(link)
        loop.add_reader(link.sock.fileno(), link.on_readable)
        links.append(link)
    fc.start()
    try:
        await stop.wait()
    finally:
        await fc.stop()
        for link in links:
            loop.remove_reader(link.sock.fileno())
            link.sock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sd_sim.mockfc", description=__doc__)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--params", type=Path, nargs="*", default=list(DEFAULT_PARAMS))
    args = parser.parse_args(argv)
    ports.check_instance(args.instance)

    async def _main() -> None:
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        say(f"sd_sim.mockfc: låtsad flygkontroll på instans {args.instance}")
        await run(args.instance, list(args.params), stop)

    asyncio.run(_main())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
