"""The drone's video into MediaMTX: ``python -m sd_sim.video_relay --instance 0``.

Receives the camera's H.264 RTP from the simulated radio (``video_out_udp``, after
``linksim``) and publishes it, without re-encoding, to
``rtsp://127.0.0.1:<mediamtx_rtsp>/drone/<vehicle_id>/main`` over TCP as the MediaMTX
user ``publisher``. The password is read inside the process from the instance's secret
file; it never appears on the command line (visible in ``ps``) or in the URL.

When MediaMTX is down or refuses the stream, the pipeline is rebuilt after a pause, so
the relay recovers by itself once MediaMTX is back.
"""

from __future__ import annotations

import argparse
import re
import signal
import sys
import threading
from pathlib import Path
from types import FrameType
from typing import Any

from sd_sim import ports
from sd_sim.gst import GstMissingError, load_gst
from sd_sim.util import say

__all__ = ["RETRY_S", "main", "pipeline_desc", "read_secret"]

RETRY_S = 1.0
VEHICLE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,31}$")


def pipeline_desc(udp_port: int, rtsp_port: int, vehicle_id: str) -> str:
    if not VEHICLE_ID.match(vehicle_id):
        raise ValueError(f"ogiltigt vehicle_id {vehicle_id!r}")
    return (
        f'udpsrc address=127.0.0.1 port={udp_port} caps="application/x-rtp,media=video,'
        'encoding-name=H264,payload=96,clock-rate=90000" '
        "! rtpjitterbuffer latency=50 ! rtph264depay ! h264parse config-interval=-1 "
        f"! rtspclientsink name=sink location=rtsp://127.0.0.1:{rtsp_port}/drone/{vehicle_id}/main "
        "protocols=tcp latency=0"
    )


def read_secret(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"{path.name} är tom")
    return value


def _run_once(gst: Any, desc: str, password: str, stop: threading.Event) -> str:
    """Run the pipeline until an error, end of stream or stop. Returns why it ended."""
    pipe = gst.parse_launch(desc)
    sink = pipe.get_by_name("sink")
    sink.set_property("user-id", "publisher")
    sink.set_property("user-pw", password)
    bus = pipe.get_bus()
    pipe.set_state(gst.State.PLAYING)
    reason = "stop"
    try:
        while not stop.is_set():
            msg = bus.timed_pop_filtered(
                200 * gst.MSECOND, gst.MessageType.ERROR | gst.MessageType.EOS
            )
            if msg is None:
                continue
            if msg.type == gst.MessageType.EOS:
                reason = "eos"
            else:
                err, _debug = msg.parse_error()
                reason = f"error: {err.message}"
            break
    finally:
        pipe.set_state(gst.State.NULL)
    return reason


def run(instance: int, secret_file: Path, vehicle_id: str, stop: threading.Event) -> int:
    gst = load_gst()
    p = ports.for_instance(instance)
    desc = pipeline_desc(p["video_out_udp"], p["mediamtx_rtsp"], vehicle_id)
    password = read_secret(secret_file)
    say(
        f"video_relay: 127.0.0.1:{p['video_out_udp']} -> "
        f"rtsp://127.0.0.1:{p['mediamtx_rtsp']}/drone/{vehicle_id}/main"
    )
    while not stop.is_set():
        reason = _run_once(gst, desc, password, stop)
        if stop.is_set():
            break
        say(f"video_relay: strömmen bröts ({reason.replace(password, '***')}); försöker igen")
        stop.wait(RETRY_S)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Publish the simulator's video to MediaMTX.")
    ap.add_argument("--instance", type=int, default=0)
    ap.add_argument("--secret-file", type=Path, required=True)
    ap.add_argument("--vehicle-id", default="sim-01")
    args = ap.parse_args(argv)
    stop = threading.Event()

    def _stop(_sig: int, _frame: FrameType | None) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        return run(args.instance, args.secret_file, args.vehicle_id, stop)
    except (GstMissingError, OSError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
