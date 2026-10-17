"""The synthetic camera: ``python -m sd_sim.testvideo --port <video_raw_udp>``.

With ``video=test`` (builtin physics, CI) this stands in for the Gazebo camera and sends
to the same port: GStreamer's moving ball (``videotestsrc pattern=ball``), 640 x 480 at
15 frames/s, with the time code strip (``sd_sim.timecode``) drawn into every frame just
before the H.264 encoder (``x264enc``, or ``openh264enc`` when the ugly plug-ins are
missing), as RTP to ``127.0.0.1:<port>``.

The frames go through Python between two pipelines (``appsink`` -> strip -> ``appsrc``):
GStreamer hands out read-only buffers, and copying 460 kB per frame is cheap.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from dataclasses import dataclass
from types import FrameType
from typing import Any

from sd_sim import timecode
from sd_sim.gst import GstMissingError, has_element, load_gst
from sd_sim.util import say

__all__ = ["Picture", "encoder_desc", "main", "sink_desc", "source_desc"]

BITRATE_KBPS = 800  # at least 800 kb/s, so the strip survives the encoder
PULL_TIMEOUT_NS = 1_000_000_000


@dataclass(frozen=True)
class Picture:
    width: int = 640
    height: int = 480
    fps: int = 15

    @property
    def caps(self) -> str:
        return (
            f"video/x-raw,format=I420,width={self.width},height={self.height},"
            f"framerate={self.fps}/1"
        )


def source_desc(pic: Picture) -> str:
    return (
        f"videotestsrc is-live=true pattern=ball ! {pic.caps} "
        "! appsink name=raw max-buffers=2 drop=true sync=false emit-signals=false"
    )


def encoder_desc(pic: Picture, *, x264: bool) -> str:
    if x264:
        return (
            f"x264enc tune=zerolatency bitrate={BITRATE_KBPS} key-int-max={pic.fps} "
            "speed-preset=ultrafast"
        )
    return (
        f"openh264enc bitrate={BITRATE_KBPS * 1000} gop-size={pic.fps} complexity=low "
        "rate-control=bitrate"
    )


def sink_desc(pic: Picture, port: int, *, x264: bool) -> str:
    return (
        f'appsrc name=src is-live=true format=time do-timestamp=true caps="{pic.caps}" '
        f"! {encoder_desc(pic, x264=x264)} ! video/x-h264,profile=constrained-baseline "
        "! rtph264pay pt=96 config-interval=1 "
        f"! udpsink host=127.0.0.1 port={port} sync=false async=false"
    )


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def run(port: int, pic: Picture, stop: threading.Event) -> int:
    gst = load_gst()
    x264 = has_element(gst, "x264enc")
    if not x264 and not has_element(gst, "openh264enc"):
        sys.stderr.write("Ingen H.264-kodare (x264enc eller openh264enc) finns.\n")
        return 1
    source = gst.parse_launch(source_desc(pic))
    sink = gst.parse_launch(sink_desc(pic, port, x264=x264))
    raw: Any = source.get_by_name("raw")
    src: Any = sink.get_by_name("src")
    sink.set_state(gst.State.PLAYING)
    source.set_state(gst.State.PLAYING)
    frame_bytes = pic.width * pic.height * 3 // 2
    say(f"testbild: {pic.width}x{pic.height} {pic.fps} fps -> 127.0.0.1:{port}")
    code = 0
    try:
        while not stop.is_set():
            sample = raw.emit("try-pull-sample", PULL_TIMEOUT_NS)
            if sample is None:
                if raw.get_property("eos"):
                    code = 1
                    break
                continue
            buf = sample.get_buffer()
            data = bytearray(buf.extract_dup(0, buf.get_size()))
            if len(data) < frame_bytes:
                continue
            timecode.stamp_i420(data, pic.width, pic.height, _now_ms())
            if src.emit("push-buffer", gst.Buffer.new_wrapped(bytes(data))) != gst.FlowReturn.OK:
                code = 1
                break
    finally:
        source.set_state(gst.State.NULL)
        sink.set_state(gst.State.NULL)
    return code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Synthetic camera with a time code strip.")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args(argv)
    stop = threading.Event()

    def _stop(_sig: int, _frame: FrameType | None) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        return run(args.port, Picture(args.width, args.height, args.fps), stop)
    except GstMissingError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
