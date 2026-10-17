"""Synthetic camera for CI and ``video=test``: a moving ball plus the time strip.

Python draws each raw I420 frame (the strip is stamped with ``CLOCK_REALTIME`` the
moment the frame is made) and feeds it to ``gst-launch-1.0`` on stdin, which encodes
H.264 (``x264enc``, else ``openh264enc``) and sends RTP to the camera's UDP port, the
same port the Gazebo camera uses. Nothing secret is on the command line.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import time
from collections.abc import Callable

from sd_sim.video_relay.timecode import SQUARE, STRIP_WIDTH, strip_row

__all__ = ["FPS", "HEIGHT", "WIDTH", "Frames", "encoder", "gst_argv", "run_source"]

WIDTH = 1280  # the 48-square strip needs at least 768 px
HEIGHT = 720
FPS = 15
BITRATE_KBPS = 2000
BALL = 64
_BG_Y, _BG_U, _BG_V = 41, 240, 110  # dark blue
_BALL_Y = 210


def encoder(have: Callable[[str], bool]) -> list[str]:
    """The H.264 encoder elements: x264enc if installed, else openh264enc."""
    if have("x264enc"):
        return [
            "x264enc",
            "tune=zerolatency",
            f"bitrate={BITRATE_KBPS}",
            f"key-int-max={FPS}",
            "speed-preset=ultrafast",
        ]
    return [
        "openh264enc",
        f"bitrate={BITRATE_KBPS * 1000}",
        f"gop-size={FPS}",
        "complexity=low",
        "rate-control=bitrate",
    ]


def gst_argv(port: int, enc: list[str]) -> list[str]:
    frame = WIDTH * HEIGHT * 3 // 2
    return [
        "gst-launch-1.0",
        "-q",
        "fdsrc",
        "fd=0",
        f"blocksize={frame}",
        "!",
        "rawvideoparse",
        "format=i420",
        f"width={WIDTH}",
        f"height={HEIGHT}",
        f"framerate={FPS}/1",
        "!",
        *enc,
        "!",
        "rtph264pay",
        "pt=96",
        "config-interval=1",
        "mtu=1200",
        "!",
        "udpsink",
        "host=127.0.0.1",
        f"port={port}",
        "sync=false",
    ]


def gst_has(element: str) -> bool:
    tool = shutil.which("gst-inspect-1.0")
    if tool is None:
        return False
    res = subprocess.run(  # noqa: S603 - fixed tool, element name from this module
        [tool, "--exists", element], capture_output=True, check=False
    )
    return res.returncode == 0


class Frames:
    """Raw I420 frames: a dark background, a bouncing ball and the time strip."""

    def __init__(self, width: int = WIDTH, height: int = HEIGHT) -> None:
        if width < STRIP_WIDTH or height < 2 * BALL + SQUARE:
            raise ValueError("bilden är för liten för tidskodsremsan")
        self.w, self.h = width, height
        y = bytes([_BG_Y]) * (width * height)
        u = bytes([_BG_U]) * (width * height // 4)
        v = bytes([_BG_V]) * (width * height // 4)
        base = bytearray(y + u + v)
        # Neutral chroma under the strip, so black and white stay black and white.
        cw = width // 2
        for row in range(SQUARE // 2):
            for plane in (width * height, width * height * 5 // 4):
                start = plane + row * cw
                base[start : start + STRIP_WIDTH // 2] = bytes([128]) * (STRIP_WIDTH // 2)
        self.base = bytes(base)
        self.ball_row = bytes([_BALL_Y]) * BALL

    def ball_at(self, n: int) -> tuple[int, int]:
        span_x, span_y = self.w - BALL, self.h - BALL - 2 * SQUARE
        x = (n * 9) % (2 * span_x)
        y = (n * 5) % (2 * span_y)
        return (x if x < span_x else 2 * span_x - x), 2 * SQUARE + (
            y if y < span_y else 2 * span_y - y
        )

    def make(self, ms: int, n: int) -> bytes:
        frame = bytearray(self.base)
        row = strip_row(ms)
        for y in range(SQUARE):
            frame[y * self.w : y * self.w + STRIP_WIDTH] = row
        bx, by = self.ball_at(n)
        for y in range(by, by + BALL):
            frame[y * self.w + bx : y * self.w + bx + BALL] = self.ball_row
        return bytes(frame)


def run_source(port: int, *, seconds: float | None = None) -> int:
    """Send the synthetic picture to ``127.0.0.1:port`` until stopped (or ``seconds``)."""
    argv = gst_argv(port, encoder(gst_has))
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE)  # noqa: S603 - fixed argv
    stdin = proc.stdin
    if stdin is None:
        raise RuntimeError("gst-launch-1.0 fick ingen stdin")
    frames = Frames()
    period = 1.0 / FPS
    start = time.monotonic()
    n = 0
    try:
        while seconds is None or time.monotonic() - start < seconds:
            if proc.poll() is not None:
                return proc.returncode or 1
            stdin.write(frames.make(time.time_ns() // 1_000_000, n))
            stdin.flush()
            n += 1
            delay = start + n * period - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        with contextlib.suppress(BrokenPipeError):
            stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0
