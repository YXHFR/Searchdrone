"""Helpers for the video tests against a running MediaMTX (``just video-up``).

* :class:`VideoChain`: the simulator's video path without a flight controller: the
  synthetic camera with the time code strip (``video=test``, the builtin-physics source),
  the simulated radio (``linksim``) and ``video_relay`` into MediaMTX, started with the
  launcher's own steps.
* :func:`rtsp_samples`: frames read back from MediaMTX over RTSP, decoded with the strip.
* :func:`try_publish`: one publishing attempt with a given user and password.
* :class:`GroundServer`: ``sd-ground serve`` as a process, with a one-time login code.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from sd_sim import config, paths, services, timecode
from sd_sim.gst import load_gst
from sd_sim.launch import VIDEO_API_PASSWORD_KEY, VIDEO_API_USER, Stack, read_env_value
from sd_sim.util import LOOPBACK_OPENER, new_run_dir

REPO = paths.REPO
VID = "sim-01"


def require_video() -> bool:
    """CI sets ``SD_SIM__REQUIRE_VIDEO=1``: a missing MediaMTX then fails instead of skips."""
    return os.environ.get("SD_SIM__REQUIRE_VIDEO") == "1"


def api_request(
    path: str, *, method: str = "GET", body: bytes | None = None, auth: bool = True
) -> tuple[int, dict[str, Any] | None]:
    """(status, JSON body) from MediaMTX's API; status 0 when it does not answer."""
    headers = {"Content-Type": "text/plain"} if body is not None else {}
    if auth:
        password = read_env_value(REPO / ".env", VIDEO_API_PASSWORD_KEY)
        token = base64.b64encode(f"{VIDEO_API_USER}:{password}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    url = f"http://127.0.0.1:9997{path}"
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with LOOPBACK_OPENER.open(req, timeout=3) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except (urllib.error.URLError, OSError, ValueError):
        return 0, None


def api_get(path: str) -> dict[str, Any] | None:
    status, body = api_request(path)
    return body if status == 200 else None


def need_mediamtx() -> dict[str, Any]:
    cfg = api_get("/v3/config/global/get")
    if cfg is None:
        msg = "MediaMTX svarar inte på 127.0.0.1:9997 (kör just video-up)"
        if require_video():
            pytest.fail(msg)
        pytest.skip(msg)
    return cfg


def wait_for(check: Any, timeout: float, what: str) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return
        time.sleep(0.25)
    raise AssertionError(f"{what}: inte inom {timeout} s")


def stream_ready(vehicle_id: str = VID) -> bool:
    state = api_get(f"/v3/paths/get/drone/{vehicle_id}/main")
    return state is not None and state.get("ready") is True


@dataclass
class VideoChain:
    stack: Stack
    run_dir: Path

    @classmethod
    @contextmanager
    def running(cls, **overrides: str) -> Iterator[VideoChain]:
        values = {"physics": "builtin", "gui": "false", "video": "test", "instance": "0"}
        cfg = config.load({**values, **overrides})
        run_dir = new_run_dir(0, "video")
        stack = Stack(cfg, run_dir, observer=False)
        try:
            stack.check_ports_free()
            stack.make_secrets()
            stack.start_link_processes()
            stack.start_video_source()
            stack.start_video_relay()
            yield cls(stack, run_dir)
        finally:
            stack.down()


def _pipeline_samples(desc: str, count: int, timeout_s: float) -> list[int | None]:
    gst = load_gst()
    pipe = gst.parse_launch(desc)
    sink = pipe.get_by_name("s")
    pipe.set_state(gst.State.PLAYING)
    out: list[int | None] = []
    end = time.monotonic() + timeout_s
    try:
        while len(out) < count and time.monotonic() < end:
            sample = sink.emit("try-pull-sample", gst.SECOND)
            if sample is None:
                continue
            buf = sample.get_buffer()
            caps = sample.get_caps().get_structure(0)
            width = caps.get_value("width")
            data = buf.extract_dup(0, buf.get_size())
            now = time.time_ns() // 1_000_000
            value = timecode.decode_luma(data, width)
            out.append(None if value is None else now - value)
    finally:
        pipe.set_state(gst.State.NULL)
    return out


def rtsp_samples(count: int, timeout_s: float = 30.0) -> list[int | None]:
    """Delay (ms) per frame read from MediaMTX over RTSP; None for unreadable frames."""
    desc = (
        f"rtspsrc location=rtsp://127.0.0.1:8554/drone/{VID}/main protocols=tcp latency=0 "
        "! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw,format=GRAY8 "
        "! appsink name=s sync=false max-buffers=1 drop=true"
    )
    samples = _pipeline_samples(desc, count + 10, timeout_s)
    # The frames before the first key frame decode to grey; drop that start.
    first = next((i for i, s in enumerate(samples) if s is not None), len(samples))
    return samples[first:][:count]


def rtsp_frame_count(seconds: float) -> int:
    """Frames decoded from MediaMTX over RTSP in ``seconds`` (after the first one)."""
    gst = load_gst()
    desc = (
        f"rtspsrc location=rtsp://127.0.0.1:8554/drone/{VID}/main protocols=tcp latency=0 "
        "! rtph264depay ! h264parse ! avdec_h264 ! appsink name=s sync=false"
    )
    pipe = gst.parse_launch(desc)
    sink = pipe.get_by_name("s")
    pipe.set_state(gst.State.PLAYING)
    count = 0
    try:
        if sink.emit("try-pull-sample", 10 * gst.SECOND) is None:
            return 0
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if sink.emit("try-pull-sample", gst.SECOND) is not None:
                count += 1
    finally:
        pipe.set_state(gst.State.NULL)
    return count


def delay_summary(samples: list[int | None]) -> dict[str, Any]:
    valid = [s for s in samples if s is not None and abs(s) < 60_000]
    return {
        "samples": len(samples),
        "valid": len(valid),
        "invalid": len(samples) - len(valid),
        "median_ms": statistics.median(valid) if valid else None,
        "max_ms": max(valid) if valid else None,
    }


def try_publish(user: str, password: str, vehicle_id: str, timeout_s: float = 8.0) -> str:
    """Publish a short test picture; returns "published" or the error MediaMTX caused."""
    gst = load_gst()
    desc = (
        "videotestsrc is-live=true ! video/x-raw,width=320,height=240,framerate=10/1 "
        "! x264enc tune=zerolatency speed-preset=ultrafast ! h264parse "
        f"! rtspclientsink name=sink location=rtsp://127.0.0.1:8554/drone/{vehicle_id}/main "
        "protocols=tcp"
    )
    pipe = gst.parse_launch(desc)
    sink = pipe.get_by_name("sink")
    sink.set_property("user-id", user)
    sink.set_property("user-pw", password)
    bus = pipe.get_bus()
    pipe.set_state(gst.State.PLAYING)
    end = time.monotonic() + timeout_s
    result = "timeout"
    try:
        while time.monotonic() < end:
            msg = bus.timed_pop_filtered(200 * gst.MSECOND, gst.MessageType.ERROR)
            if msg is not None:
                err, _ = msg.parse_error()
                result = f"error: {err.message}"
                break
            if stream_ready(vehicle_id):
                result = "published"
                break
    finally:
        pipe.set_state(gst.State.NULL)
    return result


def listeners(proto: str) -> list[tuple[str, int]]:
    """(address, port) of every listening socket, from ``ss`` (``t`` = TCP, ``u`` = UDP)."""
    flag = "-Hltn" if proto == "t" else "-Hlun"
    out = subprocess.run(["ss", flag], capture_output=True, text=True, check=True).stdout
    found: list[tuple[str, int]] = []
    for line in out.splitlines():
        cols = line.split()
        if len(cols) < 4:
            continue
        local = cols[3]
        host, _, port = local.rpartition(":")
        if port.isdigit():
            found.append((host.strip("[]"), int(port)))
    return found


def ffprobe(path: Path) -> dict[str, Any]:
    exe = shutil.which("ffprobe")
    assert exe is not None, "ffprobe saknas (apt-get install ffmpeg)"
    out = subprocess.run(
        [exe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out)


@dataclass
class GroundServer:
    """``sd-ground serve`` with the repository's config (needs ``just db-up migrate``)."""

    run_dir: Path
    env: dict[str, str] = field(default_factory=dict[str, str])
    proc: subprocess.Popen[bytes] | None = None

    def _cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "sd_ground", *args],
            cwd=REPO,
            env={**os.environ, **self.env},
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    def start(self) -> None:
        init = self._cli("init", *(["--lan"] if "SD_GROUND__LAN__ENABLED" in self.env else []))
        assert init.returncode == 0, init.stdout + init.stderr
        log = (self.run_dir / "sd-ground.log").open("wb")
        self.proc = subprocess.Popen(
            services.ground_serve_argv(),
            cwd=REPO,
            env={**os.environ, **self.env},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        wait_for(self._healthy, 60, "sd-ground startar")

    def _healthy(self) -> bool:
        assert self.proc is not None
        if self.proc.poll() is not None:
            tail = (self.run_dir / "sd-ground.log").read_text(errors="replace")[-3000:]
            raise AssertionError(f"sd-ground avslutades:\n{tail}")
        try:
            with LOOPBACK_OPENER.open("http://127.0.0.1:8080/api/v1/health", timeout=2) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def health(self) -> dict[str, Any]:
        with LOOPBACK_OPENER.open("http://127.0.0.1:8080/api/v1/health", timeout=2) as r:
            return json.loads(r.read())

    def login_code(self) -> str:
        res = self._cli("login-link", "--print-code")
        assert res.returncode == 0, res.stderr
        return res.stdout.strip().splitlines()[-1]

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    @classmethod
    @contextmanager
    def running(cls, run_dir: Path, **env: str) -> Iterator[GroundServer]:
        server = cls(run_dir, dict(env))
        try:
            server.start()
            yield server
        finally:
            server.stop()


def playwright(
    base_url: str,
    code: str,
    result: Path,
    *,
    ignore_https: bool = False,
    spec: str = "e2e/video/live.spec.ts",
) -> int:
    """One browser test in ``software/panel/e2e/video``: exit code of Playwright."""
    env = {
        **os.environ,
        "SD_VIDEO_BASE_URL": base_url,
        "SD_VIDEO_LOGIN_CODE": code,
        "SD_VIDEO_RESULT": str(result),
        "SD_VIDEO_IGNORE_HTTPS": "1" if ignore_https else "0",
    }
    cmd = [
        "pnpm",
        "--filter",
        "@searchdrone/panel",
        "exec",
        "playwright",
        "test",
        "-c",
        "playwright.video.config.ts",
        spec,
    ]
    return subprocess.run(cmd, cwd=REPO, env=env, check=False, timeout=180).returncode
