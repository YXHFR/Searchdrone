"""The simulator's video pieces without GStreamer or MediaMTX.

The time code strip, the pipeline descriptions and how the launcher starts
``video_relay`` (password only in a 0600 file, never on the command line).
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import pytest

from sd_sim import config, launch, testvideo, timecode, util, video_relay
from sd_sim.errors import InfraError

W, H = 640, 480
T = 1_792_000_123_456


def frame() -> bytearray:
    return bytearray([77]) * (W * H * 3 // 2)


def test_bits_round_trip() -> None:
    bits = timecode.bits_of(T)
    assert len(bits) == 48
    assert timecode.value_of(bits) == T
    assert timecode.bits_of(1)[-1] == 1 and sum(timecode.bits_of(1)) == 1
    assert timecode.bits_of(2**47)[0] == 1
    assert timecode.value_of(timecode.bits_of(2**50 + 5)) == 5  # only the 48 lowest bits


def test_stamp_then_decode() -> None:
    f = frame()
    timecode.stamp_i420(f, W, H, T)
    assert timecode.decode_luma(f[: W * H], W) == T
    # The strip covers the top 32 rows; below it the picture is untouched.
    assert f[32 * W] == 77
    # Neutral chroma under the strip, untouched below it.
    u0 = W * H
    assert f[u0] == timecode.NEUTRAL_UV and f[u0 + 16 * (W // 2)] == 77


def test_layout_two_rows_of_24() -> None:
    f = frame()
    timecode.stamp_i420(f, W, H, 1)  # only the last box (row 2, column 24) is white
    white = [(x, y) for y in range(32) for x in range(W) if f[y * W + x] == 255]
    assert min(x for x, _ in white) == 23 * 16 and max(x for x, _ in white) == 24 * 16 - 1
    assert min(y for _, y in white) == 16 and max(y for _, y in white) == 31


def test_one_uncertain_box_is_accepted_two_are_not() -> None:
    f = frame()
    timecode.stamp_i420(f, W, H, T)
    luma = bytearray(f[: W * H])

    def grey(index: int) -> None:
        row, col = divmod(index, 24)
        for y in range(row * 16, row * 16 + 16):
            luma[y * W + col * 16 : y * W + col * 16 + 16] = bytes([128]) * 16

    grey(3)
    assert timecode.decode_luma(luma, W) is not None
    grey(40)
    assert timecode.decode_luma(luma, W) is None


def test_too_narrow_or_too_small() -> None:
    with pytest.raises(ValueError, match="får inte plats"):
        timecode.stamp_i420(bytearray(320 * 240 * 2), 320, 240, T)
    with pytest.raises(ValueError, match="I420"):
        timecode.stamp_i420(bytearray(10), W, H, T)


def test_pipeline_descriptions() -> None:
    pic = testvideo.Picture()
    src = testvideo.source_desc(pic)
    assert "videotestsrc is-live=true pattern=ball" in src
    assert "width=640,height=480,framerate=15/1" in src
    sink = testvideo.sink_desc(pic, 5600, x264=True)
    assert "x264enc tune=zerolatency bitrate=800" in sink
    assert "udpsink host=127.0.0.1 port=5600" in sink
    assert "openh264enc bitrate=800000" in testvideo.sink_desc(pic, 5600, x264=False)


def test_relay_pipeline_has_no_credentials() -> None:
    desc = video_relay.pipeline_desc(5602, 8554, "sim-01")
    assert "udpsrc address=127.0.0.1 port=5602" in desc
    assert "location=rtsp://127.0.0.1:8554/drone/sim-01/main" in desc
    assert "protocols=tcp" in desc
    assert "@" not in desc and "user-pw" not in desc
    with pytest.raises(ValueError):
        video_relay.pipeline_desc(5602, 8554, "../x")


def test_read_secret(tmp_path: Path) -> None:
    path = tmp_path / "s"
    path.write_text(" pw \n")
    assert video_relay.read_secret(path) == "pw"
    path.write_text("\n")
    with pytest.raises(ValueError):
        video_relay.read_secret(path)


def test_read_env_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("A=1\nSD_GROUND__VIDEO_PUBLISH_PASSWORD='from-file'\n")
    monkeypatch.delenv("SD_GROUND__VIDEO_PUBLISH_PASSWORD", raising=False)
    assert launch.read_env_value(env, "SD_GROUND__VIDEO_PUBLISH_PASSWORD") == "from-file"
    monkeypatch.setenv("SD_GROUND__VIDEO_PUBLISH_PASSWORD", "from-env")
    assert launch.read_env_value(env, "SD_GROUND__VIDEO_PUBLISH_PASSWORD") == "from-env"
    assert launch.read_env_value(tmp_path / "missing", "B") == ""


class Recorder:
    def __init__(self) -> None:
        self.started: list[tuple[str, list[str]]] = []

    def start(self, name: str, argv: list[str], **_: Any) -> Any:
        self.started.append((name, argv))
        return None


def _stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, password: str) -> launch.Stack:
    (tmp_path / ".env").write_text(f"SD_GROUND__VIDEO_PUBLISH_PASSWORD={password}\n")
    monkeypatch.delenv("SD_GROUND__VIDEO_PUBLISH_PASSWORD", raising=False)
    cfg = config.load({"physics": "builtin", "video": "test", "instance": "0"}, environ={})
    stack = launch.Stack(cfg, tmp_path / "run", observer=False, repo=tmp_path)
    stack.group = Recorder()  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setattr(stack, "_wait", lambda *a, **k: None)
    return stack


def test_relay_skipped_without_mediamtx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _stack(tmp_path, monkeypatch, "pw-123456")
    monkeypatch.setattr(stack, "mediamtx_get", lambda path: None)
    stack.start_video_relay()
    assert stack.group.started == []  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("password", ["", "CHANGE_ME"])
def test_relay_refuses_a_placeholder_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, password: str
) -> None:
    stack = _stack(tmp_path, monkeypatch, password)
    monkeypatch.setattr(stack, "mediamtx_get", lambda path: {})
    with pytest.raises(InfraError, match="SD_GROUND__VIDEO_PUBLISH_PASSWORD"):
        stack.start_video_relay()


def test_relay_password_only_in_a_private_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _stack(tmp_path, monkeypatch, "pw-123456")
    monkeypatch.setattr(stack, "mediamtx_get", lambda path: {})
    stack.start_video_relay()
    ((name, argv),) = stack.group.started  # pyright: ignore[reportAttributeAccessIssue]
    assert name == "video_relay"
    assert not any("pw-123456" in a for a in argv)
    secret = Path(argv[argv.index("--secret-file") + 1])
    assert secret.read_text() == "pw-123456"
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600


@pytest.mark.parametrize(("instance", "video"), [("1", "test"), ("0", "off")])
def test_no_relay_off_instance_zero_or_without_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, instance: str, video: str
) -> None:
    stack = _stack(tmp_path, monkeypatch, "pw-123456")
    stack.cfg.instance = int(instance)
    stack.cfg.video = video
    monkeypatch.setattr(stack, "mediamtx_get", lambda path: {})
    stack.start_video_relay()
    stack.start_video_source()
    assert stack.group.started == []  # pyright: ignore[reportAttributeAccessIssue]


def test_test_picture_runs_the_strip_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _stack(tmp_path, monkeypatch, "pw-123456")
    stack.start_video_source()
    ((name, argv),) = stack.group.started  # pyright: ignore[reportAttributeAccessIssue]
    assert name == "testvideo"
    assert argv[1:] == ["-m", "sd_sim.testvideo", "--port", "5600"]


def test_mediamtx_refused_password_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import urllib.error

    stack = _stack(tmp_path, monkeypatch, "pw-123456")
    seen: list[str] = []

    def refuse(req: Any, timeout: float) -> Any:
        seen.append(req.get_header("Authorization"))
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(util.LOOPBACK_OPENER, "open", refuse)
    with pytest.raises(InfraError, match="SD_GROUND__VIDEO_API_PASSWORD"):
        stack.mediamtx_get("/v3/config/global/get")
    assert seen and seen[0].startswith("Basic ")
