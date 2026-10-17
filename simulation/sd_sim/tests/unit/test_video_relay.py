"""The synthetic camera, its time strip and the RTSP relay, without MediaMTX.

A small fake RTSP server checks the publisher's handshake, that RTP packets arrive
unchanged and that the password is only in the Authorization header.
"""

from __future__ import annotations

import asyncio
import base64
import struct
from pathlib import Path

import pytest

from sd_sim import launch
from sd_sim.video_relay import rtsp
from sd_sim.video_relay.__main__ import main as relay_main
from sd_sim.video_relay.source import HEIGHT, WIDTH, Frames, encoder, gst_argv
from sd_sim.video_relay.timecode import BITS, SQUARE, STRIP_WIDTH, decode_luma, strip_row

KNOWN_MS = 1_792_272_650_651


def test_strip_round_trip() -> None:
    row = strip_row(KNOWN_MS)
    assert len(row) == STRIP_WIDTH == BITS * SQUARE
    value, unsure = decode_luma(lambda x, _y: row[x])
    assert value == KNOWN_MS & ((1 << BITS) - 1)
    assert unsure == 0


def test_strip_is_msb_first() -> None:
    row = strip_row(1)
    assert row[-1] > 128 and row[0] < 128  # only the last square is white
    row = strip_row(1 << 47)
    assert row[0] > 128 and row[-1] < 128


def test_frame_has_strip_and_ball() -> None:
    frames = Frames()
    a, b = frames.make(KNOWN_MS, 0), frames.make(KNOWN_MS + 1, 1)
    assert len(a) == WIDTH * HEIGHT * 3 // 2
    value, unsure = decode_luma(lambda x, y: a[y * WIDTH + x])
    assert (value, unsure) == (KNOWN_MS, 0)
    assert decode_luma(lambda x, y: b[y * WIDTH + x])[0] == KNOWN_MS + 1
    assert frames.ball_at(0) != frames.ball_at(1)  # the picture moves
    # Neutral chroma under the strip.
    u0 = WIDTH * HEIGHT
    assert a[u0 : u0 + STRIP_WIDTH // 2] == bytes([128]) * (STRIP_WIDTH // 2)


def test_frame_too_small_is_refused() -> None:
    with pytest.raises(ValueError, match="för liten"):
        Frames(640, 480)


def test_encoder_choice_and_argv() -> None:
    x264 = encoder(lambda name: name == "x264enc")
    assert x264[0] == "x264enc"
    assert "bitrate=2000" in x264  # at least 800 kb/s
    assert encoder(lambda _name: False)[0] == "openh264enc"
    argv = gst_argv(5600, x264)
    assert argv[0] == "gst-launch-1.0"
    assert "host=127.0.0.1" in argv
    assert "port=5600" in argv
    assert argv.index("x264enc") < argv.index("rtph264pay")  # the strip is drawn before encoding


def test_url_never_carries_credentials() -> None:
    assert rtsp.stream_url(8554, "sim-01") == "rtsp://127.0.0.1:8554/drone/sim-01/main"
    with pytest.raises(ValueError):
        rtsp._host_port("rtsp://publisher:pw@127.0.0.1:8554/x")  # pyright: ignore[reportPrivateUsage]
    pub = rtsp.RtspPublisher(rtsp.stream_url(1, "sim-01"), "publisher", "s3cret-pw", 1)
    assert "s3cret-pw" not in repr(pub)


def test_password_file_placeholders_refused(tmp_path: Path) -> None:
    pw = tmp_path / "pw"
    pw.write_text("CHANGE_ME\n")
    with pytest.raises(SystemExit, match="platshållare"):
        relay_main(["relay", "--udp-port", "1", "--password-file", str(pw)])


def test_bad_vehicle_id_refused(tmp_path: Path) -> None:
    pw = tmp_path / "pw"
    pw.write_text("x" * 20)
    with pytest.raises(SystemExit, match="fordons-id"):
        relay_main(["relay", "--udp-port", "1", "--vehicle", "../x", "--password-file", str(pw)])


def test_launcher_reads_password_from_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(launch.VIDEO_PASSWORD_KEY, raising=False)
    assert launch.video_password(tmp_path) == ""
    (tmp_path / ".env").write_text(f"{launch.VIDEO_PASSWORD_KEY}=CHANGE_ME\n")
    assert launch.video_password(tmp_path) == ""
    (tmp_path / ".env").write_text(f"OTHER=1\n{launch.VIDEO_PASSWORD_KEY}='abc123xyz'\n")
    assert launch.video_password(tmp_path) == "abc123xyz"
    monkeypatch.setenv(launch.VIDEO_PASSWORD_KEY, "from-env-1")
    assert launch.video_password(tmp_path) == "from-env-1"


class FakeRtspServer:
    """Answers ANNOUNCE/SETUP/RECORD and collects interleaved RTP packets."""

    def __init__(self, *, refuse: bool = False) -> None:
        self.refuse = refuse
        self.requests: list[tuple[str, dict[str, str], bytes]] = []
        self.packets: list[bytes] = []
        self.got_packets = asyncio.Event()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                first = await reader.readexactly(1)
                if first == b"$":
                    head = await reader.readexactly(3)
                    self.packets.append(await reader.readexactly(struct.unpack(">H", head[1:])[0]))
                    if len(self.packets) >= 3:
                        self.got_packets.set()
                    continue
                line = (first + await reader.readline()).decode()
                headers: dict[str, str] = {}
                while (raw := (await reader.readline()).decode().strip()) != "":
                    k, _, v = raw.partition(":")
                    headers[k.strip().lower()] = v.strip()
                body = await reader.readexactly(int(headers.get("content-length", "0")))
                self.requests.append((line.strip(), headers, body))
                status = "401 Unauthorized" if self.refuse else "200 OK"
                extra = "Session: abc123;timeout=60\r\n" if line.startswith("SETUP") else ""
                writer.write(
                    f"RTSP/1.0 {status}\r\nCSeq: {headers['cseq']}\r\n{extra}\r\n".encode()
                )
                # Some RTCP from the server, which the publisher must skip.
                writer.write(b"$\x01\x00\x04abcd")
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()


async def _run_publisher(server: FakeRtspServer, packets: list[bytes]) -> rtsp.RtspPublisher:
    srv = await asyncio.start_server(server.handle, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    loop = asyncio.get_running_loop()
    sock_udp, _ = await loop.create_datagram_endpoint(asyncio.DatagramProtocol, ("127.0.0.1", 0))
    udp_port = sock_udp.get_extra_info("sockname")[1]
    sock_udp.close()
    pub = rtsp.RtspPublisher(rtsp.stream_url(port, "sim-01"), "publisher", "pw-123456", udp_port)
    task = asyncio.create_task(pub.run())
    sender, _ = await loop.create_datagram_endpoint(
        asyncio.DatagramProtocol, remote_addr=("127.0.0.1", udp_port)
    )
    try:
        for _ in range(100):
            if pub.stats.recording or pub.stats.refused:
                break
            await asyncio.sleep(0.02)
        for packet in packets:
            sender.sendto(packet)
        if not server.refuse:
            await asyncio.wait_for(server.got_packets.wait(), 3)
    finally:
        sender.close()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        srv.close()
        await srv.wait_closed()
    return pub


def _rtp(seq: int) -> bytes:
    return bytes([0x80, 96]) + struct.pack(">HII", seq, seq * 6000, 1234) + b"payload"


def test_publisher_handshake_and_forwarding() -> None:
    server = FakeRtspServer()
    packets = [_rtp(i) for i in range(3)]
    asyncio.run(_run_publisher(server, [b"not rtp", *packets]))
    methods = [r[0].split()[0] for r in server.requests]
    assert methods[:3] == ["ANNOUNCE", "SETUP", "RECORD"]
    announce = server.requests[0]
    assert announce[1]["content-type"] == "application/sdp"
    assert b"H264/90000" in announce[2]
    assert "interleaved=0-1" in server.requests[1][1]["transport"]
    assert "mode=record" in server.requests[1][1]["transport"]
    assert server.requests[2][1]["session"] == "abc123"
    auth = base64.b64encode(b"publisher:pw-123456").decode()
    assert all(r[1]["authorization"] == f"Basic {auth}" for r in server.requests)
    assert all("pw-123456" not in r[0] for r in server.requests)  # never in the URL
    assert server.packets == packets  # unchanged, in order, the non-RTP datagram dropped


def test_publisher_refused_password() -> None:
    server = FakeRtspServer(refuse=True)
    pub = asyncio.run(_run_publisher(server, []))
    assert pub.stats.refused >= 1
    assert "401" in pub.stats.last_error
    assert server.packets == []
