"""Relay H.264 RTP from UDP to MediaMTX as an RTSP publisher (TCP, interleaved).

The packets are forwarded as they arrive: no decoding, no re-encoding, so the relay
adds no picture delay. The publish password is read from a file by the process and
sent only in the ``Authorization`` header; it is never on the command line or in the
URL (``ps`` would show it).

Protocol: ANNOUNCE (SDP) -> SETUP (``RTP/AVP/TCP;interleaved=0-1;mode=record``) ->
RECORD, then each RTP packet as ``$ 0 <len16> <packet>``. Lost connections are retried
with backoff; a refused password (401/403) is logged and retried slower.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import struct
from dataclasses import dataclass, field

__all__ = ["PublishRefusedError", "RelayStats", "RtspPublisher", "sdp", "stream_url"]

RTSP_TIMEOUT_S = 5.0
KEEPALIVE_S = 20.0
MAX_RTP = 65535


class PublishRefusedError(Exception):
    """MediaMTX refused the publisher (wrong password or not allowed)."""


class RtspError(Exception):
    """Unexpected RTSP answer."""


def stream_url(port: int, vehicle_id: str) -> str:
    return f"rtsp://127.0.0.1:{port}/drone/{vehicle_id}/main"


def sdp() -> str:
    return (
        "v=0\r\n"
        "o=- 0 0 IN IP4 127.0.0.1\r\n"
        "s=sd_sim\r\n"
        "c=IN IP4 127.0.0.1\r\n"
        "t=0 0\r\n"
        "m=video 0 RTP/AVP 96\r\n"
        "a=rtpmap:96 H264/90000\r\n"
        "a=fmtp:96 packetization-mode=1\r\n"
        "a=control:trackID=0\r\n"
    )


@dataclass
class RelayStats:
    received: int = 0
    sent: int = 0
    dropped: int = 0
    sessions: int = 0
    refused: int = 0
    last_error: str = ""
    recording: bool = False
    extra: dict[str, int] = field(default_factory=dict)


class _Udp(asyncio.DatagramProtocol):
    def __init__(self, queue: asyncio.Queue[bytes], stats: RelayStats) -> None:
        self.queue = queue
        self.stats = stats

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self.stats.received += 1
        if len(data) < 12 or data[0] >> 6 != 2:  # not RTP version 2
            self.stats.dropped += 1
            return
        if self.queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()
            self.stats.dropped += 1
        self.queue.put_nowait(data)


class RtspPublisher:
    """Publish RTP packets from ``udp_port`` to ``url`` until cancelled."""

    def __init__(
        self,
        url: str,
        user: str,
        password: str,
        udp_port: int,
        *,
        stats: RelayStats | None = None,
    ) -> None:
        self.url = url
        self.udp_port = udp_port
        self.stats = stats or RelayStats()
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        self._auth = f"Basic {token}"
        self._cseq = 0
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=512)

    def __repr__(self) -> str:  # never the password
        return f"RtspPublisher({self.url!r}, udp_port={self.udp_port})"

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: _Udp(self.queue, self.stats), local_addr=("127.0.0.1", self.udp_port)
        )
        delay = 1.0
        try:
            while True:
                try:
                    await self.publish_once()
                    delay = 1.0
                except PublishRefusedError as exc:
                    self.stats.refused += 1
                    self.stats.last_error = str(exc)
                    delay = 10.0
                except (OSError, RtspError, asyncio.IncompleteReadError, TimeoutError) as exc:
                    self.stats.last_error = f"{type(exc).__name__}: {exc}"[:200]
                    delay = min(delay * 2, 10.0)
                self.stats.recording = False
                await asyncio.sleep(delay)
        finally:
            transport.close()

    async def publish_once(self) -> None:
        host, port = _host_port(self.url)
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), RTSP_TIMEOUT_S)
        try:
            session = await self.handshake(reader, writer)
            self.stats.sessions += 1
            self.stats.recording = True
            # Start from fresh packets: old ones would only add delay.
            while not self.queue.empty():
                self.queue.get_nowait()
            drain = asyncio.create_task(_drain(reader))
            try:
                await self._pump(writer, session, drain)
            finally:
                drain.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await drain
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def handshake(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> str:
        body = sdp().encode()
        await self._request(
            reader,
            writer,
            "ANNOUNCE",
            self.url,
            {"Content-Type": "application/sdp"},
            body,
        )
        headers = await self._request(
            reader,
            writer,
            "SETUP",
            f"{self.url}/trackID=0",
            {"Transport": "RTP/AVP/TCP;unicast;interleaved=0-1;mode=record"},
        )
        session = headers.get("session", "").split(";")[0].strip()
        if not session:
            raise RtspError("SETUP gav ingen Session")
        await self._request(reader, writer, "RECORD", self.url, {"Session": session})
        return session

    async def _pump(
        self, writer: asyncio.StreamWriter, session: str, drain: asyncio.Task[None]
    ) -> None:
        loop = asyncio.get_running_loop()
        last_keepalive = loop.time()
        while not drain.done():
            try:
                packet = await asyncio.wait_for(self.queue.get(), 1.0)
            except TimeoutError:
                packet = None
            if packet is not None and len(packet) <= MAX_RTP:
                writer.write(b"$\x00" + struct.pack(">H", len(packet)) + packet)
                self.stats.sent += 1
            if loop.time() - last_keepalive > KEEPALIVE_S:
                writer.write(self._message("OPTIONS", self.url, {"Session": session}, b""))
                last_keepalive = loop.time()
            await writer.drain()
        if exc := drain.exception():
            raise exc

    def _message(self, method: str, url: str, headers: dict[str, str], body: bytes) -> bytes:
        self._cseq += 1
        lines = [f"{method} {url} RTSP/1.0", f"CSeq: {self._cseq}", f"Authorization: {self._auth}"]
        lines += [f"{k}: {v}" for k, v in headers.items()]
        lines.append("User-Agent: sd_sim.video_relay")
        if body:
            lines.append(f"Content-Length: {len(body)}")
        return ("\r\n".join(lines) + "\r\n\r\n").encode() + body

    async def _request(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes = b"",
    ) -> dict[str, str]:
        writer.write(self._message(method, url, headers, body))
        await writer.drain()
        status, reply = await asyncio.wait_for(read_response(reader), RTSP_TIMEOUT_S)
        if status in (401, 403):
            raise PublishRefusedError(f"{method}: {status} (fel publiceringslösenord?)")
        if status != 200:
            raise RtspError(f"{method}: {status}")
        return reply


async def read_response(reader: asyncio.StreamReader) -> tuple[int, dict[str, str]]:
    """One RTSP response (skipping interleaved binary frames); headers in lower case."""
    while True:
        first = await reader.readexactly(1)
        if first == b"$":
            head = await reader.readexactly(3)
            await reader.readexactly(struct.unpack(">H", head[1:])[0])
            continue
        line = first + await reader.readline()
        break
    parts = line.decode("latin-1").split()
    if len(parts) < 2 or not parts[0].startswith("RTSP/") or not parts[1].isdigit():
        raise RtspError(f"ogiltigt svar: {line[:60]!r}")
    headers: dict[str, str] = {}
    while True:
        raw = (await reader.readline()).decode("latin-1").strip()
        if not raw:
            break
        name, _, value = raw.partition(":")
        headers[name.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0") or 0)
    if length:
        await reader.readexactly(length)
    return int(parts[1]), headers


async def _drain(reader: asyncio.StreamReader) -> None:
    """Read and discard what the server sends while recording (RTCP, keepalive answers)."""
    while True:
        status, _ = await read_response(reader)
        if status in (401, 403):
            raise PublishRefusedError(f"keepalive: {status}")


def _host_port(url: str) -> tuple[str, int]:
    rest = url.removeprefix("rtsp://").split("/", 1)[0]
    if "@" in rest:
        raise ValueError("inloggningsuppgifter hör inte hemma i adressen")
    host, _, port = rest.rpartition(":")
    return host, int(port)
