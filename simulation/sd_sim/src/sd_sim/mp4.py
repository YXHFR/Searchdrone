"""What an MP4 recording holds, read from its boxes: the video codec and the length.

Enough for the end-to-end test's check of a mission's video (MediaMTX records
fragmented MP4): the sample entry of each track (``avc1`` = H.264) and the length from
``mdhd``, or, in a fragmented file whose ``mdhd`` says 0, the sum of the fragments'
sample durations (``tfhd`` defaults and ``trun`` entries).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Mp4Info", "Mp4Track", "probe"]

_CONTAINERS = frozenset({b"moov", b"trak", b"mdia", b"minf", b"stbl", b"moof", b"traf", b"mvex"})


@dataclass
class Mp4Track:
    track_id: int = 0
    handler: str = ""
    codec: str = ""
    timescale: int = 0
    duration: int = 0  # in timescale units (mdhd, else the fragments)
    default_sample_duration: int = 0  # trex

    @property
    def seconds(self) -> float:
        return self.duration / self.timescale if self.timescale else 0.0


@dataclass
class Mp4Info:
    tracks: list[Mp4Track] = field(default_factory=list[Mp4Track])

    def video(self) -> Mp4Track | None:
        return next((t for t in self.tracks if t.handler == "vide"), None)


def _boxes(data: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1:
            if pos + 16 > end:
                return
            (size,) = struct.unpack_from(">Q", data, pos + 8)
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            return
        yield kind, pos + header, pos + size
        pos += size


def _full(data: bytes, at: int) -> tuple[int, int]:
    """(version, flags) of a full box whose body starts at ``at``."""
    word = struct.unpack_from(">I", data, at)[0]
    return word >> 24, word & 0xFFFFFF


def _mdhd(data: bytes, body: int, track: Mp4Track) -> None:
    version, _ = _full(data, body)
    if version == 1:
        track.timescale, track.duration = struct.unpack_from(">IQ", data, body + 20)
    else:
        track.timescale, track.duration = struct.unpack_from(">II", data, body + 12)


def _trun(data: bytes, body: int, default: int) -> int:
    _, flags = _full(data, body)
    (count,) = struct.unpack_from(">I", data, body + 4)
    pos = body + 8
    if flags & 0x1:
        pos += 4  # data offset
    if flags & 0x4:
        pos += 4  # first sample flags
    if not flags & 0x100:
        return count * default
    per = 4 * sum(1 for bit in (0x100, 0x200, 0x400, 0x800) if flags & bit)
    total = 0
    for i in range(count):
        total += struct.unpack_from(">I", data, pos + i * per)[0]
    return total


def _tfhd(data: bytes, body: int) -> tuple[int, int | None]:
    _, flags = _full(data, body)
    (track_id,) = struct.unpack_from(">I", data, body + 4)
    pos = body + 8
    if flags & 0x1:
        pos += 8  # base data offset
    if flags & 0x2:
        pos += 4  # sample description index
    duration = struct.unpack_from(">I", data, pos)[0] if flags & 0x8 else None
    return track_id, duration


def probe(path: Path) -> Mp4Info:
    data = path.read_bytes()
    info = Mp4Info()
    by_id: dict[int, Mp4Track] = {}
    fragments: dict[int, int] = {}

    def walk(start: int, end: int, track: Mp4Track | None, frag: list[int]) -> None:
        for kind, body, stop in _boxes(data, start, end):
            if kind == b"trak":
                t = Mp4Track()
                info.tracks.append(t)
                walk(body, stop, t, frag)
                by_id[t.track_id] = t
            elif kind == b"traf":
                walk(body, stop, None, [0, 0])
            elif kind in _CONTAINERS:
                walk(body, stop, track, frag)
            elif kind == b"tkhd" and track is not None:
                version, _ = _full(data, body)
                track.track_id = struct.unpack_from(
                    ">I", data, body + (20 if version == 1 else 12)
                )[0]
            elif kind == b"mdhd" and track is not None:
                _mdhd(data, body, track)
            elif kind == b"hdlr" and track is not None:
                track.handler = data[body + 8 : body + 12].decode("ascii", errors="replace")
            elif kind == b"stsd" and track is not None:
                entries = list(_boxes(data, body + 8, stop))
                if entries:
                    track.codec = entries[0][0].decode("ascii", errors="replace")
            elif kind == b"trex":
                track_id, _, default = struct.unpack_from(">III", data, body + 4)
                fragments.setdefault(-track_id, default)
            elif kind == b"tfhd":
                track_id, default = _tfhd(data, body)
                frag[0] = track_id
                frag[1] = default if default is not None else fragments.get(-track_id, 0)
            elif kind == b"trun":
                fragments[frag[0]] = fragments.get(frag[0], 0) + _trun(data, body, frag[1])

    walk(0, len(data), None, [0, 0])
    for track_id, t in by_id.items():
        t.default_sample_duration = fragments.get(-track_id, 0)
        if t.duration == 0:
            t.duration = fragments.get(track_id, 0)
    return info
