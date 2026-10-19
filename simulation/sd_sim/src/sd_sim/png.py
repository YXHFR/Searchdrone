"""A minimal, deterministic PNG writer (8-bit RGB or RGBA) for generated textures.

The same array always gives the same bytes: no timestamps, a fixed zlib level and no
optional chunks unless ``text`` is given. Rows use PNG filter 0 (none).
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ZLIB_LEVEL = 6


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def encode(image: NDArray[np.uint8], text: Mapping[str, str] | None = None) -> bytes:
    """An 8-bit RGB or RGBA PNG; ``text`` adds Latin-1 ``tEXt`` chunks (keyword: text)."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("väntade en uint8-bild med 3 eller 4 kanaler")
    height, width, channels = image.shape
    color_type = 2 if channels == 3 else 6
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    rows = np.empty((height, 1 + width * channels), dtype=np.uint8)
    rows[:, 0] = 0
    rows[:, 1:] = image.reshape(height, width * channels)
    body = zlib.compress(rows.tobytes(), ZLIB_LEVEL)
    notes = b"".join(
        _chunk(b"tEXt", k.encode("latin-1") + b"\0" + v.encode("latin-1"))
        for k, v in (text or {}).items()
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + notes
        + _chunk(b"IDAT", body)
        + _chunk(b"IEND", b"")
    )


def encode_gray16(image: NDArray[np.uint16]) -> bytes:
    """A 16-bit greyscale PNG (Gazebo heightmaps)."""
    if image.dtype != np.uint16 or image.ndim != 2:
        raise ValueError("väntade en tvådimensionell uint16-bild")
    height, width = image.shape
    header = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    rows = np.zeros((height, 1 + 2 * width), dtype=np.uint8)
    rows[:, 1:] = image.astype(">u2").view(np.uint8).reshape(height, 2 * width)
    body = zlib.compress(rows.tobytes(), ZLIB_LEVEL)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", body)
        + _chunk(b"IEND", b"")
    )


def write(path: Path, image: NDArray[np.uint8], text: Mapping[str, str] | None = None) -> None:
    path.write_bytes(encode(image, text))


def decode(data: bytes) -> NDArray[np.uint8]:
    """Read a PNG written by :func:`encode` or by Gazebo (8-bit RGB/RGBA, any filter)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("inte en PNG-fil")
    pos, idat = 8, b""
    width = height = channels = 0
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, depth, color_type = struct.unpack(">IIBB", body[:10])
            if depth != 8 or color_type not in (2, 6):
                raise ValueError("bara 8-bitars RGB eller RGBA stöds")
            channels = 3 if color_type == 2 else 4
        elif kind == b"IDAT":
            idat += body
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(height, 1 + width * channels)
    out = np.zeros((height, width * channels), dtype=np.int32)
    prev = np.zeros(width * channels, dtype=np.int32)
    for y in range(height):
        kind, line = raw[y, 0], raw[y, 1:].astype(np.int32)
        if kind == 0:
            cur = line
        elif kind == 2:
            cur = (line + prev) & 0xFF
        else:
            cur = np.zeros_like(line)
            for x in range(width * channels):
                a = cur[x - channels] if x >= channels else 0
                b = prev[x]
                c = prev[x - channels] if x >= channels else 0
                if kind == 1:
                    pred = a
                elif kind == 3:
                    pred = (a + b) // 2
                else:
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                cur[x] = (line[x] + pred) & 0xFF
        out[y] = cur
        prev = cur
    return out.reshape(height, width, channels).astype(np.uint8)


def read(path: Path) -> NDArray[np.uint8]:
    return decode(path.read_bytes())
