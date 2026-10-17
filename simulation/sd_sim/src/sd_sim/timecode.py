"""The machine-readable time code strip in the synthetic video.

Every frame of the test picture carries the wall clock (``CLOCK_REALTIME``) in
milliseconds, as the 48 lowest bits, most significant bit first, in 48 boxes of
16 x 16 px at the top of the picture: two rows of 24 boxes (a 640 px wide picture has
room for 40 boxes in one row). A white box is 1, a black box 0. The rest of the top
two box rows is black. The panel's ``e2e/helpers/frameClock.ts`` reads it back from the
decoded video and compares it with ``Date.now()``; :func:`decode_luma` does the same
on a luma plane here.

The strip is written into the raw I420 frame before the H.264 encoder, so the measured
delay covers encoding, the simulated radio, MediaMTX, WebRTC and decoding.
"""

from __future__ import annotations

__all__ = [
    "BITS",
    "BOXES_PER_ROW",
    "BOX_PX",
    "STRIP_ROWS",
    "bits_of",
    "decode_luma",
    "stamp_i420",
    "value_of",
]

BITS = 48
BOX_PX = 16
BOXES_PER_ROW = 24
STRIP_ROWS = BITS // BOXES_PER_ROW
MASK = (1 << BITS) - 1
WHITE_Y = 255
BLACK_Y = 0
NEUTRAL_UV = 128
# A box's mean must be this far from the threshold to count as certain.
UNCERTAIN_LOW = 64
UNCERTAIN_HIGH = 192
THRESHOLD = 128


def bits_of(value_ms: int) -> list[int]:
    """The 48 lowest bits of ``value_ms``, most significant first."""
    value = value_ms & MASK
    return [(value >> (BITS - 1 - i)) & 1 for i in range(BITS)]


def value_of(bits: list[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | (bit & 1)
    return value


def strip_height(width: int) -> int:
    if width < BOXES_PER_ROW * BOX_PX:
        raise ValueError(f"tidskodsremsan får inte plats i bilden ({width} px)")
    return STRIP_ROWS * BOX_PX


def stamp_i420(frame: bytearray, width: int, height: int, value_ms: int) -> None:
    """Write the strip for ``value_ms`` into one I420 frame, in place.

    Luma: the boxes white or black, the rest of the strip rows black. Chroma: neutral
    grey under the whole strip, so a box decodes to pure white or black.
    """
    rows = strip_height(width)
    if len(frame) < width * height * 3 // 2:
        raise ValueError("ramen är mindre än en I420-bild")
    bits = bits_of(value_ms)
    black_row = bytes([BLACK_Y]) * width
    for y in range(rows):
        start = y * width
        frame[start : start + width] = black_row
    white_box = bytes([WHITE_Y]) * BOX_PX
    for index, bit in enumerate(bits):
        if not bit:
            continue
        row, col = divmod(index, BOXES_PER_ROW)
        x0 = col * BOX_PX
        for y in range(row * BOX_PX, (row + 1) * BOX_PX):
            start = y * width + x0
            frame[start : start + BOX_PX] = white_box
    chroma_w, chroma_rows = width // 2, rows // 2
    u_plane = width * height
    v_plane = u_plane + chroma_w * (height // 2)
    grey = bytes([NEUTRAL_UV]) * chroma_w
    for y in range(chroma_rows):
        for plane in (u_plane, v_plane):
            start = plane + y * chroma_w
            frame[start : start + chroma_w] = grey


def decode_luma(luma: bytes | bytearray, width: int) -> int | None:
    """Read the strip back from a luma plane (one byte per pixel, row by row).

    Each box is the mean of its middle 8 x 8 px, threshold 128. At most one box may be
    uncertain (mean between 64 and 192); a sample with more is invalid (``None``).
    """
    strip_height(width)
    bits: list[int] = []
    uncertain = 0
    for index in range(BITS):
        row, col = divmod(index, BOXES_PER_ROW)
        x0, y0 = col * BOX_PX + 4, row * BOX_PX + 4
        total = 0
        for y in range(y0, y0 + 8):
            start = y * width + x0
            total += sum(luma[start : start + 8])
        mean = total / 64
        if UNCERTAIN_LOW < mean < UNCERTAIN_HIGH:
            uncertain += 1
        bits.append(1 if mean >= THRESHOLD else 0)
    if uncertain > 1:
        return None
    return value_of(bits)
