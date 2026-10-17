"""The machine-readable time strip in the top row of the synthetic picture.

48 squares of 16 x 16 px from the top-left corner carry the lowest 48 bits of
``CLOCK_REALTIME`` in milliseconds, most significant bit first: white = 1, black = 0.
The panel's end-to-end test reads them back from the browser (``frameClock.ts``) and
compares with ``Date.now()``; :func:`decode_luma` is the same rule in Python for tests.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = [
    "BITS",
    "BLACK",
    "SQUARE",
    "STRIP_WIDTH",
    "WHITE",
    "decode_luma",
    "strip_row",
]

BITS = 48
SQUARE = 16
STRIP_WIDTH = BITS * SQUARE  # 768 px: the picture must be at least this wide
WHITE = 235  # studio-range luma, so the encoder keeps clean black and white
BLACK = 16
_MASK = (1 << BITS) - 1


def bits_of(ms: int) -> list[int]:
    value = ms & _MASK
    return [(value >> (BITS - 1 - i)) & 1 for i in range(BITS)]


def strip_row(ms: int) -> bytes:
    """One luma row (``STRIP_WIDTH`` bytes) of the strip for ``ms``; repeat it 16 times."""
    row = bytearray()
    for bit in bits_of(ms):
        row += bytes([WHITE if bit else BLACK]) * SQUARE
    return bytes(row)


def decode_luma(pixel: Callable[[int, int], int]) -> tuple[int, int]:
    """Read the strip with ``pixel(x, y) -> 0..255``: (milliseconds, unsure squares).

    Each square is the mean of its middle 8 x 8 pixels against the threshold 128. A square
    whose mean is within 32 of the threshold counts as unsure.
    """
    value = 0
    unsure = 0
    for i in range(BITS):
        x0 = i * SQUARE + 4
        total = sum(pixel(x, y) for y in range(4, 12) for x in range(x0, x0 + 8))
        mean = total / 64
        if abs(mean - 128) < 32:
            unsure += 1
        value = (value << 1) | (1 if mean >= 128 else 0)
    return value, unsure
