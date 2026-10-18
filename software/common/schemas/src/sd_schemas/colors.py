"""Colour names for the search profile and the colour score, with their CIELAB references.

Twelve base colours plus ``unknown``. Black, white and grey are told apart by lightness
and chroma, not hue; the others by a CIEDE2000 distance to their reference. The values
are starting values, calibrated later against the matching bank (``COLOR_TABLE_VERSION``).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, NamedTuple

__all__ = [
    "COLOR_TABLE",
    "COLOR_TABLE_VERSION",
    "SHADE_L_SHIFT",
    "ColorName",
    "ColorRef",
]

COLOR_TABLE_VERSION: Final = "colors_v1"
SHADE_L_SHIFT: Final = 15.0  # "light" / "dark" move the reference L* by this much


class ColorName(StrEnum):
    BLACK = "black"
    WHITE = "white"
    GRAY = "gray"
    RED = "red"
    ORANGE = "orange"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    PURPLE = "purple"
    PINK = "pink"
    BROWN = "brown"
    BEIGE = "beige"
    UNKNOWN = "unknown"


class ColorRef(NamedTuple):
    """One colour: the Swedish label, the English word for prompts and its reference.

    ``delta_e`` is the CIEDE2000 tolerance for chromatic colours. The achromatic ones
    (black, white, grey) use the lightness and chroma bounds instead and leave it None.
    """

    sv: str
    en: str
    lab: tuple[float, float, float]  # L*, a*, b* (D65)
    delta_e: float | None = None
    l_min: float | None = None  # inclusive
    l_max: float | None = None  # inclusive
    c_max: float | None = None  # exclusive


COLOR_TABLE: Final[dict[ColorName, ColorRef]] = {
    ColorName.BLACK: ColorRef("svart", "black", (15, 0, 0), l_max=25, c_max=15),
    ColorName.WHITE: ColorRef("vit", "white", (92, 0, 0), l_min=80, c_max=12),
    ColorName.GRAY: ColorRef("grå", "gray", (55, 0, 0), l_min=25, l_max=80, c_max=12),
    ColorName.RED: ColorRef("röd", "red", (45, 65, 45), delta_e=22),
    ColorName.ORANGE: ColorRef("orange", "orange", (65, 45, 70), delta_e=20),
    ColorName.YELLOW: ColorRef("gul", "yellow", (88, 0, 80), delta_e=20),
    ColorName.GREEN: ColorRef("grön", "green", (50, -45, 35), delta_e=25),
    ColorName.BLUE: ColorRef("blå", "blue", (40, 10, -55), delta_e=25),
    ColorName.PURPLE: ColorRef("lila", "purple", (40, 45, -40), delta_e=22),
    ColorName.PINK: ColorRef("rosa", "pink", (70, 40, 0), delta_e=20),
    ColorName.BROWN: ColorRef("brun", "brown", (38, 15, 25), delta_e=18),
    ColorName.BEIGE: ColorRef("beige", "beige", (78, 3, 18), delta_e=15),
}
"""Every colour except ``unknown``, which has no reference and never scores."""
