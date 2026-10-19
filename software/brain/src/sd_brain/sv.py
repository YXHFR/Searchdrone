"""Normalising the operator's Swedish for the prompt ("Tolkat: …").

The result is put *next to* the original line, never instead of it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from sd_schemas.sv_commands import parse_number_sv

__all__ = ["Said", "height_m", "interpret", "read"]

_NUMBER_WORD = re.compile(r"\b([a-zåäö]+)\b")
_ALT = re.compile(r"(?:på|till|i)\s+(\S+)\s*(?:meters?|metres?|m)\b(?!/)")
_KMH = re.compile(r"(\d+(?:[.,]\d+)?|[a-zåäö]+)\s*km/h")
_MS = re.compile(r"(\d+(?:[.,]\d+)?)\s*m/s")
_COORD = re.compile(r"(-?\d{1,2})[.,](\d+),?\s+(-?\d{1,3})[.,](\d+)")
_PIN = re.compile(r"\b(?:p|punkt|nål|nal)\s?(\d{1,3})\b")
_SECTOR = re.compile(r"\b(?:sektor|sök)\s+([a-z]{1,2})\b")
_NOT_SECTORS = ("av", "en", "ut")  # "sök av", "sök en", "sök ut" name no sector
_BOX = re.compile(r"(\S+)\s*(?:x|gånger|ganger|\u00d7)\s*(\S+)\s*(?:meter|m)\b")


def _num(token: str) -> float | None:
    return parse_number_sv(token.strip(".,"))


def _fmt(value: float) -> str:
    return f"{value:g}".replace(".", ",")


def height_m(text: str) -> float | None:
    """The height the operator named ("på 40 meter", "till sextio m"), or ``None``."""
    m = _ALT.search(unicodedata.normalize("NFKC", text).lower())
    return _num(m.group(1)) if m else None


@dataclass(frozen=True)
class Said:
    """What :func:`read` recognised in a line (``None`` / empty: not named)."""

    alt_m: float | None = None
    speed_ms: float | None = None
    coord: tuple[float, float] | None = None
    pins: tuple[str, ...] = ()
    sectors: tuple[str, ...] = ()
    box_m: tuple[float, float] | None = None  # "100 gånger 100 meter": width, height


def read(text: str) -> Said:
    """The heights, speed, coordinates, pins, sectors and box size a line names."""
    t = unicodedata.normalize("NFKC", text).lower()
    speed: float | None = None
    m = _KMH.search(t)
    if m and (kmh := _num(m.group(1))) is not None:
        speed = round(kmh / 3.6, 1)
    else:
        m = _MS.search(t)
        if m:
            speed = _num(m.group(1))
    coord: tuple[float, float] | None = None
    m = _COORD.search(t)
    if m:
        coord = (float(f"{m.group(1)}.{m.group(2)}"), float(f"{m.group(3)}.{m.group(4)}"))
    box: tuple[float, float] | None = None
    m = _BOX.search(t)
    if m and (a := _num(m.group(1))) is not None and (b := _num(m.group(2))) is not None:
        box = (a, b)
    return Said(
        alt_m=height_m(t),
        speed_ms=speed,
        coord=coord,
        pins=tuple(sorted({f"P{n}" for n in _PIN.findall(t)})),
        sectors=tuple(sorted({s.upper() for s in _SECTOR.findall(t) if s not in _NOT_SECTORS})),
        box_m=box,
    )


def interpret(text: str) -> str | None:
    """``"höjd=40 m; punkt=P1; ruta=100x100 m"``, or ``None`` when nothing is recognised."""
    said = read(text)
    parts: list[str] = []
    if said.alt_m is not None:
        parts.append(f"höjd={_fmt(said.alt_m)} m")
    if said.speed_ms is not None:
        parts.append(f"fart={_fmt(said.speed_ms)} m/s")
    if said.coord is not None:
        parts.append(f"koordinat=({said.coord[0]}, {said.coord[1]})")
    if said.pins:
        parts.append("punkt=" + ",".join(said.pins))
    if said.sectors:
        parts.append("sektor=" + ",".join(said.sectors))
    if said.box_m is not None:
        parts.append(f"ruta={_fmt(said.box_m[0])}x{_fmt(said.box_m[1])} m")
    return "; ".join(parts) if parts else None
