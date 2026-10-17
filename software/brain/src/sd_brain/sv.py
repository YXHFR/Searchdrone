"""Normalising the operator's Swedish for the prompt ("Tolkat: …").

The result is put *next to* the original line, never instead of it.
"""

from __future__ import annotations

import re
import unicodedata

from sd_schemas.sv_commands import parse_number_sv

__all__ = ["height_m", "interpret"]

_NUMBER_WORD = re.compile(r"\b([a-zåäö]+)\b")
_ALT = re.compile(r"(?:på|till|i)\s+(\S+)\s*(?:meters?|metres?|m)\b(?!/)")
_KMH = re.compile(r"(\d+(?:[.,]\d+)?|[a-zåäö]+)\s*km/h")
_MS = re.compile(r"(\d+(?:[.,]\d+)?)\s*m/s")
_COORD = re.compile(r"(-?\d{1,2})[.,](\d+),?\s+(-?\d{1,3})[.,](\d+)")
_PIN = re.compile(r"\b(?:p|punkt|nål|nal)\s?(\d{1,3})\b")
_SECTOR = re.compile(r"\b(?:sektor|sök)\s+([a-z]{1,2})\b")
_BOX = re.compile(r"(\S+)\s*(?:x|gånger|ganger|\u00d7)\s*(\S+)\s*(?:meter|m)\b")


def _num(token: str) -> float | None:
    return parse_number_sv(token.strip(".,"))


def _fmt(value: float) -> str:
    return f"{value:g}".replace(".", ",")


def height_m(text: str) -> float | None:
    """The height the operator named ("på 40 meter", "till sextio m"), or ``None``."""
    m = _ALT.search(unicodedata.normalize("NFKC", text).lower())
    return _num(m.group(1)) if m else None


def interpret(text: str) -> str | None:
    """``"höjd=40 m; punkt=P1; ruta=100x100 m"``, or ``None`` when nothing is recognised."""
    t = unicodedata.normalize("NFKC", text).lower()
    parts: list[str] = []
    if (alt := height_m(t)) is not None:
        parts.append(f"höjd={_fmt(alt)} m")
    m = _KMH.search(t)
    if m and (kmh := _num(m.group(1))) is not None:
        parts.append(f"fart={_fmt(round(kmh / 3.6, 1))} m/s")
    else:
        m = _MS.search(t)
        if m and (ms := _num(m.group(1))) is not None:
            parts.append(f"fart={_fmt(ms)} m/s")
    m = _COORD.search(t)
    if m:
        lat = float(f"{m.group(1)}.{m.group(2)}")
        lon = float(f"{m.group(3)}.{m.group(4)}")
        parts.append(f"koordinat=({lat}, {lon})")
    pins = sorted({f"P{n}" for n in _PIN.findall(t)})
    if pins:
        parts.append("punkt=" + ",".join(pins))
    sectors = sorted({s.upper() for s in _SECTOR.findall(t) if s not in ("av", "en", "ut")})
    if sectors:
        parts.append("sektor=" + ",".join(sectors))
    m = _BOX.search(t)
    if m and (a := _num(m.group(1))) is not None and (b := _num(m.group(2))) is not None:
        parts.append(f"ruta={_fmt(a)}x{_fmt(b)} m")
    return "; ".join(parts) if parts else None
