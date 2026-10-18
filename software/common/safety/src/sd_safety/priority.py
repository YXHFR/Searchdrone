"""The order of the gate's reason codes and the operator text for a decision."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Final

from sd_schemas.reason_codes import GATE_REASON_CODES

__all__ = ["REASON_PRIORITY", "message_sv", "sort_codes"]

REASON_PRIORITY: Final = tuple(GATE_REASON_CODES)  # highest first
_RANK: Final = {code: i for i, code in enumerate(REASON_PRIORITY)}


def sort_codes(codes: Iterable[str]) -> list[str]:
    """Unique codes in priority order. An unknown code is a programming error."""
    return sorted(set(codes), key=lambda code: _RANK[code])


def _sv(number: float, decimals: int = 0) -> str:
    """Swedish decimal comma: 1,2."""
    return f"{number:.{decimals}f}".replace(".", ",")


def _detailed(code: str, details: Mapping[str, Any], skill: str | None) -> str | None:
    """A text with the numbers behind the code, when the details have them."""
    try:
        if code == "ROUTE_BATTERY":
            return (
                "Batteriet räcker inte för hela vägen och flygningen hem: "
                f"{_sv(details['route_m'] / 1000, 1)} km kräver ungefär "
                f"{_sv(details['needed_pct'])} % inklusive reserven på "
                f"{_sv(details['rtl_pct'])} %, men batteriet har {_sv(details['battery_pct'])} %. "
                "Rita en kortare väg eller byt batteri."
            )
        if code == "PATH_TOO_LONG":
            return (
                f"Vägen är för lång: {_sv(details['route_m'] / 1000, 1)} km, "
                f"högst {_sv(details['path_max_m'] / 1000, 1)} km."
            )
        if code == "OUTSIDE_CHARTER" and skill == "follow_path":
            by = details.get("outside_by_m")
            where = f" ({_sv(by)} m utanför)" if isinstance(by, int | float) else ""
            return f"Vägen går utanför sökområdet{where}. Rita om den inom uppdragsramen."
    except (KeyError, TypeError, ValueError):
        return None
    return None


def message_sv(
    codes: list[str], details: Mapping[str, Any] | None = None, skill: str | None = None
) -> str:
    """The text for the first code, plus how many more there are."""
    if not codes:
        return "Ordern är tillåten."
    text = _detailed(codes[0], details or {}, skill) or GATE_REASON_CODES[codes[0]]
    if len(codes) > 1:
        text = f"{text} (och {len(codes) - 1} till)"
    return text
