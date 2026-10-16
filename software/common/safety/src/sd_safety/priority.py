"""The order of the gate's reason codes and the operator text for a decision."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from sd_schemas.reason_codes import GATE_REASON_CODES

__all__ = ["REASON_PRIORITY", "message_sv", "sort_codes"]

REASON_PRIORITY: Final = tuple(GATE_REASON_CODES)  # highest first
_RANK: Final = {code: i for i, code in enumerate(REASON_PRIORITY)}


def sort_codes(codes: Iterable[str]) -> list[str]:
    """Unique codes in priority order. An unknown code is a programming error."""
    return sorted(set(codes), key=lambda code: _RANK[code])


def message_sv(codes: list[str]) -> str:
    """The text for the first code, plus how many more there are."""
    if not codes:
        return "Ordern är tillåten."
    text = GATE_REASON_CODES[codes[0]]
    if len(codes) > 1:
        text = f"{text} (och {len(codes) - 1} till)"
    return text
