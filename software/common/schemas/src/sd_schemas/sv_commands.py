"""Deterministic Swedish chat interpretation, without any language model.

* :func:`classify_override` - a chat line that *is* STOP or FLY HOME ("stopp", "stop",
  "flyg hem"). It may only ever return ``"stop"`` or ``"return_home"``, never a release
  of a stop: only the Resume button releases one. The AI commander's quick stop check
  uses it as the floor when no model answers in time.
* :func:`is_resume_request` - the operator asks to continue (answered with "press
  Resume"; it never releases anything).
* :func:`is_takeoff_request` - the operator asks the drone to start.
* :func:`parse_command` - the fixed command mode used while the language model is down.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from pydantic import Field

from sd_schemas.base import SdModel

__all__ = [
    "OverrideIntent",
    "ParsedCommand",
    "classify_override",
    "is_resume_request",
    "is_takeoff_request",
    "normalize",
    "parse_command",
    "parse_number_sv",
]

OverrideIntent = Literal["stop", "return_home"]

_PUNCT = re.compile(r"[.!?,;:¡¿\"'()]+")
_SPACE = re.compile(r"\s+")
_OVERRIDE_FILLERS = frozenset({"nu", "direkt", "genast", "now"})
_FILLERS = frozenset({"nu", "direkt", "genast", "snalla", "tack", "du", "now", "please"})
_NEGATIONS = frozenset({"inte", "ej", "aldrig", "icke", "not", "dont", "don't", "never"})
_QUESTION_WORDS = frozenset(
    {"hur", "vad", "kan", "ska", "nar", "varfor", "vilken", "vilket", "vilka", "far", "bor"}
)

_STOP_WORDS = frozenset({"stopp", "stop"})
_HOME_PHRASES = frozenset({"flyg hem"})
_RESUME_PHRASES = frozenset(
    {"fortsatt", "kor vidare", "fortsatt sokningen", "slapp stopp", "fortsatt flyga"}
)


def normalize(text: str) -> str:
    """NFKC, lower case, punctuation removed, single spaces."""
    t = unicodedata.normalize("NFKC", text).lower()
    t = t.replace("\u2019", "'")
    t = _PUNCT.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


def _fold(text: str) -> str:
    """Tolerate a/o instead of å/ä/ö (and é)."""
    return text.replace("å", "a").replace("ä", "a").replace("ö", "o").replace("é", "e")


def _words(text: str) -> list[str]:
    return _fold(normalize(text)).split()


def classify_override(text: str) -> OverrideIntent | None:
    """``"stop"`` or ``"return_home"`` when the whole line is the word, else ``None``.

    Exact match after normalisation and dropping the fillers ``nu``, ``direkt``,
    ``genast`` and ``now``; a repeated word ("stopp, stopp!") counts too. Everything
    else (``stanna inte``, ``fortsätt``, longer sentences) is ``None``. Never returns
    anything that releases a stop.
    """
    words = [w for w in normalize(text).split() if w not in _OVERRIDE_FILLERS]
    if words and all(w in _STOP_WORDS for w in words):  # "stopp", "stopp stopp!"
        return "stop"
    line = " ".join(words)
    if re.fullmatch(r"flyg hem(?: flyg hem)*", line):
        return "return_home"
    return None


def is_resume_request(text: str) -> bool:
    """The operator asks to continue (``fortsätt``, ``kör vidare`` …), without negation."""
    words = _words(text)
    if any(w in _NEGATIONS for w in words):
        return False
    line = " ".join(w for w in words if w not in _FILLERS)
    return line in _RESUME_PHRASES


_AFTER_START = r"(?:$| (?:upp|nu|och|pa|till|direkt|dronaren)\b)"
_TAKEOFF_RE = re.compile(
    r"\b(?:starta|lyft)" + _AFTER_START + r"|\bkor igang\b"
    r"|\b(?:flyg|ak) till (?:p ?\d{1,3}|punkt \d+|nal \d+)\b"
    r"|\bsok(?: av)? (?:sektor [a-z]{1,2}|omradet|rutan|ytan)\b"
)
# Talk about a later start is not a request now ("vi ska starta om en timme").
_LATER_RE = re.compile(
    r"\b(?:senare|imorgon|i morgon|ikvall|om en (?:stund|timme|minut)|vi ska|ska vi)\b"
)


def is_takeoff_request(text: str) -> bool:
    """An instruction to start now: ``starta``, ``lyft``, ``kör igång``, ``flyg till P1``,
    ``sök sektor A`` / ``sök området`` / ``sök rutan``.

    Not a question (ends with ``?`` or starts with a question word) and not negated
    (``inte``, ``ej``, ``aldrig``, ``vänta med``).
    """
    if text.strip().endswith("?"):
        return False
    words = _words(text)
    if not words or words[0] in _QUESTION_WORDS:
        return False
    if any(w in _NEGATIONS for w in words):
        return False
    line = " ".join(words)
    if "vanta med" in line or _LATER_RE.search(line):
        return False
    return _TAKEOFF_RE.search(line) is not None


# --- numbers ----------------------------------------------------------------------------

_UNITS = {
    "noll": 0, "en": 1, "ett": 1, "tva": 2, "tre": 3, "fyra": 4, "fem": 5, "sex": 6,
    "sju": 7, "atta": 8, "nio": 9,
}  # fmt: skip
_TEENS = {
    "tio": 10, "elva": 11, "tolv": 12, "tretton": 13, "fjorton": 14, "femton": 15,
    "sexton": 16, "sjutton": 17, "arton": 18, "nitton": 19,
}  # fmt: skip
_TENS = {
    "tjugo": 20, "trettio": 30, "fyrtio": 40, "femtio": 50, "sextio": 60, "sjuttio": 70,
    "attio": 80, "nittio": 90,
}  # fmt: skip


def _below_hundred(word: str) -> int | None:
    if word == "":
        return 0
    if word in _UNITS:
        return _UNITS[word]
    if word in _TEENS:
        return _TEENS[word]
    for tens_word, tens in _TENS.items():
        if word.startswith(tens_word):
            rest = word[len(tens_word) :]
            if rest == "":
                return tens
            if rest in _UNITS and _UNITS[rest] > 0 and rest != "en":
                return tens + _UNITS[rest]
            if rest == "en":  # "tjugoen"
                return tens + 1
            return None
    return None


def parse_number_sv(word: str) -> float | None:
    """A number written as digits (``40``, ``2,5``) or a Swedish number word 0-200."""
    w = _fold(word.strip().lower())
    if re.fullmatch(r"-?\d+([.,]\d+)?", w):
        return float(w.replace(",", "."))
    hundreds = 0
    for prefix, value in (("tvahundra", 200), ("etthundra", 100), ("hundra", 100)):
        if w.startswith(prefix):
            hundreds, w = value, w[len(prefix) :]
            break
    rest = _below_hundred(w)
    if rest is None or (hundreds == 0 and w == ""):
        return None
    total = hundreds + rest
    return float(total) if total <= 200 else None


# --- command mode -----------------------------------------------------------------------


class ParsedCommand(SdModel):
    kind: Literal["skill", "status", "help"]
    skill: str | None = None
    params: dict[str, Any] = Field(default_factory=dict[str, Any])


_NUM = r"(?P<n>[0-9a-z.,]+)"
_ALT = rf"(?: (?:pa|till|i) {_NUM} ?(?:meter|meters|metres|m))?"
_PIN = r"(?P<pin>p ?\d{1,3}|punkt \d{1,3}|nal \d{1,3})"
_COORD = r"(?P<lat>-?\d{1,2}[.,]\d+),? (?P<lon>-?\d{1,3}[.,]\d+)"

_COMMANDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"^(?:starta|lyft)(?: upp)?{_ALT}$"), "takeoff"),
    (re.compile(rf"^(?:flyg|ak|ga) till {_PIN}{_ALT}$"), "goto_pin"),
    (re.compile(rf"^(?:flyg|ak|ga) till {_COORD}{_ALT}$"), "goto_point"),
    (re.compile(rf"^sok(?: av)?(?: sektor)? (?P<sector>[a-z]{{1,2}}){_ALT}$"), "survey"),
    (re.compile(r"^(?:hall position|hall positionen|svava)$"), "hold"),
    (re.compile(r"^fortsatt$"), "resume"),
    (re.compile(r"^(?:status|lage|laget)$"), "status"),
    (re.compile(r"^(?:hjalp|help)$"), "help"),
)


def _clean_command(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[!?;:\"'()]+", " ", t)
    t = re.sub(r"\.(?!\d)", " ", t)  # keep decimal points
    t = re.sub(r"(\d),(?!\d)", r"\1 ", t)  # "-35.36, 149.16" keeps the number
    t = re.sub(r"(?<=\d)(m)\b", r" \1", t)  # "40m" -> "40 m"
    words = [w for w in _fold(t).split() if w not in _FILLERS]
    return " ".join(words)


def _pin_id(raw: str) -> str:
    return "P" + re.sub(r"\D", "", raw)


def _alt(match: re.Match[str]) -> dict[str, float] | None:
    n = match.groupdict().get("n")
    if n is None:
        return {}
    value = parse_number_sv(n)
    return None if value is None else {"alt_m": value}


def parse_command(text: str, pins: list[str], sectors: list[str]) -> ParsedCommand | None:
    """A fixed Swedish command, or ``None`` ("Förstod inte").

    ``pins`` and ``sectors`` are the known ids (``"P1"``, ``"A"``); an unknown one gives
    ``None``. ``håll`` alone is not a command; ``håll position`` is.
    """
    line = _clean_command(text)
    for pattern, kind in _COMMANDS:
        m = pattern.fullmatch(line)
        if m is None:
            continue
        if kind in ("status", "help"):
            return ParsedCommand(kind=kind)
        if kind in ("hold", "resume"):
            return ParsedCommand(kind="skill", skill=kind)
        alt = _alt(m)
        if alt is None:
            return None
        if kind == "takeoff":
            return ParsedCommand(kind="skill", skill="takeoff", params=alt)
        if kind == "goto_pin":
            pin = _pin_id(m.group("pin"))
            if pin not in pins:
                return None
            return ParsedCommand(kind="skill", skill="goto", params={"pin": pin, **alt})
        if kind == "goto_point":
            lat = float(m.group("lat").replace(",", "."))
            lon = float(m.group("lon").replace(",", "."))
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                return None
            return ParsedCommand(kind="skill", skill="goto", params={"lat": lat, "lon": lon, **alt})
        sector = m.group("sector").upper()
        if sector not in sectors:
            return None
        return ParsedCommand(kind="skill", skill="survey_sector", params={"sector": sector, **alt})
    return None
