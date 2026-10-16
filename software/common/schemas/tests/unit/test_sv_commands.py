"""The deterministic Swedish chat interpreter (AC-B08)."""

from __future__ import annotations

from typing import Any

import pytest

from sd_schemas.sv_commands import (
    classify_override,
    is_resume_request,
    is_takeoff_request,
    parse_command,
    parse_number_sv,
)

RESUME_PHRASES = [
    "fortsätt",
    "Fortsätt!",
    "kör vidare",
    "fortsätt sökningen",
    "släpp stopp",
    "fortsätt flyga",
    "fortsätt nu",
    "fortsatt",
]


@pytest.mark.parametrize("text", [*RESUME_PHRASES, "resume", "Fortsätt."])
def test_classify_override_never_resume(text: str) -> None:
    assert classify_override(text) is None


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("stopp", "stop"),
        ("Stopp!", "stop"),
        ("STOP!", "stop"),
        ("stop", "stop"),
        ("stop now", "stop"),
        ("stopp nu", "stop"),
        ("flyg hem", "return_home"),
        ("Flyg hem!", "return_home"),
        ("Stopp, stopp!", "stop"),
        ("stopp stopp stop", "stop"),
        ("flyg hem, flyg hem", "return_home"),
        ("stanna inte", None),
        ("don't stop", None),
        ("inte hem", None),
        ("stopp inte", None),
        ("håll", None),
        ("vänta", None),
        ("paus", None),
        ("fortsätt", None),
        ("stanna här en stund", None),
        ("flyg hem sen", None),
    ],
)
def test_classify_override(text: str, want: str | None) -> None:
    assert classify_override(text) == want


def test_resume_request() -> None:
    for text in RESUME_PHRASES:
        assert is_resume_request(text), text
    assert not is_resume_request("fortsätt inte")
    assert not is_resume_request("fortsätt aldrig")
    assert not is_resume_request("vad ska vi göra")


@pytest.mark.parametrize(
    "text",
    [
        "starta",
        "Starta och flyg till P1",
        "lyft",
        "Lyft till 40 meter",
        "kör igång",
        "flyg till P2",
        "flyg till punkt 3",
        "sök sektor A",
        "sök området",
        "sök av rutan",
        "Sök rutan nu",
        "Starta upp drönaren",
    ],
)
def test_takeoff_request_yes(text: str) -> None:
    assert is_takeoff_request(text)


@pytest.mark.parametrize(
    "text",
    [
        "Hur länge kan vi flyga?",
        "Kan du flyga i regn?",
        "Vad ska vi söka efter?",
        "När ska vi starta?",
        "Ska vi lyfta nu?",
        "Får vi starta?",
        "Varför startar du inte",
        "starta inte",
        "lyft inte än",
        "Vänta med att starta",
        "Vilken sektor ska vi söka",
        "hej",
        "",
        "sök efter en man i röd jacka",
        "lyft fram kartan",
        "vi ska starta om en timme",
        "sök området senare",
        "starta imorgon",
        "flyg till P1 om en stund",
        "Det är dags att lyfta blicken",
    ],
)
def test_takeoff_request_no(text: str) -> None:
    assert not is_takeoff_request(text)


PINS = ["P1", "P2", "P12"]
SECTORS = ["A", "B", "AB"]
COMMANDS: list[tuple[str, Any]] = [
    ("starta", ("takeoff", {})),
    ("Starta", ("takeoff", {})),
    ("starta på 40 meter", ("takeoff", {"alt_m": 40.0})),
    ("starta på fyrtio meter", ("takeoff", {"alt_m": 40.0})),
    ("lyft", ("takeoff", {})),
    ("lyft till 30 m", ("takeoff", {"alt_m": 30.0})),
    ("lyft till 30m", ("takeoff", {"alt_m": 30.0})),
    ("lyft till trettio m", ("takeoff", {"alt_m": 30.0})),
    ("lyft upp", ("takeoff", {})),
    ("starta på hundratjugo meter", ("takeoff", {"alt_m": 120.0})),
    ("flyg till P1", ("goto", {"pin": "P1"})),
    ("Flyg till p1", ("goto", {"pin": "P1"})),
    ("flyg till P 1", ("goto", {"pin": "P1"})),
    ("flyg till punkt 2", ("goto", {"pin": "P2"})),
    ("flyg till nål 2", ("goto", {"pin": "P2"})),
    ("flyg till nal 2", ("goto", {"pin": "P2"})),
    ("flyg till P12", ("goto", {"pin": "P12"})),
    ("flyg till P1 på 50 meter", ("goto", {"pin": "P1", "alt_m": 50.0})),
    ("flyg till P1 på femtio meter", ("goto", {"pin": "P1", "alt_m": 50.0})),
    ("flyg till P1 på sextio m", ("goto", {"pin": "P1", "alt_m": 60.0})),
    ("ak till P1", ("goto", {"pin": "P1"})),
    ("åk till P2", ("goto", {"pin": "P2"})),
    ("gå till P1", ("goto", {"pin": "P1"})),
    ("flyg till P1 nu", ("goto", {"pin": "P1"})),
    ("flyg till P1 direkt", ("goto", {"pin": "P1"})),
    ("flyg till -35,3610 149,1680", ("goto", {"lat": -35.361, "lon": 149.168})),
    ("flyg till -35.3610, 149.1680", ("goto", {"lat": -35.361, "lon": 149.168})),
    (
        "flyg till -35.3610 149.1680 på 45 m",
        ("goto", {"lat": -35.361, "lon": 149.168, "alt_m": 45.0}),
    ),
    ("sök sektor A", ("survey_sector", {"sector": "A"})),
    ("sok sektor a", ("survey_sector", {"sector": "A"})),
    ("sök sektor B på 40 meter", ("survey_sector", {"sector": "B", "alt_m": 40.0})),
    ("sök A", ("survey_sector", {"sector": "A"})),
    ("sök b", ("survey_sector", {"sector": "B"})),
    ("sök av sektor A", ("survey_sector", {"sector": "A"})),
    ("sök sektor AB", ("survey_sector", {"sector": "AB"})),
    ("sök sektor A på trettiofem meter", ("survey_sector", {"sector": "A", "alt_m": 35.0})),
    ("håll position", ("hold", {})),
    ("Håll position!", ("hold", {})),
    ("hall position", ("hold", {})),
    ("håll positionen", ("hold", {})),
    ("sväva", ("hold", {})),
    ("svava", ("hold", {})),
    ("fortsätt", ("resume", {})),
    ("fortsatt", ("resume", {})),
    ("status", ("status", None)),
    ("läge", ("status", None)),
    ("lage", ("status", None)),
    ("hjälp", ("help", None)),
    ("hjalp", ("help", None)),
    ("håll", None),
    ("flyg till P9", None),
    ("sök sektor C", None),
    ("flyg till -95,1 149,1", None),
    ("starta på tjohej meter", None),
    ("flyg hem", None),
    ("stopp", None),
    ("landa här", None),
    ("kamera 45 grader", None),
    ("vad gör du", None),
    ("", None),
]


@pytest.mark.parametrize(("text", "want"), COMMANDS, ids=[c[0] or "empty" for c in COMMANDS])
def test_parse_command(text: str, want: Any) -> None:
    got = parse_command(text, PINS, SECTORS)
    if want is None:
        assert got is None
    elif want[1] is None:
        assert got is not None and got.kind == want[0]
    else:
        assert got is not None and got.kind == "skill"
        assert (got.skill, got.params) == want


def test_sixty_command_phrases() -> None:
    assert len(COMMANDS) >= 60


@pytest.mark.parametrize(
    ("word", "value"),
    [
        ("0", 0.0),
        ("noll", 0.0),
        ("en", 1.0),
        ("tolv", 12.0),
        ("tjugoen", 21.0),
        ("fyrtiofem", 45.0),
        ("hundra", 100.0),
        ("etthundra", 100.0),
        ("hundrafemtio", 150.0),
        ("tvåhundra", 200.0),
        ("tvahundra", 200.0),
        ("2,5", 2.5),
        ("tvåhundraett", None),
        ("tjugofemtio", None),
        ("hej", None),
    ],
)
def test_numbers(word: str, value: float | None) -> None:
    assert parse_number_sv(word) == value
