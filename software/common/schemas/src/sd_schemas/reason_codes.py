"""Every reason code with its Swedish text for the operator.

``GATE_REASON_CODES`` is in priority order (highest first); ``sd_safety.priority``
builds ``REASON_PRIORITY`` from it, and a test there checks the order.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from sd_schemas.charter import SECTOR_MAX_AREA_M2

__all__ = [
    "APPROVAL_OUTCOME_CODES",
    "CHARTER_VALIDATION_CODES",
    "GATE_REASON_CODES",
    "LIMITS_CODES",
    "ONBOARD_REASON_CODES",
    "approval_codes",
    "is_known",
    "text_sv",
]


def _sv_int(number: float) -> str:
    """Swedish thousands separator: 250 000."""
    return f"{number:,.0f}".replace(",", "\u00a0")


GATE_REASON_CODES: Final = MappingProxyType(
    {
        "SKILL_UNKNOWN": "Okänd färdighet.",
        "SCHEMA_INVALID": "Ordern har fel parametrar.",
        "SKILL_NOT_IMPLEMENTED": "Färdigheten finns inte än.",
        "FORBIDDEN_ACTION": "AI:n får inte använda den här färdigheten.",
        "AUTONOMY_CHANGE_FORBIDDEN": "Bara operatören kan ändra autonomiläget.",
        "SKILL_NOT_ALLOWED": "Färdigheten är inte tillåten i uppdragsramen.",
        "AI_DISABLED": "AI:n är avstängd för uppdraget.",
        "MISSION_STATE_INVALID": "Det går inte i uppdragets nuvarande läge.",
        "RELEASE_REQUIRED": "Tryck Fortsätt först.",
        "RESUME_NEEDS_OPERATOR": "Drönaren är stoppad. Bara operatören kan trycka Fortsätt.",
        "OPERATOR_OVERRIDE_ACTIVE": "Operatören har tagit över. AI:n får inte flytta drönaren.",
        "FAILSAFE_ACTIVE": "Flygkontrollerns nödläge eller batterihemflygning pågår.",
        "CONTROL_HELD": "Någon annan styr drönaren just nu.",
        "MANUAL_CONTROL_ACTIVE": "Piloten styr för hand. AI:n får vänta.",
        "CHARTER_NOT_APPROVED": "Det finns ingen godkänd uppdragsram.",
        "CHARTER_NOT_INSTALLED": "Uppdragsramen är inte installerad i drönaren än.",
        "CHARTER_VERSION_MISMATCH": "Ordern gäller en annan version av uppdragsramen.",
        "FENCE_NOT_VERIFIED": "Staketet i flygkontrollern är inte kontrollerat.",
        "LINK_DOWN": "Länken till drönaren är nere.",
        "STATE_STALE": "Lägesbilden är för gammal.",
        "STATE_CHANGED": "Läget har ändrats sedan ordern skapades.",
        "INVALID_TRANSITION": "Färdigheten går inte i drönarens nuvarande läge.",
        "CLEARANCE_REQUIRED": "Någon måste bekräfta att det är fritt runt drönaren.",
        "CLEARANCE_EXPIRED": "Bekräftelsen att det är fritt runt drönaren har gått ut.",
        "RC_LINK_DOWN": (
            "RC-sändaren är inte påslagen eller har ingen kontakt med drönaren. "
            "Slå på den innan start."
        ),
        "TAKEOFF_PRECONDITION": "Drönaren är inte redo att lyfta.",
        "BATTERY_RESERVE": "Batteriet är nere på reserven.",
        "BATTERY_UNKNOWN": "Batterinivån är okänd.",
        "ALT_ABOVE_LIMIT": "Höjden är över uppdragets gräns.",
        "ALT_ABOVE_CHARTER": "Höjden är över uppdragsramens tak.",
        "ALT_BELOW_CHARTER": "Höjden är under uppdragsramens golv.",
        "OUTSIDE_CHARTER": "Målet ligger utanför sökområdet.",
        "AREA_BEYOND_FC_FENCE": "Området ligger för långt från hemplatsen för staketet.",
        "PILOT_DISTANCE": "Målet ligger för långt från piloten.",
        "SPEED_ABOVE_LIMIT": "Farten är över uppdragets gräns.",
        "SPEED_ABOVE_CHARTER": "Farten är över uppdragsramens gräns.",
        "OPERATOR_ABSENT": "Ingen operatör är ansluten.",
        "PENDING_LIMIT": "En order väntar redan på godkännande.",
        "RATE_LIMIT": "För många order på kort tid.",
        "NEEDS_CLEARANCE": "Bekräfta att det är fritt runt drönaren.",
        "NEEDS_APPROVAL_SKILL": "Färdigheten kräver operatörens godkännande.",
        "NEEDS_APPROVAL_MODE_A": "I läge A godkänner operatören varje steg.",
    }
)

APPROVAL_OUTCOME_CODES: Final = MappingProxyType(
    {
        "APPROVAL_DENIED": "Operatören avvisade ordern.",
        "APPROVAL_TIMEOUT": "Ingen svarade på godkännandet i tid.",
        "APPROVAL_SUPERSEDED": "Godkännandet ersattes av ett nyare läge.",
    }
)

CHARTER_VALIDATION_CODES: Final = MappingProxyType(
    {
        "AREA_INVALID": "Sökområdet är ogiltigt eller för stort.",
        "HOME_OUTSIDE_AREA": "Hemplatsen ligger utanför sökområdet.",
        "SECTOR_OUTSIDE_AREA": "En sektor ligger utanför sökområdet.",
        "SECTOR_TOO_LARGE": f"En sektor är större än {_sv_int(SECTOR_MAX_AREA_M2)} m².",
        "AREA_BEYOND_RANGE": "Sökområdet ligger för långt från piloten.",
        "AREA_BEYOND_FC_FENCE": GATE_REASON_CODES["AREA_BEYOND_FC_FENCE"],
        "BUFFER_INVALID": "Staketets marginal är ogiltig.",
        "ALT_BAND_INVALID": "Höjdbandet är ogiltigt.",
        "CHARTER_EXCEEDS_LIMITS": "Uppdragsramen är lösare än uppdragets gränser.",
        "SKILL_UNKNOWN": GATE_REASON_CODES["SKILL_UNKNOWN"],
        "SKILL_NOT_IMPLEMENTED": GATE_REASON_CODES["SKILL_NOT_IMPLEMENTED"],
        "AREA_CHANGE_REQUIRES_LANDING": "Sökområdet kan bara krympa medan drönaren flyger.",
        "BASE_VERSION_STALE": "Förslaget bygger på en gammal version av ramen.",
        "AUTONOMY_CHANGE_FORBIDDEN": GATE_REASON_CODES["AUTONOMY_CHANGE_FORBIDDEN"],
        "LIMITS_CHANGE_FORBIDDEN": "Uppdragets gränser ändras bara av piloten.",
        "VALIDATION": "Fel i indata.",
    }
)

LIMITS_CODES: Final = MappingProxyType(
    {
        "LIMIT_ABOVE_ABSOLUTE": "Gränsen är över det yttre taket.",
        "LIMITS_BELOW_ACTIVE_CHARTER": "Den aktiva uppdragsramen ryms inte inom de nya gränserna.",
        "LIMITS_CHANGE_REQUIRES_LANDING": "Gränser över standardvärdena kan bara höjas på marken.",
    }
)

ONBOARD_REASON_CODES: Final = MappingProxyType(
    {
        "ONB_PREFLIGHT_NOT_OK": "Förflygningskontrollerna är inte godkända.",
        "ONB_RC_LINK_DOWN": (
            "RC-sändaren är inte påslagen eller har ingen kontakt med drönaren. "
            "Slå på den innan start."
        ),
        "ONB_CLEARANCE_MISSING": "Ingen bekräftelse att det är fritt runt drönaren.",
        "ONB_CLEARANCE_STALE": "Bekräftelsen att det är fritt runt drönaren har gått ut.",
        "ONB_CLEARANCE_MISMATCH": "Bekräftelsen gäller ett annat uppdrag eller en annan ram.",
        "ONB_CLEARANCE_REMOTE": "Bekräftelsen kom inte från platsen.",
        "ONB_NOT_ON_GROUND": "Drönaren står inte desarmerad på marken.",
        "ONB_ARM_NOT_ALLOWED_IN_PROFILE": "Ombordsdatorn får inte armera i den här profilen.",
        "ONB_ARM_REFUSED": "Flygkontrollern vägrade armera.",
        "ONB_TAKEOFF_TIMEOUT": "Starten tog för lång tid.",
        "ONB_MODE_CHANGE_TIMEOUT": "Flygkontrollern bytte inte läge i tid.",
        "ONB_MISSION_UPLOAD_FAILED": "Uppladdningen av sökrutten misslyckades.",
        "ONB_MISSION_READBACK_MISMATCH": "Sökrutten i flygkontrollern stämmer inte.",
        "ONB_SURVEY_TOO_LARGE": "Sökningen har för många punkter.",
        "ONB_UNKNOWN_SECTOR": "Okänd sektor.",
        "ONB_NOTHING_TO_RESUME": "Det finns inget att fortsätta med.",
        "ONB_SKILL_TIMEOUT": "Färdigheten tog för lång tid.",
        "ONB_WRONG_EXECUTOR": "Färdigheten körs på marken, inte i drönaren.",
        "ONB_CONTROL_HELD": "Någon annan styr drönaren just nu.",
        "ONB_RTL_ACTIVE": "Drönaren är på väg hem.",
        "ONB_FAILSAFE_ACTIVE": "Flygkontrollerns nödläge är aktivt.",
        "ONB_INVALID_TRANSITION": "Färdigheten går inte i drönarens nuvarande läge.",
        "ONB_STALE_ORDER": "Ordern bygger på en gammal lägesbild.",
        "ONB_APPROVAL_MISSING": "Ordern saknar operatörens godkännande.",
        "ONB_APPROVAL_MISMATCH": "Godkännandet gäller en annan order.",
        "ONB_BATTERY_RTL_ACTIVE": "Batterihemflygning pågår.",
        "ONB_RANGE_MUST_DECREASE": "Drönaren måste närma sig piloten.",
        "ONB_NAV_DEGRADED": "Navigeringen är osäker.",
        "ONB_STOP_LATCHED": "Drönaren är stoppad. Operatören måste trycka Fortsätt.",
        "ONB_ORDER_EXPIRED": "Ordern hann gå ut.",
        "ONB_RELEASE_MODE_UNSUPPORTED": "Släppet går inte i flygkontrollerns nuvarande läge.",
        "ONB_CHARTER_IN_FLIGHT": "Uppdragsramen kan inte bytas så här under flygning.",
        "ONB_CHARTER_EXCEEDS_HARD_LIMITS": "Uppdragsramen går utanför de hårda gränserna.",
        "ONB_GATE_DISAGREEMENT": "Ombordsdatorns kontroll sa nej trots att marken sa ja.",
    }
)

_ALL = (
    GATE_REASON_CODES,
    APPROVAL_OUTCOME_CODES,
    CHARTER_VALIDATION_CODES,
    LIMITS_CODES,
    ONBOARD_REASON_CODES,
)


def is_known(code: str) -> bool:
    return any(code in table for table in _ALL)


def text_sv(code: str) -> str:
    """The Swedish text for a code; KeyError for an unknown code."""
    for table in _ALL:
        if code in table:
            return table[code]
    raise KeyError(code)


def approval_codes() -> frozenset[str]:
    """Gate codes that ask for approval instead of rejecting."""
    return frozenset(c for c in GATE_REASON_CODES if c.startswith("NEEDS_"))
