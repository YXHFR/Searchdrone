"""The demo fixture: home, pilot, pins, the search box and the test charters.

``load()`` reads ``simulation/scenarios/demo_fixture.yaml`` (or another file with the same
shape). A charter with ``base: <name>`` inherits every field it does not set from that
charter (one level only). A charter with ``exceed_limits: true`` gets limits looser than
the hard caps when the caller passes the caps (``limits=``), so no limit number is written
in the fixture itself; without them it keeps its base's values (``limits_applied`` is
False). Unknown keys raise ``FixtureError``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sd_sim import paths

DEFAULT_PATH = paths.REPO / "simulation" / "scenarios" / "demo_fixture.yaml"
EXIT_CODE = 2

TOP_KEYS = {"home", "pilot", "operator", "vehicle_id", "pins", "survey_box", "charters"}
HOME_KEYS = {"lat", "lon", "alt_amsl_m", "heading_deg"}
POINT_KEYS = {"east_m", "north_m"}
PIN_KEYS = {"id", "east_m", "north_m", "alt_m"}
BOX_KEYS = {"center_pin", "size_m", "alt_m"}
CHARTER_KEYS = {
    "area_enu",
    "geofence_buffer_m",
    "alt_min_agl_m",
    "alt_max_agl_m",
    "default_alt_agl_m",
    "max_speed_ms",
    "max_range_from_pilot_m",
    "battery_rtl_pct",
    "allowed_skills",
    "sectors",
}
CHARTER_META_KEYS = {"base", "exceed_limits"}
SECTOR_KEYS = {"sector_id", "name", "from", "line_spacing_m"}
# How far past the caps an exceed_limits charter goes.
EXCEED_ALT_M = 30.0
EXCEED_SPEED_MS = 5.0
EXCEED_BATTERY_PCT = 5.0
LIMIT_KEYS = ("max_alt_agl_m", "max_speed_ms", "min_battery_rtl_pct")

Xy = tuple[float, float]


class FixtureError(ValueError):
    """The fixture file does not have the expected shape (exit code 2)."""


@dataclass(frozen=True)
class Home:
    lat: float
    lon: float
    alt_amsl_m: float
    heading_deg: float

    def sitl_arg(self) -> str:
        """``lat,lon,alt,heading`` for SITL's ``--home``."""
        return f"{self.lat:.6f},{self.lon:.6f},{self.alt_amsl_m:.1f},{self.heading_deg:g}"


@dataclass(frozen=True)
class Pin:
    id: str
    east_m: float
    north_m: float
    alt_m: float | None = None

    @property
    def xy(self) -> Xy:
        return (self.east_m, self.north_m)


@dataclass(frozen=True)
class SurveyBox:
    center_pin: str
    size_m: Xy
    alt_m: float
    center: Xy

    def corners(self) -> list[Xy]:
        cx, cy = self.center
        hx, hy = self.size_m[0] / 2, self.size_m[1] / 2
        return [(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)]


@dataclass(frozen=True)
class Charter:
    name: str
    area_enu: tuple[Xy, ...]
    geofence_buffer_m: float
    alt_min_agl_m: float
    alt_max_agl_m: float
    default_alt_agl_m: float
    max_speed_ms: float
    max_range_from_pilot_m: float
    battery_rtl_pct: float
    allowed_skills: tuple[str, ...]
    sectors: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    # exceed_limits charters: the looser values are only set when load() got the caps;
    # without them the charter keeps its base's values and limits_applied is False.
    exceeds_limits: bool = False
    limits_applied: bool = False


@dataclass(frozen=True)
class Fixture:
    home: Home
    pilot: Xy
    operator: str
    vehicle_id: str
    pins: dict[str, Pin]
    survey_box: SurveyBox
    charters: dict[str, Charter]


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FixtureError(f"{where}: väntade en tabell, fick {type(value).__name__}")
    return value


def _keys(
    data: Mapping[str, Any],
    allowed: set[str],
    where: str,
    *,
    required: bool = True,
    optional: frozenset[str] = frozenset(),
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise FixtureError(f"{where}: okända nycklar {', '.join(unknown)}")
    if required:
        missing = sorted(allowed - set(data) - optional)
        if missing:
            raise FixtureError(f"{where}: saknar {', '.join(missing)}")


def _num(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FixtureError(f"{where}: väntade ett tal, fick {value!r}")
    if not math.isfinite(float(value)):
        raise FixtureError(f"{where}: talet måste vara ändligt")
    return float(value)


def _xy(value: Any, where: str) -> Xy:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise FixtureError(f"{where}: väntade [öster, norr]")
    return (_num(value[0], where), _num(value[1], where))


def _charter(
    name: str,
    raw: Mapping[str, Any],
    bases: Mapping[str, Mapping[str, Any]],
    limits: Mapping[str, float] | None,
) -> Charter:
    where = f"charters.{name}"
    _keys(raw, CHARTER_KEYS | CHARTER_META_KEYS, where, required=False)
    merged: dict[str, Any] = {}
    base_name = raw.get("base")
    if base_name is not None:
        if base_name not in bases:
            raise FixtureError(f"{where}: basramen {base_name!r} finns inte")
        base = bases[base_name]
        if "base" in base:
            raise FixtureError(f"{where}: basramen {base_name!r} har själv en bas (ett led)")
        merged.update({k: v for k, v in base.items() if k not in CHARTER_META_KEYS})
    merged.update({k: v for k, v in raw.items() if k not in CHARTER_META_KEYS})
    _keys(merged, CHARTER_KEYS, where)
    exceed = raw.get("exceed_limits", False)
    if not isinstance(exceed, bool):
        raise FixtureError(f"{where}.exceed_limits: väntade true eller false")
    if exceed and limits is not None:
        if any(k not in limits for k in LIMIT_KEYS):
            raise FixtureError(f"{where}: exceed_limits kräver taken ({', '.join(LIMIT_KEYS)})")
        merged["alt_max_agl_m"] = limits["max_alt_agl_m"] + EXCEED_ALT_M
        merged["max_speed_ms"] = limits["max_speed_ms"] + EXCEED_SPEED_MS
        merged["battery_rtl_pct"] = limits["min_battery_rtl_pct"] - EXCEED_BATTERY_PCT
    area = merged["area_enu"]
    if not isinstance(area, list) or len(area) < 3:
        raise FixtureError(f"{where}.area_enu: väntade minst 3 hörn")
    skills = merged["allowed_skills"]
    if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
        raise FixtureError(f"{where}.allowed_skills: väntade en lista med namn")
    sectors = merged["sectors"]
    if not isinstance(sectors, list):
        raise FixtureError(f"{where}.sectors: väntade en lista")
    for i, sector in enumerate(sectors):
        _keys(_mapping(sector, f"{where}.sectors[{i}]"), SECTOR_KEYS, f"{where}.sectors[{i}]")
    numbers = {
        k: _num(merged[k], f"{where}.{k}")
        for k in CHARTER_KEYS - {"area_enu", "allowed_skills", "sectors"}
    }
    return Charter(
        name=name,
        exceeds_limits=exceed,
        limits_applied=exceed and limits is not None,
        area_enu=tuple(_xy(p, f"{where}.area_enu") for p in area),
        allowed_skills=tuple(skills),
        sectors=tuple(sectors),
        **numbers,
    )


def parse(data: Any, limits: Mapping[str, float] | None = None) -> Fixture:
    """Build a ``Fixture`` from already-parsed YAML."""
    top = _mapping(data, "fixturen")
    _keys(top, TOP_KEYS, "fixturen")
    home_raw = _mapping(top["home"], "home")
    _keys(home_raw, HOME_KEYS, "home")
    home = Home(**{k: _num(home_raw[k], f"home.{k}") for k in HOME_KEYS})
    pilot_raw = _mapping(top["pilot"], "pilot")
    _keys(pilot_raw, POINT_KEYS, "pilot")
    pins: dict[str, Pin] = {}
    if not isinstance(top["pins"], list):
        raise FixtureError("pins: väntade en lista")
    for i, raw in enumerate(top["pins"]):
        where = f"pins[{i}]"
        pin = _mapping(raw, where)
        _keys(pin, PIN_KEYS, where, optional=frozenset({"alt_m"}))
        pid = str(pin["id"])
        if pid in pins:
            raise FixtureError(f"{where}: nålen {pid} finns två gånger")
        alt = pin.get("alt_m")
        pins[pid] = Pin(
            pid,
            _num(pin["east_m"], where),
            _num(pin["north_m"], where),
            None if alt is None else _num(alt, where),
        )
    box_raw = _mapping(top["survey_box"], "survey_box")
    _keys(box_raw, BOX_KEYS, "survey_box")
    center_pin = str(box_raw["center_pin"])
    if center_pin not in pins:
        raise FixtureError(f"survey_box.center_pin: nålen {center_pin} finns inte")
    box = SurveyBox(
        center_pin,
        _xy(box_raw["size_m"], "survey_box.size_m"),
        _num(box_raw["alt_m"], "survey_box.alt_m"),
        pins[center_pin].xy,
    )
    raw_charters = _mapping(top["charters"], "charters")
    bases = {k: _mapping(v, f"charters.{k}") for k, v in raw_charters.items()}
    charters = {name: _charter(name, raw, bases, limits) for name, raw in bases.items()}
    return Fixture(
        home=home,
        pilot=(_num(pilot_raw["east_m"], "pilot"), _num(pilot_raw["north_m"], "pilot")),
        operator=str(top["operator"]),
        vehicle_id=str(top["vehicle_id"]),
        pins=pins,
        survey_box=box,
        charters=charters,
    )


def load(path: Path = DEFAULT_PATH, limits: Mapping[str, float] | None = None) -> Fixture:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FixtureError(f"{path}: {exc}") from exc
    return parse(data, limits)
