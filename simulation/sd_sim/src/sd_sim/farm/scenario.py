"""One farm round's settings, drawn from a seed: the same seed always gives the same round.

Drawn per round: the world, the day of the year and the (solar) time of day with the
sun's direction, the weather, the flight height, the people (model, pose, clothes
colour, place, heading; the first one is the search target) and decoys that are not
people (a backpack, a tarp, clothes, a log). The people and decoys stand inside the area
the round searches, which the caller gives as a rectangle in metres east/north of home.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from sd_sim import objects
from sd_sim.farm.people import CATALOG, Person

SCHEMA = "simfarm.scenario/v1"
HEIGHTS_M = (25.0, 60.0)
MAX_PEOPLE = 4
MAX_DECOYS = 6
MIN_GAP_M = 6.0
EDGE_M = 5.0
MIN_SUN_ELEVATION_DEG = 5.0
DAYS = (60, 320)  # March to mid-November: the season without snow cover

# name: sun intensity, ambient light, cloud cover 0-1, haze 0-1, shadows
WEATHER: dict[str, tuple[float, float, float, float, bool]] = {
    "clear": (1.0, 0.35, 0.0, 0.0, True),
    "cloudy": (0.75, 0.45, 0.5, 0.0, True),
    "overcast": (0.35, 0.6, 1.0, 0.1, False),
    "haze": (0.6, 0.55, 0.3, 0.5, True),
}
# Clothes colours that searches meet: bright jackets and dull outdoor wear alike.
CLOTHES: dict[str, tuple[float, float, float]] = {
    "red": (0.75, 0.08, 0.08),
    "orange": (0.9, 0.4, 0.05),
    "yellow": (0.9, 0.8, 0.1),
    "blue": (0.1, 0.2, 0.6),
    "green": (0.15, 0.4, 0.15),
    "olive": (0.33, 0.35, 0.18),
    "brown": (0.4, 0.27, 0.15),
    "grey": (0.45, 0.45, 0.47),
    "black": (0.06, 0.06, 0.07),
    "white": (0.88, 0.88, 0.86),
}
TINTED_SHARE = 0.5
LYING_SHARE = 0.4  # of the standing models
# name: size (length, width, height), colours to draw from
DECOYS: dict[str, tuple[tuple[float, float, float], tuple[str, ...]]] = {
    "backpack": ((0.35, 0.25, 0.5), ("red", "blue", "green", "black", "orange")),
    "tarp": ((2.0, 1.5, 0.03), ("blue", "green", "orange", "grey")),
    "clothes": ((0.6, 0.4, 0.1), ("red", "blue", "yellow", "black", "white")),
    "log": ((2.5, 0.3, 0.3), ("brown",)),
}


@dataclass(frozen=True)
class Sun:
    elevation_deg: float
    azimuth_deg: float  # clockwise from north

    def direction(self) -> tuple[float, float, float]:
        """Unit vector the sunlight travels along, world frame (x east, y north, z up)."""
        el = math.radians(self.elevation_deg)
        az = math.radians(self.azimuth_deg)
        towards_sun = (math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el))
        return (-towards_sun[0], -towards_sun[1], -towards_sun[2])


def sun_position(lat_deg: float, day_of_year: int, solar_hour: float) -> Sun:
    """The sun's elevation and azimuth at local solar time (good to a degree or two)."""
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + day_of_year) / 365)
    lat = math.radians(lat_deg)
    hour_angle = math.radians(15.0 * (solar_hour - 12.0))
    sin_el = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)
    el = math.asin(max(-1.0, min(1.0, sin_el)))
    az = math.atan2(
        math.sin(hour_angle),
        math.cos(hour_angle) * math.sin(lat) - math.tan(decl) * math.cos(lat),
    )
    return Sun(round(math.degrees(el), 2), round((math.degrees(az) + 180.0) % 360.0, 2))


@dataclass(frozen=True)
class Decoy:
    name: str
    kind: str
    east_m: float
    north_m: float
    yaw_deg: float
    colour: str

    def box(self) -> objects.Box:
        size, _ = DECOYS[self.kind]
        return objects.Box(
            name=self.name,
            east_m=self.east_m,
            north_m=self.north_m,
            size_m=size,
            color=CLOTHES[self.colour],
            yaw_deg=self.yaw_deg,
            truth={"cls": "decoy", "role": "decoy", "kind": self.kind, "colour": self.colour},
        )


@dataclass(frozen=True)
class FarmScenario:
    seed: int
    world: str
    day_of_year: int
    solar_hour: float
    sun: Sun
    weather: str
    height_m: float
    people: tuple[Person, ...]
    decoys: tuple[Decoy, ...]
    area: tuple[float, float, float, float]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def light(self) -> dict[str, Any]:
        sun, ambient, clouds, haze, shadows = WEATHER[self.weather]
        # A low sun is weaker: scale by the air mass, roughly.
        low = max(0.2, min(1.0, math.sin(math.radians(self.sun.elevation_deg)) * 2.0))
        return {
            "sun_intensity": round(sun * low, 3),
            "ambient": ambient,
            "clouds": clouds,
            "haze": haze,
            "shadows": shadows,
            "sun_direction": [round(v, 4) for v in self.sun.direction()],
        }

    def objects(self) -> list[objects.WorldObject]:
        return [*self.people, *(d.box() for d in self.decoys)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "seed": self.seed,
            "world": self.world,
            "day_of_year": self.day_of_year,
            "solar_hour": self.solar_hour,
            "sun": asdict(self.sun),
            "weather": self.weather,
            "light": self.light,
            "height_m": self.height_m,
            "area_m": list(self.area),
            "people": [
                {
                    "name": p.name,
                    "model": p.model,
                    "pose": p.pose,
                    "east_m": p.east_m,
                    "north_m": p.north_m,
                    "yaw_deg": p.yaw_deg,
                    "tint": list(p.tint) if p.tint else None,
                    "is_target": p.is_target,
                }
                for p in self.people
            ],
            "decoys": [asdict(d) for d in self.decoys],
            **self.extra,
        }


def _spots(
    rng: random.Random, area: tuple[float, float, float, float], count: int
) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = area
    if x1 - x0 < 2 * EDGE_M or y1 - y0 < 2 * EDGE_M:
        raise ValueError("sökområdet är för litet")
    out: list[tuple[float, float]] = []
    if count == 0:
        return out
    for _ in range(20_000):
        e = round(rng.uniform(x0 + EDGE_M, x1 - EDGE_M), 2)
        n = round(rng.uniform(y0 + EDGE_M, y1 - EDGE_M), 2)
        if math.hypot(e, n) < objects.HOME_KEEP_OUT_M + 3.0:
            continue
        if all(math.hypot(e - a, n - b) >= MIN_GAP_M for a, b in out):
            out.append((e, n))
            if len(out) == count:
                return out
    raise ValueError(f"hittade inte {count} lediga platser i sökområdet")


def _person(rng: random.Random, i: int, spot: tuple[float, float], target: bool) -> Person:
    model = rng.choice(CATALOG)
    if model.pose == "sitting":
        pose = "sitting"
    else:
        pose = "lying" if rng.random() < LYING_SHARE else "standing"
    tint = None
    if rng.random() < TINTED_SHARE:
        tint = CLOTHES[rng.choice(sorted(CLOTHES))]
    return Person(
        name=f"person_{i}",
        model=model.id,
        pose=pose,
        east_m=spot[0],
        north_m=spot[1],
        yaw_deg=round(rng.uniform(0.0, 360.0), 1),
        tint=tint,
        is_target=target,
    )


def sample(
    seed: int,
    worlds: Sequence[str],
    area: tuple[float, float, float, float],
    lat_deg: float,
) -> FarmScenario:
    """The round for ``seed``: ``worlds`` to draw from, the search ``area`` (x0, y0, x1,
    y1 metres east/north of home) and the world's latitude for the sun."""
    if not worlds:
        raise ValueError("ingen värld att välja bland")
    rng = random.Random(f"simfarm:{seed}")  # noqa: S311 - scenery, not security
    world = rng.choice(list(worlds))
    day = rng.randint(*DAYS)
    for _ in range(200):
        hour = round(rng.uniform(4.0, 20.0), 2)
        sun = sun_position(lat_deg, day, hour)
        if sun.elevation_deg >= MIN_SUN_ELEVATION_DEG:
            break
    else:
        hour, sun = 12.0, sun_position(lat_deg, day, 12.0)
    weather = rng.choice(sorted(WEATHER))
    low, high = HEIGHTS_M
    height = float(round(rng.uniform(low, high) / 5.0) * 5.0)
    n_people = rng.randint(1, MAX_PEOPLE)
    n_decoys = rng.randint(0, MAX_DECOYS)
    spots = _spots(rng, area, n_people + n_decoys)
    people = tuple(_person(rng, i, spots[i], i == 0) for i in range(n_people))
    decoys = []
    for k, spot in enumerate(spots[n_people:]):
        kind = rng.choice(sorted(DECOYS))
        decoys.append(
            Decoy(
                name=f"decoy_{k}",
                kind=kind,
                east_m=spot[0],
                north_m=spot[1],
                yaw_deg=round(rng.uniform(0.0, 180.0), 1),
                colour=rng.choice(DECOYS[kind][1]),
            )
        )
    return FarmScenario(
        seed=seed,
        world=world,
        day_of_year=day,
        solar_hour=hour,
        sun=sun,
        weather=weather,
        height_m=height,
        people=people,
        decoys=tuple(decoys),
        area=area,
    )
