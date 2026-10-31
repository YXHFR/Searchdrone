"""``OracleDetector`` and ``SyntheticStillSource``: the onboard camera and detector in the
search scenarios, drawn from a detector profile (``det_oracle_v0``) and the world's truth.

``SyntheticStillSource`` is the stills camera: on each poll (at most ``rate_hz``) it
renders what a downward camera on the drone would see from the flight controller's
latest position and attitude: a dry meadow, and every person and decoy of the ``World`` in
view, drawn in their clothing colours from above (``sd_geo.project``, the same pinhole
model the georeferencing inverts). It remembers what it put in each still.

``OracleDetector`` is the detector source. For each still it draws, per object in view:

* a person is found with the profile's recall for its size in pixels (under 16, 16-32,
  over 32) times its occlusion factor; its box is the drawn outline, a pixel or two off;
* a decoy is taken for a person with its ``lure`` chance;
* false positives on plain ground: Poisson with ``fp_per_12mp`` scaled to the still's
  megapixels;
* scores: Beta(score_hit) for a person, Beta(score_fp) for anything else;
* the latency: log-normal through the profile's p50 and p95 (the source waits it).

Both run inside the real perception task (``OraclePerception``, a ``PerceptionLoop``):
georeferencing, crops, the report and the executor's keep-out cylinders are the
product's own. Every detection says ``source = "oracle"`` and every still
``synthetic``: nothing here is a measure of a real detector (the profile is modelled,
not measured). Both refuse to start outside the simulator (``SD_SIM=1``).
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from PIL import Image, ImageDraw

from sd_detect.api import BBox, Detection, Frame, SourceKind
from sd_detect.sources import BuiltSources
from sd_detect.still import Still
from sd_geo import PinholeCamera
from sd_geo.georef import Attitude, project
from sd_mav.vehicle import FcTelemetry
from sd_onboard.perception.loop import PerceptionLoop
from sd_onboard.perception.sources import CapturedStill
from sd_schemas.colors import COLOR_TABLE
from sd_schemas.geo import local_xy
from sd_sim import paths
from sd_sim.search_world import Thing, World

__all__ = [
    "DEFAULT_PROFILE",
    "NOT_IN_SIMULATOR",
    "DetProfile",
    "OracleDetector",
    "OraclePerception",
    "SyntheticStillSource",
    "View",
    "lab_to_rgb",
    "load_det_profile",
]

DEFAULT_PROFILE = "det_oracle_v0"
PROFILES = paths.REPO / "tools" / "bench" / "profiles"
NOT_IN_SIMULATOR = "Orakeldetektorn körs bara i simulatorn (SD_SIM=1)."
Z95 = 1.6448536
GRASS = (128, 124, 84)  # a dry meadow: a green jacket stands out from it
SKIN = (196, 150, 120)
MIN_AGL_M = 2.0
VIEWS_KEPT = 16  # stills whose truth waits for the detector
FP_SIZE_M = (0.4, 1.4)
POS_SIGMA_M = 1.2  # the receiver's horizontal error, per axis
ATT_SIGMA_DEG = 0.7  # the attitude estimate's error, per axis


def _need_sim(env: Mapping[str, str] | None) -> None:
    if (os.environ if env is None else env).get("SD_SIM") != "1":
        raise RuntimeError(NOT_IN_SIMULATOR)


@dataclass(frozen=True)
class DetProfile:
    profile_id: str
    recall: Mapping[str, float]  # "lt16", "16to32", "gt32"
    occlusion: Mapping[str, float]  # "0-30", "30-60", "60-90"
    fp_per_12mp: float
    score_hit: tuple[float, float]
    score_fp: tuple[float, float]
    latency_ms: tuple[float, float]  # p50, p95
    label: Literal["modelled", "measured"]

    def p_found(self, px: float, occlusion_pct: float) -> float:
        r = self.recall["lt16" if px < 16 else "16to32" if px <= 32 else "gt32"]
        o = "0-30" if occlusion_pct < 30 else "30-60" if occlusion_pct < 60 else "60-90"
        return r * self.occlusion[o]


def load_det_profile(name_or_path: str | Path = DEFAULT_PROFILE) -> DetProfile:
    """A profile by name (``tools/bench/profiles/<name>.json``) or from a file."""
    path = Path(name_or_path)
    if path.suffix != ".json":
        path = PROFILES / f"{name_or_path}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return DetProfile(
        profile_id=str(raw["profile_id"]),
        recall={k: float(v) for k, v in raw["recall"].items()},
        occlusion={k: float(v) for k, v in raw["occlusion_factor"].items()},
        fp_per_12mp=float(raw["fp_per_12mp"]),
        score_hit=(float(raw["score_hit"]["a"]), float(raw["score_hit"]["b"])),
        score_fp=(float(raw["score_fp"]["a"]), float(raw["score_fp"]["b"])),
        latency_ms=(float(raw["latency_ms"]["p50"]), float(raw["latency_ms"]["p95"])),
        label="measured" if not raw.get("not_measured") else "modelled",
    )


def lab_to_rgb(lab: tuple[float, float, float]) -> tuple[int, int, int]:
    """CIELAB (D65) to 8-bit sRGB, clipped."""
    l_, a, b = lab
    fy = (l_ + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200

    def inv(f: float) -> float:
        return f**3 if f**3 > 0.008856 else (f - 16 / 116) / 7.787

    x, y, z = 0.95047 * inv(fx), 1.0 * inv(fy), 1.08883 * inv(fz)
    lin = (
        3.2406 * x - 1.5372 * y - 0.4986 * z,
        -0.9689 * x + 1.8758 * y + 0.0415 * z,
        0.0557 * x - 0.2040 * y + 1.0570 * z,
    )

    def gamma(c: float) -> int:
        c = max(0.0, min(1.0, c))
        v = 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
        return round(255 * v)

    r, g, bl = (gamma(c) for c in lin)
    return r, g, bl


def _rgb(name: str | None, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if name is None:
        return default
    ref = COLOR_TABLE[name]  # type: ignore[index]
    return lab_to_rgb(tuple(float(v) for v in ref.lab))  # type: ignore[arg-type]


# A thing's parts seen from above, in metres: x along its heading, y to its left.
Rect = tuple[float, float, float, float]  # x0, x1, y0, y1
_PARTS: dict[str, list[tuple[str, Rect]]] = {
    "lying": [
        ("lower", (-0.85, -0.05, -0.15, 0.15)),
        ("upper", (-0.05, 0.65, -0.28, 0.28)),
        ("upper", (0.0, 0.55, -0.38, -0.28)),  # the jacket's sleeves, arms at the sides
        ("upper", (0.0, 0.55, 0.28, 0.38)),
        ("head", (0.65, 0.87, -0.1, 0.1)),
        ("backpack", (0.0, 0.55, 0.38, 0.8)),
    ],
    "sitting": [
        ("lower", (-0.55, -0.1, -0.18, 0.18)),
        ("upper", (-0.1, 0.3, -0.23, 0.23)),
        ("head", (0.3, 0.45, -0.1, 0.1)),
        ("backpack", (-0.1, 0.25, 0.23, 0.5)),
    ],
    "standing": [
        ("backpack", (-0.35, -0.12, -0.2, 0.2)),
        ("upper", (-0.15, 0.15, -0.25, 0.25)),
        ("head", (-0.1, 0.1, -0.1, 0.1)),
    ],
}


@dataclass(frozen=True)
class View:
    """An object in a still: its box (in the picture) and its size in pixels."""

    thing: Thing
    bbox: BBox
    px: float


class SyntheticStillSource:
    """The stills camera over the world (see the module docstring)."""

    def __init__(
        self,
        world: World,
        telemetry: Callable[[], FcTelemetry],
        camera: PinholeCamera,
        *,
        rate_hz: float = 1.0,
        monotonic: Callable[[], float] = time.monotonic,
        env: Mapping[str, str] | None = None,
    ) -> None:
        _need_sim(env)
        self.world = world
        self.telemetry = telemetry
        self.camera = camera
        self.period_s = 1.0 / rate_hz
        self.monotonic = monotonic
        self._last: float | None = None
        self._views: OrderedDict[int, list[View]] = OrderedDict()
        self._lock = threading.Lock()
        self.stills = 0
        shape = (camera.height_px, camera.width_px, 1)
        noise = np.random.default_rng(world.seed).normal(0, 9, shape)
        grass = np.clip(np.array(GRASS, dtype=np.float64) + noise, 0, 255).astype(np.uint8)
        self._grass = Image.fromarray(grass, "RGB")
        self._noise = np.random.default_rng(world.seed + 1)

    def _outline(
        self, thing: Thing, att: Attitude, dx: float, dy: float, h: float
    ) -> list[tuple[str, list[tuple[float, float]]]]:
        """The thing's parts as picture polygons (``dx``, ``dy``: east and north from the
        camera to the thing)."""
        heading = math.radians(thing.heading_deg)
        ce, se = math.sin(heading), math.cos(heading)  # x axis: east and north parts

        def to_px(x: float, y: float) -> tuple[float, float] | None:
            east = dx + x * ce - y * se
            north = dy + x * se + y * ce
            return project(east, north, h, self.camera, att)

        if thing.person:
            parts = [(n, r) for n, r in _PARTS[thing.pose] if n != "backpack" or thing.backpack]
        elif thing.kind == "rock":
            s = thing.size_m / 2
            parts = [("rock", (-s, s, -0.7 * s, 0.7 * s))]
        elif thing.kind == "log":
            s = thing.size_m / 2
            parts = [("log", (-s, s, -0.15, 0.15))]
        elif thing.kind == "tarp":
            s = thing.size_m / 2
            parts = [("tarp", (-s, s, -0.6 * s, 0.6 * s))]
        else:  # a backpack on the ground
            parts = [("item", (-0.25, 0.25, -0.17, 0.17))]
        out = []
        for name, (x0, x1, y0, y1) in parts:
            pts = [to_px(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
            if any(p is None for p in pts):
                return []
            out.append((name, [p for p in pts if p is not None]))
        return out

    def _fill(self, thing: Thing, part: str) -> tuple[int, int, int]:
        if part == "upper":
            return _rgb(thing.upper, (110, 110, 110))
        if part == "lower":
            return _rgb(thing.lower, (60, 60, 70))
        if part == "backpack":
            return _rgb(thing.backpack, (90, 90, 90))
        if part == "head":
            return SKIN
        if part == "log":
            return _rgb(thing.color or "brown", (100, 70, 40))
        return _rgb(thing.color or "gray", (120, 120, 120))

    def render(self, t: FcTelemetry) -> tuple[Frame, list[View]]:
        cam = self.camera
        img = self._grass.copy()
        views: list[View] = []
        h = t.alt_rel_home_m
        if h >= MIN_AGL_M:
            draw = ImageDraw.Draw(img)
            # Where the camera really was: the telemetry's position and attitude are off
            # by the receiver's and the attitude estimate's error (georeferencing's error).
            g = self._noise
            ge, gn = g.normal(0, POS_SIGMA_M), g.normal(0, POS_SIGMA_M)
            att = Attitude(
                t.roll_deg + g.normal(0, ATT_SIGMA_DEG),
                t.pitch_deg + g.normal(0, ATT_SIGMA_DEG),
                t.yaw_deg + g.normal(0, ATT_SIGMA_DEG),
            )
            reach = 2.0 * h + 10.0  # nothing farther is in a downward view
            for thing in self.world.things:
                dx, dy = local_xy(t.lat, t.lon, *self.world.where(thing))
                dx, dy = dx - ge, dy - gn
                if math.hypot(dx, dy) > reach:
                    continue
                parts = self._outline(thing, att, dx, dy, h)
                if not parts:
                    continue
                us = [u for _, pts in parts for u, _ in pts]
                vs = [v for _, pts in parts for _, v in pts]
                u0, u1, v0, v1 = min(us), max(us), min(vs), max(vs)
                cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
                if not (0 <= cu < cam.width_px and 0 <= cv < cam.height_px):
                    continue
                for name, pts in parts:
                    draw.polygon(pts, fill=self._fill(thing, name))
                x0, y0 = max(0.0, u0), max(0.0, v0)
                x1, y1 = min(float(cam.width_px), u1), min(float(cam.height_px), v1)
                if x1 - x0 < 1 or y1 - y0 < 1:
                    continue
                views.append(
                    View(thing, BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0), max(u1 - u0, v1 - v0))
                )
        frame = Frame(
            width=cam.width_px,
            height=cam.height_px,
            rgb=img.tobytes(),
            ts_mono_ms=int(self.monotonic() * 1000),
        )
        return frame, views

    def poll(self) -> CapturedStill | None:
        now = self.monotonic()
        if self._last is not None and now - self._last < self.period_s:
            return None
        self._last = now
        frame, views = self.render(self.telemetry())
        still = Still(frame=frame)
        with self._lock:
            self._views[id(still)] = views
            while len(self._views) > VIEWS_KEPT:
                self._views.popitem(last=False)
        self.stills += 1
        return CapturedStill(still=still, name=f"oracle-{self.stills:06d}", synthetic=True)

    def take(self, still: Still) -> list[View]:
        """What the source put in ``still`` (once)."""
        with self._lock:
            return self._views.pop(id(still), [])


@dataclass(frozen=True)
class Drawn:
    """One detection the oracle made, for the run's record (truth included)."""

    thing_id: str | None  # None: a false positive on plain ground
    person: bool
    target: bool
    score: float
    px: float


class OracleDetector:
    name: ClassVar[str] = "det_oracle"
    license: ClassVar[str] = "Apache-2.0"
    source: ClassVar[SourceKind] = "oracle"

    def __init__(
        self,
        stills: SyntheticStillSource,
        profile: DetProfile | None = None,
        *,
        seed: int = 0,
        sleep: bool = True,
        env: Mapping[str, str] | None = None,
    ) -> None:
        _need_sim(env)
        self.stills = stills
        self.profile = profile or load_det_profile()
        self.rng = random.Random(seed)  # noqa: S311 - a simulation, not a secret
        self.sleep = sleep
        self.drawn: list[Drawn] = []
        self.in_view: dict[str, int] = {}  # thing id -> stills it was in

    def _latency_s(self) -> float:
        p50, p95 = self.profile.latency_ms
        sigma = (math.log(p95) - math.log(p50)) / Z95
        return self.rng.lognormvariate(math.log(p50), sigma) / 1000

    def _box(self, b: BBox, w: int, h: int) -> BBox:
        j = [self.rng.uniform(-1.5, 1.5) for _ in range(4)]
        x0, y0 = max(0.0, b.x + j[0]), max(0.0, b.y + j[1])
        x1, y1 = min(float(w), b.x + b.w + j[2]), min(float(h), b.y + b.h + j[3])
        return BBox(x=x0, y=y0, w=max(1.0, x1 - x0), h=max(1.0, y1 - y0))

    def _poisson(self, lam: float) -> int:
        n, p, limit = 0, 1.0, math.exp(-lam)
        while True:
            p *= self.rng.random()
            if p <= limit:
                return n
            n += 1

    def detect_still(self, still: Still) -> list[Detection]:
        views = self.stills.take(still)
        f = still.frame
        prof = self.profile
        out: list[Detection] = []
        for v in views:
            t = v.thing
            self.in_view[t.thing_id] = self.in_view.get(t.thing_id, 0) + 1
            p = prof.p_found(v.px, t.occlusion_pct) if t.person else t.lure
            if self.rng.random() >= p:
                continue
            a, b = prof.score_hit if t.person else prof.score_fp
            score = round(self.rng.betavariate(a, b), 4)
            out.append(
                Detection(label="person", score=score, bbox=self._box(v.bbox, f.width, f.height))
            )
            self.drawn.append(Drawn(t.thing_id, t.person, t.target, score, v.px))
        # False positives on plain ground, as large as a person would be here.
        cam = self.stills.camera
        h = max(MIN_AGL_M, self.stills.telemetry().alt_rel_home_m)
        ppm = cam.focal_px / h
        for _ in range(self._poisson(prof.fp_per_12mp * f.width * f.height / 12e6)):
            side = self.rng.uniform(*FP_SIZE_M) * ppm
            x, y = self.rng.uniform(0, f.width - side), self.rng.uniform(0, f.height - side)
            score = round(self.rng.betavariate(*prof.score_fp), 4)
            out.append(Detection(label="person", score=score, bbox=BBox(x=x, y=y, w=side, h=side)))
            self.drawn.append(Drawn(None, False, False, score, side))
        if self.sleep:
            time.sleep(self._latency_s())  # in the perception task's worker thread
        return out


class OraclePerception(PerceptionLoop):
    """The product's perception task with the oracle as its only detector source (under
    the register's ``model`` name, the source the person class names)."""

    def __init__(self, detector: OracleDetector, *args: Any, **kwargs: Any) -> None:
        kwargs["sources"] = BuiltSources(sources={"model": detector})
        super().__init__(detector.stills, *args, **kwargs)
        self.detector = detector
