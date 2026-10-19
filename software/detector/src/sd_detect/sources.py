"""Detector sources by name, and the steps every still goes through.

``SOURCES`` maps a register ``source`` name to a factory that builds that source from
the register. A new class needs a register row; a new kind of source also needs an entry
here. ``build_sources`` builds every source the register names, ``detect_all`` runs them
on a still and keeps only labels the register gives to that source, and
``keep_plausible`` drops detections whose size on the ground does not fit their class.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from sd_detect.api import Detection
from sd_detect.classes import ClassRegistry, RegistryError
from sd_detect.color import ColorDetector
from sd_detect.oracle import SimLabelDetector
from sd_detect.still import Still, StillDetector

__all__ = [
    "SOURCES",
    "BuiltSources",
    "SourceFactory",
    "build_sources",
    "detect_all",
    "keep_plausible",
]

SourceFactory = Callable[[ClassRegistry, Mapping[str, str]], StillDetector | None]
"""Builds a source from the register and the environment; ``None`` when the source is
not built yet."""


def _color(registry: ClassRegistry, env: Mapping[str, str]) -> StillDetector:
    del env
    return ColorDetector.from_registry(registry)


def _oracle(registry: ClassRegistry, env: Mapping[str, str]) -> StillDetector:
    return SimLabelDetector.from_registry(registry, env)


def _model_not_built(registry: ClassRegistry, env: Mapping[str, str]) -> StillDetector | None:
    del registry, env  # no trained detector yet: its classes are simply not reported
    return None


SOURCES: Mapping[str, SourceFactory] = {
    "color": _color,
    "oracle": _oracle,
    "model": _model_not_built,
}


@dataclass
class BuiltSources:
    sources: dict[str, StillDetector] = field(default_factory=dict[str, StillDetector])
    """Register source name -> the running source."""
    skipped: dict[str, str] = field(default_factory=dict[str, str])
    """Source name -> why it does not run (not built yet, simulator only)."""


def build_sources(
    registry: ClassRegistry,
    factories: Mapping[str, SourceFactory] = SOURCES,
    env: Mapping[str, str] | None = None,
) -> BuiltSources:
    """One source per ``source`` name in the register. The oracle runs only in the
    simulator (``SD_SIM=1``); anywhere else its classes are skipped, never faked."""
    env = os.environ if env is None else env
    in_sim = env.get("SD_SIM") == "1"
    out = BuiltSources()
    for name in dict.fromkeys(spec.source for spec in registry):
        factory = factories.get(name)
        if factory is None:
            raise RegistryError(f"the register names the source {name!r}, which does not exist")
        if name == "oracle" and not in_sim:
            out.skipped[name] = "bara i simulatorn"
            continue
        source = factory(registry, env)
        if source is None:
            out.skipped[name] = "inte byggd än"
        else:
            out.sources[name] = source
    return out


def detect_all(
    still: Still, sources: Mapping[str, StillDetector], registry: ClassRegistry
) -> list[Detection]:
    """Every source on the still (by register source name); a label the register does not
    give to that source is dropped (a source cannot invent classes). ``source`` on each
    detection is the source's kind (model, color or oracle)."""
    out: list[Detection] = []
    for name, source in sources.items():
        allowed = {spec.id for spec in registry if spec.source == name}
        out.extend(
            d.model_copy(update={"source": source.source})
            for d in source.detect_still(still)
            if d.label in allowed
        )
    return out


def keep_plausible(
    detections: list[Detection], px_per_m: float, registry: ClassRegistry
) -> list[Detection]:
    """Drop detections whose longest side on the ground falls outside their class's
    ``size_m`` (``px_per_m``: picture pixels per metre on the ground)."""
    if not px_per_m > 0:
        raise ValueError("px_per_m must be positive")
    return [
        d for d in detections if registry.get(d.label).fits_size(max(d.bbox.w, d.bbox.h) / px_per_m)
    ]
