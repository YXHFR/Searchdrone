"""Geometry helpers in a local east/north plane (metres): search patterns, coverage and
georeferencing (pixel to ground).

Projection between WGS84 and the plane, distances and containment are the gate's
(``sd_safety.targets``), so the gate and the executor measure with the same geometry.
"""

from importlib.metadata import version

from sd_geo.coverage import coverage_fraction
from sd_geo.georef import (
    Attitude,
    PinholeCamera,
    footprint,
    ground_offset,
    project,
    px_per_m,
    sigma95_m,
)
from sd_geo.patterns import PatternError, lawnmower

__version__ = version("sd-geo")

__all__ = [
    "Attitude",
    "PatternError",
    "PinholeCamera",
    "__version__",
    "coverage_fraction",
    "footprint",
    "ground_offset",
    "lawnmower",
    "project",
    "px_per_m",
    "sigma95_m",
]
