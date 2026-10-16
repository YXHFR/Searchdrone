"""Geometry helpers in a local east/north plane (metres): search patterns and coverage.

Projection between WGS84 and the plane, distances and containment are the gate's
(``sd_safety.targets``), so the gate and the executor measure with the same geometry.
"""

from importlib.metadata import version

from sd_geo.coverage import coverage_fraction
from sd_geo.patterns import PatternError, lawnmower

__version__ = version("sd-geo")

__all__ = ["PatternError", "__version__", "coverage_fraction", "lawnmower"]
