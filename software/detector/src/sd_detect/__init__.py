"""The detector interface, the class register and the detector sources."""

from importlib.metadata import version

from sd_detect.api import SOURCES as SOURCE_KINDS
from sd_detect.api import BBox, Detection, Detector, Frame
from sd_detect.classes import ClassRegistry, ClassSpec, RegistryError, default_registry
from sd_detect.color import ColorDetector
from sd_detect.fake import NOT_BUILT, NullDetector
from sd_detect.oracle import SimLabelDetector
from sd_detect.sources import SOURCES, build_sources, detect_all, keep_plausible
from sd_detect.still import SimBox, Still, StillDetector
from sd_detect.tiling import Tile, TiledDetector, tile_grid

__all__ = [
    "NOT_BUILT",
    "SOURCES",
    "SOURCE_KINDS",
    "BBox",
    "ClassRegistry",
    "ClassSpec",
    "ColorDetector",
    "Detection",
    "Detector",
    "Frame",
    "NullDetector",
    "RegistryError",
    "SimBox",
    "SimLabelDetector",
    "Still",
    "StillDetector",
    "Tile",
    "TiledDetector",
    "__version__",
    "build_sources",
    "default_registry",
    "detect_all",
    "keep_plausible",
    "tile_grid",
]

__version__ = version("sd-detect")
