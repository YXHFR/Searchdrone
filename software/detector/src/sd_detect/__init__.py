"""The person-detector interface and a null detector."""

from importlib.metadata import version

from sd_detect.api import BBox, Detection, Detector, Frame
from sd_detect.fake import NOT_BUILT, NullDetector

__all__ = ["NOT_BUILT", "BBox", "Detection", "Detector", "Frame", "NullDetector", "__version__"]

__version__ = version("sd-detect")
