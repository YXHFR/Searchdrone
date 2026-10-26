"""Where a lost person walks while we search: a terrain-steered motion model for the
probability map (stopped mass plus walkers in eight headings), and the parameters per
kind of missing person.

numpy only, so both the flight-side map (``sd_search``) and the simulator use it.
"""

from importlib.metadata import version

from sd_motion.agent import walk
from sd_motion.model import DT_S, MotionState, advance, negative, start_state, step
from sd_motion.params import Params, for_category
from sd_motion.terrain import DIRS, Terrain

__version__ = version("sd-motion")

__all__ = [
    "DIRS",
    "DT_S",
    "MotionState",
    "Params",
    "Terrain",
    "__version__",
    "advance",
    "for_category",
    "negative",
    "start_state",
    "step",
    "walk",
]
