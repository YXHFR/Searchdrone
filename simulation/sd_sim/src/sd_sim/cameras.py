"""The stills cameras on ``sd_quad14``: the main camera and, when asked for, a tele camera.

Both look straight down from the same camera link (no gimbal), each with a
bounding-box twin of the same picture for the simulator's labels. Each publishes on its
own Gazebo Transport topics, ``<prefix>/<sensor>`` and ``<prefix>/<boxes>``, and gets its
own ``gz_stills capture`` process and still folder.

The main camera is ``world.STILLS`` (Raspberry Pi Camera Module 3, 4608 x 2592 at 66
degrees, rendered at half its width and height, 2304 x 1296: a quarter of the pixels, for
Gazebo's render cost). The tele camera is a Raspberry Pi HQ Camera (Sony IMX477, 4056 x
3040 px, 1.55 um pixels) behind an M12 tele lens of about 20 degrees horizontally. The
simulator renders it at the main camera's scale, half its width and height: 2028 x 1520.
The two keep their real ratio of pixels per degree (about 2.9x): a person is about three
times as many pixels across in the tele still as in the main one from the same height.
``tele=true`` (``SimConfig``) turns it on; ``tele_hfov_deg``, ``tele_width`` and
``tele_height`` change it, e.g. to the lens's field of view once it is measured, or to the
sensor's full 4056 x 3040 (slow to render). The onboard computer places a smaller still
of its camera's shape with the same field of view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "TELE_HFOV_DEG",
    "TELE_SENSOR_PX",
    "TELE_SIZE_PX",
    "StillsCamera",
    "tele",
]

TELE_SENSOR_PX = (4056, 3040)  # the HQ Camera's sensor, for reference
TELE_SIZE_PX = (2028, 1520)  # rendered: the sensor at the main camera's scale (half)
TELE_HFOV_DEG = 20.0  # the M12 tele lens on the HQ Camera, as the seller states it
TELE_RATE_HZ = 1.0  # as the main stills camera


@dataclass(frozen=True)
class StillsCamera:
    """One downward stills camera and its bounding-box twin."""

    sensor: str  # the Gazebo sensor's name, and its topic under the instance's prefix
    boxes: str  # the bounding-box twin's name and topic
    width: int
    height: int
    hfov_deg: float
    rate_hz: float = 1.0

    def __post_init__(self) -> None:
        if not (16 <= self.width <= 8192 and 16 <= self.height <= 8192):
            raise ValueError(f"{self.sensor}: bilden ska vara 16-8192 px på varje led")
        if not 1.0 <= self.hfov_deg <= 120.0:
            raise ValueError(f"{self.sensor}: synfältet ska vara 1-120 grader")
        if self.rate_hz <= 0:
            raise ValueError(f"{self.sensor}: bildtakten ska vara över 0")

    @property
    def hfov_rad(self) -> float:
        return round(math.radians(self.hfov_deg), 6)

    @property
    def px_per_deg(self) -> float:
        return self.width / self.hfov_deg

    def context(self, topic_prefix: str) -> dict[str, Any]:
        """What the model template needs for this camera."""
        return {
            "sensor": self.sensor,
            "boxes": self.boxes,
            "width": self.width,
            "height": self.height,
            "hfov_rad": self.hfov_rad,
            "rate_hz": self.rate_hz,
            "topic_prefix": topic_prefix,
        }


def tele(
    width: int = TELE_SIZE_PX[0], height: int = TELE_SIZE_PX[1], hfov_deg: float = TELE_HFOV_DEG
) -> StillsCamera:
    """The tele camera (Raspberry Pi HQ Camera with a 20-degree M12 lens) as rendered."""
    return StillsCamera("tele", "tele_boxes", width, height, hfov_deg, TELE_RATE_HZ)
