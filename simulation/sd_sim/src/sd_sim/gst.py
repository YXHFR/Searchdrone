"""GStreamer from Python (``gi.repository.Gst``) for the video processes.

PyGObject comes from the distribution (``python3-gi``, installed by the simulator setup
and present in the SITL image), not from PyPI: it needs the system's GObject
introspection data. The project's virtual environment does not see the distribution's
packages, so when ``gi`` is missing this adds the distribution's package folder at the
END of ``sys.path`` (the environment's own packages keep priority) and tries again.
Only the video processes import this module.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

__all__ = ["DIST_PACKAGES", "GstMissingError", "has_element", "load_gst"]

DIST_PACKAGES = ("/usr/lib/python3/dist-packages",)
HINT = (
    "GStreamer för Python saknas. Kör just sim-setup (eller: sudo apt-get install python3-gi "
    "gir1.2-gstreamer-1.0 gstreamer1.0-plugins-good gstreamer1.0-plugins-bad "
    "gstreamer1.0-plugins-ugly gstreamer1.0-rtsp)."
)


class GstMissingError(RuntimeError):
    """GStreamer or its Python bindings are not installed."""


def _import_gi() -> Any:
    try:
        import gi  # pyright: ignore[reportMissingImports]
    except ImportError:
        for folder in DIST_PACKAGES:
            if (Path(folder) / "gi").is_dir() and folder not in sys.path:
                sys.path.append(folder)
        try:
            import gi  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise GstMissingError(HINT) from exc
    return gi


def load_gst() -> Any:
    """``Gst``, initialised. Raises :class:`GstMissingError` with an install hint."""
    gi = _import_gi()
    try:
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # pyright: ignore[reportMissingImports]
    except (ImportError, ValueError) as exc:
        raise GstMissingError(HINT) from exc
    Gst.init(None)
    return Gst


def has_element(gst: Any, name: str) -> bool:
    return gst.ElementFactory.find(name) is not None
