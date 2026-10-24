"""Which person detector the simulator's onboard side runs: ``detector=oracle`` or
``detector=model``.

``oracle`` (the default, and what CI runs) keeps the stand-in: the search chain's
``OracleDetector`` on synthetic stills, and ``sd-onboard`` without a model. ``model``
runs the owner's trained detector (``tools/train/README.md``): its manifest comes from
``model=<path to .json>`` or else ``SD_DETECTOR_MODEL``, and ``sd-onboard`` gets it as
``SD_DETECTOR_MODEL``. ``choose`` checks the manifest and the model file's SHA-256
before anything starts and refuses to start on any doubt; it never falls back to the
stand-in. (``sd-onboard`` itself still only turns perception off on a bad model, since
it may be restarted in the air.) ``SD_SIM__DETECTOR`` sets the mode like the other
``SD_SIM__`` settings.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from sd_detect.onnx import ModelError, check_model
from sd_detect.sources import MODEL_ENV

__all__ = [
    "DETECTORS",
    "MODE_ENV",
    "ORACLE",
    "DetectorChoice",
    "DetectorError",
    "choose",
]

DETECTORS = ("oracle", "model")
MODE_ENV = "SD_SIM__DETECTOR"


class DetectorError(ValueError):
    """The chosen detector cannot run (Swedish text for the person starting it)."""


@dataclass(frozen=True)
class DetectorChoice:
    mode: str = "oracle"
    manifest: Path | None = None  # set for "model", checked

    @property
    def model(self) -> bool:
        return self.mode == "model"

    @property
    def name(self) -> str:
        """For reports (``search_metrics.json``): ``model:<manifest name>``, or the mode."""
        return f"model:{self.manifest.stem}" if self.manifest is not None else self.mode

    def env(self) -> dict[str, str]:
        """What a process running the detector needs: the manifest, or nothing."""
        return {} if self.manifest is None else {MODEL_ENV: str(self.manifest)}


ORACLE = DetectorChoice()


def choose(
    detector: str | None = None,
    model: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> DetectorChoice:
    """The detector from ``detector=`` and ``model=`` (else the environment), checked."""
    env = os.environ if environ is None else environ
    mode = (detector or env.get(MODE_ENV, "") or "oracle").strip().lower()
    if mode not in DETECTORS:
        raise DetectorError(f"detector= ska vara en av {', '.join(DETECTORS)}, inte {mode!r}")
    if mode == "oracle":
        if model:
            raise DetectorError("model= gäller bara med detector=model")
        return ORACLE
    raw = (model or env.get(MODEL_ENV, "")).strip()
    if not raw:
        raise DetectorError(
            f"detector=model kräver model=<sökväg till modellens .json> eller {MODEL_ENV}. "
            "Se tools/train/README.md."
        )
    manifest = Path(raw).expanduser().resolve()
    try:
        check_model(manifest)
    except ModelError as exc:
        raise DetectorError(f"detector=model: {exc} (manifest {manifest})") from exc
    return DetectorChoice("model", manifest)
