"""Manufacturer 3D models for the detailed STEP export (`just cad-detailed`).

    uv run --group cad python hardware/cad/fpv7/vendor.py

downloads every model in vendor.yaml into build/cad/vendor/, checks the archive's SHA-256
and extracts only the named STEP file. A model that is already there with the right
checksum is not downloaded again. The files are never committed: some licences do not
allow it, and the Raspberry Pi 5 model alone is 78 MB.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import cadquery as cq
import yaml

HERE = Path(__file__).resolve().parent
MODELS = HERE / "vendor.yaml"
VENDOR_DIR = HERE.parents[2] / "build" / "cad" / "vendor"
# The download servers refuse Python's default user agent.
USER_AGENT = "searchdrone-cad/0 (+https://github.com/YXHFR/Searchdrone)"


SCHEMA_VERSION = 1
# A response: a readable stream that also says which URL it finally came from.
Opener = Callable[[urllib.request.Request], Any]


def models(path: Path = MODELS) -> list[dict[str, Any]]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path.name}: schema_version must be {SCHEMA_VERSION}")
    return doc["models"]


def _urlopen(req: urllib.request.Request) -> Any:
    return urllib.request.urlopen(req, timeout=120)  # noqa: S310 - fetch() allows https only


def _copy_capped(src: IO[bytes], dst: IO[bytes], limit: int, what: str) -> None:
    """Copy at most ``limit`` bytes; a longer stream is an error."""
    copied = 0
    while chunk := src.read(1 << 20):
        copied += len(chunk)
        if copied > limit:
            raise ValueError(f"{what}: more than {limit} bytes")
        dst.write(chunk)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def step_path(model: dict[str, Any], folder: Path = VENDOR_DIR) -> Path:
    return folder / f"{model['name']}.step"


def fetch(model: dict[str, Any], folder: Path = VENDOR_DIR, opener: Opener = _urlopen) -> Path:
    """Download and check one model; return the path of its STEP file.

    Only https is allowed, also after redirects. The archive must match the pinned
    SHA-256 and stay under ``max_download_bytes``; the STEP file is the one named member,
    at most ``max_bytes``. A matching archive already on disk is used as it is.
    """
    name = model["name"]
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / f"{name}.zip"
    if not archive.exists() or sha256(archive) != model["sha256"]:
        if not model["url"].startswith("https://"):
            raise ValueError(f"{name}: only https downloads are allowed")
        tmp = archive.with_suffix(".part")
        req = urllib.request.Request(model["url"], headers={"User-Agent": USER_AGENT})  # noqa: S310 - https checked above
        try:
            with opener(req) as resp, tmp.open("wb") as out:
                if not resp.geturl().startswith("https://"):
                    raise ValueError(f"{name}: redirected away from https")
                _copy_capped(resp, out, model["max_download_bytes"], name)
            tmp.replace(archive)
        finally:
            tmp.unlink(missing_ok=True)
    digest = sha256(archive)
    if digest != model["sha256"]:
        archive.unlink()
        raise ValueError(f"{name}: SHA-256 {digest} does not match vendor.yaml")
    target = step_path(model, folder)
    with zipfile.ZipFile(archive) as zf:
        info = zf.getinfo(model["member"])
        if info.file_size > model["max_bytes"]:
            raise ValueError(f"{name}: {info.file_size} bytes is over max_bytes")
        # Only the named member is read, written under our own file name.
        with zf.open(info) as src, target.open("wb") as dst:
            _copy_capped(src, dst, model["max_bytes"], name)
    return target


def placed(model: dict[str, Any], anchor: cq.Shape, folder: Path = VENDOR_DIR) -> cq.Shape:
    """The model moved onto the envelope part it replaces (see vendor.yaml)."""
    shape = cq.importers.importStep(str(step_path(model, folder))).val()
    box = anchor.BoundingBox()
    shape = shape.translate(cq.Vector(-model["centre_x"], -model["centre_y"], 0))
    if model["rotate_z"]:
        shape = shape.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), model["rotate_z"])
    return shape.translate(cq.Vector(box.center.x, box.center.y, box.zmin + model["z_offset"]))


def available(folder: Path = VENDOR_DIR) -> bool:
    """Every model's STEP file is there (``fetch`` has run)."""
    return all(step_path(m, folder).exists() for m in models())


def main() -> int:
    for model in models():
        path = fetch(model)
        sys.stdout.write(f"{model['name']}: {path} ({model['license']})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
