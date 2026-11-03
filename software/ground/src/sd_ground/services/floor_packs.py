"""The height-floor map packs the ground station holds, in ``<data_dir>/floor/<id>/``
(``sd_safety.floor_pack``; written by ``tools/terrain/fetch_floor_map.py`` or the
simulator): the list the pilot picks from on the charter card, and the check of that pick.

A pack counts only whole: every file must match its hash (:func:`load`, the same loader
the drone uses). The pilot's pick names the hash the list showed, and the ground refuses
a pack it cannot read or whose hash has changed since: a charter never names a pack the
ground has not checked, so a closer look never comes down by a map nobody chose.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Final

from sd_ground.logging import get_logger
from sd_ground.services.missions import FlowError
from sd_safety.floor_pack import FLOOR_PACK_FILES, FloorPack, FloorPackError, load_floor_pack
from sd_schemas.api import FloorPackChoice, FloorPackInfo
from sd_schemas.charter import FLOOR_PACK_NAME_PATTERN

if TYPE_CHECKING:
    from sd_ground.services.missions import MissionFlow

__all__ = ["FLOOR_PACKS_MAX", "choose", "listing", "load", "root"]

log = get_logger(__name__)

FLOOR_PACKS_MAX: Final = 64  # in the list, by name
_ATTRIBUTION_MAX: Final = 500


def root(flow: MissionFlow) -> Path:
    return flow.data_root() / "floor"


def load(folder: Path) -> FloorPack:
    """The pack in ``folder``, checked whole. Kept in memory while its files stay the
    same (the hash of ``floor_pack.toml`` and every file's size and time): a pack written
    anew is read anew."""
    try:
        meta = (folder / "floor_pack.toml").read_bytes()
        stamp = tuple(
            (s.st_size, s.st_mtime_ns) for s in ((folder / n).stat() for n in FLOOR_PACK_FILES)
        )
    except OSError as exc:
        raise FloorPackError(f"{folder.name}: not a readable floor pack ({exc})") from exc
    return _load(folder, hashlib.sha256(meta).hexdigest(), stamp)


@lru_cache(maxsize=8)
def _load(folder: Path, sha256: str, stamp: tuple[tuple[int, int], ...]) -> FloorPack:
    del stamp  # only the cache's key
    pack = load_floor_pack(folder)
    if pack.sha256 != sha256:
        raise FloorPackError(f"{folder.name}: changed while it was read")
    return pack


def listing(flow: MissionFlow) -> list[FloorPackInfo]:
    """Every whole pack under ``<data_dir>/floor/``, by name. A folder that is not a
    pack's name (a write in progress is ``.<name>.writing``) or not a whole pack is left
    out and logged."""
    base = root(flow)
    if not base.is_dir():
        return []
    out: list[FloorPackInfo] = []
    for folder in sorted(base.iterdir()):
        if len(out) >= FLOOR_PACKS_MAX:
            break
        if not folder.is_dir() or not re.match(FLOOR_PACK_NAME_PATTERN, folder.name):
            continue
        try:
            pack = load(folder)
        except FloorPackError as exc:
            log.warning("floor_pack.unreadable", pack=folder.name, error=str(exc))
            continue
        rows, cols = pack.dtm_cm.shape
        out.append(
            FloorPackInfo(
                floor_pack_id=folder.name,
                floor_pack_sha256=pack.sha256,
                attribution=pack.attribution[:_ATTRIBUTION_MAX],
                origin=pack.origin,
                size_east_m=cols,
                size_north_m=rows,
            )
        )
    return out


def choose(flow: MissionFlow, choice: FloorPackChoice) -> tuple[str | None, str | None]:
    """The pilot's pick as the charter's ``(floor_pack_id, floor_pack_sha256)``, refused
    (409) when the ground does not have that pack whole or its hash is not the one the
    pilot saw. "Ingen höjdkarta" is ``(None, None)``."""
    name, sha = choice.floor_pack_id, choice.floor_pack_sha256
    if name is None or sha is None:
        return None, None
    try:
        pack = load(root(flow) / name)
    except FloorPackError as exc:
        log.warning("floor_pack.refused", pack=name, error=str(exc))
        raise FlowError(
            409,
            "FLOOR_PACK_UNAVAILABLE",
            f"Höjdkartan {name} saknas på markstationen eller är skadad. Välj en annan, "
            "eller Ingen höjdkarta.",
        ) from exc
    if pack.sha256 != sha:
        raise FlowError(
            409,
            "FLOOR_PACK_CHANGED",
            f"Höjdkartan {name} har ändrats sedan listan visades. Välj den igen.",
        )
    return name, sha
