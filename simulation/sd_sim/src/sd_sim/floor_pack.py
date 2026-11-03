"""Map packs (``sd_safety.floor_pack``) built from a simulated world, with map errors.

The height floor on the drone counts from a map pack: ground height, tree height and
known power lines (spec "Titta närmare" §6.2). In the simulator the pack comes from the
world itself (``sd_sim.obstacles.Scene``), so the same floor code as on the real drone
flies in the simulator, and "the map is the truth" is where every test starts:

* the ground (``dtm_cm``) is the world's ground in the middle of each 1 m cell;
* the tree height (``chm_dm``) is, per cell, the highest point of any crown or trunk over
  that cell, above the cell's ground, rounded up to the decimetre;
* every power line in the world is in ``lines.geojson``; poles and wires are not trees,
  so they are not in ``chm_dm`` (a real map has them only by chance).

``MapErrors`` puts in the errors a real map has (spec §16.1): every tree a few metres
too low (``tree_bias_m``), a share of the trees missing (``missing_trees``), holes of
unknown cells (``holes``), an old or unknown scan (``scan_year``). A single tree the
map has wrong is a ``SimTree`` with its own ``map_height`` (0: not in the map).

The pack is written in the loader's format and read back with
``sd_safety.floor_pack.load_floor_pack``, so a pack this module writes is one the drone
reads. TN-3 (``tools/terrain/fetch_floor_map.py``) brings a shared writer to
``sd_safety.floor_pack``; ``_write`` here is the smallest that fits the loader until
then.

``python -m sd_sim.floor_pack --world sd_lake_forest --out DIR`` writes one by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from sd_safety.floor_pack import (
    CHM_UNKNOWN_DM,
    FLOOR_PACK_FILES,
    FLOOR_PACK_FORMAT,
    FloorPack,
    load_floor_pack,
)
from sd_safety.targets import Plane
from sd_schemas.charter import FLOOR_PACK_NAME_PATTERN
from sd_sim import paths
from sd_sim.obstacles import Scene, SimTree, for_world, frustum_top, placed_parts
from sd_sim.sitl import CMAC_HOME

__all__ = ["DEFAULT_SCAN_CELL_M", "MapErrors", "build", "grids", "main"]

DEFAULT_SCAN_CELL_M: Final = 2500  # Lantmäteriet's scan squares
ATTRIBUTION: Final = "Searchdrone simulator: {world}, ground and trees from the world itself."
INT16_MAX_DM: Final = int(np.iinfo(np.int16).max)


@dataclass(frozen=True)
class MapErrors:
    """What the map has wrong on purpose. ``tree_bias_m``: every tree that much too low
    (never below 0). ``missing_trees``: the share of the trees left out (drawn with
    ``seed``). ``holes``: circles ``(east, north, radius)`` of unknown cells (-1).
    ``scan_year``: the scan's year everywhere (0 = unknown; ``None`` = this year, a
    fresh scan)."""

    tree_bias_m: float = 0.0
    missing_trees: float = 0.0
    holes: tuple[tuple[float, float, float], ...] = ()
    scan_year: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.tree_bias_m <= 40.0:
            raise ValueError("tree_bias_m ska ligga mellan 0 och 40")
        if not 0.0 <= self.missing_trees <= 1.0:
            raise ValueError("missing_trees ska ligga mellan 0 och 1")
        if any(r <= 0 for _, _, r in self.holes):
            raise ValueError("ett hål har en radie större än 0")
        if self.scan_year is not None and not (
            self.scan_year == 0 or 1990 <= self.scan_year <= 2100
        ):
            raise ValueError("scan_year ska vara 0 (okänt) eller ett år")

    def map_height(self, tree: SimTree, rng: random.Random) -> float:
        """How tall the map says ``tree`` is (0: not in the map)."""
        missing = rng.random() < self.missing_trees  # drawn for every tree: a fixed sequence
        if tree.map_height is not None:
            return tree.map_height
        if missing:
            return 0.0
        return max(tree.height - self.tree_bias_m, 0.0)


@dataclass(frozen=True)
class Grids:
    """The pack's grids before they are written (row 0 north, 1 m cells, centred on
    home)."""

    dtm_cm: NDArray[np.int32]
    chm_dm: NDArray[np.int16]
    scan_year: NDArray[np.uint16]
    scan_cell_m: int
    lines: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])


def grids(
    scene: Scene,
    *,
    half_m: float | None = None,
    errors: MapErrors = MapErrors(),  # noqa: B008 - frozen, never changed
    scan_cell_m: int = DEFAULT_SCAN_CELL_M,
) -> Grids:
    """The pack's grids for the square of ``half_m`` around home (the world's own size
    by default)."""
    half = scene.half if half_m is None else half_m
    n = 2 * math.ceil(half)
    centres = np.arange(n) - n / 2 + 0.5  # cell centres, west to east
    east, north = np.meshgrid(centres, centres[::-1])
    ground = scene.ground.at(east, north)
    dtm_cm = np.round((scene.home_alt + ground) * 100).astype(np.int32)
    top = ground.copy()  # the highest point over each cell, world z
    rng = random.Random(errors.seed)  # noqa: S311 - a test's map error, not security
    for tree in scene.trees:
        h = errors.map_height(tree, rng)
        for x, y, z0, r0, z1, r1 in placed_parts(tree, h):
            r = max(r0, r1)
            c0, c1 = _cells(x - r, x + r, n)
            r_0, r_1 = _cells(-(y + r), -(y - r), n)  # rows count from the north
            if c0 > c1 or r_0 > r_1:
                continue
            ce, cn = east[r_0 : r_1 + 1, c0 : c1 + 1], north[r_0 : r_1 + 1, c0 : c1 + 1]
            # The distance from the axis to the nearest point of each cell.
            dx = np.maximum(np.abs(ce - x) - 0.5, 0.0)
            dy = np.maximum(np.abs(cn - y) - 0.5, 0.0)
            z = frustum_top(np.hypot(dx, dy), z0, r0, z1, r1)
            patch = top[r_0 : r_1 + 1, c0 : c1 + 1]
            np.fmax(patch, z, out=patch)
    chm = np.clip(np.ceil(np.round((top - ground) * 10, 6)), 0, INT16_MAX_DM).astype(np.int16)
    for hx, hy, hr in errors.holes:
        chm[np.hypot(east - hx, north - hy) <= hr] = CHM_UNKNOWN_DM
    year = datetime.now(UTC).year if errors.scan_year is None else errors.scan_year
    tiles = -(-n // scan_cell_m)
    scan = np.full((tiles, tiles), year, dtype=np.uint16)
    plane = Plane(*scene.origin)
    lines = []
    for line in scene.lines:
        coords = []
        for x, y in (line.a, line.b):
            lat, lon = plane.latlon(x, y)
            coords.append([round(lon, 8), round(lat, 8)])
        lines.append(
            {
                "type": "Feature",
                "properties": {"name": line.name, "source": "simulator"},
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        )
    return Grids(dtm_cm, chm, scan, scan_cell_m, lines)


def _cells(lo: float, hi: float, n: int) -> tuple[int, int]:
    """The columns (or rows, with north negated) whose cells meet ``[lo, hi]``."""
    return max(math.floor(lo + n / 2), 0), min(math.floor(hi + n / 2), n - 1)


def build(
    scene: Scene,
    folder: Path,
    *,
    name: str,
    half_m: float | None = None,
    errors: MapErrors = MapErrors(),  # noqa: B008 - frozen, never changed
    scan_cell_m: int = DEFAULT_SCAN_CELL_M,
) -> FloorPack:
    """Write the scene's pack to ``folder`` and return it as the drone loads it."""
    if not re.match(FLOOR_PACK_NAME_PATTERN, name):
        raise ValueError(f"paketets namn ska matcha {FLOOR_PACK_NAME_PATTERN}")
    g = grids(scene, half_m=half_m, errors=errors, scan_cell_m=scan_cell_m)
    about = {
        "world": scene.name,
        "tree_bias_m": errors.tree_bias_m,
        "missing_trees": errors.missing_trees,
        "holes": len(errors.holes),
    }
    _write(folder, name, scene, g, ATTRIBUTION.format(world=scene.name), about)
    return load_floor_pack(folder)


def _write(
    folder: Path,
    name: str,
    scene: Scene,
    g: Grids,
    attribution: str,
    about: dict[str, Any],
) -> None:
    """The files of ``sd_safety.floor_pack``'s format, each with its hash in
    ``floor_pack.toml``. A folder that is there already is replaced."""
    files: dict[str, bytes] = {
        "dtm_cm.npy": _npy(g.dtm_cm),
        "chm_dm.npy": _npy(g.chm_dm),
        "scan_year.npy": _npy(g.scan_year),
        "lines.geojson": (
            json.dumps({"type": "FeatureCollection", "features": g.lines}) + "\n"
        ).encode(),
    }
    rows, cols = g.dtm_cm.shape
    hashes = "".join(
        f'"{f}" = "{hashlib.sha256(files[f]).hexdigest()}"\n' for f in FLOOR_PACK_FILES
    )
    about_toml = "".join(f"{k} = {json.dumps(v)}\n" for k, v in about.items())
    toml = (
        f"format = {FLOOR_PACK_FORMAT}\n"
        f"name = {json.dumps(name)}\n"
        f"attribution = {json.dumps(attribution)}\n"
        f"\n[origin]\nlat = {scene.origin[0]!r}\nlon = {scene.origin[1]!r}\n"
        f"\n[grid]\nsize_m = [{cols}, {rows}]\nscan_cell_m = {g.scan_cell_m}\n"
        f"\n[about]\n{about_toml}"
        f"\n[sha256]\n{hashes}"
    )
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for file_name, data in files.items():
        (folder / file_name).write_bytes(data)
    (folder / "floor_pack.toml").write_text(toml, encoding="utf-8")


def _npy(grid: NDArray[Any]) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(grid), allow_pickle=False)
    return buf.getvalue()


# --- command line -----------------------------------------------------------------------


def _hole(text: str) -> tuple[float, float, float]:
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("ett hål skrivs east,north,radie")
    return parts[0], parts[1], parts[2]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sd_sim.floor_pack", description=__doc__)
    ap.add_argument("--world", default="sd_lake_forest")
    ap.add_argument("--home", default=CMAC_HOME, help="lat,lon,alt,heading (den platta världen)")
    ap.add_argument("--name", default="", help="paketets namn (förval: sim-<värld>)")
    ap.add_argument("--out", type=Path, required=True, help="mappen paketet skrivs till")
    ap.add_argument("--half-m", type=float, default=None)
    ap.add_argument("--tree-bias-m", type=float, default=0.0)
    ap.add_argument("--missing-trees", type=float, default=0.0)
    ap.add_argument("--hole", type=_hole, action="append", default=[])
    ap.add_argument("--scan-year", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    try:
        scene = for_world(args.world, args.home, repo=paths.REPO)
        errors = MapErrors(
            args.tree_bias_m, args.missing_trees, tuple(args.hole), args.scan_year, args.seed
        )
        name = args.name or "sim-" + args.world.removeprefix("sd_").replace("_", "-")
        pack = build(scene, args.out, name=name, half_m=args.half_m, errors=errors)
    except (ValueError, OSError) as exc:
        print(f"Kartpaketet kunde inte skrivas: {exc}")  # noqa: T201 - command-line output
        return 2
    rows, cols = pack.dtm_cm.shape
    print(f"Skrev {pack.name} ({cols} x {rows} m) till {args.out}: sha256 {pack.sha256}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
