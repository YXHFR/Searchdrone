"""The lake-forest world's ground: a generated meadow texture on a mesh that repeats it.

A flat one-colour ground gives camera pictures with nothing in them but one green, which
the farm's quality check rightly refuses. The texture is drawn here from noise (grass, dry
streaks, bare soil), so it is our own work under the project's licence and needs no photo
or map. It repeats every ``PERIOD_M`` (the noise wraps at the edges, so there is no seam)
and is written once per ``VERSION`` under ``build/sim/worlds/meadow/``.

Soil is yellow-brown, never red: the colour detector of the red-block scenario must see
nothing on the ground.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sd_sim import png

VERSION = 1
PERIOD_M = 60.0
TEXTURE_PX = 2048  # 3 cm per pixel
SEED = 20261019

Grid = NDArray[np.floating[Any]]

GRASS = np.array([92.0, 132.0, 58.0])
MEADOW = np.array([146.0, 152.0, 84.0])
DRY = np.array([182.0, 168.0, 96.0])
SOIL = np.array([118.0, 104.0, 72.0])


def _hash01(col: NDArray[np.int64], row: NDArray[np.int64], seed: int) -> Grid:
    h = (col.astype(np.uint32) * np.uint32(0x9E3779B1)) ^ (
        row.astype(np.uint32) * np.uint32(0x85EBCA77)
    )
    h ^= np.uint32(seed * 0xC2B2AE3D & 0xFFFFFFFF)
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x2C1B3C6D)
    h ^= h >> np.uint32(13)
    return (h & np.uint32(0xFFFFFF)).astype(np.float32) / np.float32(0x1000000)


def periodic_noise(px: int, cells: int, seed: int, warp: tuple[Grid, Grid] | None = None) -> Grid:
    """Smooth noise in [0, 1) on a ``px`` x ``px`` picture with ``cells`` features across,
    wrapping at the edges. ``warp`` (two grids, in cells) bends the lattice so the features
    do not line up with the picture's axes."""
    g = ((np.arange(px) + 0.5) * cells / px).astype(np.float32)
    gy, gx = np.meshgrid(g, g, indexing="ij")
    if warp is not None:
        gx, gy = gx + warp[0], gy + warp[1]
    col, row = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    fx, fy = gx - col, gy - row
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    x0, y0 = col % cells, row % cells
    x1, y1 = (x0 + 1) % cells, (y0 + 1) % cells
    lattice = np.arange(cells)
    table = _hash01(*np.meshgrid(lattice, lattice, indexing="xy"), seed)  # [y, x]
    a, b = table[y0, x0], table[y0, x1]
    c, d = table[y1, x0], table[y1, x1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _cells(scale_m: float) -> int:
    return max(1, round(PERIOD_M / scale_m))


@functools.cache
def texture(px: int = TEXTURE_PX, seed: int = SEED) -> NDArray[np.uint8]:
    """The meadow, sRGB, row 0 at the top."""
    # A slow bend of the whole picture, wrapping like everything else (in metres).
    bend = 6.0 * (periodic_noise(px, _cells(15.0), seed + 90) - 0.5)
    bend2 = 6.0 * (periodic_noise(px, _cells(15.0), seed + 91) - 0.5)

    def noise(scale_m: float, salt: int, octaves: int = 1) -> Grid:
        total, amp, norm = np.zeros((px, px), np.float32), 1.0, 0.0
        for i in range(octaves):
            cells = _cells(scale_m / 2**i)
            warp = (bend * cells / PERIOD_M, bend2 * cells / PERIOD_M)
            total += amp * periodic_noise(px, cells, seed + salt + 101 * i, warp)
            norm += amp
            amp *= 0.5
        return total / norm

    patches = noise(20.0, 1, 3)
    base = GRASS + (MEADOW - GRASS) * np.clip(2.2 * patches - 0.7, 0, 1)[..., None]
    dry = np.clip(2.5 * noise(4.0, 3, 3) - 1.25, 0, 1)
    base = base * (1 - 0.5 * dry[..., None]) + DRY * 0.5 * dry[..., None]
    soil = np.clip(5.0 * (noise(5.0, 4, 3) - 0.66), 0, 1)
    base = base * (1 - soil[..., None]) + SOIL * soil[..., None]
    grain = noise(0.6, 5, 2) - 0.5
    fine = periodic_noise(px, _cells(0.12), seed + 6) - 0.5
    shade = 1.0 + 0.25 * (patches - 0.5) + 0.30 * grain + 0.16 * fine
    rgb = base * shade[..., None]
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


@functools.cache
def _texture_png() -> bytes:
    return png.encode(texture())


def ground_obj(size_m: float, mtl: str) -> str:
    """One square ``size_m`` across at z = 0, the texture repeated every ``PERIOD_M``."""
    h = size_m / 2
    r = size_m / PERIOD_M
    return (
        "\n".join(
            [
                f"mtllib {mtl}",
                "usemtl meadow",
                f"v {-h:g} {-h:g} 0",
                f"v {h:g} {-h:g} 0",
                f"v {h:g} {h:g} 0",
                f"v {-h:g} {h:g} 0",
                "vt 0 0",
                f"vt {r:g} 0",
                f"vt {r:g} {r:g}",
                f"vt 0 {r:g}",
                "vn 0 0 1",
                "f 1/1/1 2/2/1 3/3/1",
                "f 1/1/1 3/3/1 4/4/1",
            ]
        )
        + "\n"
    )


MTL = """newmtl meadow
Ka 0.600 0.600 0.600
Kd 1.000 1.000 1.000
Ks 0.020 0.020 0.020
Ns 10
map_Kd {texture}
"""


def write(out: Path, size_m: float) -> Path:
    """The ground mesh, its material and texture under ``out``; returns the mesh's path.
    Unchanged files are not written again."""
    out.mkdir(parents=True, exist_ok=True)
    tex = out / f"meadow_v{VERSION}.png"
    if not tex.is_file():
        tmp = tex.with_suffix(".tmp")
        tmp.write_bytes(_texture_png())
        tmp.replace(tex)
    mtl = out / "ground.mtl"
    obj = out / f"ground_{size_m:g}.obj"
    for path, text in ((mtl, MTL.format(texture=tex.name)), (obj, ground_obj(size_m, mtl.name))):
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
    return obj
