"""PNG pictures of the fpv7 assembly, drawn by a small z-buffer rasteriser in numpy (no
display and no OpenGL needed): ``fpv7_views.png`` shows the drone from three angles and
``fpv7_closeup.png`` the centre of the frame up close.

Every part is tessellated, projected orthographically and filled triangle by triangle into
a depth buffer at twice the output resolution, shaded by the angle to a light fixed to the
camera, outlined where the part or the surface direction changes, and scaled down. The
propeller blades are drawn where the drone has them (``display``), otherwise the swept
discs as translucent rings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch

if TYPE_CHECKING:
    from assembly import Drone, Part

# (title, elevation, azimuth) in degrees: the eye's direction from the drone, azimuth from
# +x (forward) towards +y (left).
VIEWS = [
    ("Snett framifrån", 28.0, 40.0),
    ("Från höger sida (fram till höger)", 0.0, -90.0),
    ("Underifrån (fram uppåt)", -90.0, 0.0),
]
CLOSEUP = ("Ramens mitt, snett framifrån (utan kåpan)", 32.0, 35.0)
# The close-up's box (mm): the plates, the stack, the computer and the camera.
CLOSEUP_BOX = ((-80.0, 80.0), (-60.0, 60.0), (-45.0, 100.0))
SUPERSAMPLE = 2
BACKGROUND = np.array([0.97, 0.97, 0.96])
TRANSLUCENT_ALPHA = 0.3

GROUP_NAMES = {
    "frame": "ram (kolfiber, aluminium)",
    "fastener": "skruvar och muttrar",
    "motor": "motorer",
    "prop": "propellrar",
    "battery": "batteri",
    "electronics": "elektronik",
    "computer": "Raspberry Pi 5 med AI HAT+",
    "camera": "kamera",
    "printed": "utskrivna delar",
    "cover": "kåpor (utskrivna)",
    "cable": "kablar och antenner",
}


@dataclass(frozen=True)
class Mesh:
    name: str
    triangles: np.ndarray  # (n, 3, 3) in mm
    color: np.ndarray  # (3,)
    translucent: bool


def meshes(drone: Drone, colors: dict[str, tuple[float, float, float]]) -> list[Mesh]:
    shown: list[Part] = list(drone.parts)
    blades = bool(drone.display)
    if blades:
        shown = [pt for pt in shown if pt.group != "prop"] + list(drone.display)
    out = []
    for pt in shown:
        vertices, triangles = pt.solid.tessellate(0.3, 0.6)
        if not triangles:
            continue
        v = np.array([(p.x, p.y, p.z) for p in vertices])
        out.append(
            Mesh(
                pt.name,
                v[np.array(triangles)],
                np.array(colors[pt.group]),
                pt.group == "prop" and not blades,
            )
        )
    return out


def camera(elevation: float, azimuth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit vectors (right, up, towards the eye) of a camera at the given angles."""
    el, az = math.radians(elevation), math.radians(azimuth)
    eye = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    world_up = np.array([0.0, 0.0, 1.0]) if abs(eye[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    right = np.cross(world_up, eye)
    right /= np.linalg.norm(right)
    return right, np.cross(eye, right), eye


def rasterise(
    meshes_: list[Mesh],
    view: tuple[float, float],
    size: tuple[int, int],
    box: tuple[tuple[float, float], ...] | None = None,
) -> np.ndarray:
    """An RGB image (height, width, 3) in 0..1 of the meshes seen from ``view``
    (elevation, azimuth), fitted to ``size`` (width, height) pixels. With ``box`` (x, y
    and z ranges in mm) the picture is fitted to that box instead, cutting off what lies
    outside it."""
    right, up, eye = camera(*view)
    w, h = size[0] * SUPERSAMPLE, size[1] * SUPERSAMPLE
    light = -0.35 * right + 0.55 * up + 0.75 * eye
    light /= np.linalg.norm(light)

    projected = []
    for k, m in enumerate(meshes_):
        tri = m.triangles
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(normal, axis=1)
        keep = length > 1e-12
        tri, normal = tri[keep], normal[keep] / length[keep, None]
        shade = 0.38 + 0.62 * np.abs(normal @ light)
        uvz = np.stack([tri @ right, tri @ up, tri @ eye], axis=-1)
        projected.append((k, uvz, shade))

    if box is None:
        all_uv = np.concatenate([p[1].reshape(-1, 3) for p in projected])
    else:
        corners = np.array(np.meshgrid(*box)).reshape(3, -1).T
        all_uv = np.stack([corners @ right, corners @ up, corners @ eye], axis=-1)
    lo, hi = all_uv.min(axis=0), all_uv.max(axis=0)
    margin = 0.04
    scale = min(w * (1 - 2 * margin) / (hi[0] - lo[0]), h * (1 - 2 * margin) / (hi[1] - lo[1]))
    cu, cv = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2

    depth = np.full((h, w), -np.inf)
    ids = np.full((h, w), -1, dtype=np.int64)
    shading = np.zeros((h, w))
    glass_depth = np.full((h, w), -np.inf)
    glass_color = np.zeros((h, w, 3))
    for k, uvz, shade in projected:
        px = (uvz[:, :, 0] - cu) * scale + w / 2
        py = h / 2 - (uvz[:, :, 1] - cv) * scale
        translucent = meshes_[k].translucent
        for t in range(len(uvz)):
            _fill(
                px[t], py[t], uvz[t, :, 2], shade[t], k, translucent,
                depth, ids, shading, glass_depth, glass_color, meshes_[k].color,
            )  # fmt: skip

    colors = np.array([m.color for m in meshes_])
    img = np.where(
        (ids >= 0)[..., None], colors[np.maximum(ids, 0)] * shading[..., None], BACKGROUND
    )
    # Outlines: where the part changes, or the depth or the shading jumps.
    edge = np.zeros((h, w), dtype=bool)
    step = 2.5 / scale * SUPERSAMPLE  # mm of depth between neighbouring pixels that is a jump
    for axis in (0, 1):
        d_id = np.diff(ids, axis=axis) != 0
        finite = np.isfinite(depth)
        d_depth = np.abs(np.diff(np.where(finite, depth, -1e9), axis=axis)) > max(step, 1.5)
        d_shade = np.abs(np.diff(shading, axis=axis)) > 0.14
        jump = d_id | d_depth | d_shade
        if axis == 0:
            edge[1:] |= jump
        else:
            edge[:, 1:] |= jump
    img = np.where(edge[..., None], img * 0.45, img)
    # Translucent discs in front of whatever is behind them.
    glass = glass_depth > depth
    img = np.where(
        glass[..., None], (1 - TRANSLUCENT_ALPHA) * img + TRANSLUCENT_ALPHA * glass_color, img
    )
    # Scale down by averaging SUPERSAMPLE x SUPERSAMPLE blocks (antialiasing).
    s = SUPERSAMPLE
    return img.reshape(size[1], s, size[0], s, 3).mean(axis=(1, 3))


def _fill(
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    shade: float,
    k: int,
    translucent: bool,
    depth: np.ndarray,
    ids: np.ndarray,
    shading: np.ndarray,
    glass_depth: np.ndarray,
    glass_color: np.ndarray,
    color: np.ndarray,
) -> None:
    """Fill one projected triangle into the buffers, keeping the nearest surface."""
    h, w = depth.shape
    x0, x1 = max(math.floor(px.min()), 0), min(math.ceil(px.max()), w - 1)
    y0, y1 = max(math.floor(py.min()), 0), min(math.ceil(py.max()), h - 1)
    if x0 > x1 or y0 > y1:
        return
    (ax, bx, cx), (ay, by, cy) = px, py
    area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    if abs(area) < 1e-12:
        return
    xs = np.arange(x0, x1 + 1) + 0.5
    ys = np.arange(y0, y1 + 1)[:, None] + 0.5
    w0 = ((bx - xs) * (cy - ys) - (cx - xs) * (by - ys)) / area
    w1 = ((cx - xs) * (ay - ys) - (ax - xs) * (cy - ys)) / area
    w2 = 1 - w0 - w1
    inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
    if not inside.any():
        return
    z = w0 * pz[0] + w1 * pz[1] + w2 * pz[2]
    if translucent:
        region = glass_depth[y0 : y1 + 1, x0 : x1 + 1]
        nearer = inside & (z > region)
        region[nearer] = z[nearer]
        glass_color[y0 : y1 + 1, x0 : x1 + 1][nearer] = color * (0.75 + 0.25 * shade)
        return
    region = depth[y0 : y1 + 1, x0 : x1 + 1]
    nearer = inside & (z > region)
    region[nearer] = z[nearer]
    ids[y0 : y1 + 1, x0 : x1 + 1][nearer] = k
    shading[y0 : y1 + 1, x0 : x1 + 1][nearer] = shade


def _legend(ax: Axes, colors: dict[str, tuple[float, float, float]], groups: set[str]) -> None:
    handles = [
        Patch(facecolor=colors[g], edgecolor="0.3", label=GROUP_NAMES.get(g, g))
        for g in colors
        if g in groups
    ]
    ax.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))  # fmt: skip


def views(drone: Drone, colors: dict[str, tuple[float, float, float]], path: Path) -> None:
    """Three views of the whole drone side by side in ``path``, and a close-up of the
    centre without the canopy in ``fpv7_closeup.png`` next to it."""
    ms = meshes(drone, colors)
    groups = {pt.group for pt in drone.parts}
    fig = plt.figure(figsize=(19, 7.2), dpi=100)
    for n, (title, elev, azim) in enumerate(VIEWS, start=1):
        ax = fig.add_subplot(1, 3, n)
        ax.imshow(rasterise(ms, (elev, azim), (620, 560)), interpolation="lanczos")
        ax.set_title(title)
        ax.set_axis_off()
    _legend(fig.add_axes((0, 0, 1, 0.06), frameon=False), colors, groups)
    fig.suptitle("Searchdrone fpv7 (mm; kablar ungefärliga)")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.07, wspace=0.03)
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)

    no_cover = [m for m in ms if m.name != "canopy"]
    title, elev, azim = CLOSEUP
    fig = plt.figure(figsize=(14, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(rasterise(no_cover, (elev, azim), (1300, 900), CLOSEUP_BOX), interpolation="lanczos")
    ax.set_title(title)
    ax.set_axis_off()
    _legend(fig.add_axes((0, 0, 1, 0.05), frameon=False), colors, groups)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)
    fig.savefig(path.with_name("fpv7_closeup.png"), metadata={"Software": None})
    plt.close(fig)
