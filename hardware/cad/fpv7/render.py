"""A PNG of the fpv7 assembly from three angles, drawn with matplotlib (no display needed)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

if TYPE_CHECKING:
    from assembly import Drone

# (title, elevation, azimuth) in degrees. Azimuth 0 looks along -x (from the front).
VIEWS = [
    ("Snett framifrån", 25.0, 35.0),
    ("Från sidan (fram till höger)", 0.0, -90.0),
    ("Underifrån (fram uppåt)", -90.0, 0.0),
]


def views(drone: Drone, colors: dict[str, tuple[float, float, float]], path: Path) -> None:
    meshes = []
    for pt in drone.parts:
        vertices, triangles = pt.solid.tessellate(0.5, 0.3)
        pts = [(v.x, v.y, v.z) for v in vertices]
        faces = [[pts[i] for i in tri] for tri in triangles]
        alpha = 0.35 if pt.group == "prop" else 1.0
        meshes.append((faces, colors[pt.group], alpha))
    lo = [min(p[i] for faces, _, _ in meshes for f in faces for p in f) for i in range(3)]
    hi = [max(p[i] for faces, _, _ in meshes for f in faces for p in f) for i in range(3)]

    fig = plt.figure(figsize=(18, 6.5), dpi=110)
    for n, (title, elev, azim) in enumerate(VIEWS, start=1):
        ax = fig.add_subplot(1, 3, n, projection="3d")
        for faces, color, alpha in meshes:
            ax.add_collection3d(
                Poly3DCollection(faces, facecolors=color, linewidths=0, alpha=alpha, shade=True)
            )
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]), zoom=1.15)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle("Searchdrone fpv7, grundmodell (mm, propellrarna som genomskinliga skivor)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
