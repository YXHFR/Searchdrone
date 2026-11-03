"""Other clothes for the person models: copies of a model whose garments are recoloured.

A solid ``tint`` paints the whole model one colour, face and hands too, with no folds or
seams, which no real person looks like from the air. A **look** keeps the model's own
textures and shading and changes only the clothes: the top (shirt, jacket) and the
bottom (trousers), each to a colour from ``TOPS`` and ``BOTTOMS``. The recolouring keeps
each pixel's brightness relative to its garment, so creases, seams and shadows stay.

``make(model, top, bottom)`` writes ``build/simfarm/assets/looks/<model>-<top>-<bottom>/``
once: the mesh as a hard link (or a copy) and the textures, the recoloured ones as new
files and the rest as links to the fetched model's. Gazebo finds it as
``model://<look>`` (``looks_dir`` on ``GZ_SIM_RESOURCE_PATH``). Nothing is committed or
uploaded: the looks are made from the fetched models (CC0 / CC BY 4.0, credited as the
model they come from).

Which pixels are a garment is told by colour, per texture (``GARMENTS``): the models'
garments are a white T-shirt and blue jeans, a blue top and jeans, or a hi-vis jacket.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sd_sim import paths
from sd_sim.farm import people, scenario

# Rec. 709 brightness weights of red, green and blue.
LUMA = np.array(
    [
        0.2126,
        0.7152,
        0.0722,
    ],
    np.float32,
)

Mask = Callable[[NDArray[np.float32]], NDArray[np.float32]]  # rgb 0-1 -> weight 0-1

TOPS: dict[str, tuple[float, float, float]] = dict(scenario.CLOTHES)
BOTTOMS: dict[str, tuple[float, float, float] | None] = {
    "jeans": None,  # the model's own
    "black": (0.07, 0.07, 0.08),
    "grey": (0.38, 0.38, 0.4),
    "khaki": (0.55, 0.49, 0.35),
    "olive": (0.3, 0.32, 0.18),
    "brown": (0.36, 0.25, 0.16),
    "navy": (0.1, 0.12, 0.28),
}


def _hsv(rgb: NDArray[np.float32]) -> tuple[NDArray[np.float32], ...]:
    mx, mn = rgb.max(axis=2), rgb.min(axis=2)
    v = mx
    s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    d = np.maximum(mx - mn, 1e-6)
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4))
    return (h * 60.0).astype(np.float32), s.astype(np.float32), v.astype(np.float32)


Region = Callable[
    [NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]], NDArray[np.bool_]
]


def blue_in(region: Region) -> Mask:
    """Blue cloth where ``region(x, y, saturation)`` holds (x, y: the place in the
    texture, 0-1)."""

    def mask(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
        h, s, v = _hsv(rgb)
        y, x = np.mgrid[0 : rgb.shape[0], 0 : rgb.shape[1]].astype(np.float32)
        inside = region(x / rgb.shape[1], y / rgb.shape[0], s)
        return ((h > 185) & (h < 255) & (s > 0.12) & (v > 0.08) & inside).astype(np.float32)

    return mask


# The casual suits' textures: the shirts fill the top 40 % and the sleeves (a saturated
# blue) the right edge below them; the trousers are the rest.
def _suit_top(
    x: NDArray[np.float32], y: NDArray[np.float32], s: NDArray[np.float32]
) -> NDArray[np.bool_]:
    return (y < 0.41) | ((x > 0.7) & (s > 0.6))


_SUIT_TOP = blue_in(_suit_top)
_SUIT_BOTTOM = blue_in(lambda x, y, s: ~_suit_top(x, y, s))


def hi_vis(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    h, s, v = _hsv(rgb)
    return ((h > 45) & (h < 95) & (s > 0.35) & (v > 0.3)).astype(np.float32)


def not_background(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """A garment texture's own pixels (its UV islands), not the black around them."""
    return (rgb.max(axis=2) > 0.04).astype(np.float32)


# model -> texture file -> [(part, mask)]; a part is "top" or "bottom".
GARMENTS: dict[str, dict[str, list[tuple[str, Mask]]]] = {
    "sd_person_standing": {
        "materials/textures/tshirt02_texture.png": [("top", not_background)],
        "materials/textures/jeans_basic_diffuse.png": [("bottom", not_background)],
    },
    "sd_person_walking": {
        "materials/textures/tshirt02_texture.png": [("top", not_background)],
        "materials/textures/jeans_basic_diffuse.png": [("bottom", not_background)],
    },
    "sd_person_casual_female": {
        "materials/textures/female_casualsuit01_diffuse.png": [
            ("top", _SUIT_TOP),
            ("bottom", _SUIT_BOTTOM),
        ],
        "materials/textures/female_casualsuit02_diffuse.png": [
            ("top", _SUIT_TOP),
            ("bottom", _SUIT_BOTTOM),
        ],
    },
    "sd_person_rescue_randy": {
        "materials/textures/rescue_randy_albedo.png": [("top", hi_vis)],
    },
    "sd_person_sitting": {
        "materials/textures/RescueRandy_Diffuse.jpg": [("top", hi_vis)],
    },
}


def looks_dir(repo: Path = paths.REPO) -> Path:
    return repo / "build" / "simfarm" / "assets" / "looks"


def name(model: str, top: str, bottom: str) -> str:
    if model not in people.BY_ID or top not in TOPS or bottom not in BOTTOMS:
        raise ValueError(f"okänd look: {model}, {top}, {bottom}")
    return f"{model}-{top}-{bottom}"


def recolour(
    rgb: NDArray[np.float32], weight: NDArray[np.float32], colour: tuple[float, float, float]
) -> NDArray[np.float32]:
    """``rgb`` (0-1) with the weighted pixels turned to ``colour``: each keeps its
    brightness relative to the garment's typical brightness, so folds and seams stay."""
    lum = rgb @ LUMA
    inside = weight > 0.5
    ref = float(np.median(lum[inside])) if inside.any() else 0.5
    shade = np.clip(lum / max(ref, 1e-3), 0.0, 1.8)[..., None]
    target = np.clip(np.array(colour, np.float32)[None, None, :] * shade, 0.0, 1.0)
    w = weight[..., None]
    return (rgb * (1 - w) + target * w).astype(np.float32)


def _source(model: str, repo: Path, env: Mapping[str, str]) -> Path:
    for d in people.search_dirs(env, repo):
        if (d / model).is_dir():
            return d / model
    raise people.PeopleError(f"{model} saknas. Kör: just sim-farm-setup")


def _link(src: Path, dest: Path) -> None:
    try:
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)


def make(
    model: str,
    top: str,
    bottom: str,
    repo: Path = paths.REPO,
    env: Mapping[str, str] | None = None,
) -> str:
    """The look's folder, made once; returns its name (``model://<name>``)."""
    from PIL import Image  # the dev group's Pillow, as render_stills

    look = name(model, top, bottom)
    dest = looks_dir(repo) / look
    if (dest / ".done").is_file():
        return look
    src = _source(model, repo, os.environ if env is None else env)
    tmp = dest.with_name(f".{look}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    colours = {"top": TOPS[top], "bottom": BOTTOMS[bottom]}
    garments = GARMENTS.get(model, {})
    for path in sorted(p for p in src.rglob("*") if p.is_file() and p.name != ".sha256"):
        rel = path.relative_to(src)
        out = tmp / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        parts = [(part, m) for part, m in garments.get(rel.as_posix(), []) if colours[part]]
        if not parts:
            if path.suffix == ".dae":
                _link(path, out)
            else:
                out.symlink_to(path.resolve())
            continue
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"), np.float32) / 255.0
        for part, mask in parts:
            colour = colours[part]
            assert colour is not None  # noqa: S101 - filtered above
            rgb = recolour(rgb, mask(rgb), colour)
        Image.fromarray((rgb * 255 + 0.5).astype(np.uint8)).save(out)
    (tmp / ".done").write_text(f"{model} {top} {bottom}\n", encoding="utf-8")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.replace(dest)
    return look
