"""Build ``aerial-frame.jpg``, the made-up nadir camera frame the panel tests stream as video.

Everything is drawn from seeded noise: a meadow, a gravel path, tree crowns with shadows and
one small person in a red jacket at the forest edge. No photo or map data goes in.

    uv run python software/panel/e2e/fixtures/make_aerial_frame.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
OUT = HERE / "aerial-frame.jpg"
W, H = 1280, 720
RNG = np.random.default_rng(20261018)


def fractal(octaves: int = 6, base: int = 4) -> np.ndarray:
    """Value noise in 0..1, summed over octaves with bilinear upsampling."""
    acc = np.zeros((H, W))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = base * 2**o
        grid = Image.fromarray((RNG.random((n * H // W + 2, n + 2)) * 255).astype(np.uint8))
        acc += amp * np.asarray(grid.resize((W, H), Image.Resampling.BICUBIC), dtype=float) / 255
        total += amp
        amp *= 0.55
    return acc / total


def blend(img: np.ndarray, mask: np.ndarray, colour: np.ndarray) -> None:
    img[:] = img * (1 - mask[..., None]) + colour * mask[..., None]


def soft(mask: np.ndarray, radius: float) -> np.ndarray:
    im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(radius)), dtype=float) / 255


def main() -> None:
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    tone, fine = fractal(), fractal(octaves=4, base=64)
    dry = np.array([150, 150, 92.0])
    lush = np.array([78, 104, 52.0])
    img = lush + (dry - lush) * np.clip(tone * 1.4 - 0.35, 0, 1)[..., None]
    img *= (0.86 + 0.28 * fine)[..., None]
    blades = soft(RNG.random((H, W)), 0.7)
    img *= (0.9 + 0.2 * blades)[..., None]

    # A gravel path crossing the frame on a gentle curve, with two darker wheel ruts.
    centre = 470 - 0.28 * xx + 40 * np.sin(xx / 260)
    d = np.abs(yy - centre)
    blend(img, soft((d < 15).astype(float), 3) * (0.75 + 0.25 * fine), np.array([148, 128, 98.0]))
    for off in (-6, 6):
        blend(
            img,
            soft((np.abs(yy - centre - off) < 1.6).astype(float), 1.2) * 0.35,
            np.array([96, 82, 62.0]),
        )

    # Forest edge along the top, then scattered single trees.
    crowns: list[tuple[float, float, float]] = []
    for x in np.arange(-20, W + 40, 34):
        for _ in range(3):
            crowns.append(
                (x + RNG.normal(0, 14), RNG.uniform(-30, 150) - 0.04 * x, RNG.uniform(24, 46))
            )
    for _ in range(26):
        x, y = RNG.uniform(0, W), RNG.uniform(220, H)
        if abs(y - (470 - 0.28 * x + 40 * np.sin(x / 260))) > 40:
            crowns.append((x, y, RNG.uniform(12, 26)))

    leaf = soft(RNG.random((H, W)), 1.6)
    leaf = (leaf - leaf.mean()) / leaf.std()
    blobs = [
        (x + RNG.normal(0, 0.35 * r), y + RNG.normal(0, 0.35 * r), r * RNG.uniform(0.45, 0.75))
        for x, y, r in crowns
        for _ in range(RNG.integers(4, 8))
    ]
    shadow = np.zeros((H, W))
    for x, y, r in blobs:
        shadow = np.maximum(
            shadow, ((xx - x - 0.5 * r) ** 2 + (yy - y - 0.4 * r) ** 2 < r**2) * 1.0
        )
    img *= (1 - 0.5 * soft(shadow, 4))[..., None]

    for x, y, r in sorted(blobs, key=lambda c: c[1]):
        x0, x1 = int(max(0, x - r - 3)), int(min(W, x + r + 4))
        y0, y1 = int(max(0, y - r - 3)), int(min(H, y + r + 4))
        if x0 >= x1 or y0 >= y1:
            continue
        sx, sy = xx[y0:y1, x0:x1] - x, yy[y0:y1, x0:x1] - y
        dist = np.sqrt(sx**2 + sy**2) / r
        lf = leaf[y0:y1, x0:x1]
        inside = np.clip((1.0 + 0.12 * lf - dist) * 5, 0, 1)
        light = np.clip(0.6 + 0.35 * (-(sx + sy) / (1.4 * r)) + 0.18 * lf, 0.2, 1.15)
        hue = np.array([38, 60, 32.0]) * RNG.uniform(0.85, 1.12) + RNG.normal(0, 3, 3)
        colour = hue * light[..., None]
        patch = img[y0:y1, x0:x1]
        patch[:] = patch * (1 - inside[..., None]) + colour * inside[..., None]

    # One person, seen from above, about 0.5 x 0.3 m at this height: a red jacket, a pale head.
    px, py = 760.0, 205.0
    person = ((xx - px) / 7.5) ** 2 + ((yy - py) / 4.5) ** 2 < 1
    blend(
        img,
        soft(((xx - px - 4) / 8) ** 2 + ((yy - py - 5) / 5) ** 2 < 1, 2) * 0.5,
        np.array([40, 46, 30.0]),
    )
    blend(img, soft(person.astype(float), 0.8), np.array([176, 38, 36.0]))
    blend(
        img,
        soft((((xx - px - 8) / 3) ** 2 + ((yy - py) / 3) ** 2 < 1) * 1.0, 0.6),
        np.array([196, 170, 140.0]),
    )

    vignette = 1 - 0.18 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2) * 2
    img *= vignette[..., None]
    img += RNG.normal(0, 3.0, img.shape)
    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.5)
    )
    out.save(OUT, quality=82, optimize=True, progressive=False)
    sys.stdout.write(f"{OUT.name}: {OUT.stat().st_size} bytes\n")


if __name__ == "__main__":
    main()
