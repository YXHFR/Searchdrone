"""``ColorScorer``: the code's own look at a candidate's crops against the search profile.

For each crop: the box is found in the crop, the background colours are estimated by
k-means (k = 3, CIELAB) on a frame 15 % of the box's size around it, and the box's pixels
further than CIEDE2000 15 from every background colour count as the person. Fewer than
``MIN_PERSON_PX`` such pixels and the crop says nothing. The share of each profile colour
uses the tolerances of ``sd_schemas.colors.COLOR_TABLE`` (a ``light`` or ``dark`` shade
moves the reference L* by ``SHADE_L_SHIFT``). With two or more usable crops the shares
are their mean; with one, that crop's (crops are taken largest first).

Verdicts per field (the code never says more than the pixels can):

* ``match``: the main colour's share is at least 0.15 (outer, upper, full body, backpack,
  bag) or 0.08 (everything else).
* ``conflict``: only outer, upper and full body with certainty ``certain`` or ``likely``,
  when the main colour has under 0.03 and another saturated colour (C* > 30) has at least
  0.40. An exclusion ("inte röd") naming a colour with a share of at least 0.40 gives
  ``conflict`` on the upper garments.
* ``not_visible`` otherwise. Lower garments, headwear, footwear, hair and items are only
  ever ``match`` or ``not_visible``; the likely state is never judged by the code.
"""

from __future__ import annotations

import io
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from PIL import Image

from sd_schemas.colors import COLOR_TABLE, COLOR_TABLE_VERSION, SHADE_L_SHIFT, ColorName
from sd_schemas.link import BBox
from sd_schemas.search import CandidateAssessment, FieldVerdict, SearchProfile
from sd_search.score import match_score

__all__ = [
    "CROPS_MAX",
    "MIN_PERSON_PX",
    "CropView",
    "assess_colors",
    "box_in_crop",
    "ciede2000",
    "color_shares",
    "person_pixels",
    "srgb_to_lab",
]

Lab = npt.NDArray[np.float64]

MIN_PERSON_PX: Final = 20
BACKGROUND_DE: Final = 15.0
FRAME_FRACTION: Final = 0.15
CROPS_MAX: Final = 3
MATCH_MAIN: Final = 0.15
MATCH_OTHER: Final = 0.08
CONFLICT_MAIN_MAX: Final = 0.03
CONFLICT_OTHER_MIN: Final = 0.40
SATURATED_C: Final = 30.0
_KMEANS_ITERS: Final = 10
_BIG_PARTS: Final = frozenset({"outer", "upper", "full_body"})
_BIG_ITEMS: Final = frozenset({"backpack", "bag"})
# Swedish forms an exclusion may use ("inte röd jacka", "ingen blå mössa", "rött").
_FORMS: Final[dict[ColorName, tuple[str, ...]]] = {
    ColorName.BLACK: ("svart", "svarta", "black"),
    ColorName.WHITE: ("vit", "vitt", "vita", "white"),
    ColorName.GRAY: ("grå", "grått", "gråa", "gray", "grey"),
    ColorName.RED: ("röd", "rött", "röda", "red"),
    ColorName.ORANGE: ("orange", "orangea", "oranga"),
    ColorName.YELLOW: ("gul", "gult", "gula", "yellow"),
    ColorName.GREEN: ("grön", "grönt", "gröna", "green"),
    ColorName.BLUE: ("blå", "blått", "blåa", "blue"),
    ColorName.PURPLE: ("lila", "purple"),
    ColorName.PINK: ("rosa", "pink"),
    ColorName.BROWN: ("brun", "brunt", "bruna", "brown"),
    ColorName.BEIGE: ("beige",),
}
_WORD = re.compile(r"[a-zåäö]+")


@dataclass(frozen=True)
class CropView:
    """One crop of a candidate: its JPEG and where the detector's box lies in it."""

    detection_id: str
    px_size: float
    jpeg: bytes
    box: BBox  # in the crop's pixels


def box_in_crop(bbox: BBox, crop_box: BBox, width: int, height: int) -> BBox:
    """The detector's box (still pixels) in a crop cut from ``crop_box`` of the still and
    scaled to ``width`` x ``height``."""
    sx = width / crop_box.w if crop_box.w > 0 else 1.0
    sy = height / crop_box.h if crop_box.h > 0 else 1.0
    x0 = min(max(0.0, (bbox.x - crop_box.x) * sx), float(width))
    y0 = min(max(0.0, (bbox.y - crop_box.y) * sy), float(height))
    x1 = min(max(0.0, (bbox.x + bbox.w - crop_box.x) * sx), float(width))
    y1 = min(max(0.0, (bbox.y + bbox.h - crop_box.y) * sy), float(height))
    return BBox(x=x0, y=y0, w=max(0.0, x1 - x0), h=max(0.0, y1 - y0))


def srgb_to_lab(rgb: npt.NDArray[np.uint8]) -> Lab:
    """sRGB (0-255) to CIELAB (D65), any shape ending in 3."""
    c = rgb.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    # sRGB (D65) to XYZ, in units of 1e-7.
    m = (
        np.array(
            [
                [4124564, 3575761, 1804375],
                [2126729, 7151522, 721750],
                [193339, 1191920, 9503041],
            ]
        )
        / 1e7
    )
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    lab = np.empty_like(f)
    lab[..., 0] = 116 * f[..., 1] - 16
    lab[..., 1] = 500 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200 * (f[..., 1] - f[..., 2])
    return lab


def ciede2000(lab1: Lab, lab2: Lab) -> npt.NDArray[np.float64]:
    """CIEDE2000 between rows of ``lab1`` (N x 3) and ``lab2`` (N x 3 or 3)."""
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    cm7 = ((c1 + c2) / 2) ** 7
    g = 0.5 * (1 - np.sqrt(cm7 / (cm7 + 25.0**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dl, dc = l2 - l1, c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(c1p * c2p == 0, 0.0, dh)
    dhh = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))
    lm, cm = (l1 + l2) / 2, (c1p + c2p) / 2
    hsum = h1p + h2p
    hm = np.where(
        np.abs(h1p - h2p) > 180, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2), hsum / 2
    )
    hm = np.where(c1p * c2p == 0, hsum, hm)
    t = (
        1
        - 0.17 * np.cos(np.radians(hm - 30))
        + 0.24 * np.cos(np.radians(2 * hm))
        + 0.32 * np.cos(np.radians(3 * hm + 6))
        - 0.20 * np.cos(np.radians(4 * hm - 63))
    )
    sl = 1 + 0.015 * (lm - 50) ** 2 / np.sqrt(20 + (lm - 50) ** 2)
    sc = 1 + 0.045 * cm
    sh = 1 + 0.015 * cm * t
    cm7p = cm**7
    rt = (
        -2
        * np.sqrt(cm7p / (cm7p + 25.0**7))
        * np.sin(np.radians(60 * np.exp(-(((hm - 275) / 25) ** 2))))
    )
    return np.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dhh / sh) ** 2 + rt * (dc / sc) * (dhh / sh))


def _kmeans(px: Lab, k: int) -> Lab:
    """Deterministic k-means: the centres start at lightness quantiles."""
    order = px[np.argsort(px[:, 0], kind="stable")]
    idx = [int((2 * i + 1) * len(order) / (2 * k)) for i in range(k)]
    centres = order[idx].copy()
    for _ in range(_KMEANS_ITERS):
        d = np.linalg.norm(px[:, None, :] - centres[None, :, :], axis=2)
        nearest = np.argmin(d, axis=1)
        for j in range(k):
            members = px[nearest == j]
            if len(members):
                centres[j] = members.mean(axis=0)
    return centres


def person_pixels(lab: Lab, box: BBox) -> Lab | None:
    """The box's pixels that differ from the background around it (N x 3), or ``None``
    if there is no frame around the box to learn the background from."""
    h, w = lab.shape[:2]
    x0, y0 = round(box.x), round(box.y)
    x1, y1 = round(box.x + box.w), round(box.y + box.h)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    pad = max(2, round(FRAME_FRACTION * max(x1 - x0, y1 - y0)))
    fx0, fy0, fx1, fy1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    ring_mask = np.zeros((h, w), dtype=bool)
    ring_mask[fy0:fy1, fx0:fx1] = True
    ring_mask[y0:y1, x0:x1] = False
    ring = lab[ring_mask]
    if len(ring) < 3 * 3:
        return None
    centres = _kmeans(ring, 3)
    inside = lab[y0:y1, x0:x1].reshape(-1, 3)
    d = np.stack([ciede2000(inside, c) for c in centres], axis=1)
    return inside[np.min(d, axis=1) > BACKGROUND_DE]


def _ref(color: ColorName, shade: str | None) -> tuple[float, float, float]:
    ref = COLOR_TABLE[color].lab
    shift = {"light": SHADE_L_SHIFT, "dark": -SHADE_L_SHIFT}.get(shade or "", 0.0)
    return (min(100.0, max(0.0, ref[0] + shift)), ref[1], ref[2])


def _in_color(px: Lab, color: ColorName, shade: str | None = None) -> npt.NDArray[np.bool_]:
    ref = COLOR_TABLE[color]
    if ref.delta_e is None:  # black, white and grey: lightness and chroma bounds
        chroma = np.hypot(px[:, 1], px[:, 2])
        ok = chroma < (ref.c_max if ref.c_max is not None else np.inf)
        if ref.l_min is not None:
            ok &= px[:, 0] >= ref.l_min
        if ref.l_max is not None:
            ok &= px[:, 0] <= ref.l_max
        return ok
    return ciede2000(px, np.array(_ref(color, shade))) <= ref.delta_e


def color_shares(px: Lab, shades: dict[ColorName, str | None] | None = None) -> dict[str, float]:
    """The share of the person's pixels in each colour (``<color>``) and the share that is
    also saturated (``sat:<color>``, chromatic colours only)."""
    shades = shades or {}
    out: dict[str, float] = {}
    n = len(px)
    if n == 0:
        return out
    saturated = np.hypot(px[:, 1], px[:, 2]) > SATURATED_C
    for color, ref in COLOR_TABLE.items():
        mask = _in_color(px, color, shades.get(color))
        out[color.value] = float(mask.mean())
        if ref.delta_e is not None:
            out[f"sat:{color.value}"] = float((mask & saturated).mean())
    return out


def _excluded_colors(excluded: Sequence[str]) -> set[ColorName]:
    found: set[ColorName] = set()
    for text in excluded:
        words = set(_WORD.findall(text.lower()))
        found |= {c for c, forms in _FORMS.items() if words & set(forms)}
    return found


def _shades(profile: SearchProfile) -> dict[ColorName, str | None]:
    """The shade each main colour is asked in (the first garment that names it wins)."""
    out: dict[ColorName, str | None] = {}
    for g in profile.garments:
        out.setdefault(g.colors[0], g.shade)
    return out


def _verdicts(profile: SearchProfile, shares: dict[str, float] | None) -> dict[str, FieldVerdict]:
    fields: dict[str, FieldVerdict] = dict.fromkeys(profile.field_keys, "not_visible")
    if shares is None:
        return fields

    def share(c: ColorName) -> float:
        return 0.0 if c is ColorName.UNKNOWN else shares.get(c.value, 0.0)

    for i, g in enumerate(profile.garments):
        main = g.colors[0]
        if main is ColorName.UNKNOWN:
            continue
        big = g.part in _BIG_PARTS
        if share(main) >= (MATCH_MAIN if big else MATCH_OTHER):
            fields[f"garment:{i}"] = "match"
        elif (
            big
            and g.certainty in ("certain", "likely")
            and share(main) < CONFLICT_MAIN_MAX
            and any(
                shares.get(f"sat:{c.value}", 0.0) >= CONFLICT_OTHER_MIN
                for c in COLOR_TABLE
                if c is not main and COLOR_TABLE[c].delta_e is not None
            )
        ):
            fields[f"garment:{i}"] = "conflict"
    for i, item in enumerate(profile.items):
        if not item.colors or item.colors[0] is ColorName.UNKNOWN:
            continue
        need = MATCH_MAIN if item.kind in _BIG_ITEMS else MATCH_OTHER
        if share(item.colors[0]) >= need:
            fields[f"item:{i}"] = "match"
    if profile.hair is not None and share(profile.hair.color) >= MATCH_OTHER:
        fields["hair"] = "match"
    excluded = [c for c in _excluded_colors(profile.excluded) if share(c) >= CONFLICT_OTHER_MIN]
    if excluded:
        for i, g in enumerate(profile.garments):
            if g.part == "upper":
                fields[f"garment:{i}"] = "conflict"
    return fields


def _decode(jpeg: bytes) -> Lab | None:
    try:
        with Image.open(io.BytesIO(jpeg)) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    return srgb_to_lab(rgb)


def _explain(shares: dict[str, float] | None, n_crops: int) -> str:
    if shares is None:
        return "Personen syns inte tydligt nog i utsnitten för en färgbedömning."
    top = sorted(
        ((v, k) for k, v in shares.items() if not k.startswith("sat:") and v >= 0.05),
        reverse=True,
    )[:4]
    names = ", ".join(f"{COLOR_TABLE[ColorName(k)].sv} {v:.2f}".replace(".", ",") for v, k in top)
    crops = "ett utsnitt" if n_crops == 1 else f"{n_crops} utsnitt"
    return f"Färgandelar från {crops}: {names or 'inga tydliga färger'}."[:300]


def assess_colors(
    candidate_id: str,
    profile: SearchProfile | None,
    crops: Sequence[CropView],
    *,
    person_p: float,
    latency_ms: int = 0,
) -> CandidateAssessment:
    """The code's assessment (engine ``code``) of a candidate's crops. Without an approved
    profile only the colour shares are recorded (``profile_version`` 0, no fields)."""
    if profile is not None and not profile.approved:
        raise ValueError("only an approved search profile is used for matching")
    shades = _shades(profile) if profile is not None else {}
    used: list[str] = []
    per_crop: list[dict[str, float]] = []
    person_px = 0
    for crop in sorted(crops, key=lambda c: (-c.px_size, c.detection_id)):
        if len(used) >= CROPS_MAX:
            break
        lab = _decode(crop.jpeg)
        if lab is None:
            continue
        px = person_pixels(lab, crop.box)
        if px is None or len(px) < MIN_PERSON_PX:
            continue
        used.append(crop.detection_id)
        per_crop.append(color_shares(px, shades))
        person_px += len(px)
    shares: dict[str, float] | None = None
    if per_crop:
        shares = {k: round(sum(s[k] for s in per_crop) / len(per_crop), 4) for k in per_crop[0]}
    fields = _verdicts(profile, shares) if profile is not None else {}
    match = match_score(profile, fields) if profile is not None else None
    evidence: dict[str, float] = {"person_px": float(person_px), "crops": float(len(used))}
    if shares is not None:
        evidence |= {k: v for k, v in shares.items() if not k.startswith("sat:") and v > 0}
    return CandidateAssessment.model_validate(
        {
            "candidate_id": candidate_id,
            "profile_version": 0 if profile is None else profile.profile_version,
            "engine": "code",
            "model_id": f"sd_search.color:{COLOR_TABLE_VERSION}",
            "person": "unsure",  # colours never say whether it is a person
            "person_p": round(min(1.0, max(0.0, person_p)), 4),
            "fields": fields,
            "overall": 0.5 if match is None else match,
            "explanation_sv": _explain(shares, len(used)),
            "crops_used": used,
            "latency_ms": max(0, latency_ms),
            "evidence": evidence,
        }
    )
