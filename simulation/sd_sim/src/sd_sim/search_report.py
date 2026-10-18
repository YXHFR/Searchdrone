"""The evaluation and record of a search scenario run (SK-RED): finds against the ground
truth, pictures for people to look at, and the numbers.

The ground truth is the world's facit (``sd_sim.world.world_truth``: every object's place,
label and ``truth``, which for scenario objects holds ``cls`` and ``role``).
``match_finds`` pairs each target in the ground truth with the nearest find of its class
(within ``max_m``) and reports the position error, the missed targets and the finds that
match no target (false finds). ``draw_map`` draws the search box, the ground truth, the
finds with their errors and the flown track from above; ``annotate_still`` draws the
detector's boxes on a kept still. ``write`` puts everything in ``<run>/search/``.

Only the simulator has a ground truth: nothing here runs outside the tests.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sd_safety.targets import Plane

__all__ = ["Match", "Result", "annotate_still", "draw_map", "match_finds", "write"]

Xy = tuple[float, float]


@dataclass(frozen=True)
class Match:
    target: str
    find_id: str
    error_m: float
    target_enu: Xy
    find_enu: Xy


@dataclass
class Result:
    label: str
    max_m: float
    matches: list[Match] = field(default_factory=list[Match])
    missed: list[str] = field(default_factory=list[str])
    false_finds: list[str] = field(default_factory=list[str])

    @property
    def ok(self) -> bool:
        return not self.missed and not self.false_finds

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "max_m": self.max_m,
            "ok": self.ok,
            "matches": [asdict(m) for m in self.matches],
            "missed": self.missed,
            "false_finds": self.false_finds,
            "max_error_m": max((m.error_m for m in self.matches), default=None),
        }


def match_finds(
    truth: dict[str, Any],
    finds: Sequence[dict[str, Any]],
    label: str = "red_block",
    max_m: float = 3.0,
) -> Result:
    """Greedy nearest pairing of targets of ``label`` and finds of ``label``."""
    plane = Plane(truth["home"]["lat"], truth["home"]["lon"])
    targets = [o for o in truth["objects"] if _cls(o) == label]
    cands = [f for f in finds if f["label"] == label]
    pairs = sorted(
        (
            math.dist((t["east_m"], t["north_m"]), plane.xy(f["lat"], f["lon"])),
            t["name"],
            f["find_id"],
        )
        for t in targets
        for f in cands
    )
    out = Result(label, max_m)
    used_t: set[str] = set()
    used_f: set[str] = set()
    by_name = {t["name"]: t for t in targets}
    by_id = {f["find_id"]: f for f in cands}
    for dist, name, fid in pairs:
        if dist > max_m or name in used_t or fid in used_f:
            continue
        used_t.add(name)
        used_f.add(fid)
        t, f = by_name[name], by_id[fid]
        out.matches.append(
            Match(
                name,
                fid,
                round(dist, 2),
                (t["east_m"], t["north_m"]),
                tuple(round(v, 2) for v in plane.xy(f["lat"], f["lon"])),  # type: ignore[arg-type]
            )
        )
    out.missed = sorted(set(by_name) - used_t)
    out.false_finds = sorted(set(by_id) - used_f)
    out.matches.sort(key=lambda m: m.target)
    return out


def _cls(obj: dict[str, Any]) -> str | None:
    return obj.get("truth", {}).get("cls")


_COLORS = {
    "red_block": (220, 30, 30),
    "car": (40, 90, 200),
    "building": (130, 60, 40),
    "road": (90, 90, 90),
    "person": (240, 160, 0),
}


def draw_map(
    truth: dict[str, Any],
    finds: Sequence[dict[str, Any]],
    result: Result,
    track: Iterable[tuple[float, float]],
    path: Path,
    box: tuple[float, float, float, float],
    title: str,
    size_px: int = 900,
) -> Path:
    """The search box (``box``: x0, y0, x1, y1 in metres east/north of home) from above:
    ground truth (squares), finds (circles), a line and the error from each target to its
    find, and the flown track (grey)."""
    from PIL import Image, ImageDraw  # test tooling only (dev dependency)

    plane = Plane(truth["home"]["lat"], truth["home"]["lon"])
    x0, y0, x1, y1 = box
    pad = 25.0
    span = max(x1 - x0, y1 - y0) + 2 * pad
    scale = size_px / span

    def px(e: float, n: float) -> Xy:
        return ((e - (x0 - pad)) * scale, size_px - (n - (y0 - pad)) * scale)

    img = Image.new("RGB", (size_px, size_px), (206, 222, 190))
    d = ImageDraw.Draw(img)
    d.rectangle([px(x0, y1), px(x1, y0)], outline=(60, 60, 60), width=2)
    pts = [px(*plane.xy(lat, lon)) for lat, lon in track]
    if len(pts) > 1:
        d.line(pts, fill=(120, 120, 120), width=1)
    for o in truth["objects"]:
        cls = _cls(o)
        if cls is None or cls == "road":
            continue
        cx, cy = px(o["east_m"], o["north_m"])
        r = 7 if cls == "red_block" else 10
        d.rectangle([cx - r, cy - r, cx + r, cy + r], outline=_COLORS.get(cls, (0, 0, 0)), width=3)
        d.text((cx + r + 3, cy - r), f"{o['name']} ({o['truth'].get('role', '')})", fill=(0, 0, 0))
    for f in finds:
        e, n = plane.xy(f["lat"], f["lon"])
        cx, cy = px(e, n)
        d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=_COLORS.get(f["label"], (0, 0, 0)))
    for m in result.matches:
        a, b = px(*m.target_enu), px(*m.find_enu)
        d.line([a, b], fill=(0, 0, 0), width=2)
        d.text((b[0] + 8, b[1] + 4), f"{m.error_m:.2f} m", fill=(0, 0, 0))
    d.text(
        (10, 10),
        f"{title}: {len(result.matches)} of "
        f"{len(result.matches) + len(result.missed)} blocks within {result.max_m:g} m, "
        f"{len(result.false_finds)} false",
        fill=(0, 0, 0),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def annotate_still(jpeg: Path, report: dict[str, Any], path: Path) -> Path:
    """A kept still with the detector's boxes, class, source and score drawn on it."""
    from PIL import Image, ImageDraw  # test tooling only (dev dependency)

    with Image.open(jpeg) as im:
        img = im.convert("RGB")
    d = ImageDraw.Draw(img)
    for det in report["detections"]:
        b = det["bbox"]
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        color = _COLORS.get(det["label"], (255, 255, 255))
        d.rectangle([x - 3, y - 3, x + w + 3, y + h + 3], outline=color, width=3)
        d.text(
            (x, max(0, y - 14)), f"{det['label']} {det['source']} {det['score']:.2f}", fill=color
        )
    pose = report["pose"]
    d.text((10, 10), f"{pose['agl_m']:.0f} m, yaw {pose['yaw_deg']:.0f}", fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=90)
    return path


def write(run_dir: Path, summary: dict[str, Any]) -> Path:
    out = run_dir / "search"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "result.json"
    path.write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    return path
