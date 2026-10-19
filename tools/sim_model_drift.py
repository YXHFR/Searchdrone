"""Compare a freshly generated sd_fpv7 model folder with the committed one.

Used by ``tools/sim_model_from_cad.py --check``. Standard library only, so the comparison
can be tested without the CAD kernel (cadquery).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

JSON_TOLERANCE = 1e-6


def close(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float | int) and isinstance(b, float | int) and not isinstance(a, bool):
        return math.isclose(a, b, rel_tol=JSON_TOLERANCE, abs_tol=JSON_TOLERANCE * 1e-3)
    return a == b


def differences(fresh: Path, committed: Path) -> list[str]:
    names = {p.relative_to(fresh) for p in fresh.rglob("*") if p.is_file()}
    have = {p.relative_to(committed) for p in committed.rglob("*") if p.is_file()}
    have = {p for p in have if p.parts[0] in ("meshes",) or p.name == "cad.json"}
    out = [f"saknas: {p}" for p in sorted(names - have)]
    out += [f"finns inte i CAD-modellen: {p}" for p in sorted(have - names)]
    for rel in sorted(names & have):
        a, b = (fresh / rel).read_bytes(), (committed / rel).read_bytes()
        same = close(json.loads(a), json.loads(b)) if rel.suffix == ".json" else a == b
        if not same:
            out.append(f"skiljer sig från CAD-modellen: {rel}")
    return out
