"""Pick the detector to use from measured results: ``just pick-detector``.

    just bench-detector model=person_yolox_s part=val
    just bench-detector model=person_damoyolo_s part=val
    just pick-detector

Keys: ``results`` (result files, comma separated; default every
``build/bench/det_*_val.json``), ``data``.

The rule, in order: the highest recall on people over 32 px wins; a model within 2
percentage points of it is as good on that, and among those the higher recall at
16-32 px wins; within 2 points there too, the faster one (median time per picture)
wins. When the val part holds several datasets, each recall is the lowest of the
datasets' own recalls for that size, never the pooled one: the datasets differ a lot in
size (KIOS has far more people than HERIDAL), and a model that gets worse on one of
them must not win on the strength of another. Picked on the val part, never on test:
the test part is only measured once, on the chosen model. Every result must come from
the same pictures (the same ``source`` and picture count); a quick look with ``limit=``
is never ranked against a full run.
Prints the setting that makes the on-board computer use it.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bench.common import RESULTS, BenchError, data_dir, parse_args, say, warn

KEYS = ("results", "data")
TIE = 0.02
BUILT_IN = frozenset({"null", "facit"})


def _recall(body: dict[str, Any], cls: str) -> float:
    """The weakest dataset's recall for ``cls``; the pooled one for older results."""
    rows = [
        row["by_class"][cls]
        for row in (body.get("recall_by_dataset") or {}).values()
        if "by_class" in row and row["by_class"][cls]["n"]
    ]
    if not rows:
        return float(body["recall"][cls]["recall"])
    return min(float(r["recall"]) for r in rows)


def _per_set(body: dict[str, Any]) -> str:
    parts = [
        f"{name} {row['by_class']['gt32']['recall']:.3f}/{row['by_class']['16to32']['recall']:.3f}"
        for name, row in sorted((body.get("recall_by_dataset") or {}).items())
        if "by_class" in row
    ]
    return f" ({', '.join(parts)})" if len(parts) > 1 else ""


def _fields(body: dict[str, Any]) -> tuple[str, float, float, float]:
    """What the rule reads from a result; a missing or wrong field raises."""
    return (
        str(body["model"]),
        _recall(body, "gt32"),
        _recall(body, "16to32"),
        float(body["latency_ms"]["p50"]),
    )


def pick(bodies: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The chosen result (rule in the module doc)."""
    if not bodies:
        raise BenchError("Inga mätresultat att välja bland.", 2)
    best = max(_recall(b, "gt32") for b in bodies)
    left = [b for b in bodies if _recall(b, "gt32") >= best - TIE]
    best = max(_recall(b, "16to32") for b in left)
    left = [b for b in left if _recall(b, "16to32") >= best - TIE]
    return min(left, key=lambda b: (float(b["latency_ms"]["p50"]), b["model"]))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv, KEYS, "pick-detector")
        files = (
            [Path(p) for p in args["results"].split(",") if p]
            if args.get("results")
            else sorted(RESULTS.glob("det_*_val.json"))
        )
        bodies = []
        for f in files:
            try:
                body = json.loads(f.read_text(encoding="utf-8"))
                _fields(body)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise BenchError(f"Kan inte läsa mätresultatet {f}: {exc!r}", 2) from exc
            if body.get("schema") != "bench.detector/v1" or body["model"] in BUILT_IN:
                continue
            if " val " not in str(body.get("source")):
                warn(f"hoppar över {f.name}: inte mätt på val-delen")
                continue
            bodies.append(body)
        same = {(str(b.get("source")), b.get("n_pictures")) for b in bodies}
        if len(same) > 1:
            listed = "; ".join(
                f"{b['model']}: {b.get('source')}, {b.get('n_pictures')} bilder" for b in bodies
            )
            raise BenchError(f"Resultaten är mätta på olika bilder och jämförs inte: {listed}", 2)
        chosen = pick(bodies)
        if " limit=" in str(chosen.get("source")):
            warn("bara ett snabbtest (limit=); mät på hela val-delen innan du väljer på riktigt")
        for b in sorted(bodies, key=lambda b: b["model"]):
            say(
                f"  {b['model']}: > 32 px {_recall(b, 'gt32'):.3f}, 16-32 px "
                f"{_recall(b, '16to32'):.3f}, {b['latency_ms']['p50']} ms{_per_set(b)}"
            )
        manifest = data_dir(args) / "models" / "detector" / f"{chosen['model']}.json"
        say(f"Vald: {chosen['model']}.")
        say(f"Den inbyggda datorn (sd_onboard) använder den med: SD_DETECTOR_MODEL={manifest}")
        return 0
    except BenchError as exc:
        warn(str(exc))
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
