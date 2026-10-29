"""Detector profiles: what a measured detector does, in the form the simulator's
detector stand-in reads (``tools/bench/profiles/det_<model>_v<N>.json``).

    python -m bench.profiles model=damoyolo_s version=1 result=build/bench/det_damoyolo_s_test.json

A profile holds recall per size class at the fixed false-alarm rate, the false-alarm
rate, the score distributions of hits and false alarms (Beta, fitted by the method of
moments) and the time per picture. It is made from a result measured on one dataset
(``sets=heridal``): a result pooled over several is refused, since its recall would be
mostly that of the set with the most people. ``occlusion_factor`` cannot be measured on the bank
(the boxes say nothing of cover); it is carried over from the start profile and listed
under ``not_measured``. ``det_oracle_v0.json`` is the start profile from the published
HERIDAL figures, used until a measurement exists; anything built on it is modelled,
not measured.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from bench.common import BENCH, BenchError, parse_args, say, warn
from bench.metrics import SIZE_CLASSES

PROFILES = BENCH / "profiles"
START = PROFILES / "det_oracle_v0.json"
KEYS = ("model", "version", "result", "out")
_NAME = re.compile(r"^[a-z0-9_]{1,40}$")


def build_profile(
    model: str, version: int, result: dict[str, object], start: dict[str, object]
) -> dict[str, object]:
    if result.get("schema") != "bench.detector/v1":
        raise BenchError("Resultatfilen har fel format (väntat bench.detector/v1).", 2)
    sets = result.get("recall_by_dataset") or {}
    assert isinstance(sets, dict)  # noqa: S101 - schema above
    if len(sets) > 1:
        # a pooled recall is mostly the set with the most people, not the terrain searched
        raise BenchError(
            f"Resultatet är mätt på flera datamängder ({', '.join(sorted(sets))}); en profil "
            "görs av en mätning på en enda (sets=heridal).",
            2,
        )
    recall = result["recall"]
    latency = result["latency_ms"]
    assert isinstance(recall, dict) and isinstance(latency, dict)  # noqa: S101 - schema above
    not_measured = ["occlusion_factor"]
    profile: dict[str, object] = {
        "profile_id": f"det_{model}_v{version}",
        "source": f"bench {result['source']} {result['measured']}",
        "recall": {c: recall[c]["recall"] for c in SIZE_CLASSES},
        "recall_n": {c: recall[c]["n"] for c in SIZE_CLASSES},
        "occlusion_factor": start["occlusion_factor"],
        "fp_per_12mp": result["fp_per_12mp_target"],
        "score_hit": result.get("score_hit") or start["score_hit"],
        "score_fp": result.get("score_fp") or start["score_fp"],
        "latency_ms": {"p50": latency["p50"], "p95": latency["p95"]},
    }
    for key in ("score_hit", "score_fp"):
        if not result.get(key):
            not_measured.append(key)
    profile["not_measured"] = not_measured
    return profile


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv, KEYS, "bench profiles")
        model, version = args.get("model", ""), int(args.get("version", "1"))
        if not _NAME.match(model):
            raise BenchError("model= ska vara små bokstäver, siffror och _", 2)
        result = json.loads(Path(args["result"]).read_text(encoding="utf-8"))
        start = json.loads(START.read_text(encoding="utf-8"))
        profile = build_profile(model, version, result, start)
        out = Path(args.get("out") or PROFILES / f"det_{model}_v{version}.json")
        out.write_text(json.dumps(profile, indent=1) + "\n", encoding="utf-8")
        say(f"skrev {out}")
        return 0
    except KeyError as exc:
        warn(f"saknar {exc}")
        return 2
    except BenchError as exc:
        warn(str(exc))
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
