"""``sd-searchsim train|eval key=value ...`` (behind ``just rl-train`` and ``just rl-eval``).

train: size=full|ci (default full), seed=<0..9>, workers=<int>, out=<policy file>
eval:  size=full|ci (default full), workers=<int>, policy=<policy file>, out=<result file>
both:  world=synthetic|real (default synthetic), sites=<folder from just rl-maps>,
       belief_motion=diffuse|lpb (default diffuse; lpb walks the map by sd_motion, real only),
       miss_correlation=<0..1> (default 0.8; 0 gives the numbers from before it existed)

The policy file and the results land in ``build/rl/`` by default, which git ignores:
trained weights never go into the repository.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import NoReturn

from sd_searchsim import evaluate, policies, train
from sd_searchsim.config import BELIEF_MOTIONS, SIZES, WORLDS, SimConfig
from sd_searchsim.realworld import load_sites

DEFAULT_DIR = Path("build/rl")
_WORLD_KEYS = {"world", "sites", "belief_motion", "miss_correlation"}
KEYS = {
    "train": {"size", "seed", "workers", "out", *_WORLD_KEYS},
    "eval": {"size", "workers", "policy", "out", *_WORLD_KEYS},
}


def _fail(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def _parse(command: str, argv: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in argv:
        key, sep, value = raw.partition("=")
        if not sep or not key:
            _fail(f"rl-{command}: '{raw}' är inte på formen nyckel=värde (t.ex. size=ci).")
        if key not in KEYS[command]:
            known = ", ".join(sorted(KEYS[command]))
            _fail(f"rl-{command}: okänd nyckel '{key}'. Tillåtna nycklar: {known}.")
        out[key] = value
    if out.get("size", "full") not in SIZES:
        _fail(f"rl-{command}: size ska vara {' eller '.join(SIZES)}.")
    if out.get("world", "synthetic") not in WORLDS:
        _fail(f"rl-{command}: world ska vara {' eller '.join(WORLDS)}.")
    belief = out.get("belief_motion", "diffuse")
    if belief not in BELIEF_MOTIONS:
        _fail(f"rl-{command}: belief_motion ska vara {' eller '.join(BELIEF_MOTIONS)}.")
    if belief == "lpb" and out.get("world", "synthetic") != "real":
        _fail(f"rl-{command}: belief_motion=lpb behöver world=real (vatten och stigar).")
    return out


def _int(args: dict[str, str], key: str, default: int, command: str) -> int:
    try:
        value = int(args.get(key, default))
    except ValueError:
        _fail(f"rl-{command}: {key} ska vara ett heltal.")
    if value < 0:
        _fail(f"rl-{command}: {key} får inte vara negativt.")
    return value


def _share(args: dict[str, str], key: str, default: float, command: str) -> float:
    try:
        value = float(args.get(key, default))
    except ValueError:
        _fail(f"rl-{command}: {key} ska vara ett tal mellan 0 och 1.")
    if not 0.0 <= value <= 1.0:
        _fail(f"rl-{command}: {key} ska vara mellan 0 och 1.")
    return value


def _write_json(path: Path, doc: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in KEYS:
        _fail("sd-searchsim: kommandot ska vara train eller eval.")
    command, args = argv[0], _parse(argv[0], argv[1:])
    size_name = args.get("size", "full")
    size = SIZES[size_name]
    workers = _int(args, "workers", os.cpu_count() or 1, command) or 1
    cfg = SimConfig(
        world=args.get("world", "synthetic"),
        belief_motion=args.get("belief_motion", "diffuse"),
        miss_correlation=_share(args, "miss_correlation", SimConfig.miss_correlation, command),
    )
    if cfg.world == "real":
        cfg = dataclasses.replace(cfg, sites_dir=args.get("sites", cfg.sites_dir))
        try:
            n_sites = len(load_sites(cfg.sites_dir, cfg.grid_cells))
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
            _fail(f"rl-{command}: {exc}")
        if n_sites < 10:
            _fail(f"rl-{command}: bara {n_sites} platser i {cfg.sites_dir}; minst 10 behövs.")
        print(f"Verkliga platser: {n_sites} i {cfg.sites_dir}.", flush=True)

    if command == "train":
        seed = _int(args, "seed", 0, command)
        if seed > train.MAX_SEED:
            _fail(f"rl-train: seed ska vara 0 till {train.MAX_SEED}.")
        out = Path(args.get("out", DEFAULT_DIR / "policy.json"))
        theta, log = train.train(
            size, cfg, seed=seed, workers=workers, progress=lambda m: print(m, flush=True)
        )
        meta = {
            "size": size_name,
            "world": cfg.world,
            "seed": seed,
            "run_size": dataclasses.asdict(size),
            "sim_config": dataclasses.asdict(cfg),
            "val_score_start": log.start_val,
            "val_score_best": log.best_val,
            "best_iteration": log.best_iteration,
            "seconds": round(log.seconds, 1),
            "iterations": log.iterations,
        }
        policies.save(out, theta, meta)
        print(
            f"klart på {log.seconds:.0f} s: validering {log.start_val:.4f} (girig) -> "
            f"{log.best_val:.4f} (varv {log.best_iteration}). Policyn ligger i {out}."
        )
        return 0

    policy_arg = args.get("policy")
    policy_path = Path(policy_arg) if policy_arg else DEFAULT_DIR / "policy.json"
    theta = None
    if policy_path.exists():
        try:
            theta = policies.load(policy_path)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            _fail(f"rl-eval: {exc}")
    elif policy_arg:
        _fail(f"rl-eval: policyfilen {policy_path} finns inte.")
    else:
        print(f"Ingen policy i {policy_path} (kör just rl-train); jämför bara mönstren.")
    result = evaluate.evaluate(size.eval_scenarios, theta, cfg, workers=workers)
    result["size"] = size_name
    result["world"] = cfg.world
    result["belief_motion"] = cfg.belief_motion
    result["miss_correlation"] = cfg.miss_correlation
    result["policy_file"] = str(policy_path) if theta is not None else None
    out = Path(args.get("out", DEFAULT_DIR / "eval.json"))
    _write_json(out, result)
    for name in result["ranking_by_score"]:
        c = result["choosers"][name]
        print(
            f"{name:11s} betyg {c['score']['mean']:.3f}  hittad {c['p_find']['mean']:.1%}  "
            f"väntad tid {c['expected_minutes']['mean']:.0f} min"
        )
    for name, d in result.get("policy_vs", {}).items():
        diff = d["score_diff"]
        print(
            f"policy - {name}: {diff['mean']:+.3f} (95 %: {diff['ci95_low']:+.3f} till "
            f"{diff['ci95_high']:+.3f}), bättre/lika/sämre {d['wins']}/{d['ties']}/{d['losses']}"
        )
    for cat, row in result.get("by_category", {}).items():
        best = max(row, key=lambda name: row[name])
        cells = "  ".join(f"{name} {v:.3f}" for name, v in row.items())
        print(f"{cat:12s} {cells}  (bäst: {best})")
    print(f"Resultatet ligger i {out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
