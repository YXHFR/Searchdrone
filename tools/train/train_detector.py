"""Fine-tune a person detector on the image bank: ``just train-detector model=yolox_s``.

    just train-detector model=yolox_s          # or model=damoyolo_s
    just train-detector model=yolox_s simfarm=1   # first a few epochs on the sim farm
    just train-detector model=yolox_s init=<old>/best.pt out=<new folder>   # go on from
                                                  # earlier weights, in a new folder

Keys: ``model`` (yolox_s, damoyolo_s), ``epochs`` (default 20), ``batch`` (default 8,
fits an 8 GB graphics card), ``sets`` (default heridal,kios,wisard: the fetched ones
that may be trained on, mixed), ``init`` (a ``best.pt`` of the same model to start from
instead of the COCO weights; it must lie outside ``out``), ``simfarm`` (1: pre-train on
the synthetic pictures first, ``simfarm_epochs`` of them, default 3), ``device`` (auto,
cuda, cpu), ``data``, ``out`` (default ``<data>/train/<model>``), ``limit`` (only the
first pictures: for trying it out), ``workers`` (picture loaders, default 4), ``seed``.

Stops and goes on: the state is saved to ``last.pt`` every few minutes and after every
epoch. Run the same command again and it carries on from there (a different command,
say another ``batch``, starts afresh in a new folder with ``fresh=1``). The best epoch
on the val part goes to ``best.pt``; ``just export-detector`` makes the ONNX file from it.
The val AP50 is printed per dataset; the best epoch is the one with the highest mean of
them, so the set with the most people does not decide alone.

Never trained on: the test part (locked in ``tools/bench/splits/``) and every set whose
licence allows measuring only. The weights stay on this computer.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import families
import numpy as np
import torch
from bench.common import BenchError, parse_args, say, warn
from evaluate import ap50
from tiles import TILE_PX, Tile, build_cache, load_tile, pictures
from torch.utils.data import DataLoader, Dataset

KEYS = (
    "model", "epochs", "batch", "sets", "simfarm", "simfarm_epochs", "device", "data",
    "out", "limit", "workers", "seed", "fresh", "lr", "save_minutes", "tile", "init",
)  # fmt: skip
SAVE_MINUTES = 10.0


@dataclass(frozen=True)
class Settings:
    model: str
    epochs: int = 20
    batch: int = 8
    sets: tuple[str, ...] = ("heridal", "kios", "wisard")
    simfarm_epochs: int = 0
    lr: float = 0.01  # per 64 pictures, as YOLOX
    seed: int = 0
    limit: int = 0
    tile: int = TILE_PX
    init: str = ""  # a best.pt to start from instead of COCO ("": COCO)

    def key(self) -> dict[str, Any]:
        body = json.loads(json.dumps(asdict(self)))  # as saved: lists, not tuples
        if not self.init:
            del body["init"]  # a run started before init= existed carries on
        return body


class TileSet(Dataset[tuple[np.ndarray, np.ndarray]]):
    def __init__(self, folder: Path, tiles: Sequence[Tile], tile: int, pad: int, augment: bool):
        self.folder, self.tiles, self.tile, self.pad = folder, list(tiles), tile, pad
        self.augment = augment

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        # augmentation only, not a secret
        rng = random.Random(hash((i, time.monotonic_ns()))) if self.augment else None  # noqa: S311
        return load_tile(self.folder, self.tiles[i], self.tile, self.pad, rng)


def collate(batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    images = torch.from_numpy(np.stack([b[0] for b in batch]))
    return images, [torch.from_numpy(b[1]) for b in batch]


class Ema:
    """Exponential moving average of the weights, as YOLOX trains (decay ramps up)."""

    def __init__(self, net: torch.nn.Module, decay: float = 0.9998, updates: int = 0):
        import copy

        self.net = copy.deepcopy(net).eval()
        self.updates = updates
        self.decay = decay
        for p in self.net.parameters():
            p.requires_grad_(False)

    def update(self, net: torch.nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        with torch.no_grad():
            own = self.net.state_dict()
            for k, v in net.state_dict().items():
                if v.dtype.is_floating_point:
                    own[k].mul_(d).add_(v.detach(), alpha=1 - d)
                else:
                    own[k].copy_(v)


def pick_device(name: str) -> torch.device:
    if name in ("", "auto"):
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise BenchError(
            'Inget grafikkort syns (CUDA). Se "What you need" i tools/train/README.md, '
            "eller kör device=cpu.",
            2,
        )
    return torch.device(name)


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step / max(1, warmup)) ** 2
    t = (step - warmup) / max(1, total - warmup)
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))


def evaluate_val(
    net: torch.nn.Module,
    model: str,
    loader: DataLoader[Any],
    device: torch.device,
    datasets: Sequence[str],
) -> dict[str, float]:
    """AP50 per dataset (``datasets``: each val tile's, in the loader's order)."""
    net.eval()
    results = []
    with torch.no_grad():
        for images, boxes in loader:
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                pred_boxes, scores = families.predict(net, model, images.to(device))
            for pb, ps, truth in zip(
                pred_boxes.float().cpu(), scores.float().cpu(), boxes, strict=True
            ):
                keep = ps >= 0.01
                results.append((pb[keep].numpy(), ps[keep].numpy(), truth.numpy()))
    if len(results) != len(datasets):
        raise BenchError("val-rutorna och deras datamängder stämmer inte överens", 1)
    return {
        name: ap50([r for r, d in zip(results, datasets, strict=True) if d == name])
        for name in sorted(set(datasets))
    }


def val_score(per_set: dict[str, float]) -> float:
    """The best epoch's measure: the mean of the per-dataset AP50."""
    return sum(per_set.values()) / len(per_set) if per_set else 0.0


def save(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def run_stage(
    *,
    stage: str,
    settings: Settings,
    net: torch.nn.Module,
    ema: Ema,
    train_tiles: tuple[Path, list[Tile]],
    val_tiles: tuple[Path, list[Tile]] | None,
    epochs: int,
    out: Path,
    device: torch.device,
    workers: int,
    save_minutes: float,
    resume: dict[str, Any] | None,
    lineage: dict[str, Any] | None = None,
) -> None:
    """``lineage``: what the weights have seen in all, ``init`` included (``trained_on``,
    ``pretrained_on``, ``base_weights``, ``started_from``); saved in ``best.pt`` for the
    manifest's credits."""
    fam = families.FAMILIES[settings.model]
    lineage = lineage or {}
    folder, tiles = train_tiles
    if not tiles:
        raise BenchError("Inga träningsrutor. Är bilderna hämtade? Kör: just bench-fetch yes=1", 2)
    steps_per_epoch = max(1, len(tiles) // settings.batch)
    total = steps_per_epoch * epochs
    warmup = min(total // 10, 3 * steps_per_epoch) if stage == "train" else 0
    base_lr = settings.lr * settings.batch / 64
    groups: tuple[list[Any], list[Any], list[Any]] = ([], [], [])  # bn, weights, biases
    for m in net.modules():
        if hasattr(m, "bias") and isinstance(m.bias, torch.nn.Parameter):
            groups[2].append(m.bias)
        if isinstance(m, torch.nn.BatchNorm2d):
            groups[0].append(m.weight)
        elif hasattr(m, "weight") and isinstance(m.weight, torch.nn.Parameter):
            groups[1].append(m.weight)
    opt = torch.optim.SGD(groups[0], lr=base_lr, momentum=0.9, nesterov=True)
    opt.add_param_group({"params": groups[1], "weight_decay": 5e-4})
    opt.add_param_group({"params": groups[2]})
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    epoch, step_in_epoch, best = 0, 0, -1.0
    if resume is not None:
        opt.load_state_dict(resume["opt"])
        scaler.load_state_dict(resume["scaler"])
        epoch, step_in_epoch, best = resume["epoch"], resume["step"], resume["best"]
        say(f"{stage}: fortsätter från epok {epoch + 1}, steg {step_in_epoch}")
    loader_kw: dict[str, Any] = {
        "num_workers": workers,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
    }
    val_loader = None
    if val_tiles is not None and val_tiles[1]:
        val_loader = DataLoader(
            TileSet(val_tiles[0], val_tiles[1], settings.tile, fam.pad, augment=False),
            batch_size=settings.batch,
            **loader_kw,
        )
    last_save = time.monotonic()
    dataset = TileSet(folder, tiles, settings.tile, fam.pad, augment=True)

    def snapshot(e: int, s: int) -> dict[str, Any]:
        return {
            "stage": stage,
            "settings": settings.key(),
            "epoch": e,
            "step": s,
            "best": best,
            "net": net.state_dict(),
            "ema": ema.net.state_dict(),
            "ema_updates": ema.updates,
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
        }

    while epoch < epochs:
        order = list(range(len(tiles)))
        random.Random(settings.seed * 1000 + epoch).shuffle(order)  # noqa: S311 - repeatable order
        order = order[step_in_epoch * settings.batch : steps_per_epoch * settings.batch]
        loader = DataLoader(
            torch.utils.data.Subset(dataset, order),
            batch_size=settings.batch,
            shuffle=False,
            drop_last=True,
            **loader_kw,
        )
        net.train()
        started, seen, losses = time.monotonic(), 0, []
        for images, boxes in loader:
            step = epoch * steps_per_epoch + step_in_epoch
            lr = lr_at(step, total, warmup, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            images = images.to(device, non_blocking=True)
            targets = families.train_targets(settings.model, boxes, device)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = families.loss(net, images, targets)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(net)
            step_in_epoch += 1
            seen += len(images)
            losses.append(float(loss.detach()))
            if step_in_epoch % 50 == 0:
                rate = seen / max(1e-6, time.monotonic() - started)
                left = (steps_per_epoch - step_in_epoch) * settings.batch / max(rate, 1e-6)
                say(
                    f"{stage} epok {epoch + 1}/{epochs}  steg {step_in_epoch}/{steps_per_epoch}  "
                    f"förlust {np.mean(losses[-50:]):.3f}  {rate:.1f} rutor/s  "
                    f"~{left / 60:.0f} min kvar av epoken"
                )
            if time.monotonic() - last_save > save_minutes * 60:
                save(out / "last.pt", snapshot(epoch, step_in_epoch))
                last_save = time.monotonic()
        epoch, step_in_epoch = epoch + 1, 0
        score, per_set = None, {}
        if val_loader is not None:
            assert val_tiles is not None  # noqa: S101 - val_loader is made from it
            sets = [t.source.split(":", 1)[0] for t in val_tiles[1]]
            per_set = evaluate_val(ema.net, settings.model, val_loader, device, sets)
            score = val_score(per_set)
            if score > best:
                best = score
                save(
                    out / "best.pt",
                    {
                        "model": settings.model,
                        "settings": settings.key(),
                        "ema": ema.net.state_dict(),
                        "val_ap50": score,
                        "val_ap50_by_dataset": per_set,
                        "epoch": epoch,
                        "stage": stage,
                        **lineage,
                    },
                )
        mean = np.mean(losses) if losses else float("nan")
        say(
            f"{stage} epok {epoch}/{epochs} klar: förlust {mean:.3f}"
            + (f", val AP50 {score:.3f} (bäst {best:.3f})" if score is not None else "")
            + "".join(f"; {name} {ap:.3f}" for name, ap in per_set.items())
        )
        save(out / "last.pt", snapshot(epoch, 0))
        last_save = time.monotonic()
    if val_loader is None:
        save(
            out / "best.pt",
            {
                "model": settings.model,
                "settings": settings.key(),
                "ema": ema.net.state_dict(),
                "val_ap50": None,
                "epoch": epoch,
                "stage": stage,
                **lineage,
            },
        )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv, KEYS, "train-detector")
        model = args.get("model", "")
        if model not in families.FAMILIES:
            raise BenchError(f"model= ska vara en av {', '.join(families.FAMILIES)}", 2)
        if not args.get("data"):
            raise BenchError("data= saknas (just train-detector fyller i den).", 2)
        data = Path(args["data"]).expanduser().resolve()
        simfarm = args.get("simfarm", "0") == "1"
        settings = Settings(
            model=model,
            epochs=int(args.get("epochs", "20")),
            batch=int(args.get("batch", "8")),
            sets=tuple(s for s in args.get("sets", "heridal,kios,wisard").split(",") if s),
            simfarm_epochs=int(args.get("simfarm_epochs", "3")) if simfarm else 0,
            lr=float(args.get("lr", "0.01")),
            seed=int(args.get("seed", "0")),
            limit=int(args.get("limit", "0")),
            tile=int(args.get("tile", str(TILE_PX))),
            init=str(Path(args["init"]).expanduser().resolve()) if args.get("init") else "",
        )
        if settings.tile % 32 or not 64 <= settings.tile <= 1280:
            raise BenchError("tile= ska vara delbart med 32 och mellan 64 och 1280", 2)
        out = Path(args.get("out") or data / "train" / model).expanduser().resolve()
        if settings.init and Path(settings.init).is_relative_to(out):
            raise BenchError(
                f"init= ligger i out-mappen ({out}) och skulle skrivas över. Ange en annan "
                f"out=, till exempel out={data / 'train' / (model + '_mix')}",
                2,
            )
        out.mkdir(parents=True, exist_ok=True)
        device = pick_device(args.get("device", "auto"))
        say(f"Tränar på {'grafikkortet (cuda)' if device.type == 'cuda' else 'processorn (cpu)'}.")
        workers = int(args.get("workers", "4"))
        save_minutes = float(args.get("save_minutes", str(SAVE_MINUTES)))
        torch.manual_seed(settings.seed)
        return train(settings, data, out, device, workers, save_minutes, args.get("fresh") == "1")
    except BenchError as exc:
        warn(str(exc))
        return exc.code
    except torch.OutOfMemoryError:
        warn(
            "Grafikkortets minne räckte inte. Kör igen med en mindre batch, som "
            "batch=4 fresh=1 (fresh=1 börjar om, eftersom batchen är en del av inställningarna)."
        )
        return 1
    except KeyboardInterrupt:
        warn("Avbruten. Kör samma kommando igen för att fortsätta där det sparades senast.")
        return 130


def _available(data: Path, sets: Sequence[str]) -> list[str]:
    from bench.datasets import index_path

    have = [s for s in sets if index_path(data, s).is_file()]
    for s in sets:
        if s not in have:
            warn(f"{s} är inte hämtad och tränas inte på (just bench-fetch yes=1 sets={s}).")
    return have


def trained_sets(sets: Sequence[str]) -> list[str]:
    """The sets of ``sets`` that are really trained on: measure-only ones are left out,
    as ``tiles.pictures`` leaves them out, so they are never credited as seen."""
    from bench.datasets import load_register

    register = load_register()
    return [s for s in sets if s in register and "train" in register[s].use]


def load_init(net: torch.nn.Module, model: str, path: Path) -> dict[str, Any]:
    """Start from an earlier run's best weights (its EMA) instead of COCO's. Returns what
    those weights have seen (``trained_on``, ``pretrained_on``), so their credit stays,
    and ``started_from``: the file's name, SHA-256 and epoch (never its local path)."""
    if not path.is_file():
        raise BenchError(f"init={path} finns inte.", 2)
    ck: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("model") != model or "ema" not in ck:
        raise BenchError(f"init={path} är inte en best.pt för {model}.", 2)
    net.load_state_dict(ck["ema"])
    earlier = ck.get("settings", {})
    seen: dict[str, Any] = {
        "trained_on": list(ck.get("trained_on", trained_sets(earlier.get("sets", [])))),
        "pretrained_on": list(
            ck.get("pretrained_on", ["simfarm"] if earlier.get("simfarm_epochs") else [])
        ),
        "started_from": {
            "file": f"{path.parent.name}/{path.name}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "epoch": ck.get("epoch"),
            "base_weights": ck.get("base_weights", f"COCO {model} (Apache-2.0)"),
        },
    }
    trained = ", ".join(seen["trained_on"])
    say(f"{model}: startar från {path} (tränad på {trained or 'okänt'}, epok {ck.get('epoch')})")
    return seen


def train(
    settings: Settings,
    data: Path,
    out: Path,
    device: torch.device,
    workers: int,
    save_minutes: float,
    fresh: bool,
) -> int:
    weights = data / "models" / "pretrained" / families.PRETRAINED[settings.model]
    if not weights.is_file():
        raise BenchError(
            f"COCO-vikterna saknas ({weights.name}). Kör: just detector-fetch yes=1", 2
        )
    sets = _available(data, settings.sets)
    if not sets:
        raise BenchError("Inga bilder att träna på. Kör: just bench-fetch yes=1", 2)
    settings = replace(settings, sets=tuple(sets))  # what is really trained on
    state_file = out / "settings.json"
    last = out / "last.pt"
    resume: dict[str, Any] | None = None
    if last.is_file() and not fresh:
        saved = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else None
        if saved != settings.key():
            raise BenchError(
                f"{out} har en träning med andra inställningar. Kör med samma inställningar, "
                "ange en annan out=, eller börja om med fresh=1.",
                2,
            )
        resume = torch.load(last, map_location="cpu", weights_only=True)
    state_file.write_text(json.dumps(settings.key(), indent=1) + "\n", encoding="utf-8")

    net = families.build(settings.model)
    person = families.load_coco(net, settings.model, weights)
    say(f"{settings.model}: COCO-vikter inlästa, personklassen startar från {len(person)} lager")
    seen: dict[str, Any] = {"trained_on": [], "pretrained_on": []}
    if settings.init:
        seen = load_init(net, settings.model, Path(settings.init))
    lineage: dict[str, Any] = {
        "trained_on": sorted({*seen["trained_on"], *trained_sets(settings.sets)}),
        "pretrained_on": sorted(
            {*seen["pretrained_on"], *(["simfarm"] if settings.simfarm_epochs else [])}
        ),
        "base_weights": f"COCO {settings.model} (Apache-2.0)",
    }
    if "started_from" in seen:
        start = seen["started_from"]
        lineage["started_from"] = start
        lineage["base_weights"] = (
            f"{start['file']} (sha256 {start['sha256'][:12]}, epoch {start['epoch']}), "
            f"itself from {start['base_weights']}"
        )
    ema = Ema(net)
    if resume is not None:
        net.load_state_dict(resume["net"])
        ema.net.load_state_dict(resume["ema"])
        ema.updates = resume["ema_updates"]
    net.to(device)
    ema.net.to(device)

    stages: list[tuple[str, int]] = []
    if settings.simfarm_epochs:
        stages.append(("simfarm", settings.simfarm_epochs))
    stages.append(("train", settings.epochs))
    for stage, epochs in stages:
        if (
            resume is not None
            and resume["stage"] != stage
            and [s for s, _ in stages].index(resume["stage"]) > [s for s, _ in stages].index(stage)
        ):
            continue  # this stage is done
        stage_resume = resume if resume is not None and resume["stage"] == stage else None
        if stage == "simfarm":
            items = pictures(data, "pretrain", ["simfarm"], settings.limit)
            train_tiles = (
                out / "tiles" / "simfarm",
                build_cache(items, out / "tiles" / "simfarm", settings.tile),
            )
            val = None
        else:
            items = pictures(data, "train", sets, settings.limit)
            train_tiles = (
                out / "tiles" / "train",
                build_cache(items, out / "tiles" / "train", settings.tile),
            )
            val_items = pictures(data, "val", sets, settings.limit)
            val = (
                out / "tiles" / "val",
                build_cache(val_items, out / "tiles" / "val", settings.tile, empty_share=0.1),
            )
        say(f"{stage}: {len(train_tiles[1])} rutor från {len(items)} bilder")
        run_stage(
            stage=stage,
            settings=settings,
            net=net,
            ema=ema,
            train_tiles=train_tiles,
            val_tiles=val,
            epochs=epochs,
            out=out,
            device=device,
            workers=workers,
            save_minutes=save_minutes,
            resume=stage_resume,
            lineage=lineage,
        )
        resume = None
    say(
        f"Klart. Bästa vikterna: {out / 'best.pt'}. "
        f"Nästa steg: just export-detector model={settings.model}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
