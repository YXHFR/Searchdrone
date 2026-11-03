"""Fine-tune a person detector on the image bank: ``just train-detector model=yolox_s``.

    just train-detector model=yolox_s          # or model=yolox_m, model=damoyolo_s
    just train-detector model=yolox_s simfarm=1   # first a few epochs on the sim farm
    just train-detector model=yolox_s init=<old>/best.pt out=<new folder>   # go on from
                                                  # earlier weights, in a new folder

Keys: ``model`` (yolox_s, yolox_m, damoyolo_s), ``epochs`` (default 20), ``batch``
(default 8, 4 for yolox_m: what fits an 8 GB graphics card with room to spare), ``sets`` (default
heridal,kios,wisard: the fetched ones that may be trained on, mixed), ``init`` (a
``best.pt`` of the same model to start from instead of the COCO weights; it must lie
outside ``out``; one of another size, say a yolox_s run for yolox_m, is refused: the
layers differ in width), ``simfarm`` (1: pre-train on the synthetic pictures first,
``simfarm_epochs`` of them, default 3), ``device`` (auto, cuda, cpu), ``data``, ``out``
(default ``<data>/train/<model>``), ``limit`` (only the first pictures: for trying it
out), ``workers`` (picture loaders, default 4), ``seed``,
``balance`` (``sets``, the default: each epoch draws as many tiles from every set, so a
big set does not drown the others; ``tiles``: every tile once, so a set counts by its
size; or weights, ``balance=heridal:2,kios:2,kios_carperson:1,seadronessee:1``, naming
every set), ``pick`` (the sets whose val AP50 picks the best epoch; default the sets in
``sets`` that are measured on, HERIDAL and KIOS, or all of them if none is),
``upscale`` (1, the default, or 2-4: each tile covers ``tile / upscale`` picture pixels
and is enlarged to ``tile``, so a small person is seen larger; the exported manifest
carries it, so the drone enlarges its tiles the same way, and a 12 MP still takes about
``upscale``² times as many tiles).

Stops and goes on: the state is saved to ``last.pt`` every few minutes and after every
epoch. Run the same command again and it carries on from there (a different command,
say another ``batch``, starts afresh in a new folder with ``fresh=1``). The best epoch
on the val part goes to ``best.pt``; ``just export-detector`` makes the ONNX file from it.
The val AP50 is printed per dataset; the best epoch is the one whose lowest AP50 over the
``pick`` sets is highest, so a gain on the set with the most people never hides a loss on
another.

Never trained on: the test part (locked in ``tools/bench/splits/``) and every set whose
licence allows measuring only (a GPL or unlicensed one in ``sets`` is refused). The
weights stay on this computer.
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
from tiles import MAX_UPSCALE, TILE_PX, Tile, build_cache, check_upscale, load_tile, pictures
from torch.utils.data import DataLoader, Dataset

KEYS = (
    "model", "epochs", "batch", "sets", "simfarm", "simfarm_epochs", "device", "data",
    "out", "limit", "workers", "seed", "fresh", "lr", "save_minutes", "tile", "init",
    "balance", "pick", "upscale",
)  # fmt: skip
SAVE_MINUTES = 10.0
BALANCE_MODES = ("sets", "tiles")


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
    balance: str = "sets"  # sets, tiles, or weights "heridal:2,kios:1"
    pick: tuple[str, ...] = ()  # the sets that pick the best epoch (): all
    upscale: int = 1  # each tile covers tile / upscale picture pixels, enlarged to tile

    def key(self) -> dict[str, Any]:
        body = json.loads(json.dumps(asdict(self)))  # as saved: lists, not tuples
        if not self.init:
            del body["init"]  # a run started before init= existed carries on
        # a run started before balance= and pick= existed did what these say: it carries on
        if self.balance == "tiles":
            del body["balance"]
        if not self.pick or set(self.pick) == set(self.sets):
            del body["pick"]
        if self.upscale == 1:
            del body["upscale"]  # a run started before upscale= carries on
        return body


def parse_balance(text: str, sets: Sequence[str]) -> dict[str, float] | None:
    """Each set's share of an epoch's tiles (``None``: ``tiles``, every tile once)."""
    if text == "tiles":
        return None
    if text == "sets":
        return dict.fromkeys(sets, 1.0)
    weights: dict[str, float] = {}
    for part in text.split(","):
        name, _, value = part.partition(":")
        try:
            weight = float(value)
        except ValueError:
            weight = -1.0
        if not (weight > 0 and math.isfinite(weight)):
            raise BenchError(
                f"balance= ska vara sets, tiles eller vikter som heridal:2,kios:1 (fick {part})",
                2,
            )
        weights[name] = weight
    if set(weights) != set(sets):
        raise BenchError(
            f"balance= ska ge en vikt till varje set i sets= ({', '.join(sets)}), inga andra.", 2
        )
    return weights


def epoch_order(sources: Sequence[str], balance: str, seed: int, epoch: int) -> list[int]:
    """The tiles of one epoch, in order (``sources``: each tile's set). As many as there
    are tiles, so an epoch has as many steps whatever the balance. ``tiles``: each once.
    Otherwise each set gets its share of them: a smaller set is gone through whole, as
    often as it fits, and the rest drawn at random; a bigger one is drawn from. The same
    seed and epoch give the same order, so a stopped run carries on where it was."""
    rng = random.Random(seed * 1000 + epoch)  # noqa: S311 - repeatable order
    if balance == "tiles":
        order = list(range(len(sources)))
        rng.shuffle(order)
        return order
    by_set: dict[str, list[int]] = {}
    for i, name in enumerate(sources):
        by_set.setdefault(name, []).append(i)
    names = sorted(by_set)
    if balance == "sets":
        weights = dict.fromkeys(names, 1.0)
    else:  # weights for sets with no tiles here (limit=) are left out
        given = parse_balance(balance, balance_sets(balance))
        assert given is not None  # noqa: S101 - not tiles
        weights = {n: given.get(n, 0.0) for n in names}
        if not any(weights.values()):
            raise BenchError("balance= ger ingen vikt till något set som har rutor.", 2)
    total, share = len(sources), sum(weights.values())
    exact = {n: total * weights[n] / share for n in names}
    quota = {n: math.floor(exact[n]) for n in names}
    # the tiles rounding left over go to the sets that lost the most to it
    for n in sorted(names, key=lambda n: (quota[n] - exact[n], n))[: total - sum(quota.values())]:
        quota[n] += 1
    order: list[int] = []
    for n in names:
        whole, rest = divmod(quota[n], len(by_set[n]))
        order.extend(by_set[n] * whole)
        order.extend(rng.sample(by_set[n], rest))
    rng.shuffle(order)
    return order


def balance_sets(balance: str) -> list[str]:
    """The sets weights ``heridal:2,kios:1`` name."""
    return [part.partition(":")[0] for part in balance.split(",")]


def set_of(tile: Tile) -> str:
    return tile.source.split(":", 1)[0]


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


def picked(per_set: dict[str, float], pick: Sequence[str] = ()) -> list[str]:
    """The sets of ``per_set`` that pick the best epoch: those in ``pick``, or all of
    them if none of those has val tiles (or ``pick`` is empty)."""
    return [s for s in per_set if s in pick] or list(per_set)


def val_score(per_set: dict[str, float], pick: Sequence[str] = ()) -> float:
    """The best epoch's measure: the lowest per-dataset AP50 of the ``pick`` sets, so a
    gain on one set never makes up for a loss on another (``pick-detector`` ranks models
    the same way)."""
    return min(per_set[s] for s in picked(per_set, pick)) if per_set else 0.0


def default_pick(sets: Sequence[str]) -> tuple[str, ...]:
    """The sets of ``sets`` that are measured on (the land sets), or all if none is."""
    from bench.datasets import load_register

    register = load_register()
    measured = tuple(s for s in sets if s in register and "measure" in register[s].use)
    return measured or tuple(sets)


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
    sources = [set_of(t) for t in tiles]
    # the simulator's pictures are one set: nothing to balance, so its order stays as it was
    balance = settings.balance if stage == "train" else "tiles"
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
    if stage == "train":
        drawn = [sources[i] for i in epoch_order(sources, balance, settings.seed, 0)]
        say(
            f"{stage}: rutor per epok (balance={balance}): "
            + ", ".join(f"{n} {drawn.count(n)} av {sources.count(n)}" for n in sorted(set(sources)))
            + f"; bästa epoken väljs på {', '.join(settings.pick) or 'alla'}"
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
        order = epoch_order(sources, balance, settings.seed, epoch)
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
            score = val_score(per_set, settings.pick)
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
                        "val_ap50_pick": picked(per_set, settings.pick),
                        "balance": settings.balance,
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


def _int_key(args: dict[str, str], key: str, default: int) -> int:
    raw = args.get(key, str(default))
    try:
        return int(raw)
    except ValueError:
        raise BenchError(f"{key}= ska vara ett heltal, inte {raw!r}", 2) from None


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
            batch=_int_key(args, "batch", families.FAMILIES[model].batch),
            sets=tuple(s for s in args.get("sets", "heridal,kios,wisard").split(",") if s),
            simfarm_epochs=int(args.get("simfarm_epochs", "3")) if simfarm else 0,
            lr=float(args.get("lr", "0.01")),
            seed=int(args.get("seed", "0")),
            limit=int(args.get("limit", "0")),
            tile=int(args.get("tile", str(TILE_PX))),
            init=str(Path(args["init"]).expanduser().resolve()) if args.get("init") else "",
            balance=args.get("balance", "sets"),
            pick=tuple(s for s in args.get("pick", "").split(",") if s),
            upscale=_int_key(args, "upscale", 1),
        )
        parse_balance(settings.balance, settings.sets)  # a typo stops here, not hours later
        stray = [s for s in settings.pick if s not in settings.sets]
        if stray:
            raise BenchError(f"pick= får bara nämna set i sets= (inte {', '.join(stray)}).", 2)
        if settings.tile % 32 or not 64 <= settings.tile <= 1280:
            raise BenchError("tile= ska vara delbart med 32 och mellan 64 och 1280", 2)
        try:
            span = check_upscale(settings.tile, settings.upscale)
        except ValueError:
            raise BenchError(
                f"upscale= ska vara 1-{MAX_UPSCALE} och dela tile= ({settings.tile})", 2
            ) from None
        if span < 32:
            raise BenchError(f"tile= / upscale= ska vara minst 32 bildpunkter, inte {span}", 2)
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
    as ``tiles.pictures`` leaves them out, so they are never credited as seen. A set no
    longer in the register stays, so the export refuses it rather than drop its credit."""
    from bench.datasets import load_register

    register = load_register()
    return [s for s in sets if s not in register or "train" in register[s].use]


def refuse_no_grant(sets: Sequence[str]) -> None:
    """A GPL or unlicensed set is refused outright, not left out quietly: asking to train
    on one is a mistake to see, and not even its val part may pick the best epoch."""
    from bench.datasets import load_register

    register = load_register()
    refused = [s for s in sets if s in register and register[s].no_grant]
    if refused:
        raise BenchError(
            f"{', '.join(refused)} får bara mätas på, aldrig tränas på (licensen ger ingen "
            "rätt att bygga vidare). Ta bort dem ur sets=.",
            2,
        )


def seen_by(ck: dict[str, Any]) -> tuple[list[str], list[str]]:
    """What a ``best.pt``'s weights have seen: (``trained_on``, ``pretrained_on``). A
    checkpoint from before those fields gets them from its settings: the sets that may be
    trained on, and the simulator's pictures when it had simfarm epochs."""
    settings = ck.get("settings", {})
    trained_on = ck["trained_on"] if "trained_on" in ck else trained_sets(settings.get("sets", []))
    pretrained_on = ck.get("pretrained_on", ["simfarm"] if settings.get("simfarm_epochs") else [])
    return list(trained_on), list(pretrained_on)


def load_init(net: torch.nn.Module, model: str, path: Path) -> dict[str, Any]:
    """Start from an earlier run's best weights (its EMA) instead of COCO's. Returns what
    those weights have seen (``trained_on``, ``pretrained_on``), so their credit stays,
    and ``started_from``: the file's name, SHA-256 and epoch (never its path or folder)."""
    if not path.is_file():
        raise BenchError(f"init={path} finns inte.", 2)
    ck: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    if "ema" not in ck:
        raise BenchError(f"init={path} är inte en best.pt för {model}.", 2)
    if ck.get("model") != model:
        raise BenchError(
            f"init={path} är tränad som {ck.get('model')}, inte {model}. De är olika "
            "nätverk (yolox_s och yolox_m har olika bredd) och kan inte ta varandras vikter. "
            f"Ta bort init= så startar {model} från COCO-vikterna.",
            2,
        )
    try:
        net.load_state_dict(ck["ema"])
    except RuntimeError:  # same name, other shapes: never half-loaded
        raise BenchError(f"init={path} passar inte nätverket {model}.", 2) from None
    trained_on, pretrained_on = seen_by(ck)
    seen: dict[str, Any] = {
        "trained_on": trained_on,
        "pretrained_on": pretrained_on,
        "started_from": {
            "file": path.name,
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
    refuse_no_grant(settings.sets)
    weights = data / "models" / "pretrained" / families.PRETRAINED[settings.model]
    if not weights.is_file():
        raise BenchError(
            f"COCO-vikterna saknas ({weights.name}). Kör: just detector-fetch yes=1", 2
        )
    sets = _available(data, settings.sets)
    if not sets:
        raise BenchError("Inga bilder att träna på. Kör: just bench-fetch yes=1", 2)
    # what is really trained on, and the sets among them that pick the best epoch
    pick = tuple(s for s in sets if s in settings.pick) or default_pick(sets)
    balance = settings.balance
    if balance not in BALANCE_MODES:  # weights: only for the sets really there
        weights = dict(zip(balance_sets(balance), balance.split(","), strict=True))
        balance = ",".join(weights[s] for s in sets)
    settings = replace(settings, sets=tuple(sets), pick=pick, balance=balance)
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
                build_cache(
                    items, out / "tiles" / "simfarm", settings.tile, upscale=settings.upscale
                ),
            )
            val = None
        else:
            items = pictures(data, "train", sets, settings.limit)
            train_tiles = (
                out / "tiles" / "train",
                build_cache(
                    items, out / "tiles" / "train", settings.tile, upscale=settings.upscale
                ),
            )
            val_items = pictures(data, "val", sets, settings.limit)
            val = (
                out / "tiles" / "val",
                build_cache(
                    val_items,
                    out / "tiles" / "val",
                    settings.tile,
                    empty_share=0.1,
                    upscale=settings.upscale,
                ),
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
