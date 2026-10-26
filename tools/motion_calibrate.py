"""Calibrate the motion model against where lost people were found (`just lpb-calibrate`).

Lie & Skaar (2016) give the straight-line distance from the last seen point to where
people were found, per category (25/50/75 %). This tool fits two numbers per category in
``sd_motion/motion.toml``: a scale on the turning rates (how straight people walk) and a
scale on the mobility hours (how long they walk), and writes them back with the fit.

The ground is flat forest, 30 x 30 km of 50 m cells, with a track every 500 m both ways;
the walk starts on a track halfway between two crossings. Stopping is the same in every
cell here, so where people stop is a mix of where the walkers are at each moment,
weighted by how many stop then. One walk without stopping per turning scale therefore
serves every mobility scale: the walk is the slow part, the mix is cheap.

Running it again starts from the written values, so a second run moves them little.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np

from sd_motion import MotionState, Params, Terrain, advance
from sd_motion.params import load_all

TOML = Path(__file__).resolve().parents[1] / "software/common/motion/src/sd_motion/motion.toml"

# Lie & Skaar (2016), found distance in km at 25/50/75 %.
TARGETS_KM = {
    "dementia": (0.27, 0.80, 3.15),
    "despondent": (0.37, 0.98, 2.50),
    "hiker": (0.64, 1.71, 3.35),
    "gatherer": (1.26, 1.69, 2.78),
}
TOLERANCE = 0.25
TURN_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
MOBILITY_SCALES = tuple(float(x) for x in np.geomspace(0.125, 8.0, 49))
SNAPSHOT_S = 600.0
GONE = 1e-3  # stop walking when this little is still on the map


@dataclass(frozen=True)
class World:
    size_km: float = 30.0
    cell_m: float = 50.0
    track_every_m: float = 500.0
    hours: float = 48.0
    dt_s: float = 30.0

    def build(self) -> tuple[Terrain, tuple[int, int], np.ndarray]:
        """The terrain, the start cell and each cell's distance from it (km)."""
        n = round(self.size_km * 1000 / self.cell_m)
        every = round(self.track_every_m / self.cell_m)
        track = np.zeros((n, n), bool)
        track[::every, :] = True
        track[:, ::every] = True
        terrain = Terrain.from_cover(np.ones((n, n), bool), track, np.zeros((n, n), bool),
                                     self.cell_m)  # fmt: skip
        mid = (n // 2) // every * every
        start = (mid, mid + every // 2)
        rr, cc = np.indices((n, n))
        dist_km = np.hypot(rr - start[0], cc - start[1]) * self.cell_m / 1000
        return terrain, start, dist_km


@dataclass(frozen=True)
class Walk:
    """Where the walkers are over time when nobody stops."""

    times_h: np.ndarray  # (T,)
    shares: np.ndarray  # (T, B + 1): share of the walkers per distance bin, last = off the map
    bin_km: float


def walk(world: World, params: Params, turn_scale: float) -> Walk:
    terrain, start, dist_km = world.build()
    bin_km = world.cell_m / 1000
    bins = (dist_km / bin_km).astype(np.int64).ravel()
    nbins = int(bins.max()) + 1
    walker = replace(
        params,
        turn45=params.turn45 * turn_scale,
        turn_any=params.turn_any * turn_scale,
        mobility=((0.0, 1.0), (1e6, 1.0)),
    )
    shape = terrain.shape
    state = MotionState(np.zeros(shape), np.zeros((8, *shape)), 0.0, 0.0)
    state.moving[:, start[0], start[1]] = 1.0 / 8.0
    times, shares = [], []
    t_h = 0.0
    while True:
        on_map = np.bincount(bins, weights=state.moving.sum(axis=0).ravel(), minlength=nbins)
        times.append(t_h)
        shares.append(np.append(on_map, state.row))
        if t_h >= world.hours or on_map.sum() < GONE:
            break
        state = advance(state, terrain, walker, SNAPSHOT_S, dt_s=world.dt_s)
        t_h += SNAPSHOT_S / 3600
    return Walk(np.array(times), np.array(shares), bin_km)


def scale_mobility(params: Params, scale: float) -> Params:
    return replace(params, mobility=tuple((h * scale, s) for h, s in params.mobility))


def found_quartiles(w: Walk, params: Params) -> tuple[float, float, float]:
    """Distances (km) within which 25/50/75 % are found, for this mobility curve.
    People still walking at the end of the walk count where the walkers were then."""
    s = np.array([params.still_moving(t) for t in w.times_h])
    stop_now = np.append(s[:-1] - s[1:], s[-1])
    mid = np.vstack([(w.shares[:-1] + w.shares[1:]) / 2, w.shares[-1:]])
    found = stop_now @ mid
    found[0] += 1.0 - s[0]  # never moved
    cdf = np.cumsum(found) / found.sum()
    edges = np.arange(1, len(cdf) + 1) * w.bin_km  # bin i ends at (i + 1) bins; last: off map
    edges[-1] = math.inf

    def at(q: float) -> float:
        i = int(np.searchsorted(cdf, q))
        if i >= len(cdf) - 1:
            return math.inf
        lo = cdf[i - 1] if i else 0.0
        frac = (q - lo) / (cdf[i] - lo) if cdf[i] > lo else 0.0
        return float(edges[i] - w.bin_km * (1.0 - frac))

    return at(0.25), at(0.5), at(0.75)


def error(got: tuple[float, ...], target: tuple[float, ...]) -> float:
    return sum(
        ((g - t) / t) ** 2 if math.isfinite(g) else 1e6 for g, t in zip(got, target, strict=True)
    )


@dataclass(frozen=True)
class Fit:
    category: str
    turn_scale: float
    mobility_scale: float
    quartiles_km: tuple[float, float, float]

    @property
    def misses(self) -> list[str]:
        target = TARGETS_KM[self.category]
        return [
            f"{pct} % ({got:.2f} km, target {t:.2f})"
            for pct, got, t in zip((25, 50, 75), self.quartiles_km, target, strict=True)
            if not abs(got - t) <= TOLERANCE * t
        ]


def _walk_job(args: tuple[World, str, Params, float]) -> tuple[str, float, Walk]:
    world, category, params, turn_scale = args
    return category, turn_scale, walk(world, params, turn_scale)


def calibrate(world: World, table: dict[str, Params], workers: int) -> list[Fit]:
    jobs = [(world, c, table[c], ts) for c in TARGETS_KM for ts in TURN_SCALES]
    best: dict[str, tuple[float, Fit]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for category, turn_scale, w in pool.map(_walk_job, jobs):
            for ms in MOBILITY_SCALES:
                q = found_quartiles(w, scale_mobility(table[category], ms))
                e = error(q, TARGETS_KM[category])
                if category not in best or e < best[category][0]:
                    best[category] = (e, Fit(category, turn_scale, ms, q))
    return [best[c][1] for c in TARGETS_KM]


def _fmt(x: float) -> str:
    return f"{x:.4g}"


def write_back(text: str, table: dict[str, Params], fits: list[Fit], today: date) -> str:
    """The toml text with each fitted row's turning rates, mobility hours and a
    ``calibrated`` line replaced in place, comments kept."""
    for fit in fits:
        p = table[fit.category]
        mobility = ", ".join(f"[{_fmt(h * fit.mobility_scale)}, {_fmt(s)}]" for h, s in p.mobility)
        got = "/".join(f"{q:.2f}" for q in fit.quartiles_km)
        want = "/".join(f"{q:.2f}" for q in TARGETS_KM[fit.category])
        verdict = "all within 25 %" if not fit.misses else "misses " + "; ".join(fit.misses)
        values = {
            "mobility": f"[{mobility}]",
            "turn45": _fmt(p.turn45 * fit.turn_scale),
            "turn_any": _fmt(p.turn_any * fit.turn_scale),
            "calibrated": f'"{today.isoformat()}: found within {got} km at 25/50/75 % '
            f'(Lie & Skaar: {want}); {verdict}"',
        }
        text = _set_keys(text, fit.category, values)
    return text


def _set_keys(text: str, section: str, values: dict[str, str]) -> str:
    lines = text.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.strip() == f"[{section}]")
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith("[")),
        len(lines),
    )
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    missing = dict(values)
    for i in range(start + 1, end):
        m = re.match(r"(\w+)\s*=", lines[i])
        if m and m.group(1) in missing:
            lines[i] = f"{m.group(1)} = {missing.pop(m.group(1))}\n"
    extra = [f"{k} = {v}\n" for k, v in missing.items()]
    return "".join(lines[:end] + extra + lines[end:])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit motion.toml to Lie & Skaar's distances.")
    ap.add_argument("--toml", type=Path, default=TOML)
    ap.add_argument("--size-km", type=float, default=World.size_km)
    ap.add_argument("--cell-m", type=float, default=World.cell_m)
    ap.add_argument("--hours", type=float, default=World.hours)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="print the fit, write nothing")
    args = ap.parse_args(argv)
    world = World(size_km=args.size_km, cell_m=args.cell_m, hours=args.hours)
    table = load_all(args.toml.read_text(encoding="utf-8"))
    fits = calibrate(world, table, args.workers)
    for f in fits:
        q = "/".join(f"{x:.2f}" for x in f.quartiles_km)
        t = "/".join(f"{x:.2f}" for x in TARGETS_KM[f.category])
        status = "ok" if not f.misses else "MISSES " + "; ".join(f.misses)
        print(f"{f.category:11s} turns x{f.turn_scale:g} hours x{f.mobility_scale:.2f}: "
              f"{q} km (target {t}) {status}")  # fmt: skip
    if not args.dry_run:
        text = args.toml.read_text(encoding="utf-8")
        args.toml.write_text(write_back(text, table, fits, date.today()), encoding="utf-8")  # noqa: DTZ011
        print(f"wrote {args.toml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
