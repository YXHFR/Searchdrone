"""The search scenarios' measures: one run's ``search_metrics.json`` and
``search_report.md``, and the nightly campaign's requirements over many runs.

A run's measures (``RunMetrics``) come from the scenario: what the scripted operator
saw and decided, the ground's rows (candidates, detections, the stills' times), the
probability map's sectors at the end and the cylinders the drone held. ``write_run``
puts them in the run's folder; ``search_report.md`` is a short Swedish summary (time to
find, POS, the queue, the engines). Neither ever holds the search profile's text: only
field keys (``garment:0``) and their verdicts.

``aggregate`` reads every ``search_metrics.json`` under a folder and checks the nightly
requirements per scenario (SK01, SK02, SK03, SK04, SK06 and, for every run, the latency
from still to queue and the cylinders). The detector and the image model are oracles
with modelled start profiles, so every result is labelled "modellerat, ej uppmätt".

``python -m sd_sim.eval_search aggregate <runs> <out>`` writes ``search_requirements.md``
and ``search_requirements.json`` in ``<out>`` and exits 1 when a requirement fails.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

__all__ = [
    "LABEL",
    "Requirement",
    "RunMetrics",
    "aggregate",
    "main",
    "wilson_lower",
    "write_run",
]

LABEL: Final = "modellerat, ej uppmätt"
METRICS = "search_metrics.json"
REPORT = "search_report.md"
Z95_ONE_SIDED: Final = 1.6448536
Z95_TWO_SIDED: Final = 1.959964
LATENCY_P50_MAX_MS: Final = 3000.0
SK04_MIN_SIGHTINGS: Final = 10  # verdicts that saw the hiker's top, over all runs
SK06_FACTOR: Final = 0.9
TOO_FEW: Final = "för få data"


@dataclass
class RunMetrics:
    """The measures of one search run (``search_metrics.json``)."""

    scenario: str
    seed: int
    label: str = LABEL
    detector: str = ""  # the detector oracle's profile id
    matcher: str = ""  # the image-model oracle's profile id, "" when off
    with_map: bool | None = None  # SK06's pairing: the run with or without the map
    limit_s: float = 900.0
    has_target: bool = True
    time_to_confirmed_s: float | None = None  # take-off -> the target's confirmation
    found: bool = False  # confirmed within limit_s
    false_confirms: int = 0
    geo_error_m: float | None = None
    queue_rank_target: int | None = None  # its place when it was first ready (1 = top)
    target_ever_low: bool = False
    pos_end: float | None = None  # sum of POA x mean POD over the sectors at the end
    battery_used_pct: float | None = None
    pos_per_batt_pct: float | None = None
    false_candidates_per_hour: dict[str, float] = field(default_factory=dict[str, float])
    latency_still_to_queue_ms: dict[str, float | None] = field(
        default_factory=lambda: {"p50": None, "p95": None}
    )
    keepout_min_distance_m: float | None = None
    keepouts: int = 0
    cloud_cost_sek: float = 0.0
    candidates: int = 0
    decisions: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    fields_by_thing: dict[str, dict[str, str]] = field(default_factory=dict[str, dict[str, str]])
    # The upper garment's verdict (``garment:0``) of every card of a thing that is not
    # merged, so the campaign counts each time the image model judged a person's top.
    upper_by_thing: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
    confirmed_thing: str | None = None
    engines: list[str] = field(default_factory=list[str])
    queue_end: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    flight_s: float | None = None
    invariants_broken: list[str] = field(default_factory=list[str])

    def finish(self) -> RunMetrics:
        """The derived measures (``found``, POS per battery percent)."""
        t = self.time_to_confirmed_s
        self.found = self.has_target and t is not None and t <= self.limit_s
        if self.pos_end is not None and self.battery_used_pct:
            self.pos_per_batt_pct = round(self.pos_end / self.battery_used_pct, 5)
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def wilson_lower(successes: int, n: int, z: float = Z95_TWO_SIDED) -> float:
    """The Wilson score interval's lower bound for a share."""
    if n == 0:
        return 0.0
    p = successes / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / (1 + z * z / n)


def _fmt(value: float | None, unit: str = "", digits: int = 0) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}{unit}".replace(".", ",")


def report_md(m: RunMetrics) -> str:
    """``search_report.md``: a short Swedish summary of one run."""
    lines = [
        f"# Sökrapport {m.scenario}, frö {m.seed}",
        "",
        f"*{m.label}: detektorn är oraklet `{m.detector}`"
        + (f", bildmodellen oraklet `{m.matcher}`" if m.matcher else ", ingen bildmodell")
        + ".*",
        "",
        "| Mått | Värde |",
        "|---|---|",
    ]
    if m.has_target:
        found = "ja" if m.found else "nej"
        lines.append(f"| Hittad inom {_fmt(m.limit_s / 60)} min | {found} |")
        lines.append(f"| Tid lyft → bekräftelse | {_fmt(m.time_to_confirmed_s, ' s')} |")
        lines.append(f"| Lägesfel bekräftat fynd | {_fmt(m.geo_error_m, ' m', 1)} |")
        lines.append(f"| Facitkandidatens plats i kön | {m.queue_rank_target or '-'} |")
    else:
        lines.append("| Ingen person i världen | - |")
    lines += [
        f"| Felaktiga bekräftelser | {m.false_confirms} |",
        f"| POS vid slutet | {_fmt(m.pos_end, '', 3)} |",
        f"| POS per batteriprocent | {_fmt(m.pos_per_batt_pct, '', 4)} |",
        f"| Kandidater | {m.candidates} |",
        f"| Stillbild → kö p50 / p95 | {_fmt(m.latency_still_to_queue_ms.get('p50'), ' ms')}"
        f" / {_fmt(m.latency_still_to_queue_ms.get('p95'), ' ms')} |",
        f"| Cylindrar / minsta avstånd till kant | {m.keepouts} / "
        f"{_fmt(m.keepout_min_distance_m, ' m', 1)} |",
        f"| Motorer | {', '.join(m.engines) or '-'} |",
        f"| Molnkostnad | {_fmt(m.cloud_cost_sek, ' kr', 2)} |",
        "",
        "## Kön vid slutet",
        "",
        "| K | Läge | Prioritet | Klass | Motor | Facit |",
        "|---|---|---|---|---|---|",
    ]
    for c in m.queue_end[:10]:
        lines.append(
            f"| K{c['seq']} | {c['state']} | {_fmt(c['priority'], '', 2)} | "
            f"{c['priority_class']} | {c.get('engine') or '-'} | {c.get('truth') or '-'} |"
        )
    if m.decisions:
        lines += ["", "## Operatörens beslut", ""]
        for d in m.decisions:
            lines.append(f"- {_fmt(d['t_s'])} s: K{d['seq']} {d['decision']} ({d['truth']})")
    if m.invariants_broken:
        lines += ["", "## Brutna invarianter", ""] + [f"- {b}" for b in m.invariants_broken]
    return "\n".join(lines) + "\n"


def write_run(run_dir: Path, m: RunMetrics) -> Path:
    """``search_metrics.json`` and ``search_report.md`` in ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / METRICS).write_text(
        json.dumps(m.as_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    path = run_dir / REPORT
    path.write_text(report_md(m), encoding="utf-8")
    return path


# --- the nightly campaign ---------------------------------------------------------------


@dataclass
class Requirement:
    scenario: str
    text: str
    value: str
    ok: bool
    n: int
    too_few: bool = False  # too few data to judge: never ok


def _load(root: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(root.rglob(METRICS)):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return runs


def _share(runs: Sequence[Mapping[str, Any]], pred: Callable[[Mapping[str, Any]], bool]) -> float:
    return sum(1 for r in runs if pred(r)) / len(runs) if runs else 0.0


def _hit(runs: Sequence[Mapping[str, Any]], k: int) -> float:
    return _share(runs, lambda r: r.get("queue_rank_target") is not None
                  and r["queue_rank_target"] <= k)  # fmt: skip


def _req(scenario: str, text: str, value: str, ok: bool, n: int) -> Requirement:
    return Requirement(scenario, text, value, ok, n)


def _sk01(runs: list[dict[str, Any]]) -> list[Requirement]:
    n = len(runs)
    k = sum(1 for r in runs if r["found"])
    errs = [r["geo_error_m"] for r in runs if r.get("geo_error_m") is not None]
    med = statistics.median(errs) if errs else None
    hit1 = _hit(runs, 1)
    return [
        _req("SK01", "hittad inom 15 min: andel ≥ 0,95 och Wilson-nedre ≥ 0,88",
             f"{k}/{n}, nedre {wilson_lower(k, n):.2f}",
             n > 0 and k / n >= 0.95 and wilson_lower(k, n) >= 0.88, n),
        _req("SK01", "lägesfel median < 10 m", _fmt(med, " m", 1), med is not None and med < 10, n),
        _req("SK01", "kön Hit@1 ≥ 0,80", f"{hit1:.2f}", hit1 >= 0.80, n),
    ]  # fmt: skip


def _sk02(runs: list[dict[str, Any]]) -> list[Requirement]:
    n = len(runs)
    found = _share(runs, lambda r: r["found"])
    hit5 = _hit(runs, 5)
    never_low = sum(1 for r in runs if not r.get("target_ever_low"))
    return [
        _req("SK02", "hittad inom 20 min: andel ≥ 0,85", f"{found:.2f}", found >= 0.85, n),
        _req("SK02", "kön Hit@5 ≥ 0,90", f"{hit5:.2f}", hit5 >= 0.90, n),
        _req("SK02", "facitpersonen aldrig i låg-listan: andel ≥ 0,95 (Wilson-nedre visas)",
             f"{never_low}/{n} = {never_low / n if n else 0:.2f}, "
             f"Wilson-nedre {wilson_lower(never_low, n):.2f}",
             n > 0 and never_low / n >= 0.95, n),
    ]  # fmt: skip


def _sk03(runs: list[dict[str, Any]]) -> list[Requirement]:
    n = len(runs)
    confirms = sum(int(r["false_confirms"]) for r in runs)
    e2 = [b for r in runs for b in r.get("invariants_broken", []) if b.startswith("E2")]
    return [
        _req(
            "SK03", "ingen bekräftelse i någon körning", str(confirms), n > 0 and confirms == 0, n
        ),
        _req("SK03", "alla kandidater kvar i databasen (E2)", str(len(e2)), not e2, n),
    ]


def _sk04(runs: list[dict[str, Any]]) -> list[Requirement]:
    n = len(runs)
    first = _share(
        runs, lambda r: r.get("confirmed_thing") == "target" and r["false_confirms"] == 0
    )
    tops = [v for r in runs for v in (r.get("upper_by_thing") or {}).get("green-jacket", [])]
    seen = [v for v in tops if v in ("conflict", "match")]
    conflict = sum(v == "conflict" for v in seen)
    text = (
        f"vandraren: conflict / (conflict + match) på överdel ≥ 0,80, "
        f"minst {SK04_MIN_SIGHTINGS} sedda"
    )
    if len(seen) < SK04_MIN_SIGHTINGS:
        hiker = Requirement(
            "SK04", text, f"{len(seen)} sedda av {len(tops)} kort", False, n, too_few=True
        )
    else:
        share = conflict / len(seen)
        hiker = _req(
            "SK04",
            text,
            f"{conflict}/{len(seen)} = {share:.2f} ({len(tops)} kort)",
            share >= 0.80,
            n,
        )
    return [
        _req("SK04", "rätt person bekräftad före vandraren i ≥ 90 %", f"{first:.2f}",
             first >= 0.90, n),
        hiker,
    ]  # fmt: skip


def _sk06(runs: list[dict[str, Any]]) -> list[Requirement]:
    """Paired seeds: the time with the map against ``SK06_FACTOR`` x the time without (a run
    that did not find the person counts its limit), one-sided 95 % upper bound of the mean
    below 0."""
    pairs: dict[int, dict[bool, float]] = {}
    for r in runs:
        if r.get("with_map") is None:
            continue
        t = r["time_to_confirmed_s"] if r["found"] else r["limit_s"]
        pairs.setdefault(int(r["seed"]), {})[bool(r["with_map"])] = float(t)
    diffs = [p[True] - SK06_FACTOR * p[False] for p in pairs.values() if len(p) == 2]
    n = len(diffs)
    text = f"tid med karta ≤ {_fmt(SK06_FACTOR, '', 1)} x utan (parat, ensidigt 95 %)"
    if n < 2:
        return [_req("SK06", text, f"{n} par", False, n)]
    mean = statistics.fmean(diffs)
    upper = mean + Z95_ONE_SIDED * statistics.stdev(diffs) / math.sqrt(n)
    return [
        _req("SK06", text, f"medel {mean:+.0f} s, övre {upper:+.0f} s ({n} par)", upper < 0, n)
    ]  # fmt: skip


SCENARIO_RULES: Final[dict[str, Callable[[list[dict[str, Any]]], list[Requirement]]]] = {
    "SK01": _sk01,
    "SK02": _sk02,
    "SK03": _sk03,
    "SK04": _sk04,
    "SK06": _sk06,
}


def _all(runs: list[dict[str, Any]]) -> list[Requirement]:
    p50s = [r["latency_still_to_queue_ms"]["p50"] for r in runs
            if r.get("latency_still_to_queue_ms", {}).get("p50") is not None]  # fmt: skip
    worst = max(p50s) if p50s else None
    dists = [r["keepout_min_distance_m"] for r in runs
             if r.get("keepout_min_distance_m") is not None]  # fmt: skip
    nearest = min(dists) if dists else None
    e8 = [b for r in runs for b in r.get("invariants_broken", []) if b.startswith("E8")]
    return [
        _req("alla", "stillbild → kö p50 < 3 000 ms i varje körning", _fmt(worst, " ms"),
             worst is None or worst < LATENCY_P50_MAX_MS, len(p50s)),
        _req("alla", "ingen bana i en cylinder (minsta avstånd ≥ 0)",
             f"{_fmt(nearest, ' m', 1)}, {len(e8)} brott", not e8, len(dists)),
    ]  # fmt: skip


def aggregate(runs: Iterable[Mapping[str, Any]]) -> list[Requirement]:
    by: dict[str, list[dict[str, Any]]] = {}
    every = [dict(r) for r in runs]
    for r in every:
        by.setdefault(str(r["scenario"]), []).append(r)
    out: list[Requirement] = []
    for scenario, rule in SCENARIO_RULES.items():
        if scenario in by:
            out += rule(by[scenario])
    return out + _all(every)


def requirements_md(reqs: Sequence[Requirement], n_runs: int) -> str:
    lines = [
        "# Sökkedjan, nattliga krav",
        "",
        f"*{LABEL}: detektorn och bildmodellen är orakel med modellerade startprofiler. "
        f"{n_runs} körningar.*",
        "",
        "| Scenario | Krav | Värde | Körningar | Resultat |",
        "|---|---|---|---|---|",
    ]
    for r in reqs:
        result = TOO_FEW if r.too_few else "ok" if r.ok else "EJ UPPFYLLT"
        lines.append(f"| {r.scenario} | {r.text} | {r.value} | {r.n} | {result} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sd_sim.eval_search")
    sub = parser.add_subparsers(dest="cmd", required=True)
    agg = sub.add_parser("aggregate", help="kontrollera kampanjens krav")
    agg.add_argument("runs", type=Path)
    agg.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    runs = _load(args.runs)
    if not runs:
        sys.stderr.write(f"inga {METRICS} under {args.runs}\n")
        return 1
    reqs = aggregate(runs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "search_requirements.md").write_text(
        requirements_md(reqs, len(runs)), encoding="utf-8"
    )
    (args.out / "search_requirements.json").write_text(
        json.dumps({"label": LABEL, "runs": len(runs),
                    "requirements": [asdict(r) for r in reqs]}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )  # fmt: skip
    failed = [r for r in reqs if not r.ok]
    for r in failed:
        verdict = TOO_FEW.upper() if r.too_few else "EJ UPPFYLLT"
        sys.stderr.write(f"{verdict} {r.scenario}: {r.text} ({r.value})\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
