# pyright: strict
"""Build the static status page: what the drone can do on ``main`` right now.

Usage::

    python tools/status_page/build.py results=<dir> out=<dir> sha=<commit> date=<text>
        [run_url=<url>] [repo_url=<url>] [history=<file>] [film=<dir>]

* ``results``: a folder with the simulator tests' ``result.json`` files, at any depth
  (the downloaded ``sitl-results-*`` artifacts of one run of the ``sitl`` workflow).
* ``out``: where ``index.html``, ``history.json`` and the film go (created; must be empty).
* ``sha``, ``date``, ``run_url``: the commit, the time and the workflow run the results
  belong to. ``repo_url`` links the commits (default the public repository).
* ``history``: the previous page's ``history.json``, if any. The new run goes first and
  the 20 newest runs are kept, one per commit.
* ``film``: a folder with ``demo.mp4`` (H.264) and/or ``demo.webm`` (VP9); copied as is.

One row per capability in ``capabilities.toml``: green when every result of its test
passed (and the measured value was within its limit), red when one did not, grey when
the run has no result for it. The page holds no time of its own: the same input gives
the same bytes.
"""

from __future__ import annotations

import html
import json
import re
import shutil
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

HERE = Path(__file__).resolve().parent
CAPABILITIES = HERE / "capabilities.toml"
DEFAULT_REPO_URL = "https://github.com/YXHFR/Searchdrone"
HISTORY_LEN = 20
FILMS = (("demo.mp4", "video/mp4"), ("demo.webm", "video/webm"))
KEYS = ("results", "out", "sha", "date", "run_url", "repo_url", "history", "film")
REQUIRED = ("results", "out", "sha", "date")
STATUSES = ("pass", "fail", "none")
STATUS_TEXT = {"pass": "Grön", "fail": "Röd", "none": "Ingen data"}
DASH = "\u2013"  # shown where a value is missing
STATUS_ICON = {"pass": "✓", "fail": "✗", "none": DASH}
_SHA = re.compile(r"^[0-9a-f]{7,40}$")


@dataclass(frozen=True)
class Capability:
    group: str
    name: str
    variant: str
    scenario: str | None = None
    measure: str | None = None


@dataclass(frozen=True)
class Row:
    cap: Capability
    status: str
    expected: str
    measured: str
    scenario: str
    when: str
    sha: str | None
    note: str


def load_capabilities(path: Path = CAPABILITIES) -> tuple[list[str], list[Capability]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    groups = [str(g) for g in data["groups"]]
    caps = [
        Capability(
            group=str(c["group"]),
            name=str(c["name"]),
            variant=str(c["variant"]),
            scenario=c.get("scenario"),
            measure=c.get("measure"),
        )
        for c in cast(list[dict[str, Any]], data["capability"])
    ]
    unknown = sorted({c.group for c in caps} - set(groups))
    if unknown:
        raise ValueError(f"okända grupper i {path.name}: {unknown}")
    names = [c.name for c in caps]
    if len(names) != len(set(names)):
        raise ValueError(f"samma namn två gånger i {path.name}")
    return groups, caps


def load_results(folder: Path) -> list[dict[str, Any]]:
    """Every readable ``result.json`` with a variant, oldest first."""
    out: list[dict[str, Any]] = []
    for path in sorted(folder.rglob("result.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            res = cast(dict[str, Any], data)
            if isinstance(res.get("variant"), str):
                out.append(res)
    out.sort(key=lambda r: (str(r.get("started_utc", "")), str(r.get("variant"))))
    return out


def number(value: float) -> str:
    """Swedish decimal comma, at most two decimals, no trailing zeros."""
    digits = 2 if abs(value) < 10 else 1 if abs(value) < 100 else 0
    text = f"{value:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace(".", ",").replace("-", "\u2212")  # a real minus sign


def with_unit(value: float, unit: str) -> str:
    if unit == "%":
        return f"{number(value)} %"
    return f"{number(value)} {unit}".strip()


def is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def criteria(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw: object = result.get("criteria")
    items = cast(list[object], raw) if isinstance(raw, list) else []
    return [cast(dict[str, Any], c) for c in items if isinstance(c, dict)]


def criterion(result: dict[str, Any], text: str) -> dict[str, Any] | None:
    return next((c for c in criteria(result) if c.get("text") == text), None)


def expected_of(c: dict[str, Any]) -> str:
    limit, unit = c.get("limit"), str(c.get("unit") or "")
    if not is_number(limit):
        return "inom gränsen"
    word = "minst" if c.get("bound") == "min" else "högst"
    return f"{word} {with_unit(cast(float, limit), unit)}"


def matches(cap: Capability, result: dict[str, Any]) -> bool:
    if str(result["variant"]).split("[", 1)[0] != cap.variant:
        return False
    return cap.scenario is None or result.get("scenario") == cap.scenario


def failed_how(result: dict[str, Any]) -> str:
    for c in criteria(result):
        if c.get("id") == "T" and isinstance(c.get("measured"), str):
            return str(c["measured"])
    return str(result.get("error") or result.get("infra_failure") or "underkänt")


def evaluate(cap: Capability, results: list[dict[str, Any]]) -> Row:
    mine = [r for r in results if matches(cap, r)]
    if not mine:
        return Row(cap, "none", DASH, DASH, DASH, DASH, None, "Ingen körning i den här omgången.")
    passed = [r for r in mine if r.get("passed") is True]
    latest = mine[-1]
    status = "pass" if len(passed) == len(mine) else "fail"
    note = f"{len(passed)} av {len(mine)} körningar gröna." if len(mine) > 1 else ""
    if cap.measure is None:
        expected = "alla kontroller i testet godkända"
        measured = "godkända" if latest.get("passed") is True else failed_how(latest)
    else:
        with_value = [r for r in mine if criterion(r, cap.measure) is not None]
        shown = with_value[-1] if with_value else None
        c = criterion(shown, cap.measure) if shown else None
        if c is None:
            expected = DASH
            measured = "testet nådde inte mätningen"
        else:
            expected = expected_of(c)
            value = c.get("measured")
            measured = (
                with_unit(cast(float, value), str(c.get("unit") or ""))
                if is_number(value)
                else DASH
            )
            if c.get("passed") is False:
                status = "fail"
    versions: object = latest.get("versions")
    sha = cast(dict[str, Any], versions).get("git_sha") if isinstance(versions, dict) else None
    started = str(latest.get("started_utc") or "")
    when = started.replace("T", " ").removesuffix("Z")[:16] + " UTC" if started else DASH
    return Row(
        cap,
        status,
        expected,
        measured,
        str(latest.get("scenario") or DASH),
        when,
        sha if isinstance(sha, str) and _SHA.match(sha) else None,
        note,
    )


def clean_history(raw: object, names: set[str]) -> list[dict[str, Any]]:
    """Only well-formed entries of an earlier page's history (it comes over the web)."""
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for entry in cast(list[object], raw):
        if not isinstance(entry, dict):
            continue
        e = cast(dict[str, Any], entry)
        sha, date, url, status = e.get("sha"), e.get("date"), e.get("run_url"), e.get("status")
        if not (isinstance(sha, str) and _SHA.match(sha) and isinstance(date, str)):
            continue
        if not isinstance(status, dict):
            continue
        st = {
            str(k): str(v)
            for k, v in cast(dict[object, object], status).items()
            if k in names and v in STATUSES
        }
        good_url = isinstance(url, str) and url.startswith("https://github.com/")
        out.append(
            {"sha": sha, "date": date[:40], "run_url": url if good_url else "", "status": st}
        )
    return out


def next_history(previous: list[dict[str, Any]], entry: dict[str, Any]) -> list[dict[str, Any]]:
    """The new run first, one entry per commit, the newest ``HISTORY_LEN``."""
    rest = [e for e in previous if e["sha"] != entry["sha"]]
    return [entry, *rest][:HISTORY_LEN]


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--text:#16181d;--muted:#5b6170;--line:#dde1e8;
--pass:#1a7f37;--pass-bg:#dafbe1;--fail:#c4202b;--fail-bg:#ffe3e3;--none:#6e7581;
--none-bg:#eceef2;--link:#0a58ca}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#181b21;--text:#e8eaef;
--muted:#a1a7b3;--line:#2b303a;--pass:#4ac26b;--pass-bg:#11301b;--fail:#ff6b6b;
--fail-bg:#3a1416;--none:#8b93a1;--none-bg:#242832;--link:#7cb3ff}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:16px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--link)}
main{max-width:960px;margin:0 auto;padding:16px}
h1{font-size:1.5rem;margin:.25rem 0}
h2{font-size:1.15rem;margin:1.75rem 0 .5rem}
.sub{color:var(--muted);margin:0 0 1rem}
.summary{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 1rem;padding:0;list-style:none}
.summary li{padding:6px 12px;border-radius:999px;font-weight:600}
.film{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.film video{display:block;width:100%;height:auto;border-radius:8px;background:#000}
.film p{margin:.5rem 0 0;color:var(--muted);font-size:.9rem}
.caps{list-style:none;margin:0;padding:0;display:grid;gap:8px}
.cap{background:var(--card);border:1px solid var(--line);border-left:6px solid var(--none);
border-radius:10px;padding:10px 12px}
.cap.pass{border-left-color:var(--pass)}.cap.fail{border-left-color:var(--fail)}
.top{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}
.name{font-weight:600}
.pill{flex:none;font-size:.85rem;font-weight:700;padding:2px 10px;border-radius:999px}
.pass .pill,.summary .pass{color:var(--pass);background:var(--pass-bg)}
.fail .pill,.summary .fail{color:var(--fail);background:var(--fail-bg)}
.none .pill,.summary .none{color:var(--none);background:var(--none-bg)}
dl{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;margin:8px 0 0;font-size:.92rem}
dt{color:var(--muted)}dd{margin:0;overflow-wrap:anywhere}
.note{margin:6px 0 0;font-size:.85rem;color:var(--muted)}
.hist{display:flex;gap:3px;margin-top:8px;flex-wrap:wrap}
.hist span{width:14px;height:14px;border-radius:3px;background:var(--none-bg);
border:1px solid var(--line)}
.hist .pass{background:var(--pass)}.hist .fail{background:var(--fail)}
.hist .none{background:var(--none-bg)}
footer{margin:2rem 0 1rem;color:var(--muted);font-size:.85rem}
@media (min-width:720px){.caps{grid-template-columns:1fr 1fr}}
"""


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def commit_link(repo_url: str, sha: str | None) -> str:
    if not sha:
        return DASH
    return f'<a href="{esc(repo_url)}/commit/{esc(sha)}"><code>{esc(sha[:7])}</code></a>'


def render_row(row: Row, history: list[dict[str, Any]], repo_url: str) -> str:
    cells: list[str] = []
    for e in reversed(history):  # oldest first, the newest on the right
        st = e["status"].get(row.cap.name, "none")
        label = f"{e['sha'][:7]} {e['date']}: {STATUS_TEXT[st]}"
        cells.append(f'<span class="{st}" title="{esc(label)}"></span>')
    note = f'<p class="note">{esc(row.note)}</p>' if row.note else ""
    last = esc(row.when) + (f" · {commit_link(repo_url, row.sha)}" if row.sha else "")
    return (
        f'<li class="cap {row.status}"><div class="top"><span class="name">{esc(row.cap.name)}'
        f'</span><span class="pill">{STATUS_ICON[row.status]} {STATUS_TEXT[row.status]}</span>'
        f"</div><dl><dt>Förväntat</dt><dd>{esc(row.expected)}</dd>"
        f"<dt>Uppmätt</dt><dd>{esc(row.measured)}</dd>"
        f"<dt>Scenario</dt><dd>{esc(row.scenario)}</dd>"
        f"<dt>Senast</dt><dd>{last}</dd></dl>"
        f"{note}"
        f'<div class="hist" role="img" aria-label="{esc(history_label(row, history))}">'
        f"{''.join(cells)}</div></li>"
    )


def history_label(row: Row, history: list[dict[str, Any]]) -> str:
    counts = dict.fromkeys(STATUSES, 0)
    for e in history:
        counts[e["status"].get(row.cap.name, "none")] += 1
    return (
        f"Senaste {len(history)} körningarna: {counts['pass']} gröna, {counts['fail']} röda, "
        f"{counts['none']} utan data"
    )


def render(
    groups: list[str],
    rows: list[Row],
    history: list[dict[str, Any]],
    *,
    sha: str,
    date: str,
    run_url: str,
    repo_url: str,
    films: list[tuple[str, str]],
) -> str:
    counts = {s: sum(r.status == s for r in rows) for s in STATUSES}
    if films:
        sources = "".join(f'<source src="{esc(n)}" type="{esc(t)}">' for n, t in films)
        film = (
            '<div class="film"><video controls playsinline muted preload="metadata">'
            f"{sources}Din webbläsare kan inte spela filmen.</video>"
            "<p>Demon i simulatorn med den skriptade befälhavaren: panelen med karta, chatt, "
            "sökmönster, STOPP och FLYG HEM. Uppspelad snabbare än verkligheten.</p></div>"
        )
    else:
        film = '<div class="film"><p>Ingen film än.</p></div>'
    run = f' · <a href="{esc(run_url)}">körningen</a>' if run_url else ""
    sections: list[str] = []
    for group in groups:
        mine = [r for r in rows if r.cap.group == group]
        if mine:
            items = "".join(render_row(r, history, repo_url) for r in mine)
            sections.append(f'<h2>{esc(group)}</h2><ul class="caps">{items}</ul>')
    return (
        '<!doctype html><html lang="sv"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Searchdrone: läget</title>"
        '<meta name="description" content="Vad Searchdrone klarar i simulatorn just nu">'
        f"<style>{CSS}</style></head><body><main>"
        "<h1>Searchdrone: läget just nu</h1>"
        f'<p class="sub">Simulatortesterna på <code>main</code>, commit '
        f"{commit_link(repo_url, sha)}, {esc(date)}{run}.</p>"
        '<ul class="summary">'
        f'<li class="pass">{counts["pass"]} gröna</li>'
        f'<li class="fail">{counts["fail"]} röda</li>'
        f'<li class="none">{counts["none"]} utan data</li></ul>'
        f"<h2>Demofilm</h2>{film}{''.join(sections)}"
        "<footer>Varje förmåga är ett test i simulatorn (ArduPilot SITL) som körs vid varje "
        "ändring på main och varje natt. Rutorna visar de senaste körningarna, äldst till "
        f'vänster. Källkod: <a href="{esc(repo_url)}">Searchdrone</a>.</footer>'
        "</main></body></html>\n"
    )


def parse(argv: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for arg in argv:
        key, sep, value = arg.partition("=")
        if not sep or key not in KEYS:
            raise SystemExit(f"status_page: okänd inställning '{arg}' (nycklar: {', '.join(KEYS)})")
        values[key] = value
    missing = [k for k in REQUIRED if not values.get(k)]
    if missing:
        raise SystemExit(f"status_page: saknas: {', '.join(missing)}")
    if not _SHA.match(values["sha"]):
        raise SystemExit("status_page: sha= ska vara en commit-SHA")
    return values


def build(values: dict[str, str]) -> Path:
    groups, caps = load_capabilities()
    rows = [evaluate(c, load_results(Path(values["results"]))) for c in caps]
    names = {c.name for c in caps}
    previous: object = []
    if values.get("history") and Path(values["history"]).is_file():
        try:
            previous = json.loads(Path(values["history"]).read_text(encoding="utf-8"))
        except ValueError:
            previous = []
    run_url = values.get("run_url", "")
    if not run_url.startswith("https://github.com/"):
        run_url = ""
    entry: dict[str, Any] = {
        "sha": values["sha"],
        "date": values["date"],
        "run_url": run_url,
        "status": {r.cap.name: r.status for r in rows},
    }
    history = next_history(clean_history(previous, names), entry)
    out = Path(values["out"])
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"status_page: {out} är inte tom")
    films: list[tuple[str, str]] = []
    film_dir = Path(values["film"]) if values.get("film") else None
    for name, mime in FILMS:
        if film_dir and (film_dir / name).is_file():
            shutil.copyfile(film_dir / name, out / name)
            films.append((name, mime))
    page = render(
        groups,
        rows,
        history,
        sha=values["sha"],
        date=values["date"],
        run_url=run_url,
        repo_url=values.get("repo_url") or DEFAULT_REPO_URL,
        films=films,
    )
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out / "index.html"


def main(argv: list[str] | None = None) -> int:
    path = build(parse(sys.argv[1:] if argv is None else argv))
    print(f"status_page: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
