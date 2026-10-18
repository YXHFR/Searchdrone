"""Acceptance checks for the release v0.1.0: ``just accept [phase=pre|post] [sha=<sha>]``.

Run on the owner's computer, in the WSL clone, with ``gh`` logged in as the owner. Every
GitHub call only reads (``gh api`` GET). The list of requirement IDs comes from the local
acceptance file named by ``$SD_ACCEPT_IDS`` (never in the repository; the format is in
``tools/gen_traceability.py``); without it the script stops with exit 2.

* ``phase=pre`` checks every ID except the after-release ones (``POST_IDS``),
  ``phase=post`` only those. ``sha`` is the release commit (default: ``main``'s head).
* A rows (``typ = "A"``) are checked here: CI results on the commit, local checks
  (``reuse lint``, the forbidden-file guard, gitleaks, licences, commit e-mails) and the
  reports in ``data/reports/``. An A row without a check here is red.
* M rows (``typ = "M"``) are what the owner has seen. They are read from
  ``data/reports/accept-manual-v0.1.0.toml`` (template:
  ``tools/accept/accept-manual.example.toml``), one table per ID with ``pass``, ``by``,
  ``date`` and ``note``. Only the owner fills it in; an agent never writes
  ``pass = true``. A missing table gives ``pass = null``. ``A + M`` rows need both.

The report is ``data/reports/accept-v0.1.0-<phase>-<UTC time>.json``: one row per ID with
``id``, ``typ``, ``pass`` (true/false/null), ``detalj``, ``dod_ref`` and ``sha``. Exit 0
only when every row passes; 1 when one does not; 2 when the script cannot run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO = "YXHFR/Searchdrone"
MANUAL_FILE = "data/reports/accept-manual-v0.1.0.toml"
REPORTS = "data/reports"
POST_IDS = frozenset({"V-02", "V-04", "R-04b"})
REQUIRED_CHECKS = ("ci-ok", "sitl-ok", "hardware-ok")
NIGHTLY_RUNS = 10
REPORT_MAX_AGE_D = 7
GPU = "RTX 3070"
GIF_URL = f"https://github.com/{REPO}/releases/download/v0.1.0/showcase-stage1.gif"
README_HEADINGS = ("Safety", "Quick start", "Requirements", "Licen")
# The ports the README must warn about (never exposed to the internet).
README_PORTS = (":8080", ":8443", ":7700", ":5433", ":8889")
RULESETS = ("main-protection", "release-tags")
TAG = "v0.1.0"


class Gh(Protocol):
    def api(self, path: str) -> Any: ...


class Runner(Protocol):
    def run(self, argv: Sequence[str]) -> tuple[int, str]: ...


class GhCli:
    """``gh api`` as the logged-in owner, read only (GET, paginated lists flattened)."""

    def api(self, path: str) -> Any:
        res = subprocess.run(  # noqa: S603 - fixed argv, the path is built here
            ["gh", "api", "--method", "GET", "--paginate", "--slurp", path],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if res.returncode != 0:
            raise CheckError(f"gh api {path}: {res.stderr.strip()[:200]}")
        pages = json.loads(res.stdout)
        if len(pages) == 1:
            return pages[0]
        if all(isinstance(p, list) for p in pages):
            return [item for p in pages for item in p]
        if all(isinstance(p, dict) for p in pages):
            merged: dict[str, Any] = dict(pages[0])
            for p in pages[1:]:
                for key, value in p.items():
                    if isinstance(value, list) and isinstance(merged.get(key), list):
                        merged[key] = merged[key] + value
            return merged
        return pages


class LocalRunner:
    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, argv: Sequence[str]) -> tuple[int, str]:
        exe = shutil.which(argv[0])
        if exe is None:
            return 127, f"{argv[0]} saknas"
        res = subprocess.run(  # noqa: S603 - argv is built here
            [exe, *argv[1:]],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        return res.returncode, (res.stdout + res.stderr)[-2000:]


class CheckError(Exception):
    """A check that could not be made (counts as red, with the reason)."""


@dataclass
class Row:
    id: str
    typ: str
    dod_ref: str
    sha: str
    passed: bool | None = None
    detalj: str = ""

    def as_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "typ": self.typ,
            "pass": self.passed,
            "detalj": self.detalj,
            "dod_ref": self.dod_ref,
            "sha": self.sha,
        }


@dataclass
class Context:
    sha: str
    root: Path
    gh: Gh
    runner: Runner
    now: dt.datetime
    _cache: dict[str, Any] = field(default_factory=dict[str, Any])

    def api(self, path: str) -> Any:
        if path not in self._cache:
            self._cache[path] = self.gh.api(path)
        return self._cache[path]

    def check_runs(self) -> dict[str, str]:
        """Check-run name -> conclusion on the commit (the latest run of each name)."""
        data = self.api(f"repos/{REPO}/commits/{self.sha}/check-runs?per_page=100")
        runs = data.get("check_runs", []) if isinstance(data, dict) else []
        out: dict[str, str] = {}
        for run in sorted(runs, key=lambda r: str(r.get("started_at") or "")):
            out[str(run["name"])] = str(run.get("conclusion") or run.get("status") or "")
        return out


Result = tuple[bool, str]
Check = Callable[[Context], Result]


# --- the checks ---------------------------------------------------------------------


def _checks_green(names: Sequence[str]) -> Check:
    def check(ctx: Context) -> Result:
        runs = ctx.check_runs()
        bad = [f"{n}={runs.get(n, 'saknas')}" for n in names if runs.get(n) != "success"]
        return (
            not bad,
            "gröna: " + ", ".join(names) if not bad else "inte gröna: " + ", ".join(bad),
        )

    return check


def _command(argv: Sequence[str], what: str) -> Check:
    def check(ctx: Context) -> Result:
        code, out = ctx.runner.run(argv)
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        return code == 0, f"{what}: kod {code}" + (f" ({tail[:160]})" if code else "")

    return check


def _private_patterns(ctx: Context) -> Result:
    patterns = ctx.root / "config/local/private-patterns.txt"
    if not patterns.is_file():
        return False, "saknas: config/local/private-patterns.txt"
    return _command(
        ["uv", "run", "tools/check_forbidden_files.py", "--private-patterns", str(patterns)],
        "privata mönster",
    )(ctx)


def _nightly(ctx: Context) -> Result:
    """F-01: the PR suite on the commit, and 10 of 10 end-to-end runs in one scheduled or
    on-demand run of sitl.yml on it (its ten parallel ``sitl-e2e`` jobs)."""
    ok, detail = _checks_green(["sitl-ok"])(ctx)
    if not ok:
        return ok, detail
    runs = ctx.api(f"repos/{REPO}/actions/workflows/sitl.yml/runs?head_sha={ctx.sha}&per_page=50")
    best = 0
    for run in runs.get("workflow_runs", []) if isinstance(runs, dict) else []:
        if run.get("event") not in ("schedule", "workflow_dispatch"):
            continue
        jobs = ctx.api(f"repos/{REPO}/actions/runs/{run['id']}/jobs?per_page=100")
        e2e = [
            j
            for j in (jobs.get("jobs", []) if isinstance(jobs, dict) else [])
            if str(j.get("name", "")).startswith("sitl-e2e")
        ]
        good = sum(1 for j in e2e if j.get("conclusion") == "success")
        if len(e2e) >= NIGHTLY_RUNS and good == len(e2e):
            return True, f"sitl-ok grön; körning {run['id']}: {good} av {len(e2e)}"
        best = max(best, good)
    return False, f"ingen körning med {NIGHTLY_RUNS} av {NIGHTLY_RUNS} (bäst: {best})"


def _latest_report(ctx: Context, prefix: str) -> tuple[Path | None, dict[str, Any]]:
    paths = sorted((ctx.root / REPORTS).glob(f"{prefix}-*.json"))
    if not paths:
        return None, {}
    try:
        return paths[-1], json.loads(paths[-1].read_text(encoding="utf-8"))
    except ValueError:
        return paths[-1], {}


def _machine_report(prefix: str) -> Check:
    """F-05/F-06: the newest report is fresh, from the owner's graphics card, on the commit."""

    def check(ctx: Context) -> Result:
        path, data = _latest_report(ctx, prefix)
        if path is None:
            return False, f"ingen rapport {REPORTS}/{prefix}-*.json"
        age = ctx.now - dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC)
        problems = []
        if age > dt.timedelta(days=REPORT_MAX_AGE_D):
            problems.append(f"äldre än {REPORT_MAX_AGE_D} dagar")
        if GPU not in str(data.get("gpu_name", "")):
            problems.append(f"gpu_name är inte {GPU}")
        if data.get("git_sha") != ctx.sha:
            problems.append("git_sha är inte release-commiten")
        if data.get("passed") is False:
            problems.append("rapporten är inte godkänd")
        return not problems, f"{path.name}: " + ("ok" if not problems else ", ".join(problems))

    return check


def _readme_check(ctx: Context) -> Result:
    path, data = _latest_report(ctx, "readme-check")
    if path is None:
        return False, f"ingen rapport {REPORTS}/readme-check-*.json"
    ok = data.get("code") == 0 and data.get("git_sha") == ctx.sha and bool(data.get("method"))
    return ok, f"{path.name}: kod {data.get('code')}, sätt {data.get('method')}"


def _readme(ctx: Context) -> Result:
    """V-05: English README with the safety warning, quick start, requirements, licences."""
    text = (ctx.root / "README.md").read_text(encoding="utf-8")
    missing = [h for h in README_HEADINGS if not re.search(rf"^#+ .*{h}", text, re.M | re.I)]
    missing += [p for p in README_PORTS if p not in text]
    missing += [
        c
        for c in ("bash tools/setup/wsl-setup.sh", "bash tools/bootstrap.sh", "just demo")
        if c not in text
    ]
    return not missing, "README komplett" if not missing else "saknas i README: " + ", ".join(
        missing
    )


def _rulesets(ctx: Context) -> Result:
    names = {str(r.get("name")) for r in ctx.api(f"repos/{REPO}/rulesets")}
    missing = [n for n in RULESETS if n not in names]
    extra = ["bootstrap-protection"] if "bootstrap-protection" in names else []
    ok = not missing and not extra
    return ok, f"rulesets: {', '.join(sorted(names))}" + ("" if ok else " (fel uppsättning)")


def _merged_gates(ctx: Context) -> Result:
    tool = ctx.root / "tools/check_merged_pr_gates.py"
    if not tool.is_file():
        return False, "inte byggt än (B1-18): tools/check_merged_pr_gates.py"
    code, _ = ctx.runner.run(["uv", "run", str(tool.relative_to(ctx.root)), "--sha", ctx.sha])
    return code == 0, f"check_merged_pr_gates: kod {code}"


def _hardware_owner_ok(ctx: Context) -> Result:
    """R-09: every merged pull request touching hardware/ or CODEOWNERS has the owner's OK
    (the label owner-ok after the owner's OK in the chat; GitHub's Approve is not
    possible on the owner's own pull requests)."""
    pulls = ctx.api(f"repos/{REPO}/pulls?state=closed&per_page=100")
    missing = []
    for pr in pulls:
        if not pr.get("merged_at"):
            continue
        files = ctx.api(f"repos/{REPO}/pulls/{pr['number']}/files?per_page=100")
        touched = any(
            str(f["filename"]).startswith("hardware/") or f["filename"] == ".github/CODEOWNERS"
            for f in files
        )
        labels = {str(lbl.get("name")) for lbl in pr.get("labels", [])}
        if touched and "owner-ok" not in labels:
            missing.append(f"#{pr['number']}")
    return not missing, "alla har owner-ok" if not missing else "utan owner-ok: " + ", ".join(
        missing
    )


def _tag(ctx: Context) -> Result:
    ref = ctx.api(f"repos/{REPO}/git/ref/tags/{TAG}")
    obj = ref.get("object", {})
    sha = obj.get("sha")
    if obj.get("type") == "tag":
        sha = ctx.api(f"repos/{REPO}/git/tags/{sha}").get("object", {}).get("sha")
    release = ctx.api(f"repos/{REPO}/releases/tags/{TAG}")
    ok = sha == ctx.sha and release.get("draft") is False
    ok_runs, detail = _checks_green(REQUIRED_CHECKS)(ctx)
    return (
        ok and ok_runs,
        f"taggen pekar på {str(sha)[:12]}, utkast={release.get('draft')}; {detail}",
    )


def _gif(ctx: Context) -> Result:
    code, _ = ctx.runner.run(["curl", "-sfIL", "-o", "/dev/null", GIF_URL])
    readme = GIF_URL in (ctx.root / "README.md").read_text(encoding="utf-8")
    _, files = ctx.runner.run(["git", "ls-files", "*.gif", "*.mp4"])
    ok = code == 0 and readme and not files.strip()
    return (
        ok,
        f"GIF svarar: {code == 0}, README länkar: {readme}, inga .gif/.mp4: {not files.strip()}",
    )


def _release_assets(ctx: Context) -> Result:
    patterns = ctx.root / "config/local/private-patterns.txt"
    if not patterns.is_file():
        return False, "saknas: config/local/private-patterns.txt"
    out = ctx.root / "build/accept-assets"
    shutil.rmtree(out, ignore_errors=True)
    code, _ = ctx.runner.run(["gh", "release", "download", TAG, "--repo", REPO, "--dir", str(out)])
    if code != 0:
        return False, f"gh release download: kod {code}"
    names = [str(p.relative_to(ctx.root)) for p in out.rglob("*") if p.is_file()]
    code, _ = ctx.runner.run(
        ["uv", "run", "tools/check_forbidden_files.py", "--private-patterns", str(patterns), *names]
    )
    return code == 0, f"{len(names)} tillgångar, privata mönster: kod {code}"


def automatic_checks() -> dict[str, Check]:
    """ID -> its automatic check. Rows proven by the pull-request CI on the commit (the
    flight and safety suites, the panel's browser tests) need its required checks green."""
    ci = _checks_green(("ci-ok", "sitl-ok"))
    checks: dict[str, Check] = {
        "R-01": _command(["uv", "run", "reuse", "lint"], "reuse lint"),
        "R-02": _command(["uv", "run", "tools/check_forbidden_files.py"], "förbjudna filer"),
        "R-03": _command(
            ["gitleaks", "git", "--redact", "--exit-code", "1", "--config", ".gitleaks.toml", "."],
            "gitleaks",
        ),
        "R-04": _private_patterns,
        "R-04b": _release_assets,
        "R-05": _checks_green(REQUIRED_CHECKS),
        "R-06": _rulesets,
        "R-07": _command(["uv", "run", "tools/check_licenses.py"], "licenser"),
        "R-08": _merged_gates,
        "R-09": _hardware_owner_ok,
        "R-10": _command(
            ["uv", "run", "tools/check_commit_emails.py", "--head", "HEAD"], "commit-e-post"
        ),
        "R-11": _command(
            ["uv", "run", "pytest", "-q", "tools/tests/test_hardware_changes.py"], "hårdvarujobben"
        ),
        "F-01": _nightly,
        "F-05": _machine_report("eval"),
        "F-06": _machine_report("perf"),
        "F-07": _checks_green(("e2e",)),
        "V-01": _readme_check,
        "V-02": _gif,
        "V-04": _tag,
        "V-05": _readme,
    }
    for rid in ("F-03", "F-08", "F-09", "F-10", "F-11", "F-12", "F-13"):
        checks[rid] = ci
    for n in range(1, 26):
        checks.setdefault(f"S-{n:02d}", ci)
    return checks


# --- the run ------------------------------------------------------------------------


def load_ids(path: Path) -> list[dict[str, Any]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return [dict(e) for e in data.get("id", [])]


def load_manual(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {k: dict(v) for k, v in tomllib.loads(path.read_text(encoding="utf-8")).items()}


def _manual(entry: dict[str, Any] | None) -> tuple[bool | None, str]:
    if entry is None:
        return None, f"saknas i {MANUAL_FILE}"
    passed = entry.get("pass")
    who = f"{entry.get('by', '?')} {entry.get('date', '?')}".strip()
    note = str(entry.get("note", ""))[:200]
    if passed is True and entry.get("by") and entry.get("date"):
        return True, f"ägaren: {who} {note}".strip()
    if passed is True:
        return False, "pass = true utan by och date"
    return False, f"ägaren: inte godkänd ({who}) {note}".strip()


def evaluate(
    ids: list[dict[str, Any]],
    phase: str,
    ctx: Context,
    manual: dict[str, dict[str, Any]],
    checks: dict[str, Check] | None = None,
) -> list[Row]:
    checks = automatic_checks() if checks is None else checks
    rows: list[Row] = []
    for entry in ids:
        rid = str(entry["id"])
        if (rid in POST_IDS) != (phase == "post"):
            continue
        typ = str(entry.get("typ", "A")).replace(" ", "")
        row = Row(rid, str(entry.get("typ", "A")), str(entry.get("dod_ref", "")), ctx.sha)
        parts: list[str] = []
        results: list[bool | None] = []
        if "A" in typ:
            check = checks.get(rid)
            if check is None:
                results.append(False)
                parts.append("ingen automatisk kontroll")
            else:
                try:
                    ok, detail = check(ctx)
                except (CheckError, OSError, ValueError, KeyError, TypeError) as exc:
                    ok, detail = False, f"kunde inte kontrolleras: {exc}"
                results.append(ok)
                parts.append(detail)
        if "M" in typ:
            ok_m, detail_m = _manual(manual.get(rid))
            results.append(ok_m)
            parts.append(detail_m)
        if any(r is False for r in results):
            row.passed = False
        elif any(r is None for r in results):
            row.passed = None
        else:
            row.passed = True
        row.detalj = "; ".join(parts)
        rows.append(row)
    return rows


def resolve_sha(gh: Gh, sha: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", sha):
        return sha
    data = gh.api(f"repos/{REPO}/commits/{sha or 'main'}")
    return str(data["sha"])


def main(
    argv: list[str] | None = None,
    *,
    gh: Gh | None = None,
    runner: Runner | None = None,
    root: Path = REPO_ROOT,
    now: dt.datetime | None = None,
    checks: dict[str, Check] | None = None,
) -> int:
    ap = argparse.ArgumentParser(prog="just accept")
    ap.add_argument("--phase", choices=("pre", "post"), default="pre")
    ap.add_argument("--sha", default="")
    args = ap.parse_args(argv)
    raw_ids = os.environ.get("SD_ACCEPT_IDS", "")
    if not raw_ids or not Path(raw_ids).is_file():
        print(
            "Acceptanslistan saknas. Sätt SD_ACCEPT_IDS till sökvägen till den lokala filen.",
            file=sys.stderr,
        )
        return 2
    ids_path = Path(raw_ids)
    gh = gh or GhCli()
    runner = runner or LocalRunner(root)
    now = now or dt.datetime.now(dt.UTC)
    try:
        sha = resolve_sha(gh, args.sha)
    except (CheckError, KeyError) as exc:
        print(f"Commiten gick inte att slå upp: {exc}", file=sys.stderr)
        return 2
    ctx = Context(sha=sha, root=root, gh=gh, runner=runner, now=now)
    rows = evaluate(load_ids(ids_path), args.phase, ctx, load_manual(root / MANUAL_FILE), checks)
    out_dir = root / REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"accept-v0.1.0-{args.phase}-{now:%Y%m%dT%H%M%SZ}.json"
    out.write_text(
        json.dumps([r.as_json() for r in rows], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for r in rows:
        mark = {True: "ok ", False: "NEJ", None: " ? "}[r.passed]
        print(f"[{mark}] {r.id:<6} {r.typ:<5} {r.detalj}")
    green = bool(rows) and all(r.passed is True for r in rows)
    print(f"Rapport: {out.relative_to(root)}. {'Allt godkänt.' if green else 'Inte godkänt.'}")
    return 0 if green else 1


if __name__ == "__main__":
    sys.exit(main())
