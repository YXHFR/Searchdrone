# pyright: strict
"""Acceptance checks for the release ``v0.1.0``.

Usage: ``python tools/accept/v0_1_0.py [phase=pre|post] [sha=<sha>]``

Runs on the owner's own clone, with ``gh`` logged in as the owner. Every GitHub call is
read-only and goes through :func:`gh`, which refuses anything else.

The requirement IDs come from a local TOML file that is never in the repository; its
path is ``$SD_ACCEPT_IDS`` (required; the repository names no path to it). It is the
same file ``tools/gen_traceability.py`` reads: one ``[[id]]`` table per requirement with
``id``, ``typ`` (``A``, ``M`` or ``A + M``), ``dod_ref``, ``sc`` and ``tests``. This script
owns the check behind each ID and which phase it belongs to: ``post`` is
:data:`POST_IDS` (after the release is published), ``pre`` is everything else.

Manual rows (``M``, and the manual half of ``A + M``) are read from
``data/reports/accept-manual-v0.1.0.toml`` (gitignored; template in
``tools/accept/accept-manual.example.toml``), one table per ID::

    [F-02]
    pass = true
    by = "owner"
    date = "2026-10-20"
    note = "3 of 3 runs"

Only the owner fills that file in; this script only reads it. An ID missing there has
``pass = null``. A check that cannot be automated yet reports ``pass = false`` and says
why; nothing passes by default.

Output: ``data/reports/accept-v0.1.0-<phase>-<UTC time>.json`` with one row per ID
(``id``, ``typ``, ``pass``, ``detalj``, ``dod_ref``, ``sha``) and a short table on stdout.
Exit code 0 only when every row of the phase has ``pass = true``; 1 otherwise; 2 for
bad arguments or a missing or unreadable ID list or manual file.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, cast

TOOLS = Path(__file__).resolve().parent.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import check_forbidden_files as cff  # noqa: E402 - needs tools/ on sys.path first
import gen_traceability  # noqa: E402

REPO = TOOLS.parent
VERSION = "0.1.0"
TAG = f"v{VERSION}"
DEFAULT_REPO_SLUG = "YXHFR/Searchdrone"
MANUAL_FILE = f"data/reports/accept-manual-v{VERSION}.toml"
REPORT_DIR = "data/reports"
PATTERNS_FILE = "config/local/private-patterns.txt"
STAGE1_FILE = "showcase/out/stage1.json"
EMAIL_EXCEPTIONS = "tools/accept/email_exceptions.txt"
GIF_NAME = "showcase-stage1.gif"
CMD_TIMEOUT_S = 1800
MAX_REPORT_AGE = dt.timedelta(days=7)
EXIT_USAGE = 2

PHASES = ("pre", "post")
POST_IDS = frozenset({"V-02", "V-04", "R-04b"})
# Every ID of the v0.1.0 list (08 section 5). An ID the local list lacks is reported red.
KNOWN_IDS = (
    *(f"R-{n:02d}" for n in range(1, 12)),
    "R-04b",
    *(f"F-{n:02d}" for n in range(1, 14)),
    *(f"S-{n:02d}" for n in range(1, 26)),
    *(f"V-{n:02d}" for n in range(1, 7)),
)
# Required checks on main (rev 6 owner decisions) and the ones a push to main reports.
REQUIRED_CHECKS = (
    "ci-ok",
    "sitl-ok",
    "hardware-ok",
    "security-ok",
    "gate/review",
    "gate/owner",
    "gate/pr-title",
)
MAIN_CHECKS = ("ci-ok", "sitl-ok", "hardware-ok", "security-ok")
# Hardware test pull requests (B0-06): a file of this kind must turn hardware-ok red,
# a change to only hardware/drone.yaml must keep it green.
HW_RED = (".inp", ".kicad_sch", ".kicad_pcb", ".step", ".FCStd", "hardware/README.md")
HW_GREEN = "hardware/drone.yaml"
README_NEEDS = (
    "bash tools/setup/wsl-setup.sh",
    "bash tools/bootstrap.sh",
    "just demo",
    ":8080",
    ":8443",
    ":7700",
    ":5433",
    ":8889",
    ".gif",
)
README_HEADINGS = (r"security", r"quick\s*start", r"requirement", r"licen[cs]e")
EN_WORDS = frozenset({"the", "and", "is", "to", "of", "with", "you"})
SV_WORDS = frozenset({"och", "är", "att", "det", "för", "med", "du"})

TEXT = {
    True: "ja",
    False: "nej",
    None: "saknas",
}


# --- results -------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    ok: bool
    detail: str


@dataclass(frozen=True)
class Row:
    id: str
    typ: str
    passed: bool | None
    detalj: str
    dod_ref: str
    sha: str

    def as_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "typ": self.typ,
            "pass": self.passed,
            "detalj": self.detalj,
            "dod_ref": self.dod_ref,
            "sha": self.sha,
        }


@dataclass
class Ctx:
    """Everything a check needs; ``cache`` holds answers shared by several rows."""

    root: Path
    sha: str
    repo: str
    entries: dict[str, dict[str, Any]]
    now: dt.datetime
    cache: dict[str, Any] = field(default_factory=lambda: {})


class UsageError(Exception):
    """Bad arguments or input files: exit code 2."""


class GhError(RuntimeError):
    """A failed or refused ``gh`` call."""


# --- the two ways out of the process (tests replace both) ------------------------------


@dataclass(frozen=True)
class Proc:
    code: int
    out: str
    err: str


def run(cmd: list[str], cwd: Path) -> Proc:
    """Run a local command without a shell. A missing program gives code 127."""
    try:
        p = subprocess.run(  # noqa: S603 - fixed argument lists, no shell
            cmd, cwd=cwd, capture_output=True, text=True, check=False, timeout=CMD_TIMEOUT_S
        )
    except FileNotFoundError:
        return Proc(127, "", f"{cmd[0]} finns inte")
    except subprocess.TimeoutExpired:
        return Proc(124, "", f"{cmd[0]} tog längre tid än {CMD_TIMEOUT_S} s")
    return Proc(p.returncode, p.stdout, p.stderr)


READ_ONLY_GH = frozenset(
    {
        ("api",),
        ("run", "list"),
        ("run", "view"),
        ("release", "view"),
        ("release", "download"),
        ("pr", "list"),
    }
)
WRITE_FLAGS = frozenset({"-X", "--method", "-f", "-F", "--field", "--raw-field", "--input"})


def ensure_read_only(args: list[str]) -> None:
    """Refuse every ``gh`` call that could change something on GitHub."""
    if not args:
        raise GhError("gh utan argument")
    key = (args[0],) if args[0] == "api" else tuple(args[:2])
    if key not in READ_ONLY_GH:
        raise GhError(f"gh {' '.join(args[:2])} är inte ett läsande anrop")
    for a in args:
        long_form = a.startswith(tuple(f"{f}=" for f in WRITE_FLAGS if f.startswith("--")))
        short_form = a[:2] in WRITE_FLAGS and not a.startswith("--")
        if a in WRITE_FLAGS or long_form or short_form:
            raise GhError(f"gh med {a} är inte tillåtet (bara läsande anrop)")


def gh(args: list[str]) -> str:
    """The only way this script talks to GitHub: a read-only ``gh`` call, stdout back."""
    ensure_read_only(args)
    proc = run(["gh", *args], REPO)
    if proc.code != 0:
        raise GhError(f"gh {' '.join(args[:2])}: kod {proc.code}: {last_line(proc.err)}")
    return proc.out


def gh_json(args: list[str]) -> Any:
    return json.loads(gh(args) or "null")


def obj(value: Any) -> dict[str, Any]:
    """A JSON object, or an empty one for anything else."""
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def arr(value: Any) -> list[Any]:
    """A JSON array, or an empty one for anything else."""
    return cast(list[Any], value) if isinstance(value, list) else []


# --- small helpers -------------------------------------------------------------------


def last_line(text: str, limit: int = 160) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (lines[-1] if lines else "")[:limit]


def git(ctx: Ctx, *args: str) -> Proc:
    return run(["git", *args], ctx.root)


def local_guard(ctx: Ctx) -> Check | None:
    """Local checks read the working tree, so it must be exactly ``sha`` and clean."""
    if "guard" not in ctx.cache:
        head = git(ctx, "rev-parse", "HEAD").out.strip()
        dirty = git(ctx, "status", "--porcelain", "--untracked-files=no").out.strip()
        guard: Check | None = None
        if head != ctx.sha:
            guard = Check(
                False,
                f"arbetskopian står på {head[:12] or '?'}, inte {ctx.sha[:12]}; "
                f"kör git checkout {ctx.sha[:12]} först",
            )
        elif dirty:
            guard = Check(False, "arbetskopian har ändrade filer; kontrollerna kräver rent träd")
        ctx.cache["guard"] = guard
    return ctx.cache["guard"]


def local_cmd(ctx: Ctx, cmd: list[str], what: str, missing: str | None = None) -> Check:
    guard = local_guard(ctx)
    if guard is not None:
        return guard
    proc = run(cmd, ctx.root)
    if proc.code == 127:
        return Check(False, missing or f"{cmd[0]} finns inte")
    if proc.code != 0:
        tail = last_line(proc.out) or last_line(proc.err)
        return Check(False, f"{what}: kod {proc.code}" + (f": {tail}" if tail else ""))
    return Check(True, f"{what}: grön")


def py_tool(script: str, *args: str) -> list[str]:
    return [sys.executable, f"tools/{script}", *args]


def checks_on(ctx: Ctx, sha: str) -> dict[str, str]:
    """Check run and commit status name -> conclusion (``success``, ``failure`` ...)."""
    key = f"checks:{sha}"
    if key not in ctx.cache:
        out: dict[str, str] = {}
        status = gh_json(["api", f"repos/{ctx.repo}/commits/{sha}/status?per_page=100"])
        for s in arr(obj(status).get("statuses")):
            out.setdefault(str(s.get("context")), str(s.get("state")))
        runs = gh_json(["api", f"repos/{ctx.repo}/commits/{sha}/check-runs?per_page=100"])
        for r in arr(obj(runs).get("check_runs")):
            state = r.get("conclusion") if r.get("status") == "completed" else r.get("status")
            out[str(r.get("name"))] = str(state)
        ctx.cache[key] = out
    return ctx.cache[key]


def need_green(checks: dict[str, str], names: tuple[str, ...]) -> list[str]:
    return [f"{n}={checks.get(n, 'saknas')}" for n in names if checks.get(n) != "success"]


def stamp_of(path: Path) -> dt.datetime:
    m = re.search(r"(\d{8}T\d{6}Z)", path.name)
    if m:
        return dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.UTC)
    return dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.UTC)


def latest_report(ctx: Ctx, glob: str) -> tuple[Path, dict[str, Any]] | None:
    paths = sorted((ctx.root / REPORT_DIR).glob(glob), key=stamp_of)
    if not paths:
        return None
    data = json.loads(paths[-1].read_text(encoding="utf-8"))
    return paths[-1], obj(data)


def readme_at(ctx: Ctx) -> str | None:
    proc = git(ctx, "show", f"{ctx.sha}:README.md")
    return proc.out if proc.code == 0 else None


def private_patterns(ctx: Ctx) -> list[tuple[int, Callable[[str], bool]]] | None:
    path = ctx.root / PATTERNS_FILE
    return cff.load_private_patterns(path) if path.is_file() else None


def scan_blobs(
    blobs: list[tuple[str, bytes]], patterns: list[tuple[int, Callable[[str], bool]]]
) -> int:
    """Count private-pattern and secret findings in files that are not in git."""
    found = 0
    for name, data in blobs:
        enc = "latin-1" if cff.is_binary(data) else "utf-8"
        lines = data.decode(enc, "replace").splitlines()
        found += len(cff.check_private(name, lines, patterns)) + len(cff.check_secrets(name, lines))
        if PurePosixPath(name).suffix.lower() in cff.IMAGE_EXTENSIONS and cff.has_exif_gps(data):
            found += 1
    return found


def unpack(path: Path) -> list[tuple[str, bytes]]:
    """A release asset and, for wheels and tarballs, every file inside it."""
    blobs = [(path.name, path.read_bytes())]
    name = path.name.lower()
    if name.endswith((".whl", ".zip")):
        with zipfile.ZipFile(path) as zf:
            blobs += [(f"{path.name}/{i.filename}", zf.read(i)) for i in zf.infolist()]
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path) as tf:
            for member in tf.getmembers():
                f = tf.extractfile(member) if member.isfile() else None
                if f is not None:
                    blobs.append((f"{path.name}/{member.name}", f.read()))
    return blobs


# --- the automatic checks ------------------------------------------------------------


def r01(ctx: Ctx) -> Check:
    return local_cmd(ctx, ["uv", "run", "--no-sync", "reuse", "lint"], "reuse lint")


def r02(ctx: Ctx) -> Check:
    return local_cmd(ctx, py_tool("check_forbidden_files.py"), "check_forbidden_files")


def r03(ctx: Ctx) -> Check:
    cmd = ["gitleaks", "git", "--config", ".gitleaks.toml", "--redact", "--exit-code", "1", "."]
    return local_cmd(ctx, cmd, "gitleaks (hela historiken)", "gitleaks är inte installerat")


def r04(ctx: Ctx) -> Check:
    patterns = private_patterns(ctx)
    if patterns is None:
        return Check(False, f"{PATTERNS_FILE} saknas")
    repo = local_cmd(
        ctx,
        py_tool("check_forbidden_files.py", "--private-patterns", PATTERNS_FILE),
        "privata mönster i repot",
    )
    if not repo.ok:
        return repo
    stage1 = ctx.root / STAGE1_FILE
    if not stage1.is_file():
        return Check(False, f"repot rent, men {STAGE1_FILE} saknas (B1-18)")
    hits = scan_blobs([(stage1.name, stage1.read_bytes())], patterns)
    if hits:
        return Check(False, f"{hits} fynd i {STAGE1_FILE}")
    return Check(True, f"0 fynd i repot och i {STAGE1_FILE}")


def r04b(ctx: Ctx) -> Check:
    patterns = private_patterns(ctx)
    if patterns is None:
        return Check(False, f"{PATTERNS_FILE} saknas")
    with tempfile.TemporaryDirectory(prefix="sd-accept-") as tmp:
        gh(["release", "download", TAG, "--repo", ctx.repo, "--dir", tmp])
        assets = sorted(p for p in Path(tmp).iterdir() if p.is_file())
        if not assets:
            return Check(False, f"releasen {TAG} har inga tillgångar")
        blobs = [b for p in assets for b in unpack(p)]
    hits = scan_blobs(blobs, patterns)
    if hits:
        return Check(False, f"{hits} fynd i release-tillgångarna")
    return Check(True, f"0 fynd i {len(assets)} tillgångar ({len(blobs)} filer)")


def r05(ctx: Ctx) -> Check:
    bad = need_green(checks_on(ctx, ctx.sha), MAIN_CHECKS)
    if bad:
        return Check(False, "inte gröna: " + ", ".join(bad))
    return Check(True, ", ".join(MAIN_CHECKS) + " gröna")


def r06(ctx: Ctx) -> Check:
    listing = arr(gh_json(["api", f"repos/{ctx.repo}/rulesets?includes_parents=false"]))
    by_name = {str(r.get("name")): obj(r) for r in listing}
    problems: list[str] = []
    if "bootstrap-protection" in by_name:
        problems.append("bootstrap-protection finns kvar")
    for name in ("main-protection", "release-tags"):
        if name not in by_name:
            problems.append(f"{name} saknas")
    if "main-protection" in by_name:
        rid = by_name["main-protection"]["id"]
        detail = gh_json(["api", f"repos/{ctx.repo}/rulesets/{rid}"])
        problems += main_ruleset_problems(obj(detail))
    if "release-tags" in by_name:
        rel = by_name["release-tags"]
        if rel.get("target") != "tag" or rel.get("enforcement") != "active":
            problems.append("release-tags är inte ett aktivt ruleset för taggar")
    if problems:
        return Check(False, "; ".join(problems))
    return Check(True, "main-protection och release-tags enligt B0-07b, inget bootstrap-protection")


def main_ruleset_problems(rs: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if rs.get("enforcement") != "active":
        out.append("main-protection är inte aktivt")
    if "bypass_actors" not in rs:
        out.append("bypass_actors syns inte (kör med ägarens gh-inloggning)")
    elif rs["bypass_actors"]:
        out.append("main-protection har bypass-aktörer")
    rules = {str(r.get("type")): obj(r.get("parameters")) for r in arr(rs.get("rules"))}
    for t in ("pull_request", "non_fast_forward", "deletion", "required_status_checks"):
        if t not in rules:
            out.append(f"regeln {t} saknas")
    status_rule = rules.get("required_status_checks", {})
    have = {str(c.get("context")) for c in arr(status_rule.get("required_status_checks"))}
    missing = [c for c in REQUIRED_CHECKS if c not in have]
    if missing:
        out.append("obligatoriska kontroller saknas: " + ", ".join(missing))
    methods = rules.get("pull_request", {}).get("allowed_merge_methods")
    if methods is not None and arr(methods) != ["squash"]:
        out.append(f"merge-sätt {methods}, bara squash tillåts")
    return out


def merged_pr_gates(ctx: Ctx) -> Check:
    if not (ctx.root / "tools/check_merged_pr_gates.py").is_file():
        return Check(False, "tools/check_merged_pr_gates.py inte byggt än (B1-18)")
    if "pr_gates" not in ctx.cache:
        root = git(ctx, "rev-list", "--max-parents=0", ctx.sha).out.split()
        if not root:
            return Check(False, "hittar ingen rotcommit")
        cmd = py_tool("check_merged_pr_gates.py", "--since", root[0], "--sha", ctx.sha)
        ctx.cache["pr_gates"] = local_cmd(ctx, cmd, "check_merged_pr_gates")
    return ctx.cache["pr_gates"]


def r07(ctx: Ctx) -> Check:
    return local_cmd(ctx, py_tool("check_licenses.py"), "licenskontrollen")


def r10(ctx: Ctx) -> Check:
    args = ["--head", ctx.sha]
    if (ctx.root / EMAIL_EXCEPTIONS).is_file():
        args += ["--exceptions", EMAIL_EXCEPTIONS]
    proc = run(py_tool("check_commit_emails.py", *args), ctx.root)
    if proc.code != 0:
        return Check(False, f"commit-e-post: kod {proc.code}: {last_line(proc.err or proc.out)}")
    sec = obj(obj(gh_json(["api", f"repos/{ctx.repo}"])).get("security_and_analysis"))
    off = [
        k
        for k in ("secret_scanning", "secret_scanning_push_protection")
        if obj(sec.get(k)).get("status") != "enabled"
    ]
    if off:
        return Check(False, "commit-e-post ok, men inte påslaget: " + ", ".join(off))
    return Check(True, f"{last_line(proc.out)}; secret scanning och push protection på")


def r11(ctx: Ctx) -> Check:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    unit = local_cmd(ctx, [*cmd, "tools/tests/test_hardware_changes.py"], "test_hardware_changes")
    if not unit.ok:
        return unit
    prs = gh_json(
        [
            "pr",
            "list",
            "--repo",
            ctx.repo,
            "--state",
            "closed",
            "--limit",
            "200",
            "--json",
            "number,mergedAt,headRefOid,files",
        ]
    )
    closed = [obj(p) for p in arr(prs)]
    test_prs = sorted((p for p in closed if not p.get("mergedAt")), key=lambda p: int(p["number"]))
    problems: list[str] = []
    for kind in (*HW_RED, HW_GREEN):
        want = "success" if kind == HW_GREEN else "failure"
        match = [p for p in test_prs if hw_pr_matches(p, kind)]
        if not match:
            problems.append(f"ingen stängd test-PR för {kind}")
            continue
        pr = match[-1]
        got = checks_on(ctx, str(pr["headRefOid"])).get("hardware-ok", "saknas")
        if got != want:
            problems.append(f"#{pr['number']} ({kind}): hardware-ok={got}, väntat {want}")
    if problems:
        return Check(False, "; ".join(problems))
    return Check(True, "test_hardware_changes grön; test-PR:erna röda/gröna som väntat")


def hw_pr_matches(pr: dict[str, Any], kind: str) -> bool:
    paths = [str(obj(f).get("path")) for f in arr(pr.get("files"))]
    if kind == HW_GREEN:
        return paths == [HW_GREEN]
    if "/" in kind:
        return kind in paths
    return any(p.startswith("hardware/") and p.endswith(kind) for p in paths)


def f01(ctx: Ctx) -> Check:
    pr_ci = need_green(checks_on(ctx, ctx.sha), ("sitl-ok",))
    if pr_ci:
        return Check(False, "SITL på sha inte grön: " + ", ".join(pr_ci))
    runs = gh_json(
        [
            "run",
            "list",
            "--repo",
            ctx.repo,
            "--workflow",
            "sitl.yml",
            "--commit",
            ctx.sha,
            "--limit",
            "50",
            "--json",
            "databaseId,event,status,conclusion,headSha",
        ]
    )
    nightly = [
        r
        for r in arr(runs)
        if r.get("event") in ("schedule", "workflow_dispatch") and r.get("headSha") == ctx.sha
    ]
    for r in nightly:
        view = gh_json(["run", "view", str(r["databaseId"]), "--repo", ctx.repo, "--json", "jobs"])
        jobs = {str(j.get("name")): j.get("conclusion") for j in arr(obj(view).get("jobs"))}
        if jobs.get("sitl-nightly") == "success":
            return Check(True, f"sitl-ok grön; sitl-nightly grön i körning {r['databaseId']}")
    return Check(False, f"ingen grön sitl-nightly-körning med head_sha = {ctx.sha[:12]}")


def gpu_report(ctx: Ctx, glob: str, what: str) -> Check:
    found = latest_report(ctx, glob)
    if found is None:
        return Check(False, f"ingen {what}-rapport i {REPORT_DIR}/ ({glob})")
    path, data = found
    problems: list[str] = []
    age = ctx.now - stamp_of(path)
    if age > MAX_REPORT_AGE:
        problems.append(f"{age.days} dagar gammal")
    if "RTX 3070" not in str(data.get("gpu_name") or ""):
        problems.append(f"gpu_name={data.get('gpu_name')!r}")
    if data.get("git_sha") != ctx.sha:
        problems.append(f"git_sha={str(data.get('git_sha'))[:12]}, inte sha")
    if data.get("passed") is not True:
        problems.append(f"passed={data.get('passed')!r}")
    if problems:
        return Check(False, f"{path.name}: " + ", ".join(problems))
    return Check(True, f"{path.name}: godkänd, RTX 3070, git_sha = sha")


def f05(ctx: Ctx) -> Check:
    return gpu_report(ctx, "eval-grunden-*.json", "eval")


def f06(ctx: Ctx) -> Check:
    return gpu_report(ctx, "perf-*.json", "prestanda")


def traced(ctx: Ctx, rid: str) -> Check | None:
    """The row's tests (local ID list) exist at ``sha``; ``None`` when they do."""
    guard = local_guard(ctx)
    if guard is not None:
        return guard
    if "tests" not in ctx.cache:
        ctx.cache["tests"] = gen_traceability.test_functions(ctx.root)
    found = gen_traceability.problems({"id": [ctx.entries[rid]]}, ctx.cache["tests"])
    return Check(False, "; ".join(found)) if found else None


def ci_rows(
    names: tuple[str, ...], job_prefix: str | None, label: str
) -> Callable[[Ctx, str], Check]:
    """A row proven by CI on ``sha``: the named checks green and the row's tests exist."""

    def check(ctx: Ctx, rid: str) -> Check:
        trace = traced(ctx, rid)
        if trace is not None:
            return trace
        checks = checks_on(ctx, ctx.sha)
        bad = need_green(checks, names)
        if job_prefix is not None:
            jobs = {n: c for n, c in checks.items() if n.startswith(job_prefix)}
            if not jobs:
                bad.append(f"ingen körning {job_prefix}* på sha")
            bad += [f"{n}={c}" for n, c in sorted(jobs.items()) if c != "success"]
        if bad:
            return Check(False, f"{label}: " + ", ".join(bad))
        return Check(True, f"{label} grön på sha; testerna finns")

    return check


def v01(ctx: Ctx) -> Check:
    found = latest_report(ctx, "readme-check-*.json")
    if found is None:
        return Check(False, f"ingen {REPORT_DIR}/readme-check-*.json (B1-18)")
    path, data = found
    problems: list[str] = []
    if data.get("git_sha") != ctx.sha:
        problems.append("git_sha är inte sha")
    if data.get("exit_code") != 0:
        problems.append(f"exit_code={data.get('exit_code')!r}")
    if not data.get("method"):
        problems.append("sättet (method) saknas")
    if problems:
        return Check(False, f"{path.name}: " + ", ".join(problems))
    return Check(True, f"{path.name}: kod 0, sätt {data['method']}")


def gif_url(ctx: Ctx) -> str:
    return f"https://github.com/{ctx.repo}/releases/download/{TAG}/{GIF_NAME}"


def v02(ctx: Ctx) -> Check:
    url = gif_url(ctx)
    curl = ["curl", "-sIL", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "30", url]
    proc = run(curl, ctx.root)
    problems: list[str] = []
    if proc.code != 0 or proc.out.strip()[-3:] != "200":
        problems.append(f"GIF:en svarar {proc.out.strip()[-3:] or proc.code}, inte 200")
    readme = readme_at(ctx)
    if readme is None or url not in readme:
        problems.append("README länkar inte till GIF:en")
    tracked = git(ctx, "ls-tree", "-r", "--name-only", ctx.sha).out.splitlines()
    media = [p for p in tracked if p.lower().endswith((".gif", ".mp4"))]
    if media:
        problems.append(f"{len(media)} .gif/.mp4 i git")
    if problems:
        return Check(False, "; ".join(problems))
    return Check(True, "GIF:en svarar 200, README länkar dit, inga .gif/.mp4 i git")


def v04(ctx: Ctx) -> Check:
    ref = obj(obj(gh_json(["api", f"repos/{ctx.repo}/git/ref/tags/{TAG}"])).get("object"))
    target = ref.get("sha")
    if ref.get("type") == "tag":  # annotated tag: follow it to the commit
        tag = obj(gh_json(["api", f"repos/{ctx.repo}/git/tags/{target}"]))
        target = obj(tag.get("object")).get("sha")
    problems: list[str] = []
    if target != ctx.sha:
        problems.append(f"{TAG} pekar på {str(target)[:12]}, inte sha")
    green = r05(ctx)
    if not green.ok:
        problems.append(f"R-05: {green.detail}")
    rel = obj(gh_json(["release", "view", TAG, "--repo", ctx.repo, "--json", "isDraft,body"]))
    if rel.get("isDraft") is not False:
        problems.append("releasen är ett utkast")
    if "just demo" not in str(rel.get("body") or ""):
        problems.append("releasetexten säger inte hur man kör demon (just demo)")
    if problems:
        return Check(False, "; ".join(problems))
    return Check(True, f"{TAG} på sha, R-05 grön, releasen publicerad")


def v05(ctx: Ctx) -> Check:
    readme = readme_at(ctx)
    if readme is None:
        return Check(False, "README.md finns inte på sha")
    problems = [f"saknar {s!r}" for s in README_NEEDS if s not in readme]
    headings = "\n".join(ln for ln in readme.splitlines() if ln.lstrip().startswith("#")).lower()
    problems += [f"rubrik saknas: {h}" for h in README_HEADINGS if not re.search(h, headings)]
    words = re.findall(r"[a-zåäö]+", readme.lower())
    en = sum(w in EN_WORDS for w in words)
    sv = sum(w in SV_WORDS for w in words)
    if en <= 3 * sv:
        problems.append(f"ser inte engelsk ut ({en} engelska mot {sv} svenska småord)")
    if problems:
        return Check(False, "; ".join(problems))
    return Check(True, "engelsk README med snabbstart, säkerhetsvarning, krav, licenser och GIF")


SITL = ci_rows(("sitl-ok",), "sitl (", "sitl")
PANEL = ci_rows(("ci-ok", "panel-e2e"), None, "panel-e2e")
OFFLINE = ci_rows(("sitl-ok",), "offline", "offline")


def bound(check: Callable[[Ctx, str], Check], *ids: str) -> dict[str, Callable[[Ctx], Check]]:
    def one(rid: str) -> Callable[[Ctx], Check]:
        return lambda ctx: check(ctx, rid)

    return {rid: one(rid) for rid in ids}


AUTO: dict[str, Callable[[Ctx], Check]] = {
    "R-01": r01,
    "R-02": r02,
    "R-03": r03,
    "R-04": r04,
    "R-04b": r04b,
    "R-05": r05,
    "R-06": r06,
    "R-07": r07,
    "R-08": merged_pr_gates,
    "R-09": merged_pr_gates,
    "R-10": r10,
    "R-11": r11,
    "F-01": f01,
    "F-05": f05,
    "F-06": f06,
    "V-01": v01,
    "V-02": v02,
    "V-04": v04,
    "V-05": v05,
    **bound(PANEL, "F-03", "F-13"),
    **bound(OFFLINE, "F-07"),
    **bound(SITL, "F-08", "F-09", "F-10", "F-11", "F-12"),
    **bound(SITL, *(f"S-{n:02d}" for n in range(1, 26) if n != 24)),
}
# Rows with a part the owner signs off in the manual file (M and A + M).
MANUAL_IDS = frozenset({"F-02", "F-04", "R-06", "S-24", "V-01", "V-02", "V-03", "V-06"})


# --- manual rows ---------------------------------------------------------------------


def load_manual(root: Path) -> dict[str, Any]:
    path = root / MANUAL_FILE
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise UsageError(f"kan inte läsa {MANUAL_FILE}: {exc}") from exc


def manual(entry: Any) -> tuple[bool | None, str]:
    if not isinstance(entry, dict):
        return None, f"saknas i {MANUAL_FILE}"
    e = obj(entry)
    by = str(e.get("by") or "").strip()
    note = str(e.get("note") or "").strip()
    date = str(e.get("date") or "").strip()
    if e.get("pass") is not True:
        return False, "ägaren har inte godkänt" + (f": {note}" if note else "")
    try:
        dt.date.fromisoformat(date)
    except ValueError:
        return False, "pass = true men date är inte YYYY-MM-DD"
    if not by:
        return False, "pass = true men by saknas"
    return True, f"godkänd av {by} {date}" + (f": {note}" if note else "")


# --- evaluation ----------------------------------------------------------------------


def parts(typ: str) -> set[str]:
    return set(re.findall(r"[AM]", typ.upper()))


def automatic(ctx: Ctx, rid: str) -> Check:
    fn = AUTO.get(rid)
    if fn is None:
        return Check(False, f"ingen automatisk kontroll för {rid} i v0_1_0.py")
    try:
        return fn(ctx)
    except (GhError, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        return Check(False, f"kunde inte kontrollera: {exc}"[:300])


def evaluate(ctx: Ctx, rid: str, manual_data: dict[str, Any]) -> Row:
    entry = ctx.entries.get(rid)
    if entry is None:
        return Row(rid, "?", False, "saknas i ID-listan (SD_ACCEPT_IDS)", "", ctx.sha)
    typ = str(entry.get("typ", ""))
    # The script's own knowledge adds parts, so a typo in the list cannot drop one.
    want = parts(typ)
    if rid in AUTO:
        want.add("A")
    if rid in MANUAL_IDS:
        want.add("M")
    results: list[tuple[str, bool | None, str]] = []
    if "A" in want:
        a = automatic(ctx, rid)
        results.append(("A", a.ok, a.detail))
    if "M" in want:
        results.append(("M", *manual(manual_data.get(rid))))
    if not results:
        return Row(rid, typ, False, f"okänd typ {typ!r}", str(entry.get("dod_ref", "")), ctx.sha)
    states = [r[1] for r in results]
    passed = False if False in states else None if None in states else True
    detail = results[0][2] if len(results) == 1 else "; ".join(f"{k}: {d}" for k, _, d in results)
    return Row(rid, typ, passed, detail, str(entry.get("dod_ref", "")), ctx.sha)


def phase_ids(phase: str, entries: dict[str, dict[str, Any]]) -> list[str]:
    ids = list(dict.fromkeys([*KNOWN_IDS, *entries]))
    return [i for i in ids if (i in POST_IDS) == (phase == "post")]


# --- input ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> dict[str, str]:
    values = {"phase": "pre", "sha": ""}
    for raw in argv:
        key, sep, value = raw.partition("=")
        if not sep or key not in values:
            raise UsageError(f"'{raw}' känns inte igen. Använd phase=pre|post och sha=<sha>.")
        values[key] = value
    if values["phase"] not in PHASES:
        raise UsageError(f"phase måste vara pre eller post, inte '{values['phase']}'.")
    return values


def load_ids(root: Path) -> dict[str, dict[str, Any]]:
    given = os.environ.get("SD_ACCEPT_IDS", "")
    if not given:
        raise UsageError("SD_ACCEPT_IDS är inte satt. Sätt den till ID-listans sökväg.")
    path = Path(given) if Path(given).is_absolute() else root / given
    if not path.is_file():
        raise UsageError(f"ID-listan {given} saknas (SD_ACCEPT_IDS).")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise UsageError(f"kan inte läsa ID-listan {given}: {exc}") from exc
    entries: dict[str, dict[str, Any]] = {}
    for e in arr(data.get("id")):
        entry = obj(e)
        if entry.get("id"):
            entries[str(entry["id"])] = entry
    if not entries:
        raise UsageError(f"ID-listan {given} har inga [[id]]-poster.")
    return entries


def resolve_sha(root: Path, sha: str) -> str:
    proc = run(["git", "rev-parse", "--verify", f"{sha or 'main'}^{{commit}}"], root)
    full = proc.out.strip()
    if proc.code != 0 or not re.fullmatch(r"[0-9a-f]{40}", full):
        raise UsageError(f"hittar inte commiten '{sha or 'main'}' (git rev-parse).")
    return full


# --- output --------------------------------------------------------------------------


def write_report(root: Path, phase: str, sha: str, now: dt.datetime, rows: list[Row]) -> Path:
    out_dir = root / REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"accept-v{VERSION}-{phase}-{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    report = {
        "version": VERSION,
        "phase": phase,
        "sha": sha,
        "created_at": now.isoformat(timespec="seconds"),
        "pass": all(r.passed is True for r in rows),
        "rows": [r.as_json() for r in rows],
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def print_table(rows: list[Row]) -> None:
    print(f"{'ID':<6} {'Typ':<5} {'Godkänd':<7} Detalj")
    for r in rows:
        print(f"{r.id:<6} {r.typ.replace(' ', ''):<5} {TEXT[r.passed]:<7} {r.detalj}")


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    root = root or REPO
    try:
        args = parse_args(list(sys.argv[1:] if argv is None else argv))
        entries = load_ids(root)
        manual_data = load_manual(root)
        sha = resolve_sha(root, args["sha"])
    except UsageError as exc:
        print(f"accept: {exc}", file=sys.stderr)
        return EXIT_USAGE
    now = dt.datetime.now(dt.UTC)
    repo = os.environ.get("SD_ACCEPT_REPO") or DEFAULT_REPO_SLUG
    ctx = Ctx(root=root, sha=sha, repo=repo, entries=entries, now=now)
    rows = [evaluate(ctx, rid, manual_data) for rid in phase_ids(args["phase"], entries)]
    path = write_report(root, args["phase"], sha, now, rows)
    print_table(rows)
    green = sum(r.passed is True for r in rows)
    print(f"\n{green} av {len(rows)} rader godkända ({args['phase']}, {sha[:12]}).")
    print(f"Rapport: {path.relative_to(root)}")
    return 0 if green == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
