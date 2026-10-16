"""Block files that must never reach the public repository.

Runs in pre-commit (``--staged``), in the pre-push hook (``--range origin/main..HEAD``)
and in CI (default: every file in ``git ls-files``). Every finding is printed as
``path:line: rule: message`` and the exit code is 1 if anything was found.

Rules:
  a) forbidden paths (planning, agent files, secrets, personal config)
  b) forbidden file types outside ``testdata/`` (logs, media, maps, keys)
  c) files larger than 1 MB (lockfiles excepted)
  d) secret-like content
  e) coordinates far from the public test sites in ``tools/allowed_coordinates.toml``
  f) the owner's private search patterns (``--private-patterns``); only file, line
     and the pattern's line number are printed, never the pattern or the matched text
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import math
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_FILE_BYTES = 1024 * 1024
MAX_FIXTURE_BYTES = 200 * 1024
SIZE_EXEMPT = {"uv.lock", "pnpm-lock.yaml"}
COORD_EXEMPT_FILES = {"uv.lock", "pnpm-lock.yaml"}
# Scanner configuration that has to name secret markers to find them.
SECRET_SCAN_EXEMPT = {".gitleaks.toml"}
MISSING_PATTERNS_MSG = "Skapa config/local/private-patterns.txt (får vara tom)"
DEFAULT_PATTERNS = "config/local/private-patterns.txt"
DEFAULT_ALLOWLIST = "tools/allowed_coordinates.toml"
CMAC = ("CMAC (ArduPilot SITL default)", -35.363261, 149.165230, 5000.0)
EARTH_RADIUS_M = 6_371_000.0
GPS_IFD_TAG = 0x8825  # 34853

ROOT_ONLY_DIRS = ("docs", "planning", "research", "notes")
ANYWHERE_GLOBS = (
    "**/CLAUDE.md",
    "**/CLAUDE.local.md",
    "**/.claude/**",
    "**/.mcp.json",
    "**/secrets/**",
    "**/*.local.*",
    "**/*.local.txt",
    "**/cloudflared/**",
    "**/.cloudflared/**",
)
CONFIG_ALLOWED = ("config/*.example.toml", "config/mediamtx.yml")
FORBIDDEN_EXTENSIONS = {
    ".tlog",
    ".bin",
    ".ulg",
    ".mp4",
    ".webm",
    ".mkv",
    ".mov",
    ".avi",
    ".gif",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".heic",
    ".tif",
    ".tiff",
    ".kml",
    ".kmz",
    ".gpx",
    ".waypoints",
    ".plan",
    ".pem",
    ".key",
    ".p12",
    ".crt",
}
PANEL_PUBLIC = "software/panel/public/"
PANEL_PUBLIC_OK = {".png", ".svg", ".ico"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".heic", ".gif"}
# Built from parts so this file does not match its own rule.
SECRET_MARKERS = ("Tunnel" + "Secret", "Account" + "Tag", "-----" + "BEGIN")

# Coordinate key pairs (latitude key, longitude key alternatives).
KEY_PAIRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("lat", ("lon", "lng")),
    ("latitude", ("longitude",)),
    ("latitude_deg", ("longitude_deg",)),
    ("SIM_OPOS_LAT", ("SIM_OPOS_LNG",)),
    ("home_lat", ("home_lon",)),
)
# x/y count only as 1e7-scaled integers (MAVLink MISSION_ITEM_INT, QGC plans).
INT_ONLY_PAIRS: tuple[tuple[str, tuple[str, ...]], ...] = (("x", ("y",)),)
NEAR_LINES = 3
NUM = r"(-?\d{1,10}(?:[.,]\d+)?)"
HOME_PATTERNS = (
    re.compile(r"--home[= ]\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)"),
    re.compile(r"SIM_HOME\s*=\s*\"?(-?\d+\.\d+),\s*(-?\d+\.\d+)"),
    re.compile(r"home=(-?\d+\.\d+),(-?\d+\.\d+)"),
)
FREE_PAIR = re.compile(r"(?<![\d.])(-?\d{1,2})\.(\d{3,})\s*,\s*(-?\d{1,3})\.(\d{3,})(?![\d])")
DMS = re.compile(
    r"(\d{1,2})°\s*(\d{1,2})['\u2032]\s*(\d{1,2}(?:\.\d+)?)[\"\u2033]?\s*([NS])[,\s]+"
    r"(\d{1,3})°\s*(\d{1,2})['\u2032]\s*(\d{1,2}(?:\.\d+)?)[\"\u2033]?\s*([EW])"
)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.message}"


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float
    radius_m: float


# --- helpers -------------------------------------------------------------------------


def _match(path: str, pattern: str) -> bool:
    """Glob match where ``**/`` may also match zero directories."""
    if fnmatch.fnmatchcase(path, pattern):
        return True
    return pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:])


def in_testdata(path: str) -> bool:
    return "testdata" in PurePosixPath(path).parts[:-1]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def load_places(path: Path) -> list[Place]:
    if not path.is_file():
        return [Place(*CMAC)]
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return [
        Place(str(p["name"]), float(p["lat"]), float(p["lon"]), float(p["radius_m"]))
        for p in data.get("place", [])
    ]


def is_binary(data: bytes) -> bool:
    return b"\0" in data[:8192]


# --- rules ---------------------------------------------------------------------------


def check_path(path: str) -> list[Finding]:
    """Rules a and b: forbidden locations and file types."""
    out: list[Finding] = []
    parts = PurePosixPath(path).parts
    name = parts[-1]
    if parts[0] in ROOT_ONLY_DIRS and len(parts) > 1:
        out.append(Finding(path, 0, "forbidden-path", f"{parts[0]}/ is never committed"))
    for glob in ANYWHERE_GLOBS:
        if _match(path, glob):
            out.append(Finding(path, 0, "forbidden-path", f"matches {glob}"))
            break
    if (name == ".env" or name.startswith(".env.")) and name != ".env.example":
        out.append(Finding(path, 0, "forbidden-path", "local environment file"))
    if (
        parts
        and parts[0] == "config"
        and not any(fnmatch.fnmatchcase(path, g) for g in CONFIG_ALLOWED)
    ):
        out.append(
            Finding(path, 0, "forbidden-path", "only config/*.example.toml and config/mediamtx.yml")
        )
    ext = PurePosixPath(path).suffix.lower()
    panel_icon = path.startswith(PANEL_PUBLIC) and ext in PANEL_PUBLIC_OK
    if ext in FORBIDDEN_EXTENSIONS and not in_testdata(path) and not panel_icon:
        out.append(Finding(path, 0, "forbidden-type", f"{ext} files are not committed"))
    return out


def check_size_and_exif(path: str, data: bytes) -> list[Finding]:
    """Rules b (fixture size, EXIF GPS) and c (size)."""
    out: list[Finding] = []
    if len(data) > MAX_FILE_BYTES and path not in SIZE_EXEMPT:
        out.append(Finding(path, 0, "too-large", f"{len(data)} bytes, limit {MAX_FILE_BYTES}"))
    fixture = in_testdata(path) or path.startswith(PANEL_PUBLIC)
    if fixture and len(data) > MAX_FIXTURE_BYTES:
        out.append(
            Finding(
                path, 0, "too-large", f"fixture is {len(data)} bytes, limit {MAX_FIXTURE_BYTES}"
            )
        )
    if fixture and PurePosixPath(path).suffix.lower() in IMAGE_EXTENSIONS and has_exif_gps(data):
        out.append(Finding(path, 0, "exif-gps", "image contains EXIF GPS data"))
    return out


def has_exif_gps(data: bytes) -> bool:
    from PIL import Image  # dev dependency, imported only when an image is checked

    try:
        with Image.open(io.BytesIO(data)) as img:
            exif = img.getexif()
            return GPS_IFD_TAG in exif or bool(exif.get_ifd(GPS_IFD_TAG))
    except Exception:  # an unreadable image cannot be proven clean
        return True


def check_secrets(path: str, lines: list[str]) -> list[Finding]:
    """Rule d: secret-like content."""
    if path in SECRET_SCAN_EXEMPT:
        return []
    if in_testdata(path) and path.endswith(".test.pem"):
        return []
    out: list[Finding] = []
    for no, line in enumerate(lines, 1):
        for marker in SECRET_MARKERS:
            if marker in line:
                out.append(Finding(path, no, "secret-like", "secret-like content"))
                break
    return out


def _parse_num(raw: str) -> tuple[float, int, bool]:
    """Return (value, decimals, is_integer) for a number with '.' or ',' decimals."""
    norm = raw.replace(",", ".")
    if "." in norm:
        return float(norm), len(norm.split(".")[1]), False
    return float(norm), 0, True


def _key_regex(key: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])\W{{0,6}}?\s*{NUM}")


_KEY_RE: dict[str, re.Pattern[str]] = {}


def _keyed_values(lines: list[str], key: str, int_only: bool) -> list[tuple[int, float, bool]]:
    rx = _KEY_RE.setdefault(key, _key_regex(key))
    found: list[tuple[int, float, bool]] = []
    for no, line in enumerate(lines, 1):
        for m in rx.finditer(line):
            value, decimals, is_int = _parse_num(m.group(1))
            digits = len(m.group(1).lstrip("-"))
            if is_int and digits >= 7:
                found.append((no, value * 1e-7, False))
            elif not int_only and not is_int and decimals >= 3:
                found.append((no, value, decimals == 3))
    return found


def find_coordinates(lines: list[str]) -> list[tuple[int, float, float]]:
    """Rule e patterns 1-6. Returns (line, lat, lon) candidates."""
    found: list[tuple[int, float, float]] = []
    key_lines: set[int] = set()
    for pairs, int_only in ((KEY_PAIRS, False), (INT_ONLY_PAIRS, True)):
        for lat_key, lon_keys in pairs:
            lats = _keyed_values(lines, lat_key, int_only)
            lons = [v for k in lon_keys for v in _keyed_values(lines, k, int_only)]
            if not int_only:
                key_lines.update(n for n, _, _ in lats)
                key_lines.update(n for n, _, _ in lons)
            for lat_no, lat, _ in lats:
                for lon_no, lon, _ in lons:
                    if abs(lat_no - lon_no) <= NEAR_LINES:
                        found.append((lat_no, lat, lon))
    for no, line in enumerate(lines, 1):
        for rx in HOME_PATTERNS:
            for m in rx.finditer(line):
                found.append((no, float(m.group(1)), float(m.group(2))))
        for m in FREE_PAIR.finditer(line):
            lat = float(f"{m.group(1)}.{m.group(2)}")
            lon = float(f"{m.group(3)}.{m.group(4)}")
            three = len(m.group(2)) == 3 or len(m.group(4)) == 3
            near_key = any(abs(no - k) <= NEAR_LINES for k in key_lines)
            if three and not near_key:
                continue
            found.append((no, lat, lon))
        for m in DMS.finditer(line):
            lat = int(m.group(1)) + int(m.group(2)) / 60 + float(m.group(3)) / 3600
            lon = int(m.group(5)) + int(m.group(6)) / 60 + float(m.group(7)) / 3600
            lat = -lat if m.group(4) == "S" else lat
            lon = -lon if m.group(8) == "W" else lon
            found.append((no, lat, lon))
    return [(n, la, lo) for n, la, lo in found if -90 <= la <= 90 and -180 <= lo <= 180]


def check_coordinates(path: str, lines: list[str], places: list[Place]) -> list[Finding]:
    if path in COORD_EXEMPT_FILES or "generated" in PurePosixPath(path).parts:
        return []
    out: list[Finding] = []
    seen: set[int] = set()
    for no, lat, lon in find_coordinates(lines):
        if no in seen:
            continue
        if any(haversine_m(lat, lon, p.lat, p.lon) <= p.radius_m for p in places):
            continue
        seen.add(no)
        out.append(Finding(path, no, "coordinate", "coordinate outside the public test sites"))
    return out


def load_private_patterns(path: Path) -> list[tuple[int, Callable[[str], bool]]]:
    patterns: list[tuple[int, Callable[[str], bool]]] = []
    for no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("re:"):
            rx = re.compile(line[3:], re.IGNORECASE)
            patterns.append((no, lambda s, rx=rx: rx.search(s) is not None))
        else:
            needle = line.casefold()
            patterns.append((no, lambda s, needle=needle: needle in s.casefold()))
    return patterns


def check_private(
    path: str, lines: list[str], patterns: list[tuple[int, Callable[[str], bool]]]
) -> list[Finding]:
    out: list[Finding] = []
    for pat_no, matches in patterns:
        if matches(path):
            out.append(Finding(path, 0, "private-pattern", f"matches pattern line {pat_no}"))
        for no, line in enumerate(lines, 1):
            if matches(line):
                out.append(Finding(path, no, "private-pattern", f"matches pattern line {pat_no}"))
    return out


# --- file sources --------------------------------------------------------------------


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(  # noqa: S603 - fixed git arguments, no shell
        ["git", *args],  # noqa: S607 - git from PATH
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


Reader = Callable[[str], bytes | None]


def _tree_reader(root: Path) -> Reader:
    def from_tree(p: str) -> bytes | None:
        f = root / p
        return f.read_bytes() if f.is_file() and not f.is_symlink() else None

    return from_tree


def list_files(root: Path, mode: str, rev_range: str | None) -> tuple[list[str], Reader]:
    from_tree = _tree_reader(root)

    def from_rev(rev: str) -> Callable[[str], bytes | None]:
        def read(p: str) -> bytes | None:
            try:
                return _git(root, "show", f"{rev}:{p}")
            except subprocess.CalledProcessError:
                return None

        return read

    if mode == "staged":
        names = _git(root, "diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")
        return _split(names), from_rev("")  # ":path" reads the index
    if mode == "range" and rev_range:
        names = _git(root, "diff", "--name-only", "-z", "--diff-filter=ACMR", rev_range)
        tip = rev_range.split("..")[-1] or "HEAD"
        return _split(names), from_rev(tip)
    return _split(_git(root, "ls-files", "-z")), from_tree


def _split(raw: bytes) -> list[str]:
    return [p for p in raw.decode("utf-8", "surrogateescape").split("\0") if p]


# --- main ----------------------------------------------------------------------------


def check_files(
    files: Iterable[str],
    read: Callable[[str], bytes | None],
    places: list[Place],
    patterns: list[tuple[int, Callable[[str], bool]]] | None,
) -> list[Finding]:
    findings: list[Finding] = []
    for path in files:
        findings += check_path(path)
        data = read(path)
        if data is None:
            continue
        findings += check_size_and_exif(path, data)
        if is_binary(data):
            continue
        lines = data.decode("utf-8", "replace").splitlines()
        findings += check_secrets(path, lines)
        findings += check_coordinates(path, lines, places)
        if patterns:
            findings += check_private(path, lines, patterns)
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Block files that must never reach the public repo.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true", help="check files staged for commit")
    group.add_argument("--range", dest="rev_range", help="check files changed in A..B")
    ap.add_argument("--private-patterns", type=Path, help=f"pattern file, e.g. {DEFAULT_PATTERNS}")
    ap.add_argument("--allowlist", type=Path, default=Path(DEFAULT_ALLOWLIST))
    ap.add_argument("--root", type=Path, default=Path())
    ap.add_argument("files", nargs="*", help="check these files (relative to --root) instead")
    args = ap.parse_args(argv)
    root: Path = args.root.resolve()

    exit_code = 0
    patterns = None
    if args.private_patterns is not None:
        pfile = root / args.private_patterns  # an absolute path stays absolute
        if pfile.is_file():
            patterns = load_private_patterns(pfile)
        elif os.environ.get("CI", "").lower() != "true":
            print(MISSING_PATTERNS_MSG, file=sys.stderr)
            exit_code = 1

    if args.files:
        files, read = list(args.files), _tree_reader(root)
    else:
        mode = "staged" if args.staged else "range" if args.rev_range else "all"
        files, read = list_files(root, mode, args.rev_range)

    allowlist = args.allowlist if args.allowlist.is_absolute() else root / args.allowlist
    findings = check_files(files, read, load_places(allowlist), patterns)
    for f in findings:
        print(f)
    if findings:
        print(f"{len(findings)} problem(s) found. Fix them; never commit around this check.")
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
