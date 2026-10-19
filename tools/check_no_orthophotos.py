"""Fail if anything that looks like an orthophoto is in git, a CI artifact or a CI cache.

Orthophotos (Lantmäteriet's, with the owner's own login) live only on the owner's
computer, in the gitignored ``build/``. Nothing here downloads one, but a copied file, a
world built from one or a snapshot of that world must never be committed, uploaded or
cached either. Usage:

- ``check_no_orthophotos.py PATH...``: every file under the folders or files given, by
  name and by content whatever it is called (CI runs it on what it uploads or caches);
  a path that does not exist is fine;
- ``check_no_orthophotos.py --git``: every file in every commit reachable from any ref,
  so a photo that was committed once and deleted again is found too (CI's lint job has
  the whole history).

A file is flagged when:

- its name is an orthophoto's: ``ortho*.npy``, ``ortho.toml``, ``*orto*``/``*ortho*``
  rasters, or an aerial format (JPEG 2000, ECW, MrSID);
- it is a NumPy array of bytes shaped like a picture (height x width x 3 or 4, both at
  least ``MIN_SIDE_PX``), the shape ``just ortho-fetch`` writes;
- it is a GeoTIFF (a TIFF with geographic tags);
- it is a world's manifest, a snapshot's metrics or attribution that says it was built
  from an orthophoto.

Exit code 0 when nothing was found, 1 otherwise. Standard library only.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import os
import struct
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

MIN_SIDE_PX = 256
NAME_GLOBS = (
    "ortho*.npy",
    "ortho.toml",
    "ortho.part.*",
    "*orto*.tif",
    "*orto*.tiff",
    "*ortho*.tif",
    "*ortho*.tiff",
    "*.jp2",
    "*.j2k",
    "*.ecw",
    "*.sid",
)
# What sd_sim writes next to anything built from a photo (terrain.py, snapshot.py).
MARKERS = (b'"ground_texture": "orthophoto"', "Lantmäteriet's orthophoto".encode())
MARKER_SUFFIXES = {".json", ".txt", ".toml"}
# Files that name the markers to find or write them (this check and its tests).
MARKER_EXEMPT = {"tools/check_no_orthophotos.py", "tools/tests/test_check_no_orthophotos.py"}
CONTENT_SUFFIXES = {".npy", ".tif", ".tiff"} | MARKER_SUFFIXES
GEOTIFF_TAGS = {33550, 33922, 34264, 34735}  # pixel scale, tie point, transform, geo keys
TIFF_HEAD = 64 * 1024


def name_rule(path: str) -> str | None:
    name = PurePosixPath(path).name.lower()
    for glob in NAME_GLOBS:
        if fnmatch.fnmatchcase(name, glob):
            return f"named like an orthophoto ({glob})"
    return None


def npy_shape(data: bytes) -> tuple[str, tuple[int, ...]] | None:
    """The dtype and shape of a .npy file from its header, or None."""
    if not data.startswith(b"\x93NUMPY") or len(data) < 10:
        return None
    major = data[6]
    if major == 1:
        (length,) = struct.unpack_from("<H", data, 8)
        start = 10
    else:
        if len(data) < 12:
            return None
        (length,) = struct.unpack_from("<I", data, 8)
        start = 12
    try:
        header = ast.literal_eval(data[start : start + length].decode("latin-1"))
        return str(header["descr"]), tuple(int(v) for v in header["shape"])
    except (ValueError, SyntaxError, KeyError, TypeError):
        return ("?", ())


def npy_rule(data: bytes) -> str | None:
    found = npy_shape(data)
    if found is None:
        return None
    descr, shape = found
    if descr == "?":
        return "a NumPy file whose header cannot be read"
    picture = len(shape) == 3 and shape[2] in (3, 4) and min(shape[:2]) >= MIN_SIDE_PX
    if descr.lstrip("<>|=") == "u1" and picture:
        return f"a picture as a NumPy array ({shape[0]} x {shape[1]} x {shape[2]} bytes)"
    return None


def tiff_rule(data: bytes) -> str | None:
    """GeoTIFF: a classic TIFF whose first directory has geographic tags."""
    if data[:4] not in (b"II*\x00", b"MM\x00*"):
        return None
    end = "<" if data[:2] == b"II" else ">"
    try:
        (offset,) = struct.unpack_from(f"{end}I", data, 4)
        (count,) = struct.unpack_from(f"{end}H", data, offset)
        tags = {struct.unpack_from(f"{end}H", data, offset + 2 + 12 * i)[0] for i in range(count)}
    except struct.error:
        return None
    return "a GeoTIFF" if tags & GEOTIFF_TAGS else None


def marker_rule(path: str, data: bytes) -> str | None:
    if path in MARKER_EXEMPT or PurePosixPath(path).suffix.lower() not in MARKER_SUFFIXES:
        return None
    for marker in MARKERS:
        if marker in data:
            return "says it was built from an orthophoto"
    return None


def check(path: str, data: bytes | None) -> list[str]:
    out = [r for r in (name_rule(path),) if r]
    if data is not None:
        out += [r for r in (npy_rule(data), tiff_rule(data), marker_rule(path, data)) if r]
    return out


def needs_content(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in CONTENT_SUFFIXES


# --- sources -----------------------------------------------------------------------


def walk(paths: list[Path]) -> Iterator[tuple[str, Path]]:
    for root in paths:
        if root.is_file():
            yield root.as_posix(), root
        elif root.is_dir():
            for folder, _, files in os.walk(root):
                for name in sorted(files):
                    f = Path(folder) / name
                    yield f.as_posix(), f


def _read_head(path: Path) -> bytes:
    """The start of the file: enough for every header rule; markers need the whole text."""
    suffix = path.suffix.lower()
    with path.open("rb") as fh:
        return fh.read() if suffix in MARKER_SUFFIXES else fh.read(TIFF_HEAD)


def check_paths(paths: list[Path]) -> list[str]:
    findings = []
    for name, f in walk(paths):
        if f.is_symlink():
            continue
        data = _read_head(f)
        findings += [f"{name}: {why}" for why in check(name, data)]
    return findings


def _git(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(  # noqa: S603 - fixed git arguments, no shell
        ["git", *args],  # noqa: S607 - git from PATH
        cwd=root,
        input=stdin,
        check=True,
        capture_output=True,
    ).stdout


def check_git(root: Path) -> list[str]:
    """Every blob in every commit reachable from any ref, by the names it had."""
    objects = _git(root, "rev-list", "--objects", "--all").decode("utf-8", "surrogateescape")
    named: dict[str, set[str]] = {}
    for line in objects.splitlines():
        sha, _, path = line.partition(" ")
        if path:
            named.setdefault(sha, set()).add(path)
    findings: set[str] = set()
    wanted = []
    for sha, names in named.items():
        for name in names:
            findings.update(f"{name} (git {sha[:12]}): {why}" for why in check(name, None))
            if needs_content(name):
                wanted.append(sha)
    if wanted:
        batch = _git(root, "cat-file", "--batch", stdin="\n".join(wanted).encode() + b"\n")
        at = 0
        for sha in wanted:
            nl = batch.index(b"\n", at)
            head = batch[at:nl].split()
            at = nl + 1
            if len(head) < 3 or head[1] != b"blob":
                continue
            size = int(head[2])
            data = batch[at : at + size]
            at += size + 1
            for name in named[sha]:
                findings.update(f"{name} (git {sha[:12]}): {why}" for why in check(name, data))
    return sorted(findings)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fail if anything looks like an orthophoto.")
    ap.add_argument("--git", action="store_true", help="every file in the git history")
    ap.add_argument("--root", type=Path, default=Path())
    ap.add_argument("paths", nargs="*", type=Path, help="folders or files to check")
    args = ap.parse_args(argv)
    if not args.git and not args.paths:
        ap.error("ge --git eller sökvägar att kontrollera")
    findings = check_git(args.root) if args.git else []
    findings += check_paths(args.paths)
    for f in findings:
        print(f)
    if findings:
        print(
            f"{len(findings)} fil(er) som liknar ortofoton. Ortofoton får bara finnas på "
            "ägarens dator i build/: aldrig i git, i CI-artefakter eller i CI-cachen."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
