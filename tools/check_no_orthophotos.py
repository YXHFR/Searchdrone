"""Fail if anything that looks like an orthophoto is in git, a CI artifact or a CI cache.

Orthophotos (Lantmäteriet's, with the owner's own login) live only on the owner's
computer, in the gitignored ``build/``. Nothing here downloads one, but a copied file, a
world built from one or a snapshot of that world must never be committed, uploaded or
cached either. Usage:

- ``check_no_orthophotos.py PATH...``: every file under the folders or files given, by
  name and by content whatever it is called (CI runs it on what it uploads or caches);
  a path that does not exist is fine;
- ``check_no_orthophotos.py --git``: every file in every commit reachable from any ref,
  by name and by content, so a photo that was committed once and deleted again is found
  too (CI's lint job has the whole history).

A file is flagged when:

- its name is an orthophoto's: ``ortho*.npy``, ``ortho.toml``, ``*orto*``/``*ortho*``
  rasters, or an aerial format (JPEG 2000, ECW, MrSID);
- it is a NumPy array of bytes shaped like a picture (height x width x 3 or 4, both at
  least ``MIN_SIDE_PX``), the shape ``just ortho-fetch`` writes;
- it is a GeoTIFF (a classic or Big TIFF with geographic tags);
- it is a world's manifest, a snapshot's metrics or attribution, or a ground texture
  (PNG, by the text chunk ``terrain.py`` writes) that says it was built from an
  orthophoto;
- it is a symbolic link under a folder given (``upload-artifact`` would follow it).

What it cannot see: a photo saved as an ordinary picture (PNG, JPEG, WebP) by hand,
with nothing that says where it came from. Such pictures are kept out of git by
``check_forbidden_files.py`` (no images outside ``testdata/`` and the panel's assets) and
none is made in CI, where no photo is ever fetched.

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
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import BinaryIO

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
# What sd_sim writes next to anything built from a photo (terrain.py, snapshot.py), and
# into the ground textures themselves (terrain.PHOTO_PNG_TEXT).
MARKERS = (b'"ground_texture": "orthophoto"', "Lantmäteriet's orthophoto".encode())
PNG_MARKER = b"ground_texture=orthophoto"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MARKER_SUFFIXES = {".json", ".txt", ".toml"}
# Files that name the markers to find or write them (this check and its tests).
MARKER_EXEMPT = {"tools/check_no_orthophotos.py", "tools/tests/test_check_no_orthophotos.py"}
GEOTIFF_TAGS = {33550, 33922, 34264, 34735}  # pixel scale, tie point, transform, geo keys
HEAD = 64 * 1024  # enough for a .npy header and a PNG's chunks before its pixels


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


def tiff_tags(read: Callable[[int, int], bytes]) -> set[int] | None:
    """The tags of a TIFF's first directory, wherever in the file it is; None when it is
    not a TIFF. ``read(offset, size)`` gives the file's bytes there."""
    head = read(0, 16)
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        big = False
    elif head[:4] in (b"II+\x00", b"MM\x00+"):
        big = True
    else:
        return None
    end = "<" if head[:2] == b"II" else ">"
    try:
        if big:
            (offset,) = struct.unpack_from(f"{end}Q", head, 8)
            (count,) = struct.unpack_from(f"{end}Q", read(offset, 8))
            entry, first = 20, 8
        else:
            (offset,) = struct.unpack_from(f"{end}I", head, 4)
            (count,) = struct.unpack_from(f"{end}H", read(offset, 2))
            entry, first = 12, 2
        table = read(offset + first, entry * min(count, 4096))
        return {struct.unpack_from(f"{end}H", table, entry * i)[0] for i in range(min(count, 4096))}
    except struct.error:
        return set()


def tiff_rule(read: Callable[[int, int], bytes]) -> str | None:
    """GeoTIFF: a TIFF whose first directory has geographic tags."""
    tags = tiff_tags(read)
    return "a GeoTIFF" if tags and tags & GEOTIFF_TAGS else None


def marker_rule(path: str, data: bytes) -> str | None:
    if data.startswith(PNG_SIGNATURE):
        return "a picture that says it was made from an orthophoto" if PNG_MARKER in data else None
    if path in MARKER_EXEMPT or PurePosixPath(path).suffix.lower() not in MARKER_SUFFIXES:
        return None
    for marker in MARKERS:
        if marker in data:
            return "says it was built from an orthophoto"
    return None


def _from_bytes(data: bytes) -> Callable[[int, int], bytes]:
    return lambda offset, size: data[offset : offset + size]


def check(
    path: str, data: bytes | None, read: Callable[[int, int], bytes] | None = None
) -> list[str]:
    """What looks like an orthophoto about this file. ``data`` is its start (all of it
    for a text file); ``read(offset, size)``, when given, reaches the rest (a TIFF's
    directory can be anywhere)."""
    out = [r for r in (name_rule(path),) if r]
    if data is not None:
        rules = (npy_rule(data), tiff_rule(read or _from_bytes(data)), marker_rule(path, data))
        out += [r for r in rules if r]
    return out


# --- sources -----------------------------------------------------------------------


def walk(paths: list[Path]) -> Iterator[tuple[str, Path]]:
    """Every file and symbolic link under the paths given (links are not followed)."""
    for root in paths:
        if root.is_symlink() or root.is_file():
            yield root.as_posix(), root
        elif root.is_dir():
            for folder, dirs, files in os.walk(root):
                links = [d for d in dirs if (Path(folder) / d).is_symlink()]
                for name in sorted([*files, *links]):
                    f = Path(folder) / name
                    yield f.as_posix(), f


def check_paths(paths: list[Path]) -> list[str]:
    findings = []
    for name, f in walk(paths):
        if f.is_symlink():
            findings.append(f"{name}: a symbolic link (uploads follow it; it is not checked)")
            continue
        if not f.is_file():
            continue
        with f.open("rb") as fh:
            data = fh.read() if f.suffix.lower() in MARKER_SUFFIXES else fh.read(HEAD)

            def read(offset: int, size: int, fh: BinaryIO = fh) -> bytes:
                fh.seek(offset)
                return fh.read(size)

            findings += [f"{name}: {why}" for why in check(name, data, read)]
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
    """Every blob in every commit reachable from any ref, by the names it had and by its
    content, whatever it was called."""
    objects = _git(root, "rev-list", "--objects", "--all").decode("utf-8", "surrogateescape")
    named: dict[str, set[str]] = {}
    for line in objects.splitlines():
        sha, _, path = line.partition(" ")
        if path:
            named.setdefault(sha, set()).add(path)
    findings: set[str] = set()
    for sha, names in named.items():
        for name in names:
            findings.update(f"{name} (git {sha[:12]}): {why}" for why in check(name, None))
    with subprocess.Popen(
        ["git", "cat-file", "--batch"],  # noqa: S607 - git from PATH
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    ) as cat:
        if cat.stdin is None or cat.stdout is None:  # never: both are pipes
            raise RuntimeError("git cat-file gav inga rör")
        for sha, names in named.items():
            cat.stdin.write(sha.encode() + b"\n")
            cat.stdin.flush()
            head = cat.stdout.readline().split()
            if len(head) < 3:
                continue
            data = cat.stdout.read(int(head[2]) + 1)[:-1]
            if head[1] != b"blob":
                continue
            for name in names:
                findings.update(f"{name} (git {sha[:12]}): {why}" for why in check(name, data))
        cat.stdin.close()
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
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if findings and summary:  # so a blocked upload can be told apart from a failed test
        with Path(summary).open("a", encoding="utf-8") as fh:
            fh.write("### Orthophoto check\n\n" + "".join(f"- `{f}`\n" for f in findings))
    if findings:
        print(
            f"{len(findings)} fil(er) som liknar ortofoton. Ortofoton får bara finnas på "
            "ägarens dator i build/: aldrig i git, i CI-artefakter eller i CI-cachen."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
