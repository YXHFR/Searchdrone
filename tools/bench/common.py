"""Shared parts of the image bank tools (``tools/bench``): settings, the data folder,
downloads with resume and byte ranges, and safe file names.

Everything the bank downloads lands under ``<data_dir>/datasets/`` (outside the
repository, never committed, uploaded or cached in CI); results go to ``build/bench/``
(gitignored) and hold only numbers.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import BinaryIO

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "tools" / "bench"
RESULTS = ROOT / "build" / "bench"
USER_AGENT = "searchdrone-bench (+https://github.com/YXHFR/Searchdrone)"
TIMEOUT_S = 120
CHUNK = 1 << 20
# One request asks for at most this much: some hosts (Google Drive) answer an open-ended
# request for a big file with a quota page but serve bounded ranges.
RANGE_SPAN = 256 << 20


class BenchError(Exception):
    """A Swedish message for the owner and the exit code (1 a check failed, 2 bad use)."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


# --- settings -------------------------------------------------------------------------


def parse_args(argv: Sequence[str], allowed: Sequence[str], recipe: str) -> dict[str, str]:
    """``key=value`` settings (the repository's rule for just recipes)."""
    out: dict[str, str] = {}
    for arg in argv:
        key, sep, value = arg.partition("=")
        if not sep or not key:
            raise BenchError(f"just {recipe}: '{arg}' är inte på formen nyckel=värde.", 2)
        if key not in allowed:
            known = ", ".join(sorted(allowed))
            raise BenchError(f"just {recipe}: okänd nyckel '{key}'. Tillåtna nycklar: {known}.", 2)
        out[key] = value
    return out


def flag(args: Mapping[str, str], key: str) -> bool:
    value = args.get(key, "0").lower()
    if value not in ("0", "1", "true", "false", "no", "yes"):
        raise BenchError(f"{key} ska vara 1 eller 0, inte '{value}'.", 2)
    return value in ("1", "true", "yes")


def data_dir(args: Mapping[str, str], env: Mapping[str, str] | None = None) -> Path:
    """``data=<path>``, else ``SD_DATA_DIR`` (environment or ``.env``), else ``data_dir``
    in ``config/ground.toml``."""
    env = os.environ if env is None else env
    if args.get("data"):
        return Path(args["data"]).expanduser().resolve()
    if env.get("SD_DATA_DIR"):
        return Path(env["SD_DATA_DIR"]).expanduser().resolve()
    dotenv = ROOT / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if line.startswith("SD_DATA_DIR=") and line.partition("=")[2].strip():
                return Path(line.partition("=")[2].strip()).expanduser().resolve()
    from sd_ground.config import ConfigError, load

    try:
        return load(ROOT).data_dir
    except ConfigError:
        pass
    raise BenchError("Datamappen är okänd. Kör: just setup-config (eller ange data=<mapp>).", 2)


def say(message: str) -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --- names ----------------------------------------------------------------------------

_SAFE_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ +()-]{0,200}$")


def safe_relpath(name: str) -> PurePosixPath:
    """An archive member name as a relative path that stays inside its folder: no
    absolute paths, no ``..``, no odd characters. Anything else is refused."""
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or len(path.parts) > 8:
        raise BenchError(f"Otillåtet filnamn i arkivet: {name!r}")
    for part in path.parts:
        if part in (".", "..") or not _SAFE_PART.match(part):
            raise BenchError(f"Otillåtet filnamn i arkivet: {name!r}")
    return path


def inside(base: Path, rel: PurePosixPath) -> Path:
    target = (base / Path(*rel.parts)).resolve()
    if not target.is_relative_to(base.resolve()):
        raise BenchError(f"Filen hamnar utanför {base}: {rel}")
    return target


# --- downloads ------------------------------------------------------------------------

Opener = Callable[[urllib.request.Request], BinaryIO]


class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to another https address (hosts move files to storage)."""

    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> urllib.request.Request | None:
        if not newurl.startswith("https://"):
            raise BenchError(f"Servern skickar vidare till en adress utan https: {newurl}", 2)
        return super().redirect_request(req, fp, code, msg, headers, newurl)  # type: ignore[arg-type]


_OPENER = urllib.request.build_opener(_HttpsOnlyRedirects)


def _open(req: urllib.request.Request) -> BinaryIO:
    return _OPENER.open(req, timeout=TIMEOUT_S)  # https only: request() and the redirects


def request(url: str, headers: Mapping[str, str] | None = None) -> urllib.request.Request:
    if not url.startswith("https://"):
        raise BenchError(f"Bara https-adresser hämtas: {url}", 2)
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})  # noqa: S310


def _content_range_total(value: str | None) -> int | None:
    if not value or "/" not in value:
        return None
    total = value.rsplit("/", 1)[1].strip()
    return int(total) if total.isdigit() else None


def _refuse_html(resp: BinaryIO, url: str) -> None:
    ctype = str(getattr(resp, "headers", {}).get("Content-Type", ""))
    if ctype.startswith("text/html"):
        raise BenchError(
            f"Servern svarade med en webbsida i stället för filen ({url}). Google Drive gör så "
            "när dagens nedladdningskvot är slut: försök igen om ett dygn."
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    url: str,
    dest: Path,
    size: int,
    sha256: str,
    opener: Opener = _open,
    progress: Callable[[int, int], None] | None = None,
    span: int = RANGE_SPAN,
) -> None:
    """Download ``url`` to ``dest`` in ranges of at most ``span`` bytes, resuming a
    ``dest.partial`` left by a stopped run, and check its size and SHA-256. A wrong file
    is never kept: it is renamed ``*.bad`` and the run stops."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == size and sha256_file(dest) == sha256:
        return
    part = dest.with_name(dest.name + ".partial")
    have = part.stat().st_size if part.is_file() else 0
    if have > size:
        part.unlink()
        have = 0
    while have < size:
        before = have
        try:
            end = min(have + span, size) - 1
            resp = opener(request(url, {"Range": f"bytes={have}-{end}"}))
        except (urllib.error.URLError, OSError) as exc:
            raise BenchError(
                f"Nedladdningen avbröts ({exc}). Kör samma kommando igen: den fortsätter."
            ) from exc
        with resp:
            _refuse_html(resp, url)
            if getattr(resp, "status", 206) == 200 and have:
                part.unlink()  # the server ignored the range: start over
                have = before = 0
            got = getattr(resp, "headers", {}).get("Content-Range")
            if have and got is not None and not got.startswith(f"bytes {have}-"):
                raise BenchError(f"Servern svarade inte med den begärda delen av filen: {url}")
            total = _content_range_total(got)
            if total is not None and total != size:
                raise BenchError(f"Filen på servern är {total} byte, väntat {size}: {url}")
            with part.open("ab") as fh:
                while have <= size and (chunk := resp.read(CHUNK)):
                    fh.write(chunk)
                    have += len(chunk)
                    if progress:
                        progress(have, size)
        if have == before:
            raise BenchError(f"Servern skickade inget mer efter {have} byte: {url}")
    got = sha256_file(part)
    if part.stat().st_size != size or got != sha256:
        bad = dest.with_name(dest.name + ".bad")
        part.replace(bad)
        raise BenchError(
            f"Kontrollsumman stämmer inte för {dest.name} (SHA-256 {got}, väntat {sha256}). "
            f"Filen ligger kvar som {bad.name} och används inte."
        )
    part.replace(dest)


class RangeFile(io.RawIOBase):
    """A remote file read with HTTP byte ranges, so ``zipfile`` can read single members
    of a large archive without downloading all of it."""

    def __init__(self, url: str, opener: Opener = _open) -> None:
        super().__init__()
        self.url = url
        self._opener = opener
        self._pos = 0
        with opener(request(url, {"Range": "bytes=0-0"})) as resp:
            _refuse_html(resp, url)
            total = _content_range_total(getattr(resp, "headers", {}).get("Content-Range"))
        if total is None:
            raise BenchError(f"Servern ger inte delar av filen (HTTP range): {url}")
        self.size = total

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def readinto(self, buffer: bytearray | memoryview) -> int:  # type: ignore[override]
        n = min(len(buffer), self.size - self._pos, RANGE_SPAN)
        if n <= 0:
            return 0
        req = request(self.url, {"Range": f"bytes={self._pos}-{self._pos + n - 1}"})
        with self._opener(req) as resp:
            _refuse_html(resp, self.url)
            got = getattr(resp, "headers", {}).get("Content-Range") or ""
            if getattr(resp, "status", 206) != 206 or not got.startswith(f"bytes {self._pos}-"):
                raise BenchError(f"Servern svarade inte med den begärda delen av filen: {self.url}")
            data = resp.read(n)
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)
