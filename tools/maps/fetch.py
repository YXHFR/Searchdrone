"""Download the offline map pack for the panel: ``just maps-fetch [key=value ...]``.

Keys (all optional): ``bbox=min_lon,min_lat,max_lon,max_lat``, ``maxzoom=15``,
``name=demo``, ``dry=1`` (only show how large the pack would be).

1. Installs ``go-pmtiles`` in the locked version as ``~/.local/bin/sd-pmtiles-<version>``
   (SHA-256 of the download and of the program are checked, ``tools/maps/maps-extract.toml``).
   The build's ETag must match ``protomaps_etag``; tile data is otherwise trusted on TLS.
2. Cuts the area out of the Protomaps build named in ``config/maps.toml`` (or the example
   file) with ``pmtiles extract``, straight into ``<data_dir>/tiles/<name>.pmtiles``.
3. Copies glyphs and sprites from ``basemaps-assets`` at a fixed commit to
   ``<data_dir>/tiles/assets/`` and checks their digest.
4. Writes ``<data_dir>/tiles/<name>.json``: build, bbox, maxzoom and SHA-256 of the pack.

Needs internet only while it runs. Nothing is written inside the repository. Exit codes:
0 done, 1 a check failed, 2 bad arguments or the build no longer exists.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sd_ground.api.tiles import PACK_NAME

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "tools" / "maps" / "maps-extract.toml"
KEYS = ("bbox", "maxzoom", "name", "dry")
BUILDS_PAGE = "https://maps.protomaps.com/builds"
TIMEOUT_S = 60
MAX_BINARY_BYTES = 200 * 1024 * 1024
USER_AGENT = "searchdrone-maps-fetch (+https://github.com/YXHFR/Searchdrone)"

Bbox = tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
Runner = Callable[[Sequence[str]], None]


class FetchError(Exception):
    """A Swedish message for the owner and the exit code."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Settings:
    build: str
    etag: str  # the build's ETag: the archive is checked to be exactly this build
    name: str
    maxzoom: int
    bbox: Bbox
    dry: bool


# --- settings -----------------------------------------------------------------------


def parse_args(argv: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for arg in argv:
        key, sep, value = arg.partition("=")
        if not sep or not key or arg.startswith("-"):
            raise FetchError(f"Skriv inställningar som nyckel=värde, inte '{arg}'.", 2)
        if key not in KEYS:
            raise FetchError(f"okänd nyckel '{key}' (giltiga: {', '.join(KEYS)})", 2)
        out[key] = value
    return out


def bbox_around(lat: float, lon: float, half_m: float) -> Bbox:
    dlat = half_m / 110_540
    dlon = half_m / (111_320 * math.cos(math.radians(lat)))
    return (round(lon - dlon, 6), round(lat - dlat, 6), round(lon + dlon, 6), round(lat + dlat, 6))


def parse_bbox(text: str) -> Bbox:
    try:
        parts = [float(p) for p in text.split(",")]
    except ValueError:
        parts = []
    if len(parts) != 4:
        raise FetchError("bbox ska vara min_lon,min_lat,max_lon,max_lat.", 2)
    min_lon, min_lat, max_lon, max_lat = parts
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise FetchError("bbox är ogiltig (min måste vara mindre än max, inom ±180/±90).", 2)
    return min_lon, min_lat, max_lon, max_lat


def load_settings(root: Path, args: dict[str, str]) -> Settings:
    conf: dict[str, Any] = {}
    for path in (root / "config" / "maps.example.toml", root / "config" / "maps.toml"):
        if path.is_file():
            conf.update(tomllib.loads(path.read_text(encoding="utf-8")).get("maps", {}))
    build = str(conf.get("protomaps_build", ""))
    if not (len(build) == 8 and build.isdigit()):
        raise FetchError("protomaps_build i config/maps.toml ska vara ett datum YYYYMMDD.", 2)
    name = args.get("name", str(conf.get("name", "demo")))
    if not re.fullmatch(PACK_NAME, name):
        raise FetchError("name: högst 64 tecken a-z, 0-9, - och _ (först a-z eller 0-9).", 2)
    try:
        maxzoom = int(args.get("maxzoom") or conf.get("maxzoom", 15))
    except ValueError:
        raise FetchError("maxzoom ska vara ett heltal.", 2) from None
    if not 0 <= maxzoom <= 15:
        raise FetchError("maxzoom ska vara 0 till 15 (kartan förstoras själv över 15).", 2)
    if "bbox" in args:
        bbox = parse_bbox(args["bbox"])
    else:
        bbox = bbox_around(
            float(conf["center_lat"]), float(conf["center_lon"]), float(conf["half_size_m"])
        )
    etag = str(conf.get("protomaps_etag", ""))
    if not etag:
        raise FetchError("protomaps_etag saknas i config/maps.toml (se exempelfilen).", 2)
    return Settings(build, etag, name, maxzoom, bbox, args.get("dry", "0") in ("1", "true"))


def data_dir(root: Path) -> Path:
    """The ground station's data root (``data_dir`` in config/ground.toml)."""
    from sd_ground.config import ConfigError, load

    try:
        return load(root).data_dir
    except ConfigError as exc:
        raise FetchError("config/ground.toml går inte att läsa: " + "; ".join(exc.errors)) from None


# --- downloads ----------------------------------------------------------------------


def _request(url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
    if not url.startswith("https://"):
        raise FetchError(f"Bara https-adresser hämtas: {url}")
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})  # noqa: S310


def download(url: str) -> bytes:
    with urllib.request.urlopen(_request(url), timeout=TIMEOUT_S) as resp:  # noqa: S310
        return resp.read(MAX_BINARY_BYTES + 1)


def url_status(url: str) -> tuple[int, str]:
    """HTTP status and ETag of the first byte (some hosts refuse HEAD and unnamed clients)."""
    try:
        req = _request(url, {"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # noqa: S310
            return int(resp.status), str(resp.headers.get("ETag", ""))
    except urllib.error.HTTPError as exc:
        return int(exc.code), ""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_pmtiles(
    lock: dict[str, Any], bin_dir: Path, fetch: Callable[[str], bytes] = download
) -> Path:
    """``bin_dir/pmtiles`` in the locked version; downloaded and checked if needed."""
    # Its own name, so a pmtiles the user installed for something else is never replaced.
    target = bin_dir / f"sd-pmtiles-{lock['version']}"
    if target.is_file() and sha256(target.read_bytes()) == lock["binary_sha256"]:
        return target
    blob = fetch(lock["url"])
    if sha256(blob) != lock["sha256"]:
        raise FetchError("go-pmtiles: kontrollsumman stämmer inte. Inget installerades.")
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        member = tar.getmember("pmtiles")  # only this regular file is read from the archive
        handle = tar.extractfile(member) if member.isfile() else None
        if handle is None:
            raise FetchError("go-pmtiles: arkivet saknar programmet pmtiles.")
        program = handle.read(MAX_BINARY_BYTES + 1)
    if sha256(program) != lock["binary_sha256"]:
        raise FetchError("go-pmtiles: kontrollsumman för programmet stämmer inte.")
    bin_dir.mkdir(parents=True, exist_ok=True)
    tmp = bin_dir / f".{target.name}.download"
    tmp.write_bytes(program)
    tmp.chmod(0o755)
    tmp.replace(target)
    return target


def build_url(lock: dict[str, Any], build: str) -> str:
    return str(lock["build"]["url_template"]).format(build=build)


def check_build(
    url: str, s: Settings, status: Callable[[str], tuple[int, str]] = url_status
) -> None:
    """The build exists and is the one in the config (same ETag), so the tiles that
    ``pmtiles extract`` reads come from that archive and no other."""
    build = s.build
    code, etag = status(url)
    if code == 404:
        raise FetchError(
            f"Kartbygget {build} finns inte längre. Välj ett nytt på {BUILDS_PAGE} "
            "och ändra protomaps_build.",
            2,
        )
    if code >= 400:
        raise FetchError(f"Kartbygget {build} svarar {code}. Försök igen senare.")
    if etag.strip('W/"') != s.etag.strip('W/"'):
        raise FetchError(
            f"Kartbygget {build} har ändrats (ETag {etag or 'saknas'}, väntade {s.etag}). "
            "Inget hämtades."
        )


def run(cmd: Sequence[str]) -> None:
    subprocess.run(list(cmd), check=True)  # noqa: S603 - fixed program, no shell


def extract(program: Path, url: str, out: Path, s: Settings, runner: Runner = run) -> None:
    bbox = ",".join(f"{v:.6f}" for v in s.bbox)
    cmd = [str(program), "extract", url, str(out), f"--bbox={bbox}", f"--maxzoom={s.maxzoom}"]
    if s.dry:
        cmd.append("--dry-run")
    try:
        runner(cmd)
    except subprocess.CalledProcessError as exc:
        raise FetchError(f"pmtiles extract misslyckades (kod {exc.returncode}).") from None


# --- glyphs and sprites -------------------------------------------------------------


def asset_files(root: Path, paths: Sequence[str]) -> list[Path]:
    """Every regular file under the locked paths, sorted. Symlinks are refused."""
    out: list[Path] = []
    for rel in paths:
        p = root / rel
        if p.is_symlink():
            raise FetchError(f"basemaps-assets: {rel} är en symlänk.")
        items = [p] if p.is_file() else sorted(p.rglob("*"))
        for item in items:
            if item.is_symlink():
                raise FetchError(f"basemaps-assets: {item.relative_to(root)} är en symlänk.")
            if item.is_file():
                out.append(item)
    return sorted(out, key=lambda q: q.relative_to(root).as_posix())


def tree_digest(root: Path, files: Sequence[Path]) -> str:
    """SHA-256 over lines ``<sha256 of file>  <path>``, like ``sha256sum``, sorted by path."""
    lines = "".join(f"{sha256(f.read_bytes())}  {f.relative_to(root).as_posix()}\n" for f in files)
    return sha256(lines.encode())


def checkout_assets(lock: dict[str, Any], work: Path, runner: Runner = run) -> None:
    """A sparse checkout of the locked commit (only the locked paths are downloaded)."""
    git = shutil.which("git") or "git"
    runner([git, "init", "-q", str(work)])
    runner([git, "-C", str(work), "config", "core.sparseCheckout", "true"])
    (work / ".git" / "info").mkdir(parents=True, exist_ok=True)
    # Non-cone patterns: "/fonts/OFL.txt" matches the file, "/sprites/v4" the whole directory.
    (work / ".git" / "info" / "sparse-checkout").write_text(
        "".join(f"/{p}\n" for p in lock["paths"]), encoding="utf-8"
    )
    runner([git, "-C", str(work), "fetch", "-q", "--depth=1", "--filter=blob:none",
            lock["repo"], lock["commit"]])  # fmt: skip
    runner([git, "-C", str(work), "-c", "advice.detachedHead=false", "checkout", "-q",
            lock["commit"]])  # fmt: skip


def install_assets(lock: dict[str, Any], tiles: Path, runner: Runner = run) -> int:
    with tempfile.TemporaryDirectory(prefix="sd-maps-") as tmp:
        work = Path(tmp) / "assets"
        try:
            checkout_assets(lock, work, runner)
        except subprocess.CalledProcessError as exc:
            raise FetchError(
                f"basemaps-assets kunde inte hämtas (git, kod {exc.returncode})."
            ) from None
        files = asset_files(work, lock["paths"])
        if len(files) != lock["files"] or tree_digest(work, files) != lock["sha256"]:
            raise FetchError("basemaps-assets: kontrollsumman stämmer inte. Inget kopierades.")
        staging = tiles / ".assets.new"
        shutil.rmtree(staging, ignore_errors=True)
        for f in files:
            dest = staging / f.relative_to(work)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dest)
        old = tiles / ".assets.old"
        shutil.rmtree(old, ignore_errors=True)
        if (tiles / "assets").exists():
            (tiles / "assets").rename(old)
        staging.rename(tiles / "assets")
        shutil.rmtree(old, ignore_errors=True)
        return len(files)


# --- main ---------------------------------------------------------------------------


def fetch(
    s: Settings,
    tiles: Path,
    lock: dict[str, Any],
    *,
    bin_dir: Path,
    runner: Runner = run,
    fetch_url: Callable[[str], bytes] = download,
    status: Callable[[str], tuple[int, str]] = url_status,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
) -> Path | None:
    program = ensure_pmtiles(lock["go_pmtiles"], bin_dir, fetch_url)
    url = build_url(lock, s.build)
    check_build(url, s, status)
    tiles.mkdir(parents=True, exist_ok=True)
    final = tiles / f"{s.name}.pmtiles"
    if s.dry:
        extract(program, url, tiles / f".{s.name}.dry.pmtiles", s, runner)
        return None
    part = tiles / f".{s.name}.part.pmtiles"
    part.unlink(missing_ok=True)
    extract(program, url, part, s, runner)
    if not part.is_file():
        raise FetchError("pmtiles extract skrev ingen fil.")
    count = install_assets(lock["assets"], tiles, runner)
    part.replace(final)
    with final.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    manifest = {
        "name": s.name,
        "protomaps_build": s.build,
        "source": url,
        "bbox": list(s.bbox),
        "maxzoom": s.maxzoom,
        "sha256": digest,
        "bytes": final.stat().st_size,
        "assets_commit": lock["assets"]["commit"],
        "asset_files": count,
        "created_at": now().isoformat(timespec="seconds"),
    }
    (tiles / f"{s.name}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return final


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        settings = load_settings(ROOT, args)
        lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
        tiles = data_dir(ROOT) / "tiles"
        bin_dir = Path(os.environ.get("SD_MAPS_BIN_DIR", "~/.local/bin")).expanduser()
        out = fetch(settings, tiles, lock, bin_dir=bin_dir)
    except FetchError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    if out is None:
        print("Provkörning klar: inget sparades.")
    else:
        size_mb = out.stat().st_size / 1e6
        print(f"Kartpaketet är klart: {out} ({size_mb:.1f} MB). Starta om sd-ground.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
