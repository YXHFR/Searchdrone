"""Fetch the real trees of a simulator site: ``just trees-fetch site=<name>``.

Every tree is found in Skogsstyrelsen's tree height raster (Trädhöjd, 1 m, from
Lantmäteriet's forest laser scan) and gets its species from Naturvårdsverket's land cover
map (Nationella Marktäckedata 2023, 10 m, CC0). The result is a short table, one row per
tree, in the site's local east/north plane:

    east_m,north_m,height_m,species        species: spruce | pine | deciduous

How a tree is found: the raster is smoothed a little (3 x 3), and every cell of at least
``MIN_TREE_M`` that is the highest within a crown radius that grows with its height
(``crown_radius``) is a tree top. The tree's height is the unsmoothed raster's highest
value next to its top. Its species is the land cover class under it (``species_for``).

The table is written to ``build/trees/<site>/`` with ``trees.toml`` (where and when the
data came from, how many trees, how well their heights match the raster) and
``chm_dm.npy`` (the raster on the site's 1 m grid, row 0 north, decimetres) and is never
committed: the tree height raster's licence could not be confirmed (Skogsstyrelsen's
general terms say CC0, the service itself says CC BY, and the download feed restricts
redistribution), so only the code is public and anyone can fetch the same data. A local
site (``config/site.local.toml``) writes into its own ``build/sites/<name>/``.

Skogsstyrelsen publishes the raster without an account only as one file per county, a
zip archive of several GB whose single TIFF is compressed with Deflate64, so a part of it
cannot be read directly. This tool streams the archive (several ranges in parallel),
decompresses it on the fly and stops as soon as it has the rows that cover the site;
nothing of the county file is stored. The TIFF stores one row per strip, so the rows are
cut out as they pass and written as a small TIFF for GDAL.

Needs internet, GDAL's command-line tools (``gdalwarp``, ``gdal_translate``,
``gdaltransform``) and the Python package ``inflate64`` (Deflate64) only while it runs.
Exit codes: 0 done, 1 a download or a check failed, 2 bad arguments.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import hashlib
import http.client
import io
import json
import math
import struct
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_site
from fetch_site import ROOT, FetchError, SiteDef

KEYS = ("site",)
# Tree height raster (Trädhöjd, Laserdata skog generation 2) per county, without account.
SKS_HOST = "geodpags.skogsstyrelsen.se"
SKS_COUNTY_ZIP = "https://geodpags.skogsstyrelsen.se/geodataport/data/sksTradhojd{county:02d}.zip"
# Skogsstyrelsen's server refuses requests without a browser-like User-Agent.
SKS_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) searchdrone-trees-fetch "
    "(+https://github.com/YXHFR/Searchdrone)"
)
NMD_WMS = "https://geodata.naturvardsverket.se/inspire/lc-nmd/ows"
NMD_LAYER = "LC.LandCoverRaster.Bas.2023.v2.x"
NMD_CELL_M = 10.0
SWEREF = "EPSG:3006"
CHUNK = 16 * 1024 * 1024
PARALLEL = 8
TRIES = 4
TIMEOUT_S = 120
# Rows of the raster are fetched this far outside the site, so the warp has no edge.
MARGIN_M = 30.0
MIN_TREE_M = 3.0
# Tops closer than this are one tree (the highest is kept).
MERGE_M = 1.5
SPECIES = ("spruce", "pine", "deciduous")

# Land cover classes (NMD2023 basskikt v2.x) with trees, and the species share of each.
# 111-118 on dry land, 121-128 on wetland, in the same order.
_FOREST_SHARES: dict[int, tuple[float, float, float]] = {
    1: (0.0, 1.0, 0.0),  # tallskog
    2: (1.0, 0.0, 0.0),  # granskog
    3: (0.5, 0.5, 0.0),  # barrblandskog
    4: (0.4, 0.35, 0.25),  # lövblandad barrskog
    5: (0.0, 0.0, 1.0),  # triviallövskog
    6: (0.0, 0.0, 1.0),  # ädellövskog
    7: (0.0, 0.0, 1.0),  # triviallövskog med ädellövinslag
    8: (0.2, 0.2, 0.6),  # temporärt ej skog (young trees after a felling)
}
# Trees outside the forest classes (parks, gardens, lake shores) are deciduous.
_OTHER_SHARES = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class Tree:
    east: float
    north: float
    height: float
    species: str


# --- settings ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for arg in argv:
        key, sep, value = arg.partition("=")
        if not sep or key not in KEYS:
            raise FetchError(f"okänd inställning {arg!r} (använd site=<namn>)", 2)
        out[key] = value
    if "site" not in out:
        raise FetchError("ange site=<namn>", 2)
    if not fetch_site.SITE_NAME.match(out["site"]):
        raise FetchError("site: a-z, 0-9 och _, högst 32 tecken, först en bokstav", 2)
    return out


def county_of(name: str, root: Path = ROOT) -> int:
    """The county (länskod) whose tree height file covers the site: ``[trees] county``."""
    public = root / "simulation" / "gazebo" / "sites" / name / "site.toml"
    if public.is_file():
        data = tomllib.loads(public.read_text(encoding="utf-8"))
    else:
        local = root / "config" / "site.local.toml"
        data = tomllib.loads(local.read_text(encoding="utf-8")).get("sites", {}).get(name, {})
    county = data.get("trees", {}).get("county")
    if isinstance(county, bool) or not isinstance(county, int) or not 1 <= county <= 25:
        raise FetchError(f"{name}: [trees] county ska vara länskoden (1-25) för platsen", 2)
    return county


def out_dir(site: SiteDef, root: Path = ROOT) -> Path:
    return root / "build" / "trees" / site.name if site.public else site.out


# --- network ----------------------------------------------------------------------------


def _request(url: str, headers: Mapping[str, str] = {}, length: int | None = None) -> bytes:
    """The body of ``url``; a body that is not ``length`` bytes long is retried."""
    host = urllib.parse.urlparse(url).hostname
    ua = SKS_USER_AGENT if host == SKS_HOST else fetch_site.USER_AGENT
    req = urllib.request.Request(url, headers={"User-Agent": ua, **headers})  # noqa: S310
    last: Exception | str | None = None
    for attempt in range(TRIES):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # noqa: S310
                data = resp.read()
            if length is None or len(data) == length:
                return data
            last = f"fick {len(data)} av {length} byte"
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            ConnectionError,
        ) as exc:
            last = exc
        time.sleep(2**attempt)
    raise FetchError(f"kunde inte hämta {url}: {last}")


def file_size(url: str) -> int:
    req = urllib.request.Request(  # noqa: S310
        url, method="HEAD", headers={"User-Agent": SKS_USER_AGENT}
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # noqa: S310
            return int(resp.headers["Content-Length"])
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        KeyError,
        ValueError,
    ) as exc:
        raise FetchError(f"kunde inte läsa storleken på {url}: {exc}") from exc


def stream(url: str, size: int, start: int = 0) -> Generator[bytes]:
    """The file from ``start`` in order, fetched as ranges in parallel. Closing the
    iterator stops the download."""

    def get(lo: int) -> bytes:
        hi = min(lo + CHUNK, size) - 1
        return _request(url, {"Range": f"bytes={lo}-{hi}"}, hi - lo + 1)

    with cf.ThreadPoolExecutor(PARALLEL) as pool:
        pending: list[cf.Future[bytes]] = []
        nxt = start
        try:
            while pending or nxt < size:
                while len(pending) < 2 * PARALLEL and nxt < size:
                    pending.append(pool.submit(get, nxt))
                    nxt += CHUNK
                yield pending.pop(0).result()
        finally:
            for f in pending:
                f.cancel()


# --- the tree height raster ---------------------------------------------------------------


@dataclass
class TiffInfo:
    width: int
    height: int
    offsets: NDArray[np.uint64]
    counts: NDArray[np.uint64]
    origin: tuple[float, float]  # west, north of the top-left cell's corner
    cell: float
    tags: dict[int, tuple[int, int, bytes]]  # other tags: type, count, value bytes


_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 11: 4, 12: 8, 16: 8, 17: 8}


def parse_bigtiff(head: bytes) -> TiffInfo:
    """The first image of a little-endian BigTIFF whose tags lie within ``head``."""
    if head[:4] != b"II\x2b\x00":
        raise FetchError("trädhöjdsfilen är inte en BigTIFF i little-endian")
    ifd = struct.unpack_from("<Q", head, 8)[0]
    (n,) = struct.unpack_from("<Q", head, ifd)
    tags: dict[int, tuple[int, int, bytes]] = {}
    for i in range(n):
        tag, typ, count = struct.unpack_from("<HHQ", head, ifd + 8 + 20 * i)
        size = _TYPE_SIZE.get(typ, 1) * count
        raw = head[ifd + 20 + 20 * i : ifd + 28 + 20 * i]
        if size > 8:
            at = struct.unpack("<Q", raw)[0]
            if at + size > len(head):
                raise FetchError("trädhöjdsfilens taggar ligger inte i början av filen")
            raw = head[at : at + size]
        tags[tag] = (typ, count, raw[:size])

    def ints(tag: int) -> NDArray[np.uint64]:
        typ, _, raw = tags[tag]
        dtype = {3: "<u2", 4: "<u4", 16: "<u8"}[typ]
        return np.frombuffer(raw, dtype=dtype).astype(np.uint64)

    def floats(tag: int) -> NDArray[np.float64]:
        return np.frombuffer(tags[tag][2], dtype="<f8")

    width, height = int(ints(256)[0]), int(ints(257)[0])
    if int(ints(278)[0]) != 1 or int(ints(258)[0]) != 16 or int(ints(277)[0]) != 1:
        raise FetchError("trädhöjdsfilen har inte en rad per strip med 16 bitar")
    scale, tie = floats(33550), floats(33922)
    if abs(scale[0] - scale[1]) > 1e-9 or tie[0] != 0 or tie[1] != 0:
        raise FetchError("trädhöjdsfilen har ett oväntat rutnät")
    offsets, counts = ints(273), ints(279)
    if not (np.diff(offsets.astype(np.int64)) > 0).all():
        raise FetchError("trädhöjdsfilens rader ligger inte i ordning")
    keep = {t: v for t, v in tags.items() if t not in (256, 257, 273, 279, 33922)}
    return TiffInfo(
        width, height, offsets, counts, (float(tie[3]), float(tie[4])), float(scale[0]), keep
    )


def write_bigtiff(info: TiffInfo, row0: int, strips: list[bytes]) -> bytes:
    """A BigTIFF with rows ``row0 ..`` of the source (its other tags unchanged)."""
    rows = len(strips)
    west, north = info.origin
    tie = struct.pack("<6d", 0, 0, 0, west, north - row0 * info.cell, 0)
    entries: dict[int, tuple[int, int, bytes]] = dict(info.tags)
    entries[256] = (16, 1, struct.pack("<Q", info.width))
    entries[257] = (16, 1, struct.pack("<Q", rows))
    entries[33922] = (12, 6, tie)
    counts = np.array([len(s) for s in strips], dtype="<u8")
    data_at = 16
    blob = b"".join(strips)
    offsets = (data_at + np.concatenate([[0], np.cumsum(counts)[:-1]])).astype("<u8")
    entries[273] = (16, rows, offsets.tobytes())
    entries[279] = (16, rows, counts.tobytes())
    ifd_at = data_at + len(blob)
    ifd_at += ifd_at % 2
    extra_at = ifd_at + 8 + 20 * len(entries) + 8
    head = bytearray()
    extra = bytearray()
    for tag in sorted(entries):
        typ, count, raw = entries[tag]
        if len(raw) <= 8:
            value = raw.ljust(8, b"\0")
        else:
            value = struct.pack("<Q", extra_at + len(extra))
            extra += raw + (b"\0" if len(raw) % 2 else b"")
        head += struct.pack("<HHQ", tag, typ, count) + value
    out = io.BytesIO()
    out.write(b"II\x2b\x00\x08\x00\x00\x00" + struct.pack("<Q", ifd_at))
    out.write(blob)
    out.write(b"\0" * (ifd_at - data_at - len(blob)))
    out.write(struct.pack("<Q", len(entries)) + bytes(head) + struct.pack("<Q", 0))
    out.write(bytes(extra))
    return out.getvalue()


def site_bounds_sweref(site: SiteDef, extra_m: float) -> tuple[float, float, float, float]:
    """(west, south, east, north) in SWEREF 99 TM that cover the site plus ``extra_m``."""
    h = site.half_size_m + extra_m
    pts = [(x, y) for x in (-h, 0.0, h) for y in (-h, 0.0, h)]
    xy = transform(pts, site.proj, SWEREF)
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    return min(xs), min(ys), max(xs), max(ys)


def transform(
    points: Sequence[tuple[float, float]], src: str, dst: str
) -> list[tuple[float, float]]:
    res = subprocess.run(  # noqa: S603 - fixed program
        ["gdaltransform", "-s_srs", src, "-t_srs", dst, "-output_xy"],  # noqa: S607
        input="".join(f"{x} {y}\n" for x, y in points),
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise FetchError(f"gdaltransform misslyckades: {res.stderr.strip()[-300:]}")
    return [(float(a), float(b)) for a, b in (line.split()[:2] for line in res.stdout.splitlines())]


def zip_entries(head: bytes) -> list[tuple[str, int, int]]:
    """(name, data offset, compressed size) of the archive's leading local entries; a size
    of -1 when the entry is the last one readable from ``head`` (zip64, size unknown)."""
    out = []
    at = 0
    while at + 30 <= len(head):
        sig, _, flag, method, _, _, _, csize, _, nlen, elen = struct.unpack_from(
            "<IHHHHHIIIHH", head, at
        )
        if sig != 0x04034B50:
            break
        if method != 9:
            raise FetchError(f"oväntad komprimering {method} i trädhöjdsarkivet")
        name = head[at + 30 : at + 30 + nlen].decode()
        data = at + 30 + nlen + elen
        if csize == 0xFFFFFFFF or flag & 8:
            out.append((name, data, -1))
            break
        out.append((name, data, csize))
        at = data + csize
    return out


def fetch_chm(site: SiteDef, county: int, work: Path) -> tuple[Path, dict[str, Any]]:
    """The tree height raster's rows over the site, as a small GeoTIFF in SWEREF 99 TM."""
    import inflate64  # type: ignore[import-not-found]  # LGPL-2.1+, Deflate64

    url = SKS_COUNTY_ZIP.format(county=county)
    size = file_size(url)
    head = _request(url, {"Range": "bytes=0-65535"})
    entries = [e for e in zip_entries(head) if e[0].lower().endswith(".tif")]
    if not entries:
        raise FetchError(f"{url}: ingen TIFF först i arkivet")
    name, data_at, _ = entries[0]
    west, south, east, north = site_bounds_sweref(site, MARGIN_M)
    inflater = inflate64.Inflater()
    buf = bytearray()
    pos = 0  # uncompressed bytes before buf[0]
    info: TiffInfo | None = None
    need: tuple[int, int, int, int] | None = None  # row0, row1, first byte, end byte
    strips: list[bytes] = []
    fetched = 0
    t0 = time.monotonic()
    gen = stream(url, size, data_at)
    try:
        for chunk in gen:
            fetched += len(chunk)
            buf += inflater.inflate(chunk)
            if info is None:
                if len(buf) < 4 * 1024 * 1024:
                    continue
                info = parse_bigtiff(bytes(buf))
                c0 = math.floor((west - info.origin[0]) / info.cell)
                c1 = math.ceil((east - info.origin[0]) / info.cell)
                r0 = math.floor((info.origin[1] - north) / info.cell)
                r1 = math.ceil((info.origin[1] - south) / info.cell)
                if c0 < 0 or r0 < 0 or c1 > info.width or r1 > info.height:
                    raise FetchError(f"platsen ligger utanför länets trädhöjdsfil ({url})")
                first = int(info.offsets[r0])
                end = int(info.offsets[r1 - 1] + info.counts[r1 - 1])
                need = (r0, r1, first, end)
            if need is None or info is None:
                raise FetchError(f"trädhöjdsfilen saknar sitt TIFF-huvud ({url})")
            r0, r1, first, end = need
            if pos + len(buf) < end:
                # Keep only what may still be needed.
                drop = max(0, min(len(buf), first - pos))
                del buf[:drop]
                pos += drop
                if fetched % (64 * CHUNK) < CHUNK:
                    print(
                        f"    {fetched / 1e9:.1f} av högst {size / 1e9:.1f} GB "
                        f"({(time.monotonic() - t0):.0f} s)"
                    )
                continue
            for row in range(r0, r1):
                a = int(info.offsets[row]) - pos
                strips.append(bytes(buf[a : a + int(info.counts[row])]))
            break
    finally:
        gen.close()
    if info is None or need is None or not strips:
        raise FetchError(f"{url}: arkivet tog slut före platsens rader")
    out = work / "chm_rows.tif"
    out.write_bytes(write_bigtiff(info, need[0], strips))
    meta = {
        "chm_file": url.rsplit("/", 1)[-1],
        "chm_entry": name,
        "chm_rows": f"{need[0]}-{need[1] - 1}",
        "chm_downloaded_gb": round(fetched / 1e9, 2),
    }
    return out, meta


def warp_to_site(src: Path, site: SiteDef, work: Path, cell: float, stem: str) -> NDArray[Any]:
    """``src`` on the site's plane, cells of ``cell`` m centred like the world (row 0
    north), nearest neighbour, as float32."""
    edge = site.half_size_m
    n = round(2 * edge / cell)
    raw = work / f"{stem}.raw"
    fetch_site._gdal(
        [
            "gdalwarp",
            "-q",
            "-overwrite",
            "-t_srs",
            site.proj,
            "-te",
            str(-edge),
            str(-edge),
            str(edge),
            str(edge),
            "-ts",
            str(n),
            str(n),
            "-r",
            "near",
            "-ot",
            "Float32",
            "-dstnodata",
            "-9999",
            "-of",
            "ENVI",
            str(src),
            str(raw),
        ]
    )
    return np.fromfile(raw, dtype="<f4").reshape(n, n)


# --- trees ----------------------------------------------------------------------------------


def crown_radius(height: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """The search radius for a top of this height (m): about half the crown width of
    Swedish conifers (2-3 m wide at 10 m, 4-6 m at 25 m), never under 1.5 m."""
    return np.clip(0.8 + 0.09 * height, 1.5, 4.0)


def smooth3(z: NDArray[np.float64]) -> NDArray[np.float64]:
    p = np.pad(z, 1, mode="edge")
    k = ((1, 2, 1), (2, 4, 2), (1, 2, 1))
    out = np.zeros_like(z)
    for i in range(3):
        for j in range(3):
            out += k[i][j] * p[i : i + z.shape[0], j : j + z.shape[1]]
    return out / 16.0


def disc_max(z: NDArray[np.float64], radius: float) -> NDArray[np.float64]:
    r = math.floor(radius)
    p = np.pad(z, r, mode="constant", constant_values=-np.inf)
    out = np.full_like(z, -np.inf)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= radius * radius:
                out = np.maximum(out, p[r + dy : r + dy + z.shape[0], r + dx : r + dx + z.shape[1]])
    return out


def find_tops(
    chm: NDArray[np.float64], cell: float, half: float
) -> list[tuple[float, float, float]]:
    """(east, north, height) of every tree top in a height grid (m, row 0 north)."""
    s = smooth3(chm)
    radius = crown_radius(s) / cell
    is_top = np.zeros(s.shape, dtype=bool)
    for r in np.unique(np.round(radius * 2) / 2):
        sel = np.isclose(np.round(radius * 2) / 2, r)
        is_top |= sel & (s >= disc_max(s, float(r)) - 1e-9) & (s >= MIN_TREE_M)
    rows, cols = np.nonzero(is_top)
    peak = disc_max(chm, 1.5)  # the raw raster's highest value next to the top
    tops = sorted(
        ((float(peak[r, c]), int(r), int(c)) for r, c in zip(rows, cols, strict=True)),
        reverse=True,
    )
    kept: list[tuple[float, float, float]] = []
    taken = np.zeros(s.shape, dtype=bool)
    m = max(1, math.ceil(MERGE_M / cell))
    for h, r, c in tops:
        if taken[r, c]:
            continue
        taken[max(r - m, 0) : r + m + 1, max(c - m, 0) : c + m + 1] = True
        east = -half + (c + 0.5) * cell
        north = half - (r + 0.5) * cell
        kept.append((east, north, h))
    return kept


def _rand01(key: str) -> float:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64


def species_for(nmd_class: int, key: str) -> str:
    shares = _FOREST_SHARES.get(nmd_class % 10, _OTHER_SHARES)
    if not (111 <= nmd_class <= 128 and nmd_class % 10 != 9 and nmd_class != 120):
        shares = _OTHER_SHARES
    p = _rand01(key)
    acc = 0.0
    for name, share in zip(SPECIES, shares, strict=True):
        acc += share
        if p < acc:
            return name
    return SPECIES[-1]


# --- land cover --------------------------------------------------------------------------------


def nmd_legend() -> dict[tuple[int, int, int], int]:
    raw = _request(
        f"{NMD_WMS}?service=WMS&version=1.3.0&request=GetLegendGraphic&format=application/json"
        f"&layer={NMD_LAYER}"
    )
    out: dict[tuple[int, int, int], int] = {}
    for legend in json.loads(raw).get("Legend", []):
        for rule in legend.get("rules", []):
            for sym in rule.get("symbolizers", []):
                for e in sym.get("Raster", {}).get("colormap", {}).get("entries", []):
                    c = str(e["color"]).lstrip("#")
                    out.setdefault(
                        (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)),
                        int(float(e["quantity"])),
                    )
    if not any(111 <= v <= 128 for v in out.values()):
        raise FetchError("Naturvårdsverkets teckenförklaring saknar skogsklasserna")
    return out


def fetch_nmd(site: SiteDef, work: Path) -> Path:
    """NMD2023 classes over the site on its own 10 m grid, as an ASCII grid in SWEREF."""
    from PIL import Image  # MIT-CMU (dev dependency): the WMS answer is a PNG

    west, south, east, north = site_bounds_sweref(site, 2 * NMD_CELL_M)
    west = math.floor(west / NMD_CELL_M) * NMD_CELL_M
    south = math.floor(south / NMD_CELL_M) * NMD_CELL_M
    east = math.ceil(east / NMD_CELL_M) * NMD_CELL_M
    north = math.ceil(north / NMD_CELL_M) * NMD_CELL_M
    w, h = round((east - west) / NMD_CELL_M), round((north - south) / NMD_CELL_M)
    query = urllib.parse.urlencode(
        {
            "service": "WMS",
            "version": "1.3.0",
            "request": "GetMap",
            "layers": NMD_LAYER,
            "styles": "",
            "crs": SWEREF,
            # WMS 1.3.0 with EPSG:3006: northing first.
            "bbox": f"{south},{west},{north},{east}",
            "width": str(w),
            "height": str(h),
            "format": "image/png",
            "transparent": "false",
        }
    )
    img = np.asarray(Image.open(io.BytesIO(_request(f"{NMD_WMS}?{query}"))).convert("RGB"))
    if img.shape[:2] != (h, w):
        raise FetchError(f"marktäckekartan kom som {img.shape}, inte {h} x {w}")
    legend = nmd_legend()
    classes = np.zeros((h, w), dtype=np.int32)
    flat = img.reshape(-1, 3)
    keys = flat[:, 0].astype(np.int64) << 16 | flat[:, 1].astype(np.int64) << 8 | flat[:, 2]
    lut = {r << 16 | g << 8 | b: v for (r, g, b), v in legend.items()}
    unknown = 0
    out = classes.reshape(-1)
    for key in np.unique(keys):
        v = lut.get(int(key))
        if v is None:
            unknown += int((keys == key).sum())
            continue
        out[keys == key] = v
    if unknown > 0.02 * keys.size:
        raise FetchError(f"{unknown} rutor i marktäckekartan har en okänd färg")
    asc = work / "nmd.asc"
    asc.write_text(
        f"ncols {w}\nnrows {h}\nxllcorner {west}\nyllcorner {south}\ncellsize {NMD_CELL_M}\n"
        "NODATA_value 0\n" + "\n".join(" ".join(str(v) for v in row) for row in classes) + "\n",
        encoding="ascii",
    )
    tif = work / "nmd.tif"
    fetch_site._gdal(["gdal_translate", "-q", "-a_srs", SWEREF, str(asc), str(tif)])
    return tif


# --- main ------------------------------------------------------------------------------------


def write_trees(path: Path, trees: Sequence[Tree]) -> None:
    lines = ["east_m,north_m,height_m,species"]
    lines += [f"{t.east:.1f},{t.north:.1f},{t.height:.1f},{t.species}" for t in trees]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def run(argv: Sequence[str], root: Path = ROOT) -> int:
    args = parse_args(argv)
    site, _ = fetch_site.load_site(args["site"], root)
    county = county_of(site.name, root)
    dest = out_dir(site, root)
    dest.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.now(dt.UTC).date().isoformat()
    with tempfile.TemporaryDirectory(prefix="sd-trees-") as tmp:
        work = Path(tmp)
        print(f"==> trädhöjd (Skogsstyrelsen, län {county:02d}; strömmas, sparas inte)")
        chm_tif, meta = fetch_chm(site, county, work)
        chm = warp_to_site(chm_tif, site, work, 1.0, "chm").astype(np.float64)
        # Decimetres; 0 where the scan had no returns, negative or nodata outside.
        chm = np.where(chm > 0, chm / 10.0, 0.0)
        print("==> marktäcke (Naturvårdsverket, NMD2023)")
        nmd = warp_to_site(fetch_nmd(site, work), site, work, 1.0, "nmd").astype(np.int32)
    half = site.half_size_m
    tops = find_tops(chm, 1.0, half)
    trees = []
    n = chm.shape[0]
    for e, nn, h in tops:
        r = min(int((half - nn) // 1.0), n - 1)
        c = min(int((e + half) // 1.0), n - 1)
        key = f"{site.name}:{e:.1f},{nn:.1f}"
        trees.append(Tree(round(e, 1), round(nn, 1), round(h, 1), species_for(int(nmd[r, c]), key)))
    trees.sort(key=lambda t: (-t.north, t.east))
    write_trees(dest / "trees.csv", trees)
    # The raster itself on the site's 1 m grid (decimetres), for the world's ground and for
    # checking the built trees against it; it stays in build/ like the table.
    np.save(dest / "chm_dm.npy", np.round(chm * 10.0).astype(np.uint16))
    # How well the table's heights match the raster where the tree stands (the raster's
    # highest value within the tree's crown radius).
    errors = []
    for t in trees:
        r = int((half - t.north) // 1.0)
        c = int((t.east + half) // 1.0)
        rad = float(crown_radius(np.array([t.height]))[0])
        k = math.ceil(rad)
        win = chm[max(r - k, 0) : r + k + 1, max(c - k, 0) : c + k + 1]
        errors.append(abs(t.height - float(win.max())))
    counts = {s: sum(t.species == s for t in trees) for s in SPECIES}
    canopy = float((chm >= MIN_TREE_M).mean())
    lines = [
        "# Written by tools/terrain/fetch_trees.py (just trees-fetch). Not committed: see the",
        "# tool's docstring for why.",
        f'retrieved = "{today}"',
        'chm_source = "Skogsstyrelsen, Trädhöjd från Laserdata skog, baserad på Laserdata '
        'Nedladdning, skog (c) Lantmäteriet; bearbetad"',
        *(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}" for k, v in meta.items()),
        'species_source = "NMD2023 basskikt v2.1, Naturvårdsverket (CC0)"',
        f"trees = {len(trees)}",
        *(f"trees_{s} = {c}" for s, c in counts.items()),
        f"canopy_share = {canopy:.3f}",
        f"height_vs_raster_mean_abs_m = {float(np.mean(errors)) if errors else 0.0:.3f}",
        f"height_max_m = {max((t.height for t in trees), default=0.0):.1f}",
    ]
    (dest / "trees.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Klart: {dest / 'trees.csv'}: {len(trees)} träd "
        + ", ".join(f"{s} {c}" for s, c in counts.items())
        + f"; medelfel mot rastret {float(np.mean(errors)) if errors else 0.0:.2f} m"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(list(sys.argv[1:] if argv is None else argv))
    except FetchError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
