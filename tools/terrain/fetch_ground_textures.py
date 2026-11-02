"""Fetch the photo farm's ground pictures: ``just ground-pictures-fetch``.

The site's terrain texture (``sd_sim.terrain``) is painted at about 15 cm per pixel: fine
for a search flight's video, but a drone photo of 1-2 cm per pixel shows grass blades,
needles and gravel. The photo farm (``sd_sim.farm.stills``) lays a detailed ground
under its pictures (``sd_sim.farm.ground``), made from the painted colours and these
photographed ground textures from Poly Haven (CC0, https://polyhaven.com/license), one
per kind of ground (``TEXTURES``).

Everything lands in ``build/ground_textures/``, never in the repository: each texture's
diffuse map at 2k (``<asset>.jpg``) and ``ground.json`` with each one's kind, its size on
the ground in metres (Poly Haven's own ``dimensions``), the download's address and MD5
sum (checked against the API's).

Needs internet only while it runs. Exit codes: 0 done, 1 a download or a check failed,
2 a bad argument.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_site import ROOT, USER_AGENT, FetchError

INFO = "https://api.polyhaven.com/info/{asset}"
FILES = "https://api.polyhaven.com/files/{asset}"
DOWNLOAD_PREFIX = "https://dl.polyhaven.org/"
RESOLUTION = "2k"
TRIES = 4
# kind of ground -> Poly Haven texture assets (CC0)
TEXTURES: dict[str, tuple[str, ...]] = {
    "grass": ("aerial_grass_rock", "sparse_grass"),
    "forest": ("forest_floor", "forest_leaves_02", "forest_ground_04"),
    "path": ("gravel_road", "dirt_aerial_02"),
    "rock": ("aerial_ground_rock",),
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
    last: Exception | None = None
    for attempt in range(TRIES):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
                return resp.read()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise FetchError(f"kunde inte hämta {url}: {last}")


def fetch(asset: str, dest: Path) -> dict[str, Any]:
    info = json.loads(_get(INFO.format(asset=asset)))
    files = json.loads(_get(FILES.format(asset=asset)))
    try:
        entry = files["Diffuse"][RESOLUTION]["jpg"]
        width_mm, height_mm = (float(v) for v in info["dimensions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FetchError(f"Poly Haven: {asset} saknar diffuse {RESOLUTION} eller mått") from exc
    url = str(entry["url"])
    if not url.startswith(DOWNLOAD_PREFIX):
        raise FetchError(f"Poly Haven: oväntad adress {url}")
    target = dest / f"{asset}.jpg"
    if not (target.is_file() and _md5(target.read_bytes()) == entry["md5"]):
        data = _get(url)
        if _md5(data) != entry["md5"]:
            raise FetchError(f"Poly Haven: {url} har fel MD5")
        tmp = target.with_name(f".{target.name}.part")
        tmp.write_bytes(data)
        tmp.replace(target)
    return {
        "file": target.name,
        "size_m": [round(width_mm / 1000, 3), round(height_mm / 1000, 3)],
        "url": url,
        "md5": entry["md5"],
        "licence": "CC0-1.0",
        "source": f"https://polyhaven.com/a/{asset}",
    }


def _md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    if list(sys.argv[1:] if argv is None else argv):
        print("ground-pictures-fetch tar inga inställningar", file=sys.stderr)
        return 2
    dest = ROOT / "build" / "ground_textures"
    dest.mkdir(parents=True, exist_ok=True)
    print("==> markbilder (Poly Haven, CC0)")
    out: dict[str, Any] = {"textures": {}}
    try:
        for kind, assets in TEXTURES.items():
            for asset in assets:
                out["textures"][asset] = {"kind": kind, **fetch(asset, dest)}
                print(f"    {asset} ({kind})")
    except FetchError as exc:
        print(str(exc), file=sys.stderr)
        return exc.code
    (dest / "ground.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"Klart: {dest} ({len(out['textures'])} bilder)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
