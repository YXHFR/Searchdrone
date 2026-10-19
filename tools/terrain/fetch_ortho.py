"""Fetch a site's orthophoto for the simulator's photo mode: ``just ortho-fetch site=<name>``.

Lantmäteriet's newest colour orthophoto (0.16 m where it has one) over the site, with the
owner's own Geotorget login (``fetch_site.lm_login``: ``config/geotorget.local.toml`` or
``SD_LM_USER``/``SD_LM_PASSWORD``), warped onto the site's plane at
``fetch_site.ORTHO_CELL_M`` per pixel. It is written to ``build/ortho/<site>/`` for a
public site (``ortho.npy``, row 0 north, and ``ortho.toml``, where and when it came
from) and to ``build/sites/<site>/`` for a local one. Neither is ever committed, cached
or uploaded: the photo stays on the owner's computer. The world generator
(``sd_sim.terrain``) drapes it over the open ground when the file is there and builds
the world without it otherwise.

Refuses to run in CI (``CI`` or ``GITHUB_ACTIONS`` set): no runner may hold the photo.
Needs internet and GDAL's command-line tools. Exit codes: 0 done, 1 a download or a
check failed, 2 bad arguments, no login or in CI.
"""

from __future__ import annotations

import datetime as dt
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_site
from fetch_site import ROOT, FetchError, SiteDef

IN_CI = fetch_site.ORTHO_IN_CI


def out_dir(site: SiteDef, root: Path = ROOT) -> Path:
    return root / "build" / "ortho" / site.name if site.public else site.out


def run(argv: Sequence[str], root: Path = ROOT) -> int:
    if fetch_site.in_ci():  # fetch_site.fetch_ortho refuses too, for every other caller
        raise FetchError(IN_CI, 2)
    args = fetch_site.parse_args(argv)
    site, _ = fetch_site.load_site(args["site"], root)
    login = fetch_site.lm_login(root)
    if not login:
        raise FetchError(fetch_site.LOGIN_MISSING, 2)
    dest = out_dir(site, root)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"==> ortofoto (Lantmäteriet) för {site.name}, {fetch_site.ORTHO_CELL_M} m per pixel")
    with tempfile.TemporaryDirectory(prefix="sd-ortho-") as tmp:
        partial = dest / "ortho.part.npy"
        meta = fetch_site.fetch_ortho(site, login, Path(tmp), partial)
        partial.replace(dest / "ortho.npy")
    lines = [
        "# Written by tools/terrain/fetch_ortho.py (just ortho-fetch). Local only: never commit.",
        f'retrieved = "{dt.datetime.now(dt.UTC).date().isoformat()}"',
        f'collection = "{meta["collection"]}"',
        f"year = {meta['year']}",
        f"source_cell_m = {meta['source_cell_m']}",
        f"cell_m = {meta['cell_m']}",
        f"tiles = {meta['tiles']}",
        'licence = "CC BY 4.0, (c) Lantmäteriet"',
    ]
    (dest / "ortho.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Klart: {dest / 'ortho.npy'} ({meta['collection']}, {meta['year']}). "
        f"Starta fotoläget med: just sim-photo"
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
