"""Photo mode: ``just sim-photo [site=<name>] [gi=vct|off] [other sim keys]``.

The simulator in a site world whose open ground is the site's orthophoto
(``just ortho-fetch``, with the owner's own Geotorget login; it stays in ``build/``),
with the real trees standing on it (``just trees-fetch``) and global illumination for
the camera (``gi=vct``, the default here). Checks that the photo is there and says what
to run when it is not, then starts ``just sim`` with the site's world. Everything else
is ``just sim``'s: the same keys, the same stop (``just stop``).

Exit codes: those of ``just sim``; 2 when the photo is missing or an argument is bad.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sd_sim import _args, launch, paths, site
from sd_sim.util import say

DEFAULT_SITE = "hellasgarden"
OWN_KEYS = ("site",)
LOGIN_EXAMPLE = Path("config") / "geotorget.example.toml"
LOGIN_FILE = Path("config") / "geotorget.local.toml"


def missing_photo(name: str, repo: Path) -> str:
    """What to do when the site has no orthophoto yet, in the owner's words."""
    steps = []
    if not (repo / LOGIN_FILE).is_file():
        steps.append(
            f"Kopiera {LOGIN_EXAMPLE} till {LOGIN_FILE} och skriv in ditt Geotorget-konto "
            f"(Lantmäteriet) där; kör sedan: chmod 600 {LOGIN_FILE}"
        )
    steps.append(f"Hämta fotot: just ortho-fetch site={name}")
    steps.append("Starta igen: just sim-photo")
    lines = [f"Ortofoto saknas för {name}. Fotoläget behöver det:"]
    lines += [f"  {i}. {s}" for i, s in enumerate(steps, 1)]
    lines.append("Fotot hämtas bara till build/ på din dator och laddas aldrig upp.")
    return "\n".join(lines)


def launch_argv(argv: list[str], repo: Path = paths.REPO) -> list[str]:
    """The ``sd_sim.launch`` arguments for photo mode, after the checks."""
    own = [a for a in argv if a.partition("=")[0] in OWN_KEYS]
    rest = [a for a in argv if a not in own]
    if any(a.partition("=")[0] == "world" for a in rest):
        _args.fail("sd_sim.photo: välj plats med site=<namn>, inte world=")
    name = _args.parse(own, OWN_KEYS, "sd_sim.photo").get("site", DEFAULT_SITE)
    try:
        s = site.load(name, repo)
    except site.SiteError as exc:
        _args.fail(f"sd_sim.photo: {exc}")
    if s.ortho is None:
        _args.fail(missing_photo(name, repo))
    if s.trees_folder is None or not (s.trees_folder / "trees.csv").is_file():
        say(
            f"Obs: {name} har inga riktiga träd än, skogen blir genererad. "
            f"Hämta dem med: just trees-fetch site={name}"
        )
    gi = [] if any(a.startswith("gi=") for a in rest) else ["gi=vct"]
    return ["up", f"world={s.world}", *gi, *rest]


def main(argv: list[str] | None = None) -> int:
    return launch.main(launch_argv(list(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    sys.exit(main())
