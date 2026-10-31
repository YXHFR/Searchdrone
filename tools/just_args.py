"""Parse ``key=value`` arguments for just recipes that are not simulation recipes.

just treats ``NAME=VALUE`` after the recipe name as a positional argument, so recipes
with settings are variadic and pass everything here unchanged:

    python tools/just_args.py <recipe> [key=value ...]

Unknown recipes or keys exit with code 2 and a Swedish explanation. ``dry=1`` prints
the parsed arguments as JSON (used by the tests). A recipe whose ticket is not built
yet prints ``inte byggt än (<ticket>)`` and exits with code 2; a built one exits 0 and
the justfile goes on with the real work.

The simulation entry points have the same rules in ``sd_sim._args``; keep the
messages of the two in step.
"""

from __future__ import annotations

import json
import sys

NOT_BUILT_EXIT = 2

# recipe -> (ticket that builds it, or None when it is built, allowed keys). A built
# recipe only has its keys checked here (exit 0) and does its work in the justfile.
RECIPES: dict[str, tuple[str | None, frozenset[str]]] = {
    "dev": (None, frozenset({"physics", "gui", "brain", "instance"})),
    "e2e": (None, frozenset({"runs", "offline"})),
    "e2e-search": (None, frozenset({"runs", "detector", "model"})),
    "demo-search": (None, frozenset({"detector", "model"})),
    "accept": (None, frozenset({"phase", "sha"})),
    "perf": (None, frozenset({"runs", "gui", "eval"})),
    "showcase": ("B1-18", frozenset({"stage", "brain"})),
    "brain-eval-sim": (None, frozenset({"flights", "seed", "model"})),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in RECIPES:
        known = ", ".join(sorted(RECIPES))
        print(f"just_args: okänt recept. Kända recept: {known}.", file=sys.stderr)
        return 2
    recipe, rest = argv[0], argv[1:]
    ticket, allowed = RECIPES[recipe]
    values: dict[str, str] = {}
    for raw in rest:
        key, sep, value = raw.partition("=")
        if not sep or not key:
            print(f"just {recipe}: '{raw}' är inte på formen nyckel=värde.", file=sys.stderr)
            return 2
        if key != "dry" and key not in allowed:
            known = ", ".join(sorted(allowed)) or "(inga)"
            msg = f"just {recipe}: okänd nyckel '{key}'. Tillåtna nycklar: {known}."
            print(msg, file=sys.stderr)
            return 2
        values[key] = value
    if values.get("dry") == "1":
        print(json.dumps({"recipe": recipe, "args": rest}))
        return 0
    if ticket is None:
        return 0
    print(f"inte byggt än ({ticket})")
    return NOT_BUILT_EXIT


if __name__ == "__main__":
    sys.exit(main())
