"""Fetch the published detector weights: ``just detector-fetch yes=1``.

Keys: ``yes`` (1: fetch; without it only the plan is shown), ``only`` (comma list of
names in ``models.toml``), ``data``.

Each file in ``tools/train/models.toml`` goes to ``<data>/models/<file>`` and is checked
against its size and SHA-256; a stopped download carries on. A ready-to-use detector
(``kind = "detector"``) also gets its manifest (``sd.detector-model/v1``) next to it, so
``just bench-detector model=<name>`` and ``SD_DETECTOR_MODEL`` can use it. Runs with the
workspace's Python (no PyTorch needed).
"""

from __future__ import annotations

import json
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bench.common import BenchError, data_dir, download, flag, parse_args, say, warn
from check_licenses import license_allowed

from sd_detect.onnx import MANIFEST_SCHEMA, MODEL_LICENCE

REGISTER = Path(__file__).with_name("models.toml")
KEYS = ("yes", "only", "data")


def load_models(path: Path = REGISTER) -> list[dict[str, Any]]:
    rows = tomllib.loads(path.read_text(encoding="utf-8")).get("model", [])
    for row in rows:
        if not license_allowed(row["licence"]):
            raise BenchError(f"models.toml: {row['name']} har licensen {row['licence']}", 2)
        if row["kind"] == "detector" and row["licence"] != MODEL_LICENCE:
            raise BenchError(f"models.toml: {row['name']} ska ha licensen {MODEL_LICENCE}", 2)
        if not row["url"].startswith("https://") or len(row["sha256"]) != 64:
            raise BenchError(f"models.toml: {row['name']} saknar https eller SHA-256", 2)
        if row["kind"] not in ("pretrained", "detector"):
            raise BenchError(f"models.toml: {row['name']} har okänd kind", 2)
    return rows


def manifest_for(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "name": row["name"],
        "onnx": Path(row["file"]).name,
        "sha256": row["sha256"],
        "family": row["family"],
        "model": row["name"],
        "input_px": row["input_px"],
        "tile_px": row["input_px"],
        "classes": ["person"],  # COCO: person is class 0
        "licence": row["licence"],
        "trained_on": [
            {"dataset": "coco", "licence": "CC-BY-4.0", "attribution": row["attribution"]}
        ],
        "publish": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv, KEYS, "detector-fetch")
        rows = load_models()
        only = [s for s in args.get("only", "").split(",") if s]
        unknown = set(only) - {r["name"] for r in rows}
        if unknown:
            raise BenchError(f"okända namn: {', '.join(sorted(unknown))}", 2)
        rows = [r for r in rows if not only or r["name"] in only]
        data = data_dir(args)
        total = sum(r["size_bytes"] for r in rows) / 1e6
        if not flag(args, "yes"):
            say(f"Hämtar {len(rows)} filer ({total:.0f} MB) till {data / 'models'}:")
            for r in rows:
                say(f"  {r['name']}: {r['size_bytes'] / 1e6:.0f} MB, {r['licence']}")
            say("Kör med yes=1 för att hämta.")
            return 2
        for r in rows:
            dest = data / "models" / r["file"]
            say(f"{r['name']}: {dest}")
            download(r["url"], dest, r["size_bytes"], r["sha256"])
            if r["kind"] == "detector":
                manifest = dest.with_suffix(".json")
                manifest.write_text(json.dumps(manifest_for(r), indent=1) + "\n", encoding="utf-8")
        say("Klart.")
        return 0
    except BenchError as exc:
        warn(str(exc))
        return exc.code


if __name__ == "__main__":
    raise SystemExit(main())
