"""Download the local AI model to ``models/`` and check its sha256 (``just models-fetch``).

Reads ``software/brain/runtime.lock.toml``. The file comes only from the locked Hugging
Face repository; a size or checksum that does not match is an error and the partial
file is removed, so llama-server never starts with an unchecked model. A finished,
correct file is not downloaded again. ``fallback=true`` fetches the reserve model.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tomllib
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "software/brain/runtime.lock.toml"
MODELS = ROOT / "models"
HF = "https://huggingface.co"
CHUNK = 1 << 20


class FetchError(RuntimeError):
    pass


def load_lock(path: Path = LOCK) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def url_for(entry: dict[str, Any]) -> str:
    repo, name = str(entry["hf_repo"]), str(entry["file"])
    if "/" not in repo or ".." in repo or "/" in name or ".." in name:
        raise FetchError("ogiltigt repo eller filnamn i runtime.lock.toml")
    return f"{HF}/{repo}/resolve/main/{name}"


def fetch(entry: dict[str, Any], dest_dir: Path, client: httpx.Client) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / str(entry["file"])
    want = str(entry["sha256"]).lower()
    if len(want) != 64:
        raise FetchError("sha256 saknas i runtime.lock.toml")
    if target.is_file() and sha256_of(target) == want:
        print(f"{target.name} finns redan och stämmer")
        return target
    part = target.with_suffix(target.suffix + ".part")
    part.unlink(missing_ok=True)
    print(f"laddar ner {target.name} (ca {int(entry.get('size', 0)) / 1e9:.1f} GB) ...")
    h = hashlib.sha256()
    size = 0
    with client.stream("GET", url_for(entry), follow_redirects=True) as resp:
        if resp.status_code != 200:
            raise FetchError(f"Hugging Face svarade HTTP {resp.status_code}")
        with part.open("wb") as fh:
            for block in resp.iter_bytes(CHUNK):
                fh.write(block)
                h.update(block)
                size += len(block)
    expected_size = entry.get("size")
    if (expected_size is not None and size != int(expected_size)) or h.hexdigest() != want:
        part.unlink(missing_ok=True)
        raise FetchError(f"{target.name}: kontrollsumman stämmer inte, filen är borttagen")
    part.replace(target)
    print(f"{target.name}: kontrollsumman stämmer")
    return target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("args", nargs="*", help="fallback=true hämtar reservmodellen")
    ns = ap.parse_args(argv)
    opts = dict(a.split("=", 1) for a in ns.args if "=" in a)
    section = "model_fallback" if opts.get("fallback") in ("1", "true", "yes") else "model"
    entry = load_lock()[section]
    try:
        with httpx.Client(timeout=httpx.Timeout(30.0, read=120.0)) as client:
            fetch(entry, MODELS, client)
    except (FetchError, httpx.HTTPError) as exc:
        print(f"fel: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
