"""tools/models_fetch.py and tools/brain_token.py (B1-10)."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import brain_token
import httpx
import models_fetch
import pytest

DATA = b"gguf-test-bytes" * 100


def _entry(data: bytes = DATA, **kw: object) -> dict[str, object]:
    return {
        "hf_repo": "Org/Repo-GGUF",
        "file": "model.gguf",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        **kw,
    }


def _client(body: bytes, status: int = 200) -> httpx.Client:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.host == "huggingface.co"
        assert req.url.path == "/Org/Repo-GGUF/resolve/main/model.gguf"
        return httpx.Response(status, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_lock_file_is_complete() -> None:
    lock = models_fetch.load_lock()
    assert lock["llama_cpp"]["image"].startswith("ghcr.io/ggml-org/llama.cpp:server-cuda12-b")
    assert "@sha256:" in lock["llama_cpp"]["image"]
    for section in ("model", "model_fallback"):
        assert len(lock[section]["sha256"]) == 64
        assert lock[section]["file"].endswith(".gguf")
    compose = (models_fetch.ROOT / "compose.yaml").read_text()
    assert lock["llama_cpp"]["image"] in compose
    assert lock["model"]["file"] in compose


def test_fetch_checks_the_sum(tmp_path: Path) -> None:
    target = models_fetch.fetch(_entry(), tmp_path, _client(DATA))
    assert target.read_bytes() == DATA
    # A correct file is not fetched again.
    models_fetch.fetch(_entry(), tmp_path, _client(b"", status=500))
    with pytest.raises(models_fetch.FetchError, match="kontrollsumman"):
        models_fetch.fetch(_entry(), tmp_path / "b", _client(DATA[:-1] + b"x"))
    assert not any((tmp_path / "b").iterdir())
    with pytest.raises(models_fetch.FetchError, match="HTTP 404"):
        models_fetch.fetch(_entry(), tmp_path / "c", _client(b"", status=404))
    with pytest.raises(models_fetch.FetchError):
        models_fetch.fetch(_entry(sha256="abc"), tmp_path / "d", _client(DATA))


@pytest.mark.parametrize(
    "entry",
    [
        {"hf_repo": "../x", "file": "a.gguf"},
        {"hf_repo": "Org/Repo", "file": "../a.gguf"},
        {"hf_repo": "noslash", "file": "a.gguf"},
    ],
)
def test_url_refuses_odd_names(entry: dict[str, str]) -> None:
    with pytest.raises(models_fetch.FetchError):
        models_fetch.url_for(entry)


def test_brain_token(tmp_path: Path) -> None:
    path = tmp_path / "secrets" / "brain_token"
    brain_token.write_token(path)
    first = path.read_text().strip()
    assert len(first) >= 43
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    brain_token.main([str(path)])
    assert path.read_text().strip() != first
