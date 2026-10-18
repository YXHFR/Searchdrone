"""vendor.py downloads offline: a small zip and a fake opener stand in for the server."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Any

import pytest
import vendor


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, url: str) -> None:
        super().__init__(data)
        self.url = url

    def geturl(self) -> str:
        return self.url


def make_zip(member: str, payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member, payload)
        zf.writestr("../escape.step", b"never written")
    return buf.getvalue()


@pytest.fixture
def archive() -> bytes:
    return make_zip("part.step", b"ISO-10303-21;\n" + b"x" * 100)


@pytest.fixture
def model(archive: bytes) -> dict[str, Any]:
    return {
        "name": "part",
        "url": "https://example.invalid/part.zip",
        "sha256": hashlib.sha256(archive).hexdigest(),
        "member": "part.step",
        "max_download_bytes": 10_000,
        "max_bytes": 1_000,
    }


def opener_for(data: bytes, url: str = "https://example.invalid/part.zip") -> Any:
    calls: list[str] = []

    def opener(req: Any) -> FakeResponse:
        calls.append(req.full_url)
        return FakeResponse(data, url)

    opener.calls = calls  # type: ignore[attr-defined]
    return opener


def test_fetch_extracts_only_the_named_member(
    model: dict[str, Any], archive: bytes, tmp_path: Path
) -> None:
    path = vendor.fetch(model, tmp_path, opener_for(archive))
    assert path == tmp_path / "part.step"
    assert path.read_bytes().startswith(b"ISO-10303-21;")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["part.step", "part.zip"]
    assert not (tmp_path.parent / "escape.step").exists()


def test_a_checked_archive_on_disk_is_not_downloaded_again(
    model: dict[str, Any], archive: bytes, tmp_path: Path
) -> None:
    (tmp_path / "part.zip").write_bytes(archive)
    opener = opener_for(b"")
    vendor.fetch(model, tmp_path, opener)
    assert opener.calls == []


def test_checksum_mismatch_is_refused_and_removed(model: dict[str, Any], tmp_path: Path) -> None:
    other = make_zip("part.step", b"something else")
    with pytest.raises(ValueError, match="SHA-256"):
        vendor.fetch(model, tmp_path, opener_for(other))
    assert not (tmp_path / "part.zip").exists()
    assert not (tmp_path / "part.step").exists()


def test_plain_http_is_refused(model: dict[str, Any], archive: bytes, tmp_path: Path) -> None:
    opener = opener_for(archive)
    with pytest.raises(ValueError, match="https"):
        vendor.fetch({**model, "url": "http://example.invalid/part.zip"}, tmp_path, opener)
    assert opener.calls == []


def test_redirect_away_from_https_is_refused(
    model: dict[str, Any], archive: bytes, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="redirected"):
        vendor.fetch(model, tmp_path, opener_for(archive, url="http://example.invalid/x.zip"))
    assert list(tmp_path.iterdir()) == []


def test_oversize_download_is_refused_and_cleaned_up(
    model: dict[str, Any], archive: bytes, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="more than"):
        vendor.fetch({**model, "max_download_bytes": 50}, tmp_path, opener_for(archive))
    assert list(tmp_path.iterdir()) == []


def test_oversize_member_is_refused(model: dict[str, Any], archive: bytes, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_bytes"):
        vendor.fetch({**model, "max_bytes": 10}, tmp_path, opener_for(archive))
    assert not (tmp_path / "part.step").exists()


def test_vendor_yaml_is_pinned_and_https(tmp_path: Path) -> None:
    for m in vendor.models():
        assert m["url"].startswith("https://")
        assert len(m["sha256"]) == 64
        assert int(m["sha256"], 16) >= 0
        assert m["max_download_bytes"] > 0
        assert m["max_bytes"] > 0
    bad = tmp_path / "vendor.yaml"
    bad.write_text("schema_version: 2\nmodels: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        vendor.models(bad)
