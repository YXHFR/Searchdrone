"""``just maps-fetch`` (tools/maps/fetch.py), without network: downloads, ``pmtiles`` and
``git`` are replaced by fakes. The "extract" copies the small committed test pack, so the
whole flow runs on a real PMTiles file."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import math
import shutil
import tarfile
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from maps import fetch

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "software" / "panel" / "e2e" / "fixtures" / "fixture.pmtiles"
CMAC_LAT, CMAC_LON = -35.363261, 149.165230
PROGRAM = b"#!/bin/sh\necho fake pmtiles\n"


def _tarball(members: dict[str, bytes], symlink: str | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if symlink:
            info = tarfile.TarInfo(symlink)
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
    return buf.getvalue()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


TARBALL = _tarball({"LICENSE": b"BSD", "pmtiles": PROGRAM})
ASSET_FILES = {
    "fonts/OFL.txt": b"OFL",
    "fonts/Noto Sans Regular/0-255.pbf": b"glyph",
    "sprites/v4/dark.json": b"{}",
    "sprites/v4/dark.png": b"png",
}


def _assets_digest() -> str:
    lines = "".join(f"{_sha(ASSET_FILES[p])}  {p}\n" for p in sorted(ASSET_FILES))
    return _sha(lines.encode())


def _lock() -> dict[str, Any]:
    return {
        "go_pmtiles": {
            "version": "9.9.9",
            "url": "https://example.invalid/pmtiles.tar.gz",
            "sha256": _sha(TARBALL),
            "binary_sha256": _sha(PROGRAM),
        },
        "build": {"url_template": "https://example.invalid/{build}.pmtiles"},
        "assets": {
            "repo": "https://example.invalid/assets",
            "commit": "0" * 40,
            "paths": ["fonts/OFL.txt", "fonts/Noto Sans Regular", "sprites/v4"],
            "files": len(ASSET_FILES),
            "sha256": _assets_digest(),
        },
    }


class FakeTools:
    """Stands in for ``pmtiles extract`` and ``git``; records every command."""

    def __init__(self, assets: dict[str, bytes] | None = None) -> None:
        self.commands: list[list[str]] = []
        self.assets = ASSET_FILES if assets is None else assets

    def __call__(self, cmd: Sequence[str]) -> None:
        cmd = list(cmd)
        self.commands.append(cmd)
        if cmd[1] == "extract":
            if "--dry-run" not in cmd:
                shutil.copyfile(FIXTURE, cmd[3])
        elif cmd[1] == "init":
            Path(cmd[3]).mkdir(parents=True)
            (Path(cmd[3]) / ".git").mkdir()
        elif "checkout" in cmd:
            work = Path(cmd[2])
            for rel, data in self.assets.items():
                (work / rel).parent.mkdir(parents=True, exist_ok=True)
                (work / rel).write_bytes(data)


def _settings(**kw: Any) -> fetch.Settings:
    base: dict[str, Any] = {
        "build": "20260811",
        "etag": '"abc-1"',
        "name": "demo",
        "maxzoom": 15,
        "bbox": (149.13, -35.39, 149.20, -35.33),
        "dry": False,
    }
    base.update(kw)
    return fetch.Settings(**base)


def _run(
    tmp_path: Path, s: fetch.Settings, tools: FakeTools, status: int = 206, etag: str = '"abc-1"'
) -> Path | None:
    return fetch.fetch(
        s,
        tmp_path / "data" / "tiles",
        _lock(),
        bin_dir=tmp_path / "bin",
        runner=tools,
        fetch_url=lambda _: TARBALL,
        status=lambda _: (status, etag),
        now=lambda: dt.datetime(2026, 10, 16, tzinfo=dt.UTC),
    )


# --- settings -----------------------------------------------------------------------


def test_defaults_are_cmac_plus_minus_3_km() -> None:
    s = fetch.load_settings(ROOT, {})
    assert s.etag
    assert (s.build, s.name, s.maxzoom, s.dry) == ("20260811", "demo", 15, False)
    min_lon, min_lat, max_lon, max_lat = s.bbox
    assert min_lat < CMAC_LAT < max_lat and min_lon < CMAC_LON < max_lon
    height_m = (max_lat - min_lat) * 110_540
    width_m = (max_lon - min_lon) * 111_320 * math.cos(math.radians(CMAC_LAT))
    assert abs(height_m - 6000) < 5 and abs(width_m - 6000) < 5


def test_keys_override_config() -> None:
    s = fetch.load_settings(
        ROOT,
        {"bbox": "149.14,-35.38,149.19,-35.34", "maxzoom": "13", "name": "sim-area", "dry": "1"},
    )
    assert s.bbox == (149.14, -35.38, 149.19, -35.34)
    assert (s.maxzoom, s.name, s.dry) == (13, "sim-area", True)


def test_site_centres_the_pack_on_its_home() -> None:
    """The pack for a simulator site is centred on the same home as its world and SITL."""
    s = fetch.load_settings(ROOT, {"site": "hellasgarden"})
    home = tomllib.loads(
        (ROOT / "simulation" / "gazebo" / "sites" / "hellasgarden" / "site.toml").read_text()
    )["home"]
    min_lon, min_lat, max_lon, max_lat = s.bbox
    assert abs((min_lat + max_lat) / 2 - home["lat"]) < 1e-9
    assert abs((min_lon + max_lon) / 2 - home["lon"]) < 1e-9
    assert abs((max_lat - min_lat) * 110_540 - 6000) < 40


def test_local_config_wins_over_example(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    shutil.copyfile(
        ROOT / "config" / "maps.example.toml", tmp_path / "config" / "maps.example.toml"
    )
    (tmp_path / "config" / "maps.toml").write_text('[maps]\nname = "other"\nhalf_size_m = 500\n')
    s = fetch.load_settings(tmp_path, {})
    assert s.name == "other"
    assert abs((s.bbox[3] - s.bbox[1]) * 110_540 - 1000) < 5


@pytest.mark.parametrize(
    ("args", "text"),
    [
        (["foo=1"], "okänd nyckel 'foo'"),
        (["--bbox=1,2,3,4"], "nyckel=värde"),
        (["bbox"], "nyckel=värde"),
    ],
)
def test_bad_arguments_exit_2(
    args: list[str], text: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert fetch.main(args) == 2
    assert text in capsys.readouterr().err


@pytest.mark.parametrize(
    "args",
    [
        {"bbox": "1,2,3"},
        {"bbox": "150,-35,149,-34"},
        {"bbox": "a,b,c,d"},
        {"maxzoom": "16"},
        {"maxzoom": "x"},
        {"name": "../evil"},
        {"name": ""},
        {"name": "-demo"},
        {"name": "Demo"},
        {"name": "a" * 65},
        {"site": "nowhere"},
        {"site": "../evil"},
        {"site": "hellasgarden", "bbox": "149.14,-35.38,149.19,-35.34"},
    ],
)
def test_bad_values_exit_2(args: dict[str, str]) -> None:
    with pytest.raises(fetch.FetchError) as err:
        fetch.load_settings(ROOT, args)
    assert err.value.code == 2


def test_the_locked_build_is_a_date() -> None:
    example = tomllib.loads((ROOT / "config" / "maps.example.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads(fetch.LOCK.read_text(encoding="utf-8"))
    build = example["maps"]["protomaps_build"]
    assert dt.date(int(build[:4]), int(build[4:6]), int(build[6:])) < dt.date(2026, 10, 1)
    assert lock["go_pmtiles"]["url"].startswith("https://github.com/protomaps/go-pmtiles/")
    assert lock["assets"]["repo"] == "https://github.com/protomaps/basemaps-assets"
    for key in ("sha256", "binary_sha256"):
        assert len(lock["go_pmtiles"][key]) == 64
    assert len(lock["assets"]["commit"]) == 40


# --- the whole flow -----------------------------------------------------------------


def test_fetch_writes_pack_assets_and_manifest(tmp_path: Path) -> None:
    tools = FakeTools()
    out = _run(tmp_path, _settings(), tools)
    tiles = tmp_path / "data" / "tiles"
    assert out is not None
    assert out == tiles / "demo.pmtiles"
    assert out.read_bytes() == FIXTURE.read_bytes()
    extract = next(c for c in tools.commands if c[1] == "extract")
    assert extract[0] == str(tmp_path / "bin" / "sd-pmtiles-9.9.9")
    assert extract[2] == "https://example.invalid/20260811.pmtiles"
    assert extract[4:] == ["--bbox=149.130000,-35.390000,149.200000,-35.330000", "--maxzoom=15"]
    for rel, data in ASSET_FILES.items():
        assert (tiles / "assets" / rel).read_bytes() == data
    manifest = json.loads((tiles / "demo.json").read_text())
    assert manifest["protomaps_build"] == "20260811"
    assert manifest["maxzoom"] == 15
    assert manifest["bbox"] == [149.13, -35.39, 149.20, -35.33]
    assert manifest["sha256"] == _sha(FIXTURE.read_bytes())
    assert manifest["created_at"] == "2026-10-16T00:00:00+00:00"
    assert not list(tiles.glob(".*")), "no temporary files are left"
    # The program was installed from the checked archive, executable.
    assert (tmp_path / "bin" / "sd-pmtiles-9.9.9").read_bytes() == PROGRAM
    assert (tmp_path / "bin" / "sd-pmtiles-9.9.9").stat().st_mode & 0o777 == 0o755


def test_the_ground_station_reads_the_fetched_pack(tmp_path: Path) -> None:
    from sd_ground.api.tiles import read_center

    out = _run(tmp_path, _settings(), FakeTools())
    assert out is not None
    center = read_center(out)
    assert center is not None
    assert abs(center.lat - CMAC_LAT) < 0.01 and abs(center.lon - CMAC_LON) < 0.01


def test_dry_run_saves_nothing(tmp_path: Path) -> None:
    tools = FakeTools()
    assert _run(tmp_path, _settings(dry=True), tools) is None
    assert "--dry-run" in next(c for c in tools.commands if c[1] == "extract")
    assert not any(c[1] == "init" for c in tools.commands)
    assert list((tmp_path / "data" / "tiles").iterdir()) == []


def test_gone_build_exits_2_with_the_message(tmp_path: Path) -> None:
    with pytest.raises(fetch.FetchError) as err:
        _run(tmp_path, _settings(), FakeTools(), status=404)
    assert err.value.code == 2
    assert str(err.value) == (
        "Kartbygget 20260811 finns inte längre. Välj ett nytt på "
        "https://maps.protomaps.com/builds och ändra protomaps_build."
    )


def test_changed_build_is_refused(tmp_path: Path) -> None:
    tools = FakeTools()
    with pytest.raises(fetch.FetchError, match="har ändrats") as err:
        _run(tmp_path, _settings(), tools, etag='"other-2"')
    assert err.value.code == 1
    assert not any(c[1] == "extract" for c in tools.commands)


def test_weak_etag_matches(tmp_path: Path) -> None:
    assert _run(tmp_path, _settings(), FakeTools(), etag='W/"abc-1"') is not None


def test_a_pmtiles_of_the_user_is_left_alone(tmp_path: Path) -> None:
    (tmp_path / "pmtiles").write_bytes(b"someone else's program")
    fetch.ensure_pmtiles(_lock()["go_pmtiles"], tmp_path, lambda _: TARBALL)
    assert (tmp_path / "pmtiles").read_bytes() == b"someone else's program"


def test_other_http_errors_exit_1(tmp_path: Path) -> None:
    with pytest.raises(fetch.FetchError) as err:
        _run(tmp_path, _settings(), FakeTools(), status=503)
    assert err.value.code == 1


def test_wrong_asset_digest_copies_nothing_and_keeps_old_assets(tmp_path: Path) -> None:
    tiles = tmp_path / "data" / "tiles"
    (tiles / "assets").mkdir(parents=True)
    (tiles / "assets" / "old.txt").write_text("kept")
    bad = dict(ASSET_FILES) | {"sprites/v4/dark.png": b"tampered"}
    with pytest.raises(fetch.FetchError, match="kontrollsumman"):
        _run(tmp_path, _settings(), FakeTools(bad))
    assert (tiles / "assets" / "old.txt").read_text() == "kept"
    assert not (tiles / "demo.pmtiles").exists()


def test_extra_asset_file_fails(tmp_path: Path) -> None:
    extra = dict(ASSET_FILES) | {"sprites/v4/extra.json": b"{}"}
    with pytest.raises(fetch.FetchError):
        _run(tmp_path, _settings(), FakeTools(extra))


def test_sparse_checkout_of_the_locked_commit(tmp_path: Path) -> None:
    tools = FakeTools()
    _run(tmp_path, _settings(), tools)
    git = [c for c in tools.commands if c[1] != "extract"]
    fetch_cmd = next(c for c in git if "fetch" in c)
    assert fetch_cmd[-2:] == ["https://example.invalid/assets", "0" * 40]
    assert "--filter=blob:none" in fetch_cmd
    assert git[-1][-1] == "0" * 40  # checks out exactly the locked commit


# --- go-pmtiles ---------------------------------------------------------------------


def test_wrong_archive_checksum_installs_nothing(tmp_path: Path) -> None:
    lock = _lock()["go_pmtiles"]
    with pytest.raises(fetch.FetchError, match="kontrollsumman"):
        fetch.ensure_pmtiles(lock, tmp_path, lambda _: TARBALL + b"x")
    assert not (tmp_path / "sd-pmtiles-9.9.9").exists()


def test_wrong_program_checksum_installs_nothing(tmp_path: Path) -> None:
    lock = _lock()["go_pmtiles"] | {"binary_sha256": "0" * 64}
    with pytest.raises(fetch.FetchError, match="för programmet"):
        fetch.ensure_pmtiles(lock, tmp_path, lambda _: TARBALL)
    assert not (tmp_path / "sd-pmtiles-9.9.9").exists()


def test_symlink_in_archive_is_refused(tmp_path: Path) -> None:
    blob = _tarball({"LICENSE": b"x"}, symlink="pmtiles")
    lock = _lock()["go_pmtiles"] | {"sha256": _sha(blob)}
    with pytest.raises(fetch.FetchError, match="saknar programmet"):
        fetch.ensure_pmtiles(lock, tmp_path, lambda _: blob)


def test_installed_program_is_reused(tmp_path: Path) -> None:
    (tmp_path / "sd-pmtiles-9.9.9").write_bytes(PROGRAM)

    def no_download(_: str) -> bytes:
        raise AssertionError("must not download again")

    assert (
        fetch.ensure_pmtiles(_lock()["go_pmtiles"], tmp_path, no_download)
        == tmp_path / "sd-pmtiles-9.9.9"
    )


def test_only_https_is_fetched() -> None:
    with pytest.raises(fetch.FetchError, match="https"):
        fetch.download("http://example.invalid/x")


# --- assets digest ------------------------------------------------------------------


def test_tree_digest_matches_sha256sum_lines(tmp_path: Path) -> None:
    for rel, data in ASSET_FILES.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(data)
    files = fetch.asset_files(tmp_path, _lock()["assets"]["paths"])
    assert [f.relative_to(tmp_path).as_posix() for f in files] == sorted(ASSET_FILES)
    assert fetch.tree_digest(tmp_path, files) == _assets_digest()


def test_symlinked_asset_is_refused(tmp_path: Path) -> None:
    (tmp_path / "sprites" / "v4").mkdir(parents=True)
    (tmp_path / "sprites" / "v4" / "dark.json").symlink_to("/etc/hostname")
    with pytest.raises(fetch.FetchError, match="symlänk"):
        fetch.asset_files(tmp_path, ["sprites/v4"])
