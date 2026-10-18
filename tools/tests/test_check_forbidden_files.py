"""Tests for tools/check_forbidden_files.py.

Test coordinates, secrets and private patterns are built at runtime so this file
itself passes the checker.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import check_forbidden_files as cff
import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "check_forbidden_files.py"
PLACES = [cff.Place(*cff.CMAC)]
# A made-up point in the middle of the Atlantic, far from any test site.
FAR_LAT, FAR_LON = "30." + "0000", "-40." + "0000"
FAR_LAT3, FAR_LON3 = "30." + "123", "-40." + "456"
FAR_LAT_E7, FAR_LON_E7 = "3" + "00000000", "-4" + "00000000"
NEAR_LAT, NEAR_LON = "-35." + "363261", "149." + "165230"


def run(root: Path, *args: str, ci: bool | None = None) -> subprocess.CompletedProcess[str]:
    env = {"PATH": "/usr/bin:/bin"}
    if ci is not None:
        env["CI"] = "true" if ci else "false"
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def write(root: Path, rel: str, content: str | bytes) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content, encoding="utf-8")
    return rel


def findings(root: Path, rel: str) -> list[cff.Finding]:
    def read(p: str) -> bytes | None:
        f = root / p
        return f.read_bytes() if f.is_file() else None

    return cff.check_files([(rel, read)], PLACES, None)


# --- a) paths ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "docs/x.md",
        "planning/x.md",
        "research/x.md",
        "notes/x.md",
        "CLAUDE.md",
        "software/x/CLAUDE.md",
        "software/x/CLAUDE.local.md",
        "software/x/.claude/settings.json",
        ".mcp.json",
        "secrets/brain_token",
        ".env",
        ".env.production",
        "config/ground.toml",
        "config/local/sim.toml",
        "config/local/mediamtx.yml",
        "config/local/x",
        "software/x/settings.local.toml",
        "notes.local.txt",
        "cloudflared/config.yml",
        "config/local/x.example.toml",
        "docs/report.pdf",
        ".cloudflared/cert.json",
    ],
)
def test_forbidden_paths(path: str) -> None:
    assert any(f.rule == "forbidden-path" for f in cff.check_path(path)), path


@pytest.mark.parametrize(
    "path",
    [
        "config/mediamtx.yml",
        "config/ground.example.toml",
        ".env.example",
        "software/brain/src/sd_brain/planning/x.py",
        "software/ground/src/sd_ground/missions/x.py",
        "software/panel/src/data/x.ts",
        "software/common/schemas/generated/x.json",
        "README.md",
    ],
)
def test_allowed_paths(path: str) -> None:
    assert cff.check_path(path) == [], path


# --- b) types, fixture size and EXIF ------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["software/ground/forbidden-test.gif", "x.tlog", "flight.BIN", "a/b/route.plan", "k.pem"],
)
def test_forbidden_types(path: str) -> None:
    assert any(f.rule == "forbidden-type" for f in cff.check_path(path))


def test_types_allowed_in_testdata_and_panel_icons() -> None:
    assert cff.check_path("software/ground/tests/testdata/frame.png") == []
    assert cff.check_path("software/panel/public/icon.png") == []
    assert cff.check_path("software/panel/public/photo.jpg") != []


def _png(gps: bool) -> bytes:
    img = Image.new("RGB", (4, 4))
    exif = Image.Exif()
    if gps:
        exif[cff.GPS_IFD_TAG] = {1: "N", 2: (30.0, 0.0, 0.0)}
    buf = io.BytesIO()
    img.save(buf, format="PNG", exif=exif)
    return buf.getvalue()


def test_exif_gps_rejected(tmp_path: Path) -> None:
    rel = write(tmp_path, "software/x/tests/testdata/gps.png", _png(gps=True))
    assert any(f.rule == "exif-gps" for f in findings(tmp_path, rel))
    rel = write(tmp_path, "software/x/tests/testdata/clean.png", _png(gps=False))
    assert findings(tmp_path, rel) == []


def test_fixture_size_limit(tmp_path: Path) -> None:
    rel = write(tmp_path, "software/x/tests/testdata/big.json", "a" * (cff.MAX_FIXTURE_BYTES + 1))
    assert any(f.rule == "too-large" for f in findings(tmp_path, rel))


# --- c) size ---------------------------------------------------------------------------


def test_large_file_rejected_but_lockfiles_exempt(tmp_path: Path) -> None:
    big = "x" * (cff.MAX_FILE_BYTES + 10)
    rel = write(tmp_path, "software/ground/tests/blob.txt", big)
    assert any(f.rule == "too-large" for f in findings(tmp_path, rel))
    rel = write(tmp_path, "uv.lock", big)
    assert findings(tmp_path, rel) == []


# --- d) secrets -------------------------------------------------------------------------


@pytest.mark.parametrize("marker", cff.SECRET_MARKERS)
def test_secret_markers(tmp_path: Path, marker: str) -> None:
    rel = write(tmp_path, "software/x/conf.json", f'{{"v": "{marker}"}}\n')
    assert any(f.rule == "secret-like" for f in findings(tmp_path, rel))


def test_test_pem_in_testdata_allowed(tmp_path: Path) -> None:
    rel = write(tmp_path, "software/x/testdata/k.test.pem", cff.SECRET_MARKERS[2] + " TEST\n")
    assert findings(tmp_path, rel) == []


# --- e) coordinates ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rel", "body"),
    [
        ("simulation/scenarios/x.yaml", f"home:\n  lat: {FAR_LAT}\n  lon: {FAR_LON}\n"),
        (
            "simulation/gazebo/worlds/x.sdf",
            f"<latitude_deg>{FAR_LAT}</latitude_deg>\n<longitude_deg>{FAR_LON}</longitude_deg>\n",
        ),
        ("simulation/sitl/x.parm", f"SIM_OPOS_LAT {FAR_LAT}\nSIM_OPOS_LNG {FAR_LON}\n"),
        ("justfile", f"sim:\n    run --home={FAR_LAT},{FAR_LON},10,0\n"),
        ("x.geojson", f'{{"coordinates": [{FAR_LON}, {FAR_LAT}]}}\n'),
        ("software/x/a.py", f"POINT = ({FAR_LAT}, {FAR_LON})\n"),
        ("software/x/a.ts", f"const p = {{ lat: {FAR_LAT}, lng: {FAR_LON} }};\n"),
        ("software/x/b.toml", f"lat = {FAR_LAT3}\nlon = {FAR_LON3}\n"),
        ("software/x/c.json", f'{{"lat": {FAR_LAT_E7}, "lon": {FAR_LON_E7}}}\n'),
        ("software/x/d.md", "Meet at 30°0'0\"N, 40°0'0\"W.\n"),
        ("software/x/e.plan", f'{{"x": {FAR_LAT_E7}, "y": {FAR_LON_E7}}}\n'),
        ("software/x/f.yaml", "latitude: 30,1234\nlongitude: -40,5678\n"),
        ("software/x/g.toml", "home_lat = 30.0001\nhome_lon = -40.0001\n"),
    ],
)
def test_coordinates_far_away_fail(tmp_path: Path, rel: str, body: str) -> None:
    write(tmp_path, rel, body)
    assert any(f.rule == "coordinate" for f in findings(tmp_path, rel)), body


def test_coordinates_inside_allowlist_pass(tmp_path: Path) -> None:
    rel = write(tmp_path, "simulation/scenarios/x.yaml", f"lat: {NEAR_LAT}\nlon: {NEAR_LON}\n")
    assert findings(tmp_path, rel) == []


def test_free_pair_where_both_orders_are_valid_positions() -> None:
    """At a site like 59 N 18 E the swapped pair (18 N 59 E) is a real position too; the
    order that lies inside the site decides, in GeoJSON (lon, lat) and in (lat, lon) form."""
    site = cff.Place("site", 59.2884, 18.1595, 2000.0)
    for body in ("[18.15953, 59.28842]", "(59.28842, 18.15953)"):
        assert cff.check_coordinates("x.geojson", [body], [site]) == []
    far_lon, far_lat = 18 + 0.95953, 59 + 0.98842  # about 80 km away, built so this file passes
    far = cff.check_coordinates("x.geojson", [f"[{far_lon:.5f}, {far_lat:.5f}]"], [site])
    assert [f.rule for f in far] == ["coordinate"]
    # Keyed values have one order: a swapped keyed pair is still flagged.
    lat, lon = 18 + 0.15953, 59 + 0.28842
    keyed = [f"lat: {lat:.5f}", f"lon: {lon:.5f}"]
    assert [f.rule for f in cff.check_coordinates("x.yaml", keyed, [site])] == ["coordinate"]


def test_three_decimal_pair_without_key_is_ignored(tmp_path: Path) -> None:
    rel = write(tmp_path, "software/x/a.py", "RATIO = (30.123, -40.456)\n")
    assert findings(tmp_path, rel) == []


def test_lockfile_and_generated_not_scanned(tmp_path: Path) -> None:
    body = f"{FAR_LAT}, {FAR_LON}\n"
    for rel in ("pnpm-lock.yaml", "software/common/schemas/generated/x.json"):
        write(tmp_path, rel, body)
        assert findings(tmp_path, rel) == []


# --- f) private patterns ---------------------------------------------------------------


def test_private_pattern_hit_does_not_leak(tmp_path: Path) -> None:
    secret_word = "zq" + "xvprivatemarker"
    write(tmp_path, "config/local/private-patterns.txt", f"# comment\n\n{secret_word.upper()}\n")
    write(tmp_path, "software/x/a.py", f"# {secret_word}\n")
    res = run(
        tmp_path, "--private-patterns", "config/local/private-patterns.txt", "software/x/a.py"
    )
    assert res.returncode == 1
    assert "software/x/a.py:1: private-pattern: matches pattern line 3" in res.stdout
    assert secret_word not in (res.stdout + res.stderr).lower()


def test_private_regex_pattern(tmp_path: Path) -> None:
    write(tmp_path, "p.txt", "re:" + r"sec\d{3}word" + "\n")
    write(tmp_path, "a.txt", "SEC123WORD\n")
    assert run(tmp_path, "--private-patterns", "p.txt", "a.txt").returncode == 1


def test_missing_pattern_file_blocks_outside_ci(tmp_path: Path) -> None:
    write(tmp_path, "a.txt", "ok\n")
    res = run(tmp_path, "--private-patterns", "missing.txt", "a.txt", ci=False)
    assert res.returncode == 1
    assert cff.MISSING_PATTERNS_MSG in res.stderr
    res = run(tmp_path, "--private-patterns", "missing.txt", "a.txt", ci=True)
    assert res.returncode == 0


def test_empty_pattern_file_is_fine(tmp_path: Path) -> None:
    write(tmp_path, "p.txt", "")
    write(tmp_path, "a.txt", "anything\n")
    assert run(tmp_path, "--private-patterns", "p.txt", "a.txt").returncode == 0


# --- git modes --------------------------------------------------------------------------


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "test@users.noreply.github.com")
    _git(tmp_path, "config", "user.name", "test")
    write(tmp_path, "README.md", "ok\n")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_ls_files_mode(repo: Path) -> None:
    assert run(repo).returncode == 0
    write(repo, "notes.local.txt", "x\n")
    _git(repo, "add", "notes.local.txt")  # the scratch repo has no .gitignore
    res = run(repo)
    assert res.returncode == 1
    assert "notes.local.txt:0: forbidden-path" in res.stdout


def test_staged_mode_reads_index(repo: Path) -> None:
    rel = write(repo, "software/x/a.py", f"P = ({FAR_LAT}, {FAR_LON})\n")
    _git(repo, "add", rel)
    write(repo, rel, "P = None\n")  # working tree differs from the index
    res = run(repo, "--staged")
    assert res.returncode == 1
    assert "software/x/a.py:1: coordinate" in res.stdout


def test_range_mode(repo: Path) -> None:
    rel = write(repo, "software/x/a.txt", cff.SECRET_MARKERS[0] + "\n")
    _git(repo, "add", rel)
    _git(repo, "commit", "-q", "-m", "add")
    res = run(repo, "--range", "HEAD~1..HEAD")
    assert res.returncode == 1
    assert "secret-like" in res.stdout


def test_repository_itself_is_clean() -> None:
    root = Path(__file__).resolve().parents[2]
    res = run(root, ci=True)
    assert res.returncode == 0, res.stdout


def test_script_has_no_literal_marker() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert cff.check_secrets("tools/check_forbidden_files.py", text.splitlines()) == []


def test_range_mode_checks_every_commit(repo: Path) -> None:
    write(repo, "docs/plan.md", "internal\n")
    _git(repo, "add", "docs/plan.md")
    _git(repo, "commit", "-q", "-m", "add")
    _git(repo, "rm", "-q", "docs/plan.md")
    _git(repo, "commit", "-q", "-m", "remove")
    res = run(repo, "--range", "HEAD~2..HEAD")
    assert res.returncode == 1
    assert "docs/plan.md:0: forbidden-path" in res.stdout


def test_range_mode_uses_pre_commit_push_refs(repo: Path) -> None:
    rel = write(repo, "software/x/a.txt", cff.SECRET_MARKERS[1] + "\n")
    _git(repo, "add", rel)
    _git(repo, "commit", "-q", "-m", "add")
    env_res = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(repo), "--range", "HEAD..HEAD"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PRE_COMMIT_FROM_REF": "HEAD~1", "PRE_COMMIT_TO_REF": "HEAD"},
        check=False,
    )
    assert env_res.returncode == 1
    assert "secret-like" in env_res.stdout


def test_three_dot_range_rejected(repo: Path) -> None:
    res = run(repo, "--range", "HEAD~0...HEAD")
    assert res.returncode != 0
    assert "A..B" in res.stderr


def test_binary_file_still_checked_for_private_patterns(tmp_path: Path) -> None:
    secret_word = "zq" + "xvbinarymarker"
    write(tmp_path, "p.txt", secret_word + "\n")
    write(tmp_path, "blob.dat", b"\0\1\2" + secret_word.encode() + b"\n")
    res = run(tmp_path, "--private-patterns", "p.txt", "blob.dat")
    assert res.returncode == 1
    assert "blob.dat:1: private-pattern" in res.stdout


def test_private_pattern_in_path_hides_path(tmp_path: Path) -> None:
    secret_word = "zq" + "xvpathmarker"
    write(tmp_path, "p.txt", secret_word + "\n")
    rel = write(tmp_path, f"software/{secret_word}.txt", "ok\n")
    res = run(tmp_path, "--private-patterns", "p.txt", rel)
    assert res.returncode == 1
    assert "<path hidden>" in res.stdout
    assert secret_word not in res.stdout + res.stderr


def test_tracked_symlink_rejected(repo: Path) -> None:
    (repo / "link").symlink_to("/etc/hostname")
    _git(repo, "add", "link")
    res = run(repo)
    assert res.returncode == 1
    assert "link:0: symlink" in res.stdout


def test_geojson_lon_lat_with_large_longitude(tmp_path: Path) -> None:
    lon, lat = "-122." + "4194", "37." + "7749"
    rel = write(tmp_path, "software/x/a.geojson", f'{{"coordinates": [{lon}, {lat}]}}\n')
    assert any(f.rule == "coordinate" for f in findings(tmp_path, rel))
    rel = write(tmp_path, "software/x/b.geojson", f'{{"coordinates": [{NEAR_LON}, {NEAR_LAT}]}}\n')
    assert findings(tmp_path, rel) == []


def test_unreadable_git_content_fails_closed(tmp_path: Path) -> None:
    def broken(p: str) -> bytes | None:
        raise cff.UnreadableError(p)

    found = cff.check_files([("software/x/a.py", broken)], PLACES, None)
    assert [f.rule for f in found] == ["unreadable"]
