"""Rules for the committed simulator files."""

import re
import subprocess
from pathlib import Path

from sd_sim import paths

ROOT = paths.REPO
SIM = ROOT / "simulation"
DOCKERFILE = SIM / "docker" / "sitl.Dockerfile"
KNOWN_PARAMS = SIM / "sitl" / "params" / "known_params_Copter-4.7.1.txt"
KNOWN_KEYS = SIM / "sitl" / "frames" / "known_keys.txt"


def tracked(prefix: str) -> list[Path]:
    """Committed files plus new files that are not ignored (never gitignored local files)."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", prefix],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode()
    return sorted({ROOT / name for name in out.split("\0") if name and (ROOT / name).is_file()})


def read_list(path: Path) -> list[str]:
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def test_versions_consistent() -> None:
    lock = paths.versions()
    text = DOCKERFILE.read_text(encoding="utf-8")
    default = re.search(r"^ARG UBUNTU_IMAGE=(\S+)$", text, re.M)
    assert default is not None and default.group(1) == lock["ubuntu_image"]
    assert re.fullmatch(r"ubuntu:\S+@sha256:[0-9a-f]{64}", lock["ubuntu_image"])
    assert "ver ardupilot_tag" in text and "ver ardupilot_commit" in text
    assert lock["ardupilot_tag"] == "Copter-4.7.1"
    assert re.fullmatch(r"[0-9a-f]{40}", lock["ardupilot_commit"])
    assert re.fullmatch(r"[0-9a-f]{40}", lock["ardupilot_gazebo_commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", lock["uv_sha256"])
    for name in ("ARDUPILOT_REF", "FIRMWARE.lock"):
        assert not list(ROOT.glob(f"**/{name}")), name
    assert KNOWN_PARAMS.name == f"known_params_{lock['ardupilot_tag']}.txt"


def test_dockerfile_copies_only_allowed_files() -> None:
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if not words or words[0] not in ("COPY", "ADD"):
            continue
        assert words[0] == "COPY", line
        if any(w.startswith("--from=") for w in words):
            continue
        sources = [w for w in words[1:-1] if not w.startswith("--")]
        for src in sources:
            assert src == "simulation/versions.lock" or src.startswith("simulation/docker/"), line


def test_dockerignore_sends_only_the_lock() -> None:
    lines = read_list(SIM / "docker" / "sitl.Dockerfile.dockerignore")
    assert lines == ["*", "!simulation/versions.lock", "!simulation/docker/known_params.py"]
    # Everything the Dockerfile copies must be sent to the builder.
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if words[:1] == ["COPY"] and not any(w.startswith("--from=") for w in words):
            assert f"!{words[1]}" in lines, line


def test_wsl_setup_uses_locked_versions() -> None:
    text = (ROOT / "tools" / "setup" / "wsl-setup.sh").read_text(encoding="utf-8")
    for key in ("ardupilot_tag", "ardupilot_commit", "ardupilot_gazebo_commit", "uv_sha256"):
        assert f"ver {key}" in text, key
    assert "/mnt/" in text  # refuses a clone under /mnt/c


def test_known_lists() -> None:
    params = set(read_list(KNOWN_PARAMS))
    assert {"MAV_GCS_SYSID", "FS_GCS_ENABLE", "MOT_THST_HOVER", "SERIAL5_PROTOCOL"} <= params
    # Names that no longer exist in 4.7 must not be accepted.
    assert not {"SYSID_MYGCS", "SYSID_THISMAV", "WPNAV_SPEED", "RTL_ALT", "ARMING_CHECK"} & params
    keys = read_list(KNOWN_KEYS)
    assert {"mass", "diagonal_size", "moment_inertia", "num_motors"} <= set(keys)


def test_param_names_exist() -> None:
    """Every name in a committed parameter file exists in Copter-4.7.1."""
    known = set(read_list(KNOWN_PARAMS))
    for parm in sorted((SIM / "sitl" / "params").rglob("*.parm")):
        for line in read_list(parm):
            name = line.split()[0].split(",")[0]
            assert name in known, f"{parm.name}: unknown parameter {name}"


SPEC_REF = re.compile(
    r"(?i)\b0[1-8] ?§|\b(0[1-8] )?[DNS][0-9]+b?\b.*\b01\b|\bS-[0-9]{2}\b|\bB[0-9]-[0-9]{2}\b"
    r"|\b(EV|AC-B)-?[0-9]{2}\b|docs/spec"
)
# The only allowed mention of a ticket id: the "not built yet" message of a stub.
NOT_BUILT = re.compile(r"not_built\(\"B[0-9]-[0-9]{2}\"\)|inte byggt än \(B[0-9]-[0-9]{2}\)")
THIS_FILE = Path(__file__).resolve()
TEXT_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".txt",
    ".json",
    ".lock",
    ".parm",
    ".sh",
    "",
}


def test_no_spec_refs() -> None:
    hits = []
    for path in tracked("simulation"):
        if path.suffix not in TEXT_SUFFIXES and not path.name.endswith(".Dockerfile"):
            continue
        if path.name.startswith("known_params_") or path.resolve() == THIS_FILE:
            continue  # ArduPilot's own parameter names; the patterns above
        for no, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if SPEC_REF.search(NOT_BUILT.sub("", line)):
                hits.append(f"{path.relative_to(ROOT)}:{no}")
    assert not hits, hits


def test_no_localhost_pilot() -> None:
    for top in ("software", "config", "simulation"):
        for path in tracked(top):
            if (
                path.suffix in {".py", ".toml", ".yaml", ".yml", ".md", ".json"}
                and path.resolve() != THIS_FILE
            ):
                assert "localhost_is_pilot" not in path.read_text(encoding="utf-8").lower(), path
