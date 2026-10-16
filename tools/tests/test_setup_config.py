import shutil
import stat
from pathlib import Path

import pytest
import setup_config

ROOT = Path(__file__).resolve().parents[2]


def _project(tmp_path: Path) -> Path:
    shutil.copyfile(ROOT / ".env.example", tmp_path / ".env.example")
    (tmp_path / "config").mkdir()
    for example in (ROOT / "config").glob("*.example.toml"):
        shutil.copyfile(example, tmp_path / "config" / example.name)
    return tmp_path


def _env(root: Path) -> dict[str, str]:
    out = {}
    for line in (root / ".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k] = v
    return out


def test_fills_generated_secrets(tmp_path: Path) -> None:
    root = _project(tmp_path)
    assert setup_config.main([str(root)]) == 0
    env = _env(root)
    kinds = setup_config.example_keys(root / ".env.example")
    for key, kind in kinds.items():
        if kind == "generated":
            assert len(env[key]) == 32, key
            assert env[key].lower() != "change_me"
        if kind == "optional":
            assert env[key] == ""
    assert env["SD_DATA_DIR"]
    assert stat.S_IMODE((root / ".env").stat().st_mode) == 0o600
    assert stat.S_IMODE((root / "secrets").stat().st_mode) == 0o700


def test_existing_values_and_configs_kept(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / ".env").write_text("SD_GROUND__DB_PASSWORD=keepme\nSD_LLM_API_KEY=CHANGE_ME\n")
    (root / "config" / "ground.toml").write_text("# mine\n")
    setup_config.main([str(root)])
    env = _env(root)
    assert env["SD_GROUND__DB_PASSWORD"] == "keepme"
    assert len(env["SD_LLM_API_KEY"]) == 32
    assert (root / "config" / "ground.toml").read_text() == "# mine\n"
    assert (root / "config" / "brain.toml").is_file()


def test_idempotent(tmp_path: Path) -> None:
    root = _project(tmp_path)
    setup_config.main([str(root)])
    first = _env(root)
    setup_config.main([str(root)])
    assert _env(root) == first


def test_existing_world_readable_env_becomes_private(tmp_path: Path) -> None:
    root = _project(tmp_path)
    env = root / ".env"
    env.write_text("SD_LLM_API_KEY=\n")
    env.chmod(0o644)
    setup_config.main([str(root)])
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert not list(root.glob(".*.tmp"))


def test_symlinked_env_refused(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "elsewhere").write_text("")
    (root / ".env").symlink_to(root / "elsewhere")
    with pytest.raises(SystemExit):
        setup_config.main([str(root)])
