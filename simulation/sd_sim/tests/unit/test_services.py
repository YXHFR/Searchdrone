"""The environments the launcher gives sd-ground, sd-onboard and sd-brain."""

from __future__ import annotations

from pathlib import Path

import pytest

from sd_sim import ports, services
from sd_sim.errors import InfraError
from sd_sim.launch import Stack


def _paths(tmp_path: Path) -> services.StackPaths:
    return services.StackPaths(
        repo=tmp_path / "repo",
        run_dir=tmp_path / "repo/runs/r",
        secrets_dir=tmp_path / "repo/build/sim/i1/secrets",
        data_dir=tmp_path / "repo/runs/r/data",
        onboard_dir=tmp_path / "data/sim/i1/onboard",
    )


@pytest.mark.parametrize("instance", [0, 1, 7])
def test_ground_env_uses_the_instance_ports(tmp_path: Path, instance: int) -> None:
    p = ports.for_instance(instance)
    env = services.ground_env(p, _paths(tmp_path), instance, video=False, lan=False, test_rig=False)
    assert env["SD_GROUND__HTTP__PANEL_BIND"] == f"127.0.0.1:{p['ground_panel']}"
    assert env["SD_GROUND__HTTP__COMMANDER_BIND"] == f"127.0.0.1:{p['ground_commander']}"
    assert env["SD_GROUND__SDLINK__BIND"] == f"127.0.0.1:{p['ground_sdlink']}"
    assert env["SD_GROUND__MAVLINK__LISTEN"] == f"udpin:127.0.0.1:{p['ground_mavlink']}"
    assert env["SD_GROUND__DB__URL"].endswith(f"/searchdrone_i{instance}")
    assert env["SD_GROUND__DATA_DIR"] == env["SD_DATA_DIR"]
    assert env["SD_GROUND__AUTH__CLI_MAY_CONFIRM"] == "false"
    assert env["SD_GROUND__REMOTE__ENABLED"] == "false"


def test_only_the_test_rig_lets_cli_tokens_confirm(tmp_path: Path) -> None:
    p = ports.for_instance(0)
    env = services.ground_env(p, _paths(tmp_path), 0, video=True, lan=True, test_rig=True)
    assert env["SD_GROUND__AUTH__CLI_MAY_CONFIRM"] == "true"
    assert env["SD_GROUND__VIDEO__ENABLED"] == "true"
    assert env["SD_GROUND__LAN__ENABLED"] == "true"


def test_no_secret_values_in_any_environment(tmp_path: Path) -> None:
    p = ports.for_instance(1)
    sp = _paths(tmp_path)
    envs = [
        services.ground_env(p, sp, 1, video=False, lan=False, test_rig=True),
        services.onboard_env(p, sp),
        services.brain_env(p, sp, "scripted", tmp_path / "SC01.yaml"),
    ]
    for env in envs:
        for key in env:
            assert not key.endswith(("PASSWORD", "TOKEN", "KEY")), key


def test_onboard_keeps_its_token_and_state_outside_the_repository(tmp_path: Path) -> None:
    sp = _paths(tmp_path)
    env = services.onboard_env(ports.for_instance(1), sp)
    for key in ("SD_ONBOARD__LINK_TOKEN_FILE", "SD_ONBOARD__STATE_DIR"):
        assert not Path(env[key]).is_relative_to(sp.repo), key
    assert env["SD_ONBOARD__REQUIRE_RC"] == "false"
    assert env["SD_ONBOARD__SDLINK_URL"].startswith("ws://127.0.0.1:")


def test_scripted_brain_needs_a_script(tmp_path: Path) -> None:
    p = ports.for_instance(0)
    with pytest.raises(InfraError):
        services.brain_env(p, _paths(tmp_path), "scripted", None)
    env = services.brain_env(p, _paths(tmp_path), "llm_local", None)
    assert "SD_BRAIN__BACKEND__SCRIPTED_FILE" not in env
    assert env["SD_BRAIN__GROUND__COMMANDER_URL"] == (
        f"http://127.0.0.1:{p['ground_commander']}/commander/v1"
    )


def test_cloud_off_empties_the_cloud_model(tmp_path: Path) -> None:
    p = ports.for_instance(0)
    default = services.brain_env(p, _paths(tmp_path), "llm_local", None)
    assert "SD_BRAIN__CLOUD__MODEL" not in default
    off = services.brain_env(p, _paths(tmp_path), "llm_local", None, cloud=False)
    assert off["SD_BRAIN__CLOUD__MODEL"] == ""
    assert {k: v for k, v in off.items() if k != "SD_BRAIN__CLOUD__MODEL"} == default


def test_health_reads_parts() -> None:
    h = services.Health(True, {"db": "ok", "brain": "down"})
    assert h.ok("db") and not h.ok("db", "brain")


def test_stack_starts_no_services_unless_asked(tmp_path: Path) -> None:
    from sd_sim import config

    cfg = config.load({"physics": "builtin", "gui": "false", "video": "off", "instance": "3"})
    assert Stack(cfg, tmp_path).services is False
    assert Stack(cfg, tmp_path, services=True).services is True


class _StartedError(Exception):
    pass


def test_stack_without_cloud_gives_brain_no_cloud_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sd_sim import config

    cfg = config.load({"physics": "builtin", "gui": "false", "video": "off", "instance": "3"})
    cfg.brain = "grammar"  # no llama-server start
    assert Stack(cfg, tmp_path).cloud is True
    stack = Stack(cfg, tmp_path, services=True, cloud=False)
    seen: dict[str, str] = {}

    def fake_start(name: str, argv: list[str], env: dict[str, str]) -> None:
        seen.update(env)
        raise _StartedError

    monkeypatch.setattr(stack, "stack_paths", lambda: _paths(tmp_path))
    monkeypatch.setattr(stack, "_start_service", fake_start)
    with pytest.raises(_StartedError):
        stack.start_brain()
    assert seen["SD_BRAIN__CLOUD__MODEL"] == ""
    assert seen["SD_BRAIN__BACKEND__KIND"] == "grammar"
