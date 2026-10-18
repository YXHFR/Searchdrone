"""Health counters, the config hash, the CLI and the app wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from brain_helpers import REG, with_stub

from sd_brain.ai_config import ai_config_hash, prompt_text
from sd_brain.app import build, registry_mismatch
from sd_brain.cli import _git_sha, _ram_total_gb, main
from sd_brain.config import BrainConfig, CloudConfig
from sd_brain.health import HealthState
from sd_brain.runtime.base import Sampling
from sd_brain.secrets import Secrets
from sd_schemas.skills import SKILL_REGISTRY, SkillSpecPublic

ROOT = Path(__file__).resolve().parents[4]


class _Clock:
    t = 0.0

    def __call__(self) -> float:
        return self.t


def test_health_turn_rules() -> None:
    c = _Clock()
    h = HealthState(c)
    for _ in range(2):
        h.turn(False)
    assert not h.degraded
    h.turn(True)  # a good turn resets the count
    for _ in range(3):
        h.turn(False)
    assert h.degraded and not h.may_recover()
    c.t = 59
    assert not h.may_recover()
    c.t = 61
    assert h.may_recover()
    h.recovered()
    assert not h.degraded and h.turn_fails == 0
    s = HealthState(c)
    s.force_degrade()
    assert s.degraded and not s.may_recover()
    c.t = 200
    assert s.may_recover()
    s.turn(True)
    assert s.turn_fails == 0


def test_ai_config_hash() -> None:
    h = ai_config_hash(REG, Sampling())
    assert h == ai_config_hash(REG, Sampling()) and len(h) == 64
    assert h != ai_config_hash(REG, Sampling(temperature=0.3))
    cloud = ai_config_hash(REG, Sampling(), cloud=CloudConfig())
    assert h != cloud
    assert cloud != ai_config_hash(REG, Sampling(), cloud=CloudConfig(reasoning_effort="high"))
    assert cloud != ai_config_hash(REG, Sampling(), cloud=CloudConfig(fallback_model="other/m"))
    assert h != ai_config_hash(with_stub("report"), Sampling())
    assert "You are the mission commander AI" in prompt_text("system_v1.md")


def test_registry_mismatch() -> None:
    public = [
        SkillSpecPublic.model_validate(json.loads(s.model_dump_json()))
        for s in SKILL_REGISTRY.public()
    ]
    assert registry_mismatch(SKILL_REGISTRY, public) == []
    # A ground station of another build: one skill is still a stub there.
    assert registry_mismatch(with_stub("report"), public) == ["report"]
    assert registry_mismatch(SKILL_REGISTRY, public[1:]) == [public[0].name]


async def test_build_wires_everything() -> None:
    secrets = Secrets(brain_token="t" * 40, cloud_api_key="c" * 40)
    b = build(BrainConfig.model_validate({"cloud": {"model": "vendor/model"}}), secrets)
    assert b.runtime.engine() == "cloud"
    assert b.loop.heartbeat() == {"status": "idle", "backend": "cloud", "engine": "cloud"}
    assert b.loop.decide is not None and b.loop.decide.available() == []
    await b.http.aclose()
    assert b.cloud_http is not None
    await b.cloud_http.aclose()
    offline = build(BrainConfig(), Secrets(brain_token="t" * 40))
    assert offline.runtime.engine() == "command" and offline.cloud_http is None
    beat = offline.loop.heartbeat()
    assert beat["engine"] == "command" and beat["status"] == "degraded"
    assert beat["backend"] == "grammar"  # the command mode answers
    await offline.http.aclose()


def test_build_scripted_needs_file() -> None:
    from sd_brain.app import StartError

    with pytest.raises(StartError):
        build(
            BrainConfig.model_validate({"backend": {"kind": "scripted"}}),
            Secrets("t" * 40),
        )


def test_cli_config_check_and_eval(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert main(["config-check", "--config", str(ROOT / "config/brain.example.toml")]) == 0
    assert main(["config-check", "--config", "/nonexistent.toml"]) == 1
    capsys.readouterr()
    old = tmp_path / "brain.toml"
    old.write_text('profile = "field"\n[cloud.provider]\nzdr = false\n')
    assert main(["config-check", "--config", str(old)]) == 0
    assert "zdr = false" in capsys.readouterr().err
    scen = ROOT / "software/brain/src/sd_brain/evals/scenarios"
    gold = ROOT / "software/brain/tests/golden"
    assert (
        main(
            [
                "eval",
                "--scenarios",
                str(scen),
                "--golden",
                str(gold),
                "--only",
                "ev01,ev31",
                "--config",
                "/none",
            ]
        )
        == 0
    )
    assert "EV-01: ok" in capsys.readouterr().out


def test_cli_perf_cloud_skips_without_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SD_CLOUD_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    assert main(["perf-cloud", "--runs", "1", "--report-dir", str(tmp_path / "r")]) == 0
    assert "hoppar över molnets svarstid" in capsys.readouterr().out
    assert not (tmp_path / "r").exists()


def test_cli_eval_cloud_needs_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SD_CLOUD_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    assert main(["eval", "--backend", "cloud", "--config", "/none"]) == 1


def test_cli_helpers(tmp_path: Path) -> None:
    assert _git_sha(tmp_path) is None
    assert _ram_total_gb() is None or _ram_total_gb() > 0  # type: ignore[operator]
