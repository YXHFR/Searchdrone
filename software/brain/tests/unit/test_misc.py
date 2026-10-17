"""Health counters, the config hash, the CLI and the app wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from brain_helpers import REG, with_stub

from sd_brain.ai_config import ai_config_hash, model_sha256, prompt_text
from sd_brain.app import build, registry_mismatch
from sd_brain.checkmodel import vram_peak, write_report
from sd_brain.cli import _git_sha, _ram_total_gb, main
from sd_brain.config import BrainConfig
from sd_brain.health import HealthState
from sd_brain.runtime.base import Sampling
from sd_brain.secrets import Secrets
from sd_schemas.skills import SKILL_REGISTRY, SkillSpecPublic

ROOT = Path(__file__).resolve().parents[4]


class _Clock:
    t = 0.0

    def __call__(self) -> float:
        return self.t


def test_health_probe_and_turn_rules() -> None:
    c = _Clock()
    h = HealthState(c, has_probe=True)
    for _ in range(2):
        h.probe(False)
    assert not h.degraded
    h.probe(False)
    assert h.degraded and h.cause == "probe" and not h.may_recover()
    h.probe(True)
    h.probe(True)
    assert h.may_recover() and not h.needs_test_call()
    h.recovered()
    for _ in range(3):
        h.turn(False)
    assert h.degraded and h.cause == "turn"
    h.probe(True)
    h.probe(True)
    assert not h.may_recover()
    c.t = 61
    assert h.may_recover() and h.needs_test_call()
    s = HealthState(c, has_probe=False)
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
    assert h != ai_config_hash(REG, Sampling(), cloud_model="vendor/model")
    assert h != ai_config_hash(with_stub("report"), Sampling())
    assert len(model_sha256()) == 64
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
    secrets = Secrets(brain_token="t" * 40, llm_api_key="k" * 40, cloud_api_key="c" * 40)
    b = build(BrainConfig.model_validate({"cloud": {"model": "vendor/model"}}), secrets)
    assert b.runtime.engine() == "cloud"
    assert b.loop.heartbeat()["engine"] == "cloud"
    assert b.loop.decide is not None and b.loop.decide.available() == ["local"]
    await b.http.aclose()
    assert b.cloud_http is not None
    await b.cloud_http.aclose()
    local = build(BrainConfig(), Secrets(brain_token="t" * 40, llm_api_key="k" * 40))
    assert local.runtime.engine() == "local" and local.cloud_http is None
    await local.http.aclose()


def test_build_scripted_needs_file() -> None:
    from sd_brain.app import StartError

    with pytest.raises(StartError):
        build(
            BrainConfig.model_validate({"backend": {"kind": "scripted"}}),
            Secrets("t" * 40, "k" * 40),
        )


def test_cli_config_check_and_eval(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config-check", "--config", str(ROOT / "config/brain.example.toml")]) == 0
    assert main(["config-check", "--config", "/nonexistent.toml"]) == 1
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


def test_cli_check_model_refuses_weak_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SD_LLM_API_KEY", "CHANGE_ME")
    monkeypatch.chdir(tmp_path)
    assert main(["check-model", "--duration-s", "1"]) == 2


def test_vram_and_reports(tmp_path: Path) -> None:
    log = tmp_path / "smi.csv"
    log.write_text("3500\n6800\nbad\n4000, x\n")
    assert vram_peak(log) == 6800
    assert vram_peak(None) is None and vram_peak(tmp_path / "none") is None
    path = write_report(tmp_path / "r", {"calls": 1})
    assert json.loads(path.read_text()) == {"calls": 1}
    assert _git_sha(tmp_path) is None
    assert _ram_total_gb() is None or _ram_total_gb() > 0  # type: ignore[operator]
