"""detector=model: the trained person detector reaches sd-onboard only when chosen, and a
missing or changed model stops the start instead of falling back to the stand-in."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from sd_detect.onnx import MANIFEST_SCHEMA
from sd_detect.sources import MODEL_ENV
from sd_sim import config, detector_choice, launch, ports, procs, services, testrun
from sd_sim.detector_choice import DetectorChoice, DetectorError, choose
from sd_sim.people_scene import PeopleScene


def _manifest(folder: Path, *, sha256: str | None = None) -> Path:
    data = b"not a real graph; only its checksum is checked before the start"
    (folder / "person.onnx").write_bytes(data)
    manifest = folder / "person.json"
    body = {
        "schema": MANIFEST_SCHEMA,
        "onnx": "person.onnx",
        "family": "yolox",
        "sha256": sha256 or hashlib.sha256(data).hexdigest(),
        "input_px": 640,
        "classes": ["person"],
        "licence": "Apache-2.0",
    }
    manifest.write_text(json.dumps(body), encoding="utf-8")
    return manifest


def test_the_stand_in_is_the_default() -> None:
    choice = choose(environ={MODEL_ENV: "/somewhere/model.json"})
    assert choice == detector_choice.ORACLE and not choice.model
    assert choice.env() == {} and choice.name == "oracle"


def test_model_comes_from_the_key_or_the_environment(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    by_key = choose("model", str(manifest), environ={})
    by_env = choose(environ={detector_choice.MODE_ENV: "model", MODEL_ENV: str(manifest)})
    assert by_key == by_env == DetectorChoice("model", manifest)
    assert by_key.env() == {MODEL_ENV: str(manifest)}
    assert by_key.name == "model:person"


@pytest.mark.parametrize(
    ("detector", "model", "why"),
    [
        ("model", None, "kräver model="),
        ("model", "gone.json", "Kan inte läsa"),
        ("oracle", "x.json", "bara med detector=model"),
        ("yolo", None, "detector= ska vara"),
    ],
)
def test_a_wrong_choice_is_refused(
    tmp_path: Path, detector: str, model: str | None, why: str
) -> None:
    with pytest.raises(DetectorError, match=why):
        choose(detector, None if model is None else str(tmp_path / model), environ={})


def test_a_changed_model_is_refused(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, sha256="0" * 64)
    with pytest.raises(DetectorError, match="kontrollsumma"):
        choose("model", str(manifest), environ={})
    good = _manifest(tmp_path)
    (tmp_path / "person.onnx").unlink()
    with pytest.raises(DetectorError, match="finns inte"):
        choose("model", str(good), environ={})


def _paths(tmp_path: Path) -> services.StackPaths:
    return services.StackPaths(
        repo=tmp_path / "repo",
        run_dir=tmp_path / "repo/runs/r",
        secrets_dir=tmp_path / "repo/build/sim/i1/secrets",
        data_dir=tmp_path / "repo/runs/r/data",
        onboard_dir=tmp_path / "data/sim/i1/onboard",
    )


def _onboard_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detector: DetectorChoice, **stack: Any
) -> dict[str, str]:
    """The environment ``Stack.start_onboard`` starts sd-onboard with."""
    cfg = config.load({"physics": "builtin", "gui": "false", "video": "off", "instance": "1"})
    st = launch.Stack(cfg, tmp_path / "run", detector=detector, **stack)
    started: dict[str, dict[str, str]] = {}
    monkeypatch.setattr(st, "stack_paths", lambda: _paths(tmp_path))
    monkeypatch.setattr(st, "_wait", lambda *a, **k: None)
    monkeypatch.setattr(st, "_start_service", lambda name, argv, env: started.setdefault(name, env))
    st.start_onboard()
    return started["sd-onboard"]


def test_sd_onboard_gets_the_model_only_with_the_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    # A model named in the shell never leaks into the stand-in's run.
    monkeypatch.setenv(MODEL_ENV, str(manifest))
    plain = _onboard_env(tmp_path, monkeypatch, detector_choice.ORACLE)
    assert MODEL_ENV not in plain
    assert "SD_ONBOARD__PERCEPTION__ENABLED" not in plain  # perception stays off as before
    chosen = _onboard_env(tmp_path, monkeypatch, choose("model", str(manifest), {}))
    assert chosen[MODEL_ENV] == str(manifest)
    assert "SD_ONBOARD__PERCEPTION__ENABLED" not in chosen  # no camera without a scene
    assert set(chosen) - set(plain) == {MODEL_ENV}


def test_with_people_in_the_world_sd_onboard_reads_the_stills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sd_sim import search_world

    choice = choose("model", str(_manifest(tmp_path)), {})
    scene = PeopleScene(search_world.load("SK01", 1))
    env = _onboard_env(tmp_path, monkeypatch, choice, scene=scene)
    assert env[MODEL_ENV] == str(choice.manifest)
    assert env["SD_ONBOARD__PERCEPTION__ENABLED"] == "true"
    assert env["SD_ONBOARD__PERCEPTION__STILL_DIR"] == str(tmp_path / "run" / "stills")
    assert env["SD_ONBOARD__PERCEPTION__WIDTH_PX"] == "1280"


def test_onboard_env_has_no_model_by_default(tmp_path: Path) -> None:
    env = procs.clean_env(services.onboard_env(ports.for_instance(1), _paths(tmp_path)))
    assert MODEL_ENV not in env


def test_a_missing_model_stops_the_launch_before_anything_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def never(stack: launch.Stack) -> int:
        raise AssertionError("the stack must not start")

    monkeypatch.setattr(launch, "_run_foreground", never)
    monkeypatch.delenv(MODEL_ENV, raising=False)
    gone = str(tmp_path / "gone.json")
    assert launch.main(["up", "physics=builtin", "detector=model", f"model={gone}"]) == 2
    assert "Kan inte läsa" in capsys.readouterr().err
    assert launch.main(["up", "physics=builtin", "detector=model"]) == 2
    assert "kräver model=" in capsys.readouterr().err
    changed = _manifest(tmp_path, sha256="1" * 64)
    assert launch.main(["up", "physics=builtin", "detector=model", f"model={changed}"]) == 2
    assert "kontrollsumma" in capsys.readouterr().err


def test_the_launch_hands_the_checked_model_to_the_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[launch.Stack] = []
    monkeypatch.setattr(launch, "_run_foreground", lambda st: seen.append(st) or 0)
    manifest = _manifest(tmp_path)
    assert launch.main(["up", "physics=builtin", "detector=model", f"model={manifest}"]) == 0
    assert seen[-1].detector == DetectorChoice("model", manifest)
    assert seen[-1].scene is None  # builtin physics: no camera, so no people either
    assert launch.main(["up", "physics=builtin"]) == 0
    assert seen[-1].detector == detector_choice.ORACLE


def test_people_stand_in_the_world_only_with_the_model_and_gazebo(tmp_path: Path) -> None:
    choice = choose("model", str(_manifest(tmp_path)), {})
    gz = config.SimConfig(physics="gz", world="sd_lake_forest", vehicle="sd_quad14")
    scene = launch.people_scene_for(gz, choice)
    assert isinstance(scene, PeopleScene) and scene.world.scenario == "SK01"
    assert launch.people_scene_for(gz, detector_choice.ORACLE) is None
    builtin = config.SimConfig(physics="builtin", world="sd_lake_forest", vehicle="sd_quad14")
    assert launch.people_scene_for(builtin, choice) is None


def test_testrun_passes_the_model_only_with_the_switch(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert testrun.commands({"mode": "e2e-search"})[0][1] == {}
    env = testrun.commands({"mode": "e2e-search", "detector": "model", "model": str(manifest)})[0][
        1
    ]
    assert env == {detector_choice.MODE_ENV: "model", MODEL_ENV: str(manifest)}


@pytest.mark.parametrize(
    "values",
    [
        {"mode": "e2e-search", "detector": "model", "model": "gone.json"},
        {"mode": "e2e-search", "detector": "model"},
        {"mode": "search", "detector": "model", "model": "person.json"},
    ],
)
def test_testrun_refuses_a_bad_model_before_pytest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, values: dict[str, str]
) -> None:
    _manifest(tmp_path)
    monkeypatch.delenv(MODEL_ENV, raising=False)
    values = {k: str(tmp_path / v) if k == "model" else v for k, v in values.items()}
    with pytest.raises(SystemExit) as exc:
        testrun.commands(values)
    assert exc.value.code == 2


def test_testrun_ignores_an_inherited_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    class Done:
        returncode = 0

    def run(cmd: list[str], **kw: Any) -> Done:
        seen.update(kw["env"])
        return Done()

    monkeypatch.setenv(detector_choice.MODE_ENV, "model")
    monkeypatch.setattr(subprocess, "run", run)
    assert testrun.run_round({"mode": "e2e-search"}) == 0
    assert detector_choice.MODE_ENV not in seen
