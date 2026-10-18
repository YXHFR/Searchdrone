"""The farm loop (rounds in their own process, the stop rule, cleanup) and its status."""

from __future__ import annotations

import datetime as dt
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from sd_sim.farm import loop, status
from sd_sim.farm import round as farm_round
from sd_sim.farm.config import FarmConfig
from sd_sim.farm.host import Gate, Limits
from sd_sim.farm.store import Ledger, R2Config, UploadError

# A stand-in for ``python -m sd_sim.farm round``: the next mode from a list decides
# what the round does.
FAKE_ROUND = """
import sys, time
from pathlib import Path
from sd_sim.farm import round as r
spec_path, modes = Path(sys.argv[1]), Path(sys.argv[2])
lines = modes.read_text().splitlines()
mode, rest = lines[0], lines[1:]
modes.write_text("\\n".join(rest))
spec = r.RoundSpec.read(spec_path)
out = spec_path.parent
(out / "work").mkdir()
(out / "work" / "raw.ppm").write_text("raw")
if mode == "ok":
    (out / "images").mkdir()
    (out / "images" / "a.png").write_text("png")
    r.write_result(out, spec, {"status": "ok", "error": None, "images": 1,
                               "qc": {"labels": 2}, "grade": {"is_exam": True, "cost_sek": 1.5}})
    sys.exit(r.EXIT_OK)
if mode == "fail":
    r.write_result(out, spec, {"status": "failed", "error": "svarta bilder"})
    sys.exit(r.EXIT_QC)
if mode == "not_built":
    r.write_result(out, spec, {"status": "not_built", "error": "inte byggt"})
    sys.exit(r.EXIT_NOT_BUILT)
if mode == "hang":
    (out / "started").write_text("")
    time.sleep(60)
sys.exit(3)
"""


class FakeFarm(loop.Farm):
    modes: Path

    def worktree(self) -> Path:
        self.commit = "c0ffee" * 6 + "abcd"
        return self.repo

    def round_argv(self, spec_path: Path) -> list[str]:
        return [sys.executable, "-c", FAKE_ROUND, str(spec_path), str(self.modes)]


# A disk budget the host's own disk can never trip.
ROOMY = Limits(max_duty=1.0, max_local_gb=1e6, min_free_gb=0.0)


def farm(
    tmp_path: Path,
    modes: list[str],
    rounds: int = 0,
    r2: R2Config | None = None,
    limits: Limits = ROOMY,
) -> FakeFarm:
    cfg = FarmConfig("script", ("sd_lake_forest", "w2"), limits, r2)
    gate = Gate(
        cfg.limits,
        tmp_path / "build" / "simfarm" / "rounds",
        read_gpu=lambda: None,
        read_cpu=lambda: 0.0,
        read_idle=lambda: None,
    )
    f = FakeFarm(
        tmp_path, cfg, gate, loop.Stop(), rounds_wanted=rounds, seed=7, sleep=lambda _s: None
    )
    f.modes = tmp_path / "modes.txt"
    f.modes.write_text("\n".join(modes))
    return f


def test_passed_rounds_are_kept_without_their_work(tmp_path: Path) -> None:
    f = farm(tmp_path, ["ok", "ok"], rounds=2)
    assert f.run() == 0
    kept = sorted(p for p in f.rounds_dir.iterdir())
    assert len(kept) == 2 and not any(p.name.startswith(".") for p in kept)
    for d in kept:
        assert (d / "round.json").is_file() and (d / "images" / "a.png").is_file()
        assert not (d / "work").exists()
    rows = status.read_rounds(f.base / "rounds.jsonl")
    assert [r["seed"] for r in rows] == [7, 8] and rows[0]["labels"] == 2


def test_three_failures_in_a_row_stop_the_farm(tmp_path: Path) -> None:
    f = farm(tmp_path, ["fail", "ok", "fail", "fail", "fail", "ok"])
    assert f.run() == 1
    failed = sorted((f.base / "failed").iterdir())
    assert len(failed) == 4
    assert all(sorted(p.name for p in d.iterdir()) == ["round.json", "spec.json"] for d in failed)
    assert len(list(f.rounds_dir.iterdir())) == 1
    assert json.loads((f.base / "state.json").read_text())["state"] == "stopped"
    assert f.modes.read_text() == "ok"  # the round after the third failure never ran


def test_not_built_stops_with_exit_2(tmp_path: Path) -> None:
    f = farm(tmp_path, ["not_built"])
    assert f.run() == 2
    assert list(f.rounds_dir.iterdir()) == []
    assert status.read_rounds(f.base / "rounds.jsonl") == []


def test_stop_removes_the_half_made_round(tmp_path: Path) -> None:
    f = farm(tmp_path, ["hang"])

    def stop_when_started() -> None:
        for _ in range(200):
            if list(f.rounds_dir.glob(".partial-*/started")):
                break
            time.sleep(0.05)
        f.stop.handler(2, None)

    t = threading.Thread(target=stop_when_started)
    t.start()
    started = time.monotonic()
    assert f.run() == 0
    t.join()
    assert time.monotonic() - started < loop.STOP_GRACE_S
    assert list(f.rounds_dir.iterdir()) == []
    assert json.loads((f.base / "state.json").read_text())["reason"] == "avbruten"


def test_time_cap_stops_the_round_and_exits_cleanly(tmp_path: Path) -> None:
    f = farm(tmp_path, ["hang", "ok"])
    f.max_minutes = 1.0

    jumped: list[bool] = []

    def clock() -> float:  # jumps past the cap once the round has started
        if not jumped and any(f.rounds_dir.glob(".partial-*/started")):
            jumped.append(True)
        return time.monotonic() + (120.0 if jumped else 0.0)

    f.clock = clock
    started = time.monotonic()
    assert f.run() == 0
    assert time.monotonic() - started < loop.STOP_GRACE_S
    assert list(f.rounds_dir.iterdir()) == []  # the stopped round was removed
    assert json.loads((f.base / "state.json").read_text())["reason"] == "tidstaket nått"
    assert f.modes.read_text() == "ok"


def test_a_round_that_runs_too_long_is_an_error(tmp_path: Path) -> None:
    f = farm(tmp_path, ["hang", "ok"], rounds=1)
    f.round_timeout_s = 1.0
    assert f.run() == 0
    (failed,) = (f.base / "failed").iterdir()
    assert json.loads((failed / "round.json").read_text())["error"] == "varvet tog för lång tid"
    assert json.loads((failed / "spec.json").read_text())["worlds"] == ["sd_lake_forest", "w2"]
    assert f.streak == 1


R2 = R2Config("0123456789abcdef0123456789abcdef", "sd-simfarm", "AK", "SK")


def test_make_room_removes_only_uploaded_rounds_oldest_first(tmp_path: Path) -> None:
    f = farm(tmp_path, [], r2=R2, limits=Limits(max_duty=1.0, max_local_gb=4e-6, min_free_gb=0.0))
    for name, size in (("r1", 2000), ("r2", 2000), ("r3", 1500)):
        (f.rounds_dir / name).mkdir(parents=True)
        (f.rounds_dir / name / "data").write_bytes(b"x" * size)
    (f.base / "uploaded.jsonl").write_text('{"round": "r1"}\n{"round": "r3"}\n')
    assert f.make_room() is None  # removing r1 is enough
    assert sorted(p.name for p in f.rounds_dir.iterdir()) == ["r2", "r3"]
    (f.rounds_dir / "r2" / "more").write_bytes(b"x" * 3000)
    reason = f.make_room()  # r3 goes too, and r2 (not uploaded) still does not fit
    assert reason is not None and "tak" in reason
    assert sorted(p.name for p in f.rounds_dir.iterdir()) == ["r2"]


def test_make_room_never_removes_without_an_upload(tmp_path: Path) -> None:
    f = farm(tmp_path, [], limits=Limits(max_duty=1.0, max_local_gb=1e-6, min_free_gb=0.0))
    (f.rounds_dir / "r1").mkdir(parents=True)
    (f.rounds_dir / "r1" / "data").write_bytes(b"x" * 5000)
    (f.base / "uploaded.jsonl").write_text('{"round": "r1"}\n')  # no key: the ledger is stale
    assert f.make_room() is not None
    assert (f.rounds_dir / "r1").is_dir()


class FakeStore:
    def __init__(self, fail_on: str | None = None) -> None:
        self.sent: list[str] = []
        self.fail_on = fail_on

    def upload_round(self, d: Path) -> bool:
        if d.name == self.fail_on:
            raise UploadError("R2 svarar inte")
        self.sent.append(d.name)
        return True


def test_upload_backlog_in_order_and_waits_on_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = farm(tmp_path, [], r2=R2)
    for name in ("r1", "r2", "r3", ".partial-r4"):
        (f.rounds_dir / name).mkdir(parents=True)
    fake = FakeStore(fail_on="r2")
    monkeypatch.setattr(f, "store", lambda: fake)
    f.upload_backlog()
    assert fake.sent == ["r1"]  # stops at the error and tries again next turn
    fake.fail_on = None
    f.upload_backlog()
    assert fake.sent == ["r1", "r1", "r2", "r3"]  # never the half-made round
    f.stop.requested = True
    f.upload_backlog()
    assert len(fake.sent) == 4


def test_round_command_runs_under_nice_from_the_worktree(tmp_path: Path) -> None:
    f = farm(tmp_path, [])
    argv = loop.Farm.round_argv(f, Path("/x/spec.json"))
    assert argv[:3] == ["nice", "-n", "10"]
    assert argv[-3:] == ["sd_sim.farm", "round", "spec=/x/spec.json"]


def test_worktree_links_local_config_but_not_examples(tmp_path: Path) -> None:
    repo, wt = tmp_path / "repo", tmp_path / "wt"
    (repo / "config").mkdir(parents=True)
    (repo / "config" / "simfarm.toml").write_text("[r2]\n")
    (repo / "config" / "simfarm.example.toml").write_text("")
    (repo / "config" / "site.toml").write_text("")
    (repo / ".env").write_text("X=1\n")
    loop.Worktree(repo, wt).link_local()
    assert (wt / "config" / "site.toml").resolve() == repo / "config" / "site.toml"
    assert (wt / ".env").is_symlink()
    assert not (wt / "config" / "simfarm.example.toml").exists()
    assert not (wt / "config" / "simfarm.toml").exists()  # the R2 key stays with the loop


def test_worktree_drops_an_earlier_link_to_the_farm_config(tmp_path: Path) -> None:
    repo, wt = tmp_path / "repo", tmp_path / "wt"
    (repo / "config").mkdir(parents=True)
    (repo / "config" / "simfarm.toml").write_text("[r2]\n")
    (wt / "config").mkdir(parents=True)
    (wt / "config" / "simfarm.toml").symlink_to(repo / "config" / "simfarm.toml")
    loop.Worktree(repo, wt).link_local()
    assert not (wt / "config" / "simfarm.toml").exists()
    assert (repo / "config" / "simfarm.toml").is_file()


def test_leftover_half_made_rounds_are_removed_at_start(tmp_path: Path) -> None:
    f = farm(tmp_path, ["ok"], rounds=1)
    stale = f.rounds_dir / ".partial-old"
    (stale / "work").mkdir(parents=True)
    assert f.run() == 0
    assert not stale.exists()
    assert len(list(f.rounds_dir.iterdir())) == 1


def test_round_env_is_an_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SD_GROUND__SECRET", "x")
    monkeypatch.setenv("UV_CACHE_DIR", "/c")
    monkeypatch.setenv("HTTPS_PROXY", "http://p")
    env = loop.round_env(Path("/w"))
    assert "SD_GROUND__SECRET" not in env
    assert env["UV_CACHE_DIR"] == "/c" and env["HTTPS_PROXY"] == "http://p"
    assert "GZ_SIM_RESOURCE_PATH" in env and "PATH" in env


def test_worktree_refuses_a_folder_that_is_not_its_own(tmp_path: Path) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    inside = repo / "build" / "simfarm" / "main"
    (inside / ".git").mkdir(parents=True)  # broken: git would find the owner's checkout
    wt = loop.Worktree(repo, inside, run=lambda *a, **k: None)
    with pytest.raises(OSError, match="egen worktree"):
        wt.update()


def test_status_summary(tmp_path: Path) -> None:
    base = tmp_path
    now = dt.datetime(2026, 10, 18, 12, 0, tzinfo=dt.UTC)
    status.write_state(
        base / "state.json", {"pid": 1, "state": "paused", "reason": "GPU", "upload": "r2"}
    )
    for i, (st, hours_ago) in enumerate([("ok", 1), ("ok", 8), ("failed", 0.5)]):
        status.append_round(
            base / "rounds.jsonl",
            {
                "round_id": f"r{i}",
                "status": st,
                "error": None if st == "ok" else "svarta",
                "images": 10,
                "labels": 4,
                "finished_utc": (now - dt.timedelta(hours=hours_ago)).isoformat(),
                "grade": [
                    {"is_exam": True, "cost_sek": 2.0},
                    {"is_exam": True, "cost_sek": None},
                    None,
                ][i],
            },
        )
    (base / "uploaded.jsonl").write_text('{"round": "r1"}\n')
    s = status.summary(base, now, alive=lambda _pid: True)
    assert s["state"] == "paused" and s["reason"] == "GPU"
    assert (s["rounds_ok"], s["rounds_failed"], s["images"], s["labels"]) == (2, 1, 20, 8)
    assert s["rounds_per_hour"] == round(1 / status.WINDOW_H, 2)
    assert (s["exams"], s["exam_cost_sek"], s["exam_cost_unmeasured"]) == (2, 2.0, 1)
    assert s["not_uploaded"] == ["r0"] and s["storage"] == "R2 (EU)"
    text = status.render(s)
    assert "2.0 kr (minst; 1 prov utan uppmätt kostnad)" in text
    assert (
        "paused (GPU)" in text and "väntar på uppladdning: 1 varv" in text and "r2: svarta" in text
    )
    # A dead loop is stopped, whatever its last state said.
    assert status.summary(base, now, alive=lambda _pid: False)["state"] == "stopped"
    assert status.summary(tmp_path / "none", now)["state"] == "never run"
    assert Ledger(base / "uploaded.jsonl").rounds == {"r1"}
    assert farm_round.EXIT_NOT_BUILT == 2
