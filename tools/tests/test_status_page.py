"""tools/status_page/build.py: the status page from the simulator tests' result.json."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
from status_page import build as sp

ROOT = Path(__file__).resolve().parents[2]
SHA = "0f57504" + "0" * 33


def result(variant: str, passed: bool, *criteria: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return {
        "schema": 1,
        "scenario": kw.get("scenario", "SC02"),
        "variant": variant,
        "passed": passed,
        "started_utc": kw.get("started", "2026-10-18T10:33:26Z"),
        "versions": {"git_sha": SHA},
        "criteria": [*criteria, {"id": "T", "text": "Testet godkänt", "passed": passed,
                                 "measured": "passed" if passed else "call: failed"}],
    }  # fmt: skip


def brake(measured: float, passed: bool = True) -> dict[str, Any]:
    return {"id": "M1", "text": "BRAKE", "passed": passed, "measured": measured,
            "limit": 2.0, "unit": "s", "bound": "max"}  # fmt: skip


def write_results(folder: Path, *results: dict[str, Any]) -> Path:
    for i, res in enumerate(results):
        run = folder / f"sitl-results-sitl-{i}" / f"run-{i}"
        run.mkdir(parents=True)
        (run / "result.json").write_text(json.dumps(res), encoding="utf-8")
    return folder


def page(tmp_path: Path, name: str, results: Path, **extra: str) -> str:
    values = {"results": str(results), "out": str(tmp_path / name), "sha": SHA,
              "date": "2026-10-18 10:45 UTC",
              "run_url": "https://github.com/YXHFR/Searchdrone/actions/runs/1",
              **extra}  # fmt: skip
    return sp.build(values).read_text(encoding="utf-8")


def card(html: str, name: str) -> str:
    m = re.search(r'<li class="cap (\w+)"><div class="top"><span class="name">'
                  + re.escape(name) + r"</span>.*?</li>", html)  # fmt: skip
    assert m, name
    return m[0]


def test_same_input_gives_the_same_html(tmp_path: Path) -> None:
    results = write_results(tmp_path / "in", result("test_sc02_stop_and_resume", True, brake(0.4)))
    first = page(tmp_path, "a", results)
    second = page(tmp_path, "b", results)
    assert first == second
    assert (tmp_path / "a" / "history.json").read_bytes() == (
        tmp_path / "b" / "history.json"
    ).read_bytes()


def test_a_green_row_shows_expected_and_measured(tmp_path: Path) -> None:
    results = write_results(tmp_path / "in", result("test_sc02_stop_and_resume", True, brake(0.43)))
    html = page(tmp_path, "out", results)
    stop = card(html, "STOPP")
    assert 'class="cap pass"' in stop and "Grön" in stop
    assert "<dd>högst 2 s</dd>" in stop and "<dd>0,43 s</dd>" in stop
    assert "<dd>SC02</dd>" in stop and "2026-10-18 10:33 UTC" in stop and "0f57504" in stop


def test_a_red_capability_shows_red(tmp_path: Path) -> None:
    results = write_results(
        tmp_path / "in", result("test_sc02_stop_and_resume", False, brake(2.6, passed=False))
    )
    html = page(tmp_path, "out", results)
    stop = card(html, "STOPP")
    assert 'class="cap fail"' in stop and "✗ Röd" in stop
    assert "<dd>2,6 s</dd>" in stop
    assert '<li class="fail">2 röda</li>' in html  # both SC02 rows


def test_a_value_out_of_limit_is_red_even_if_the_test_passed(tmp_path: Path) -> None:
    results = write_results(
        tmp_path / "in", result("test_sc02_stop_and_resume", True, brake(2.6, passed=False))
    )
    assert 'class="cap fail"' in card(page(tmp_path, "out", results), "STOPP")


def test_one_red_round_of_several_makes_the_row_red(tmp_path: Path) -> None:
    results = write_results(
        tmp_path / "in",
        result("test_sc02_stop_and_resume", True, brake(0.4), started="2026-10-18T10:00:00Z"),
        result("test_sc02_stop_and_resume", False, started="2026-10-18T10:10:00Z"),
    )
    stop = card(page(tmp_path, "out", results), "STOPP")
    assert 'class="cap fail"' in stop and "1 av 2 körningar gröna." in stop
    assert "<dd>0,4 s</dd>" in stop  # the last value measured


def test_a_repeated_wait_counts_its_worst_value(tmp_path: Path) -> None:
    ok, late = brake(0.4), brake(2.6, passed=False)
    results = write_results(
        tmp_path / "in", result("test_sc02_stop_and_resume", True, late, ok, brake(0.3))
    )
    stop = card(page(tmp_path, "out", results), "STOPP")
    assert 'class="cap fail"' in stop and "<dd>2,6 s</dd>" in stop
    results = write_results(
        tmp_path / "in2", result("test_sc02_stop_and_resume", True, ok, brake(0.3))
    )
    assert "<dd>0,3 s</dd>" in card(page(tmp_path, "out2", results), "STOPP")


def test_outcome_rows_and_missing_results(tmp_path: Path) -> None:
    results = write_results(tmp_path / "in", result("test_sc18_api_guards", False, scenario="SC18"))
    html = page(tmp_path, "out", results)
    api = card(html, "API-skydd")
    assert 'class="cap fail"' in api
    assert "alla kontroller i testet godkända" in api and "call: failed" in api
    assert 'class="cap none"' in card(html, "FLYG HEM")


def test_sc00_reads_its_own_report(tmp_path: Path) -> None:
    healthy = {"id": "SC00.1", "text": "Stacken frisk (HEARTBEAT, EKF och GPS)", "passed": True,
               "measured": 41.9, "limit": 90.0, "unit": "s"}  # fmt: skip
    sc00 = result("default", True, healthy, scenario="SC00")
    plugin_record = result("test_sc00_smoke", True, scenario="SC00")
    html = page(tmp_path, "out", write_results(tmp_path / "in", sc00, plugin_record))
    row = card(html, "Simulatorn startar frisk")
    assert "<dd>högst 90 s</dd>" in row and "<dd>41,9 s</dd>" in row


def test_history_keeps_twenty_one_per_commit_and_drops_junk(tmp_path: Path) -> None:
    old = [
        {"sha": f"{i:07x}", "date": f"d{i}", "run_url": "", "status": {"STOPP": "pass"}}
        for i in range(1, 30)
    ]
    old.insert(0, {"sha": SHA, "date": "earlier", "run_url": "", "status": {"STOPP": "fail"}})
    old.append({"sha": "<script>", "date": "x", "status": {}})
    old.append({"sha": "abcdef0", "date": "<b>x</b>", "run_url": "javascript:x",
                "status": {"STOPP": "pass", "Okänd": "pass", "FLYG HEM": "grön"}})  # fmt: skip
    hist_file = tmp_path / "history.json"
    hist_file.write_text(json.dumps(old), encoding="utf-8")
    results = write_results(tmp_path / "in", result("test_sc02_stop_and_resume", True, brake(0.4)))
    html = page(tmp_path, "out", results, history=str(hist_file))
    hist = json.loads((tmp_path / "out" / "history.json").read_text(encoding="utf-8"))
    assert len(hist) == sp.HISTORY_LEN
    assert hist[0]["sha"] == SHA and hist[0]["status"]["STOPP"] == "pass"
    assert [e["sha"] for e in hist].count(SHA) == 1
    assert "<script>" not in html and "<b>x</b>" not in html
    cleaned = sp.clean_history(old, {"STOPP", "FLYG HEM"})
    junk = next(e for e in cleaned if e["sha"] == "abcdef0")
    assert junk["status"] == {"STOPP": "pass"} and junk["run_url"] == ""
    stop = card(html, "STOPP")
    assert stop.count("<span class=") == sp.HISTORY_LEN + 2  # name, pill and the cells


def test_film_is_offered_as_mp4_and_webm(tmp_path: Path) -> None:
    film = tmp_path / "film"
    film.mkdir()
    (film / "demo.mp4").write_bytes(b"mp4")
    (film / "demo.webm").write_bytes(b"webm")
    html = page(tmp_path, "out", write_results(tmp_path / "in"), film=str(film))
    assert (
        '<source src="demo.mp4" type="video/mp4"><source src="demo.webm" type="video/webm">' in html
    )
    assert "playsinline" in html and (tmp_path / "out" / "demo.webm").read_bytes() == b"webm"
    assert "Ingen film än." in page(tmp_path, "nofilm", write_results(tmp_path / "in2"))


def test_out_must_be_empty_and_arguments_are_checked(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "x").write_text("x")
    with pytest.raises(SystemExit, match="inte tom"):
        page(tmp_path, "out", write_results(tmp_path / "in"))
    with pytest.raises(SystemExit, match="okänd"):
        sp.parse(["results=a", "out=b", f"sha={SHA}", "date=d", "--x"])
    with pytest.raises(SystemExit, match="saknas"):
        sp.parse(["results=a"])
    with pytest.raises(SystemExit, match="SHA"):
        sp.parse(["results=a", "out=b", "sha=main", "date=d"])


def test_numbers_in_swedish() -> None:
    assert sp.number(0.432) == "0,43"
    assert sp.number(41.94) == "41,9"
    assert sp.number(240.0) == "240"
    assert sp.number(2.0) == "2"
    assert sp.with_unit(97.5, "%") == "97,5 %"


SOURCES = "\n".join(
    p.read_text(encoding="utf-8") for p in sorted((ROOT / "simulation" / "tests").rglob("*.py"))
)


@pytest.mark.parametrize("cap", sp.load_capabilities()[1], ids=lambda c: c.name)
def test_every_capability_names_a_real_test_and_measurement(cap: sp.Capability) -> None:
    test = "test_sc00_smoke" if cap.variant == "default" else cap.variant
    assert re.search(rf"^(async )?def {re.escape(test)}\(", SOURCES, re.M), test
    if cap.measure:
        assert f'"{cap.measure}"' in SOURCES, cap.measure


WAIT = ROOT / "tools" / "status_page" / "wait_for_sitl.sh"
REPO = "YXHFR/Searchdrone"


def sitl_run(i: int, **kw: Any) -> dict[str, Any]:
    run: dict[str, Any] = {"id": i, "html_url": f"https://github.com/{REPO}/actions/runs/{i}",
                           "head_sha": f"{i:040x}", "updated_at": "2026-10-18T10:45:31Z",
                           "status": "completed", "conclusion": "success", "event": "push",
                           "head_branch": "main",
                           "head_repository": {"full_name": REPO}}  # fmt: skip
    run.update(kw)
    return run


def wait_for(tmp_path: Path, event: str, *runs: dict[str, Any]) -> dict[str, str]:
    """wait_for_sitl.sh with a stand-in ``gh`` that answers with ``runs``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps({"workflow_runs": list(runs)}), encoding="utf-8")
    gh = bin_dir / "gh"
    gh.write_text(f'#!/bin/sh\necho "$@" >> {tmp_path}/calls\ncat {answer}\n', encoding="utf-8")
    gh.chmod(0o755)
    out = tmp_path / "out"
    out.write_text("", encoding="utf-8")
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "REPO": REPO,
           "EVENT": event, "SHA": "a" * 40, "GITHUB_OUTPUT": str(out), "WAIT_MIN": "0",
           "POLL_S": "0"}  # fmt: skip
    proc = subprocess.run(["bash", str(WAIT)], env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"exit": str(proc.returncode), "calls": (tmp_path / "calls").read_text()}
    lines = out.read_text(encoding="utf-8").splitlines()
    return dict(line.split("=", 1) for line in lines) | {"calls": (tmp_path / "calls").read_text()}


def test_nightly_page_never_shows_a_pull_requests_run(tmp_path: Path) -> None:
    got = wait_for(
        tmp_path,
        "schedule",
        sitl_run(9, event="pull_request", head_repository={"full_name": "evil/Searchdrone"}),
        sitl_run(8, event="pull_request"),
        sitl_run(7, event="push", head_repository={"full_name": "evil/Searchdrone"}),
        sitl_run(6, conclusion="cancelled"),
        sitl_run(5, event="schedule", head_repository={"full_name": "yxhfr/searchdrone"}),
        sitl_run(4),
    )
    assert got["run_id"] == "5"
    assert got["run_url"] == f"https://github.com/{REPO}/actions/runs/5"
    assert got["date"] == "2026-10-18 10:45 UTC"
    assert "branch=main&status=completed" in got["calls"]


def test_push_page_waits_for_its_own_commit(tmp_path: Path) -> None:
    got = wait_for(tmp_path, "push", sitl_run(3, conclusion="failure"))
    assert got["run_id"] == "3" and got["sha"] == f"{3:040x}"
    assert f"head_sha={'a' * 40}&event=push" in got["calls"]
    assert wait_for(tmp_path, "push", sitl_run(2, status="in_progress"))["exit"] == "1"
    assert wait_for(tmp_path, "push", sitl_run(2, conclusion="cancelled"))["exit"] == "1"
    assert wait_for(tmp_path, "release")["exit"] == "2"
