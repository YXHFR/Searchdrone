"""The farm's R2 upload (signing, resumable, never twice), its config and its politeness."""

from __future__ import annotations

import datetime as dt
import hashlib
import http.server
import json
import os
import threading
import urllib.request
from collections.abc import Mapping
from pathlib import Path

import pytest

from sd_sim.farm import config as farm_config
from sd_sim.farm import store
from sd_sim.farm.host import Gate, GpuReading, Limits, gpu

ACCOUNT = "0123456789abcdef0123456789abcdef"
NOW = dt.datetime(2026, 10, 18, 12, 0, 0, tzinfo=dt.UTC)


def test_sigv4_matches_the_aws_test_suite() -> None:
    # AWS Signature Version 4 test suite, "get-vanilla" (its published example key).
    headers = store.sigv4_headers(
        "GET",
        "https://example.amazonaws.com/",
        {},
        store.EMPTY_SHA256,
        "AKIDEXAMPLE",
        "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        dt.datetime(2015, 8, 30, 12, 36, 0, tzinfo=dt.UTC),
        region="us-east-1",
        service="service",
        sign_content_sha=False,
    )
    assert headers["authorization"] == (
        "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, "
        "SignedHeaders=host;x-amz-date, "
        "Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31"
    )


def cfg() -> store.R2Config:
    return store.R2Config(ACCOUNT, "sd-simfarm", "AK", "SECRET-VALUE")


def test_r2_config_uses_the_eu_endpoint_and_hides_the_secret() -> None:
    c = cfg()
    assert c.endpoint == f"https://{ACCOUNT}.eu.r2.cloudflarestorage.com"
    assert "SECRET" not in repr(c)
    assert store.R2Config.from_table({}) is None
    assert store.R2Config.from_table({"prefix": "x"}) is None
    with pytest.raises(ValueError):
        store.R2Config("not-an-account", "b-1", "a", "s")
    with pytest.raises(ValueError):
        store.R2Config(ACCOUNT, "Bad_Bucket", "a", "s")
    with pytest.raises(ValueError):
        store.R2Config(ACCOUNT, "bucket", "", "s")
    with pytest.raises(ValueError):
        store.R2Config(ACCOUNT, "bucket", "a", "s", prefix="../up")


class FakeBucket:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.calls: list[tuple[str, str]] = []
        self.fail_put_after: int | None = None

    def send(self, req: urllib.request.Request) -> tuple[int, Mapping[str, str]]:
        url = req.full_url
        assert url.startswith(f"https://{ACCOUNT}.eu.r2.cloudflarestorage.com/sd-simfarm/")
        key = url.split("/sd-simfarm/", 1)[1]
        method = req.get_method()
        self.calls.append((method, key))
        assert req.get_header("Authorization", "").startswith("AWS4-HMAC-SHA256 ")
        if method == "HEAD":
            if key in self.objects:
                return 200, {"x-amz-meta-sha256": self.objects[key][1]}
            return 404, {}
        assert method == "PUT"
        puts = sum(1 for m, _ in self.calls if m == "PUT")
        if self.fail_put_after is not None and puts > self.fail_put_after:
            return 500, {}
        body = req.data
        assert isinstance(body, bytes)
        sha = req.get_header("X-amz-meta-sha256", "")
        assert sha == hashlib.sha256(body).hexdigest()
        assert req.get_header("X-amz-content-sha256") == sha
        self.objects[key] = (body, sha)
        return 200, {}


def make_round(base: Path, name: str = "r1") -> Path:
    d = base / name
    for rel, text in {
        "round.json": "{}",
        "coco.json": "{}",
        "images/a.png": "png-a",
        "labels/a.txt": "0 0.5 0.5 0.1 0.1\n",
        "meta/a.json": "{}",
        "work/stills/0.ppm": "raw",
        ".round.json.tmp": "half",
    }.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)
    return d


def test_upload_is_whole_and_never_twice(tmp_path: Path) -> None:
    d = make_round(tmp_path / "rounds")
    bucket = FakeBucket()
    ledger_path = tmp_path / "uploaded.jsonl"
    s = store.R2Store(cfg(), store.Ledger(ledger_path), bucket.send, lambda: NOW)
    assert s.upload_round(d)
    keys = sorted(bucket.objects)
    assert keys == sorted(
        f"simfarm/v1/rounds/r1/{p}"
        for p in (
            "round.json",
            "coco.json",
            "images/a.png",
            "labels/a.txt",
            "meta/a.json",
            "manifest.json",
        )
    )
    puts = [k for m, k in bucket.calls if m == "PUT"]
    assert puts[-1].endswith("manifest.json")  # the manifest goes last
    manifest = json.loads(bucket.objects["simfarm/v1/rounds/r1/manifest.json"][0])
    assert len(manifest["files"]) == 5
    # Again: nothing is sent at all.
    bucket.calls.clear()
    assert not store.R2Store(
        cfg(), store.Ledger(ledger_path), bucket.send, lambda: NOW
    ).upload_round(d)
    assert bucket.calls == []
    # The ledger lost: the bucket's hashes are checked and nothing is sent again.
    ledger_path.unlink()
    bucket.calls.clear()
    assert store.R2Store(cfg(), store.Ledger(ledger_path), bucket.send, lambda: NOW).upload_round(d)
    assert [m for m, _ in bucket.calls] == ["HEAD"] * 6


def test_upload_resumes_after_an_interruption(tmp_path: Path) -> None:
    d = make_round(tmp_path / "rounds")
    bucket = FakeBucket()
    bucket.fail_put_after = 2
    ledger = tmp_path / "uploaded.jsonl"
    with pytest.raises(store.UploadError):
        store.R2Store(cfg(), store.Ledger(ledger), bucket.send, lambda: NOW).upload_round(d)
    assert not any(k.endswith("manifest.json") for k in bucket.objects)
    bucket.fail_put_after = None
    bucket.calls.clear()
    s = store.R2Store(cfg(), store.Ledger(ledger), bucket.send, lambda: NOW)
    assert s.upload_round(d)
    assert s.puts == 6 - 2  # only what was missing, plus the manifest
    # A changed file with the same name is sent again (its hash differs).
    (d / "coco.json").write_text('{"x": 1}')
    s2 = store.R2Store(cfg(), store.Ledger(ledger), bucket.send, lambda: NOW)
    s2.ledger.rounds.discard("r1")
    assert s2.upload_round(d) and s2.puts == 2  # coco.json and the new manifest


def test_upload_skips_symlinks_and_signs_encoded_keys(tmp_path: Path) -> None:
    d = make_round(tmp_path / "rounds")
    secret = tmp_path / "secret.txt"
    secret.write_text("key")
    (d / "images" / "leak.png").symlink_to(secret)
    (d / "linked").symlink_to(tmp_path, target_is_directory=True)
    assert all(not p.is_symlink() for p in store.files_of(d))
    assert "leak.png" not in {p.name for p in store.files_of(d)}
    assert not any("linked" in p.parts for p in store.files_of(d))
    os.link(secret, d / "images" / "hard.png")
    assert "hard.png" not in {p.name for p in store.files_of(d)}
    # A key that needs escaping is signed as sent, not escaped twice.
    url = "https://h.example/b/a%20b.png"
    h = store.sigv4_headers("PUT", url, {}, store.EMPTY_SHA256, "A", "S", NOW)
    h2 = store.sigv4_headers(
        "PUT", "https://h.example/b/a%2520b.png", {}, store.EMPTY_SHA256, "A", "S", NOW
    )
    assert h["authorization"] != h2["authorization"]


def test_send_refuses_a_redirect() -> None:
    hits: list[str] = []

    class Redirect(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self) -> None:
            hits.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Redirect)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/b/k"
        req = urllib.request.Request(url, method="HEAD", headers={"Authorization": "x"})
        with pytest.raises(store.UploadError):
            store._send(req)
    finally:
        server.shutdown()
    assert hits == ["/b/k"]  # the signed request never went on


def test_ledger_survives_a_cut_line(tmp_path: Path) -> None:
    path = tmp_path / "uploaded.jsonl"
    path.write_text('{"key": "a", "sha256": "1"}\n{"round": "r1"}\n{"key": "b", "sh')
    ledger = store.Ledger(path)
    assert ledger.objects == {"a": "1"} and ledger.rounds == {"r1"}


def test_config_example_is_local_only(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[4]
    c = farm_config.load(repo=repo) if not (repo / farm_config.LOCAL).exists() else None
    example = farm_config.load(repo=tmp_path)  # no files at all: the defaults
    assert example.r2 is None and example.commander == "script"
    if c is not None:
        assert c.r2 is None
    text = (repo / farm_config.EXAMPLE).read_text()
    assert 'secret_access_key = ""' in text and 'access_key_id = ""' in text


def test_config_with_a_key(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    local = tmp_path / farm_config.LOCAL
    local.write_text(
        f'[farm]\ncommander = "sd-brain"\nidle_only = true\nmax_duty = 0.5\n'
        f'[r2]\naccount_id = "{ACCOUNT}"\nbucket = "sd-simfarm"\n'
        'access_key_id = "AK"\nsecret_access_key = "SK"\n'
    )
    local.chmod(0o644)
    with pytest.raises(farm_config.FarmConfigError, match="chmod 600"):
        farm_config.load(repo=tmp_path)
    local.chmod(0o600)
    c = farm_config.load({"commander": "script"}, tmp_path)
    assert c.commander == "script" and c.limits.idle_only and c.limits.max_duty == 0.5
    assert c.r2 is not None and c.r2.bucket == "sd-simfarm"
    with pytest.raises(farm_config.FarmConfigError):
        farm_config.load({"commander": "pilot"}, tmp_path)
    with pytest.raises(farm_config.FarmConfigError):
        farm_config.load({"max_duty": "2"}, tmp_path)
    for bad in ("x", "0"):
        with pytest.raises(farm_config.FarmConfigError):
            farm_config.load({"max_failures": bad}, tmp_path)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def gate(tmp_path: Path, **kw: object) -> tuple[Gate, Clock, dict[str, object]]:
    clock = Clock()
    now: dict[str, object] = {"gpu": GpuReading(3, 50, 100, 8000), "cpu": 0.1, "idle": None}
    g = Gate(
        Limits(**kw),  # type: ignore[arg-type]
        tmp_path,
        clock=clock,
        read_gpu=lambda: now["gpu"],  # type: ignore[return-value]
        read_cpu=lambda: now["cpu"],  # type: ignore[return-value]
        read_idle=lambda: now["idle"],  # type: ignore[return-value]
    )
    return g, clock, now


def test_gate_pauses_for_games_heat_and_load(tmp_path: Path) -> None:
    g, _, now = gate(tmp_path)
    assert g.reason_to_wait() is None
    now["gpu"] = GpuReading(95, 60, 7000, 8000)
    assert "GPU:n används" in (g.reason_to_wait() or "")
    now["gpu"] = GpuReading(5, 88, 7000, 8000)
    assert "varm" in (g.reason_to_wait() or "")
    now["gpu"], now["cpu"] = None, 0.95  # no nvidia-smi: the CPU still counts
    assert "processorn" in (g.reason_to_wait() or "")


def test_gate_keeps_the_duty_cycle(tmp_path: Path) -> None:
    g, clock, _ = gate(tmp_path, max_duty=0.5)
    assert g.reason_to_wait() is None
    clock.t = 100.0
    g.round_took(90.0)
    assert "vilar" in (g.reason_to_wait() or "")
    clock.t = 200.0
    assert g.reason_to_wait() is None


def test_gate_idle_only(tmp_path: Path) -> None:
    g, _, now = gate(tmp_path, idle_only=True, idle_minutes=10)
    now["idle"] = 60.0
    assert "ledig" in (g.reason_to_wait() or "")
    now["idle"] = 3600.0
    assert g.reason_to_wait() is None
    # No way to ask Windows: a quiet GPU and CPU for ten minutes count as idle.
    g2, clock2, now2 = gate(tmp_path, idle_only=True, idle_minutes=10)
    assert "ledig" in (g2.reason_to_wait() or "")
    clock2.t = 601.0
    assert g2.reason_to_wait() is None
    now2["cpu"] = 0.5
    clock2.t = 700.0
    assert "ledig" in (g2.reason_to_wait() or "")


def test_gate_disk_budget(tmp_path: Path) -> None:
    g, _, _ = gate(tmp_path, max_local_gb=1e-6, min_free_gb=0.0)
    assert g.disk_ok() is None
    (tmp_path / "big").write_bytes(b"x" * 5000)
    assert "tak" in (g.disk_ok() or "")


def test_gpu_reading_parses_nvidia_smi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "nvidia-smi"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    got = gpu(lambda argv: "37, 61, 2048, 8192\n")
    assert got == GpuReading(37.0, 61.0, 2048.0, 8192.0)
    assert gpu(lambda argv: "garbage") is None


def test_upload_refuses_a_link(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    real.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    assert store.read_nofollow(real) == b"{}"
    with pytest.raises(OSError):
        store.read_nofollow(link)
