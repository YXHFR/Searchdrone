"""compose.yaml: no remote access in v0.1.0, and the db service stays local and pinned."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "compose.yaml"
DOCKER = shutil.which("docker")
# Dummy values so `docker compose config` can render without a real .env. Tickets that
# add services with required variables add them here.
DUMMY_ENV = {
    "SD_GROUND__DB_PASSWORD": "x",
    "SD_GROUND__VIDEO_PUBLISH_PASSWORD": "x",
    "SD_GROUND__VIDEO_API_PASSWORD": "x",
    "SD_LLM_API_KEY": "x",
    "SD_UID": "1000",
    "SD_GID": "1000",
    "SD_DATA_DIR": "/srv/sd-data",  # only rendered, never created
}


def _services() -> dict[str, Any]:
    data = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    return data["services"]


def test_no_remote_profile_or_tunnel_static() -> None:
    for name, svc in _services().items():
        assert "remote" not in svc.get("profiles", []), name
        assert "cloudflared" not in name, name
        assert "cloudflared" not in str(svc.get("image", "")), name


@pytest.mark.skipif(DOCKER is None, reason="docker is not installed")
def test_no_remote_profile_rendered(tmp_path: Path) -> None:
    env = tmp_path / "env"
    env.write_text("".join(f"{k}={v}\n" for k, v in DUMMY_ENV.items()), encoding="utf-8")
    assert DOCKER is not None
    profiles = subprocess.run(
        [DOCKER, "compose", "--env-file", str(env), "-f", str(COMPOSE), "config", "--profiles"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert profiles.returncode == 0, profiles.stderr
    assert "remote" not in profiles.stdout.split()
    images = subprocess.run(
        [
            DOCKER,
            "compose",
            "--env-file",
            str(env),
            "-f",
            str(COMPOSE),
            "--profile",
            "*",
            "config",
            "--images",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert images.returncode == 0, images.stderr
    assert "cloudflared" not in images.stdout


def test_db_service() -> None:
    db = _services()["db"]
    assert re.fullmatch(r"postgis/postgis:18-3\.6@sha256:[0-9a-f]{64}", db["image"])
    assert db["ports"] == ["127.0.0.1:5433:5432"]
    assert db["volumes"] == ["sd-pgdata:/var/lib/postgresql"]
    assert "network_mode" not in db
    # The password comes from .env and compose refuses to start without it.
    assert db["environment"]["POSTGRES_PASSWORD"].startswith("${SD_GROUND__DB_PASSWORD:?")


def test_env_example_db_passwords_empty() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    for key in ("SD_GROUND__DB_PASSWORD", "SD_GROUND__DB_APP_PASSWORD"):
        assert f"{key}=" in lines, key


def test_mediamtx_service() -> None:
    """B1-09: pinned image, signalling on loopback, the password required from .env."""
    mtx = _services()["mediamtx"]
    assert re.fullmatch(r"bluenviron/mediamtx:1\.21\.1@sha256:[0-9a-f]{64}", mtx["image"])
    assert mtx["network_mode"] == "host"
    assert "ports" not in mtx
    env = mtx["environment"]
    assert env["MTX_WEBRTCADDRESS"] == "127.0.0.1:8889"  # fixed, also in LAN mode
    assert env["MTX_HLSADDRESS"] == "127.0.0.1:8888"
    assert env["MTX_RTSPADDRESS"] == "127.0.0.1:8554"
    assert env["MTX_APIADDRESS"] == "127.0.0.1:9997"
    assert env["MTX_WEBRTCLOCALUDPADDRESS"] == "${SD_VIDEO_BIND:-127.0.0.1}:8189"
    assert env["MTX_AUTHINTERNALUSERS_0_PASS"].startswith("${SD_GROUND__VIDEO_PUBLISH_PASSWORD:?")
    assert env["MTX_AUTHINTERNALUSERS_2_PASS"].startswith("${SD_GROUND__VIDEO_API_PASSWORD:?")
    assert "./config/mediamtx.yml:/mediamtx.yml:ro" in mtx["volumes"]


def test_mediamtx_config_has_no_real_password() -> None:
    data = yaml.safe_load((ROOT / "config" / "mediamtx.yml").read_text(encoding="utf-8"))
    users = data["authInternalUsers"]
    assert [u.get("pass") for u in users] == ["CHANGE_ME", None, "CHANGE_ME"]
    # The user without a password may only read; the control API needs a password.
    assert [p["action"] for p in users[1]["permissions"]] == ["read"]
    assert [p["action"] for p in users[2]["permissions"]] == ["api"]


@pytest.mark.skipif(DOCKER is None, reason="docker is not installed")
def test_offline_overlay_renders_with_compose_yaml(tmp_path: Path) -> None:
    """compose.yaml + compose.offline.yaml: the database on sd-offline (172.31.250.0/24),
    still published on 127.0.0.1 only, and nothing else added."""
    env = tmp_path / "env"
    env.write_text("".join(f"{k}={v}\n" for k, v in DUMMY_ENV.items()), encoding="utf-8")
    assert DOCKER is not None
    res = subprocess.run(
        [
            DOCKER,
            "compose",
            "--env-file",
            str(env),
            "-f",
            str(COMPOSE),
            "-f",
            str(ROOT / "compose.offline.yaml"),
            "config",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    data = yaml.safe_load(res.stdout)
    defaults = {n for n, svc in _services().items() if not svc.get("profiles")}
    assert set(data["services"]) == defaults
    assert set(data["services"]["db"]["networks"]) == {"sd-offline"}
    assert all(p["host_ip"] == "127.0.0.1" for p in data["services"]["db"]["ports"])
    [subnet] = data["networks"]["sd-offline"]["ipam"]["config"]
    assert subnet["subnet"] == "172.31.250.0/24"
