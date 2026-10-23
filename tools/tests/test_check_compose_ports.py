"""tools/check_compose_ports.py: one test per rule."""

from __future__ import annotations

import subprocess
from pathlib import Path

import check_compose_ports as ccp
import pytest

DB_OK = """services:
  db:
    image: postgis/postgis:18-3.6
    ports:
      - "127.0.0.1:5433:5432"
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """An empty git repository: the check looks only at the files git would take."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


@pytest.mark.parametrize(
    "port",
    ['"5433:5432"', '"0.0.0.0:5433:5432"', "5432", '"[::]:5433:5432"', '"127.0.0.1::5432"'],
)
def test_port_outside_localhost_fails(port: str) -> None:
    text = f"services:\n  db:\n    ports:\n      - {port}\n"
    assert any("port:" in e for e in ccp.check_compose(text))


def test_localhost_short_and_long_syntax_pass() -> None:
    long_syntax = """services:
  db:
    ports:
      - target: 5432
        published: "5433"
        host_ip: 127.0.0.1
"""
    assert ccp.check_compose(DB_OK) == []
    assert ccp.check_compose(long_syntax) == []


def test_long_syntax_without_host_ip_fails() -> None:
    text = "services:\n  db:\n    ports:\n      - target: 5432\n        published: '5433'\n"
    assert ccp.check_compose(text)


HOST_NET = """services:
  mediamtx:
    image: bluenviron/mediamtx
    network_mode: host{comment}
    environment:
      MTX_RTSPADDRESS: "{rtsp}"
      MTX_WEBRTCLOCALUDPADDRESS: "${{SD_VIDEO_BIND:-127.0.0.1}}:8189"
"""


def test_host_network_with_reason_and_local_addresses_passes() -> None:
    text = HOST_NET.format(comment="  # host-network: WebRTC needs UDP", rtsp="127.0.0.1:8554")
    assert ccp.check_compose(text) == []


def test_host_network_reason_on_line_above_passes() -> None:
    text = HOST_NET.format(comment="", rtsp="127.0.0.1:8554").replace(
        "    network_mode: host", "    # host-network: WebRTC needs UDP\n    network_mode: host"
    )
    assert ccp.check_compose(text) == []


def test_host_network_without_reason_fails() -> None:
    errors = ccp.check_compose(HOST_NET.format(comment="", rtsp="127.0.0.1:8554"))
    assert any("host-network" in e and "saknar" in e for e in errors)


def test_host_network_address_on_all_interfaces_fails() -> None:
    text = HOST_NET.format(comment="  # host-network: x", rtsp=":8554")
    assert any("MTX_RTSPADDRESS" in e for e in ccp.check_compose(text))


def test_environment_as_list() -> None:
    text = """services:
  m:
    network_mode: host  # host-network: x
    environment:
      - MTX_APIADDRESS=0.0.0.0:9997
"""
    assert any("MTX_APIADDRESS" in e for e in ccp.check_compose(text))


MTX_OK = """rtspAddress: 127.0.0.1:8554
webrtcAddress: 127.0.0.1:8889
apiAddress: ""
rtmp: no
srt: no
moq: no
authInternalUsers:
  - user: publisher
    pass: CHANGE_ME
"""
COMPOSE_WITH_PASS = """services:
  mediamtx:
    ports: ["127.0.0.1:8554:8554"]
    environment:
      MTX_AUTHINTERNALUSERS_0_PASS: ${SD_GROUND__VIDEO_PUBLISH_PASSWORD:?set in .env}
"""


def test_mediamtx_ok_with_compose_override() -> None:
    assert ccp.check_mediamtx(MTX_OK, COMPOSE_WITH_PASS) == []


def test_mediamtx_change_me_without_override_fails() -> None:
    assert any("change-me" in e for e in ccp.check_mediamtx(MTX_OK, DB_OK))


def test_mediamtx_change_me_outside_pass_fails() -> None:
    text = MTX_OK.replace("user: publisher", "user: CHANGE_ME")
    assert any("change-me" in e for e in ccp.check_mediamtx(text, COMPOSE_WITH_PASS))


def test_mediamtx_public_address_fails() -> None:
    text = MTX_OK.replace("rtspAddress: 127.0.0.1:8554", "rtspAddress: :8554")
    assert any("rtspAddress" in e for e in ccp.check_mediamtx(text, COMPOSE_WITH_PASS))


@pytest.mark.parametrize("key", ["rtmp", "srt", "moq"])
def test_mediamtx_rtmp_and_srt_must_be_off(key: str) -> None:
    text = MTX_OK.replace(f"{key}: no", f"{key}: yes")
    assert any(f"{key} ska vara no" in e for e in ccp.check_mediamtx(text, COMPOSE_WITH_PASS))


def test_main_without_files_passes(repo: Path) -> None:
    assert ccp.main(["--root", str(repo)]) == 0


def test_main_reports_findings(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (repo / "compose.yaml").write_text(
        "services:\n  db:\n    ports: ['5433:5432']\n", encoding="utf-8"
    )
    assert ccp.main(["--root", str(repo)]) == 1
    assert "compose.yaml: port:" in capsys.readouterr().out


def test_compose_file_outside_the_root_fails(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "database").mkdir()
    (repo / "database" / "compose.yaml").write_text(DB_OK, encoding="utf-8")
    assert ccp.main(["--root", str(repo)]) == 1
    assert "database/compose.yaml: location:" in capsys.readouterr().out


def test_offline_overlay_is_allowed_and_checked(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "compose.offline.yaml").write_text(
        "services:\n  db:\n    networks: [sd-offline]\n", encoding="utf-8"
    )
    assert ccp.main(["--root", str(repo)]) == 0
    (repo / "compose.offline.yaml").write_text(
        "services:\n  db:\n    ports: ['5433:5432']\n", encoding="utf-8"
    )
    assert ccp.main(["--root", str(repo)]) == 1
    assert "compose.offline.yaml:" in capsys.readouterr().out
    (repo / "sub").mkdir()
    (repo / "sub" / "compose.offline.yaml").write_text(DB_OK, encoding="utf-8")
    assert ccp.main(["--root", str(repo)]) == 1


def test_compose_files_git_ignores_are_no_strays(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A generated compose file in an ignored folder (``just sim-farm`` writes
    build/simfarm/<name>/compose.yaml) is not the repository's: only a file git would
    take counts, tracked or new."""
    (repo / ".gitignore").write_text("build/\n", encoding="utf-8")
    (repo / "build" / "simfarm" / "main").mkdir(parents=True)
    (repo / "build" / "simfarm" / "main" / "compose.yaml").write_text(
        "services:\n  db:\n    ports: ['5433:5432']\n", encoding="utf-8"
    )
    assert ccp.main(["--root", str(repo)]) == 0
    (repo / "sim").mkdir()
    (repo / "sim" / "compose.yaml").write_text(DB_OK, encoding="utf-8")  # new, not ignored
    assert ccp.main(["--root", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "sim/compose.yaml: location:" in out and "build/" not in out


def test_mediamtx_moq_missing_fails() -> None:
    """MediaMTX turns Media over QUIC on when the key is left out."""
    text = MTX_OK.replace("moq: no\n", "")
    assert any("moq ska vara no" in e for e in ccp.check_mediamtx(text, COMPOSE_WITH_PASS))


def test_repository_files_pass() -> None:
    root = Path(__file__).resolve().parents[2]
    assert ccp.main(["--root", str(root)]) == 0


def test_mediamtx_passwordless_user_may_only_read() -> None:
    text = MTX_OK + "  - user: any\n    pass:\n    permissions: [{action: read}, {action: api}]\n"
    errors = ccp.check_mediamtx(text, COMPOSE_WITH_PASS)
    assert any("no-password" in e and "api" in e for e in errors)
    reader = MTX_OK + "  - user: any\n    pass:\n    permissions: [{action: read}]\n"
    assert ccp.check_mediamtx(reader, COMPOSE_WITH_PASS) == []


def test_mediamtx_each_change_me_needs_its_own_override() -> None:
    text = MTX_OK + "  - user: reader\n    pass:\n    permissions: [{action: read}]\n"
    text += "  - user: ground\n    pass: CHANGE_ME\n    permissions: [{action: api}]\n"
    errors = ccp.check_mediamtx(text, COMPOSE_WITH_PASS)
    assert any("ground" in e and "MTX_AUTHINTERNALUSERS_2_PASS" in e for e in errors)
    both = COMPOSE_WITH_PASS + "      MTX_AUTHINTERNALUSERS_2_PASS: ${SD_X:?set}\n"
    assert ccp.check_mediamtx(text, both) == []


@pytest.mark.parametrize(
    ("key", "ok"),
    [("MTX_WEBRTCLOCALUDPADDRESS", True), ("MTX_WEBRTCADDRESS", False), ("MTX_APIADDRESS", False)],
)
def test_only_the_media_port_may_follow_the_lan_bind(key: str, ok: bool) -> None:
    text = f"""services:
  mediamtx:
    # host-network: test
    network_mode: host
    environment:
      {key}: "${{SD_VIDEO_BIND:-127.0.0.1}}:8189"
"""
    assert (ccp.check_compose(text) == []) is ok
