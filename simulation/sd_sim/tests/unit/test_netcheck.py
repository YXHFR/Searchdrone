from pathlib import Path

import pytest

from sd_sim import netcheck, ports

SS = """\
udp   UNCONN 0      0          127.0.0.1:14561      0.0.0.0:*    users:(("python3",pid=1,fd=8))
udp   UNCONN 0      0            0.0.0.0:5501       0.0.0.0:*    users:(("arducopter",pid=2,fd=11))
udp   UNCONN 0      0            0.0.0.0:14564      0.0.0.0:*    users:(("python3",pid=3,fd=3))
tcp   LISTEN 0      100        127.0.0.1:8790       0.0.0.0:*    users:(("python3",pid=1,fd=13))
tcp   LISTEN 0      4096               *:7601             *:*    users:(("python3",pid=4,fd=5))
tcp   LISTEN 0      4096           [::1]:8080          [::]:*    users:(("python3",pid=5,fd=5))
tcp   LISTEN 0      128          0.0.0.0:22         0.0.0.0:*
"""


def test_parse_ss() -> None:
    items = netcheck.parse_ss(SS)
    assert netcheck.Listener("udp", "0.0.0.0", 5501, "arducopter") in items  # noqa: S104
    assert netcheck.Listener("tcp", "[::1]", 8080, "python3") in items
    assert len(items) == 7


def test_violations_ignore_loopback_unknown_ports_and_allowlist() -> None:
    p = ports.for_instance(0)
    bad = netcheck.violations(netcheck.parse_ss(SS), p, {("sitl_rc_in", "udp")})
    assert {(b.port, b.proto) for b in bad} == {(14564, "udp"), (7601, "tcp")}


def test_allowlist_only_rc_in() -> None:
    assert netcheck.load_allowlist() == {("sitl_rc_in", "udp")}
    assert "sitl_rc_in" in ports.load()


def test_allowlist_needs_reason(tmp_path: Path) -> None:
    path = tmp_path / "allow.yaml"
    path.write_text("- port: ground_panel\n  proto: tcp\n  reason: ''\n")
    with pytest.raises(ValueError):
        netcheck.load_allowlist(path)
