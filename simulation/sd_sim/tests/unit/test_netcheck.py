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
    assert netcheck.Listener("udp", "0.0.0.0", 5501, "arducopter", 2) in items  # noqa: S104
    assert netcheck.Listener("tcp", "[::1]", 8080, "python3", 5) in items
    assert len(items) == 7


def test_violations_ignore_loopback_unknown_ports_and_allowlist() -> None:
    p = ports.for_instance(0)
    bad = netcheck.violations(netcheck.parse_ss(SS), p, {("sitl_rc_in", "udp")})
    assert {(b.port, b.proto) for b in bad} == {(14564, "udp"), (7601, "tcp")}


def test_allowlist_is_short_and_known() -> None:
    allowed = netcheck.load_allowlist()
    assert allowed == {
        ("sitl_rc_in", "udp"),
        ("gz_discovery_msg", "udp"),
        ("gz_discovery_srv", "udp"),
        ("ephemeral:arducopter", "udp"),
        ("ephemeral:ruby", "udp"),
    }
    named = {name for name, _ in allowed if not name.startswith("ephemeral:")}
    assert named <= set(ports.load())


def test_ephemeral_rule_is_narrow() -> None:
    allow = netcheck.load_allowlist()
    p = ports.for_instance(0)
    low, _ = netcheck.ephemeral_range()  # this machine's range (WSL mirrored mode narrows it)
    sitl = netcheck.Listener("udp", "0.0.0.0", low + 1, "arducopter", 9)  # noqa: S104
    other = netcheck.Listener("udp", "0.0.0.0", low + 2, "python3", 9)  # noqa: S104
    tcp = netcheck.Listener("tcp", "0.0.0.0", low + 3, "arducopter", 9)  # noqa: S104
    on_known = netcheck.Listener("udp", "0.0.0.0", p["harness_mavlink"], "arducopter", 9)  # noqa: S104
    fixed = netcheck.Listener("udp", "0.0.0.0", 1234, "arducopter", 9)  # noqa: S104
    bad = netcheck.violations([sitl, other, tcp, on_known, fixed], p, allow, owned_pids={9})
    assert bad == [other, tcp, on_known, fixed]  # a fixed low port is never "ephemeral"


def test_ephemeral_needs_process(tmp_path: Path) -> None:
    path = tmp_path / "allow.yaml"
    path.write_text("- port: ephemeral\n  proto: udp\n  reason: x\n")
    with pytest.raises(ValueError):
        netcheck.load_allowlist(path)


def test_allowlist_needs_reason(tmp_path: Path) -> None:
    path = tmp_path / "allow.yaml"
    path.write_text("- port: ground_panel\n  proto: tcp\n  reason: ''\n")
    with pytest.raises(ValueError):
        netcheck.load_allowlist(path)
