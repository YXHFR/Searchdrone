import re
from pathlib import Path

import pytest
import yaml

from sd_sim import paths, ports

ROOT = paths.REPO


def all_ports(instance: int) -> dict[str, int]:
    out = ports.for_instance(instance)
    # SITL also binds the JSON reply port right after its FDM port.
    out["sitl_json_fdm_reply"] = out["sitl_json_fdm"] + 1
    return out


def test_ports_unique() -> None:
    seen: dict[int, str] = {}
    for instance in range(ports.MAX_INSTANCE + 1):
        for name, port in all_ports(instance).items():
            key = f"{name}@i{instance}"
            other = seen.get(port)
            # Shared rows give the same port for every instance; that is the same socket.
            if other is not None and other.split("@")[0] != name:
                pytest.fail(f"port {port}: {other} and {key} collide")
            seen[port] = key


def test_known_bases() -> None:
    p = ports.for_instance(0)
    assert p["ground_panel"] == 8080
    assert p["ground_commander"] == 7700
    assert p["postgres_local"] == 5433
    assert p["llm_http"] == 8091
    assert p["ground_remote"] == 8081
    assert p["ground_remote"] != p["llm_http"]
    assert ports.for_instance(1)["harness_mavlink"] == 14574
    assert "qgc_mavlink" not in ports.for_instance(1)


def test_instance_range() -> None:
    with pytest.raises(ValueError):
        ports.for_instance(8)


def test_example_config_ports_are_listed() -> None:
    """Every port in a committed example config is a row of ports.yaml."""
    listed = set(ports.for_instance(0).values())
    for path in sorted((ROOT / "config").glob("*.example.toml")):
        text = path.read_text(encoding="utf-8")
        found = {int(m) for m in re.findall(r"(?:port\s*=\s*|127\.0\.0\.1:)(\d{4,5})", text)}
        assert found <= listed, f"{path.name}: {sorted(found - listed)} missing in ports.yaml"


def test_rows_have_no_unknown_fields(tmp_path: Path) -> None:
    bad = tmp_path / "ports.yaml"
    bad.write_text(yaml.safe_dump({"x": {"base": 1, "per_instance": True, "colour": "red"}}))
    with pytest.raises(ValueError):
        ports.load(bad)
