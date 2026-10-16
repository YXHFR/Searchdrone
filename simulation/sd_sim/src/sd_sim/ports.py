"""Simulator ports from ``simulation/ports.yaml`` (the only source of simulator ports)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from sd_sim import paths

MAX_INSTANCE = 7


@dataclass(frozen=True)
class PortSpec:
    name: str
    base: int
    per_instance: bool
    only_instance: int | None = None
    owner: str | None = None
    env: str | None = None

    def port(self, instance: int) -> int | None:
        """The port for ``instance``, or None when the row does not exist for it."""
        check_instance(instance)
        if self.only_instance is not None and instance != self.only_instance:
            return None
        return self.base + 10 * instance if self.per_instance else self.base


def check_instance(instance: int) -> None:
    if not 0 <= instance <= MAX_INSTANCE:
        raise ValueError(f"instans måste vara 0-{MAX_INSTANCE}, inte {instance}")


def load(path: Path = paths.PORTS_YAML) -> dict[str, PortSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: väntade en tabell")
    specs: dict[str, PortSpec] = {}
    for name, row in raw.items():
        if not isinstance(row, dict) or set(row) - {
            "base",
            "per_instance",
            "only_instance",
            "owner",
            "env",
        }:
            raise ValueError(f"{path}: raden {name} har okända fält")
        specs[str(name)] = PortSpec(
            name=str(name),
            base=int(row["base"]),
            per_instance=bool(row["per_instance"]),
            only_instance=row.get("only_instance"),
            owner=row.get("owner"),
            env=row.get("env"),
        )
    return specs


def for_instance(instance: int, specs: dict[str, PortSpec] | None = None) -> dict[str, int]:
    """Every port that exists for ``instance``, by name."""
    specs = load() if specs is None else specs
    out: dict[str, int] = {}
    for name, spec in specs.items():
        port = spec.port(instance)
        if port is not None:
            out[name] = port
    return out
