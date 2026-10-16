"""``python -m sd_hwcalc check <drone.yaml> --limits <limits.yaml>``.

Writes a JSON report to stdout and a Swedish summary to stderr. Exit 0 when every limit
holds, 1 when a limit is broken or a file is missing or invalid (fail closed), 2 on a
usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from sd_hwcalc.calc import Report, check
from sd_hwcalc.model import load_drone, load_limits


def report_json(report: Report) -> dict[str, object]:
    p = report.performance
    return {
        "total_mass_kg": round(p.total_mass_kg, 4),
        "hover_power_w": round(p.hover_power_w, 2),
        "usable_energy_wh": round(p.usable_energy_wh, 2),
        "hover_time_min": round(p.hover_time_min, 2),
        "hover_thrust_fraction": round(p.hover_thrust_fraction, 4),
        "thrust_to_weight": round(p.thrust_to_weight, 3),
        "pass": report.passed,
        "failures": [
            {"limit": f.limit, "value": round(f.value, 4), "bound": f.bound, "message": f.message}
            for f in report.failures
        ],
    }


def summary(report: Report) -> str:
    p = report.performance
    lines = [
        f"Total massa:     {p.total_mass_kg:.3f} kg",
        f"Svävtid:         {p.hover_time_min:.1f} min",
        f"Svävgas:         {p.hover_thrust_fraction:.0%} av maxdragkraften",
        f"Dragkraft/vikt:  {p.thrust_to_weight:.2f}",
        f"Sväveffekt:      {p.hover_power_w:.0f} W, användbar energi {p.usable_energy_wh:.0f} Wh",
    ]
    if report.passed:
        lines.append("Alla hårdvarugränser håller.")
    else:
        lines.extend(f"FEL: {f.message}" for f in report.failures)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sd_hwcalc")
    sub = parser.add_subparsers(dest="command", required=True)
    check_cmd = sub.add_parser("check", help="check a drone file against the limits")
    check_cmd.add_argument("drone", type=Path)
    check_cmd.add_argument("--limits", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        drone = load_drone(args.drone)
        limits = load_limits(args.limits)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        sys.stderr.write(f"Kan inte läsa hårdvarufilerna: {exc}\n")
        sys.stdout.write(json.dumps({"pass": False, "error": type(exc).__name__}) + "\n")
        return 1

    report = check(drone, limits)
    sys.stdout.write(json.dumps(report_json(report), ensure_ascii=False, indent=2) + "\n")
    sys.stderr.write(summary(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
