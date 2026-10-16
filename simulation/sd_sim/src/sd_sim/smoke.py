"""Smoke flight: ``python -m sd_sim.smoke [physics=gz|builtin] [gui=true|false] [instance=N]``.

Starts SITL (with Gazebo for ``physics=gz``) with a standard vehicle, then flies
GUIDED, arm, take off to 20 m and land through the test rig's own MAVLink channel.
Writes ``runs/<run>/result.json`` and exits 0 only if every check passed within 3
minutes. While flying it checks that the simulator listens on loopback only, and with
``gui=false`` that the camera stream delivers frames.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from sd_sim import _args, config, gazebo, netcheck
from sd_sim.errors import InfraError
from sd_sim.harness import Harness
from sd_sim.launch import Stack
from sd_sim.sitl import CMAC_HOME
from sd_sim.util import new_run_dir, say

ALLOWED_KEYS = ("physics", "gui", "instance", "dry")
TAKEOFF_ALT_M = 20.0
ALT_TOLERANCE_M = 1.5
MAX_DURATION_S = 180.0
FRAME_WINDOW_S = 10.0
FRAME_SHARE = 0.9  # at least 90 % of the camera's nominal frames must arrive


def min_frames(vehicle: str) -> int:
    return int(FRAME_SHARE * gazebo.CAMERA_FPS[vehicle] * FRAME_WINDOW_S)


def harden(cfg: config.SimConfig) -> config.SimConfig:
    """Settings the smoke flight always uses, whatever local config or environment say:
    no QGroundControl path (it bypasses the gate), the public CMAC home, the standard
    vehicle, and the camera only in Gazebo server mode."""
    cfg.qgc = False
    cfg.home, cfg.personal_home = CMAC_HOME, False
    cfg.vehicle = "stock"
    cfg.video = "camera" if cfg.physics == "gz" and not cfg.gui else "off"
    return cfg


def check_listeners(stack: Stack) -> list[str]:
    if shutil.which("ss") is None:
        return ["ss saknas: kunde inte kontrollera portarna"]
    bad = netcheck.violations(
        netcheck.parse_ss(netcheck.run_ss()),
        stack.ports,
        netcheck.load_allowlist(),
        owned_pids=stack.group.owned_pids(),
    )
    return [str(item) for item in bad]


def count_frames(stack: Stack) -> int:
    def frames() -> int:
        flows: Any = stack.linksim_get("/state")["flows"]
        return int(flows["video"]["from_vehicle"]["rtp_frames"])

    gazebo.enable_camera(stack.cfg.instance)
    start = frames()
    time.sleep(FRAME_WINDOW_S)
    return frames() - start


def fly(harness: Harness, result: dict[str, Any], t0: float) -> None:
    harness.wait(lambda s: s.heartbeat, 60, "HEARTBEAT från flygkontrollen")
    harness.wait(lambda s: s.ready, 90, "position från EKF och GPS")
    result["ready_s"] = round(time.monotonic() - t0, 1)
    harness.set_mode("GUIDED")
    harness.arm()
    harness.takeoff(TAKEOFF_ALT_M)
    harness.wait(lambda s: s.rel_alt_m >= TAKEOFF_ALT_M - 0.5, 60, f"höjden {TAKEOFF_ALT_M} m")
    harness.pump(5.0)  # hover: the overshoot shows up in the maximum
    harness.set_mode("LAND")
    harness.wait(lambda s: not s.armed, 90, "landning och desarmering")


def run(cfg: config.SimConfig) -> tuple[int, Path]:
    cfg = harden(cfg)
    t0 = time.monotonic()
    run_dir = new_run_dir(cfg.instance, "smoke")
    stack = Stack(cfg, run_dir, observer=False)
    result: dict[str, Any] = {
        "schema": 1,
        "name": "sim-smoke",
        "physics": cfg.physics,
        "gui": cfg.gui,
        "instance": cfg.instance,
        "target_alt_m": TAKEOFF_ALT_M,
    }
    checks: dict[str, bool] = {}
    harness: Harness | None = None
    try:
        harness = Harness(stack.ports["harness_mavlink"])
        stack.up()
        fly_start = time.monotonic()
        fly(harness, result, t0)
        result["flight_s"] = round(time.monotonic() - fly_start, 1)
        result["max_alt_m"] = round(harness.state.max_rel_alt_m, 2)
        checks["altitude"] = abs(result["max_alt_m"] - TAKEOFF_ALT_M) <= ALT_TOLERANCE_M
        bad = check_listeners(stack)
        result["non_loopback_listeners"] = bad
        checks["loopback_only"] = not bad
        if cfg.physics == "gz" and not cfg.gui and stack.world is not None:
            frames = count_frames(stack)
            result["camera_frames_10s"] = frames
            result["camera_frames_min"] = min_frames(stack.world.vehicle)
            checks["camera"] = frames >= result["camera_frames_min"]
    except InfraError as exc:
        result["error"] = stack.mask(str(exc))
        checks["completed"] = False
    except Exception as exc:
        result["error"] = stack.mask(f"{type(exc).__name__}: {exc}")
        checks["completed"] = False
    finally:
        if harness is not None:
            harness.close()
        stack.down()
    result["duration_s"] = round(time.monotonic() - t0, 1)
    checks["duration"] = result["duration_s"] <= MAX_DURATION_S
    checks.setdefault("completed", True)
    result["checks"] = checks
    result["passed"] = all(checks.values())
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return (0 if result["passed"] else 1), run_dir


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    values = _args.parse(argv, ALLOWED_KEYS, "sd_sim.smoke")
    if values.pop("dry", "") == "1":
        say(json.dumps({"args": argv}))
        return 0
    values.setdefault("physics", "gz")
    try:
        cfg = config.load({**values, "video": "off", "qgc": "false"})
    except config.ConfigError as exc:
        _args.fail(f"sd_sim.smoke: {exc}")
    code, run_dir = run(cfg)
    summary = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    say(json.dumps(summary, indent=2, ensure_ascii=False))
    say("GODKÄNT" if code == 0 else "UNDERKÄNT")
    return code


if __name__ == "__main__":
    sys.exit(main())
