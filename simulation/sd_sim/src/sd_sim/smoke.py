"""Smoke flight: ``python -m sd_sim.smoke [physics=gz|builtin] [gui=true|false] [instance=N]
[world=<site world>] [vehicle=fpv7]``.

Starts SITL (with Gazebo for ``physics=gz``) with a standard vehicle at CMAC, or with
``world=`` the Searchdrone vehicle in a world built from a public site (at the site's
home, on its terrain), then flies
GUIDED, arm, take off to 20 m and land through the test rig's own MAVLink channel.
Writes ``runs/<run>/result.json`` and exits 0 only if every check passed within 3
minutes. While flying it checks that the simulator listens on loopback only, and with
``gui=false`` that the camera stream delivers frames.

``vehicle=fpv7`` flies the 7-inch ``sd_fpv7`` instead: at CMAC in ``sd_lake_forest``, or
in the site's world with ``world=``. The battery current while hovering is recorded as
``hover_current_a`` where the physics reports one (builtin physics).
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sd_sim import _args, config, gazebo, netcheck, site, world
from sd_sim.errors import InfraError, VehicleError
from sd_sim.harness import Harness
from sd_sim.launch import Stack
from sd_sim.sitl import CMAC_HOME
from sd_sim.util import new_run_dir, say

ALLOWED_KEYS = ("physics", "gui", "instance", "dry", "world", "vehicle", "gi")
TAKEOFF_ALT_M = 20.0
ALT_TOLERANCE_M = 1.5
MAX_DURATION_S = 180.0
FRAME_WINDOW_S = 10.0
# Listeners, camera frames and real-time factor while hovering: 10 s of frames plus a few
# gz calls; anything longer means a call hangs.
MEASURE_LIMIT_S = 60.0
# Gazebo must keep at least half of real time while hovering with the camera on.
MIN_REAL_TIME_FACTOR = 0.5
FRAME_SHARE = 0.9  # at least 90 % of the camera's nominal frames must arrive
CURRENT_WINDOW_S = 3.0


def min_frames(vehicle: str, sim_seconds: float = FRAME_WINDOW_S) -> int:
    """Frames the camera must deliver over ``sim_seconds`` of simulated time: the camera's
    rate is in simulated time, so a Gazebo running slower than real time sends fewer per
    wall-clock second without anything being wrong."""
    return int(FRAME_SHARE * gazebo.CAMERA_FPS[vehicle] * sim_seconds)


def harden(cfg: config.SimConfig, vehicle: str = "") -> config.SimConfig:
    """Settings the smoke flight always uses, whatever local config or environment say:
    no QGroundControl path (it bypasses the gate), a public home (CMAC with the standard
    vehicle, or a public site's home with sd_quad14 in that site's world), and the camera
    only in Gazebo server mode. ``vehicle`` (the smoke flight's own key, never the local
    config's) picks sd_fpv7 instead: in the site's world, else in sd_lake_forest at CMAC."""
    cfg.qgc = False
    cfg.objects = ""
    fpv7 = config.VEHICLE_ALIASES.get(vehicle, vehicle) == "sd_fpv7"
    name = site.world_sites().get(cfg.world)
    if name is not None and site.folder_for(name)[1]:  # public sites only
        cfg.home, cfg.personal_home = site.load(name).home_arg, False
        cfg.vehicle = "sd_fpv7" if fpv7 else "sd_quad14"
    elif fpv7:
        cfg.home, cfg.personal_home = CMAC_HOME, False
        cfg.world, cfg.vehicle = config.TEST_WORLD, "sd_fpv7"
    else:
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


def count_frames(stack: Stack, result: dict[str, Any] | None = None) -> int:
    """Camera frames through the radio in FRAME_WINDOW_S, and Gazebo's real-time factor
    over the same window (camera rendering is the costly part on a computer without GPU)."""

    if stack.world is None:
        raise InfraError("ingen Gazebo-värld kör")
    before = stack.video_frames()
    gazebo.enable_camera(stack.cfg.instance, streaming=lambda: stack.video_frames() > before)
    start, t_start = stack.video_frames(), gazebo.world_times(stack.world, stack.cfg.instance)
    time.sleep(FRAME_WINDOW_S)
    t_end = gazebo.world_times(stack.world, stack.cfg.instance)
    if result is not None:
        result["real_time_factor"] = gazebo.real_time_factor(t_start, t_end)
        if t_start is not None and t_end is not None and t_end[0] > t_start[0]:
            result["sim_seconds"] = round(t_end[0] - t_start[0], 2)
    return stack.video_frames() - start


def hover_current(harness: Harness, seconds: float = CURRENT_WINDOW_S) -> float | None:
    """Mean battery current (A) over ``seconds`` of hovering, or None where the physics
    reports none (Gazebo's JSON link sends no battery data, so SITL shows 0 A)."""
    samples: list[float] = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        harness.pump(0.25)
        if harness.state.battery_current_a > 0:
            samples.append(harness.state.battery_current_a)
    return round(sum(samples) / len(samples), 2) if samples else None


Measure = Callable[[dict[str, Any], dict[str, bool]], None]


def hover_until(
    harness: Harness,
    measure: Measure,
    limit_s: float,
    result: dict[str, Any],
    checks: dict[str, bool],
) -> None:
    """Run ``measure`` in a thread while the harness keeps pumping (and so sending its
    heartbeat). The thread writes into its own dictionaries, which are taken over only
    when it finishes within ``limit_s``; after that a late write can change nothing, and
    the camera check is failed."""
    own_result: dict[str, Any] = {}
    own_checks: dict[str, bool] = {}
    worker = threading.Thread(target=measure, args=(own_result, own_checks), daemon=True)
    worker.start()
    end = time.monotonic() + limit_s
    while worker.is_alive() and time.monotonic() < end:
        harness.pump(0.5)
    if worker.is_alive():
        result["measure_error"] = f"mätningen tog längre än {limit_s:g} s"
        checks["camera"] = False
        return
    result.update(own_result)
    checks.update(own_checks)


def fly(
    harness: Harness, result: dict[str, Any], t0: float, hover: Callable[[], None] = lambda: None
) -> None:
    harness.wait(lambda s: s.heartbeat, 60, "HEARTBEAT från flygkontrollen")
    harness.wait(lambda s: s.ready, 90, "position från EKF och GPS")
    result["ready_s"] = round(time.monotonic() - t0, 1)
    harness.set_mode("GUIDED")
    harness.arm()
    harness.takeoff(TAKEOFF_ALT_M)
    harness.wait(lambda s: s.rel_alt_m >= TAKEOFF_ALT_M - 0.5, 60, f"höjden {TAKEOFF_ALT_M} m")
    harness.pump(5.0)  # hover: the overshoot shows up in the maximum
    hover()
    harness.pump(1.0)
    harness.set_mode("LAND")
    harness.wait(lambda s: not s.armed, 90, "landning och desarmering")


def run(cfg: config.SimConfig, vehicle: str = "") -> tuple[int, Path]:
    cfg = harden(cfg, vehicle)
    t0 = time.monotonic()
    run_dir = new_run_dir(cfg.instance, "smoke")
    stack = Stack(cfg, run_dir, observer=False)
    result: dict[str, Any] = {
        "schema": 1,
        "name": "sim-smoke",
        "physics": cfg.physics,
        "gui": cfg.gui,
        "gi": cfg.gi,
        "instance": cfg.instance,
        "vehicle": cfg.vehicle,
        "world": cfg.world if cfg.vehicle in world.MODELS else gazebo.STOCK_WORLD,
        "home": cfg.home,
        "target_alt_m": TAKEOFF_ALT_M,
    }
    checks: dict[str, bool] = {}
    harness: Harness | None = None
    try:
        harness = Harness(stack.ports["harness_mavlink"])
        stack.up()
        fly_start = time.monotonic()

        def measure(out: dict[str, Any], out_checks: dict[str, bool]) -> None:
            try:
                out["non_loopback_listeners"] = check_listeners(stack)
                if cfg.physics == "gz" and not cfg.gui and stack.world is not None:
                    frames = count_frames(stack, out)
                    out["camera_frames_10s"] = frames
                    out["camera_frames_min"] = min_frames(
                        stack.world.vehicle, out.get("sim_seconds", FRAME_WINDOW_S)
                    )
                    out_checks["camera"] = frames >= out["camera_frames_min"]
                    rtf = out.get("real_time_factor")
                    out_checks["real_time_factor"] = rtf is not None and rtf >= MIN_REAL_TIME_FACTOR
            except Exception as exc:  # reported in result.json, never lost in the thread
                out["measure_error"] = stack.mask(f"{type(exc).__name__}: {exc}")
                out_checks["camera"] = False

        def in_hover() -> None:
            # Listeners, camera and real-time factor are measured while the vehicle hovers
            # at 20 m; the harness keeps sending its heartbeat meanwhile (GCS failsafe).
            if harness is None:
                return
            current = hover_current(harness)
            if current is not None:
                result["hover_current_a"] = current
            hover_until(harness, measure, MEASURE_LIMIT_S, result, checks)

        fly(harness, result, t0, in_hover)
        result["flight_s"] = round(time.monotonic() - fly_start, 1)
        result["max_alt_m"] = round(harness.state.max_rel_alt_m, 2)
        checks["altitude"] = abs(result["max_alt_m"] - TAKEOFF_ALT_M) <= ALT_TOLERANCE_M
        # Not measured (the measurement timed out) counts as a failed check.
        checks["loopback_only"] = result.get("non_loopback_listeners") == []
    except (InfraError, VehicleError) as exc:
        result["error"] = stack.mask(str(exc))
        checks["completed"] = False
    except Exception as exc:
        result["error"] = stack.mask(f"{type(exc).__name__}: {exc}")
        checks["completed"] = False
    finally:
        if harness is not None:
            if not all(checks.values()) or "error" in result:
                # The flight controller's own words (pre-arm reasons and the like).
                result["fc_texts"] = [stack.mask(t) for _, t in harness.state.texts[-12:]]
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
    vehicle = values.pop("vehicle", "")
    if vehicle and config.VEHICLE_ALIASES.get(vehicle, vehicle) not in config.VEHICLES:
        _args.fail(f"sd_sim.smoke: vehicle= ska vara en av {', '.join(config.VEHICLES)}")
    try:
        cfg = config.load({**values, "video": "off", "qgc": "false"})
    except config.ConfigError as exc:
        _args.fail(f"sd_sim.smoke: {exc}")
    code, run_dir = run(cfg, vehicle)
    summary = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    say(json.dumps(summary, indent=2, ensure_ascii=False))
    say("GODKÄNT" if code == 0 else "UNDERKÄNT")
    return code


if __name__ == "__main__":
    sys.exit(main())
