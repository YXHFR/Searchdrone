"""The 7-inch quad ``sd_fpv7``: the owner's drone, built from the CAD model's numbers.

``simulation/gazebo/models/sd_fpv7/cad.json`` is written by ``tools/sim_model_from_cad.py``
from ``hardware/cad/fpv7`` (and checked against it in CI). Mass, centre of gravity and the
whole inertia matrix come from there, never from a hand-written number:

- the Gazebo model (``model.sdf.jinja`` next to ``cad.json``) splits the CAD mass over the
  body, the IMU, the camera and the four propellers so that together they have exactly the
  CAD mass, centre of gravity and inertia (``links``);
- the builtin-physics frame file for SITL (``frame_json``) takes the CAD mass and inertia.

The rotors sit where the CAD motors are (an H frame, 193 x 223 mm between the axes), and
the meshes are the CAD parts. Rotor speed, thrust-to-weight and the battery are this
module's constants; the SITL parameters are in ``simulation/sitl/params/sd_fpv7-sim.parm``.
"""

from __future__ import annotations

import json
import math
from functools import cache
from typing import Any

from sd_sim import paths

MODEL_NAME = "sd_fpv7"
MODEL_DIR = paths.REPO / "simulation" / "gazebo" / "models" / MODEL_NAME
CAD_JSON = MODEL_DIR / "cad.json"

# Gazebo. The joint's top speed is below the real motor's (about 2 500 rad/s loaded on 6S):
# LiftDrag only needs the thrust at full speed, which the blade area is sized for, and a
# slower joint stays well resolved at Gazebo's 1 ms step.
MAX_ROTOR_RAD_S = 1200.0
# Full rotor speed lifts this many times the take-off weight (MOT_SPIN_MAX 0.90 sends
# PWM 1900, the top of the plugin's range, so all of it is usable).
THRUST_TO_WEIGHT = 2.5
# Where the lift acts along a blade, as a share of the propeller radius (as in sd_quad14).
BLADE_CP_SHARE = 0.47
# The rotor joint's speed loop settles in this time; its gain follows from the
# propeller's inertia, and the joint damping costs 2 % of the commanded speed.
ROTOR_TIME_CONSTANT_S = 0.0025
ROTOR_DAMPING_SHARE = 0.02
IMU_MASS_KG = 0.01
IMU_INERTIA = 1e-6
CAMERA_INERTIA = 2e-7
# Same stream as sd_quad14 (frame rate and size), the CAD camera's field of view.
CAMERA = {"width": 640, "height": 480, "fps": 15}

# Builtin physics (SITL's SIM_Frame) and the battery: Dogcom 6S2P 21700 Li-ion, 10 000 mAh.
BATTERY_MAH = 10000
CELLS = 6
CELL_NOMINAL_V = 3.6
CELL_FULL_V = 4.2
# EMAX ECO II 2807 1300KV with HQ 7x4x3 on 6S at the CAD mass: about 15.5 A in hover.
HOVER_CURRENT_A = 15.5
HOVER_THROTTLE = 0.45  # throttle out at hover in the builtin model (about 3.5 x weight)
SPIN_MIN = 0.15
SPIN_MAX = 0.90  # equals MOT_SPIN_MAX in sd_fpv7-sim.parm
# The body leans this much to fly this fast (drag of the body and the rotors).
REF_SPEED_MS = 15.0
REF_ANGLE_DEG = 25.0

Vec = list[float]
Mat = list[list[float]]

# ArduPilot's quad X order: front right and back left turn counter-clockwise seen from
# above, front left and back right clockwise. (sign of x, sign of y, counter-clockwise)
AP_QUAD_X = ((1, -1, True), (-1, 1, True), (1, 1, False), (-1, -1, False))


@cache
def cad() -> dict[str, Any]:
    return json.loads(CAD_JSON.read_text(encoding="utf-8"))


def stand_m() -> float:
    """The model origin (the CAD origin, under the bottom plate) above the leg tips."""
    return float(cad()["stand_m"])


def prop_radius_m() -> float:
    return float(cad()["prop"]["diameter_m"]) / 2


def _outer(a: Vec, b: Vec) -> Mat:
    return [[x * y for y in b] for x in a]


def _shift(mass: float, d: Vec) -> Mat:
    """Parallel-axis term: inertia of a point mass ``mass`` at ``d`` about the origin."""
    d2 = sum(v * v for v in d)
    return [[mass * ((d2 if i == j else 0.0) - d[i] * d[j]) for j in range(3)] for i in range(3)]


def _add(a: Mat, b: Mat, sign: float = 1.0) -> Mat:
    return [[a[i][j] + sign * b[i][j] for j in range(3)] for i in range(3)]


def _diag(ixx: float, iyy: float, izz: float) -> Mat:
    return [[ixx, 0.0, 0.0], [0.0, iyy, 0.0], [0.0, 0.0, izz]]


def rotor_inertia() -> tuple[float, float]:
    """(ixx, izz) of one propeller as a thin disc around its axis."""
    m, r = float(cad()["prop"]["mass_kg"]), prop_radius_m()
    return m * r * r / 4, m * r * r / 2


def rotors() -> list[dict[str, Any]]:
    """The CAD rotors in ArduPilot's quad X order, with the turning direction checked."""
    out = []
    for sx, sy, ccw in AP_QUAD_X:
        found = [
            r
            for r in cad()["rotors"]
            if math.copysign(1, r["position_m"][0]) == sx
            and math.copysign(1, r["position_m"][1]) == sy
        ]
        if len(found) != 1 or found[0]["clockwise"] == ccw:
            raise ValueError("cad.json: rotorerna stämmer inte med ArduPilots quad X")
        x, y, z = found[0]["position_m"]
        out.append(
            {
                "x": x,
                "y": y,
                "z": z,
                "ccw": ccw,
                # A CCW rotor spins with positive joint speed; its blades meet the air
                # along +y at +x. A CW rotor mirrors both.
                "sign": 1 if ccw else -1,
                "mesh": cad()["meshes"]["prop_ccw" if ccw else "prop_cw"],
            }
        )
    return out


def links() -> dict[str, dict[str, Any]]:
    """Mass, centre of mass and inertia (about the centre of mass) of every link, in the
    model frame (the CAD frame). Together they have the CAD model's mass, centre of
    gravity and inertia matrix: the body gets what the small links do not."""
    c = cad()
    rxx, rzz = rotor_inertia()
    small: dict[str, dict[str, Any]] = {
        "imu_link": {
            "mass": IMU_MASS_KG,
            "com": list(c["imu_m"]),
            "inertia": _diag(IMU_INERTIA, IMU_INERTIA, IMU_INERTIA),
        },
        "camera_link": {
            "mass": float(c["camera_mass_kg"]),
            "com": list(c["camera_m"]),
            "inertia": _diag(CAMERA_INERTIA, CAMERA_INERTIA, CAMERA_INERTIA),
        },
    }
    for i, r in enumerate(rotors()):
        small[f"rotor_{i}"] = {
            "mass": float(c["prop"]["mass_kg"]),
            "com": [r["x"], r["y"], r["z"]],
            "inertia": _diag(rxx, rxx, rzz),
        }
    total, cog = float(c["mass_kg"]), list(c["cog_m"])
    about_origin = _add(c["inertia_kg_m2"], _shift(total, cog))
    mass, moment = total, [total * v for v in cog]
    for link in small.values():
        mass -= link["mass"]
        moment = [m - link["mass"] * p for m, p in zip(moment, link["com"], strict=True)]
        about_origin = _add(
            about_origin, _add(link["inertia"], _shift(link["mass"], link["com"])), -1
        )
    com = [m / mass for m in moment]
    base = {"mass": mass, "com": com, "inertia": _add(about_origin, _shift(mass, com), -1)}
    return {"base_link": base, **small}


def composite(parts: dict[str, dict[str, Any]]) -> tuple[float, Vec, Mat]:
    """Mass, centre of mass and inertia about it of links given as ``links()`` gives them."""
    mass = sum(p["mass"] for p in parts.values())
    com = [sum(p["mass"] * p["com"][k] for p in parts.values()) / mass for k in range(3)]
    inertia = _diag(0.0, 0.0, 0.0)
    for p in parts.values():
        d = [a - b for a, b in zip(p["com"], com, strict=True)]
        inertia = _add(inertia, _add(p["inertia"], _shift(p["mass"], d)))
    return mass, com, inertia


def _r(values: Vec, digits: int = 9) -> list[float]:
    return [round(v, digits) for v in values]


def blade_cp_m() -> float:
    return BLADE_CP_SHARE * prop_radius_m()


def model_context(instance: int, ports_i: dict[str, int]) -> dict[str, Any]:
    """The template's values that come from the CAD model; ``sd_sim.world`` adds the lift."""
    c = cad()
    parts = links()
    base = parts["base_link"]
    inertia = base["inertia"]
    rxx, rzz = rotor_inertia()
    p_gain = rzz / ROTOR_TIME_CONSTANT_S
    rotor_list = rotors()
    for r in rotor_list:
        # The arm visual is part of the CAD mesh; only the rotor needs a pose.
        r["x"], r["y"], r["z"] = round(r["x"], 6), round(r["y"], 6), round(r["z"], 6)
    legs = [{"pose": _r(leg["position_m"], 6), "size": _r(leg["size_m"], 6)} for leg in c["legs"]]
    return {
        "name": f"{MODEL_NAME}_i{instance}",
        "instance": instance,
        "fdm_port_in": ports_i["sitl_json_fdm"],
        "video_port": ports_i["video_raw_udp"],
        "stand_m": stand_m(),
        "base_mass": round(base["mass"], 6),
        "base_com": _r(base["com"]),
        "base_inertia": {
            "ixx": round(inertia[0][0], 9),
            "iyy": round(inertia[1][1], 9),
            "izz": round(inertia[2][2], 9),
            "ixy": round(inertia[0][1], 9),
            "ixz": round(inertia[0][2], 9),
            "iyz": round(inertia[1][2], 9),
        },
        "body_box": {
            "pose": _r(c["body_box_m"]["centre"], 6),
            "size": _r(c["body_box_m"]["size"], 6),
        },
        "legs": legs,
        "meshes": [f"model://{MODEL_NAME}/{m}" for m in c["meshes"]["body"]],
        "mesh_scale": c["meshes"]["scale"],
        "prop_mesh": {
            True: f"model://{MODEL_NAME}/{c['meshes']['prop_ccw']}",
            False: f"model://{MODEL_NAME}/{c['meshes']['prop_cw']}",
        },
        "imu_mass": IMU_MASS_KG,
        "imu_pose": _r(parts["imu_link"]["com"], 6),
        "imu_inertia": IMU_INERTIA,
        "rotors": rotor_list,
        "rotor_mass": parts["rotor_0"]["mass"],
        "rotor_ixx": round(rxx, 10),
        "rotor_izz": round(rzz, 10),
        "rotor_p_gain": round(p_gain, 6),
        "rotor_damping": round(ROTOR_DAMPING_SHARE * p_gain, 8),
        "multiplier": MAX_ROTOR_RAD_S,
        "blade_cp": round(blade_cp_m(), 5),
        "camera_mass": parts["camera_link"]["mass"],
        "camera_pose": _r(parts["camera_link"]["com"], 6),
        "camera_inertia": CAMERA_INERTIA,
        "camera": {
            **CAMERA,
            "hfov_rad": round(math.radians(float(c["camera_hfov_deg"])), 6),
        },
    }


def frame_json() -> dict[str, Any]:
    """SITL's builtin-physics frame file (``--model x:sd_fpv7.json``) from the CAD numbers."""
    c = cad()
    inertia = c["inertia_kg_m2"]
    xs = [abs(r["position_m"][0]) for r in c["rotors"]]
    ys = [abs(r["position_m"][1]) for r in c["rotors"]]
    diagonal = 2 * math.hypot(sum(xs) / len(xs), sum(ys) / len(ys))
    radius = prop_radius_m()
    return {
        "mass": c["mass_kg"],
        "diagonal_size": round(diagonal, 4),
        "moment_inertia": [inertia[0][0], inertia[1][1], inertia[2][2]],
        "refSpd": REF_SPEED_MS,
        "refAngle": REF_ANGLE_DEG,
        "refVoltage": round(CELLS * CELL_NOMINAL_V, 2),
        "refCurrent": HOVER_CURRENT_A,
        "refAlt": 100,
        "refTempC": 20,
        "maxVoltage": round(CELLS * CELL_FULL_V, 2),
        "battCapacityAh": BATTERY_MAH / 1000,
        "refBatRes": 0.03,
        "propExpo": 0.65,
        "refRotRate": 120,
        "hoverThrOut": HOVER_THROTTLE,
        "pwmMin": 1000,
        "pwmMax": 2000,
        "spin_min": SPIN_MIN,
        "spin_max": SPIN_MAX,
        "slew_max": 0,
        "disc_area": round(len(c["rotors"]) * math.pi * radius * radius, 4),
        "mdrag_coef": 0.1,
        "num_motors": len(c["rotors"]),
    }
