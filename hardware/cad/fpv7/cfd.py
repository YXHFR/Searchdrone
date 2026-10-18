"""A steady RANS run of the air round the fpv7 body in OpenFOAM, to check aero.py.

    uv run --group cad python hardware/cad/fpv7/cfd.py CASE_DIR [--without PART ...] [--run]

writes an OpenFOAM case to CASE_DIR: the drone without its propeller discs as one STL
surface in metres, a box of air round it meshed by blockMesh and snappyHexMesh (2.5 mm
cells at the surface, 5 mm near the body, 10 mm in the wake), and simpleFoam with the
k-omega SST model at ``aero.speed`` from the front. ``--without`` leaves parts out
(``--without canopy nose_fairing`` for the drone without its covers). With ``--run`` it
also runs the case on four cores (OpenFOAM v1912 or newer must be installed; Debian and
Ubuntu: ``apt install openfoam``) and prints Cd A in m^2: the pressure on the drone's
surface integrated to a drag force over the dynamic pressure, averaged over the last third
of the iterations, plus the skin friction of a flat plate with the same wetted area.

Not part of `just cad` or CI: a run takes about ten minutes on four free cores, longer
than the whole CAD job, and needs OpenFOAM. README.md has the results.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import assembly
import numpy as np

if TYPE_CHECKING:
    from assembly import Drone

ITERATIONS = 900
SAMPLES = 9  # pressure written every 100 iterations; the last three are averaged
BODY_LENGTH = 0.3  # m, nose to tail, for the skin friction estimate
CORES = 4
# The air box (m): 0.5 m ahead of the nose, 1 m behind, 0.5 m to the sides, above, below.
DOMAIN = ((-1.0, 0.5), (-0.5, 0.5), (-0.45, 0.55))
NEAR = ((-0.2, 0.18), (-0.16, 0.16), (-0.1, 0.13))  # 5 mm cells
WAKE = ((-0.45, 0.22), (-0.2, 0.2), (-0.14, 0.17))  # 10 mm cells
BASE_CELL = 0.04  # m; the surface is refined four times, to 2.5 mm

HEADER = """FoamFile
{{
    version 2.0;
    format ascii;
    class {cls};
    object {obj};
}}
"""


def write_stl(drone: Drone, path: Path, without: Sequence[str] = ()) -> int:
    """The body (no propeller discs, none of the parts named in ``without``) as one ASCII
    STL in metres. Returns the number of triangles."""
    lines = ["solid drone"]
    n = 0
    for pt in drone.parts:
        if pt.group == "prop" or pt.name in without:
            continue
        vertices, triangles = pt.solid.tessellate(0.2, 0.3)
        v = [(p.x / 1000, p.y / 1000, p.z / 1000) for p in vertices]
        for a, b, c in triangles:
            lines.append("facet normal 0 0 0\n outer loop")
            lines += [f"  vertex {x:.6e} {y:.6e} {z:.6e}" for x, y, z in (v[a], v[b], v[c])]
            lines.append(" endloop\nendfacet")
            n += 1
    lines.append("endsolid drone\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return n


def _box(b: tuple[tuple[float, float], ...]) -> str:
    return f"min ({b[0][0]} {b[1][0]} {b[2][0]}); max ({b[0][1]} {b[1][1]} {b[2][1]});"


def files(a: dict[str, Any]) -> dict[str, str]:
    """The case's dictionaries, by path."""
    u = a["speed"]
    nu = a["kinematic_viscosity"]
    k = 1.5 * (0.01 * u) ** 2  # 1 % turbulence intensity
    omega = k**0.5 / (0.09**0.25 * 0.01)  # 10 mm length scale
    (x0, x1), (y0, y1), (z0, z1) = DOMAIN
    cells = [round((hi - lo) / BASE_CELL) for lo, hi in DOMAIN]
    out: dict[str, str] = {}

    out["system/blockMeshDict"] = (
        HEADER.format(cls="dictionary", obj="blockMeshDict")
        + f"""
convertToMeters 1;
vertices
(
    ({x0} {y0} {z0}) ({x1} {y0} {z0}) ({x1} {y1} {z0}) ({x0} {y1} {z0})
    ({x0} {y0} {z1}) ({x1} {y0} {z1}) ({x1} {y1} {z1}) ({x0} {y1} {z1})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({cells[0]} {cells[1]} {cells[2]}) simpleGrading (1 1 1) );
boundary
(
    inlet {{ type patch; faces ((1 2 6 5)); }}
    outlet {{ type patch; faces ((0 4 7 3)); }}
    sides {{ type patch; faces ((0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }}
);
"""
    )
    out["system/snappyHexMeshDict"] = (
        HEADER.format(cls="dictionary", obj="snappyHexMeshDict")
        + f"""
castellatedMesh true;
snap true;
addLayers false;
geometry
{{
    drone.stl {{ type triSurfaceMesh; name drone; }}
    near {{ type searchableBox; {_box(NEAR)} }}
    wake {{ type searchableBox; {_box(WAKE)} }}
}}
castellatedMeshControls
{{
    maxLocalCells 3000000;
    maxGlobalCells 8000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;
    features ();
    refinementSurfaces {{ drone {{ level (4 4); patchInfo {{ type wall; }} }} }}
    resolveFeatureAngle 30;
    refinementRegions
    {{
        near {{ mode inside; levels ((1e15 3)); }}
        wake {{ mode inside; levels ((1e15 2)); }}
    }}
    locationInMesh ({x0 + 0.05} {y1 - 0.05} {z1 - 0.05});
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap true;
    explicitFeatureSnap false;
    multiRegionFeatureSnap false;
}}
addLayersControls
{{
    relativeSizes true;
    layers {{}}
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}
meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-15;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
    relaxed {{ maxNonOrtho 75; }}
}}
mergeTolerance 1e-6;
"""
    )
    out["system/controlDict"] = (
        HEADER.format(cls="dictionary", obj="controlDict")
        + f"""
application simpleFoam;
startFrom latestTime;
startTime 0;
stopAt endTime;
endTime {ITERATIONS};
deltaT 1;
writeControl timeStep;
writeInterval {ITERATIONS // SAMPLES};
purgeWrite 0;
writeFormat ascii;
writePrecision 8;
timeFormat general;
timePrecision 6;
runTimeModifiable false;
"""
    )
    out["system/fvSchemes"] = (
        HEADER.format(cls="dictionary", obj="fvSchemes")
        + """
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwindV grad(U);
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.33; }
interpolationSchemes { default linear; }
snGradSchemes { default limited corrected 0.33; }
wallDist { method meshWave; }
"""
    )
    out["system/fvSolution"] = (
        HEADER.format(cls="dictionary", obj="fvSolution")
        + """
solvers
{
    p { solver GAMG; tolerance 1e-7; relTol 0.05; smoother GaussSeidel; }
    "(U|k|omega)"
    {
        solver smoothSolver; smoother GaussSeidel; tolerance 1e-8; relTol 0.1; nSweeps 1;
    }
}
SIMPLE { nNonOrthogonalCorrectors 1; consistent no; }
relaxationFactors { fields { p 0.3; } equations { U 0.7; ".*" 0.7; } }
"""
    )
    out["system/decomposeParDict"] = (
        HEADER.format(cls="dictionary", obj="decomposeParDict")
        + f"""
numberOfSubdomains {CORES};
method hierarchical;
coeffs {{ n (2 2 1); }}
"""
    )
    out["constant/transportProperties"] = (
        HEADER.format(cls="dictionary", obj="transportProperties")
        + f"""
transportModel Newtonian;
nu {nu};
"""
    )
    out["constant/turbulenceProperties"] = (
        HEADER.format(cls="dictionary", obj="turbulenceProperties")
        + """
simulationType RAS;
RAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }
"""
    )

    def field(name: str, cls: str, dims: str, internal: str, bc: dict[str, str]) -> str:
        patches = "\n".join(f"    {p} {{ {v} }}" for p, v in bc.items())
        return (
            HEADER.format(cls=cls, obj=name)
            + f"""
dimensions {dims};
internalField uniform {internal};
boundaryField
{{
    #includeEtc "caseDicts/setConstraintTypes"
{patches}
}}
"""
        )

    uvec = f"(-{u} 0 0)"
    out["0/U"] = field("U", "volVectorField", "[0 1 -1 0 0 0 0]", uvec, {
        "inlet": f"type fixedValue; value uniform {uvec};",
        "outlet": f"type inletOutlet; inletValue uniform (0 0 0); value uniform {uvec};",
        "sides": "type slip;",
        "drone": "type noSlip;",
    })  # fmt: skip
    out["0/p"] = field("p", "volScalarField", "[0 2 -2 0 0 0 0]", "0", {
        "inlet": "type zeroGradient;",
        "outlet": "type fixedValue; value uniform 0;",
        "sides": "type slip;",
        "drone": "type zeroGradient;",
    })  # fmt: skip
    out["0/k"] = field("k", "volScalarField", "[0 2 -2 0 0 0 0]", f"{k:.6g}", {
        "inlet": f"type fixedValue; value uniform {k:.6g};",
        "outlet": f"type inletOutlet; inletValue uniform {k:.6g}; value uniform {k:.6g};",
        "sides": "type slip;",
        "drone": f"type kqRWallFunction; value uniform {k:.6g};",
    })  # fmt: skip
    out["0/omega"] = field("omega", "volScalarField", "[0 0 -1 0 0 0 0]", f"{omega:.6g}", {
        "inlet": f"type fixedValue; value uniform {omega:.6g};",
        "outlet": f"type inletOutlet; inletValue uniform {omega:.6g}; value uniform {omega:.6g};",
        "sides": "type slip;",
        "drone": f"type omegaWallFunction; value uniform {omega:.6g};",
    })  # fmt: skip
    out["0/nut"] = field("nut", "volScalarField", "[0 2 -1 0 0 0 0]", "0", {
        "inlet": "type calculated; value uniform 0;",
        "outlet": "type calculated; value uniform 0;",
        "sides": "type calculated; value uniform 0;",
        "drone": "type nutkWallFunction; value uniform 0;",
    })  # fmt: skip
    return out


def write_case(case: Path, without: Sequence[str] = ()) -> None:
    p = assembly.load_params()
    drone = assembly.build(p)
    for rel, text in files(p["aero"]).items():
        (case / rel).parent.mkdir(parents=True, exist_ok=True)
        (case / rel).write_text(text, encoding="utf-8")
    write_stl(drone, case / "constant" / "triSurface" / "drone.stl", without)


def run(case: Path, a: dict[str, Any]) -> dict[str, float]:
    """Mesh and solve the case, then integrate the drag (see ``drag``)."""
    env = {**os.environ, "WM_PROJECT_DIR": os.environ.get("WM_PROJECT_DIR", "/usr/share/openfoam")}
    par = ["mpirun", "--oversubscribe", "-np", str(CORES)]
    if os.geteuid() == 0:  # containers: OpenMPI refuses root unless told
        par.insert(1, "--allow-run-as-root")
        sys.stderr.write("Running as root: mpirun gets --allow-run-as-root.\n")
    times = ",".join(str(t) for t in sample_times())
    steps = [
        ["blockMesh"],
        ["decomposePar", "-force"],
        [*par, "snappyHexMesh", "-parallel", "-overwrite"],
        [*par, "checkMesh", "-parallel", "-constant"],
        [*par, "simpleFoam", "-parallel"],
        [*par, "foamToVTK", "-parallel", "-time", times, "-no-internal", "-patches", "(drone)",
         "-fields", "(p)", "-ascii", "-legacy", "-overwrite"],
    ]  # fmt: skip
    for cmd in steps:
        name = cmd[len(par)] if cmd[0] == "mpirun" else cmd[0]
        if name == "checkMesh":
            # snappyHexMesh added the drone patch: give every processor the start fields
            # again, now that they name it.
            for proc in sorted(case.glob("processor*")):
                shutil.rmtree(proc / "0", ignore_errors=True)
                shutil.copytree(case / "0", proc / "0")
        log = case / f"log.{name}"
        with log.open("w", encoding="utf-8") as f:
            # Fixed argument lists, no shell.
            subprocess.run(cmd, cwd=case, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)  # noqa: S603
    return drag(case, a)


def sample_times() -> list[int]:
    """The iterations whose pressure is averaged: the last third of the run."""
    step = ITERATIONS // SAMPLES
    return [t for t in range(step, ITERATIONS + 1, step) if t > ITERATIONS * 2 // 3]


def read_patch(path: Path) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    """Points (m), faces and the face pressure (kinematic, m^2/s^2) of a legacy ASCII VTK
    file of one patch, as foamToVTK writes it."""
    words = path.read_text(encoding="utf-8").split()
    i = words.index("POINTS")
    n = int(words[i + 1])
    points = np.array(words[i + 3 : i + 3 + 3 * n], dtype=float).reshape(n, 3)
    i = words.index("POLYGONS")
    nf = int(words[i + 1])
    flat = [int(w) for w in words[i + 3 : i + 3 + int(words[i + 2])]]
    faces, k = [], 0
    for _ in range(nf):
        faces.append(flat[k + 1 : k + 1 + flat[k]])
        k += flat[k] + 1
    i = words.index("CELL_DATA")
    j = words.index("p", i)
    # p 1 nFaces float, then the values
    pressure = np.array(words[j + 4 : j + 4 + nf], dtype=float)
    return points, faces, pressure


def face_areas(points: np.ndarray, faces: list[list[int]]) -> np.ndarray:
    """Area vectors (m^2) of the polygon faces."""
    out = np.zeros((len(faces), 3))
    for k, f in enumerate(faces):
        v = points[f]
        out[k] = 0.5 * np.cross(v - v[0], np.roll(v, -1, axis=0) - v[0]).sum(axis=0)
    return out


def pressure_cda(path: Path, speed: float) -> float:
    """Cd A (m^2) from the pressure on the drone's surface. OpenFOAM's face normals point
    out of the air, into the body, so the pressure pushes the body along them; the air
    comes from +x, so drag is the force towards -x."""
    points, faces, p = read_patch(path)
    force = (p[:, None] * face_areas(points, faces)).sum(axis=0)
    return float(-force[0]) / (0.5 * speed**2)


def wetted_area(path: Path) -> float:
    points, faces, _ = read_patch(path)
    return float(np.linalg.norm(face_areas(points, faces), axis=1).sum())


def drag(case: Path, a: dict[str, Any]) -> dict[str, float]:
    """Cd A of the pressure (form drag) averaged over the sampled iterations, its spread
    between them, and the skin friction estimated as a turbulent flat plate of the body's
    wetted area and length (the function objects that integrate the wall shear stress do
    not run in the Debian/Ubuntu OpenFOAM v1912 build)."""
    vtk = {int(f.stem.rsplit("_", 1)[-1]): f for f in (case / "VTK").rglob("drone*.vtk")}
    values = [pressure_cda(vtk[t], a["speed"]) for t in sample_times()]
    area = wetted_area(vtk[sample_times()[-1]])
    reynolds = a["speed"] * BODY_LENGTH / a["kinematic_viscosity"]
    friction = 0.074 * reynolds**-0.2 * area
    form = sum(values) / len(values)
    return {
        "cda_pressure_m2": form,
        "cda_pressure_spread_m2": max(values) - min(values),
        "wetted_area_m2": area,
        "cda_friction_m2": friction,
        "cda_m2": form + friction,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenFOAM case of the fpv7 body.")
    parser.add_argument("case", type=Path, help="directory to write the case to")
    parser.add_argument(
        "--without", nargs="*", default=[], metavar="PART", help="parts to leave out"
    )
    parser.add_argument("--run", action="store_true", help="mesh, solve and print Cd A")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    write_case(args.case, args.without)
    if args.run:
        result = run(args.case, assembly.load_params()["aero"])
        sys.stdout.write("".join(f"{k} = {v:.5f}\n" for k, v in result.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
