"""Person models for the farm: the catalog, the fetch and the person as a world object.

The models come from Gazebo Fuel under CC0 1.0 or CC BY 4.0. The catalog is
``simulation/farm/people/<id>.toml``: where each model comes from, its pinned
version and hash, its licence and credit, and its box measured from the mesh. Credits
are in ``ATTRIBUTION.md`` there and in every round's metadata. The models
are tens of megabytes with textures, so they are never committed: ``fetch()`` downloads
each pinned version once into ``build/simfarm/assets/people/<id>/``, checks its SHA-256
and unpacks only the meshes and textures. World generation never downloads anything;
a missing model is an error that says to run ``just sim-farm-setup``.

A ``Person`` is an ``sd_sim.objects.WorldObject``: a static model at world level with the
simulator's label (``PERSON_LABEL``) so the bounding-box camera reports it, in one of
three poses (standing, sitting, lying) and either with its own textures or in one solid
colour (``tint``), which varies the clothes. ``corners()`` gives the eight corners of its
box in the world, which the label code projects into the picture.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
import shutil
import tomllib
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sd_sim import objects, paths

# The class register's simulator label for a person (bounding-box camera).
PERSON_LABEL = 2
FUEL = "https://fuel.gazebosim.org/1.0"
MAX_ZIP_BYTES = 64 * 1024 * 1024
MAX_UNPACKED_BYTES = 256 * 1024 * 1024
KEEP_DIRS = ("meshes/", "materials/")
POSES = ("standing", "sitting", "lying")
_LOOK = re.compile(r"[a-z0-9_]{1,40}-[a-z]{1,12}-[a-z]{1,12}")
LICENSES = ("CC0-1.0", "CC-BY-4.0")
# How far a model may be scaled to a drawn body height (``Person.scale``): a child of
# 1.1 m to a tall adult of 2 m, from models made 1.67-1.93 m tall.
SCALE_RANGE = (0.55, 1.25)


@dataclass(frozen=True)
class PersonModel:
    """One pinned Fuel model. ``bounds_m`` is its box in the model frame (x, y, z min and
    max, metres, z up, feet at z = 0), measured from the mesh."""

    id: str
    owner: str
    fuel_name: str
    version: int
    sha256: str
    license: str  # SPDX
    credit: str
    mesh: str  # path inside the model folder
    pose: str  # the model's own pose: "standing" or "sitting"
    bounds_m: tuple[float, float, float, float, float, float]
    albedo_map: str = ""  # a texture the SDF must name itself (PBR), else the mesh's own
    mesh_offset_z_m: float = 0.0  # added to the mesh's height so the feet are at z = 0
    stature_m: float = 0.0  # the standing body height a seated model shows (0: its box's)

    @property
    def body_height_m(self) -> float:
        """The standing body height the model is made at: its box's height for a standing
        model, ``stature_m`` for a seated one (whose box is lower)."""
        return self.stature_m or self.bounds_m[5] - self.bounds_m[4]

    @property
    def url(self) -> str:
        name = urllib.parse.quote(self.fuel_name)
        return f"{FUEL}/{urllib.parse.quote(self.owner)}/models/{name}/{self.version}/{name}.zip"

    @property
    def page(self) -> str:
        return f"https://app.gazebosim.org/{self.owner}/fuel/models/{urllib.parse.quote(self.fuel_name)}"

    def attribution(self) -> dict[str, str]:
        return {
            "model": f"{self.fuel_name} v{self.version}",
            "credit": self.credit,
            "license": self.license,
            "source": self.page,
        }


CATALOG_DIR = paths.SIMULATION / "farm" / "people"


def load_catalog(folder: Path = CATALOG_DIR) -> tuple[PersonModel, ...]:
    """Every ``<id>.toml`` in the catalog folder (each carries its model's licence)."""
    out = []
    for path in sorted(folder.glob("*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        bounds = tuple(float(v) for v in data["bounds_m"])
        if (
            len(bounds) != 6
            or data["id"] != path.stem
            or data["pose"] not in ("standing", "sitting")
        ):
            raise ValueError(f"{path}: ogiltig personmodell")
        if data["license"] not in LICENSES:
            raise ValueError(f"{path}: licensen {data['license']} är inte tillåten")
        out.append(
            PersonModel(
                id=data["id"],
                owner=data["owner"],
                fuel_name=data["fuel_name"],
                version=int(data["version"]),
                sha256=data["sha256"],
                license=data["license"],
                credit=data["credit"],
                mesh=data["mesh"],
                pose=data["pose"],
                bounds_m=(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]),
                albedo_map=data.get("albedo_map", ""),
                mesh_offset_z_m=float(data.get("mesh_offset_z_m", 0.0)),
                stature_m=float(data.get("stature_m", 0.0)),
            )
        )
    return tuple(out)


CATALOG = load_catalog()
BY_ID = {m.id: m for m in CATALOG}


class PeopleError(RuntimeError):
    """A person model is missing, has the wrong hash or a bad archive."""


def assets_dir(repo: Path = paths.REPO) -> Path:
    """``build/simfarm/assets/people``: add it to ``GZ_SIM_RESOURCE_PATH``."""
    return repo / "build" / "simfarm" / "assets" / "people"


def model_dir(model: PersonModel, repo: Path = paths.REPO) -> Path:
    return assets_dir(repo) / model.id


def _stamp(model: PersonModel) -> str:
    return f"{model.sha256}\n"


def _stamped(folder: Path, model: PersonModel) -> bool:
    stamp = folder / ".sha256"
    return stamp.is_file() and stamp.read_text(encoding="utf-8") == _stamp(model)


def is_fetched(model: PersonModel, repo: Path = paths.REPO) -> bool:
    return _stamped(model_dir(model, repo), model)


def search_dirs(env: Mapping[str, str], repo: Path = paths.REPO) -> list[Path]:
    """Where Gazebo looks for ``model://<id>``, in its order: this checkout's assets, then
    ``GZ_SIM_RESOURCE_PATH`` (in a farm round it has the owner's checkout's assets)."""
    parts = [str(assets_dir(repo)), *env.get("GZ_SIM_RESOURCE_PATH", "").split(":")]
    return [Path(p) for p in dict.fromkeys(parts) if p]


def missing(repo: Path = paths.REPO, env: Mapping[str, str] | None = None) -> list[str]:
    """The catalog models that are not fetched. Without ``env`` only this checkout's assets
    count (what ``fetch`` fills). With ``env`` a model counts where Gazebo would load it
    from: the first folder in ``search_dirs`` that has it, with the right hash stamp."""
    if env is None:
        return [m.id for m in CATALOG if not is_fetched(m, repo)]
    dirs = search_dirs(env, repo)
    out = []
    for m in CATALOG:
        found = next((d / m.id for d in dirs if (d / m.id).is_dir()), None)
        if found is None or not _stamped(found, m):
            out.append(m.id)
    return out


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed https URL
        data = resp.read(MAX_ZIP_BYTES + 1)
    if len(data) > MAX_ZIP_BYTES:
        raise PeopleError(f"{url}: större än {MAX_ZIP_BYTES} byte")
    return data


def unpack(model: PersonModel, data: bytes, dest: Path) -> None:
    """Check the hash and unpack the meshes and textures (nothing else, no path that
    leaves ``dest``) into ``dest``, replacing what was there."""
    got = hashlib.sha256(data).hexdigest()
    if got != model.sha256:
        raise PeopleError(f"{model.id}: fel SHA-256 ({got}), väntade {model.sha256}")
    tmp = dest.with_name(f".{dest.name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    total = 0
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                name = PurePosixPath(info.filename)
                if info.is_dir() or not str(name).startswith(KEEP_DIRS):
                    continue
                if name.is_absolute() or ".." in name.parts:
                    raise PeopleError(f"{model.id}: otillåten sökväg i arkivet: {name}")
                total += info.file_size
                if total > MAX_UNPACKED_BYTES:
                    raise PeopleError(f"{model.id}: arkivet är för stort uppackat")
                target = tmp.joinpath(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
        if not (tmp / model.mesh).is_file():
            raise PeopleError(f"{model.id}: {model.mesh} saknas i arkivet")
        (tmp / ".sha256").write_text(_stamp(model), encoding="utf-8")
        shutil.rmtree(dest, ignore_errors=True)
        tmp.replace(dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fetch(
    repo: Path = paths.REPO,
    download: Callable[[str], bytes] = _download,
    say: Callable[[str], None] = print,
) -> list[str]:
    """Fetch every catalog model that is not already here; returns the fetched ids."""
    got: list[str] = []
    for model in CATALOG:
        if is_fetched(model, repo):
            continue
        say(f"hämtar {model.fuel_name} v{model.version} ({model.license})")
        unpack(model, download(model.url), model_dir(model, repo))
        got.append(model.id)
    return got


def _num(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".") or "0"


def _rot_x(p: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return (p[0], c * p[1] - s * p[2], s * p[1] + c * p[2])


def _rot_z(p: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1], p[2])


@dataclass(frozen=True)
class Person:
    """A person on the ground. ``pose`` "lying" lays a standing model on its back (a
    quarter turn about its x axis) with its lowest point on the ground. ``scale`` sizes
    the whole model evenly (``SCALE_RANGE``), so one mesh can be a child or a tall adult;
    ``height_m`` is the standing body height that gives. ``look`` names a copy of the
    model with other clothes colours (``sd_sim.farm.looks``), made in ``build/`` before
    the world is loaded; it replaces ``tint``."""

    name: str
    model: str  # catalog id
    pose: str
    east_m: float
    north_m: float
    yaw_deg: float
    tint: tuple[float, float, float] | None = None
    is_target: bool = False
    up_m: float = 0.0
    scale: float = 1.0
    look: str = ""

    def __post_init__(self) -> None:
        if self.model not in BY_ID:
            raise objects.ObjectError(f"{self.name}: okänd personmodell {self.model!r}")
        if self.pose not in POSES:
            raise objects.ObjectError(f"{self.name}: okänd pose {self.pose!r}")
        own = BY_ID[self.model].pose
        if (self.pose == "sitting") != (own == "sitting"):
            raise objects.ObjectError(f"{self.name}: {self.model} kan inte vara {self.pose}")
        if self.tint is not None and not all(0.0 <= c <= 1.0 for c in self.tint):
            raise objects.ObjectError(f"{self.name}: tint ska vara tre tal mellan 0 och 1")
        low, high = SCALE_RANGE
        if not low <= self.scale <= high:
            raise objects.ObjectError(f"{self.name}: scale ska ligga mellan {low:g} och {high:g}")
        if self.look and not (_LOOK.fullmatch(self.look) and self.look.startswith(self.model)):
            raise objects.ObjectError(f"{self.name}: ogiltig look {self.look!r}")

    @classmethod
    def from_truth(cls, row: Mapping[str, Any]) -> Person:
        """The person as the world's facit file records it (``world_truth`` row)."""
        truth = row["truth"]
        tint = truth.get("tint")
        return cls(
            name=str(row["name"]),
            model=str(truth["model"]),
            pose=str(truth["pose"]),
            east_m=float(row["east_m"]),
            north_m=float(row["north_m"]),
            yaw_deg=float(row["yaw_deg"]),
            tint=(float(tint[0]), float(tint[1]), float(tint[2])) if tint else None,
            is_target=truth.get("role") == "target",
            up_m=float(row.get("up_m", 0.0)),
            scale=float(truth.get("scale", 1.0)),
            look=str(truth.get("look") or ""),
        )

    @property
    def spec(self) -> PersonModel:
        return BY_ID[self.model]

    @property
    def label(self) -> int:
        return PERSON_LABEL

    @property
    def height_m(self) -> float:
        """The person's standing body height (also when sitting or lying)."""
        return self.spec.body_height_m * self.scale

    def _bounds(self) -> tuple[float, ...]:
        return tuple(v * self.scale for v in self.spec.bounds_m)

    @property
    def roll_rad(self) -> float:
        return math.pi / 2 if self.pose == "lying" else 0.0

    def _local_corners(self) -> list[tuple[float, float, float]]:
        """The model box's corners after the pose turn, in the person's frame (origin on
        the ground under the model's origin, before yaw)."""
        x0, x1, y0, y1, z0, z1 = self._bounds()
        pts = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
        pts = [_rot_x(p, self.roll_rad) for p in pts]
        lift = -min(p[2] for p in pts)
        return [(p[0], p[1], p[2] + lift) for p in pts]

    @property
    def lift_m(self) -> float:
        """How far the turned model is raised so it rests on the ground."""
        _x0, _x1, y0, y1, z0, z1 = self._bounds()
        pts = [_rot_x((0.0, y, z), self.roll_rad) for y in (y0, y1) for z in (z0, z1)]
        return -min(p[2] for p in pts)

    @property
    def footprint_m(self) -> float:
        return max(math.hypot(p[0], p[1]) for p in self._local_corners())

    def corners(self, ground_z: float = 0.0) -> list[tuple[float, float, float]]:
        """The eight box corners in the world frame (x east, y north, z up)."""
        yaw = objects.yaw_enu(self.yaw_deg)
        out = []
        for p in self._local_corners():
            x, y, z = _rot_z(p, yaw)
            out.append((self.east_m + x, self.north_m + y, ground_z + self.up_m + z))
        return out

    @property
    def truth(self) -> Mapping[str, Any]:
        return {
            "cls": "person",
            "role": "target" if self.is_target else "person",
            "model": self.model,
            "pose": self.pose,
            "tint": list(self.tint) if self.tint is not None else None,
            **({} if self.scale == 1.0 else {"scale": self.scale}),
            **({"look": self.look} if self.look else {}),
            "height_m": round(self.height_m, 3),
            "attribution": self.spec.attribution(),
        }

    def sdf(self, pose: objects.Pose) -> str:
        spec = self.spec
        folder = self.look or spec.id
        uri = f"model://{folder}/{spec.mesh}"
        label = (
            '<plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">'
            f"<label>{PERSON_LABEL}</label></plugin>"
        )
        z = self.lift_m + spec.mesh_offset_z_m * self.scale
        size = "" if self.scale == 1.0 else f"<scale>{' '.join([_num(self.scale)] * 3)}</scale>"
        if self.tint is not None and not self.look:
            rgb = " ".join(_num(c) for c in self.tint)
            material = f"<material><ambient>{rgb} 1</ambient><diffuse>{rgb} 1</diffuse></material>"
        elif spec.albedo_map:
            material = (
                "<material><diffuse>1 1 1 1</diffuse><pbr><metal>"
                f"<albedo_map>model://{folder}/{spec.albedo_map}</albedo_map>"
                "</metal></pbr></material>"
            )
        else:
            material = ""
        return f"""<model name="{self.name}">
    <static>true</static>
    <pose>{pose.sdf()}</pose>
    <link name="link">
      <visual name="visual">
        <pose>0 0 {_num(z)} {_num(self.roll_rad)} 0 0</pose>
        <geometry><mesh><uri>{uri}</uri>{size}</mesh></geometry>{material}
      </visual>
    </link>
    {label}
  </model>"""
