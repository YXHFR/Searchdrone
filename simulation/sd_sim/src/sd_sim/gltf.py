"""A minimal binary glTF 2.0 (``.glb``) writer for the site's textured meshes.

One mesh, one primitive per material, positions, normals, tangents and texture
coordinates, 32-bit indices. Materials are glTF's metallic-roughness PBR with textures
that stay next to the file (``uri`` relative to it), so several meshes share one picture
in the renderer. ``alphaMode`` MASK cuts out the transparent parts of leaves and needles;
``doubleSided`` shows a card from both sides.

Coordinates are the site's own (east, north, up), not glTF's Y up: Gazebo reads a glTF
file's coordinates as they are, without turning Y up into Z up.

The same input always gives the same bytes.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

_FLOAT = 5126
_UINT = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963


@dataclass(frozen=True)
class PbrMaterial:
    name: str
    base_color: str  # RGBA picture; alpha below ``alpha_cutoff`` is cut out
    normal: str = ""  # tangent-space normal map, OpenGL convention (+Y up)
    # Occlusion-roughness-metallic in R, G and B: glTF reads G and B, R is occlusion.
    orm: str = ""
    alpha_cutoff: float = 0.5
    double_sided: bool = True
    roughness: float = 1.0
    metallic: float = 0.0


@dataclass
class Primitive:
    """Triangles of one material; vertices are not shared with other primitives."""

    positions: list[NDArray[np.float32]] = field(default_factory=list)
    normals: list[NDArray[np.float32]] = field(default_factory=list)
    tangents: list[NDArray[np.float32]] = field(default_factory=list)
    uvs: list[NDArray[np.float32]] = field(default_factory=list)
    indices: list[NDArray[np.uint32]] = field(default_factory=list)
    count: int = 0

    def add(
        self,
        positions: NDArray[Any],
        normals: NDArray[Any],
        tangents: NDArray[Any],
        uvs: NDArray[Any],
        faces: NDArray[Any],
    ) -> None:
        n = len(positions)
        self.positions.append(np.asarray(positions, dtype=np.float32).reshape(n, 3))
        self.normals.append(np.asarray(normals, dtype=np.float32).reshape(n, 3))
        self.tangents.append(np.asarray(tangents, dtype=np.float32).reshape(n, 4))
        self.uvs.append(np.asarray(uvs, dtype=np.float32).reshape(n, 2))
        self.indices.append(np.asarray(faces, dtype=np.uint32).reshape(-1) + self.count)
        self.count += n

    def triangles(self) -> int:
        return sum(len(i) for i in self.indices) // 3


def _pad4(data: bytes, fill: bytes = b"\0") -> bytes:
    return data + fill * (-len(data) % 4)


def glb(primitives: dict[str, Primitive], materials: Sequence[PbrMaterial], name: str) -> bytes:
    """The ``.glb`` file for ``primitives`` (material name -> triangles)."""
    used = [m for m in materials if m.name in primitives and primitives[m.name].count]
    images: list[str] = []

    def texture(uri: str) -> int:
        if uri not in images:
            images.append(uri)
        return images.index(uri)

    gl_materials = []
    for m in used:
        pbr: dict[str, Any] = {
            "baseColorTexture": {"index": texture(m.base_color)},
            "metallicFactor": m.metallic,
            "roughnessFactor": m.roughness,
        }
        mat: dict[str, Any] = {
            "name": m.name,
            "pbrMetallicRoughness": pbr,
            "alphaMode": "MASK",
            "alphaCutoff": m.alpha_cutoff,
            "doubleSided": m.double_sided,
        }
        if m.orm:
            pbr["metallicRoughnessTexture"] = {"index": texture(m.orm)}
            mat["occlusionTexture"] = {"index": texture(m.orm)}
        if m.normal:
            mat["normalTexture"] = {"index": texture(m.normal)}
        gl_materials.append(mat)

    blob = bytearray()
    views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []

    def put(array: NDArray[Any], kind: str, target: int, minmax: bool = False) -> int:
        data = np.ascontiguousarray(array).tobytes()
        views.append(
            {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target}
        )
        blob.extend(_pad4(data))
        acc: dict[str, Any] = {
            "bufferView": len(views) - 1,
            "componentType": _UINT if array.dtype == np.uint32 else _FLOAT,
            "count": int(array.shape[0]),
            "type": kind,
        }
        if minmax:
            acc["min"] = [float(v) for v in array.min(axis=0)]
            acc["max"] = [float(v) for v in array.max(axis=0)]
        accessors.append(acc)
        return len(accessors) - 1

    gl_prims = []
    for i, m in enumerate(used):
        p = primitives[m.name]
        pos = np.concatenate(p.positions)
        nor = np.concatenate(p.normals)
        tan = np.concatenate(p.tangents)
        uv = np.concatenate(p.uvs)
        idx = np.concatenate(p.indices)
        attributes = {
            "POSITION": put(pos.astype(np.float32), "VEC3", _ARRAY_BUFFER, minmax=True),
            "NORMAL": put(nor.astype(np.float32), "VEC3", _ARRAY_BUFFER),
            "TANGENT": put(tan.astype(np.float32), "VEC4", _ARRAY_BUFFER),
            "TEXCOORD_0": put(uv.astype(np.float32), "VEC2", _ARRAY_BUFFER),
        }
        gl_prims.append(
            {
                "attributes": attributes,
                "indices": put(idx.astype(np.uint32), "SCALAR", _ELEMENT_ARRAY_BUFFER),
                "material": i,
                "mode": 4,
            }
        )

    doc: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "sd_sim.gltf"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": name, "mesh": 0}],
        "meshes": [{"name": name, "primitives": gl_prims}],
        "materials": gl_materials,
        "textures": [{"source": i, "sampler": 0} for i in range(len(images))],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
        "images": [{"uri": uri} for uri in images],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    js = _pad4(json.dumps(doc, separators=(",", ":"), sort_keys=True).encode(), b" ")
    body = bytes(blob)
    total = 12 + 8 + len(js) + 8 + len(body)
    return b"".join(
        [
            struct.pack("<III", 0x46546C67, 2, total),
            struct.pack("<II", len(js), 0x4E4F534A),
            js,
            struct.pack("<II", len(body), 0x004E4942),
            body,
        ]
    )
