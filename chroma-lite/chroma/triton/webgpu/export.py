"""Export shared detector meshes and BVHs for the WGSL geometry viewer.

The v1 buffer starts with magic/version/group-count/word-count. Each 16-word
group record stores six buffer offsets (BLAS, triangles, colors, TLAS,
transforms, BLAS escape links), a float32 quantization origin/scale, and four
counts (BLAS nodes, triangles, TLAS nodes, instances). Its last two words are
reserved. Offsets address uint32 words. Transforms contain nine row-major
object-to-world rotation floats followed by three world translation floats.
The existing packed BLAS words and original triangle IDs are unchanged.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from chroma.triton.bvh import build_packed_bvh
from chroma.triton.viewer import _geometry_groups, _MeshGroup

MAX_UINT = np.uint32(0xFFFFFFFF)
GROUP_WORDS = 16


def bvh_escape_links(nodes):
    """Thread the existing packed tree without changing its nodes/triangles."""
    nodes = np.asarray(nodes, np.uint32)
    escape = np.full(len(nodes), MAX_UINT, np.uint32)
    seen = np.zeros(len(nodes), bool)
    pending = [(0, int(MAX_UINT))]
    while pending:
        index, successor = pending.pop()
        if not 0 <= index < len(nodes) or seen[index]:
            raise ValueError("BVH contains a cycle or invalid child")
        seen[index] = True
        escape[index] = successor
        word = int(nodes[index, 3])
        count, first = word >> 28, word & 0x0FFFFFFF
        for child in range(count):
            pending.append((first + child, first + child + 1 if child + 1 < count else successor))
    if not seen.all():
        raise ValueError("BVH has unreachable nodes")
    return escape


def tlas_threaded(lower, upper, left, right, instance):
    """Eight-word nodes: lo.xyz/child-or-leaf, hi.xyz/escape."""
    result = np.zeros((len(left), 8), np.uint32)
    result[:, :3] = np.asarray(lower, np.float32).view(np.uint32)
    result[:, 4:7] = np.asarray(upper, np.float32).view(np.uint32)
    seen = np.zeros(len(left), bool)
    pending = [(0, int(MAX_UINT))]
    while pending:
        index, successor = pending.pop()
        if not 0 <= index < len(left) or seen[index]:
            raise ValueError("TLAS contains a cycle or invalid child")
        seen[index] = True
        result[index, 7] = successor
        if instance[index] >= 0:
            if instance[index] >= 2**31:
                raise ValueError("too many instances")
            result[index, 3] = np.uint32(0x80000000 + int(instance[index]))
        else:
            result[index, 3] = left[index]
            pending.extend(((int(right[index]), successor), (int(left[index]), int(right[index]))))
    if not seen.all():
        raise ValueError("TLAS has unreachable nodes")
    return result


@dataclass
class ExportGroup:
    bvh: object
    escape: np.ndarray
    tlas: np.ndarray
    transforms: np.ndarray
    colors: np.ndarray
    scene_view: object


def prepare_groups(geometry, solid_colors=None):
    """Keep shared meshes; merge unique meshes into one ordinary world BVH."""
    from chroma_lar.triton_scene.instances import _build_instance_tlas, _padded_bounds
    from chroma_lar.triton_scene.primitive_adapter import instance_traversal_view
    from chroma.triton.primitives import Mesh, MeshInstance

    groups = _geometry_groups(geometry, solid_colors=solid_colors)
    shared, vertices, triangles, colors = [], [], [], []
    offset = 0
    for group in groups:
        rigid = np.allclose(
            group.rotations @ group.rotations.transpose(0, 2, 1), np.eye(3), atol=2e-6, rtol=0
        )
        if len(group.rotations) >= 4 and rigid:
            shared.append(group)
        else:
            for rotation, translation in zip(group.rotations, group.translations):
                vertices.append(group.vertices @ rotation.T + translation)
                triangles.append(group.triangles + offset)
                colors.append(group.colors)
                offset += len(group.vertices)
    if vertices:
        shared.insert(
            0,
            _MeshGroup(
                np.concatenate(vertices),
                np.concatenate(triangles),
                np.concatenate(colors),
                np.eye(3)[None],
                np.zeros((1, 3)),
            ),
        )
    result = []
    for group in shared:
        mesh = Mesh(group.vertices, group.triangles)
        instances = tuple(
            MeshInstance(str(i), mesh, r, t, 0)
            for i, (r, t) in enumerate(zip(group.rotations, group.translations))
        )
        view = instance_traversal_view(instances)
        lower, upper = _padded_bounds(
            group.vertices, view.instances.bounds_min, view.instances.bounds_max
        )
        lo, hi, left, right, inst, _ = _build_instance_tlas(lower, upper)
        bvh = build_packed_bvh(group.vertices, group.triangles)
        transforms = np.concatenate(
            (group.rotations.reshape(-1, 9), group.translations), axis=1
        ).astype(np.float32)
        result.append(
            ExportGroup(
                bvh,
                bvh_escape_links(bvh.nodes),
                tlas_threaded(lo, hi, left, right, inst),
                transforms,
                np.asarray(group.colors, np.uint32),
                view,
            )
        )
    return result


def pack_groups(groups):
    """One little-endian storage buffer; all offsets are measured in u32 words."""
    parts = [np.zeros(4 + GROUP_WORDS * len(groups), np.uint32)]
    size = len(parts[0])

    def append(array):
        nonlocal size
        words = np.ascontiguousarray(array).view(np.uint32).reshape(-1)
        start = size
        parts.append(words)
        size += len(words)
        return start

    for index, group in enumerate(groups):
        bvh = group.bvh
        header = parts[0][4 + index * GROUP_WORDS : 4 + (index + 1) * GROUP_WORDS]
        header[:6] = [
            append(bvh.nodes),
            append(bvh.triangle_vertices),
            append(group.colors),
            append(group.tlas),
            append(group.transforms),
            append(group.escape),
        ]
        header[6:9] = np.asarray(bvh.world_origin, np.float32).view(np.uint32)
        header[9] = np.asarray(bvh.world_scale, np.float32).view(np.uint32)
        header[10:14] = [bvh.node_count, bvh.triangle_count, len(group.tlas), len(group.transforms)]
    parts[0][:4] = [0x54524957, 1, len(groups), size]
    if size >= 2**32:
        raise ValueError("export exceeds 32-bit word addressing")
    return np.concatenate(parts).astype("<u4", copy=False)


def export_example(example, destination, *, name=None):
    """Write an immutable scene package; analytic wire scenes fail closed."""
    if example.boundary_layers or len(getattr(example.geometry, "wireplanes", ())):
        raise ValueError(
            "WebGPU analytic-wire precision is not validated. Use the Triton notebook for reflect3wires; no wires are omitted."
        )
    groups = prepare_groups(example.geometry, example.solid_colors)
    words = pack_groups(groups)
    name = example.metadata["name"] if name is None else name
    if not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("scene name must contain only letters, digits, underscores or hyphens")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    binary = name + ".bin"
    payload = words.tobytes()
    (destination / binary).write_bytes(payload)
    camera = example.camera
    manifest = dict(
        format="trichroma-webgpu-v1",
        name=name,
        binary=binary,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_length=len(payload),
        groups=len(groups),
        geometry=example.metadata,
        camera=dict(
            eye=list(camera.eye), target=list(camera.target), up=list(camera.up), fov=camera.fov
        ),
        rendering="opaque geometry camera rays; no optical photon transport",
        precision="WGSL f32 triangle queries; no analytic wires",
        mesh_triangles=sum(g.bvh.triangle_count for g in groups),
        instances=sum(len(g.transforms) for g in groups),
    )
    (destination / (name + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, groups


def copy_browser_assets(destination, scenes):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    assets = Path(__file__).with_name("assets")
    for filename in (
        "index.html",
        "viewer.js",
        "trace.wgsl",
        "physics.html",
        "physics.js",
        "physics.wgsl",
        "deposition.wgsl",
        "camera.wgsl",
        "camera.js",
        "camera.html",
        "theme.css",
        "fonts.css",
    ):
        shutil.copyfile(assets / filename, destination / filename)
    catalog = [dict(name=s["name"], manifest=s["name"] + ".json") for s in scenes]
    (destination / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
