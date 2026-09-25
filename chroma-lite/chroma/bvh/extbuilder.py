import importlib
import os
from pathlib import Path

import numpy as np
from pycuda.gpuarray import vec

from chroma.bvh.bvh import BVH, WorldCoords, CHILD_BITS
from chroma.gpu.bvh import merge_nodes_detailed, concatenate_layers, collapse_chains

MAX_CHILD = 2 ** (32 - CHILD_BITS) - 1


def _spread3_16(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    x = (x | (x << 16)) & np.uint64(0x00000000FF0000FF)
    x = (x | (x << 8)) & np.uint64(0x000000F00F00F00F)
    x = (x | (x << 4)) & np.uint64(0x00000C30C30C30C3)
    x = (x | (x << 2)) & np.uint64(0x0000249249249249)
    return x


def _quantize_lower(lower: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        scale = 1.0
    q = ((lower - origin) / scale).astype(np.int64)
    q = np.maximum(q - 1, 0)
    return np.clip(q, 0, 0xFFFF).astype(np.uint32)


def _quantize_upper(upper: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        scale = 1.0
    q = ((upper - origin) / scale).astype(np.int64) + 1
    return np.clip(q, 0, 0xFFFF).astype(np.uint32)


def _compute_morton(centroids: np.ndarray, origin: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        scale = 1.0
    q = ((centroids - origin) / scale).astype(np.int64)
    q = np.clip(q, 0, 0xFFFF).astype(np.uint32)
    return (
        _spread3_16(q[:, 0]) |
        (_spread3_16(q[:, 1]) << 1) |
        (_spread3_16(q[:, 2]) << 2)
    )


def _count_unique_in_sorted(a: np.ndarray) -> int:
    if len(a) == 0:
        return 0
    return int((np.ediff1d(a) > 0).sum() + 1)
from tqdm import trange

try:  # PyCUDA vector dtype may not be hashable when imported lazily
    uint4_dtype = vec.uint4
except Exception as exc:  # pragma: no cover - defensive
    raise RuntimeError("Unable to import PyCUDA vector dtype") from exc

_MODULE_NAME = "chroma.bvh._ext_bvh_builder"
_THIS_DIR = Path(__file__).resolve().parent
_CPP_SOURCE = _THIS_DIR / "_ext_bvh_builder.cpp"


def _load_native() -> object:
    try:
        return importlib.import_module(_MODULE_NAME)
    except ImportError:
        pass

    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from setuptools import Distribution

    build_temp = _THIS_DIR / "_build_ext"
    build_temp.mkdir(exist_ok=True)

    include_dir = (Path(__file__).resolve().parents[2] / "ext" / "bvh" / "src").resolve()

    ext = Pybind11Extension(
        name=_MODULE_NAME,
        sources=[str(_CPP_SOURCE)],
        include_dirs=[str(include_dir)],
        extra_compile_args=["-std=c++20", "-O3"],
    )

    dist = Distribution({"ext_modules": [ext]})
    cmd = build_ext(dist)
    cmd.build_lib = str(_THIS_DIR.parent.parent)
    cmd.build_temp = str(build_temp)
    cmd.ensure_finalized()
    cmd.run()

    return importlib.import_module(_MODULE_NAME)


_native = _load_native()



def make_ext_bvh(mesh, quality: str = "high", target_degree: int = 3, verbose: bool = False) -> BVH:
    """Build a BVH using ext/bvh for leaf construction and the existing GPU pipeline for parents."""

    quality = quality.lower()
    if quality not in {"low", "medium", "high"}:
        raise ValueError("quality must be 'low', 'medium', or 'high'")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.uint32)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("mesh.vertices must be shaped (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("mesh.triangles must be shaped (M, 3)")

    bounds, first_ids, prim_counts, prim_ids = _native.build(vertices, triangles, quality)

    bounds = np.asarray(bounds, dtype=np.float32)
    first_ids = np.asarray(first_ids, dtype=np.uint32)
    prim_counts = np.asarray(prim_counts, dtype=np.uint32)
    prim_ids = np.asarray(prim_ids, dtype=np.uint32)

    leaf_mask = prim_counts > 0
    if not np.any(leaf_mask):
        raise RuntimeError("ext/bvh builder returned no leaves")

    leaf_bounds = bounds[leaf_mask]
    leaf_first = first_ids[leaf_mask]
    leaf_tris = prim_ids[leaf_first]

    world_origin = vertices.min(axis=0)
    world_scale = float(np.max(vertices.max(axis=0) - world_origin))
    if world_scale == 0:
        world_scale = 1.0
    else:
        world_scale /= (2**16 - 2)

    lower = leaf_bounds[:, [0, 2, 4]]
    upper = leaf_bounds[:, [1, 3, 5]]
    centroids = 0.5 * (lower + upper)

    q_lower = _quantize_lower(lower, world_origin, world_scale)
    q_upper = _quantize_upper(upper, world_origin, world_scale)

    leaf_dtype = np.dtype([('x', np.uint32), ('y', np.uint32), ('z', np.uint32), ('w', np.uint32)])
    leaf_nodes = np.zeros(len(leaf_tris), dtype=leaf_dtype)
    leaf_nodes['x'] = (q_upper[:, 0] << 16) | q_lower[:, 0]
    leaf_nodes['y'] = (q_upper[:, 1] << 16) | q_lower[:, 1]
    leaf_nodes['z'] = (q_upper[:, 2] << 16) | q_lower[:, 2]
    leaf_nodes['w'] = leaf_tris.astype(np.uint32)

    morton_codes = _compute_morton(centroids, world_origin, world_scale)
    argsort = morton_codes.argsort()
    leaf_nodes = leaf_nodes[argsort]
    morton_codes = morton_codes[argsort]

    layers = [leaf_nodes]
    while len(layers[0]) > 1:
        top_layer = layers[0]
        nnodes = len(top_layer)

        nunique = _count_unique_in_sorted(morton_codes)
        while nnodes / float(nunique) < target_degree and nunique > 1:
            morton_codes >>= 1
            nunique = _count_unique_in_sorted(morton_codes)

        morton_delta = np.ediff1d(morton_codes, to_begin=np.uint64(1)).astype(np.uint64)
        parent_morton_codes = morton_codes[morton_delta > 0]
        first_child = np.argwhere(morton_delta > 0).flatten().astype(np.uint32)
        nchild = np.ediff1d(first_child, to_end=nnodes - first_child[-1]).astype(np.uint32)

        excess_children = np.argwhere(nchild > MAX_CHILD).flatten()
        if len(excess_children) > 0:
            parent_parts = np.split(parent_morton_codes, excess_children)
            first_parts = np.split(first_child, excess_children)
            nchild_parts = np.split(nchild, excess_children)

            new_parent_parts = parent_parts[:1]
            new_first_parts = first_parts[:1]

            for morton_part, first_part, nchild_part in zip(parent_parts[1:], first_parts[1:], nchild_parts[1:]):
                extra_first = np.arange(first_part[0], first_part[0] + nchild_part[0], MAX_CHILD).astype(first_part.dtype)
                new_first_parts.extend([extra_first, first_part[1:]])
                new_parent_parts.extend([np.repeat(morton_part[0], len(extra_first)), morton_part[1:]])

            parent_morton_codes = np.concatenate([p.astype(np.uint64) for p in new_parent_parts])
            first_child = np.concatenate(new_first_parts)
            nchild = np.ediff1d(first_child, to_end=nnodes - first_child[-1]).astype(np.uint32)

        if nunique > 1:
            plural = 's'
        else:
            plural = ''
        if verbose:
            print('Merging %d nodes to %d parent%s' % (nnodes, len(parent_morton_codes), plural))

        parents = merge_nodes_detailed(top_layer, first_child, nchild)
        layers = [parents] + layers
        morton_codes = parent_morton_codes

    nodes, layer_bounds = concatenate_layers(layers)
    nodes = collapse_chains(nodes, layer_bounds)
    world_coords = WorldCoords(world_origin=world_origin, world_scale=world_scale)

    return BVH(world_coords, nodes, layer_bounds[:-1])
