"""Compile an arbitrary Chroma Geometry/Detector for the production engine.

Instancing is automatic: every placed solid becomes one instance of the
bottom-level structure (BLAS) built for its mesh in the mesh's own frame.
Solids whose meshes have identical vertices and triangles share one BLAS.
Material/surface codes are stored per *variant* (a distinct combination of
mesh and per-triangle material1/material2/surface indices), so repeated PMTs
cost one table. Global triangle ids are exactly those of
``Geometry.flatten()``: ``tri_offset[solid] + local_triangle``.

Analytic wire planes follow the installed Chroma ``geometry.wireplanes``
dictionaries (FP32 frame precomputed on the host as ``GPUGeometry`` does).
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from chroma.geometry import standard_wavelengths
from chroma.triton.optics import compile_optical_tables

from .bvh_build import build_threaded_bvh

# Packed node record: lower xyz, upper xyz, escape (int32 bits), leaf (int32
# bits: first*16+count for leaves, -1 for inner nodes).
NODE_WIDTH = 8
# Instance record (float32, ints stored as bit patterns):
# 0-8 world->local matrix M (row-major), 9-11 translation d (local =
# M @ (world - d)), 12 det sign of the placement, 13 BLAS root node, 14 global
# triangle offset, 15 code offset, 16 solid id, 17 BLAS triangle-slot offset.
INSTANCE_WIDTH = 20
# Wire-plane record (float32, ints as bit patterns): origin 0-2, u_norm 3-5,
# v_norm 6-8, n_norm 9-11, pitch 12, radius 13, umin 14, umax 15, v0 16,
# k_min 17, k_max 18, surface 19, material_inner 20, material_outer 21.
WIRE_WIDTH = 24
TRI_WIDTH = 9  # v0 xyz, e1 xyz, e2 xyz (local frame)


def _f32_bits(values):
    return np.asarray(values, dtype=np.int32).view(np.float32)


def _mesh_key(mesh):
    h = hashlib.sha1()
    v = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    t = np.ascontiguousarray(mesh.triangles, dtype=np.int64)
    h.update(str(v.shape).encode())
    h.update(v.tobytes())
    h.update(t.tobytes())
    return h.hexdigest()


@dataclass
class CompiledScene:
    """Host arrays for the production engine (see module docstring)."""

    nodes: np.ndarray  # float32 [M, NODE_WIDTH]; TLAS nodes first
    tlas_node_count: int
    instances: np.ndarray  # float32 [I, INSTANCE_WIDTH], TLAS leaf order
    tri_data: np.ndarray  # float32 [S, TRI_WIDTH], all BLAS triangle slots
    tri_local: np.ndarray  # int32 [S], local triangle id of every slot
    code_m1: np.ndarray  # int32 [C]: material1 index per variant triangle
    code_m2: np.ndarray  # int32 [C]
    code_surface: np.ndarray  # int32 [C], -1 for none
    wires: np.ndarray  # float32 [W, WIRE_WIDTH]
    solid_tri_offset: np.ndarray  # int64 [nsolids+1]
    solid_id_to_channel_index: np.ndarray  # int32 [nsolids]
    optics: object  # chroma.triton.optics.OpticalTableIR
    materials: list
    surfaces: list
    triangle_count: int
    world_lower: np.ndarray
    world_upper: np.ndarray
    blas_count: int
    variant_count: int


def _wire_records(geometry, materials, surfaces):
    """W's WirePlane records; appends wire-only materials/surfaces."""
    descs = getattr(geometry, "wireplanes", None) or []
    records = []
    for desc in descs:
        surface = desc.get("surface", None)
        inner = desc.get("material_inner", None)
        outer = desc.get("material_outer", None)
        for material in (inner, outer):
            if material is not None and material not in materials:
                materials.append(material)
        if surface is not None and surface not in surfaces:
            surfaces.append(surface)
        surface_idx = -1 if surface is None else surfaces.index(surface)
        if inner is None or outer is None:
            inner_idx = outer_idx = 0
        else:
            inner_idx, outer_idx = materials.index(inner), materials.index(outer)
        u_raw = np.asarray(desc["u"], dtype=np.float32)
        v_raw = np.asarray(desc["v"], dtype=np.float32)
        u_norm = u_raw / np.linalg.norm(u_raw)
        v_orth = v_raw - np.dot(v_raw, u_norm) * u_norm
        v_norm = v_orth / np.linalg.norm(v_orth)
        n_norm = np.cross(u_norm, v_norm)
        pitch = float(np.float32(desc["pitch"]))
        v0 = float(np.float32(desc["v0"]))
        vmin = float(np.float32(desc["vmin"]))
        vmax = float(np.float32(desc["vmax"]))
        k_min = int(np.ceil((vmin - v0) / pitch)) if pitch > 0 else 0
        k_max = int(np.floor((vmax - v0) / pitch)) if pitch > 0 else 0
        rec = np.zeros(WIRE_WIDTH, np.float32)
        rec[0:3] = np.asarray(desc["origin"], dtype=np.float32)
        rec[3:6] = u_norm.astype(np.float32)
        rec[6:9] = v_norm.astype(np.float32)
        rec[9:12] = n_norm.astype(np.float32)
        rec[12] = np.float32(desc["pitch"])
        rec[13] = np.float32(desc["radius"])
        rec[14] = np.float32(desc["umin"])
        rec[15] = np.float32(desc["umax"])
        rec[16] = np.float32(desc["v0"])
        rec[17:22] = _f32_bits([k_min, k_max, surface_idx, inner_idx, outer_idx])
        records.append(rec)
    return np.stack(records) if records else np.zeros((0, WIRE_WIDTH), np.float32)


def compile_scene(geometry, *, wavelengths=None, leaf_size=4):
    """Compile ``geometry`` (flattened in place if needed, like Chroma does)."""
    if not hasattr(geometry, "mesh"):
        geometry.flatten()
    solids = geometry.solids
    if not solids:
        raise ValueError("geometry has no solids")
    rotations = geometry.solid_rotations
    displacements = geometry.solid_displacements
    counts = np.array([len(s.mesh.triangles) for s in solids], np.int64)
    tri_offset = np.concatenate([[0], np.cumsum(counts)])
    triangle_count = int(tri_offset[-1])
    if triangle_count != len(geometry.mesh.triangles):
        raise ValueError("flattened mesh does not match the solid list")

    materials = list(geometry.unique_materials)
    surfaces = list(geometry.unique_surfaces)
    m1_all = np.asarray(geometry.material1_index, np.int32)
    m2_all = np.asarray(geometry.material2_index, np.int32)
    surf_all = np.asarray(geometry.surface_index, np.int32)

    # --- unique meshes (BLAS) and code variants -------------------------
    blas_of_mesh = {}  # id(mesh) -> blas index
    blas_by_key = {}  # content key -> blas index
    blas_meshes = []
    variant_of = {}  # (blas, id(solid)) -> variant
    variant_by_key = {}
    variant_codes = []
    solid_blas = np.empty(len(solids), np.int64)
    solid_variant = np.empty(len(solids), np.int64)
    for i, solid in enumerate(solids):
        mid = id(solid.mesh)
        if mid not in blas_of_mesh:
            key = _mesh_key(solid.mesh)
            if key not in blas_by_key:
                blas_by_key[key] = len(blas_meshes)
                blas_meshes.append(solid.mesh)
            blas_of_mesh[mid] = blas_by_key[key]
        b = blas_of_mesh[mid]
        solid_blas[i] = b
        vid_key = (b, id(solid))
        if vid_key not in variant_of:
            rows = slice(tri_offset[i], tri_offset[i + 1])
            codes = (m1_all[rows], m2_all[rows], surf_all[rows])
            h = hashlib.sha1(b"%d" % b)
            for c in codes:
                h.update(c.tobytes())
            key = h.hexdigest()
            if key not in variant_by_key:
                variant_by_key[key] = len(variant_codes)
                variant_codes.append(codes)
            variant_of[vid_key] = variant_by_key[key]
        solid_variant[i] = variant_of[vid_key]

    variant_offset = np.concatenate([[0], np.cumsum([len(c[0]) for c in variant_codes])])
    code_m1 = np.concatenate([c[0] for c in variant_codes]).astype(np.int32)
    code_m2 = np.concatenate([c[1] for c in variant_codes]).astype(np.int32)
    code_surface = np.concatenate([c[2] for c in variant_codes]).astype(np.int32)

    # --- BLAS construction --------------------------------------------------
    blas_nodes, blas_tri, blas_local, blas_root, blas_slot0 = [], [], [], [], []
    node_cursor = 0
    slot_cursor = 0
    for mesh in blas_meshes:
        v = np.asarray(mesh.vertices, np.float64)
        t = np.asarray(mesh.triangles, np.int64)
        corners = v[t]  # [T,3,3]
        bvh = build_threaded_bvh(corners.min(axis=1), corners.max(axis=1), leaf_size=leaf_size)
        packed = np.zeros((bvh.node_count, NODE_WIDTH), np.float32)
        packed[:, 0:3] = bvh.lower
        packed[:, 3:6] = bvh.upper
        escape = np.where(bvh.escape >= 0, bvh.escape + node_cursor, -1)
        leaf = np.where(bvh.leaf_first >= 0, (bvh.leaf_first + slot_cursor) * 16 + bvh.leaf_count, -1)
        packed[:, 6] = _f32_bits(escape)
        packed[:, 7] = _f32_bits(leaf)
        blas_nodes.append(packed)
        tri = corners[bvh.order].astype(np.float32)
        slots = np.empty((len(bvh.order), TRI_WIDTH), np.float32)
        slots[:, 0:3] = tri[:, 0]
        slots[:, 3:6] = tri[:, 1] - tri[:, 0]
        slots[:, 6:9] = tri[:, 2] - tri[:, 0]
        blas_tri.append(slots)
        blas_local.append(bvh.order.astype(np.int32))
        blas_root.append(node_cursor)
        blas_slot0.append(slot_cursor)
        node_cursor += bvh.node_count
        slot_cursor += len(bvh.order)

    # --- instances and TLAS ---------------------------------------------------
    inst_lower = np.empty((len(solids), 3))
    inst_upper = np.empty((len(solids), 3))
    records = np.zeros((len(solids), INSTANCE_WIDTH), np.float32)
    ints = np.zeros((len(solids), 5), np.int64)
    blas_bounds = []
    for mesh in blas_meshes:
        v = np.asarray(mesh.vertices, np.float64)
        blas_bounds.append((v.min(axis=0), v.max(axis=0)))
    for i in range(len(solids)):
        r = np.asarray(rotations[i], np.float64)
        d = np.asarray(displacements[i], np.float64)
        det = np.linalg.det(r)
        if abs(det) < 1e-12:
            raise ValueError("solid %d has a singular rotation matrix" % i)
        m = np.linalg.inv(r)
        lo, hi = blas_bounds[solid_blas[i]]
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        world = corners @ r.T + d
        # Pad by a relative float32 margin: the transformed local ray and the
        # flattened vertices round differently.
        pad = 1e-5 * (1.0 + np.abs(world).max())
        inst_lower[i] = world.min(axis=0) - pad
        inst_upper[i] = world.max(axis=0) + pad
        records[i, 0:9] = m.reshape(-1).astype(np.float32)
        records[i, 9:12] = d.astype(np.float32)
        records[i, 12] = np.float32(1.0 if det > 0 else -1.0)
        ints[i] = (blas_root[solid_blas[i]], tri_offset[i],
                   variant_offset[solid_variant[i]], i, blas_slot0[solid_blas[i]])
    tlas = build_threaded_bvh(inst_lower, inst_upper, leaf_size=1)
    tlas_packed = np.zeros((tlas.node_count, NODE_WIDTH), np.float32)
    tlas_packed[:, 0:3] = tlas.lower
    tlas_packed[:, 3:6] = tlas.upper
    tlas_packed[:, 6] = _f32_bits(tlas.escape)
    tlas_packed[:, 7] = _f32_bits(np.where(tlas.leaf_first >= 0, tlas.leaf_first * 16 + tlas.leaf_count, -1))
    # BLAS node indices shift by the TLAS size.
    shift = tlas.node_count
    for packed in blas_nodes:
        esc = packed[:, 6].view(np.int32)
        packed[:, 6] = _f32_bits(np.where(esc >= 0, esc + shift, -1))
    ints[:, 0] += shift
    order = tlas.order
    records = records[order]
    ints = ints[order]
    records[:, 13:18] = _f32_bits(ints.astype(np.int32))
    nodes = np.concatenate([tlas_packed] + blas_nodes)

    wires = _wire_records(geometry, materials, surfaces)
    grid = standard_wavelengths if wavelengths is None else wavelengths
    optics = compile_optical_tables(materials, surfaces, wavelengths=grid)

    if hasattr(geometry, "solid_id_to_channel_index"):
        channel = np.asarray(geometry.solid_id_to_channel_index, np.int32)
    else:
        channel = np.full(len(solids), -1, np.int32)

    world_lower = inst_lower.min(axis=0)
    world_upper = inst_upper.max(axis=0)
    return CompiledScene(
        nodes=nodes,
        tlas_node_count=int(tlas.node_count),
        instances=records,
        tri_data=np.concatenate(blas_tri),
        tri_local=np.concatenate(blas_local),
        code_m1=code_m1,
        code_m2=code_m2,
        code_surface=code_surface,
        wires=wires,
        solid_tri_offset=tri_offset,
        solid_id_to_channel_index=channel,
        optics=optics,
        materials=materials,
        surfaces=surfaces,
        triangle_count=triangle_count,
        world_lower=world_lower,
        world_upper=world_upper,
        blas_count=len(blas_meshes),
        variant_count=len(variant_codes),
    )
