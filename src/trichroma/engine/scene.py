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
from trichroma.engine.optics import compile_optical_tables

from .bvh_build import build_sah_tree, build_threaded_bvh, thread_octants

# Packed node record: lower xyz, upper xyz, escape (int32 bits), leaf (int32
# bits: first*16+count for leaves, -1 for inner nodes).
NODE_WIDTH = 8
# Instance record (float32, ints stored as bit patterns):
# 0-8 world->local matrix M (row-major), 9-11 translation d (local =
# M @ (world - d)), 12 det sign of the placement, 13 BLAS root node (octant-0
# copy), 14 global triangle offset, 15 code offset, 16 solid id, 17 BLAS
# triangle-slot offset, 18 BLAS nodes per octant copy.
INSTANCE_WIDTH = 20
# Every tree is stored as eight threaded copies, one per ray-direction octant
# (bit a set when direction component a is negative), near child first: the
# copy for octant o starts at root + o * (nodes per copy).
OCTANTS = 8
# Wire-plane record (float32, ints as bit patterns): origin 0-2, u_norm 3-5,
# v_norm 6-8, n_norm 9-11, pitch 12, radius 13, umin 14, umax 15, v0 16,
# k_min 17, k_max 18, surface 19, material_inner 20, material_outer 21.
WIRE_WIDTH = 24
TRI_WIDTH = 9  # v0 xyz, e1 xyz, e2 xyz (local frame)
# Analytic box record (float32; ints as bits): lower xyz 0-2, upper xyz 3-5,
# then for faces (-x,+x,-y,+y,-z,+z): first box-triangle slot and count 6-17,
# the solid's global triangle id range [first, end) 18-19.
BOX_WIDTH = 20
# Box triangle record (float32; ints as bits): world v0/e1/e2 0-8, global
# triangle id 9, material1 10, material2 11, surface 12; and, in the row at
# (box's first slot + k), the face of the box triangle with local id k 13.
BOX_TRI_WIDTH = 16
MAX_GLOBAL_BOXES = 16


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
    optics: object  # trichroma.engine.optics.OpticalTableIR
    materials: list
    surfaces: list
    triangle_count: int
    world_lower: np.ndarray
    world_upper: np.ndarray
    blas_count: int
    variant_count: int
    boxes: np.ndarray  # float32 [B, BOX_WIDTH], analytic boxes tested before the TLAS
    box_tris: np.ndarray  # float32 [S, BOX_TRI_WIDTH]
    box_solids: tuple  # solid ids represented by ``boxes``


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



def detect_box(world_tri, tol=1e-4):
    """Return per-face triangle lists if ``world_tri`` [T,3,3] tiles an axis-aligned box.

    Every triangle must lie in one face plane of the mesh bounding box (within
    ``tol`` mm) and each face's triangle area must equal the face area.
    Returns (lower, upper, faces) with faces a list of 6 index arrays, or None.
    """
    lo = world_tri.reshape(-1, 3).min(axis=0)
    hi = world_tri.reshape(-1, 3).max(axis=0)
    if np.any(hi - lo <= tol):
        return None
    faces = [[] for _ in range(6)]
    for t, tri in enumerate(world_tri):
        placed = False
        for axis in range(3):
            for side, plane in ((0, lo[axis]), (1, hi[axis])):
                if np.all(np.abs(tri[:, axis] - plane) <= tol):
                    faces[2 * axis + side].append(t)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None
    for f in range(6):
        axis = f // 2
        others = [a for a in range(3) if a != axis]
        if not faces[f]:
            return None
        tris = world_tri[faces[f]][:, :, others]
        area = 0.5 * np.abs((tris[:, 1, 0] - tris[:, 0, 0]) * (tris[:, 2, 1] - tris[:, 0, 1])
                            - (tris[:, 2, 0] - tris[:, 0, 0]) * (tris[:, 1, 1] - tris[:, 0, 1])).sum()
        face_area = (hi[others[0]] - lo[others[0]]) * (hi[others[1]] - lo[others[1]])
        if abs(area - face_area) > 1e-6 * face_area + tol:
            return None
    return lo, hi, [np.asarray(f, np.int64) for f in faces]


def _pack_octants(tree, node_base, slot_base):
    """NODE_WIDTH records of ``tree``'s eight octant copies (float32 bounds
    rounded outward), escapes absolute from ``node_base``, leaves pointing at
    triangle slots from ``slot_base``."""
    m = len(tree.left)
    lower = np.asarray(tree.lower, np.float64)
    upper = np.asarray(tree.upper, np.float64)
    lo32 = lower.astype(np.float32)
    hi32 = upper.astype(np.float32)
    lo32 = np.where(lo32.astype(np.float64) > lower, np.nextafter(lo32, np.float32(-np.inf)), lo32)
    hi32 = np.where(hi32.astype(np.float64) < upper, np.nextafter(hi32, np.float32(np.inf)), hi32)
    is_leaf = tree.left < 0
    count = tree.end - tree.begin
    if np.any(count[is_leaf] > 15):
        raise ValueError("leaves hold at most 15 primitives")
    leaf = np.where(is_leaf, (tree.begin + slot_base) * 16 + count, -1)
    packed = np.zeros((OCTANTS * m, NODE_WIDTH), np.float32)
    for octant, (row, escape) in enumerate(thread_octants(tree)):
        base = octant * m
        dest = base + row
        packed[dest, 0:3] = lo32
        packed[dest, 3:6] = hi32
        packed[dest, 6] = _f32_bits(np.where(escape >= 0, escape + base + node_base, -1))
        packed[dest, 7] = _f32_bits(leaf)
    return packed


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

    # --- analytic boxes: the largest box-shaped solids ------------------
    world_vertices = np.asarray(geometry.mesh.vertices, np.float64)
    world_triangles = np.asarray(geometry.mesh.triangles, np.int64)
    candidates = []
    for i, solid in enumerate(solids):
        if counts[i] > 64:
            continue
        rows = slice(tri_offset[i], tri_offset[i + 1])
        found = detect_box(world_vertices[world_triangles[rows]])
        if found is not None:
            lo_b, hi_b, faces = found
            ext = hi_b - lo_b
            candidates.append((2 * (ext[0] * ext[1] + ext[1] * ext[2] + ext[0] * ext[2]), i, lo_b, hi_b, faces))
    candidates.sort(key=lambda c: -c[0])
    # Keep the largest boxes; test the smallest first: nested boxes (an inner
    # detector inside a cavity) then prune the outer ones.
    boxes, box_tris, box_solids = [], [], []
    for _, i, lo_b, hi_b, faces in sorted(candidates[:MAX_GLOBAL_BOXES], key=lambda c: c[0]):
        rec = np.zeros(BOX_WIDTH, np.float32)
        rec[0:3] = lo_b.astype(np.float32)
        rec[3:6] = hi_b.astype(np.float32)
        ints = []
        for f in range(6):
            ints += [len(box_tris), len(faces[f])]
            for t in faces[f]:
                g = tri_offset[i] + t
                v = world_vertices[world_triangles[g]].astype(np.float32)
                row = np.zeros(BOX_TRI_WIDTH, np.float32)
                row[0:3] = v[0]
                row[3:6] = v[1] - v[0]
                row[6:9] = v[2] - v[0]
                row[9:13] = _f32_bits([g, m1_all[g], m2_all[g], surf_all[g]])
                box_tris.append(row)
        rec[6:18] = _f32_bits(ints)
        rec[18:20] = _f32_bits([tri_offset[i], tri_offset[i + 1]])
        first_slot = ints[0]
        for f in range(6):
            for t in faces[f]:
                box_tris[first_slot + t][13] = _f32_bits([f])[0]
        boxes.append(rec)
        box_solids.append(i)
    box_set = set(box_solids)

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
    blas_nodes, blas_tri, blas_local, blas_root, blas_slot0, blas_stride = [], [], [], [], [], []
    node_cursor = 0
    slot_cursor = 0
    for mesh in blas_meshes:
        v = np.asarray(mesh.vertices, np.float64)
        t = np.asarray(mesh.triangles, np.int64)
        corners = v[t]  # [T,3,3]
        tree = build_sah_tree(corners.min(axis=1), corners.max(axis=1), leaf_size=leaf_size)
        packed = _pack_octants(tree, node_cursor, slot_cursor)
        blas_nodes.append(packed)
        tri = corners[tree.order].astype(np.float32)
        slots = np.empty((len(tree.order), TRI_WIDTH), np.float32)
        slots[:, 0:3] = tri[:, 0]
        slots[:, 3:6] = tri[:, 1] - tri[:, 0]
        slots[:, 6:9] = tri[:, 2] - tri[:, 0]
        blas_tri.append(slots)
        blas_local.append(tree.order.astype(np.int32))
        blas_root.append(node_cursor)
        blas_slot0.append(slot_cursor)
        blas_stride.append(len(tree.left))
        node_cursor += len(packed)
        slot_cursor += len(tree.order)

    # --- instances and TLAS ---------------------------------------------------
    inst_lower = np.empty((len(solids), 3))
    inst_upper = np.empty((len(solids), 3))
    records = np.zeros((len(solids), INSTANCE_WIDTH), np.float32)
    ints = np.zeros((len(solids), 6), np.int64)
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
                   variant_offset[solid_variant[i]], i, blas_slot0[solid_blas[i]], blas_stride[solid_blas[i]])
    world_lower = inst_lower.min(axis=0)
    world_upper = inst_upper.max(axis=0)
    keep = np.array([i for i in range(len(solids)) if i not in box_set], np.int64)
    if len(keep) == 0:
        keep = np.array([box_solids[0]], np.int64)  # keep one instance so the TLAS is never empty
    inst_lower, inst_upper = inst_lower[keep], inst_upper[keep]
    records, ints = records[keep], ints[keep]
    tlas = build_sah_tree(inst_lower, inst_upper, leaf_size=1)
    tlas_packed = _pack_octants(tlas, 0, 0)
    tlas_count = len(tlas.left)
    # BLAS node indices shift by the TLAS copies.
    shift = len(tlas_packed)
    for packed in blas_nodes:
        esc = packed[:, 6].view(np.int32)
        packed[:, 6] = _f32_bits(np.where(esc >= 0, esc + shift, -1))
    ints[:, 0] += shift
    order = tlas.order
    records = records[order]
    ints = ints[order]
    records[:, 13:19] = _f32_bits(ints.astype(np.int32))
    nodes = np.concatenate([tlas_packed] + blas_nodes)

    wires = _wire_records(geometry, materials, surfaces)
    grid = standard_wavelengths if wavelengths is None else wavelengths
    optics = compile_optical_tables(materials, surfaces, wavelengths=grid)

    if hasattr(geometry, "solid_id_to_channel_index"):
        channel = np.asarray(geometry.solid_id_to_channel_index, np.int32)
    else:
        channel = np.full(len(solids), -1, np.int32)

    return CompiledScene(
        nodes=nodes,
        tlas_node_count=int(tlas_count),
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
        boxes=np.stack(boxes) if boxes else np.zeros((0, BOX_WIDTH), np.float32),
        box_tris=np.stack(box_tris) if box_tris else np.zeros((0, BOX_TRI_WIDTH), np.float32),
        box_solids=tuple(box_solids),
    )
