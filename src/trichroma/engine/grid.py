"""Certified empty space for the bulk shortcut.

A uniform grid covers the scene. A cell is *occupied* if any triangle or
analytic wire plane may touch it (conservative bounding-box rasterization,
refined by a separating-axis triangle/box test for large triangles, and a
padded slab test for wire planes). Connected components of empty cells are
*certified* for one material by casting rays with the engine itself from
sample points: every ray must hit geometry, and every hit must give the same
incident material (Chroma's ``fill_state`` rule). Only certified cells are
used; each stores the largest empty box of its component that contains it,
grown face by face with a summed-volume table.

Inside such a box, Chroma's next ``fill_state`` would report that material
and a boundary farther away than the box exit, so absorption, scattering and
re-emission that happen before the (conservatively shrunk) exit are decided
exactly as Chroma decides them, without a geometry query.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EmptySpaceGrid:
    lower: np.ndarray  # float64 [3]
    cell: np.ndarray  # float64 [3]
    shape: tuple
    material: np.ndarray  # int32 [nx*ny*nz], -1 = not certified
    boxes: np.ndarray  # float32 [nx*ny*nz, 6] (lower xyz, upper xyz)
    certified_fraction: float
    components: int
    certified_components: int


def _cell_range(lo, hi, lower, cell, shape):
    i0 = np.floor((lo - lower) / cell).astype(np.int64)
    i1 = np.floor((hi - lower) / cell).astype(np.int64)
    upper = np.asarray(shape) - 1
    return np.clip(i0, 0, upper), np.clip(i1, 0, upper)


def _tri_box_overlap(v0, v1, v2, center, half):
    """Separating-axis test: triangles (N,3) each vs boxes (N,3) center/half."""
    v0, v1, v2 = v0 - center, v1 - center, v2 - center
    e0, e1, e2 = v1 - v0, v2 - v1, v0 - v2
    overlap = np.ones(len(center), bool)
    # 9 edge cross-product axes
    for e in (e0, e1, e2):
        for axis in range(3):
            a = np.zeros_like(e)
            a[:, (axis + 1) % 3] = -e[:, (axis + 2) % 3]
            a[:, (axis + 2) % 3] = e[:, (axis + 1) % 3]
            p0 = np.einsum("ij,ij->i", v0, a)
            p1 = np.einsum("ij,ij->i", v1, a)
            p2 = np.einsum("ij,ij->i", v2, a)
            r = np.einsum("ij,ij->i", half, np.abs(a))
            overlap &= ~((np.minimum(np.minimum(p0, p1), p2) > r) | (np.maximum(np.maximum(p0, p1), p2) < -r))
    # box face normals
    for axis in range(3):
        mn = np.minimum(np.minimum(v0[:, axis], v1[:, axis]), v2[:, axis])
        mx = np.maximum(np.maximum(v0[:, axis], v1[:, axis]), v2[:, axis])
        overlap &= ~((mn > half[:, axis]) | (mx < -half[:, axis]))
    # triangle normal
    n = np.cross(e0, e1)
    d = np.einsum("ij,ij->i", n, v0)
    r = np.einsum("ij,ij->i", half, np.abs(n))
    overlap &= np.abs(d) <= r
    return overlap


def _mark_triangles(occ, corners, lower, cell, pad):
    shape = occ.shape
    lo = corners.min(axis=1) - pad
    hi = corners.max(axis=1) + pad
    i0, i1 = _cell_range(lo, hi, lower, cell, shape)
    span = i1 - i0 + 1
    small = np.all(span <= 2, axis=1)
    # Small triangles: mark every cell of their (<= 2x2x2) bounding range.
    idx = np.flatnonzero(small)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ok = (dx < span[idx, 0]) & (dy < span[idx, 1]) & (dz < span[idx, 2])
                j = idx[ok]
                occ[i0[j, 0] + dx, i0[j, 1] + dy, i0[j, 2] + dz] = True
    # Large triangles: exact triangle/box overlap for every cell in range.
    half = 0.5 * cell + pad
    for t in np.flatnonzero(~small):
        gx, gy, gz = np.meshgrid(np.arange(i0[t, 0], i1[t, 0] + 1), np.arange(i0[t, 1], i1[t, 1] + 1),
                                 np.arange(i0[t, 2], i1[t, 2] + 1), indexing="ij")
        cells = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        centers = lower + (cells + 0.5) * cell
        k = len(cells)
        hit = _tri_box_overlap(np.repeat(corners[t, 0][None], k, 0), np.repeat(corners[t, 1][None], k, 0),
                               np.repeat(corners[t, 2][None], k, 0), centers, np.repeat(half[None], k, 0))
        c = cells[hit]
        occ[c[:, 0], c[:, 1], c[:, 2]] = True


def _mark_wires(occ, wires, lower, cell):
    """Occupy cells within the padded slab of every wire plane's wire set."""
    if not len(wires):
        return
    shape = occ.shape
    gx, gy, gz = np.meshgrid(*(lower[a] + (np.arange(shape[a]) + 0.5) * cell[a] for a in range(3)), indexing="ij")
    centers = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    half_diag = 0.5 * np.linalg.norm(cell)
    for rec in wires:
        o, u, v, n = rec[0:3], rec[3:6], rec[6:9], rec[9:12]
        pitch, radius, umin, umax, v0 = rec[12], rec[13], rec[14], rec[15], rec[16]
        kmin, kmax = rec[17:19].view(np.int32)
        w = centers - o
        near = np.abs(w @ n) <= radius + half_diag + 1e-3
        uu = w @ u
        vv = w @ v
        near &= (uu >= umin - half_diag - radius) & (uu <= umax + half_diag + radius)
        near &= (vv >= v0 + kmin * pitch - radius - half_diag) & (vv <= v0 + kmax * pitch + radius + half_diag)
        occ.reshape(-1)[near] = True


def _box_sum(sat, lo, hi):
    """Sum of a 3D array over inclusive boxes [lo, hi] given its padded SAT."""
    x0, y0, z0 = lo[:, 0], lo[:, 1], lo[:, 2]
    x1, y1, z1 = hi[:, 0] + 1, hi[:, 1] + 1, hi[:, 2] + 1
    return (sat[x1, y1, z1] - sat[x0, y1, z1] - sat[x1, y0, z1] - sat[x1, y1, z0]
            + sat[x0, y0, z1] + sat[x0, y1, z0] + sat[x1, y0, z0] - sat[x0, y0, z0])


def build_grid(engine, geometry, *, max_cells=1 << 21, samples_per_component=16, rays_per_sample=32,
               max_components=256, seed=12345):
    """Build and certify the empty-space grid for ``engine`` (a ProductionEngine)."""
    import torch
    from scipy import ndimage

    scene = engine.scene
    lower = np.asarray(scene.world_lower, np.float64)
    upper = np.asarray(scene.world_upper, np.float64)
    extent = np.maximum(upper - lower, 1e-3)
    cell_size = float(np.cbrt(np.prod(extent) / max_cells))
    shape = tuple(int(v) for v in np.clip(np.ceil(extent / cell_size), 1, 512))
    cell = extent / np.asarray(shape)
    occ = np.zeros(shape, bool)
    corners = np.asarray(geometry.mesh.vertices, np.float64)[np.asarray(geometry.mesh.triangles)]
    pad = 1e-3 + 4e-7 * np.max(np.abs(np.concatenate([lower, upper])))
    _mark_triangles(occ, corners, lower, cell, pad)
    _mark_wires(occ, scene.wires, lower, cell)

    labels, ncomp = ndimage.label(~occ)
    sizes = np.bincount(labels.ravel(), minlength=ncomp + 1)
    order = np.argsort(sizes[1:])[::-1][:max_components] + 1
    material_of = np.full(ncomp + 1, -1, np.int64)
    rng = np.random.default_rng(seed)
    flat_labels = labels.ravel()
    for comp in order:
        if sizes[comp] < 8:
            continue
        cells = np.flatnonzero(flat_labels == comp)
        picks = rng.choice(cells, size=min(samples_per_component, len(cells)), replace=False)
        idx = np.stack(np.unravel_index(picks, shape), axis=1)
        points = lower + (idx + rng.uniform(0.1, 0.9, idx.shape)) * cell
        origins = np.repeat(points, rays_per_sample, axis=0)
        d = rng.normal(size=(len(origins), 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        t, tri, normal, codes = engine.query(torch.from_numpy(origins.astype(np.float32)),
                                             torch.from_numpy(d.astype(np.float32)))
        tri = tri.cpu().numpy()
        if np.any(tri == -1):
            continue
        normal = normal.cpu().numpy()
        codes = codes.cpu().numpy()
        facing = np.einsum("ij,ij->i", normal, -d) > 0
        incident = np.where(facing, codes[:, 1], codes[:, 0])
        if np.all(incident == incident[0]):
            material_of[comp] = incident[0]
    cell_material = material_of[flat_labels].astype(np.int32)
    cell_material[flat_labels == 0] = -1

    # Largest boxes inside each certified component, per cell.
    boxes = np.zeros((len(flat_labels), 6), np.float32)
    certified = cell_material >= 0
    if certified.any():
        comp_mask = np.zeros(shape, bool)
        cheb = np.zeros(shape, np.int64)
        for comp in np.unique(flat_labels[certified]):
            mask = labels == comp
            cheb[mask] = ndimage.distance_transform_cdt(mask, metric="chessboard")[mask]
        comp_mask = certified.reshape(shape)
        cells = np.flatnonzero(certified)
        idx = np.stack(np.unravel_index(cells, shape), axis=1)
        r = cheb.ravel()[cells] - 1
        lab = flat_labels[cells]
        lo = np.maximum(idx - r[:, None], 0)
        hi = np.minimum(idx + r[:, None], np.asarray(shape) - 1)
        # Blocked = anything not in the same component: test via one SAT per component.
        for comp in np.unique(lab):
            sel = np.flatnonzero(lab == comp)
            blocked = (labels != comp).astype(np.int64)
            sat = np.zeros(tuple(s + 1 for s in shape), np.int64)
            sat[1:, 1:, 1:] = blocked.cumsum(0).cumsum(1).cumsum(2)
            for axis in range(3):
                for sign in (-1, 1):
                    l, h = lo[sel].copy(), hi[sel].copy()
                    bound = 0 if sign < 0 else shape[axis] - 1
                    room = (l[:, axis] - bound) if sign < 0 else (bound - h[:, axis])
                    good = np.zeros(len(sel), np.int64)
                    step = 1 << int(np.ceil(np.log2(max(1, room.max(initial=0)) + 1)))
                    while step:
                        trial = np.minimum(good + step, room)
                        tl_, th_ = l.copy(), h.copy()
                        if sign < 0:
                            tl_[:, axis] = l[:, axis] - trial
                        else:
                            th_[:, axis] = h[:, axis] + trial
                        ok = _box_sum(sat, tl_, th_) == 0
                        good = np.where(ok, trial, good)
                        step >>= 1
                    if sign < 0:
                        lo[sel, axis] = l[:, axis] - good
                    else:
                        hi[sel, axis] = h[:, axis] + good
        box_lo = lower + lo * cell
        box_hi = lower + (hi + 1) * cell
        margin = 1e-2 + 2e-6 * np.maximum(np.abs(box_lo), np.abs(box_hi))
        boxes[cells, 0:3] = (box_lo + margin).astype(np.float32)
        boxes[cells, 3:6] = (box_hi - margin).astype(np.float32)
        del comp_mask
    return EmptySpaceGrid(lower=lower, cell=cell, shape=shape, material=cell_material, boxes=boxes,
                          certified_fraction=float(certified.mean()), components=int(ncomp),
                          certified_components=int(np.count_nonzero(material_of[1:] >= 0)))
