"""
Pixel plane geometry for pixel LArTPC.

Builds a box mesh with gold pixel pads (chamfered-square octagons) on the
+/- X faces and plain steel faces on +/- Y / +/- Z.  The pixel array is centerd on
each +/- X face; any remaining border area is triangulated as steel.
"""

import numpy as np
from chroma.geometry import Mesh, Surface

# 20 triangles per cell, each referencing 3 of 13 local vertex slots:
#   0=c00(BL)  1=c10(BR)  2=c11(TR)  3=c01(TL)
#   4..11 = octagon v0..v7          12 = center
_TRI_TEMPLATE = np.array([
    # pad fan (8): center → v[k] → v[k+1]
    [12,  4,  5], [12,  5,  6], [12,  6,  7], [12,  7,  8],
    [12,  8,  9], [12,  9, 10], [12, 10, 11], [12, 11,  4],
    # corner surround (4)
    [ 2,  5,  4], [ 3,  7,  6], [ 0,  9,  8], [ 1, 11, 10],
    # side surround (8)
    [ 1,  2,  4], [ 1,  4, 11],
    [ 2,  3,  6], [ 2,  6,  5],
    [ 3,  0,  8], [ 3,  8,  7],
    [ 0,  1, 10], [ 0, 10,  9],
], dtype=np.int32)  # (20, 3)

_TRI_TEMPLATE_FLIP = _TRI_TEMPLATE[:, [0, 2, 1]]  # reversed winding for -X
_IS_PAD_CELL = np.array([True] * 8 + [False] * 12, dtype=bool)

# Octagon vertex offsets (CCW in YZ ⇒ +X outward normal)
_OFF_Y = np.array([+1, +1, -1, -1, -1, -1, +1, +1], dtype=np.float64)
_OFF_Z = np.array([+1, +1, +1, +1, -1, -1, -1, -1], dtype=np.float64)
_CHAMFER_SIGN_Y = np.array([0, -1, +1, 0, 0, +1, -1, 0], dtype=np.float64)
_CHAMFER_SIGN_Z = np.array([-1, 0, 0, -1, +1, 0, 0, +1], dtype=np.float64)


def make_pixel_face(x_pos, y_range, z_range, n_pixels_y, n_pixels_z,
                    pitch, pad_size, chamfer_radius, face_normal=1):
    """
    Create a triangulated pixel face mesh at *x = x_pos*.

    The pixel array is **centered** on the face.  Any remaining border area
    between the pixel array and the face edges is triangulated as plain
    steel-surface rectangles.

    Each pixel cell is decomposed into:
      - 8 gold pad triangles (octagonal pad, fan from center)
      - 12 FR-4 surround triangles (4 corner + 4 side quads)

    Parameters
    ----------
    x_pos : float
        X coordinate of the face plane.
    y_range, z_range : tuple of (min, max)
        Full extent of the face in Y and Z.
    n_pixels_y, n_pixels_z : int
        Number of pixels along Y and Z.
    pitch, pad_size, chamfer_radius : float
        Pixel geometry parameters (mm).
    face_normal : int
        +1 for outward normal in +X, -1 for -X.

    Returns
    -------
    vertices : (N, 3) float32
    triangles : (M, 3) int32
    is_pad : (M,) bool — True for gold pad triangles
    is_border : (M,) bool — True for steel border triangles
    """
    ny, nz = n_pixels_y, n_pixels_z
    n_cells = ny * nz
    half = pad_size / 2.0
    c = chamfer_radius

    y_min, y_max = y_range
    z_min, z_max = z_range

    # center the pixel array on the face
    pixel_wy = ny * pitch
    pixel_wz = nz * pitch
    py_min = (y_min + y_max - pixel_wy) * 0.5
    pz_min = (z_min + z_max - pixel_wz) * 0.5
    py_max = py_min + pixel_wy
    pz_max = pz_min + pixel_wz

    # 1) corner grid  (ny+1)×(nz+1)
    gy = py_min + np.arange(ny + 1) * pitch
    gz = pz_min + np.arange(nz + 1) * pitch
    gy_g, gz_g = np.meshgrid(gy, gz, indexing="ij")
    n_corners = (ny + 1) * (nz + 1)
    corner_verts = np.empty((n_corners, 3), dtype=np.float32)
    corner_verts[:, 0] = x_pos
    corner_verts[:, 1] = gy_g.ravel()
    corner_verts[:, 2] = gz_g.ravel()

    # 2) cell centers
    cy_arr = py_min + (np.arange(ny) + 0.5) * pitch
    cz_arr = pz_min + (np.arange(nz) + 0.5) * pitch
    cy_g, cz_g = np.meshgrid(cy_arr, cz_arr, indexing="ij")
    cy_f = cy_g.ravel()
    cz_f = cz_g.ravel()

    # 3) octagon vertices  n_cells × 8
    off_y = half * _OFF_Y + c * _CHAMFER_SIGN_Y
    off_z = half * _OFF_Z + c * _CHAMFER_SIGN_Z
    oct_verts = np.empty((n_cells, 8, 3), dtype=np.float32)
    oct_verts[:, :, 0] = x_pos
    oct_verts[:, :, 1] = cy_f[:, None] + off_y[None, :]
    oct_verts[:, :, 2] = cz_f[:, None] + off_z[None, :]
    oct_verts = oct_verts.reshape(-1, 3)

    # 4) center vertices
    cen_verts = np.empty((n_cells, 3), dtype=np.float32)
    cen_verts[:, 0] = x_pos
    cen_verts[:, 1] = cy_f
    cen_verts[:, 2] = cz_f

    pixel_verts = np.concatenate([corner_verts, oct_verts, cen_verts])

    iy_g, iz_g = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
    iy_f = iy_g.ravel()
    iz_f = iz_g.ravel()

    # slot map: (n_cells, 13)  → global vertex index for each local slot
    slot_map = np.empty((n_cells, 13), dtype=np.int32)
    slot_map[:, 0] = iy_f * (nz + 1) + iz_f                          # c00
    slot_map[:, 1] = (iy_f + 1) * (nz + 1) + iz_f                    # c10
    slot_map[:, 2] = (iy_f + 1) * (nz + 1) + (iz_f + 1)              # c11
    slot_map[:, 3] = iy_f * (nz + 1) + (iz_f + 1)                    # c01
    ob = n_corners + np.arange(n_cells) * 8
    slot_map[:, 4:12] = ob[:, None] + np.arange(8)[None, :]           # v0-v7
    slot_map[:, 12] = n_corners + n_cells * 8 + np.arange(n_cells)    # center

    tmpl = _TRI_TEMPLATE if face_normal > 0 else _TRI_TEMPLATE_FLIP
    pixel_tris = slot_map[:, tmpl].reshape(-1, 3)
    n_pixel_tris = len(pixel_tris)

    is_pad = np.tile(_IS_PAD_CELL, n_cells)

    # 4 strips: bottom, top, left, right
    eps = 1e-6
    rects = []
    if pz_min > z_min + eps:
        rects.append([[y_min, z_min], [y_max, z_min],
                      [y_max, pz_min], [y_min, pz_min]])
    if z_max > pz_max + eps:
        rects.append([[y_min, pz_max], [y_max, pz_max],
                      [y_max, z_max], [y_min, z_max]])
    if py_min > y_min + eps:
        rects.append([[y_min, pz_min], [py_min, pz_min],
                      [py_min, pz_max], [y_min, pz_max]])
    if y_max > py_max + eps:
        rects.append([[py_max, pz_min], [y_max, pz_min],
                      [y_max, pz_max], [py_max, pz_max]])

    if rects:
        n_r = len(rects)
        bv = np.empty((n_r * 4, 3), dtype=np.float32)
        bv[:, 0] = x_pos
        for i, r in enumerate(rects):
            for j, (yy, zz) in enumerate(r):
                bv[i * 4 + j, 1] = yy
                bv[i * 4 + j, 2] = zz

        base = np.arange(n_r)[:, None] * 4 + len(pixel_verts)
        if face_normal > 0:
            bt = np.column_stack([base, base + 1, base + 2,
                                  base, base + 2, base + 3]).reshape(-1, 3)
        else:
            bt = np.column_stack([base, base + 2, base + 1,
                                  base, base + 3, base + 2]).reshape(-1, 3)

        vertices = np.concatenate([pixel_verts, bv])
        triangles = np.concatenate([pixel_tris, bt])
        is_pad = np.concatenate([is_pad, np.zeros(len(bt), dtype=bool)])
        is_border = np.concatenate([np.zeros(n_pixel_tris, dtype=bool),
                                    np.ones(len(bt), dtype=bool)])
    else:
        vertices = pixel_verts
        triangles = pixel_tris
        is_border = np.zeros(n_pixel_tris, dtype=bool)

    return vertices, triangles, is_pad, is_border


def make_pixel_box(dx, dy, dz, n_pixels_y, n_pixels_z,
                   pitch, pad_size, chamfer_radius, center=(0, 0, 0)):
    """
    Create a box mesh with pixel-pad faces on +/- X and plain faces on +/- Y, +/- Z.

    The pixel array is centerd on each +/- X face.  Any remaining border
    area (e.g. from PMT extensions along Y) is steel.

    Parameters
    ----------
    dx, dy, dz : float
        Full box dimensions (mm).
    n_pixels_y, n_pixels_z : int
        Number of pixels along Y and Z on each +/- X face.
    pitch, pad_size, chamfer_radius : float
        Pixel geometry (mm).
    center : tuple
        Box center (x, y, z).

    Returns
    -------
    mesh : Mesh
    is_pad : (n_tri,) bool — True for gold pad triangles
    is_pixel_face : (n_tri,) bool — True for pixel-region triangles (pad + FR-4)
    """
    cx, cy, cz = center
    hx, hy, hz = dx / 2, dy / 2, dz / 2

    all_verts, all_tris = [], []
    all_is_pad, all_is_pf = [], []
    voff = 0

    simple_faces = [
        # +Y (normal +Y)
        [[cx - hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz + hz],
         [cx + hx, cy + hy, cz + hz], [cx + hx, cy + hy, cz - hz]],
        # -Y (normal -Y)
        [[cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
         [cx + hx, cy - hy, cz + hz], [cx - hx, cy - hy, cz + hz]],
        # +Z (normal +Z)
        [[cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
         [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz]],
        # -Z (normal -Z)
        [[cx - hx, cy - hy, cz - hz], [cx - hx, cy + hy, cz - hz],
         [cx + hx, cy + hy, cz - hz], [cx + hx, cy - hy, cz - hz]],
    ]
    for fv in simple_faces:
        all_verts.append(np.array(fv, dtype=np.float32))
        all_tris.append(
            np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32) + voff)
        all_is_pad.append(np.array([False, False]))
        all_is_pf.append(np.array([False, False]))
        voff += 4

    y_range = (cy - hy, cy + hy)
    z_range = (cz - hz, cz + hz)

    for sign in (+1, -1):
        x_pos = cx + sign * hx
        verts, tris, pad, border = make_pixel_face(
            x_pos, y_range, z_range,
            n_pixels_y, n_pixels_z,
            pitch, pad_size, chamfer_radius,
            face_normal=sign,
        )
        all_verts.append(verts)
        all_tris.append(tris + voff)
        all_is_pad.append(pad)
        all_is_pf.append(~border)  # pixel_face = pad + FR-4 (NOT border)
        voff += len(verts)

    vertices = np.concatenate(all_verts).astype(np.float32)
    triangles = np.concatenate(all_tris).astype(np.int32)
    is_pad = np.concatenate(all_is_pad)
    is_pixel_face = np.concatenate(all_is_pf)

    mesh = Mesh(vertices, triangles,
                remove_duplicate_vertices=False,
                remove_null_triangles=False,
                round=False)
    return mesh, is_pad, is_pixel_face


def compute_averaged_pixel_surface(pixel_surface, pcb_surface,
                                   pad_size, chamfer_radius, pitch,
                                   shield_surface=None):
    """
    Create a Surface with area-weighted average of gold-pad, FR-4, and
    (optionally) copper-shield properties.

    The fractional gold area is computed from the octagon (chamfered square)
    inscribed in each pitch cell.  When *shield_surface* is provided, the
    remaining (non-gold) area is split 50/50 between FR-4 and the shield.

    Parameters
    ----------
    pixel_surface : Surface
        Gold pad surface.
    pcb_surface : Surface
        FR-4 background surface.
    pad_size, chamfer_radius, pitch : float
        Pixel geometry (mm).
    shield_surface : Surface or None
        Optional copper (or other) shield surface.  When provided, half
        of the non-gold area is assigned to this surface.

    Returns
    -------
    Surface
        New surface named ``"averaged_pixel"`` with blended properties.
    """
    octagon_area = pad_size ** 2 - 2 * chamfer_radius ** 2
    cell_area = pitch ** 2
    f_gold = octagon_area / cell_area

    if shield_surface is not None:
        f_shield = (1.0 - f_gold) * 0.5
        f_fr4 = (1.0 - f_gold) * 0.5
    else:
        f_shield = 0.0
        f_fr4 = 1.0 - f_gold

    avg = Surface("averaged_pixel")
    for prop in ("absorb", "reflect_specular", "reflect_diffuse", "detect"):
        v_gold = getattr(pixel_surface, prop)
        v_fr4 = getattr(pcb_surface, prop)
        v_shield = getattr(shield_surface, prop) if shield_surface is not None else None

        # start from gold contribution
        if v_gold is not None:
            wl = v_gold[:, 0]
            val = f_gold * v_gold[:, 1]
        else:
            val = None

        # add FR-4 contribution
        if v_fr4 is not None:
            if val is None:
                wl = v_fr4[:, 0]
                val = f_fr4 * v_fr4[:, 1]
            else:
                val = val + f_fr4 * v_fr4[:, 1]

        # add shield contribution
        if v_shield is not None:
            if val is None:
                wl = v_shield[:, 0]
                val = f_shield * v_shield[:, 1]
            else:
                val = val + f_shield * v_shield[:, 1]

        if val is not None:
            avg.set(prop, val, wavelengths=wl)

    return avg
