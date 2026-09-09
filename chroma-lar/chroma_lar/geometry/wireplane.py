import numpy as np
from math import cos, sin
import chroma.geometry as geometry
import chroma.make as make
import chroma.detector as detector
from typing import Optional, Sequence, List, Dict, Any

__exports__ = ["add_wires", "make_wireplane", "attach_wireplanes"]


def _clip_segment_to_rectangle(p0, d, y_half, z_half):
    """
    clip an infinite line  L(t) = p0 + d*t  (d is **unit**) to the rectangle
    |y|<=y_half, |z|<=z_half  in the Y-Z plane.
    Returns (t_min, t_max) or None if the line misses the rectangle.
    """
    t_low, t_high = -np.inf, np.inf

    #   y limits
    if abs(d[0]) < 1e-12:  # line is (almost) parallel to Z
        if abs(p0[0]) > y_half:  # outside band → no hit
            return None
    else:
        t1, t2 = (-y_half - p0[0]) / d[0], (y_half - p0[0]) / d[0]
        t_low, t_high = max(t_low, min(t1, t2)), min(t_high, max(t1, t2))

    #   z limits
    if abs(d[1]) < 1e-12:  # line is (almost) parallel to Y
        if abs(p0[1]) > z_half:
            return None
    else:
        t1, t2 = (-z_half - p0[1]) / d[1], (z_half - p0[1]) / d[1]
        t_low, t_high = max(t_low, min(t1, t2)), min(t_high, max(t1, t2))

    return (t_low, t_high) if t_low < t_high else None


def make_wire_plane(
    ly,
    lz,
    wire_pitch,  # distance between wires, mm
    wire_angle,  # radians, 0 = along Y (+horizontal), π/2 = along Z (vertical)
    wire_diameter,  # mm
    offset=0.0,  # shift of the whole bundle along its normal direction, mm
    nsteps=64,  # cylinder tessellation
    nsegments=64,  # cylinder segments
):
    """
    Return a list of Mesh objects, one per wire, covering the rectangle
    |y|<=ly/2, |z|<=lz/2, oriented at `wire_angle`, with centre-to-centre pitch
    `wire_pitch` and diameter `wire_diameter`.
    """
    radius = 0.5 * wire_diameter
    y_half, z_half = 0.5 * ly, 0.5 * lz

    # unit direction vector of a wire (in the Y-Z plane)
    d = np.array([cos(wire_angle), sin(wire_angle)])  # (dy, dz)
    # unit normal (points “across” the wires, used to step pitches)
    n = np.array([-sin(wire_angle), cos(wire_angle)])

    # calculate corners of the detector in the YZ plane
    corners = np.array(
        [
            [-y_half, -z_half],
            [+y_half, -z_half],
            [-y_half, +z_half],
            [+y_half, +z_half],
        ],
        dtype=np.float32,
    )

    # project corners onto the wire direction to find coverage range
    cos_theta = cos(wire_angle)
    sin_theta = sin(wire_angle)
    r_values = corners[:, 0] * cos_theta + corners[:, 1] * sin_theta
    r_min = np.min(r_values)
    r_max = np.max(r_values)

    # calculate wire indices needed to cover this range
    idx_min_rel = int(np.floor(r_min / wire_pitch - 1e-9))
    idx_max_rel = int(np.ceil(r_max / wire_pitch + 1e-9))

    meshes = []
    base_cyl = make.cylinder(
        radius, 1.0, nsteps=nsteps,
    )  # unit-length cylinder along +Y

    # rotation that aligns +Y with direction d→
    rot_axis = np.array([sin(wire_angle), 0, 0])  # actually ±X axis
    rot_angle = wire_angle  # |angle| around X
    # pre-compute rotation matrix (Rodrigues)
    cA, sA = cos(rot_angle), sin(rot_angle)
    R = np.array([[1, 0, 0], [0, cA, -sA], [0, sA, cA]])

    # generate wires for each index in the calculated range
    for idx_rel in range(idx_min_rel, idx_max_rel + 1):
        # calculate position of this wire along normal direction
        s = idx_rel * wire_pitch + offset
        # line centre point at closest approach to rectangle centre
        p0 = n * s  # (y0, z0)

        clip = _clip_segment_to_rectangle(p0, d, y_half, z_half)
        if clip is None:
            continue  # this wire misses the rectangle entirely

        t0, t1 = clip
        seg_len = t1 - t0  # in "t" units (=mm because |d|=1)
        seg_mid = p0 + d * (0.5 * (t0 + t1))  # (y, z) of segment centre

        # clone the unit cylinder, scale to correct length, rotate, translate
        v = base_cyl.vertices.copy()
        v[:, 1] *= seg_len  # stretch along local +Y (its axis)
        v = v @ R.T  # rotate into (d) direction
        v[:, 1] += seg_mid[0]  # Y translation
        v[:, 2] += seg_mid[1]  # Z translation
        # (X remains 0 – on the Y-Z plane)

        meshes.append(
            geometry.Mesh(v, base_cyl.triangles, remove_duplicate_vertices=True)
        )

    return meshes


def add_wires(
    g: detector.Detector,
    ly: float,
    lz: float,
    active_dimensions: dict,
    wire_diameter: float,
    wire_pitch: float,
    wire_angles: list,
    wire_offsets: list,
    wire_nsteps: int,
    wire_nsegments: int,
    inner_material: geometry.Material = None,
    outer_material: geometry.Material = None,
    surface: geometry.Surface = None,
    default_optics: detector.Detector = None,
):
    # define colors for wire planes
    wire_colors = [0x000000FF, 0x0000FF00, 0x00FF0000]  # blue, green, red

    if inner_material is None:
        inner_material = default_optics.steel_material
    if outer_material is None:
        outer_material = default_optics.lar
    if surface is None:
        surface = default_optics.reflect99

    # loop through the three wire planes
    for i, (plane_angle, offset) in enumerate(zip(wire_angles, wire_offsets)):
        # generate wire meshes
        wire_meshes = make_wire_plane(
            ly=ly,
            lz=lz,
            wire_pitch=wire_pitch,
            wire_angle=plane_angle,
            wire_diameter=wire_diameter,
            offset=0.0,
            nsteps=wire_nsteps,
            nsegments=wire_nsegments,
        )

        # add wires to detector
        for mesh in wire_meshes:
            # position at x = active boundary + offset
            x_pos_1 = active_dimensions["x"][1] + offset
            x_pos_0 = active_dimensions["x"][0] - offset

            mesh_solid = geometry.Solid(
                mesh,
                material1=inner_material,
                material2=outer_material,
                color=wire_colors[i],
                surface=surface,
            )

            # add wire at +x position
            g.add_solid(mesh_solid, displacement=(x_pos_0, 0, 0))

            # add wire at -x position
            g.add_solid(mesh_solid, displacement=(x_pos_1, 0, 0))
    return g


def rect_pts(w, h):
    return [-w / 2, -w / 2, w / 2, w / 2], [-h / 2, h / 2, h / 2, -h / 2]


# def add_wire_wall(
#     g: detector.Detector,
#     ly: float,
#     lz: float,
#     active_dimensions: dict,
#     offset: float = 0.0,
#     inner_material: Optional[geometry.Material] = None,
#     outer_material: Optional[geometry.Material] = None,
#     surface: Optional[geometry.Surface] = None,
#     default_optics: Optional[detector.Detector] = None,
# ):
#     if inner_material is None:
#         inner_material = default_optics.lar
#     if outer_material is None:
#         outer_material = default_optics.lar
#     if surface is None:
#         surface = default_optics.wire_wall_surface

#     vertices = [[0, y, z] for y, z in zip(*rect_pts(ly, lz))]
#     triangles = [[0, i + 1, i + 2] for i in range(len(vertices) - 2)]
#     mesh = geometry.Mesh(vertices, triangles, remove_duplicate_vertices=True)

#     wire_solid = geometry.Solid(
#         mesh,
#         material1=inner_material,
#         material2=outer_material,
#         surface=surface,
#         color=0xA0A0A0A0,
#     )

#     x_pos_1 = active_dimensions["x"][1] + offset
#     x_pos_0 = active_dimensions["x"][0] - offset

#     g.add_solid(wire_solid, displacement=(x_pos_0, 0, 0))
#     g.add_solid(wire_solid, displacement=(x_pos_1, 0, 0))

#     return g


# analytic wire-plane simulation helpers
def make_wireplane(
    center: Sequence[float],
    u_dir: Sequence[float],
    plane_normal: Sequence[float],
    pitch: float,
    radius: float,
    length: float,
    width: float,
    first_wire_offset: float = 0.0,
    surface: Optional[geometry.Surface] = None,
    material_inner: Optional[geometry.Material] = None,
    material_outer: Optional[geometry.Material] = None,
    color: int = 0,
) -> Dict[str, Any]:
    # normalize u and n, construct v = n x u
    u = np.asarray(u_dir, dtype=np.float32)
    n = np.asarray(plane_normal, dtype=np.float32)
    if np.linalg.norm(u) == 0 or np.linalg.norm(n) == 0:
        raise ValueError("u_dir and plane_normal must be non-zero")
    u = u / np.linalg.norm(u)
    n = n / np.linalg.norm(n)
    v = np.cross(n, u)
    if np.linalg.norm(v) == 0:
        raise ValueError("u_dir must not be parallel to plane_normal")
    v = v / np.linalg.norm(v)

    half_len = 0.5 * float(length)
    half_wid = 0.5 * float(width)

    desc: Dict[str, Any] = {
        "origin": np.asarray(center, dtype=np.float32),
        "u": u.astype(np.float32),
        "v": v.astype(np.float32),
        "pitch": float(pitch),
        "radius": float(radius),
        "umin": -half_len,
        "umax": +half_len,
        "vmin": -half_wid,
        "vmax": +half_wid,
        "v0": float(first_wire_offset),
        "surface": surface,
        "material_inner": material_inner,
        "material_outer": material_outer,
        "color": int(color),
    }
    return desc


def attach_wireplanes(geom: geometry.Geometry, planes: List[Dict[str, Any]]):
    if planes is None or len(planes) == 0:
        return geom
    # ensure attribute exists
    if not hasattr(geom, "wireplanes") or geom.wireplanes is None:
        geom.wireplanes = []
    geom.wireplanes.extend(planes)
    try:
        print(
            f"analytic wireplanes attached: {len(planes)} (total={len(geom.wireplanes)})"
        )
    except Exception:
        pass
    return geom
