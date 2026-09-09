"""
Build a pixel LArTPC detector.

Pixel pads (gold on FR-4) on the +/- X faces, PMTs on +/- Y walls, cathode at X = 0.
The box Y dimension is extended to house the PMTs behind the +/- Y faces.
"""

from .pmt import generate_pmt_positions_ywalls, build_r5912_pmt
from .pixelplane import make_pixel_box, compute_averaged_pixel_surface

import numpy as np
import chroma.geometry as geometry
from chroma.geometry import Mesh
import chroma.make as make
import chroma.transform as transform
import chroma.detector as detector

try:
    from chroma.loader import create_geometry_from_obj
except ImportError:  # cuda not available
    create_geometry_from_obj = None


def in2mm(in_value):
    """Convert inches to millimeters."""
    return in_value * 25.4


def build_detector(
    # Geometry parameters
    active_dimensions={
        "x": [-2160.0, 2160.0],
        "y": [-2160.0, 2160.0],
        "z": [-2160.0, 2160.0],
    },
    cavity_scale=1.5,
    # PMT parameters (+/- Y walls)
    pmt_photocathode_surface=None,
    pmt_back_surface=None,
    pmt_glass_material=None,
    pmt_n_x_half=5,
    pmt_n_z=9,
    pmt_gap=10,            # mm from active volume Y boundary to PMT photocathode
    pmt_nsteps=20,
    pmt_diameter_in=4.38,  # inches
    pmt_downsample_factor=1,
    # Pixel parameters (+/- X faces)
    pixel_pitch=4.32,          # mm
    pixel_pad_size=2.419,      # mm
    pixel_chamfer_radius=0.43, # mm
    n_pixels_y=1000,
    n_pixels_z=1000,
    pixel_surface=None,        # gold surface for pads
    pcb_surface=None,          # FR-4 surface
    shield_surface=None,       # copper shield surface (None = no shield in average)
    pixel_simplified=False,    # use area-averaged surface instead of detailed mesh
    # Cathode parameters
    pmt_wall_margin=None,      # min distance from PMT center to wall edge (None → pmt_radius)
    cathode_thickness=6,       # mm
    cathode_clearance=None,    # min |X| for PMTs (None → auto)
    cathode_inner_material=None,
    cathode_surface=None,
    # Materials
    default_optics=None,
    target_material=None,
    # Active volume
    active_surface=None,       # surface for non-pixel faces (+/- Y, +/- Z, borders)
    # Flags
    include_cavity=True,
    include_active=True,
    include_pmts=True,
    include_pixels=True,
    include_cathode=True,
    # BVH
    flatten=True,
    pmt_coating_surface=None,
):
    """
    Build a complete pixel LArTPC detector.

    Parameters
    ----------
    active_dimensions : dict
        Dictionary with x, y, z ranges for active volume.
    cavity_scale : float
        Scale factor for cavity relative to active volume.
    pmt_n_x_half : int
        Number of PMTs in the +X half per Z-row on each +/- Y wall.
    pmt_n_z : int
        Number of PMTs along Z on each +/- Y wall.
    pmt_gap : float
        Gap between active volume Y boundary and PMT photocathode (mm).
    pmt_diameter_in : float
        PMT diameter in inches.
    pixel_pitch : float
        center-to-center pixel spacing (mm).
    pixel_pad_size : float
        Square pad side length before chamfering (mm).
    pixel_chamfer_radius : float
        Corner chamfer radius (mm).
    n_pixels_y, n_pixels_z : int
        Number of pixels along Y and Z on each +/- X face.
    pixel_surface : Surface
        Surface for gold pads.
    pcb_surface : Surface
        Surface for FR-4 PCB surround.
    cathode_thickness : float
        Cathode slab thickness (mm).
    cathode_clearance : float or None
        Min distance from X = 0 to any PMT center. If None, auto-computed.
    default_optics : object
        Optics database.
    active_surface : Surface
        Surface for non-pixel faces (+/- Y, +/- Z, steel borders).
    include_pixels : bool
        If False, the active volume is a plain box (no pixel detail).

    Returns
    -------
    detector.Detector
    """
    if target_material is None:
        target_material = default_optics.lar

    # ── dimensions ────────────────────────────────────────────────────
    lx = active_dimensions["x"][1] - active_dimensions["x"][0]
    ly = active_dimensions["y"][1] - active_dimensions["y"][0]
    lz = active_dimensions["z"][1] - active_dimensions["z"][0]

    pmt_radius = in2mm(pmt_diameter_in) / 2

    # Build PMT model (needed for depth even if include_pmts is False)
    pmt = build_r5912_pmt(
        glass_thickness=3,
        nzsteps=pmt_nsteps,
        nsteps=64,
        diameter=in2mm(pmt_diameter_in),
        outer_material=target_material,
        glass=pmt_glass_material,
        vacuum=default_optics.vacuum,
        photocathode_surface=pmt_photocathode_surface,
        coating_surface=pmt_coating_surface,
        back_surface=pmt_back_surface,
        default_optics=default_optics,
        downsample_factor=pmt_downsample_factor,
    )
    pmt_depth = abs(pmt.mesh.vertices[:, 1].min())

    # Extended Y dimension to house PMTs behind +/- Y faces
    dy_extended = ly + 2 * (pmt_depth + pmt_gap)

    g = detector.Detector(target_material)

    # cavity
    if include_cavity:
        cavity = make.box(
            cavity_scale * lx,
            cavity_scale * dy_extended,
            cavity_scale * lz,
            center=(0, 0, 0),
        )
        cavity_solid = geometry.Solid(
            cavity,
            target_material,
            default_optics.vacuum,
            color=0xFFFFFFFF,
            surface=default_optics.reflect00,
        )
        g.add_solid(cavity_solid)

    # PMTs on +/- Y walls
    pmt_positions, pmt_indices, pmt_directions = generate_pmt_positions_ywalls(
        lx=lx,
        ly=ly,
        lz=lz,
        n_x_half=pmt_n_x_half,
        n_z=pmt_n_z,
        pmt_gap=pmt_gap,
        pmt_radius=pmt_radius,
        wall_margin=pmt_wall_margin,
        cathode_clearance=cathode_clearance,
        cathode_thickness=cathode_thickness,
    )

    if include_pmts:
        for p, r, i in zip(pmt_positions, pmt_directions, pmt_indices):
            if hasattr(p, "numpy"):
                p = p.numpy()
            if hasattr(r, "numpy"):
                r = r.numpy()
            g.add_pmt(
                pmt,
                displacement=p,
                rotation=transform.gen_rot(np.array([0.0, -1.0, 0.0]), r),
                channel_type=i,
            )

    # active volume
    if include_active:
        if active_surface is None:
            active_surface = default_optics.polished_steel_surface
        if pixel_surface is None:
            pixel_surface = default_optics.gold
        if pcb_surface is None:
            pcb_surface = default_optics.polished_steel_surface

        if include_pixels and not pixel_simplified:
            # Detailed pixel mesh: gold pads + FR-4 surround on +/- X
            mesh, is_pad, is_pixel_face = make_pixel_box(
                dx=lx,
                dy=dy_extended,
                dz=lz,
                n_pixels_y=n_pixels_y,
                n_pixels_z=n_pixels_z,
                pitch=pixel_pitch,
                pad_size=pixel_pad_size,
                chamfer_radius=pixel_chamfer_radius,
                center=(0, 0, 0),
            )

            n_tri = len(mesh.triangles)
            surface_arr = np.empty(n_tri, dtype=object)
            color_arr = np.empty(n_tri, dtype=np.uint32)

            surface_arr[is_pad] = pixel_surface
            color_arr[is_pad] = 0xFFD700

            fr4_mask = is_pixel_face & ~is_pad
            surface_arr[fr4_mask] = pcb_surface
            color_arr[fr4_mask] = 0x2E8B57

            other_mask = ~is_pixel_face
            surface_arr[other_mask] = active_surface
            color_arr[other_mask] = 0xA0A0A0

            active_solid = geometry.Solid(
                mesh=mesh,
                material1=target_material,
                material2=default_optics.vacuum,
                surface=surface_arr,
                color=color_arr,
            )

        elif include_pixels and pixel_simplified:
            # Simple box with area-averaged pixel surface on the pixel
            # region of each +/- X face, steel on the border strips and
            # other faces.  This correctly separates the pixel region
            # (ly x lz centerd) from the extended-Y border (steel).
            avg_surface = compute_averaged_pixel_surface(
                pixel_surface, pcb_surface,
                pixel_pad_size, pixel_chamfer_radius, pixel_pitch,
                shield_surface=shield_surface,
            )

            mesh, is_pixel_region = _make_simplified_pixel_box(
                lx, dy_extended, lz, ly, lz)

            n_tri = len(mesh.triangles)
            surface_arr = np.empty(n_tri, dtype=object)
            color_arr = np.full(n_tri, 0xA0A0A0, dtype=np.uint32)

            surface_arr[is_pixel_region] = avg_surface
            color_arr[is_pixel_region] = 0xC8A000
            surface_arr[~is_pixel_region] = active_surface

            active_solid = geometry.Solid(
                mesh=mesh,
                material1=target_material,
                material2=default_optics.vacuum,
                surface=surface_arr,
                color=color_arr,
            )

        else:
            # Plain box, uniform surface (no pixel detail)
            active_mesh = make.box(lx, dy_extended, lz, center=(0, 0, 0))
            active_solid = geometry.Solid(
                mesh=active_mesh,
                material1=target_material,
                material2=default_optics.vacuum,
                surface=active_surface,
                color=0xA0A0A0A0,
            )

        g.add_solid(active_solid)

    # cathode
    if include_cathode:
        if cathode_inner_material is None:
            cathode_inner_material = default_optics.steel_material
        if cathode_surface is None:
            cathode_surface = default_optics.polished_steel_surface

        add_cathode(
            g,
            dy_extended,
            lz,
            cathode_thickness,
            cathode_inner_material,
            target_material,
            cathode_surface,
            default_optics,
        )

    # flatten / BVH
    if flatten and create_geometry_from_obj is not None:
        return create_geometry_from_obj(g)

    return g


def add_cathode(
    g,
    ly,
    lz,
    cathode_thickness,
    inner_material=None,
    outer_material=None,
    surface=None,
    default_optics=None,
):
    if inner_material is None:
        inner_material = default_optics.steel_material
    if outer_material is None:
        outer_material = default_optics.lar
    if surface is None:
        surface = default_optics.polished_steel_surface

    cathode = make.box(cathode_thickness, ly, lz, center=(0, 0, 0))
    cathode_solid = geometry.Solid(
        cathode,
        material1=inner_material,
        material2=outer_material,
        surface=surface,
        color=0xA0A0A0A0,
    )
    g.add_solid(cathode_solid)


def _make_simplified_pixel_box(dx, dy, dz, pixel_dy, pixel_dz):
    """
    Build a simple box (few triangles) where the +/- X faces are split into
    a central *pixel region* (``pixel_dy × pixel_dz``, centerd) and steel
    border strips for the remaining area.

    Returns ``(mesh, is_pixel_region)`` where *is_pixel_region* is a
    boolean mask over triangles.
    """
    hx, hy, hz = dx / 2, dy / 2, dz / 2
    phy, phz = pixel_dy / 2, pixel_dz / 2  # pixel region half-extents

    all_verts, all_tris = [], []
    all_is_pr = []                         # is_pixel_region flags
    voff = 0

    def _add_quad(v0, v1, v2, v3, pixel_region=False):
        """Append a 4-vertex quad (2 triangles) to the running lists."""
        nonlocal voff
        all_verts.append(np.array([v0, v1, v2, v3], dtype=np.float32))
        all_tris.append(np.array([[0,1,2],[0,2,3]], dtype=np.int32) + voff)
        all_is_pr.extend([pixel_region, pixel_region])
        voff += 4

    # 4 simple faces (+/- Y, +/- Z) — all steel
    _add_quad([-hx,+hy,-hz], [-hx,+hy,+hz], [+hx,+hy,+hz], [+hx,+hy,-hz])  # +Y
    _add_quad([-hx,-hy,-hz], [+hx,-hy,-hz], [+hx,-hy,+hz], [-hx,-hy,+hz])  # -Y
    _add_quad([-hx,-hy,+hz], [+hx,-hy,+hz], [+hx,+hy,+hz], [-hx,+hy,+hz])  # +Z
    _add_quad([-hx,-hy,-hz], [-hx,+hy,-hz], [+hx,+hy,-hz], [+hx,-hy,-hz])  # -Z

    # +/- X faces: central pixel region + border strips
    for x_sign in (+1, -1):
        x = x_sign * hx
        # Determine winding order based on face normal direction
        # For +X face (normal +X): CCW in YZ means Y then Z
        # For -X face (normal -X): reversed
        if x_sign > 0:
            def q(y0,z0, y1,z1, y2,z2, y3,z3, pr=False):
                _add_quad([x,y0,z0],[x,y1,z1],[x,y2,z2],[x,y3,z3], pr)
        else:
            def q(y0,z0, y1,z1, y2,z2, y3,z3, pr=False):
                _add_quad([x,y3,z3],[x,y2,z2],[x,y1,z1],[x,y0,z0], pr)

        # Central pixel region (pixel_dy × pixel_dz centerd)
        q(-phy,-phz, +phy,-phz, +phy,+phz, -phy,+phz, pr=True)

        # Border strips: +/- Y sides (pixel array covers full Z, only Y is extended)
        eps = 1e-6
        if hy > phy + eps:
            q(-hy,-hz, -phy,-hz, -phy,+hz, -hy,+hz)    # -Y border
            q(+phy,-hz, +hy,-hz, +hy,+hz, +phy,+hz)    # +Y border

    vertices = np.concatenate(all_verts).astype(np.float32)
    triangles = np.concatenate(all_tris).astype(np.int32)
    is_pixel_region = np.array(all_is_pr, dtype=bool)

    mesh = Mesh(vertices, triangles,
                remove_duplicate_vertices=False,
                remove_null_triangles=False,
                round=False)
    return mesh, is_pixel_region
