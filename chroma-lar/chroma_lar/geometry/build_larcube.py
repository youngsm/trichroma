from .pmt import generate_pmt_positions
from .wireplane import add_wires, make_wireplane, attach_wireplanes
from .pmt import build_r5912_pmt

import numpy as np
import chroma.geometry as geometry
import chroma.make as make
import chroma.transform as transform
import chroma.detector as detector
try:
    from chroma.loader import create_geometry_from_obj
except ImportError: # cuda not available
    create_geometry_from_obj = None
from typing import Literal  # noqa: F401


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
    cavity_scale=1.5,  # Cavity size relative to active volume
    # PMT parameters
    pmt_photocathode_surface=None,
    pmt_back_surface=None,
    pmt_glass_material=None,
    center_pmts=True,
    pmt_spacing=500,  # mm
    pmt_gap=10,  # mm from active volume boundary
    pmt_nsteps=20,  # Tessellation for PMT model
    pmt_diameter_in=8,  # in
    pmt_downsample_factor=1,  # factor by which to downsample the PMT profile
    # Wire plane parameters
    wire_diameter=0.150,  # mm
    wire_pitch=3.0,  # mm
    wire_angles=[np.pi / 2, np.pi / 3, -np.pi / 3],  # U, V, Y planes
    wire_offsets=[0.0, -3.0, -6.0],  # mm, relative to active boundary
    wire_nsteps=32,  # Tessellation for wire cylinders
    wire_nsegments=64,  # Tessellation for wire cylinders
    analytic_wires=False,  # if True, attach analytic wire-planes instead of meshes
    wire_inner_material=None,
    wire_surface=None,
    # Cathode parameters
    cathode_thickness=6,  # mm
    cathode_inner_material=None,
    cathode_surface=None,
    # Materials
    default_optics=None,
    target_material=None,
    # Active volume parameters
    active_surface=None,
    # Other options
    include_cavity=True,
    include_active=True,
    include_pmts=True,
    include_wires=True,
    include_cathode=True,
    # BVH options
    flatten=True,
    pmt_coating_surface=None,
):
    """
    Build a complete detector with all components.

    Parameters
    ----------
    active_dimensions : dict
        Dictionary with x, y, z ranges for active volume
    cavity_scale : float
        Scale factor for cavity relative to active volume
    pmt_center : bool
        Whether to center the PMTs
    pmt_spacing : float
        Spacing between PMTs in mm
    pmt_gap : float
        Gap between active volume and PMT planes in mm
    pmt_nsteps : int
        Tessellation parameter for PMT model
    pmt_downsample_factor : int
        Factor by which to downsample the PMT profile
    wire_diameter : float
        Diameter of wire in mm
    wire_pitch : float
        Spacing between wires in mm
    wire_angles : list
        List of angles (radians) for the three wire planes
    wire_offsets : list
        Offset of each wire plane from TPC boundary
    wire_nsteps : int
        Tessellation parameter for wire cylinders
    wire_nsegments : int
        Tessellation parameter for wire cylinders
    analytic_wires : bool
        Whether to use analytic wire-planes instead of meshes
    default_optics : object
        Optics database for materials
    active_surface : Surface, optional
        Surface for the active volume cube
    include_cavity : bool
        Whether to include cavity volume
    include_active : bool
        Whether to include active volume
    include_pmts : bool
        Whether to include PMTs
    include_wires : bool
        Whether to include wire planes
    include_cathode : bool
        Whether to include cathode

    Returns
    -------
    detector.Detector
        Fully constructed detector object
    """

    if target_material is None:
        target_material = default_optics.lar

    # Calculate dimensions from ranges
    lx = active_dimensions["x"][1] - active_dimensions["x"][0]
    ly = active_dimensions["y"][1] - active_dimensions["y"][0]
    lz = active_dimensions["z"][1] - active_dimensions["z"][0]

    g = detector.Detector(target_material)

    if include_cavity:
        cavity = make.box(
            cavity_scale * lx, cavity_scale * ly, cavity_scale * lz, center=(0, 0, 0)
        )
        cavity_solid = geometry.Solid(
            cavity,
            target_material,
            default_optics.vacuum,
            color=0xFFFFFFFF,
            surface=default_optics.reflect00,
        )
        g.add_solid(cavity_solid)

    # generate pmt positions
    pmt_positions, pmt_indices, pmt_directions = generate_pmt_positions(
        lx=lx,
        ly=ly,
        lz=lz,
        spacing_y=pmt_spacing,
        spacing_z=pmt_spacing,
        gap_pmt_active=pmt_gap,
        pmt_radius=in2mm(pmt_diameter_in) / 2,
        center=center_pmts,
    )

    # create pmt model
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
    if include_pmts:
        # add pmt to detector
        for p, r, i in zip(pmt_positions, pmt_directions, pmt_indices):
            # convert positions and directions if they're torch tensors
            if hasattr(p, "numpy"):
                p = p.numpy()
            if hasattr(r, "numpy"):
                r = r.numpy()

            # add pmt to detector
            g.add_pmt(
                pmt,
                displacement=p,
                rotation=transform.gen_rot(np.array([0.0, -1.0, 0]), r),
                channel_type=i,
            )

    if include_active:
        if active_surface is None:
            active_surface = default_optics.reflect99

        active = make.box(
            lx + (abs(pmt.mesh.vertices[:, 1].min()) + pmt_gap) * 2,
            ly,
            lz,
            center=(0, 0, 0),
        )

        active_solid = geometry.Solid(
            mesh=active,
            material1=target_material,
            material2=default_optics.vacuum,
            surface=active_surface,
            color=0xA0A0A0A0,
        )
        g.add_solid(active_solid)

    if include_cathode:
        if cathode_inner_material is None:
            cathode_inner_material = default_optics.steel_material
        if cathode_surface is None:
            cathode_surface = default_optics.reflect99

        add_cathode(
            g,
            ly,
            lz,
            cathode_thickness,
            cathode_inner_material,
            target_material,
            cathode_surface,
            default_optics,
        )
    if include_wires:
        if analytic_wires:
            # attach analytic periodic-cylinder planes at +/- X boundaries with offsets
            if wire_inner_material is None:
                wire_inner_material = default_optics.steel_material
            if wire_surface is None:
                wire_surface = default_optics.reflect99

            y_half, z_half = 0.5 * ly, 0.5 * lz
            corners = np.array(
                [
                    [-y_half, -z_half],
                    [y_half, -z_half],
                    [-y_half, z_half],
                    [y_half, z_half],
                ],
                dtype=np.float32,
            )

            plane_descs = []
            for i, (plane_angle, offset) in enumerate(zip(wire_angles, wire_offsets)):
                cA = np.cos(plane_angle)
                sA = np.sin(plane_angle)
                # projections for u (along wires) and v (across wires)
                r_vals = corners[:, 0] * cA + corners[:, 1] * sA
                s_vals = -corners[:, 0] * sA + corners[:, 1] * cA
                u_length = float(r_vals.max() - r_vals.min())
                v_width = float(s_vals.max() - s_vals.min())

                u_dir = (0.0, float(cA), float(sA))
                plane_normal = (1.0, 0.0, 0.0)  # +X side normal

                radius = 0.5 * float(wire_diameter)
                pitch = float(wire_pitch)

                # +X side
                center_pos = (float(active_dimensions["x"][1] + offset), 0.0, 0.0)
                plane_descs.append(
                    make_wireplane(
                        center=center_pos,
                        u_dir=u_dir,
                        plane_normal=plane_normal,
                        pitch=pitch,
                        radius=radius,
                        length=u_length,
                        width=v_width,
                        first_wire_offset=0.0,
                        surface=wire_surface,
                        material_inner=wire_inner_material,
                        material_outer=target_material,
                        color=[0x000000FF, 0x0000FF00, 0x00FF0000][i % 3],
                    )
                )

                # -X side
                center_neg = (float(active_dimensions["x"][0] - offset), 0.0, 0.0)
                plane_descs.append(
                    make_wireplane(
                        center=center_neg,
                        u_dir=u_dir,
                        plane_normal=(-1.0, 0.0, 0.0),  # -X side normal
                        pitch=pitch,
                        radius=radius,
                        length=u_length,
                        width=v_width,
                        first_wire_offset=0.0,
                        surface=wire_surface,
                        material_inner=wire_inner_material,
                        material_outer=target_material,
                        color=[0x000000FF, 0x0000FF00, 0x00FF0000][i % 3],
                    )
                )

            attach_wireplanes(g, plane_descs)
            try:
                print(f"analytic wireplanes attached: {len(plane_descs)}")
            except Exception:
                pass
        else:
            if wire_inner_material is None:
                wire_inner_material = default_optics.steel_material
            if wire_surface is None:
                wire_surface = default_optics.reflect99
            add_wires(
                g,
                ly,
                lz,
                active_dimensions,
                wire_diameter,
                wire_pitch,
                wire_angles,
                wire_offsets,
                wire_nsteps,
                wire_nsegments,
                inner_material=wire_inner_material,
                outer_material=target_material,
                surface=wire_surface,
                default_optics=default_optics,
            )

    if flatten and create_geometry_from_obj is not None:
        return create_geometry_from_obj(g)

    return g


def add_cathode(
    g: detector.Detector,
    ly: float,
    lz: float,
    cathode_thickness: float,
    inner_material: geometry.Material = None,
    outer_material: geometry.Material = None,
    surface: geometry.Surface = None,
    default_optics: detector.Detector = None,
):
    if inner_material is None:
        inner_material = default_optics.steel_material
    if outer_material is None:
        outer_material = default_optics.lar
    if surface is None:
        surface = default_optics.reflect99

    cathode = make.box(cathode_thickness, ly, lz, center=(0, 0, 0))
    cathode_solid = geometry.Solid(
        cathode,
        material1=inner_material,
        material2=outer_material,
        surface=surface,
        color=0xA0A0A0A0,
    )

    g.add_solid(cathode_solid)
