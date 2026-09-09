import numpy as np
from math import pi
import chroma.tools as tools
import chroma.geometry as geometry
import chroma.make as make
# import torch
import os
import numpy as np
import chroma.geometry as geometry

__exports__ = ["generate_pmt_positions", "generate_pmt_positions_ywalls",
               "build_r5912_pmt", "x_reflection_pmt_permutation"]


def x_reflection_pmt_permutation(pmt_coords):
    """
    Return permutation for x-reflection when using a plib built for one half.

    With n_pmt_walls=2, generate_pmt_positions returns order [all -x PMTs][all +x PMTs]
    with identical (y,z) order. So PMT i and PMT i + num_lo are at same (y,z), opposite x.
    If the plib has only the -x wall (81 PMTs), then for a +x source we look up at -x
    and get data[j] = response at -x PMT j. By symmetry, that equals the response at the
    +x PMT with the same (y,z) as -x PMT j, so output[j] = data[j] -> identity.

    If the plib has both walls (162 PMTs), then perm[i] = i + num_lo for i < num_lo
    and perm[i] = i - num_lo for i >= num_lo (half-swap).

    This function infers from pmt_coords and returns the appropriate permutation
    for mapping plib output to "output channel i = PMT at reflected position of plib PMT i".
    """
    pmt_coords = np.asarray(pmt_coords)
    if pmt_coords.ndim != 2 or pmt_coords.shape[1] != 3:
        raise ValueError("pmt_coords must be (N, 3)")
    x = pmt_coords[:, 0]
    neg = x < 0
    pos = x > 0
    n_neg = np.sum(neg)
    n_pos = np.sum(pos)
    if n_neg == 0 or n_pos == 0:
        return np.arange(len(pmt_coords), dtype=np.int64)
    if n_neg != n_pos:
        return np.arange(len(pmt_coords), dtype=np.int64)
    num_lo = n_neg
    perm = np.arange(len(pmt_coords), dtype=np.int64)
    perm[:num_lo] = np.arange(num_lo, len(pmt_coords))
    perm[num_lo:] = np.arange(num_lo)
    return perm


def in2mm(in_value):
    return in_value * 25.4


def generate_pmt_positions(
    lx,
    ly,
    lz,
    spacing_y,
    spacing_z,
    gap_pmt_active,
    n_pmt_walls=2,
    pmt_radius=50,
    center=False,
):
    """
    Generate PMT positions in a hexagonal grid pattern.
    Parameters
    ----------
    lx : float
        Length of the active volume in the x-direction.
    ly : float
        Length of the active volume in the y-direction.
    lz : float
        Length of the active volume in the z-direction.
    spacing_y : float
        Spacing between PMTs in the y-direction.
    spacing_z : float
        Spacing between PMTs in the z-direction.
    gap_pmt_active : float
        Gap between the PMT and the active volume.
    n_pmt_walls: int
        Number of walls to mount PMT (default is 2).
    Returns
    -------
    pmt_coords : torch.Tensor
        Tensor containing the x, y, z coordinates of the PMTs.
    pmt_ids : torch.Tensor
        Tensor containing the IDs of the PMTs.
    """
    
    grid_y = int(ly / spacing_y) + 1
    grid_z = int(lz / spacing_z) + 1

    # Generate hexagonal grid coordinates
    spacing_buffer_y = ly - (grid_y - 1) * spacing_y - 4 * pmt_radius
    spacing_buffer_z = lz - (grid_z - 1) * spacing_z - 4 * pmt_radius
    while spacing_buffer_y < spacing_y / 2:
        grid_y -= 1
        spacing_buffer_y = ly - (grid_y - 1) * spacing_y
    while spacing_buffer_z < 0:
        grid_z -= 1
        spacing_buffer_z = ly - (grid_z - 1) * spacing_z
    print(f"Spacing buffer in y: {spacing_buffer_y}, z: {spacing_buffer_z}")

    y_side, z_side = np.meshgrid(np.arange(grid_y), np.arange(grid_z), indexing="ij")
    y_side = y_side*spacing_y - ly/2 + (spacing_buffer_y/2 - spacing_y/4) + pmt_radius
    z_side = z_side*spacing_z - lz/2 + spacing_buffer_z/2 + 2*pmt_radius

    print(f"Total PMT number is {n_pmt_walls * grid_y * grid_z}")

    y_side = y_side.astype(np.float32)
    z_side = z_side.astype(np.float32)

    for i in range(y_side.shape[1]):
        if i % 2 == 1:
            y_side[:, i] += spacing_y / 2 - pmt_radius

    y = np.tile(y_side, (n_pmt_walls, 1, 1)).flatten()  # reshape(2, -1)
    z = np.tile(z_side, (n_pmt_walls, 1, 1)).flatten()  # reshape(2, -1)

    if center:
        y -= y.mean()
        z -= z.mean()

    if n_pmt_walls == 2:
        num_lo_pmt = num_hi_pmt = int(len(y) / 2)
        lo_x_value = -lx / 2 - gap_pmt_active
        hi_x_value = lx / 2 + gap_pmt_active
        x = np.concatenate(
            (
                np.full((num_lo_pmt,), lo_x_value),
                np.full((num_hi_pmt,), hi_x_value),
            )
        )
        normal = np.zeros((len(x), 3), dtype=np.float32)
        normal[:num_lo_pmt, 0] = 1.0  # -x side PMTs face +x direction
        normal[num_lo_pmt:, 0] = -1.0  # +x side PMTs face -x direction
    elif n_pmt_walls == 1:
        num_pmt = len(y)
        hi_x_value = lx + gap_pmt_active
        x = np.full((num_pmt,), hi_x_value)  # Single wall PMT at x=lx+gap_pmt_active
        normal = np.zeros((len(x), 3), dtype=np.float32)
        normal[:, 0] = 1.0  # +x side PMTs face +x direction
    else:
        raise ValueError("Number of PMT walls must be 1 or 2.")

    # print(x, y, z)
    # Shift every other row to create a hexagonal pattern

    # pmt_coords = torch.column_stack((x.flatten(), y.flatten(), z.flatten()))

    # change to swap between the sides
    pmt_coords = np.stack((x, y, z), axis=-1)
    return pmt_coords, np.arange(pmt_coords.shape[0], dtype=np.int32), normal

def generate_pmt_positions_ywalls(
    lx,
    ly,
    lz,
    n_x_half=5,
    n_z=9,
    pmt_gap=10.0,
    pmt_radius=55.626,
    wall_margin=None,
    cathode_clearance=None,
    cathode_thickness=6.0,
):
    """
    Generate PMT positions on +/- Y walls for a pixel TPC configuration.

    Layout: *n_x_half* × *n_z* hexagonal grid per quadrant (+X half of
    +Y wall), mirrored in X then in Y.  Alternating Z-rows are staggered
    by half a spacing in X (hex packing).  PMTs that fall outside the
    available range after the stagger are removed.

    Parameters
    ----------
    lx, ly, lz : float
        Active-volume dimensions (mm).
    n_x_half : int
        Number of PMTs in the +X half per Z-row (before hex filtering).
    n_z : int
        Number of PMTs along Z.
    pmt_gap : float
        Gap from active-volume Y boundary to PMT photocathode plane (mm).
    pmt_radius : float
        PMT radius (mm).
    wall_margin : float or None
        Minimum distance from PMT center to the outer +/- X and +/- Z wall
        edges.  If *None*, defaults to ``pmt_radius``.
    cathode_clearance : float or None
        Minimum distance from X = 0 to PMT center.  If *None*, computed
        as ``cathode_thickness / 2 + pmt_radius + 5``.
    cathode_thickness : float
        Cathode slab thickness (mm).

    Returns
    -------
    positions : (N, 3) float32
    ids : (N,) int32
    directions : (N, 3) float32 — unit normals pointing inward
    """
    if wall_margin is None:
        wall_margin = pmt_radius
    if cathode_clearance is None:
        cathode_clearance = cathode_thickness / 2 + pmt_radius + 5.0

    # Available range with wall_margin from outer edges, cathode_clearance from center
    x_min = cathode_clearance + wall_margin
    x_max = lx / 2 - wall_margin
    z_min = -lz / 2 + wall_margin
    z_max = lz / 2 - wall_margin

    # Base grid for one quadrant (+X half of +/- Y wall)
    x_arr = np.linspace(x_min, x_max, n_x_half)
    z_arr = np.linspace(z_min, z_max, n_z)
    spacing_x = (x_max - x_min) / max(n_x_half - 1, 1)

    xx, zz = np.meshgrid(x_arr, z_arr, indexing="ij")

    # Hex stagger: shift alternating Z-rows by half a spacing in X
    for j in range(n_z):
        if j % 2 == 1:
            xx[:, j] += spacing_x / 2

    # Filter positions that exceeded bounds after hex shift
    x_flat = xx.ravel()
    z_flat = zz.ravel()
    keep = (x_flat >= x_min - 1e-6) & (x_flat <= x_max + 1e-6)
    xz_half = np.column_stack([x_flat[keep], z_flat[keep]])
    n_half = len(xz_half)

    y_wall = ly / 2 + pmt_gap

    # +X half → +/- Y wall
    pos_q1 = np.column_stack([
        xz_half[:, 0], np.full(n_half, y_wall), xz_half[:, 1]])

    # Mirror in X → -X half of +/- Y wall
    pos_q2 = pos_q1.copy()
    pos_q2[:, 0] *= -1

    # Mirror in Y → +/- Y wall copies of each quadrant
    pos_q3 = pos_q1.copy();  pos_q3[:, 1] *= -1   # +X, -Y wall
    pos_q4 = pos_q2.copy();  pos_q4[:, 1] *= -1   # -X, -Y wall

    dir_py = np.tile([0.0, -1.0, 0.0], (n_half, 1)).astype(np.float32)
    dir_my = np.tile([0.0,  1.0, 0.0], (n_half, 1)).astype(np.float32)

    # Order: first half = +X side (both walls), second half = -X side (both walls)
    # PMT i and PMT i + N/2 share the same (|x|, y, z) with x sign flipped.
    pos_plus_x  = np.vstack([pos_q1, pos_q3])   # +X side: +/- Y wall, then +/- Y wall
    pos_minus_x = np.vstack([pos_q2, pos_q4])   # -X side: same order
    dir_plus_x  = np.vstack([dir_py, dir_my])
    dir_minus_x = np.vstack([dir_py, dir_my])

    positions  = np.vstack([pos_plus_x, pos_minus_x]).astype(np.float32)
    directions = np.vstack([dir_plus_x, dir_minus_x]).astype(np.float32)
    ids = np.arange(len(positions), dtype=np.int32)

    print(f"Total PMT number is {len(positions)}")

    return positions, ids, directions


def split_pmt_profile(
    y_min=65,
    split_threshold=220,
    target_diameter=190,
    downsample_factor=1,
):
    """
    Process a PMT profile by splitting it into top and bottom parts based on diameter criteria.

    Args:
        profile_path (str): Path to the CSV file containing PMT contour data
        y_min (float): Minimum y-value to consider in the profile
        split_threshold (float): Minimum y-value for considering diameter matching
        target_diameter (float): Target diameter to match for splitting the profile
        downsample_factor (int): Factor by which to downsample the profile points

    Returns:
        tuple: (cap_profile, bottom_profile, downsampled_cap, downsampled_bottom, y_threshold, actual_diameter)
    """
    # Read the profile data
    # make the path relative to the current file
    profile_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "pmt_contour_mm.csv")
    profile = tools.read_csv(profile_path)
    profile = profile[profile[:, 1] > y_min]

    # Calculate the diameter at each y position
    unique_y_values = np.unique(profile[:, 1])
    diameters = []
    y_values = []

    for y in unique_y_values:
        points_at_y = profile[profile[:, 1] == y]
        if len(points_at_y) >= 2:  # Need at least 2 points to calculate diameter
            diameter = np.max(points_at_y[:, 0]) - np.min(points_at_y[:, 0])
            diameters.append(diameter)
            y_values.append(y)

    diameters = np.array(diameters)
    y_values = np.array(y_values)

    # Find the y-value where diameter is closest to target_diameter and y > split_threshold
    valid_indices = y_values > split_threshold
    if np.any(valid_indices):
        valid_diameters = diameters[valid_indices]
        valid_y_values = y_values[valid_indices]
        closest_idx = np.argmin(np.abs(valid_diameters - target_diameter))
        y_threshold = valid_y_values[closest_idx]
        actual_diameter = valid_diameters[closest_idx]
    else:
        # Fallback if no valid points found
        y_threshold = split_threshold
        actual_diameter = 0

    # Get midpoint in X
    x_midpoint = (np.max(profile[:, 0]) + np.min(profile[:, 0])) / 2
    # Get top-most point in Y
    y_max = np.max(profile[:, 1])

    # Shift the coordinates to set the zero points
    profile[:, 0] -= x_midpoint  # X=0 at the midpoint
    profile[:, 1] = profile[:, 1] - y_max  # Y=0 at the top-most point
    y_threshold = y_threshold - y_max

    if downsample_factor > 1:
        profile = profile[::downsample_factor]

    return (
        profile,
        y_threshold,
    )


def build_r5912_pmt(
    glass_thickness=3.0,
    nzsteps=40,
    nsteps=64,
    diameter=in2mm(8),
    outer_material=None,
    glass=None,
    vacuum=None,
    photocathode_surface=None,
    back_surface=None,
    default_optics=None,
    return_individual_solids=False,
    downsample_factor=1,
    coating_surface=None,
):

    """ returns R5912 PMT solid oriented along [0,1,0]"""
    profile, y_threshold = split_pmt_profile(downsample_factor=downsample_factor)

    if outer_material is None:
        outer_material = default_optics.lar
    if glass is None:
        glass = default_optics.glass
    if vacuum is None:
        vacuum = default_optics.vacuum
    if photocathode_surface is None:
        photocathode_surface = default_optics.r5912_mod_photocathode
    if back_surface is None:
        back_surface = default_optics.glossy_surface

    # slice profile in half
    profile = profile[profile[:, 0] < 0]
    profile[:, 0] = -profile[:, 0]
    # order profile from base to face
    profile = profile[np.argsort(profile[:, 1])]
    # set x coordinate to 0.0 for first and last profile along the profile

    mask = profile[:, 1] > -150
    idx = np.argwhere(mask)[0, 0]
    profile[0, 0] = 0.0  # close it
    profile[-1, 0] = 0.0  # close it

    back_end = np.concatenate([profile[[0]], [[profile[idx, 0], profile[0, 1]]]])
    front_end = profile[mask]

    r = np.sqrt(np.sum(np.square(front_end), axis=1))
    theta = np.arctan2(front_end[:, 1], front_end[:, 0])

    new_theta = np.linspace(theta[0], theta[-1], nzsteps)
    if pi / 2 not in new_theta:
        new_theta = np.sort(np.append(new_theta, pi / 2))
    new_r = np.interp(new_theta, theta, r)
    front_end = np.asarray([new_r * np.cos(new_theta), new_r * np.sin(new_theta)]).T
    front_end[-1, 0] = 0.0  # close it
    profile = np.concatenate([back_end, front_end])

    offset_profile = tools.offset(profile, -glass_thickness)
    offset_profile[0, 0] = 0.0  # close it
    offset_profile[-1, 0] = 0.0  # close it
    

    outer_envelope_mesh = make.rotate_extrude(profile[:, 0], profile[:, 1], nsteps)
    inner_envelope_mesh = make.rotate_extrude(
        offset_profile[:, 0], offset_profile[:, 1], nsteps
    )

    # Optional WLS coating covers the outer window above the photocathode.
    # Other triangles retain the ordinary glass/LAr dielectric boundary.
    coated_window = np.mean(outer_envelope_mesh.assemble(), axis=1)[:, 1] > y_threshold
    outer_surface = None if coating_surface is None else np.where(coated_window, coating_surface, None)
    outer_envelope = geometry.Solid(
        outer_envelope_mesh, glass, outer_material, color=0xD0E0E0E0, surface=outer_surface
    )

    # photocathode starts at y=y_threshold
    photocathode = np.mean(inner_envelope_mesh.assemble(), axis=1)[:, 1] > y_threshold
    # photocathode = np.ones_like(photocathode).astype(bool)

    inner_envelope = geometry.Solid(
        inner_envelope_mesh,
        vacuum,
        glass,
        surface=np.where(photocathode, photocathode_surface, back_surface),
        color=np.where(photocathode, 0xC0FF9900, 0xD0E0E0E0),
    )

    if return_individual_solids:
        return outer_envelope, inner_envelope

    pmt = outer_envelope + inner_envelope

    # scale the pmt
    pmt.mesh.vertices[:] *= diameter / (in2mm(8))
    return pmt
