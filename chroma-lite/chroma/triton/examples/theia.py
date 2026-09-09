"""Theia-like water Cherenkov test geometry, independent of private databases.

Placement follows the supplied Theia builder: staggered cylinder caps/barrel
or six rectangular faces, a 2 m cavity margin, and inward-facing sensors.
The example uses bundled Chroma demo water/glass/QE tables and a scaled SNO
PMT profile. These are illustrative assumptions, not Theia calibration.
All distances are mm, wavelengths nm, and times ns. No scintillator, TPB,
dichroicons, or LAPPDs are included in this water-detector fixture.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chroma.detector import Detector
from chroma.event import Photons
from chroma.geometry import Solid
from chroma.make import box, cylinder_along_z
from chroma.pmt import build_pmt
from chroma.triton.optical_response import TabulatedCDF
from chroma.triton.primitives import (
    Bounds,
    BoxVolume,
    ConvexMeshVolume,
    Mesh,
    MeshInstance,
    PrimitiveScene,
)


def inward_rotation(direction):
    """Proper rotation mapping a PMT's local +Y face to direction."""
    target = np.asarray(direction, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all() or not np.any(target):
        raise ValueError("sensor direction must be a finite nonzero three-vector")
    target = target / np.linalg.norm(target)
    source = np.array([0.0, 1.0, 0.0])
    cross = np.cross(source, target)
    cosine = float(source @ target)
    sine = np.linalg.norm(cross)
    if sine < 1e-14:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    axis = cross / sine
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return cosine * np.eye(3) + (1 - cosine) * np.outer(axis, axis) + sine * skew


def sensor_layout(size=25500.0, *, coverage=0.81, diameter=508.0, towards_zero=False, offset=False):
    """Return positions/rotations, with no dependence on material names."""
    if not np.isfinite([coverage, diameter]).all() or not 0 < coverage <= 1 or diameter <= 0:
        raise ValueError("coverage must be in (0,1] and diameter positive")
    dimensions = np.asarray(size, float)
    if (
        dimensions.shape not in ((), (3,))
        or not np.isfinite(dimensions).all()
        or np.any(dimensions <= diameter)
    ):
        raise ValueError("size must be a radius or three full dimensions, larger than one sensor")
    positions, normals = [], []
    area = np.pi * (diameter / 2) ** 2
    if dimensions.ndim == 0:
        radius = float(dimensions)
        surface_area = 6 * np.pi * radius**2
        spacing = np.sqrt(surface_area / np.ceil(coverage * surface_area / area)) * 0.985
        cols = max(1, round(2 * np.pi * radius / (spacing * 1.118)))
        rows = max(1, round(2 * radius / (spacing / 1.118)))
        reach = round(1.5 * radius / spacing)
        for i in range(-reach, reach + 1):
            for j in range(-reach, reach + 1):
                x = i * spacing / 1.118
                y = (j + 0.5 * offset + 0.5 * (i % 2)) * spacing * 1.118
                if np.hypot(x, y) <= radius - diameter / 2:
                    for sign in (1, -1):
                        positions.append((x, y, sign * radius))
                        normals.append((0.0, 0.0, -sign))
        for col in range(cols):
            for row in range(rows):
                phi = 2 * np.pi * (col + 0.5 * offset + 0.5 * (row % 2)) / cols
                x, y = radius * np.cos(phi), radius * np.sin(phi)
                positions.append((x, y, row * 2 * radius / rows + spacing / 2 - radius))
                normals.append((-np.cos(phi), -np.sin(phi), 0.0))
    else:
        w, h, length = dimensions
        surface_area = 2 * (w * h + w * length + h * length)
        spacing = np.sqrt(surface_area / np.ceil(coverage * surface_area / area))
        # Rows/columns and staggering match rect_pmt_gen in the supplied code.
        for normal_axis, row_axis, col_axis in ((2, 1, 0), (1, 2, 0), (0, 2, 1)):
            rows, cols = [max(1, round(dimensions[a] / spacing)) for a in (row_axis, col_axis)]
            for i in range(-rows // 2, rows // 2):
                for j in range(-cols // 2, cols // 2):
                    for sign in (1, -1):
                        pos, normal = np.zeros(3), np.zeros(3)
                        pos[normal_axis] = sign * dimensions[normal_axis] / 2
                        pos[row_axis] = (i + (rows % 2) / 2) * spacing
                        pos[col_axis] = (j + (cols % 2) / 2 + 0.5 * (i % 2)) * spacing
                        normal[normal_axis] = -sign
                        positions.append(pos)
                        normals.append(normal)
    positions = np.asarray(positions, float)
    normals = -positions if towards_zero else np.asarray(normals)
    rotations = np.asarray([inward_rotation(n) for n in normals])
    return positions, rotations


@dataclass
class TheiaFixture:
    size: object
    coverage: float
    positions: np.ndarray
    rotations: np.ndarray
    enclosure: object
    sensor: object
    water: object
    vacuum: object
    wall: object
    packing: dict

    @property
    def channel_count(self):
        return len(self.positions)

    def primitives(self):
        if np.ndim(self.size) == 0:
            domain = ConvexMeshVolume(
                "water", Mesh(self.enclosure.vertices, self.enclosure.triangles), 0, 1
            )
        else:
            domain = BoxVolume("water", Bounds(*self.enclosure.get_bounds()), 0, 1)
        mesh = Mesh(self.sensor.mesh.vertices, self.sensor.mesh.triangles)
        return PrimitiveScene(
            (domain,),
            instances=tuple(
                MeshInstance(f"sensor:{i}", mesh, r, p, 0)
                for i, (p, r) in enumerate(zip(self.positions, self.rotations))
            ),
        )

    def detector(self):
        """Build ordinary Chroma geometry for the independent mesh transport."""
        result = Detector(self.water)
        result.add_solid(Solid(self.enclosure, self.water, self.vacuum, surface=self.wall))
        for position, rotation in zip(self.positions, self.rotations):
            result.add_pmt(self.sensor, displacement=position, rotation=rotation)
        return result


def audit_sensor_clearance(vertices, positions, rotations):
    """Prove disjoint PMT bounding cylinders, or reject uncertain placement.

    Cylinders enclose every local mesh vertex. A spatial broadphase finds all
    sphere-overlapping pairs; each must have a separating projection. This is
    conservative: an unproven pair may or may not have intersecting triangles.
    It is a fixture placement check, not a geometry-compiler optimization.
    """
    from scipy.spatial import cKDTree

    vertices = np.asarray(vertices, float)
    guard = 32 * np.finfo(np.float32).eps * (np.max(np.abs(positions)) + np.max(np.abs(vertices)))
    radius = np.max(np.linalg.norm(vertices[:, [0, 2]], axis=1)) + guard
    lower, upper = vertices[:, 1].min(), vertices[:, 1].max()
    half_height = (upper - lower) / 2 + guard
    axes = rotations[:, :, 1]
    centers = positions + (upper + lower) / 2 * axes
    sphere_radius = np.hypot(radius, half_height)
    pairs = cKDTree(centers).query_pairs(2 * sphere_radius, output_type="ndarray")
    minimum_gap = np.inf
    for start in range(0, len(pairs), 65536):
        a, b = pairs[start : start + 65536].T
        delta = centers[b] - centers[a]
        directions = np.stack(
            [
                delta,
                axes[a],
                axes[b],
                np.cross(axes[a], axes[b]),
                *[np.broadcast_to(e, delta.shape) for e in np.eye(3)],
            ],
            axis=1,
        )
        lengths = np.linalg.norm(directions, axis=2)
        directions = directions / np.where(lengths > 0, lengths, 1)[:, :, None]

        def support(normal):
            cosine = np.clip(np.abs(np.sum(directions * normal[:, None, :], axis=2)), 0, 1)
            return half_height * cosine + radius * np.sqrt(np.maximum(0, 1 - cosine * cosine))

        gaps = (
            np.abs(np.sum(directions * delta[:, None, :], axis=2))
            - support(axes[a])
            - support(axes[b])
        )
        gap = np.max(np.where(lengths > 0, gaps, -np.inf), axis=1)
        if np.any(gap <= 0):
            bad = int(np.flatnonzero(gap <= 0)[0])
            raise ValueError(
                f"cannot exclude PMT overlap for sensors {a[bad]} and {b[bad]}; reduce coverage or adjust placement"
            )
        minimum_gap = min(minimum_gap, float(gap.min()))
    return {
        "validated": True,
        "method": "separating projections of enclosing finite cylinders",
        "candidate_pairs": len(pairs),
        "rounding_guard_mm": float(guard),
        "minimum_separation_mm": float(minimum_gap) if len(pairs) else None,
    }


def build_theia(
    size=25500.0,
    *,
    coverage=0.81,
    diameter=508.0,
    towards_zero=False,
    offset=False,
    nsteps=10,
    validate_clearance=True,
):
    """Construct a shared-mesh fixture without flattening its sensor instances."""
    from chroma.demo import optics
    from chroma import demo

    sensor = build_pmt(
        str(Path(demo.__file__).parent / "sno_pmt_reduced.txt"),
        3.0,
        optics.water,
        optics.glass,
        optics.vacuum,
        optics.r7081hqe_photocathode,
        optics.black_surface,
        nsteps=nsteps,
    )
    sensor.mesh.vertices *= np.float32(diameter / 203.2)
    positions, rotations = sensor_layout(
        size, coverage=coverage, diameter=diameter, towards_zero=towards_zero, offset=offset
    )
    removed = 0
    if validate_clearance:
        # Reserve space for both the cap PMT's inward extent and a barrel
        # PMT's transverse radius. The original placement overlaps at corners.
        radius = np.max(np.linalg.norm(sensor.mesh.vertices[:, [0, 2]], axis=1))
        clearance = radius + float(sensor.mesh.vertices[:, 1].max()) + 2.0
        if np.ndim(size) == 0:
            caps = np.abs(positions[:, 2]) == size
            keep = caps | (np.abs(positions[:, 2]) < float(size) - clearance)
        else:
            half = np.asarray(size) / 2
            on_face = np.abs(positions) == half
            keep = (np.sum(on_face, axis=1) == 1) & np.all(
                on_face | (np.abs(positions) < half - clearance), axis=1
            )
        removed = int(np.count_nonzero(~keep))
        positions, rotations = positions[keep], rotations[keep]
    packing = (
        audit_sensor_clearance(sensor.mesh.vertices, positions, rotations)
        if validate_clearance
        else {"validated": False, "method": "original unchecked placement"}
    )
    packing["removed_edge_sensors"] = removed
    enclosure = (
        cylinder_along_z(float(size) + 2000.0, 2 * float(size) + 4000.0)
        if np.ndim(size) == 0
        else box(*(np.asarray(size) + 4000.0))
    )
    return TheiaFixture(
        size,
        coverage,
        positions,
        rotations,
        enclosure,
        sensor,
        optics.water,
        optics.vacuum,
        optics.black_surface,
        packing,
    )


def cherenkov_photons(
    count, *, seed=1, center=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), length=1000.0, beta=1.0
):
    """Fixed-count photons from a straight track using bundled water dispersion.

    Samples the Frank--Tamm spectral shape (1-1/(beta*n)^2)/lambda^2
    over 300--650 nm. Polarization lies in the track/photon plane. The caller
    supplies photon count; charged-particle energy loss/yield is not modeled.
    """
    from chroma.demo.optics import water

    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("photon count must be a nonnegative integer")
    if not np.isfinite([length, beta]).all() or length < 0 or not 0 < beta <= 1:
        raise ValueError("track length/beta are invalid")
    center = np.asarray(center, float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("track center must be a finite three-vector")
    rotation = inward_rotation(axis)
    track = rotation[:, 1]
    grid = np.linspace(300.0, 650.0, 701)
    index = np.interp(grid, water.refractive_index[:, 0], water.refractive_index[:, 1])
    density = np.maximum(0.0, 1 - 1 / (beta * index) ** 2) / grid**2
    if not density.any():
        raise ValueError("track is below Cherenkov threshold on the source band")
    distribution = TabulatedCDF.from_pdf(grid, density)
    rng = np.random.default_rng(seed)
    wavelength = distribution.sample(rng.random(count))
    index = np.interp(wavelength, water.refractive_index[:, 0], water.refractive_index[:, 1])
    cosine = 1 / (beta * index)
    sine = np.sqrt(np.maximum(0.0, 1 - cosine**2))
    phi = rng.uniform(0.0, 2 * np.pi, count)
    directions = cosine[:, None] * track + sine[:, None] * (
        np.cos(phi)[:, None] * rotation[:, 0] + np.sin(phi)[:, None] * rotation[:, 2]
    )
    polarization = (track - cosine[:, None] * directions) / sine[:, None]
    distance = rng.uniform(-length / 2, length / 2, count)
    return Photons(
        center + distance[:, None] * track,
        directions,
        polarization,
        wavelength,
        t=(distance + length / 2) / (beta * 299.792458),
    )
