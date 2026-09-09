"""Detector-independent geometry descriptions used to certify bulk regions.

These immutable host objects describe authoritative volumes and conservative
obstacle bounds. They do not substitute fitted analytic surfaces for input
meshes. The compiler certifies box-shaped interiors and leaves all uncertified
space to the existing exact boundary engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _vector(value, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite three-vector")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, eq=False)
class Bounds:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "lower", _vector(self.lower, "lower"))
        object.__setattr__(self, "upper", _vector(self.upper, "upper"))
        if np.any(self.lower > self.upper):
            raise ValueError("bounds are inverted")

    @property
    def volume(self):
        return float(np.prod(self.upper - self.lower))

    def overlaps(self, other):
        """Whether the open interiors intersect; touching faces are excluded."""
        return bool(np.all(self.lower < other.upper) and np.all(other.lower < self.upper))

    def encloses(self, other):
        return bool(np.all(self.lower <= other.lower) and np.all(self.upper >= other.upper))

    def contains(self, point):
        point = np.asarray(point)
        return np.all((point > self.lower) & (point < self.upper), axis=-1)

    def float32_interior(self):
        """Round faces inward; open membership excludes an exactly equal face."""
        lo = self.lower.astype(np.float32)
        hi = self.upper.astype(np.float32)
        lo = np.where(lo < self.lower, np.nextafter(lo, np.float32(np.inf)), lo)
        hi = np.where(hi > self.upper, np.nextafter(hi, np.float32(-np.inf)), hi)
        if np.any(lo >= hi):
            return None
        return Bounds(lo, hi)


@dataclass(frozen=True)
class BoxVolume:
    name: str
    bounds: Bounds
    material_inside: int
    material_outside: int

    def __post_init__(self):
        if not self.name or self.bounds.volume <= 0:
            raise ValueError("box volumes require a name and positive volume")
        for value in (self.material_inside, self.material_outside):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError("material indices must be nonnegative integers")

    def encloses(self, bounds):
        return self.bounds.encloses(bounds)

    def seed_cells(self):
        return (self.bounds,)

    def descriptor(self):
        return ["box", self.bounds.lower.tolist(), self.bounds.upper.tolist()]


@dataclass(frozen=True)
class BoundedObstacle:
    """Bounds covering every interface of a mesh, instance, or other object.

    exterior_material declares the homogeneous medium outside these bounds.
    None means it is unknown, preventing certification of an overlapping domain.
    Bounds supplied by a traversal backend should include its numerical padding.
    """

    name: str
    bounds: Bounds
    exterior_material: int | None

    def __post_init__(self):
        value = self.exterior_material
        if not self.name or (
            value is not None
            and (isinstance(value, bool) or not isinstance(value, Integral) or value < 0)
        ):
            raise ValueError("obstacles require a name and a valid exterior material")


@dataclass(frozen=True, eq=False)
class WireArray:
    """Finite parallel cylinders in an orthonormal local frame.

    Cylinder axes run along u between axial_limits; centerlines are at
    origin + v*(offset + k*pitch), first <= k <= last. Bounds conservatively
    cover the entire array for region discovery, without expanding its wires.
    """

    name: str
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    pitch: float
    radius: float
    axial_limits: tuple[float, float]
    offset: float
    first: int
    last: int
    exterior_material: int

    def __post_init__(self):
        for name in ("origin", "u", "v"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        object.__setattr__(self, "axial_limits", tuple(self.axial_limits))
        if len(self.axial_limits) != 2:
            raise ValueError("wire axial_limits must contain two values")
        frame = np.stack((self.u, self.v))
        if not np.allclose(frame @ frame.T, np.eye(2), rtol=0, atol=2e-14):
            raise ValueError("wire u/v axes must be orthonormal")
        values = (self.pitch, self.radius, *self.axial_limits, self.offset)
        if not np.isfinite(values).all() or self.pitch <= 0 or self.radius <= 0:
            raise ValueError("wire dimensions must be finite with positive pitch/radius")
        if self.axial_limits[0] >= self.axial_limits[1]:
            raise ValueError("wire axial extent must be positive")
        if (
            any(isinstance(v, bool) or not isinstance(v, Integral) for v in (self.first, self.last))
            or self.first > self.last
        ):
            raise ValueError("wire array needs a nonempty integer index range")

    def obstacle(self):
        axial = np.asarray(self.axial_limits)
        transverse = self.offset + self.pitch * np.asarray([self.first, self.last])
        corners = np.asarray(
            [self.origin + a * self.u + b * self.v for a in axial for b in transverse]
        )
        # Radius projected onto each world axis. Round outward so a grazing
        # cylinder point cannot lie outside the certificate's exclusion box.
        radial = self.radius * np.sqrt(np.maximum(0.0, 1.0 - self.u * self.u))
        scale = np.maximum(1.0, np.max(np.abs(corners), axis=0))
        padding = 32 * np.finfo(np.float64).eps * scale
        bounds = Bounds(
            np.nextafter(corners.min(0) - radial - padding, -np.inf),
            np.nextafter(corners.max(0) + radial + padding, np.inf),
        )
        return BoundedObstacle(self.name, bounds, self.exterior_material)


@dataclass(frozen=True, eq=False)
class Mesh:
    """Shared mesh geometry; certification uses its actual transformed vertices."""

    vertices: np.ndarray
    triangles: np.ndarray

    def __post_init__(self):
        vertices = np.array(self.vertices, dtype=np.float64, copy=True)
        triangles = np.array(self.triangles, copy=True)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or not len(vertices)
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("mesh vertices must be finite [N,3]")
        if (
            triangles.ndim != 2
            or triangles.shape[1:] != (3,)
            or not np.issubdtype(triangles.dtype, np.integer)
        ):
            raise ValueError("mesh triangles must be integer [N,3]")
        if not len(triangles) or np.any(triangles < 0) or np.any(triangles >= len(vertices)):
            raise ValueError("mesh triangle indices are empty or out of range")
        vertices.flags.writeable = triangles.flags.writeable = False
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)


@dataclass(frozen=True, eq=False)
class ConvexMeshVolume:
    """A closed convex input mesh, retaining its planar physical boundaries.

    This is a certificate over the supplied triangles, not a fitted sphere or
    cylinder. Nonconvex, open, and inconsistently wound meshes are rejected.
    The exact boundary backend must still trace the original input mesh.
    """

    name: str
    mesh: Mesh
    material_inside: int
    material_outside: int

    def __post_init__(self):
        vertices, triangles = self.mesh.vertices, self.mesh.triangles
        bounds = Bounds(vertices.min(0), vertices.max(0))
        BoxVolume(self.name, bounds, self.material_inside, self.material_outside)
        edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
        forward = edges[:, 0] < edges[:, 1]
        _, inverse, counts = np.unique(
            np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
        )
        balance = np.bincount(inverse, weights=np.where(forward, 1, -1))
        if np.any(counts != 2) or np.any(balance != 0):
            raise ValueError("convex volume mesh must be closed and consistently wound")
        faces = vertices[triangles]
        normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths == 0):
            raise ValueError("convex volume mesh contains a degenerate triangle")
        normals /= lengths[:, None]
        center = vertices.mean(0)
        distances = np.einsum("ij,ij->i", faces[:, 0] - center, normals)
        if np.all(distances < 0):
            normals = -normals
            distances = -distances
        tolerance = 128 * np.finfo(float).eps * max(1.0, np.abs(vertices).max())
        if np.any(distances <= tolerance):
            raise ValueError("convex volume mesh has inconsistent faces or no interior")
        # Work about a local origin to avoid cancellation after translation.
        for start in range(0, len(vertices), 256):
            projection = (vertices[start : start + 256] - center) @ normals.T
            if np.any(projection > distances + tolerance):
                raise ValueError("volume mesh is not convex")
        # Inset beyond the exact planes, covering traversal transform errors.
        guard = 32 * np.finfo(np.float32).eps * max(1.0, np.abs(vertices).max())
        for field, value in (
            ("normals", normals),
            ("distances", distances - guard),
            ("center", center),
        ):
            value.flags.writeable = False
            object.__setattr__(self, field, value)
        object.__setattr__(self, "bounds", bounds)

    def encloses(self, bounds):
        center = (bounds.lower + bounds.upper) / 2 - self.center
        half = (bounds.upper - bounds.lower) / 2
        return bool(
            self.bounds.encloses(bounds)
            and np.all(self.normals @ center + np.abs(self.normals) @ half <= self.distances)
        )

    def seed_cells(self):
        # Largest uniform contraction of the bounding box about an interior
        # point. Plane support functions derive the size for any orientation.
        half = np.minimum(self.center - self.bounds.lower, self.bounds.upper - self.center)
        support = np.abs(self.normals) @ half
        factor = min(1.0, float(np.min(self.distances / support)))
        if factor <= 0:
            return ()
        # Leave room for the center +/- half arithmetic after translation.
        half *= factor * (1.0 - 1e-12)
        bounds = Bounds(self.center - half, self.center + half)
        return (bounds,) if self.encloses(bounds) else ()

    def descriptor(self):
        return ["convex_mesh", self.mesh.vertices.tolist(), self.mesh.triangles.tolist()]


@dataclass(frozen=True, eq=False)
class MeshInstance:
    name: str
    mesh: Mesh
    rotation: np.ndarray
    translation: np.ndarray
    exterior_material: int

    def __post_init__(self):
        rotation = np.array(self.rotation, dtype=np.float64, copy=True)
        if (
            rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=2e-6)
        ):
            raise ValueError("mesh instance requires an orthonormal rotation")
        rotation.flags.writeable = False
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", _vector(self.translation, "translation"))

    def obstacle(self):
        vertices = self.mesh.vertices @ self.rotation.T + self.translation
        magnitude = np.maximum(1.0, np.max(np.abs(vertices), axis=0))
        # A conservative float32 transform/triangle tolerance envelope. A
        # backend with different intersection tolerances must supply its own
        # certified BoundedObstacle instead (as the detector adapter does).
        padding = 32 * np.finfo(np.float32).eps * (magnitude + np.max(np.abs(self.mesh.vertices)))
        return BoundedObstacle(
            self.name,
            Bounds(
                np.nextafter(vertices.min(0) - padding, -np.inf),
                np.nextafter(vertices.max(0) + padding, np.inf),
            ),
            self.exterior_material,
        )


@dataclass(frozen=True)
class PrimitiveScene:
    volumes: tuple[BoxVolume | ConvexMeshVolume, ...]
    obstacles: tuple[BoundedObstacle, ...] = ()
    wires: tuple[WireArray, ...] = ()
    instances: tuple[MeshInstance, ...] = ()

    def __post_init__(self):
        for name in ("volumes", "obstacles", "wires", "instances"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        names = [
            p.name
            for group in (self.volumes, self.obstacles, self.wires, self.instances)
            for p in group
        ]
        if len(set(names)) != len(names):
            raise ValueError("primitive names must be unique")
        for i, a in enumerate(self.volumes):
            for b in self.volumes[i + 1 :]:
                if a.bounds.encloses(b.bounds) and b.bounds.encloses(a.bounds):
                    raise ValueError("coincident box volumes have ambiguous ownership")

    def bounded_obstacles(self):
        return (
            self.obstacles
            + tuple(wire.obstacle() for wire in self.wires)
            + tuple(instance.obstacle() for instance in self.instances)
        )
