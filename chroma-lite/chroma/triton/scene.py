"""Detector-independent, immutable host scene IR for Triton backends.

The compiler consumes the same flattened arrays as :class:`GPUGeometry`, but
does not import PyCUDA, Torch, Triton, or initialize a GPU context.  Triangle
IDs remain in Chroma's authoritative flattened namespace: optical metadata,
solid IDs, detector channels, and BVH leaves all refer to those IDs.

This module deliberately stops at a CPU artifact.  Device upload and transport
are separate lifecycle stages, so compiling or caching geometry is safe in a
CPU-only process.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, IntFlag
import hashlib
import struct
from typing import Any, Optional, Tuple

import numpy as np

from .bvh import build_packed_bvh


SCENE_IR_VERSION = 1
_CHILD_MASK = np.uint32(0x0FFFFFFF)


class SceneCompileError(ValueError):
    """Raised when a Chroma geometry cannot be represented without guessing."""


def _readonly_array(
    value: Any,
    dtype: Optional[Any] = None,
    *,
    name: str,
) -> np.ndarray:
    """Return an owning, C-contiguous, read-only NumPy snapshot."""

    try:
        result = np.array(value, dtype=dtype, order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SceneCompileError("%s cannot be converted to an array" % name) from exc
    result.flags.writeable = False
    return result


def _readonly_int32_array(value: Any, *, name: str) -> np.ndarray:
    """Snapshot an integer array without silently truncating its namespace."""

    try:
        source = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise SceneCompileError("%s cannot be converted to an array" % name) from exc
    if not np.issubdtype(source.dtype, np.integer):
        raise SceneCompileError("%s must contain integers" % name)
    if source.size:
        limits = np.iinfo(np.int32)
        if np.issubdtype(source.dtype, np.unsignedinteger):
            outside = np.max(source) > limits.max
        else:
            outside = np.min(source) < limits.min or np.max(source) > limits.max
        if outside:
            raise SceneCompileError("%s is outside the int32 namespace" % name)
    return _readonly_array(source, np.int32, name=name)


def _require_shape(name: str, value: np.ndarray, shape: Tuple[Any, ...]) -> None:
    if value.ndim != len(shape):
        raise SceneCompileError("%s must have shape %r" % (name, shape))
    for actual, expected in zip(value.shape, shape):
        if expected is not None and actual != expected:
            raise SceneCompileError("%s must have shape %r" % (name, shape))


def _packed_node_array(nodes: Any) -> np.ndarray:
    """Normalize Chroma record ``uint4`` or plain ``[N,4]`` packed nodes."""

    source = np.asarray(nodes)
    if source.dtype.fields is not None:
        names = source.dtype.names or ()
        if not all(axis in names for axis in ("x", "y", "z", "w")):
            raise SceneCompileError(
                "BVH record nodes must contain x, y, z, and w fields"
            )
        normalized = np.column_stack(
            [source[axis].astype(np.uint32, copy=False) for axis in "xyzw"]
        )
    else:
        if source.ndim != 2 or source.shape[1] != 4:
            raise SceneCompileError("BVH nodes must have shape (N, 4)")
        if not np.issubdtype(source.dtype, np.integer):
            raise SceneCompileError("BVH nodes must contain integer packed words")
        normalized = source.astype(np.uint32, copy=False)
    return _readonly_array(normalized, np.uint32, name="bvh.nodes")


def _validate_packed_nodes(nodes: np.ndarray, triangle_count: int) -> None:
    if nodes.shape[0] == 0:
        raise SceneCompileError("BVH must contain at least a root node")
    words = nodes[:, 3]
    child_count = words >> np.uint32(28)
    child = words & _CHILD_MASK
    leaves = child_count == 0
    # Legacy fixed-degree BVHs may contain zero-volume padding leaves.  Their
    # child word is still a valid triangle ID, so the same range check applies.
    if np.any(child[leaves].astype(np.uint64) >= np.uint64(triangle_count)):
        raise SceneCompileError("BVH leaf refers outside the triangle array")
    inner = ~leaves
    end = child[inner].astype(np.uint64) + child_count[inner].astype(np.uint64)
    if np.any(end > np.uint64(len(nodes))):
        raise SceneCompileError("BVH inner-node child range is outside the node array")


@dataclass(frozen=True)
class HostBVH:
    """Immutable snapshot of Chroma-compatible packed BVH topology."""

    nodes: np.ndarray
    world_origin: np.ndarray
    world_scale: np.float32
    layer_offsets: Tuple[int, ...]
    source: str

    def validate(self, triangle_count: int) -> None:
        _require_shape("bvh.nodes", self.nodes, (None, 4))
        _require_shape("bvh.world_origin", self.world_origin, (3,))
        if self.nodes.dtype != np.uint32:
            raise SceneCompileError("bvh.nodes must be uint32")
        if self.world_origin.dtype != np.float32:
            raise SceneCompileError("bvh.world_origin must be float32")
        if not np.isfinite(self.world_origin).all():
            raise SceneCompileError("bvh.world_origin must be finite")
        if not np.isfinite(self.world_scale) or float(self.world_scale) <= 0.0:
            raise SceneCompileError("bvh.world_scale must be finite and positive")
        if not self.layer_offsets or self.layer_offsets[0] != 0:
            raise SceneCompileError("bvh.layer_offsets must begin at zero")
        if any(
            left >= right
            for left, right in zip(self.layer_offsets, self.layer_offsets[1:])
        ):
            raise SceneCompileError("bvh.layer_offsets must be strictly increasing")
        if self.layer_offsets[-1] >= len(self.nodes):
            raise SceneCompileError("last BVH layer offset is outside the node array")
        _validate_packed_nodes(self.nodes, triangle_count)


@dataclass(frozen=True)
class HostDetector:
    """Detector mappings and response CDFs in Chroma's channel-index namespace."""

    solid_id_to_channel_index: np.ndarray
    channel_index_to_solid_id: np.ndarray
    channel_index_to_channel_type: np.ndarray
    channel_index_to_position: np.ndarray
    time_cdf_x: np.ndarray
    time_cdf_y: np.ndarray
    charge_cdf_x: np.ndarray
    charge_cdf_y: np.ndarray

    @property
    def channel_count(self) -> int:
        return int(self.channel_index_to_solid_id.size)

    def validate(self, solid_count: int) -> None:
        count = self.channel_count
        for name in (
            "solid_id_to_channel_index",
            "channel_index_to_solid_id",
            "channel_index_to_channel_type",
        ):
            value = getattr(self, name)
            if value.ndim != 1 or value.dtype != np.int32:
                raise SceneCompileError("detector.%s must be int32[N]" % name)
        if self.solid_id_to_channel_index.shape != (solid_count,):
            raise SceneCompileError(
                "detector.solid_id_to_channel_index must contain one entry per solid"
            )
        if self.channel_index_to_channel_type.shape != (count,):
            raise SceneCompileError("detector channel arrays have inconsistent lengths")
        if self.channel_index_to_position.shape != (count, 3):
            raise SceneCompileError("detector channel positions must have shape (N, 3)")
        if self.channel_index_to_position.dtype != np.float32:
            raise SceneCompileError("detector channel positions must be float32")
        if count:
            solid_ids = self.channel_index_to_solid_id
            if np.any(solid_ids < 0) or np.any(solid_ids >= solid_count):
                raise SceneCompileError("detector channel refers to an invalid solid")
            expected = np.arange(count, dtype=np.int32)
            if not np.array_equal(
                self.solid_id_to_channel_index[solid_ids], expected
            ):
                raise SceneCompileError(
                    "detector forward and reverse channel maps are inconsistent"
                )
        valid_forward = self.solid_id_to_channel_index >= 0
        if np.any(self.solid_id_to_channel_index[valid_forward] >= count):
            raise SceneCompileError("solid-to-channel map refers to an invalid channel")
        for stem in ("time", "charge"):
            x = getattr(self, stem + "_cdf_x")
            y = getattr(self, stem + "_cdf_y")
            if x.ndim != 1 or y.shape != x.shape or x.dtype != np.float32 or y.dtype != np.float32:
                raise SceneCompileError(
                    "detector %s CDF coordinates must be matching float32 vectors" % stem
                )
            if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise SceneCompileError("detector %s CDF must be finite and nonempty" % stem)
            # GPUGeometry stores detector response tables as float32.  Very
            # close, but distinct, source abscissae can therefore collapse to
            # equal values (the Detector default charge CDF does exactly
            # this).  Preserve that established representation while still
            # rejecting a decreasing axis or CDF.
            if np.any(np.diff(x) < 0.0) or np.any(np.diff(y) < 0.0):
                raise SceneCompileError("detector %s CDF must be monotone" % stem)


@dataclass(frozen=True)
class SceneFeatures:
    """Feature inventory used for fail-closed backend capability checks."""

    is_detector: bool
    optical_feature_mask: int
    surface_models: Tuple[int, ...]
    analytic_wireplane_count: int
    uses_existing_bvh: bool


@dataclass(frozen=True)
class HostTritonScene:
    """Complete immutable CPU scene artifact in flattened Chroma namespaces."""

    schema_version: int
    vertices: np.ndarray
    triangles: np.ndarray
    bvh: HostBVH
    material1_index: np.ndarray
    material2_index: np.ndarray
    surface_index: np.ndarray
    solid_id: np.ndarray
    triangle_channel_index: np.ndarray
    optics: Any
    detector: Optional[HostDetector]
    features: SceneFeatures
    fingerprint: str

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def material_count(self) -> int:
        return len(self.optics.materials.names)

    @property
    def surface_count(self) -> int:
        return len(self.optics.surfaces.names)

    @property
    def solid_count(self) -> int:
        if self.detector is not None:
            return int(self.detector.solid_id_to_channel_index.size)
        return 0 if self.solid_id.size == 0 else int(self.solid_id.max()) + 1

    @property
    def channel_count(self) -> int:
        return 0 if self.detector is None else self.detector.channel_count

    # Descriptive aliases make the global namespace explicit to downstream
    # kernels without duplicating storage.
    @property
    def triangle_material1_index(self) -> np.ndarray:
        return self.material1_index

    @property
    def triangle_material2_index(self) -> np.ndarray:
        return self.material2_index

    @property
    def triangle_surface_index(self) -> np.ndarray:
        return self.surface_index

    @property
    def triangle_solid_id(self) -> np.ndarray:
        return self.solid_id

    def validate(self) -> None:
        _require_shape("vertices", self.vertices, (None, 3))
        _require_shape("triangles", self.triangles, (None, 3))
        if self.vertices.dtype != np.float32 or self.triangles.dtype != np.int32:
            raise SceneCompileError("vertices/triangles must be float32/int32")
        if not np.isfinite(self.vertices).all():
            raise SceneCompileError("vertices must be finite")
        if self.triangle_count == 0:
            raise SceneCompileError("scene must contain at least one triangle")
        if self.vertex_count == 0:
            raise SceneCompileError("scene must contain at least one vertex")
        if np.any(self.triangles < 0) or np.any(self.triangles >= self.vertex_count):
            raise SceneCompileError("triangle index is outside the vertex array")
        for name in (
            "material1_index",
            "material2_index",
            "surface_index",
            "solid_id",
            "triangle_channel_index",
        ):
            value = getattr(self, name)
            if value.shape != (self.triangle_count,) or value.dtype != np.int32:
                raise SceneCompileError("%s must be int32[Ntriangle]" % name)
        if np.any(self.material1_index < 0) or np.any(
            self.material1_index >= self.material_count
        ):
            raise SceneCompileError("material1_index is outside the material table")
        if np.any(self.material2_index < 0) or np.any(
            self.material2_index >= self.material_count
        ):
            raise SceneCompileError("material2_index is outside the material table")
        if np.any(self.surface_index < -1) or np.any(
            self.surface_index >= self.surface_count
        ):
            raise SceneCompileError("surface_index is outside the surface table")
        if np.any(self.solid_id < 0):
            raise SceneCompileError("solid IDs must be non-negative")
        if self.detector is None:
            if np.any(self.triangle_channel_index != -1):
                raise SceneCompileError("non-detector triangles cannot carry channels")
        else:
            self.detector.validate(self.solid_count)
            expected_channels = self.detector.solid_id_to_channel_index[self.solid_id]
            if not np.array_equal(self.triangle_channel_index, expected_channels):
                raise SceneCompileError("triangle channel map disagrees with solid map")
        self.bvh.validate(self.triangle_count)
        if hasattr(self.optics, "validate"):
            self.optics.validate()
        if self.schema_version != SCENE_IR_VERSION:
            raise SceneCompileError("unsupported HostTritonScene schema version")


def _snapshot_bvh(geometry: Any, vertices: np.ndarray, triangles: np.ndarray) -> HostBVH:
    source_bvh = getattr(geometry, "bvh", None)
    if source_bvh is None:
        built = build_packed_bvh(vertices, triangles)
        result = HostBVH(
            nodes=_readonly_array(built.nodes, np.uint32, name="bvh.nodes"),
            world_origin=_readonly_array(
                built.world_origin, np.float32, name="bvh.world_origin"
            ),
            world_scale=np.float32(built.world_scale),
            layer_offsets=tuple(map(int, built.layer_offsets)),
            source="triton_cpu",
        )
    else:
        nodes = _packed_node_array(getattr(source_bvh, "nodes", None))
        if hasattr(source_bvh, "world_coords"):
            world_origin = source_bvh.world_coords.world_origin
            world_scale = source_bvh.world_coords.world_scale
        else:
            world_origin = getattr(source_bvh, "world_origin", None)
            world_scale = getattr(source_bvh, "world_scale", None)
        if world_origin is None or world_scale is None:
            raise SceneCompileError("existing BVH is missing world-coordinate metadata")
        layer_offsets = tuple(map(int, getattr(source_bvh, "layer_offsets", ())))
        result = HostBVH(
            nodes=nodes,
            world_origin=_readonly_array(
                world_origin, np.float32, name="bvh.world_origin"
            ),
            world_scale=np.float32(world_scale),
            layer_offsets=layer_offsets,
            source="geometry",
        )
    result.validate(len(triangles))
    return result


def _snapshot_detector(geometry: Any, solid_count: int) -> Optional[HostDetector]:
    marker = getattr(geometry, "solid_id_to_channel_index", None)
    if marker is None:
        return None
    required = (
        "solid_id_to_channel_index",
        "channel_index_to_solid_id",
        "channel_index_to_channel_type",
        "channel_index_to_position",
        "time_cdf",
        "charge_cdf",
    )
    missing = [name for name in required if not hasattr(geometry, name)]
    if missing:
        raise SceneCompileError(
            "detector is missing required fields: %s" % ", ".join(missing)
        )
    try:
        time_x, time_y = geometry.time_cdf
        charge_x, charge_y = geometry.charge_cdf
    except (TypeError, ValueError) as exc:
        raise SceneCompileError("detector time/charge CDFs must be coordinate pairs") from exc
    channel_positions = np.asarray(geometry.channel_index_to_position)
    # Detector.flatten() historically emits shape (0,) when no PMTs were
    # registered.  Normalize the empty case to the IR's unambiguous (0, 3)
    # layout without changing any nonempty detector data.
    if channel_positions.size == 0:
        channel_positions = np.empty((0, 3), dtype=np.float32)
    result = HostDetector(
        solid_id_to_channel_index=_readonly_int32_array(
            geometry.solid_id_to_channel_index,
            name="detector.solid_id_to_channel_index",
        ),
        channel_index_to_solid_id=_readonly_int32_array(
            geometry.channel_index_to_solid_id,
            name="detector.channel_index_to_solid_id",
        ),
        channel_index_to_channel_type=_readonly_int32_array(
            geometry.channel_index_to_channel_type,
            name="detector.channel_index_to_channel_type",
        ),
        channel_index_to_position=_readonly_array(
            channel_positions,
            np.float32,
            name="detector.channel_index_to_position",
        ),
        time_cdf_x=_readonly_array(time_x, np.float32, name="detector.time_cdf_x"),
        time_cdf_y=_readonly_array(time_y, np.float32, name="detector.time_cdf_y"),
        charge_cdf_x=_readonly_array(
            charge_x, np.float32, name="detector.charge_cdf_x"
        ),
        charge_cdf_y=_readonly_array(
            charge_y, np.float32, name="detector.charge_cdf_y"
        ),
    )
    result.validate(solid_count)
    return result


def _hash_update(digest: Any, label: str, value: Any) -> None:
    """Canonical recursive encoding for deterministic content fingerprints."""

    encoded_label = label.encode("utf8")
    digest.update(struct.pack("<I", len(encoded_label)))
    digest.update(encoded_label)
    if isinstance(value, np.ndarray):
        digest.update(b"A")
        dtype = value.dtype.str.encode("ascii")
        digest.update(struct.pack("<I", len(dtype)))
        digest.update(dtype)
        digest.update(struct.pack("<I", value.ndim))
        for extent in value.shape:
            digest.update(struct.pack("<Q", int(extent)))
        digest.update(np.ascontiguousarray(value).tobytes(order="C"))
    elif is_dataclass(value):
        digest.update(b"D")
        for field in fields(value):
            _hash_update(digest, field.name, getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        digest.update(b"T")
        digest.update(struct.pack("<Q", len(value)))
        for index, item in enumerate(value):
            _hash_update(digest, str(index), item)
    elif isinstance(value, (IntFlag, Enum)):
        _hash_update(digest, "enum", int(value.value))
    elif value is None:
        digest.update(b"N")
    elif isinstance(value, (bool, np.bool_)):
        digest.update(b"B\x01" if bool(value) else b"B\x00")
    elif isinstance(value, (int, np.integer)):
        digest.update(b"I" + struct.pack("<q", int(value)))
    elif isinstance(value, (float, np.floating)):
        digest.update(b"F" + struct.pack("<d", float(value)))
    elif isinstance(value, str):
        data = value.encode("utf8")
        digest.update(b"S" + struct.pack("<Q", len(data)) + data)
    else:
        raise SceneCompileError(
            "cannot fingerprint unsupported value %s at %s"
            % (type(value).__name__, label)
        )


def _scene_fingerprint(
    vertices: np.ndarray,
    triangles: np.ndarray,
    bvh: HostBVH,
    material1_index: np.ndarray,
    material2_index: np.ndarray,
    surface_index: np.ndarray,
    solid_id: np.ndarray,
    triangle_channel_index: np.ndarray,
    optics: Any,
    detector: Optional[HostDetector],
    features: SceneFeatures,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"chroma.triton.HostTritonScene\0")
    _hash_update(digest, "schema_version", SCENE_IR_VERSION)
    for name, value in (
        ("vertices", vertices),
        ("triangles", triangles),
        ("bvh", bvh),
        ("material1_index", material1_index),
        ("material2_index", material2_index),
        ("surface_index", surface_index),
        ("solid_id", solid_id),
        ("triangle_channel_index", triangle_channel_index),
        ("optics", optics),
        ("detector", detector),
        ("features", features),
    ):
        _hash_update(digest, name, value)
    return digest.hexdigest()


def compile_host_scene(
    geometry: Any,
    *,
    wavelengths: Optional[Any] = None,
    times: Optional[Any] = None,
) -> HostTritonScene:
    """Compile any flattened or flattenable Chroma Geometry/Detector.

    The input is flattened in-place only when it does not already expose a
    ``mesh``.  Every returned array owns its storage and is read-only, so later
    mutation of the legacy geometry cannot alter the artifact or its digest.
    """

    if geometry is None:
        raise SceneCompileError("geometry cannot be None")
    if not hasattr(geometry, "mesh"):
        flatten = getattr(geometry, "flatten", None)
        if not callable(flatten):
            raise SceneCompileError("geometry must be flattened or provide flatten()")
        try:
            flatten()
        except Exception as exc:
            raise SceneCompileError("geometry.flatten() failed") from exc
    mesh = getattr(geometry, "mesh", None)
    if mesh is None or not hasattr(mesh, "vertices") or not hasattr(mesh, "triangles"):
        raise SceneCompileError("flattened geometry must expose mesh vertices/triangles")

    vertices = _readonly_array(mesh.vertices, np.float32, name="vertices")
    triangles = _readonly_int32_array(mesh.triangles, name="triangles")
    _require_shape("vertices", vertices, (None, 3))
    _require_shape("triangles", triangles, (None, 3))
    triangle_count = int(triangles.shape[0])
    if triangle_count == 0:
        raise SceneCompileError("scene must contain at least one triangle")
    if len(vertices) == 0 or not np.isfinite(vertices).all():
        raise SceneCompileError("scene vertices must be nonempty and finite")
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise SceneCompileError("triangle index is outside the vertex array")

    metadata = {}
    for name in ("material1_index", "material2_index", "surface_index", "solid_id"):
        if not hasattr(geometry, name):
            raise SceneCompileError("flattened geometry is missing %s" % name)
        metadata[name] = _readonly_int32_array(getattr(geometry, name), name=name)
        if metadata[name].shape != (triangle_count,):
            raise SceneCompileError("%s must contain one value per triangle" % name)

    materials = getattr(geometry, "unique_materials", None)
    surfaces = getattr(geometry, "unique_surfaces", None)
    if materials is None or surfaces is None:
        raise SceneCompileError(
            "flattened geometry must expose unique_materials and unique_surfaces"
        )
    from .optics import compile_optical_tables

    try:
        optics = compile_optical_tables(
            tuple(materials), tuple(surfaces), wavelengths=wavelengths, times=times
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise SceneCompileError("optical table compilation failed: %s" % exc) from exc

    solid_id = metadata["solid_id"]
    solid_count = (
        len(getattr(geometry, "solids"))
        if hasattr(geometry, "solids")
        else int(solid_id.max()) + 1
    )
    if solid_count <= 0 or np.any(solid_id < 0) or np.any(solid_id >= solid_count):
        raise SceneCompileError("solid_id is outside the geometry solid table")
    detector = _snapshot_detector(geometry, solid_count)
    if detector is None:
        triangle_channel_index = _readonly_array(
            np.full(triangle_count, -1, dtype=np.int32),
            np.int32,
            name="triangle_channel_index",
        )
    else:
        triangle_channel_index = _readonly_array(
            detector.solid_id_to_channel_index[solid_id],
            np.int32,
            name="triangle_channel_index",
        )

    bvh = _snapshot_bvh(geometry, vertices, triangles)
    surface_models = tuple(
        sorted(set(map(int, np.asarray(optics.surfaces.model).tolist())))
    )
    features = SceneFeatures(
        is_detector=detector is not None,
        optical_feature_mask=int(optics.feature_mask),
        surface_models=surface_models,
        analytic_wireplane_count=len(getattr(geometry, "wireplanes", ()) or ()),
        uses_existing_bvh=getattr(geometry, "bvh", None) is not None,
    )
    fingerprint = _scene_fingerprint(
        vertices,
        triangles,
        bvh,
        metadata["material1_index"],
        metadata["material2_index"],
        metadata["surface_index"],
        solid_id,
        triangle_channel_index,
        optics,
        detector,
        features,
    )
    scene = HostTritonScene(
        schema_version=SCENE_IR_VERSION,
        vertices=vertices,
        triangles=triangles,
        bvh=bvh,
        material1_index=metadata["material1_index"],
        material2_index=metadata["material2_index"],
        surface_index=metadata["surface_index"],
        solid_id=solid_id,
        triangle_channel_index=triangle_channel_index,
        optics=optics,
        detector=detector,
        features=features,
        fingerprint=fingerprint,
    )
    scene.validate()
    return scene


# Short alias for callers that already operate in the Triton namespace.
compile_scene = compile_host_scene


__all__ = [
    "HostBVH",
    "HostDetector",
    "HostTritonScene",
    "SCENE_IR_VERSION",
    "SceneCompileError",
    "SceneFeatures",
    "compile_host_scene",
    "compile_scene",
]
