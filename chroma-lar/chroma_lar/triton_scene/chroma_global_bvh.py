"""Immutable snapshots of Chroma's exact flattened global mesh and BVH.

This module is deliberately host-only.  In particular, loading the reference
detector asks Chroma's cache for an existing BVH with ``auto_build_bvh=False``;
it will never create a CUDA context or silently substitute a different tree.
That distinction matters because leaf visitation order is an observable part
of Chroma's strict-``<`` nearest-hit tie behaviour.

Schema v4 carries three fingerprints.  ``traversal_sha256`` covers stable
geometry/certificate words and can be pinned across fresh Chroma processes.
``optical_semantics_sha256`` covers the exact fixed-wavelength float32 optical
words, their per-triangle assignments, and source-ordered analytic wire-plane
descriptors after replacing process-local raw IDs with canonical names.  It is
therefore invariant under a consistent permutation of Chroma's material and
surface tables.  ``sha256`` additionally covers raw table ordering and archive
provenance.  Together the name tables and optical snapshot permit an explicit,
fail-closed remap into the deterministic tables used by the specialized
Triton scene.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import argparse
import hashlib
import io
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .compiler import DEFAULT_CONFIG, _numpy2_chroma_linalg_compat


SCHEMA_VERSION = 4
ARCHIVE_MAGIC = "chroma-lar.chroma-global-bvh"
TARGET_MESH_MD5 = "9a4f1d85f358cfdbd759117fefb449a7"
# Filled from the checked Reflect3Wires cached ``source == geometry`` BVH.
# Unlike the full archive hash, this deliberately excludes Chroma's
# process-order-dependent optical table indices.
TARGET_TRAVERSAL_SHA256 = (
    "e4c38552d318d3c45e09b5204291fae3a0d8e54dc388f9514082dc13c1c2ea58"
)
# Filled from a schema-v4 CPU export of the checked Reflect3Wires detector.
# This fingerprint is independent of raw Chroma object-table IDs.
TARGET_OPTICAL_SEMANTICS_SHA256 = (
    "aae153bf619d231df5fc20053a789e8db194c0b2873d5d38c4bdc43549bf4003"
)
_CHILD_MASK = np.uint32(0x0FFFFFFF)
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class ChromaGlobalBVHArtifactError(ValueError):
    """Raised when an exact reference artifact is absent or inconsistent."""


def _freeze(value: Any, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.flags.writeable = False
    return result


def _packed_nodes(value: Any) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.fields is not None:
        required = ("x", "y", "z", "w")
        if not all(name in source.dtype.fields for name in required):
            raise ChromaGlobalBVHArtifactError(
                "structured BVH nodes must expose x/y/z/w words"
            )
        if source.ndim != 1:
            raise ChromaGlobalBVHArtifactError(
                "structured BVH nodes must be one-dimensional"
            )
        result = np.empty((len(source), 4), dtype=np.uint32)
        for column, name in enumerate(required):
            field = np.asarray(source[name])
            if not np.issubdtype(field.dtype, np.integer):
                raise ChromaGlobalBVHArtifactError(
                    f"BVH node field {name!r} is not integral"
                )
            result[:, column] = field.astype(np.uint32, copy=False)
        return _freeze(result, np.uint32)
    if source.ndim != 2 or source.shape[1] != 4:
        raise ChromaGlobalBVHArtifactError("BVH nodes must have shape (N, 4)")
    if not np.issubdtype(source.dtype, np.integer):
        raise ChromaGlobalBVHArtifactError("BVH node words must be integral")
    return _freeze(source, np.uint32)


def _mesh_md5(vertices: np.ndarray, triangles: np.ndarray) -> str:
    # This is byte-for-byte Mesh.md5(), without depending on Chroma at import.
    digest = hashlib.md5()
    digest.update(memoryview(vertices).cast("B"))
    digest.update(memoryview(triangles).cast("B"))
    return digest.hexdigest()


def _hash_text(digest: Any, name: str, value: str) -> None:
    encoded_name = name.encode("utf8")
    encoded_value = value.encode("utf8")
    digest.update(len(encoded_name).to_bytes(4, "little"))
    digest.update(encoded_name)
    digest.update(len(encoded_value).to_bytes(8, "little"))
    digest.update(encoded_value)


def _hash_array(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    _hash_text(digest, name + ".dtype", array.dtype.str)
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    contiguous = np.ascontiguousarray(array)
    # Python's memoryview cannot cast a multidimensional array with a zero
    # extent.  Its byte contribution is empty anyway, while dtype and shape
    # above still distinguish every empty representation.
    if contiguous.nbytes:
        digest.update(memoryview(contiguous).cast("B"))


def _hash_names(
    digest: Any, name: str, values: Sequence[Optional[str]]
) -> None:
    _hash_text(digest, name + ".count", str(len(values)))
    for index, value in enumerate(values):
        item = f"{name}[{index}]"
        _hash_text(digest, item + ".kind", "none" if value is None else "text")
        if value is not None:
            _hash_text(digest, item + ".value", value)


def _optical_object_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if name is None or not str(name):
        raise ChromaGlobalBVHArtifactError(
            "optical objects require stable, nonempty names"
        )
    return str(name)


_MATERIAL_ARRAY_FIELDS = (
    "material_refractive_index",
    "material_absorption_length",
    "material_scattering_length",
    "material_num_reemission_components",
)

_SURFACE_ARRAY_FIELDS = (
    "surface_models",
    "surface_transmissive",
    "surface_thickness",
    "surface_detect",
    "surface_absorb",
    "surface_reemit",
    "surface_reflect_diffuse",
    "surface_reflect_specular",
    "surface_eta",
    "surface_k",
    "surface_reemission_cdf",
    "surface_default_probability_sum",
    "surface_default_transmit_probability",
)

_WIRE_ARRAY_FIELDS = (
    "wire_source_index",
    "wire_origin",
    "wire_u",
    "wire_v",
    "wire_pitch",
    "wire_radius",
    "wire_umin",
    "wire_umax",
    "wire_vmin",
    "wire_vmax",
    "wire_v0",
    "wire_surface_index",
    "wire_material_outer_index",
    "wire_material_inner_index",
    "wire_color",
)


_ARRAY_FIELDS = (
    "nodes",
    "world_origin",
    "layer_offsets",
    "vertices",
    "triangles",
    "solid_id",
    "material1_index",
    "material2_index",
    "surface_index",
    "colors",
    "triangle_channel_index",
    "solid_id_to_channel_index",
    "channel_index_to_solid_id",
    *_MATERIAL_ARRAY_FIELDS,
    *_SURFACE_ARRAY_FIELDS,
    *_WIRE_ARRAY_FIELDS,
)


_TRAVERSAL_ARRAY_FIELDS = (
    "nodes",
    "world_origin",
    "layer_offsets",
    "vertices",
    "triangles",
    "solid_id",
    "colors",
    "triangle_channel_index",
    "solid_id_to_channel_index",
    "channel_index_to_solid_id",
)


def _traversal_sha256(artifact: "ChromaGlobalBVHArtifact") -> str:
    """Fingerprint every stable word observable by geometric traversal."""

    digest = hashlib.sha256()
    # Keep this domain/version independent of the container schema: adding an
    # archive-only field must not invalidate an unchanged geometry certificate.
    _hash_text(digest, "domain", ARCHIVE_MAGIC + ".traversal.v1")
    _hash_text(digest, "reachable_node_count", str(artifact.reachable_node_count))
    _hash_text(digest, "stack_capacity", str(artifact.stack_capacity))
    _hash_array(
        digest, "world_scale", np.asarray(artifact.world_scale, dtype=np.float32)
    )
    for name in _TRAVERSAL_ARRAY_FIELDS:
        _hash_array(digest, name, getattr(artifact, name))
    return digest.hexdigest()


def _canonical_name_order(values: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return UTF-8-name order and the raw-index to canonical-index map."""

    order = np.asarray(
        sorted(range(len(values)), key=lambda index: values[index].encode("utf8")),
        dtype=np.int64,
    )
    inverse = np.empty(len(values), dtype=np.int32)
    inverse[order] = np.arange(len(values), dtype=np.int32)
    return order, inverse


def _canonical_surface_order(
    values: Sequence[Optional[str]],
) -> tuple[np.ndarray, np.ndarray]:
    named_raw_indices = sorted(
        (index for index, value in enumerate(values) if value is not None),
        key=lambda index: str(values[index]).encode("utf8"),
    )
    order = np.asarray(named_raw_indices, dtype=np.int64)
    inverse = np.full(len(values), -1, dtype=np.int32)
    inverse[order] = np.arange(len(order), dtype=np.int32)
    return order, inverse


def _optical_semantics_sha256(artifact: "ChromaGlobalBVHArtifact") -> str:
    """Fingerprint fixed-wavelength optical behaviour, not raw object IDs.

    Material and named-surface rows are sorted by their UTF-8 names, and every
    triangle/wire reference is translated into that canonical namespace before
    hashing.  Wire descriptor order is retained because Chroma's ascending
    analytic-plane scan makes it observable when two wire hits tie.
    """

    digest = hashlib.sha256()
    _hash_text(digest, "domain", ARCHIVE_MAGIC + ".optical-semantics.v1")
    _hash_array(
        digest,
        "optical_wavelength_nm",
        np.asarray(artifact.optical_wavelength_nm, dtype=np.float32),
    )

    material_order, canonical_material = _canonical_name_order(
        artifact.material_names
    )
    canonical_material_names = tuple(
        artifact.material_names[int(index)] for index in material_order
    )
    _hash_names(digest, "canonical_material_names", canonical_material_names)
    for name in _MATERIAL_ARRAY_FIELDS:
        _hash_array(digest, name, getattr(artifact, name)[material_order])

    surface_order, canonical_surface = _canonical_surface_order(
        artifact.surface_names
    )
    canonical_surface_names = tuple(
        artifact.surface_names[int(index)] for index in surface_order
    )
    _hash_names(digest, "canonical_surface_names", canonical_surface_names)
    for name in _SURFACE_ARRAY_FIELDS:
        _hash_array(digest, name, getattr(artifact, name)[surface_order])

    _hash_array(
        digest,
        "triangle_material1_semantic_id",
        canonical_material[artifact.material1_index],
    )
    _hash_array(
        digest,
        "triangle_material2_semantic_id",
        canonical_material[artifact.material2_index],
    )
    triangle_surface = np.full(artifact.triangle_count, -1, dtype=np.int32)
    triangle_has_surface = artifact.surface_index >= 0
    triangle_surface[triangle_has_surface] = canonical_surface[
        artifact.surface_index[triangle_has_surface]
    ]
    _hash_array(digest, "triangle_surface_semantic_id", triangle_surface)

    for name in _WIRE_ARRAY_FIELDS:
        if name in {
            "wire_surface_index",
            "wire_material_outer_index",
            "wire_material_inner_index",
        }:
            continue
        _hash_array(digest, name, getattr(artifact, name))
    _hash_array(
        digest,
        "wire_surface_semantic_id",
        canonical_surface[artifact.wire_surface_index],
    )
    _hash_array(
        digest,
        "wire_material_outer_semantic_id",
        canonical_material[artifact.wire_material_outer_index],
    )
    _hash_array(
        digest,
        "wire_material_inner_semantic_id",
        canonical_material[artifact.wire_material_inner_index],
    )
    return digest.hexdigest()


def _artifact_sha256(artifact: "ChromaGlobalBVHArtifact") -> str:
    digest = hashlib.sha256()
    _hash_text(digest, "magic", ARCHIVE_MAGIC)
    _hash_text(digest, "schema_version", str(artifact.schema_version))
    _hash_text(digest, "source", artifact.source)
    _hash_text(digest, "config_name", artifact.config_name)
    _hash_text(digest, "mesh_md5", artifact.mesh_md5)
    _hash_text(digest, "mesh_triangle_dtype", artifact.mesh_triangle_dtype)
    _hash_text(digest, "traversal_sha256", artifact.traversal_sha256)
    _hash_text(
        digest,
        "optical_semantics_sha256",
        artifact.optical_semantics_sha256,
    )
    _hash_names(digest, "material_names", artifact.material_names)
    _hash_names(digest, "surface_names", artifact.surface_names)
    _hash_text(digest, "reachable_node_count", str(artifact.reachable_node_count))
    _hash_text(digest, "stack_capacity", str(artifact.stack_capacity))
    _hash_array(
        digest, "world_scale", np.asarray(artifact.world_scale, dtype=np.float32)
    )
    _hash_array(
        digest,
        "optical_wavelength_nm",
        np.asarray(artifact.optical_wavelength_nm, dtype=np.float32),
    )
    for name in _ARRAY_FIELDS:
        _hash_array(digest, name, getattr(artifact, name))
    return digest.hexdigest()


def _require_array(
    name: str,
    value: Any,
    dtype: Any,
    ndim: int,
    tail: tuple[int, ...] = (),
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ChromaGlobalBVHArtifactError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(dtype):
        raise ChromaGlobalBVHArtifactError(
            f"{name} must have dtype {np.dtype(dtype)}, got {value.dtype}"
        )
    if value.ndim != ndim or (tail and value.shape[-len(tail) :] != tail):
        shape = "N" + "".join(f",{part}" for part in tail)
        raise ChromaGlobalBVHArtifactError(f"{name} must have shape ({shape})")
    if not value.flags.c_contiguous or not value.flags.owndata:
        raise ChromaGlobalBVHArtifactError(
            f"{name} must be a C-contiguous owning array"
        )
    if value.flags.writeable:
        raise ChromaGlobalBVHArtifactError(f"{name} must be read-only")
    return value


def _validate_topology(nodes: np.ndarray, triangle_count: int) -> tuple[int, int]:
    if not len(nodes):
        raise ChromaGlobalBVHArtifactError("BVH must contain a root node")
    words = nodes[:, 3]
    child_count = words >> np.uint32(28)
    child = words & _CHILD_MASK
    leaves = child_count == 0
    if np.any(child[leaves].astype(np.uint64) >= np.uint64(triangle_count)):
        raise ChromaGlobalBVHArtifactError(
            "BVH leaf refers outside the global triangle array"
        )
    internal = ~leaves
    ends = child[internal].astype(np.uint64) + child_count[internal].astype(
        np.uint64
    )
    if np.any(ends > np.uint64(len(nodes))):
        raise ChromaGlobalBVHArtifactError(
            "BVH internal child range is outside the node array"
        )
    root_count = int(child_count[0])
    if root_count == 0:
        raise ChromaGlobalBVHArtifactError(
            "the Chroma global-BVH compatibility path requires an internal root"
        )

    # Follow exactly the eager-sibling/LIFO range structure used by mesh.h.
    # Accepting every internal node is also a structural upper bound on the
    # number of pending ranges for any ray.
    seen_nodes = np.zeros(len(nodes), dtype=np.bool_)
    seen_triangles = np.zeros(triangle_count, dtype=np.bool_)
    seen_nodes[0] = True
    stack = [(int(child[0]), root_count)]
    maximum_pending = 1
    while stack:
        first, count = stack.pop()
        for node_index in range(first, first + count):
            if seen_nodes[node_index]:
                raise ChromaGlobalBVHArtifactError(
                    "BVH reachable topology is cyclic or multiply referenced"
                )
            seen_nodes[node_index] = True
            count_here = int(child_count[node_index])
            child_here = int(child[node_index])
            if count_here:
                stack.append((child_here, count_here))
                maximum_pending = max(maximum_pending, len(stack))
            else:
                if seen_triangles[child_here]:
                    raise ChromaGlobalBVHArtifactError(
                        "BVH has duplicate reachable leaves for one triangle"
                    )
                seen_triangles[child_here] = True
    if not np.all(seen_triangles):
        missing = int(np.count_nonzero(~seen_triangles))
        raise ChromaGlobalBVHArtifactError(
            f"BVH is missing {missing} global triangle leaves"
        )
    return int(np.count_nonzero(seen_nodes)), int(maximum_pending)


@dataclass(frozen=True)
class ChromaGlobalBVHArtifact:
    """Exact host representation consumed by global compatibility traversal.

    Material and surface names are indexed exactly like Chroma's raw flattened
    arrays.  ``None`` is retained in ``surface_names`` even though flattened
    triangles use ``-1`` for that no-surface value.
    """

    schema_version: int
    source: str
    config_name: str
    mesh_md5: str
    mesh_triangle_dtype: str
    material_names: tuple[str, ...]
    surface_names: tuple[Optional[str], ...]
    optical_wavelength_nm: np.float32
    traversal_sha256: str
    optical_semantics_sha256: str
    sha256: str
    nodes: np.ndarray
    world_origin: np.ndarray
    world_scale: np.float32
    layer_offsets: np.ndarray
    vertices: np.ndarray
    triangles: np.ndarray
    solid_id: np.ndarray
    material1_index: np.ndarray
    material2_index: np.ndarray
    surface_index: np.ndarray
    colors: np.ndarray
    triangle_channel_index: np.ndarray
    solid_id_to_channel_index: np.ndarray
    channel_index_to_solid_id: np.ndarray
    material_refractive_index: np.ndarray
    material_absorption_length: np.ndarray
    material_scattering_length: np.ndarray
    material_num_reemission_components: np.ndarray
    surface_models: np.ndarray
    surface_transmissive: np.ndarray
    surface_thickness: np.ndarray
    surface_detect: np.ndarray
    surface_absorb: np.ndarray
    surface_reemit: np.ndarray
    surface_reflect_diffuse: np.ndarray
    surface_reflect_specular: np.ndarray
    surface_eta: np.ndarray
    surface_k: np.ndarray
    surface_reemission_cdf: np.ndarray
    surface_default_probability_sum: np.ndarray
    surface_default_transmit_probability: np.ndarray
    wire_source_index: np.ndarray
    wire_origin: np.ndarray
    wire_u: np.ndarray
    wire_v: np.ndarray
    wire_pitch: np.ndarray
    wire_radius: np.ndarray
    wire_umin: np.ndarray
    wire_umax: np.ndarray
    wire_vmin: np.ndarray
    wire_vmax: np.ndarray
    wire_v0: np.ndarray
    wire_surface_index: np.ndarray
    wire_material_outer_index: np.ndarray
    wire_material_inner_index: np.ndarray
    wire_color: np.ndarray
    reachable_node_count: int
    stack_capacity: int

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def nbytes(self) -> int:
        names_nbytes = sum(len(value.encode("utf8")) for value in self.material_names)
        names_nbytes += sum(
            0 if value is None else len(value.encode("utf8"))
            for value in self.surface_names
        )
        return int(
            sum(getattr(self, name).nbytes for name in _ARRAY_FIELDS)
            + names_nbytes
            + 4
        )

    def remap_optical_indices(
        self,
        target_material_names: Sequence[str],
        target_surface_names: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map raw Chroma optical IDs into one stable compiled scene table.

        Returns per-triangle material1, material2, and surface arrays.  Names
        must be unique on both sides; a missing or ambiguous identity is an
        error instead of a guessed numeric mapping.
        """

        self.validate()

        def lookup(values: Sequence[str], description: str) -> dict[str, int]:
            result: dict[str, int] = {}
            for index, value in enumerate(values):
                if not isinstance(value, str) or not value:
                    raise ChromaGlobalBVHArtifactError(
                        f"target {description} names must be nonempty strings"
                    )
                if value in result:
                    raise ChromaGlobalBVHArtifactError(
                        f"target {description} name {value!r} is ambiguous"
                    )
                result[value] = index
            return result

        target_materials = lookup(target_material_names, "material")
        target_surfaces = lookup(target_surface_names, "surface")
        try:
            material_remap = np.asarray(
                [target_materials[name] for name in self.material_names],
                dtype=np.int32,
            )
        except KeyError as exc:
            raise ChromaGlobalBVHArtifactError(
                f"target scene has no material named {exc.args[0]!r}"
            ) from exc
        surface_remap = np.full(len(self.surface_names), -1, dtype=np.int32)
        for raw_index, name in enumerate(self.surface_names):
            if name is None:
                continue
            try:
                surface_remap[raw_index] = target_surfaces[name]
            except KeyError as exc:
                raise ChromaGlobalBVHArtifactError(
                    f"target scene has no surface named {name!r}"
                ) from exc

        mapped_surface = np.full(self.triangle_count, -1, dtype=np.int32)
        present_surface = self.surface_index >= 0
        mapped_surface[present_surface] = surface_remap[
            self.surface_index[present_surface]
        ]
        return (
            _freeze(material_remap[self.material1_index], np.int32),
            _freeze(material_remap[self.material2_index], np.int32),
            _freeze(mapped_surface, np.int32),
        )

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ChromaGlobalBVHArtifactError(
                f"unsupported schema version {self.schema_version}"
            )
        if self.source != "geometry":
            raise ChromaGlobalBVHArtifactError(
                "exact compatibility requires a BVH whose source is 'geometry'"
            )
        if not isinstance(self.config_name, str) or not self.config_name:
            raise ChromaGlobalBVHArtifactError("config_name must be nonempty")
        if not isinstance(self.material_names, tuple) or any(
            not isinstance(name, str) or not name for name in self.material_names
        ):
            raise ChromaGlobalBVHArtifactError(
                "material_names must be a tuple of nonempty strings"
            )
        if len(set(self.material_names)) != len(self.material_names):
            raise ChromaGlobalBVHArtifactError("material names must be unique")
        if not isinstance(self.surface_names, tuple) or any(
            name is not None and (not isinstance(name, str) or not name)
            for name in self.surface_names
        ):
            raise ChromaGlobalBVHArtifactError(
                "surface_names must contain only nonempty strings or None"
            )
        named_surfaces = [name for name in self.surface_names if name is not None]
        if len(set(named_surfaces)) != len(named_surfaces):
            raise ChromaGlobalBVHArtifactError("surface names must be unique")
        if self.surface_names.count(None) > 1:
            raise ChromaGlobalBVHArtifactError(
                "surface_names may contain at most one None entry"
            )
        if not isinstance(self.mesh_md5, str) or not _HEX32.fullmatch(
            self.mesh_md5
        ):
            raise ChromaGlobalBVHArtifactError("mesh_md5 must be lowercase MD5 hex")
        try:
            source_triangle_dtype = np.dtype(self.mesh_triangle_dtype)
        except TypeError as exc:
            raise ChromaGlobalBVHArtifactError(
                "mesh_triangle_dtype is not a NumPy dtype"
            ) from exc
        if not np.issubdtype(source_triangle_dtype, np.integer):
            raise ChromaGlobalBVHArtifactError(
                "mesh_triangle_dtype must describe an integer dtype"
            )
        if not isinstance(self.traversal_sha256, str) or not _HEX64.fullmatch(
            self.traversal_sha256
        ):
            raise ChromaGlobalBVHArtifactError(
                "traversal_sha256 must be lowercase SHA-256 hex"
            )
        if not isinstance(
            self.optical_semantics_sha256, str
        ) or not _HEX64.fullmatch(self.optical_semantics_sha256):
            raise ChromaGlobalBVHArtifactError(
                "optical_semantics_sha256 must be lowercase SHA-256 hex"
            )
        if not isinstance(self.sha256, str) or not _HEX64.fullmatch(self.sha256):
            raise ChromaGlobalBVHArtifactError("sha256 must be lowercase SHA-256 hex")

        nodes = _require_array("nodes", self.nodes, np.uint32, 2, (4,))
        origin = _require_array(
            "world_origin", self.world_origin, np.float32, 1
        )
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ChromaGlobalBVHArtifactError(
                "world_origin must be a finite float32[3]"
            )
        if not isinstance(self.world_scale, np.float32):
            raise ChromaGlobalBVHArtifactError("world_scale must be a float32 scalar")
        if not np.isfinite(self.world_scale) or self.world_scale <= 0:
            raise ChromaGlobalBVHArtifactError(
                "world_scale must be finite and positive"
            )
        if not isinstance(self.optical_wavelength_nm, np.float32):
            raise ChromaGlobalBVHArtifactError(
                "optical_wavelength_nm must be a float32 scalar"
            )
        if (
            not np.isfinite(self.optical_wavelength_nm)
            or self.optical_wavelength_nm <= 0
        ):
            raise ChromaGlobalBVHArtifactError(
                "optical_wavelength_nm must be finite and positive"
            )
        offsets = _require_array(
            "layer_offsets", self.layer_offsets, np.int64, 1
        )
        if (
            not len(offsets)
            or offsets[0] != 0
            or np.any(np.diff(offsets) <= 0)
            or offsets[-1] >= len(nodes)
        ):
            raise ChromaGlobalBVHArtifactError(
                "layer_offsets must increase from zero within the node array"
            )

        vertices = _require_array("vertices", self.vertices, np.float32, 2, (3,))
        triangles = _require_array(
            "triangles", self.triangles, np.int32, 2, (3,)
        )
        if not len(vertices) or not np.isfinite(vertices).all():
            raise ChromaGlobalBVHArtifactError("vertices must be finite and nonempty")
        if not len(triangles):
            raise ChromaGlobalBVHArtifactError("triangles must be nonempty")
        if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
            raise ChromaGlobalBVHArtifactError(
                "triangle index is outside the global vertex array"
            )

        triangle_count = len(triangles)
        for name in (
            "solid_id",
            "material1_index",
            "material2_index",
            "surface_index",
            "triangle_channel_index",
        ):
            value = _require_array(name, getattr(self, name), np.int32, 1)
            if value.shape != (triangle_count,):
                raise ChromaGlobalBVHArtifactError(
                    f"{name} must contain one entry per triangle"
                )
        colors = _require_array("colors", self.colors, np.uint32, 1)
        if colors.shape != (triangle_count,):
            raise ChromaGlobalBVHArtifactError(
                "colors must contain one entry per triangle"
            )
        if np.any(self.solid_id < 0):
            raise ChromaGlobalBVHArtifactError("solid IDs cannot be negative")
        if np.any(self.material1_index < 0) or np.any(self.material2_index < 0):
            raise ChromaGlobalBVHArtifactError("material indices cannot be negative")
        if (
            not self.material_names
            or np.any(self.material1_index >= len(self.material_names))
            or np.any(self.material2_index >= len(self.material_names))
        ):
            raise ChromaGlobalBVHArtifactError(
                "material index is outside material_names"
            )
        if np.any(self.surface_index < -1):
            raise ChromaGlobalBVHArtifactError("surface indices cannot be below -1")
        if np.any(self.surface_index >= len(self.surface_names)):
            raise ChromaGlobalBVHArtifactError(
                "surface index is outside surface_names"
            )
        present_triangle_surfaces = self.surface_index[self.surface_index >= 0]
        if any(
            self.surface_names[int(index)] is None
            for index in present_triangle_surfaces
        ):
            raise ChromaGlobalBVHArtifactError(
                "a nonnegative triangle surface index references None"
            )

        material_count = len(self.material_names)
        for name in (
            "material_refractive_index",
            "material_absorption_length",
            "material_scattering_length",
        ):
            value = _require_array(name, getattr(self, name), np.float32, 1)
            if value.shape != (material_count,) or not np.isfinite(value).all():
                raise ChromaGlobalBVHArtifactError(
                    f"{name} must be finite float32[Nmaterial]"
                )
        material_components = _require_array(
            "material_num_reemission_components",
            self.material_num_reemission_components,
            np.int32,
            1,
        )
        if material_components.shape != (material_count,) or np.any(
            material_components < 0
        ):
            raise ChromaGlobalBVHArtifactError(
                "material_num_reemission_components must be nonnegative "
                "int32[Nmaterial]"
            )

        surface_count = len(self.surface_names)
        for name in ("surface_models", "surface_transmissive"):
            value = _require_array(name, getattr(self, name), np.int32, 1)
            if value.shape != (surface_count,):
                raise ChromaGlobalBVHArtifactError(
                    f"{name} must be int32[Nsurface]"
                )
        for name in _SURFACE_ARRAY_FIELDS[2:]:
            value = _require_array(name, getattr(self, name), np.float32, 1)
            if value.shape != (surface_count,) or not np.isfinite(value).all():
                raise ChromaGlobalBVHArtifactError(
                    f"{name} must be finite float32[Nsurface]"
                )
        named_surface_mask = np.asarray(
            [name is not None for name in self.surface_names], dtype=np.bool_
        )
        if np.any(self.surface_models[named_surface_mask] < 0):
            raise ChromaGlobalBVHArtifactError(
                "named surface models cannot be negative"
            )
        if np.any(
            (self.surface_transmissive[named_surface_mask] != 0)
            & (self.surface_transmissive[named_surface_mask] != 1)
        ):
            raise ChromaGlobalBVHArtifactError(
                "named surface transmissive flags must be zero or one"
            )
        none_surface_mask = ~named_surface_mask
        if np.any(self.surface_models[none_surface_mask] != -1) or np.any(
            self.surface_transmissive[none_surface_mask] != -1
        ):
            raise ChromaGlobalBVHArtifactError(
                "None surface rows must use -1 model/transmissive sentinels"
            )
        expected_probability_sum = (
            self.surface_detect
            + self.surface_absorb
            + self.surface_reflect_diffuse
            + self.surface_reflect_specular
        ).astype(np.float32)
        if not np.array_equal(
            self.surface_default_probability_sum, expected_probability_sum
        ):
            raise ChromaGlobalBVHArtifactError(
                "surface default probability sums are inconsistent"
            )
        expected_transmit = (
            np.float32(1.0) - expected_probability_sum
        ).astype(np.float32)
        if not np.array_equal(
            self.surface_default_transmit_probability, expected_transmit
        ):
            raise ChromaGlobalBVHArtifactError(
                "surface default transmission probabilities are inconsistent"
            )

        wire_count = len(self.wire_source_index)
        wire_source = _require_array(
            "wire_source_index", self.wire_source_index, np.int32, 1
        )
        if not np.array_equal(wire_source, np.arange(wire_count, dtype=np.int32)):
            raise ChromaGlobalBVHArtifactError(
                "wire_source_index must preserve the complete ascending source order"
            )
        for name in ("wire_origin", "wire_u", "wire_v"):
            value = _require_array(name, getattr(self, name), np.float32, 2, (3,))
            if value.shape != (wire_count, 3) or not np.isfinite(value).all():
                raise ChromaGlobalBVHArtifactError(
                    f"{name} must be finite float32[Nwire,3]"
                )
        for name in (
            "wire_pitch",
            "wire_radius",
            "wire_umin",
            "wire_umax",
            "wire_vmin",
            "wire_vmax",
            "wire_v0",
        ):
            value = _require_array(name, getattr(self, name), np.float32, 1)
            if value.shape != (wire_count,) or not np.isfinite(value).all():
                raise ChromaGlobalBVHArtifactError(
                    f"{name} must be finite float32[Nwire]"
                )
        for name in (
            "wire_surface_index",
            "wire_material_outer_index",
            "wire_material_inner_index",
        ):
            value = _require_array(name, getattr(self, name), np.int32, 1)
            if value.shape != (wire_count,):
                raise ChromaGlobalBVHArtifactError(
                    f"{name} must be int32[Nwire]"
                )
        wire_color = _require_array("wire_color", self.wire_color, np.uint32, 1)
        if wire_color.shape != (wire_count,):
            raise ChromaGlobalBVHArtifactError(
                "wire_color must be uint32[Nwire]"
            )
        if np.any(self.wire_pitch <= 0) or np.any(self.wire_radius <= 0):
            raise ChromaGlobalBVHArtifactError(
                "wire pitch and radius must be positive"
            )
        if np.any(self.wire_umin > self.wire_umax) or np.any(
            self.wire_vmin > self.wire_vmax
        ):
            raise ChromaGlobalBVHArtifactError(
                "wire finite extents must be ordered"
            )
        if np.any(self.wire_material_outer_index < 0) or np.any(
            self.wire_material_inner_index < 0
        ):
            raise ChromaGlobalBVHArtifactError(
                "wire material indices cannot be negative"
            )
        if np.any(self.wire_material_outer_index >= material_count) or np.any(
            self.wire_material_inner_index >= material_count
        ):
            raise ChromaGlobalBVHArtifactError(
                "wire material index is outside material_names"
            )
        if np.any(self.wire_surface_index < 0) or np.any(
            self.wire_surface_index >= surface_count
        ):
            raise ChromaGlobalBVHArtifactError(
                "wire surface index is outside surface_names"
            )
        if any(
            self.surface_names[int(index)] is None
            for index in self.wire_surface_index
        ):
            raise ChromaGlobalBVHArtifactError(
                "a wire surface index references None"
            )

        forward = _require_array(
            "solid_id_to_channel_index",
            self.solid_id_to_channel_index,
            np.int32,
            1,
        )
        reverse = _require_array(
            "channel_index_to_solid_id",
            self.channel_index_to_solid_id,
            np.int32,
            1,
        )
        if not len(forward) or np.any(self.solid_id >= len(forward)):
            raise ChromaGlobalBVHArtifactError(
                "triangle solid ID is outside the solid/channel map"
            )
        if np.any(forward < -1) or np.any(forward >= len(reverse)):
            raise ChromaGlobalBVHArtifactError(
                "solid-to-channel map contains an invalid channel"
            )
        if np.any(reverse < 0) or np.any(reverse >= len(forward)):
            raise ChromaGlobalBVHArtifactError(
                "channel-to-solid map contains an invalid solid"
            )
        if len(reverse):
            expected_channels = np.arange(len(reverse), dtype=np.int32)
            if not np.array_equal(forward[reverse], expected_channels):
                raise ChromaGlobalBVHArtifactError(
                    "solid/channel maps are not mutual inverses"
                )
        if not np.array_equal(
            self.triangle_channel_index, forward[self.solid_id]
        ):
            raise ChromaGlobalBVHArtifactError(
                "triangle channel IDs disagree with triangle solid IDs"
            )

        source_triangles = np.ascontiguousarray(
            triangles.astype(source_triangle_dtype, copy=False)
        )
        if _mesh_md5(vertices, source_triangles) != self.mesh_md5:
            raise ChromaGlobalBVHArtifactError(
                "mesh MD5 does not match flattened vertex/triangle words"
            )
        reachable, capacity = _validate_topology(nodes, triangle_count)
        if reachable != self.reachable_node_count:
            raise ChromaGlobalBVHArtifactError(
                "reachable_node_count does not match BVH topology"
            )
        if capacity != self.stack_capacity:
            raise ChromaGlobalBVHArtifactError(
                "stack_capacity does not match BVH topology"
            )
        if _traversal_sha256(self) != self.traversal_sha256:
            raise ChromaGlobalBVHArtifactError(
                "traversal SHA-256 does not match its payload"
            )
        if _optical_semantics_sha256(self) != self.optical_semantics_sha256:
            raise ChromaGlobalBVHArtifactError(
                "optical-semantics SHA-256 does not match its payload"
            )
        if _artifact_sha256(self) != self.sha256:
            raise ChromaGlobalBVHArtifactError(
                "artifact SHA-256 does not match its payload"
            )


def _source_bvh(detector: Any) -> tuple[np.ndarray, np.ndarray, np.float32, np.ndarray]:
    bvh = getattr(detector, "bvh", None)
    if bvh is None:
        raise ChromaGlobalBVHArtifactError(
            "detector has no existing BVH; refusing to build a replacement"
        )
    recorded_source = getattr(bvh, "source", "geometry")
    if recorded_source != "geometry":
        raise ChromaGlobalBVHArtifactError(
            f"exact compatibility requires source='geometry', got {recorded_source!r}"
        )
    nodes = _packed_nodes(getattr(bvh, "nodes", None))
    world_coords = getattr(bvh, "world_coords", None)
    if world_coords is None:
        origin = getattr(bvh, "world_origin", None)
        scale = getattr(bvh, "world_scale", None)
    else:
        origin = getattr(world_coords, "world_origin", None)
        scale = getattr(world_coords, "world_scale", None)
    if origin is None or scale is None:
        raise ChromaGlobalBVHArtifactError(
            "existing BVH is missing world origin/scale"
        )
    offsets = getattr(bvh, "layer_offsets", None)
    if offsets is None:
        raise ChromaGlobalBVHArtifactError(
            "existing BVH is missing layer offsets"
        )
    scale_array = np.asarray(scale)
    if scale_array.shape != ():
        raise ChromaGlobalBVHArtifactError("BVH world_scale must be scalar")
    return (
        nodes,
        _freeze(origin, np.float32),
        np.float32(scale_array),
        _freeze(offsets, np.int64),
    )


_SURFACE_PROPERTY_NAMES = (
    "detect",
    "absorb",
    "reemit",
    "reflect_diffuse",
    "reflect_specular",
    "eta",
    "k",
    "reemission_cdf",
)
_MISSING = object()


def _interp_optical_property(obj: Any, name: str, wavelength_nm: np.float32) -> np.float32:
    values = np.asarray(getattr(obj, name, None))
    if values.ndim != 2 or values.shape[1] != 2 or not len(values):
        object_name = _optical_object_name(obj)
        raise ChromaGlobalBVHArtifactError(
            f"{object_name}.{name} is not a nonempty wavelength table"
        )
    if not np.isfinite(values).all():
        object_name = _optical_object_name(obj)
        raise ChromaGlobalBVHArtifactError(
            f"{object_name}.{name} contains a non-finite table word"
        )
    # This deliberately matches compiler._interp_property followed by the
    # OpticalTables float32 cast.
    return np.float32(np.interp(float(wavelength_nm), values[:, 0], values[:, 1]))


def _wire_get(wire: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(wire, Mapping):
        value = wire.get(name, _MISSING)
    else:
        value = getattr(wire, name, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise ChromaGlobalBVHArtifactError(
                f"wire-plane descriptor is missing {name}"
            )
        return default
    return value


def _append_identity_once(values: list[Any], lookup: dict[int, int], value: Any) -> None:
    if value is None:
        return
    key = id(value)
    if key not in lookup:
        lookup[key] = len(values)
        values.append(value)


def _optical_objects_and_lookups(
    detector: Any, wires: Sequence[Any]
) -> tuple[list[Any], list[Any], dict[int, int], dict[int, int]]:
    """Mirror GPUGeometry's raw tables, including wire-only objects."""

    if not hasattr(detector, "unique_materials"):
        raise ChromaGlobalBVHArtifactError(
            "flattened detector is missing unique_materials"
        )
    if not hasattr(detector, "unique_surfaces"):
        raise ChromaGlobalBVHArtifactError(
            "flattened detector is missing unique_surfaces"
        )
    materials = list(detector.unique_materials)
    surfaces = list(detector.unique_surfaces)
    if any(value is None for value in materials):
        raise ChromaGlobalBVHArtifactError("unique_materials cannot contain None")
    material_lookup = {id(value): index for index, value in enumerate(materials)}
    surface_lookup = {id(value): index for index, value in enumerate(surfaces)}
    if len(material_lookup) != len(materials):
        raise ChromaGlobalBVHArtifactError(
            "unique_materials contains a duplicate object identity"
        )
    if len(surface_lookup) != len(surfaces):
        raise ChromaGlobalBVHArtifactError(
            "unique_surfaces contains a duplicate object identity"
        )
    for wire in wires:
        _append_identity_once(
            materials, material_lookup, _wire_get(wire, "material_outer", None)
        )
        _append_identity_once(
            materials, material_lookup, _wire_get(wire, "material_inner", None)
        )
        _append_identity_once(
            surfaces, surface_lookup, _wire_get(wire, "surface", None)
        )
    return materials, surfaces, material_lookup, surface_lookup


def _resolve_wire_index(
    wire: Any,
    direct_name: str,
    object_name: str,
    objects: Sequence[Any],
    lookup: Mapping[int, int],
) -> int:
    direct = _wire_get(wire, direct_name, None)
    if direct is not None:
        try:
            index = int(direct)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ChromaGlobalBVHArtifactError(
                f"wire-plane {direct_name} is not an integer"
            ) from exc
    else:
        obj = _wire_get(wire, object_name, None)
        index = lookup.get(id(obj), -1) if obj is not None else -1
    if index < 0 or index >= len(objects):
        raise ChromaGlobalBVHArtifactError(
            f"wire-plane {object_name} is unresolved"
        )
    return index


def _optical_snapshot(
    detector: Any,
    wavelength_nm: np.float32,
) -> dict[str, Any]:
    wires = tuple(getattr(detector, "wireplanes", ()) or ())
    materials, surfaces, material_lookup, surface_lookup = (
        _optical_objects_and_lookups(detector, wires)
    )
    material_names_raw = tuple(_optical_object_name(value) for value in materials)
    material_names = tuple(str(value) for value in material_names_raw)
    surface_names = tuple(_optical_object_name(value) for value in surfaces)

    material_values = {
        "material_refractive_index": _freeze(
            [
                _interp_optical_property(value, "refractive_index", wavelength_nm)
                for value in materials
            ],
            np.float32,
        ),
        "material_absorption_length": _freeze(
            [
                _interp_optical_property(value, "absorption_length", wavelength_nm)
                for value in materials
            ],
            np.float32,
        ),
        "material_scattering_length": _freeze(
            [
                _interp_optical_property(value, "scattering_length", wavelength_nm)
                for value in materials
            ],
            np.float32,
        ),
        "material_num_reemission_components": _freeze(
            [len(getattr(value, "comp_reemission_prob", ())) for value in materials],
            np.int32,
        ),
    }

    surface_values: dict[str, list[Any]] = {
        "surface_models": [],
        "surface_transmissive": [],
        "surface_thickness": [],
        **{f"surface_{name}": [] for name in _SURFACE_PROPERTY_NAMES},
    }
    for surface in surfaces:
        if surface is None:
            surface_values["surface_models"].append(-1)
            surface_values["surface_transmissive"].append(-1)
            surface_values["surface_thickness"].append(0.0)
            for name in _SURFACE_PROPERTY_NAMES:
                surface_values[f"surface_{name}"].append(0.0)
            continue
        surface_values["surface_models"].append(int(getattr(surface, "model", 0)))
        surface_values["surface_transmissive"].append(
            int(getattr(surface, "transmissive", 0))
        )
        surface_values["surface_thickness"].append(
            np.float32(getattr(surface, "thickness", 0.0))
        )
        for name in _SURFACE_PROPERTY_NAMES:
            surface_values[f"surface_{name}"].append(
                _interp_optical_property(surface, name, wavelength_nm)
            )
    frozen_surface_values = {
        name: _freeze(
            values,
            np.int32 if name in {"surface_models", "surface_transmissive"} else np.float32,
        )
        for name, values in surface_values.items()
    }
    probability_sum = (
        frozen_surface_values["surface_detect"]
        + frozen_surface_values["surface_absorb"]
        + frozen_surface_values["surface_reflect_diffuse"]
        + frozen_surface_values["surface_reflect_specular"]
    ).astype(np.float32)
    frozen_surface_values["surface_default_probability_sum"] = _freeze(
        probability_sum, np.float32
    )
    frozen_surface_values["surface_default_transmit_probability"] = _freeze(
        np.float32(1.0) - probability_sum, np.float32
    )

    wire_values: dict[str, list[Any]] = {
        name: [] for name in _WIRE_ARRAY_FIELDS if name != "wire_source_index"
    }
    for wire in wires:
        wire_values["wire_origin"].append(_wire_get(wire, "origin"))
        wire_values["wire_u"].append(_wire_get(wire, "u"))
        wire_values["wire_v"].append(_wire_get(wire, "v"))
        wire_values["wire_pitch"].append(_wire_get(wire, "pitch"))
        wire_values["wire_radius"].append(_wire_get(wire, "radius"))
        wire_values["wire_umin"].append(_wire_get(wire, "umin", -1.0e9))
        wire_values["wire_umax"].append(_wire_get(wire, "umax", +1.0e9))
        wire_values["wire_vmin"].append(_wire_get(wire, "vmin", -1.0e9))
        wire_values["wire_vmax"].append(_wire_get(wire, "vmax", +1.0e9))
        wire_values["wire_v0"].append(_wire_get(wire, "v0", 0.0))
        wire_values["wire_surface_index"].append(
            _resolve_wire_index(
                wire, "surface_index", "surface", surfaces, surface_lookup
            )
        )
        wire_values["wire_material_outer_index"].append(
            _resolve_wire_index(
                wire,
                "material_outer_index",
                "material_outer",
                materials,
                material_lookup,
            )
        )
        wire_values["wire_material_inner_index"].append(
            _resolve_wire_index(
                wire,
                "material_inner_index",
                "material_inner",
                materials,
                material_lookup,
            )
        )
        wire_values["wire_color"].append(_wire_get(wire, "color", 0))
    frozen_wire_values: dict[str, np.ndarray] = {
        "wire_source_index": _freeze(np.arange(len(wires)), np.int32)
    }
    for name, values in wire_values.items():
        if name in {"wire_origin", "wire_u", "wire_v"}:
            source = np.asarray(values, dtype=np.float32).reshape((len(wires), 3))
            dtype = np.float32
        elif name == "wire_color":
            source, dtype = values, np.uint32
        elif name.endswith("_index"):
            source, dtype = values, np.int32
        else:
            source, dtype = values, np.float32
        frozen_wire_values[name] = _freeze(source, dtype)

    return {
        "material_names": material_names,
        "surface_names": surface_names,
        **material_values,
        **frozen_surface_values,
        **frozen_wire_values,
    }


def build_chroma_global_bvh_artifact(
    detector: Any,
    *,
    config_name: str = DEFAULT_CONFIG,
    wavelength_nm: float = 450.0,
) -> ChromaGlobalBVHArtifact:
    """Snapshot a flattened detector and its fixed-wavelength optical state."""

    try:
        optical_wavelength_nm = np.float32(wavelength_nm)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ChromaGlobalBVHArtifactError(
            "wavelength_nm must be a finite positive float32 value"
        ) from exc
    if not np.isfinite(optical_wavelength_nm) or optical_wavelength_nm <= 0:
        raise ChromaGlobalBVHArtifactError(
            "wavelength_nm must be a finite positive float32 value"
        )
    mesh = getattr(detector, "mesh", None)
    if mesh is None:
        raise ChromaGlobalBVHArtifactError(
            "detector must already be flattened before exact snapshotting"
        )
    source_vertices = np.asarray(getattr(mesh, "vertices", None))
    source_triangles = np.asarray(getattr(mesh, "triangles", None))
    vertices = _freeze(source_vertices, np.float32)
    triangles = _freeze(source_triangles, np.int32)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ChromaGlobalBVHArtifactError("mesh vertices must have shape (V, 3)")
    if triangles.ndim != 2 or triangles.shape[1:] != (3,):
        raise ChromaGlobalBVHArtifactError("mesh triangles must have shape (T, 3)")
    if not source_vertices.flags.c_contiguous or not source_triangles.flags.c_contiguous:
        raise ChromaGlobalBVHArtifactError(
            "source Chroma mesh arrays must be C-contiguous"
        )
    mesh_md5 = _mesh_md5(source_vertices, source_triangles)
    mesh_triangle_dtype = source_triangles.dtype.str
    source_md5 = getattr(mesh, "md5", None)
    if callable(source_md5) and str(source_md5()).lower() != mesh_md5:
        raise ChromaGlobalBVHArtifactError(
            "detector Mesh.md5() disagrees with its flattened words"
        )
    nodes, origin, scale, layer_offsets = _source_bvh(detector)

    def triangle_map(name: str, dtype: Any) -> np.ndarray:
        if not hasattr(detector, name):
            raise ChromaGlobalBVHArtifactError(
                f"flattened detector is missing {name}"
            )
        return _freeze(getattr(detector, name), dtype)

    solid_id = triangle_map("solid_id", np.int32)
    material1 = triangle_map("material1_index", np.int32)
    material2 = triangle_map("material2_index", np.int32)
    surface = triangle_map("surface_index", np.int32)
    colors = triangle_map("colors", np.uint32)
    optical_snapshot = _optical_snapshot(detector, optical_wavelength_nm)
    forward = _freeze(
        getattr(detector, "solid_id_to_channel_index", None), np.int32
    )
    reverse = _freeze(
        getattr(detector, "channel_index_to_solid_id", None), np.int32
    )
    triangle_channel = _freeze(forward[solid_id], np.int32)
    reachable, stack_capacity = _validate_topology(nodes, len(triangles))

    values = dict(
        schema_version=SCHEMA_VERSION,
        source="geometry",
        config_name=str(config_name),
        mesh_md5=mesh_md5,
        mesh_triangle_dtype=mesh_triangle_dtype,
        optical_wavelength_nm=optical_wavelength_nm,
        traversal_sha256="0" * 64,
        optical_semantics_sha256="0" * 64,
        sha256="0" * 64,
        nodes=nodes,
        world_origin=origin,
        world_scale=scale,
        layer_offsets=layer_offsets,
        vertices=vertices,
        triangles=triangles,
        solid_id=solid_id,
        material1_index=material1,
        material2_index=material2,
        surface_index=surface,
        colors=colors,
        triangle_channel_index=triangle_channel,
        solid_id_to_channel_index=forward,
        channel_index_to_solid_id=reverse,
        reachable_node_count=reachable,
        stack_capacity=stack_capacity,
        **optical_snapshot,
    )
    provisional = ChromaGlobalBVHArtifact(**values)
    values["traversal_sha256"] = _traversal_sha256(provisional)
    values["optical_semantics_sha256"] = _optical_semantics_sha256(
        ChromaGlobalBVHArtifact(**values)
    )
    values["sha256"] = _artifact_sha256(ChromaGlobalBVHArtifact(**values))
    artifact = ChromaGlobalBVHArtifact(**values)
    artifact.validate()
    return artifact


def _check_expected(
    artifact: ChromaGlobalBVHArtifact,
    expected_mesh_md5: Optional[str],
    expected_traversal_sha256: Optional[str],
    expected_optical_semantics_sha256: Optional[str],
    expected_sha256: Optional[str],
) -> None:
    if expected_mesh_md5 is not None and artifact.mesh_md5 != expected_mesh_md5:
        raise ChromaGlobalBVHArtifactError(
            "unexpected flattened mesh MD5: "
            f"expected {expected_mesh_md5}, got {artifact.mesh_md5}"
        )
    if (
        expected_traversal_sha256 is not None
        and artifact.traversal_sha256 != expected_traversal_sha256
    ):
        raise ChromaGlobalBVHArtifactError(
            "unexpected traversal SHA-256: "
            f"expected {expected_traversal_sha256}, "
            f"got {artifact.traversal_sha256}"
        )
    if (
        expected_optical_semantics_sha256 is not None
        and artifact.optical_semantics_sha256
        != expected_optical_semantics_sha256
    ):
        raise ChromaGlobalBVHArtifactError(
            "unexpected optical-semantics SHA-256: "
            f"expected {expected_optical_semantics_sha256}, "
            f"got {artifact.optical_semantics_sha256}"
        )
    if expected_sha256 is not None and artifact.sha256 != expected_sha256:
        raise ChromaGlobalBVHArtifactError(
            "unexpected artifact SHA-256: "
            f"expected {expected_sha256}, got {artifact.sha256}"
        )


def load_reflect3wires_chroma_global_bvh_artifact(
    config_name: str = DEFAULT_CONFIG,
    *,
    cache_dir: Optional[str | Path] = None,
    expected_mesh_md5: Optional[str] = TARGET_MESH_MD5,
    expected_traversal_sha256: Optional[str] = TARGET_TRAVERSAL_SHA256,
    expected_optical_semantics_sha256: Optional[str] = (
        TARGET_OPTICAL_SEMANTICS_SHA256
    ),
    expected_sha256: Optional[str] = None,
    wavelength_nm: float = 450.0,
    quiet: bool = True,
) -> ChromaGlobalBVHArtifact:
    """Load the target's existing cached Chroma BVH without using a GPU.

    A cache miss is an error.  Building a replacement would be especially
    dangerous here: NumPy's equal-Morton ordering and a different BVH builder
    can change Chroma's first-leaf-wins tie result.
    """

    from chroma.loader import load_bvh
    from chroma_lar.geometry.build_larcube import build_detector
    from chroma_lar.geometry.config_loader import load_config_from_file

    config = dict(load_config_from_file(config_name))
    detector_type = config.pop("detector_type", "wire")
    if detector_type != "wire":
        raise ChromaGlobalBVHArtifactError(
            "the reflect3wires global artifact requires the wire detector"
        )
    output = io.StringIO()
    stream = output if quiet else None
    with _numpy2_chroma_linalg_compat():
        if stream is None:
            detector = build_detector(**config, flatten=False)
            detector.flatten()
        else:
            with redirect_stdout(stream):
                detector = build_detector(**config, flatten=False)
                detector.flatten()

    actual_md5 = str(detector.mesh.md5()).lower()
    if expected_mesh_md5 is not None and actual_md5 != expected_mesh_md5:
        raise ChromaGlobalBVHArtifactError(
            "detector configuration does not match the expected flattened mesh: "
            f"expected {expected_mesh_md5}, got {actual_md5}"
        )
    cached = load_bvh(
        detector,
        auto_build_bvh=False,
        read_bvh_cache=True,
        update_bvh_cache=False,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )
    if cached is None:
        location = "the default Chroma cache" if cache_dir is None else str(cache_dir)
        raise ChromaGlobalBVHArtifactError(
            f"no existing Chroma BVH for mesh {actual_md5} in {location}; "
            "refusing to build a non-reference replacement"
        )
    detector.bvh = cached
    artifact = build_chroma_global_bvh_artifact(
        detector, config_name=str(config_name), wavelength_nm=wavelength_nm
    )
    _check_expected(
        artifact,
        expected_mesh_md5,
        expected_traversal_sha256,
        expected_optical_semantics_sha256,
        expected_sha256,
    )
    return artifact


_ARCHIVE_ARRAY_DTYPES = {
    "nodes": np.uint32,
    "world_origin": np.float32,
    "layer_offsets": np.int64,
    "vertices": np.float32,
    "triangles": np.int32,
    "solid_id": np.int32,
    "material1_index": np.int32,
    "material2_index": np.int32,
    "surface_index": np.int32,
    "colors": np.uint32,
    "triangle_channel_index": np.int32,
    "solid_id_to_channel_index": np.int32,
    "channel_index_to_solid_id": np.int32,
    "material_refractive_index": np.float32,
    "material_absorption_length": np.float32,
    "material_scattering_length": np.float32,
    "material_num_reemission_components": np.int32,
    "surface_models": np.int32,
    "surface_transmissive": np.int32,
    "surface_thickness": np.float32,
    "surface_detect": np.float32,
    "surface_absorb": np.float32,
    "surface_reemit": np.float32,
    "surface_reflect_diffuse": np.float32,
    "surface_reflect_specular": np.float32,
    "surface_eta": np.float32,
    "surface_k": np.float32,
    "surface_reemission_cdf": np.float32,
    "surface_default_probability_sum": np.float32,
    "surface_default_transmit_probability": np.float32,
    "wire_source_index": np.int32,
    "wire_origin": np.float32,
    "wire_u": np.float32,
    "wire_v": np.float32,
    "wire_pitch": np.float32,
    "wire_radius": np.float32,
    "wire_umin": np.float32,
    "wire_umax": np.float32,
    "wire_vmin": np.float32,
    "wire_vmax": np.float32,
    "wire_v0": np.float32,
    "wire_surface_index": np.int32,
    "wire_material_outer_index": np.int32,
    "wire_material_inner_index": np.int32,
    "wire_color": np.uint32,
}
_ARCHIVE_KEYS = frozenset(
    {
        "magic",
        "schema_version",
        "source",
        "config_name",
        "mesh_md5",
        "mesh_triangle_dtype",
        "material_names",
        "surface_names",
        "surface_name_is_none",
        "traversal_sha256",
        "optical_wavelength_nm",
        "optical_semantics_sha256",
        "sha256",
        "world_scale",
        "reachable_node_count",
        "stack_capacity",
        *_ARCHIVE_ARRAY_DTYPES,
    }
)


def save_chroma_global_bvh_artifact(
    artifact: ChromaGlobalBVHArtifact,
    path: str | Path,
    *,
    compressed: bool = False,
) -> Path:
    """Write a validated, pickle-free ``.npz`` artifact to exactly *path*."""

    artifact.validate()
    destination = Path(path)
    payload = {
        "magic": np.asarray(ARCHIVE_MAGIC),
        "schema_version": np.asarray(artifact.schema_version, dtype=np.int64),
        "source": np.asarray(artifact.source),
        "config_name": np.asarray(artifact.config_name),
        "mesh_md5": np.asarray(artifact.mesh_md5),
        "mesh_triangle_dtype": np.asarray(artifact.mesh_triangle_dtype),
        "material_names": np.asarray(artifact.material_names, dtype=np.str_),
        "surface_names": np.asarray(
            ["" if name is None else name for name in artifact.surface_names],
            dtype=np.str_,
        ),
        "surface_name_is_none": np.asarray(
            [name is None for name in artifact.surface_names], dtype=np.bool_
        ),
        "traversal_sha256": np.asarray(artifact.traversal_sha256),
        "optical_wavelength_nm": np.asarray(
            artifact.optical_wavelength_nm, dtype=np.float32
        ),
        "optical_semantics_sha256": np.asarray(
            artifact.optical_semantics_sha256
        ),
        "sha256": np.asarray(artifact.sha256),
        "world_scale": np.asarray(artifact.world_scale, dtype=np.float32),
        "reachable_node_count": np.asarray(
            artifact.reachable_node_count, dtype=np.int64
        ),
        "stack_capacity": np.asarray(artifact.stack_capacity, dtype=np.int64),
    }
    payload.update({name: getattr(artifact, name) for name in _ARCHIVE_ARRAY_DTYPES})
    writer = np.savez_compressed if compressed else np.savez
    with destination.open("wb") as output:
        writer(output, **payload)
    return destination


def _archive_scalar(archive: Any, name: str, dtype: Any = None) -> Any:
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ChromaGlobalBVHArtifactError(f"archive field {name} must be scalar")
    if dtype is not None and value.dtype != np.dtype(dtype):
        raise ChromaGlobalBVHArtifactError(
            f"archive field {name} must have dtype {np.dtype(dtype)}"
        )
    return value.item()


def _archive_names(archive: Any, name: str) -> tuple[str, ...]:
    value = np.asarray(archive[name])
    if value.ndim != 1 or value.dtype.kind != "U":
        raise ChromaGlobalBVHArtifactError(
            f"archive field {name} must be a one-dimensional Unicode array"
        )
    return tuple(str(item) for item in value.tolist())


def load_chroma_global_bvh_artifact(
    path: str | Path,
    *,
    expected_mesh_md5: Optional[str] = None,
    expected_traversal_sha256: Optional[str] = None,
    expected_optical_semantics_sha256: Optional[str] = None,
    expected_sha256: Optional[str] = None,
) -> ChromaGlobalBVHArtifact:
    """Read, fully validate, and fingerprint-check a saved host artifact."""

    source_path = Path(path)
    try:
        with np.load(source_path, allow_pickle=False) as archive:
            if frozenset(archive.files) != _ARCHIVE_KEYS:
                missing = sorted(_ARCHIVE_KEYS.difference(archive.files))
                extra = sorted(set(archive.files).difference(_ARCHIVE_KEYS))
                raise ChromaGlobalBVHArtifactError(
                    f"archive schema mismatch (missing={missing}, extra={extra})"
                )
            if _archive_scalar(archive, "magic") != ARCHIVE_MAGIC:
                raise ChromaGlobalBVHArtifactError("not a Chroma global-BVH artifact")
            arrays = {}
            for name, dtype in _ARCHIVE_ARRAY_DTYPES.items():
                raw = np.asarray(archive[name])
                if raw.dtype != np.dtype(dtype):
                    raise ChromaGlobalBVHArtifactError(
                        f"archive field {name} must have dtype {np.dtype(dtype)}"
                    )
                arrays[name] = _freeze(raw, dtype)
            world_scale = np.float32(
                _archive_scalar(archive, "world_scale", np.float32)
            )
            material_names = _archive_names(archive, "material_names")
            encoded_surface_names = _archive_names(archive, "surface_names")
            surface_none = np.asarray(archive["surface_name_is_none"])
            if (
                surface_none.dtype != np.dtype(np.bool_)
                or surface_none.ndim != 1
                or surface_none.shape != (len(encoded_surface_names),)
            ):
                raise ChromaGlobalBVHArtifactError(
                    "archive field surface_name_is_none must be bool[Nsurface]"
                )
            if any(
                bool(is_none) and name != ""
                for name, is_none in zip(encoded_surface_names, surface_none)
            ):
                raise ChromaGlobalBVHArtifactError(
                    "None surface names must use the canonical empty encoding"
                )
            surface_names = tuple(
                None if bool(is_none) else name
                for name, is_none in zip(encoded_surface_names, surface_none)
            )
            artifact = ChromaGlobalBVHArtifact(
                schema_version=int(
                    _archive_scalar(archive, "schema_version", np.int64)
                ),
                source=str(_archive_scalar(archive, "source")),
                config_name=str(_archive_scalar(archive, "config_name")),
                mesh_md5=str(_archive_scalar(archive, "mesh_md5")),
                mesh_triangle_dtype=str(
                    _archive_scalar(archive, "mesh_triangle_dtype")
                ),
                material_names=material_names,
                surface_names=surface_names,
                optical_wavelength_nm=np.float32(
                    _archive_scalar(archive, "optical_wavelength_nm", np.float32)
                ),
                traversal_sha256=str(
                    _archive_scalar(archive, "traversal_sha256")
                ),
                optical_semantics_sha256=str(
                    _archive_scalar(archive, "optical_semantics_sha256")
                ),
                sha256=str(_archive_scalar(archive, "sha256")),
                world_scale=world_scale,
                reachable_node_count=int(
                    _archive_scalar(archive, "reachable_node_count", np.int64)
                ),
                stack_capacity=int(
                    _archive_scalar(archive, "stack_capacity", np.int64)
                ),
                **arrays,
            )
    except ChromaGlobalBVHArtifactError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise ChromaGlobalBVHArtifactError(
            f"cannot read Chroma global-BVH artifact {source_path}: {exc}"
        ) from exc
    artifact.validate()
    _check_expected(
        artifact,
        expected_mesh_md5,
        expected_traversal_sha256,
        expected_optical_semantics_sha256,
        expected_sha256,
    )
    return artifact


def export_chroma_global_bvh_artifact(
    detector: Any,
    path: str | Path,
    *,
    config_name: str = DEFAULT_CONFIG,
    wavelength_nm: float = 450.0,
    compressed: bool = False,
) -> ChromaGlobalBVHArtifact:
    """Build and save an artifact from an already loaded Chroma detector."""

    artifact = build_chroma_global_bvh_artifact(
        detector, config_name=config_name, wavelength_nm=wavelength_nm
    )
    save_chroma_global_bvh_artifact(artifact, path, compressed=compressed)
    return artifact


def export_reflect3wires_chroma_global_bvh_artifact(
    path: str | Path,
    *,
    cache_dir: Optional[str | Path] = None,
    expected_mesh_md5: Optional[str] = TARGET_MESH_MD5,
    expected_traversal_sha256: Optional[str] = TARGET_TRAVERSAL_SHA256,
    expected_optical_semantics_sha256: Optional[str] = (
        TARGET_OPTICAL_SEMANTICS_SHA256
    ),
    expected_sha256: Optional[str] = None,
    wavelength_nm: float = 450.0,
    compressed: bool = False,
    quiet: bool = True,
) -> ChromaGlobalBVHArtifact:
    """Load the cached target reference and save its exact host artifact."""

    artifact = load_reflect3wires_chroma_global_bvh_artifact(
        cache_dir=cache_dir,
        expected_mesh_md5=expected_mesh_md5,
        expected_traversal_sha256=expected_traversal_sha256,
        expected_optical_semantics_sha256=expected_optical_semantics_sha256,
        expected_sha256=expected_sha256,
        wavelength_nm=wavelength_nm,
        quiet=quiet,
    )
    save_chroma_global_bvh_artifact(artifact, path, compressed=compressed)
    return artifact


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export the existing cached Reflect3Wires Chroma global BVH"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--compressed", action="store_true")
    args = parser.parse_args(argv)
    artifact = export_reflect3wires_chroma_global_bvh_artifact(
        args.output, cache_dir=args.cache_dir, compressed=args.compressed, quiet=False
    )
    print(
        f"wrote {args.output}: mesh_md5={artifact.mesh_md5} "
        f"traversal_sha256={artifact.traversal_sha256} "
        f"optical_semantics_sha256={artifact.optical_semantics_sha256} "
        f"sha256={artifact.sha256} nodes={artifact.node_count} "
        f"triangles={artifact.triangle_count} stack={artifact.stack_capacity}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI.
    raise SystemExit(_main())


__all__ = [
    "ARCHIVE_MAGIC",
    "ChromaGlobalBVHArtifact",
    "ChromaGlobalBVHArtifactError",
    "SCHEMA_VERSION",
    "TARGET_MESH_MD5",
    "TARGET_OPTICAL_SEMANTICS_SHA256",
    "TARGET_TRAVERSAL_SHA256",
    "build_chroma_global_bvh_artifact",
    "export_chroma_global_bvh_artifact",
    "export_reflect3wires_chroma_global_bvh_artifact",
    "load_chroma_global_bvh_artifact",
    "load_reflect3wires_chroma_global_bvh_artifact",
    "save_chroma_global_bvh_artifact",
]
