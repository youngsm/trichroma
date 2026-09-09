"""Compile the reflect/reflect/three-wire detector into a compact SoA scene.

This module is a detector compiler, not a second hand-written description of
the detector.  It builds the unflattened reference Chroma object and derives:

* one canonical PMT mesh with exact per-triangle optical metadata;
* rigid transforms for the PMTs in the source-reachable optical component;
* analytic box boundaries for the cavity, active enclosure, and cathode;
* pre-orthonormalized FP64 frames for the reachable wire lattices.

The cathode and active enclosure in the target configuration are opaque.  A
photon starting on one side therefore cannot reach PMTs or wires on the other
side.  The artifact records the proof inputs and global channel IDs so this
dead-scene elimination remains auditable.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext, redirect_stdout
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import io
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_CONFIG = "detector_config_reflect_reflect3wires"
FACE_ORDER = ("-x", "+x", "-y", "+y", "-z", "+z")
_SURFACE_DEFAULT_PROBABILITIES = (
    "detect",
    "absorb",
    "reflect_diffuse",
    "reflect_specular",
)
_SURFACE_PROPERTIES = _SURFACE_DEFAULT_PROBABILITIES + (
    "reemit",
    "eta",
    "k",
    "reemission_cdf",
)


def _array(value: Any, dtype: np.dtype | str | None = None) -> np.ndarray:
    """Return a C-contiguous, owning array suitable for Torch zero-copy use."""

    return np.array(value, dtype=dtype, order="C", copy=True)


def _require_shape(name: str, value: np.ndarray, tail: Tuple[int, ...]) -> None:
    if value.ndim != len(tail) + 1 or value.shape[1:] != tail:
        raise ValueError(f"{name} must have shape (N, {', '.join(map(str, tail))})")


def _identity_index(objects: Iterable[Any], *, allow_none: bool = False):
    """Create a stable first-occurrence object table keyed by identity."""

    table = []
    lookup: Dict[int, int] = {}
    for obj in objects:
        if obj is None:
            if allow_none:
                continue
            raise ValueError("None is not a valid material")
        key = id(obj)
        if key not in lookup:
            lookup[key] = len(table)
            table.append(obj)
    return tuple(table), lookup


def _object_indices(values: Sequence[Any], lookup: Mapping[int, int], *, none=-1):
    return _array(
        [none if obj is None else lookup[id(obj)] for obj in values], np.int32
    )


def _object_name(obj: Any) -> str:
    name = getattr(obj, "name", None)
    return str(name) if name is not None else type(obj).__name__


def _interp_property(obj: Any, name: str, wavelength_nm: float) -> float:
    values = getattr(obj, name)
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"{_object_name(obj)}.{name} is not a wavelength table")
    return float(np.interp(wavelength_nm, values[:, 0], values[:, 1]))


def _surface_probability_sum(surface: Any, wavelength_nm: float) -> float:
    return sum(
        _interp_property(surface, field, wavelength_nm)
        for field in _SURFACE_DEFAULT_PROBABILITIES
    )


def _surface_is_opaque(surface: Any, wavelength_nm: float) -> bool:
    # Chroma's default surface model transmits only through the probability
    # remainder.  A sum of one leaves no crossing branch.  `transmissive` is a
    # separate opt-in for specialized models and must also be disabled.
    total = _surface_probability_sum(surface, wavelength_nm)
    return total >= 1.0 - 2.0e-6 and int(getattr(surface, "transmissive", 0)) == 0


@contextmanager
def _numpy2_chroma_linalg_compat():
    """Temporarily restore NumPy 1.x's ``np.linalg.linalg`` alias.

    Chroma's profile offset helper catches ``np.linalg.linalg.LinAlgError``.
    NumPy 2 removed that module alias, turning an expected singular-line
    fallback into an AttributeError.  Keeping the shim local allows the exact
    reference builder to run in the requested ``py310_torch`` environment
    without modifying Chroma itself.
    """

    existed = hasattr(np.linalg, "linalg")
    previous = getattr(np.linalg, "linalg", None)
    if not existed:
        np.linalg.linalg = np.linalg  # type: ignore[attr-defined]
    try:
        yield
    finally:
        if existed:
            np.linalg.linalg = previous  # type: ignore[attr-defined]
        else:
            delattr(np.linalg, "linalg")


@dataclass(frozen=True)
class OpticalTables:
    """Stable scene-wide indices and Chroma-interpolated optical constants.

    Every floating array is evaluated at the scene's single compile wavelength
    with the same linear interpolation and float32 cast as ``GPUGeometry``.
    """

    material_names: Tuple[str, ...]
    material_refractive_index: np.ndarray
    material_absorption_length: np.ndarray
    material_scattering_length: np.ndarray
    material_num_reemission_components: np.ndarray
    surface_names: Tuple[str, ...]
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

    def validate(self) -> None:
        nmaterial = len(self.material_names)
        nsurface = len(self.surface_names)
        for name in (
            "material_refractive_index",
            "material_absorption_length",
            "material_scattering_length",
        ):
            value = getattr(self, name)
            if value.shape != (nmaterial,) or value.dtype != np.float32:
                raise ValueError(f"tables.{name} must be float32[Nmaterial]")
        if (
            self.material_num_reemission_components.shape != (nmaterial,)
            or self.material_num_reemission_components.dtype != np.int32
        ):
            raise ValueError(
                "tables.material_num_reemission_components must be int32[Nmaterial]"
            )
        for name in ("surface_models", "surface_transmissive"):
            value = getattr(self, name)
            if value.shape != (nsurface,) or value.dtype != np.int32:
                raise ValueError(f"tables.{name} must be int32[Nsurface]")
        for name in (
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
        ):
            value = getattr(self, name)
            if value.shape != (nsurface,) or value.dtype != np.float32:
                raise ValueError(f"tables.{name} must be float32[Nsurface]")
        expected_sum = (
            self.surface_detect
            + self.surface_absorb
            + self.surface_reflect_diffuse
            + self.surface_reflect_specular
        ).astype(np.float32)
        if not np.array_equal(self.surface_default_probability_sum, expected_sum):
            raise ValueError("compiled surface probability sums are inconsistent")
        expected_transmit = (np.float32(1.0) - expected_sum).astype(np.float32)
        if not np.array_equal(
            self.surface_default_transmit_probability, expected_transmit
        ):
            raise ValueError("compiled default surface transmission is inconsistent")


@dataclass(frozen=True)
class CanonicalPMT:
    """The single exact local-space PMT mesh shared by all instances."""

    vertices: np.ndarray
    triangles: np.ndarray
    # Indices below are local to material_names/surface_names.  A surface of
    # -1 is Chroma's exact representation of ``surface=None``.
    material_names: Tuple[str, ...]
    surface_names: Tuple[str, ...]
    material1_index: np.ndarray
    material2_index: np.ndarray
    surface_index: np.ndarray
    # Scene-wide equivalents make this directly consumable by transport.
    scene_material1_index: np.ndarray
    scene_material2_index: np.ndarray
    scene_surface_index: np.ndarray
    colors: np.ndarray
    sha256: str

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    def validate(self) -> None:
        _require_shape("pmt.vertices", self.vertices, (3,))
        _require_shape("pmt.triangles", self.triangles, (3,))
        if self.vertices.dtype != np.float32:
            raise ValueError("canonical PMT vertices must be float32")
        if self.triangles.dtype != np.int32:
            raise ValueError("canonical PMT triangles must be int32")
        ntri = self.triangle_count
        for name in (
            "material1_index",
            "material2_index",
            "surface_index",
            "scene_material1_index",
            "scene_material2_index",
            "scene_surface_index",
            "colors",
        ):
            value = getattr(self, name)
            if value.shape != (ntri,):
                raise ValueError(f"pmt.{name} must contain one value per triangle")
        if self.colors.dtype != np.uint32:
            raise ValueError("canonical PMT colors must be uint32")
        if self.triangles.size and (
            self.triangles.min() < 0 or self.triangles.max() >= len(self.vertices)
        ):
            raise ValueError("canonical PMT triangle index is out of range")


@dataclass(frozen=True)
class PMTInstances:
    """Rigid instances retained in one optically connected half-detector."""

    channel_id: np.ndarray
    solid_id: np.ndarray
    object_to_world_rotation: np.ndarray
    object_to_world_translation: np.ndarray
    world_to_object_rotation: np.ndarray
    world_to_object_translation: np.ndarray
    inward_normal: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray

    @property
    def count(self) -> int:
        return int(self.channel_id.size)

    def validate(self) -> None:
        n = self.count
        for name in ("channel_id", "solid_id"):
            value = getattr(self, name)
            if value.shape != (n,) or value.dtype != np.int32:
                raise ValueError(f"instances.{name} must be int32[N]")
        for name in ("object_to_world_rotation", "world_to_object_rotation"):
            value = getattr(self, name)
            if value.shape != (n, 3, 3) or value.dtype != np.float32:
                raise ValueError(f"instances.{name} must be float32[N,3,3]")
        for name in (
            "object_to_world_translation",
            "world_to_object_translation",
            "inward_normal",
            "bounds_min",
            "bounds_max",
        ):
            value = getattr(self, name)
            if value.shape != (n, 3) or value.dtype != np.float32:
                raise ValueError(f"instances.{name} must be float32[N,3]")
        if not np.array_equal(np.sort(self.channel_id), self.channel_id):
            raise ValueError("PMT instances must retain ascending global channel order")
        identity = np.einsum(
            "nij,njk->nik",
            self.world_to_object_rotation,
            self.object_to_world_rotation,
        )
        if not np.allclose(identity, np.eye(3), rtol=0.0, atol=2e-6):
            raise ValueError("PMT forward/inverse rotations are inconsistent")
        if np.any(self.bounds_min > self.bounds_max):
            raise ValueError("PMT instance bounds are inverted")


@dataclass(frozen=True)
class AnalyticBoxes:
    """Reference mesh boxes represented as exact slab-intersection inputs."""

    kinds: Tuple[str, ...]
    solid_id: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    # The compatibility path retains the sixteen original world-space mesh
    # triangles per box.  Production uses the cheaper bounds above; lockstep
    # replay uses these words to reproduce Chroma's Moller-Trumbore rounding.
    triangle_vertices: np.ndarray
    triangle_face: np.ndarray
    material_inside_index: np.ndarray
    material_outside_index: np.ndarray
    surface_index: np.ndarray
    color: np.ndarray
    # Faces use FACE_ORDER.  Disabled faces are unreachable from the compiled
    # source component; the full bounds and metadata remain for audit/replay.
    reachable_face_mask: np.ndarray
    collision_enabled: np.ndarray

    @property
    def count(self) -> int:
        return len(self.kinds)

    def validate(self) -> None:
        n = self.count
        if self.bounds_min.shape != (n, 3) or self.bounds_max.shape != (n, 3):
            raise ValueError("box bounds must be [N,3]")
        if self.bounds_min.dtype != np.float32 or self.bounds_max.dtype != np.float32:
            raise ValueError("box bounds must preserve Chroma float32 coordinates")
        if self.triangle_vertices.shape != (n, 16, 3, 3):
            raise ValueError("box triangle vertices must be float32[N,16,3,3]")
        if self.triangle_vertices.dtype != np.float32:
            raise ValueError("box triangle vertices must preserve Chroma float32 words")
        if self.triangle_face.shape != (n, 16) or self.triangle_face.dtype != np.int8:
            raise ValueError("box triangle faces must be int8[N,16]")
        if np.any(self.triangle_face < 0) or np.any(self.triangle_face >= 6):
            raise ValueError("box triangle face index is outside [0,6)")
        for name in (
            "solid_id",
            "material_inside_index",
            "material_outside_index",
            "surface_index",
        ):
            value = getattr(self, name)
            if value.shape != (n,) or value.dtype != np.int32:
                raise ValueError(f"boxes.{name} must be int32[N]")
        if self.color.shape != (n,) or self.color.dtype != np.uint32:
            raise ValueError("boxes.color must be uint32[N]")
        if self.reachable_face_mask.shape != (n, 6):
            raise ValueError("box reachable_face_mask must be bool[N,6]")
        if self.collision_enabled.shape != (n,):
            raise ValueError("box collision_enabled must be bool[N]")


@dataclass(frozen=True)
class AnalyticWires:
    """FP64 periodic-cylinder frames with loop invariants precomputed."""

    source_wireplane_index: np.ndarray
    origin: np.ndarray
    raw_u: np.ndarray
    raw_v: np.ndarray
    u: np.ndarray
    v: np.ndarray
    n: np.ndarray
    pitch: np.ndarray
    inv_pitch: np.ndarray
    radius: np.ndarray
    radius2: np.ndarray
    diameter: np.ndarray
    pad_v: np.ndarray
    pad_n: np.ndarray
    umin: np.ndarray
    umax: np.ndarray
    vmin: np.ndarray
    vmax: np.ndarray
    v0: np.ndarray
    kmin: np.ndarray
    kmax: np.ndarray
    surface_index: np.ndarray
    material_outer_index: np.ndarray
    material_inner_index: np.ndarray
    color: np.ndarray

    @property
    def count(self) -> int:
        return int(self.source_wireplane_index.size)

    def validate(self) -> None:
        nwire = self.count
        for name in ("origin", "raw_u", "raw_v", "u", "v", "n"):
            value = getattr(self, name)
            if value.shape != (nwire, 3) or value.dtype != np.float64:
                raise ValueError(f"wires.{name} must be float64[N,3]")
        for name in (
            "pitch",
            "inv_pitch",
            "radius",
            "radius2",
            "diameter",
            "pad_v",
            "pad_n",
            "umin",
            "umax",
            "vmin",
            "vmax",
            "v0",
        ):
            value = getattr(self, name)
            if value.shape != (nwire,) or value.dtype != np.float64:
                raise ValueError(f"wires.{name} must be float64[N]")
        for name in (
            "source_wireplane_index",
            "kmin",
            "kmax",
            "surface_index",
            "material_outer_index",
            "material_inner_index",
        ):
            value = getattr(self, name)
            if value.shape != (nwire,) or value.dtype != np.int32:
                raise ValueError(f"wires.{name} must be int32[N]")
        basis = np.stack((self.u, self.v, self.n), axis=1)
        gram = np.einsum("nij,nkj->nik", basis, basis)
        if not np.allclose(gram, np.eye(3), rtol=0.0, atol=2e-15):
            raise ValueError("wire frame is not FP64 orthonormal")
        if np.any(self.kmin > self.kmax):
            raise ValueError("wire lattice has an empty integer extent")
        if not np.allclose(self.inv_pitch * self.pitch, 1.0, atol=1e-15):
            raise ValueError("wire inverse pitch is inconsistent")


@dataclass(frozen=True)
class Reachability:
    """Evidence and masks for exact optical connected-component pruning."""

    source_x_sign: int
    source_component_bounds_min: np.ndarray
    source_component_bounds_max: np.ndarray
    cathode_probability_sum: float
    active_probability_sum: float
    cathode_opaque: bool
    active_enclosure_opaque: bool
    kept_global_channel_id: np.ndarray
    discarded_global_channel_id: np.ndarray
    kept_wireplane_index: np.ndarray
    discarded_wireplane_index: np.ndarray

    def validate(self) -> None:
        if self.source_x_sign not in (-1, 1):
            raise ValueError("source_x_sign must be -1 or +1")
        if self.source_component_bounds_min.shape != (3,) or self.source_component_bounds_max.shape != (3,):
            raise ValueError("source component bounds must be 3-vectors")
        if np.any(self.source_component_bounds_min >= self.source_component_bounds_max):
            raise ValueError("source component bounds are empty")
        if not self.cathode_opaque:
            raise ValueError("opposite detector half cannot be pruned through a transmitting cathode")
        if not self.active_enclosure_opaque:
            raise ValueError("target specialization requires an opaque active enclosure")
        kept = set(map(int, self.kept_global_channel_id))
        discarded = set(map(int, self.discarded_global_channel_id))
        if kept & discarded:
            raise ValueError("kept and discarded channel sets overlap")


def _dataclass_arrays(value: Any, prefix: str, output: Dict[str, Any]) -> None:
    if is_dataclass(value):
        for field in fields(value):
            child = getattr(value, field.name)
            key = f"{prefix}_{field.name}" if prefix else field.name
            _dataclass_arrays(child, key, output)
    elif isinstance(value, np.ndarray):
        output[prefix] = value
    elif isinstance(value, tuple) and all(isinstance(x, str) for x in value):
        output[prefix] = value
    elif np.isscalar(value) or isinstance(value, str):
        output[prefix] = value


@dataclass(frozen=True)
class CompiledReflect3WiresScene:
    """Complete CPU artifact consumed by detector-specialized Triton kernels."""

    config_name: str
    wavelength_nm: float
    total_reference_solids: int
    total_reference_channels: int
    tables: OpticalTables
    pmt: CanonicalPMT
    instances: PMTInstances
    boxes: AnalyticBoxes
    wires: AnalyticWires
    reachability: Reachability

    def validate(self) -> None:
        self.tables.validate()
        self.pmt.validate()
        self.instances.validate()
        self.boxes.validate()
        self.wires.validate()
        self.reachability.validate()
        if not np.array_equal(
            self.instances.channel_id, self.reachability.kept_global_channel_id
        ):
            raise ValueError("instance channels disagree with reachability record")
        if not np.array_equal(
            self.wires.source_wireplane_index,
            self.reachability.kept_wireplane_index,
        ):
            raise ValueError("wire descriptors disagree with reachability record")
        if self.total_reference_channels != (
            len(self.reachability.kept_global_channel_id)
            + len(self.reachability.discarded_global_channel_id)
        ):
            raise ValueError("reachability channel partition is incomplete")

    def as_dict(self) -> Dict[str, Any]:
        """Return a flat, allocation-free NumPy-friendly mapping."""

        result: Dict[str, Any] = {}
        _dataclass_arrays(self, "", result)
        return result

    def to_torch(self, device: Any = None) -> Dict[str, Any]:
        """Return the flat artifact with NumPy arrays converted to tensors.

        Importing this package does not require Torch.  The dependency is
        resolved only when this method is called.  CPU tensors share storage
        with the artifact arrays; moving to CUDA naturally allocates there.
        """

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("to_torch requires PyTorch") from exc

        output: Dict[str, Any] = {}
        for name, value in self.as_dict().items():
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value)
                output[name] = tensor.to(device=device) if device is not None else tensor
            else:
                output[name] = value
        return output


def _collect_scene_tables(geometry: Any, wavelength_nm: float):
    # Repeated PMT instances share one Solid object.  Scan each unique Solid
    # once, in first scene occurrence, to avoid 162 redundant 5k-triangle walks.
    material_stream = []
    surface_stream = []
    seen_solids = set()
    for solid in geometry.solids:
        if id(solid) in seen_solids:
            continue
        seen_solids.add(id(solid))
        material_stream.extend(solid.material1)
        material_stream.extend(solid.material2)
        surface_stream.extend(solid.surface)
    for wire in getattr(geometry, "wireplanes", ()):
        material_stream.append(wire["material_outer"])
        material_stream.append(wire["material_inner"])
        surface_stream.append(wire["surface"])

    material_objects, material_lookup = _identity_index(material_stream)
    surface_objects, surface_lookup = _identity_index(surface_stream, allow_none=True)
    material_refractive_index = _array(
        [
            _interp_property(obj, "refractive_index", wavelength_nm)
            for obj in material_objects
        ],
        np.float32,
    )
    material_absorption_length = _array(
        [
            _interp_property(obj, "absorption_length", wavelength_nm)
            for obj in material_objects
        ],
        np.float32,
    )
    material_scattering_length = _array(
        [
            _interp_property(obj, "scattering_length", wavelength_nm)
            for obj in material_objects
        ],
        np.float32,
    )

    compiled_surface_properties = {
        name: _array(
            [_interp_property(obj, name, wavelength_nm) for obj in surface_objects],
            np.float32,
        )
        for name in _SURFACE_PROPERTIES
    }
    default_probability_sum = (
        compiled_surface_properties["detect"]
        + compiled_surface_properties["absorb"]
        + compiled_surface_properties["reflect_diffuse"]
        + compiled_surface_properties["reflect_specular"]
    ).astype(np.float32)
    default_transmit_probability = (
        np.float32(1.0) - default_probability_sum
    ).astype(np.float32)

    tables = OpticalTables(
        material_names=tuple(map(_object_name, material_objects)),
        material_refractive_index=material_refractive_index,
        material_absorption_length=material_absorption_length,
        material_scattering_length=material_scattering_length,
        material_num_reemission_components=_array(
            [len(obj.comp_reemission_prob) for obj in material_objects], np.int32
        ),
        surface_names=tuple(map(_object_name, surface_objects)),
        surface_models=_array(
            [int(getattr(surface, "model", 0)) for surface in surface_objects],
            np.int32,
        ),
        surface_transmissive=_array(
            [int(getattr(surface, "transmissive", 0)) for surface in surface_objects],
            np.int32,
        ),
        surface_thickness=_array(
            [float(getattr(surface, "thickness", 0.0)) for surface in surface_objects],
            np.float32,
        ),
        surface_detect=compiled_surface_properties["detect"],
        surface_absorb=compiled_surface_properties["absorb"],
        surface_reemit=compiled_surface_properties["reemit"],
        surface_reflect_diffuse=compiled_surface_properties["reflect_diffuse"],
        surface_reflect_specular=compiled_surface_properties["reflect_specular"],
        surface_eta=compiled_surface_properties["eta"],
        surface_k=compiled_surface_properties["k"],
        surface_reemission_cdf=compiled_surface_properties["reemission_cdf"],
        surface_default_probability_sum=default_probability_sum,
        surface_default_transmit_probability=default_transmit_probability,
    )
    return tables, material_lookup, surface_lookup


def _canonical_pmt(solid: Any, material_lookup, surface_lookup) -> CanonicalPMT:
    local_materials, local_material_lookup = _identity_index(
        list(solid.material1) + list(solid.material2)
    )
    local_surfaces, local_surface_lookup = _identity_index(
        solid.surface, allow_none=True
    )

    vertices = _array(solid.mesh.vertices, np.float32)
    triangles = _array(solid.mesh.triangles, np.int32)
    material1 = _object_indices(solid.material1, local_material_lookup)
    material2 = _object_indices(solid.material2, local_material_lookup)
    surfaces = _object_indices(solid.surface, local_surface_lookup)
    colors = _array(solid.color, np.uint32)

    digest = hashlib.sha256()
    for value in (vertices, triangles, material1, material2, surfaces, colors):
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes(order="C"))
    for name in map(_object_name, local_materials):
        digest.update(name.encode("utf8") + b"\0")
    for name in map(_object_name, local_surfaces):
        digest.update(name.encode("utf8") + b"\0")

    return CanonicalPMT(
        vertices=vertices,
        triangles=triangles,
        material_names=tuple(map(_object_name, local_materials)),
        surface_names=tuple(map(_object_name, local_surfaces)),
        material1_index=material1,
        material2_index=material2,
        surface_index=surfaces,
        scene_material1_index=_object_indices(solid.material1, material_lookup),
        scene_material2_index=_object_indices(solid.material2, material_lookup),
        scene_surface_index=_object_indices(solid.surface, surface_lookup),
        colors=colors,
        sha256=digest.hexdigest(),
    )


def _locate_solids(config: Mapping[str, Any], geometry: Any):
    cursor = 0
    boxes: Dict[str, int] = {}
    if config.get("include_cavity", True):
        boxes["cavity"] = cursor
        cursor += 1

    channel_solid_ids = _array(geometry.channel_index_to_solid_id, np.int32)
    expected_pmt_ids = np.arange(cursor, cursor + len(channel_solid_ids), dtype=np.int32)
    if not np.array_equal(channel_solid_ids, expected_pmt_ids):
        raise ValueError("target detector no longer has one contiguous PMT solid block")
    cursor += len(channel_solid_ids)

    if config.get("include_active", True):
        boxes["active"] = cursor
        cursor += 1
    if config.get("include_cathode", True):
        boxes["cathode"] = cursor
        cursor += 1
    if cursor != len(geometry.solids):
        raise ValueError(
            "unexpected mesh solids after compiling PMTs/boxes; target specialization is stale"
        )
    if set(boxes) != {"cavity", "active", "cathode"}:
        raise ValueError("target compiler requires cavity, active enclosure, and cathode")
    return boxes, channel_solid_ids


def _compile_instances(
    geometry: Any,
    canonical: CanonicalPMT,
    channel_solid_ids: np.ndarray,
    source_x_sign: int,
    retain_all: bool = False,
):
    positions = np.stack(
        [geometry.solid_displacements[int(sid)] for sid in channel_solid_ids]
    ).astype(np.float32)
    channel_mask = np.sign(positions[:, 0]).astype(np.int8) == source_x_sign
    if retain_all:
        channel_mask[:] = True
    kept_channels = np.flatnonzero(channel_mask).astype(np.int32)
    discarded_channels = np.flatnonzero(~channel_mask).astype(np.int32)
    kept_solid_ids = channel_solid_ids[channel_mask]

    rotations = np.stack(
        [geometry.solid_rotations[int(sid)] for sid in kept_solid_ids]
    ).astype(np.float32)
    translations = positions[channel_mask].astype(np.float32, copy=True)
    inverse_rotations = np.transpose(rotations, (0, 2, 1)).copy()
    inverse_translations = -np.einsum(
        "nij,nj->ni", inverse_rotations, translations
    ).astype(np.float32)

    # The R5912 face is at local +Y (the body extends toward negative Y), so
    # the reference position generator's inward normals are the image of +Y.
    local_inward = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    inward = np.einsum("nij,j->ni", rotations, local_inward).astype(np.float32)

    bounds_min = np.empty((len(kept_channels), 3), dtype=np.float32)
    bounds_max = np.empty_like(bounds_min)
    for i, (rotation, translation) in enumerate(zip(rotations, translations)):
        world_vertices = canonical.vertices @ rotation.T + translation
        bounds_min[i] = world_vertices.min(axis=0)
        bounds_max[i] = world_vertices.max(axis=0)

    return (
        PMTInstances(
            channel_id=_array(kept_channels, np.int32),
            solid_id=_array(kept_solid_ids, np.int32),
            object_to_world_rotation=_array(rotations, np.float32),
            object_to_world_translation=_array(translations, np.float32),
            world_to_object_rotation=_array(inverse_rotations, np.float32),
            world_to_object_translation=_array(inverse_translations, np.float32),
            inward_normal=_array(inward, np.float32),
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        ),
        kept_channels,
        discarded_channels,
    )


def _uniform_identity(values: Sequence[Any], description: str) -> Any:
    first = values[0]
    if any(obj is not first for obj in values[1:]):
        raise ValueError(f"analytic {description} is not uniform over box triangles")
    return first


def _world_bounds(geometry: Any, solid_id: int):
    solid = geometry.solids[solid_id]
    rotation = geometry.solid_rotations[solid_id]
    translation = geometry.solid_displacements[solid_id]
    world = solid.mesh.vertices @ rotation.T + translation
    return world.min(axis=0).astype(np.float32), world.max(axis=0).astype(np.float32)


def _compile_boxes(
    geometry: Any,
    box_solid_ids: Mapping[str, int],
    material_lookup,
    surface_lookup,
    source_x_sign: int,
):
    kinds = ("cavity", "active", "cathode")
    solid_ids = []
    bounds_min = []
    bounds_max = []
    triangle_vertices = []
    triangle_faces = []
    material_inside = []
    material_outside = []
    surfaces = []
    colors = []

    for kind in kinds:
        solid_id = int(box_solid_ids[kind])
        solid = geometry.solids[solid_id]
        lo, hi = _world_bounds(geometry, solid_id)
        rotation = geometry.solid_rotations[solid_id]
        translation = geometry.solid_displacements[solid_id]
        # Geometry.flatten performs this operation into a float32 destination.
        # Round at the same point before gathering triangle vertices.
        world_vertices = np.asarray(
            np.inner(solid.mesh.vertices, rotation) + translation,
            dtype=np.float32,
        )
        world_triangles = np.ascontiguousarray(
            world_vertices[np.asarray(solid.mesh.triangles, dtype=np.int64)],
            dtype=np.float32,
        )
        if world_triangles.shape != (16, 3, 3):
            raise ValueError(f"analytic {kind} is no longer a 16-triangle box")
        edge01 = world_triangles[:, 1] - world_triangles[:, 0]
        edge12 = world_triangles[:, 2] - world_triangles[:, 1]
        raw_normal = np.cross(edge01, edge12)
        normal_axis = np.argmax(np.abs(raw_normal), axis=1)
        normal_sign = raw_normal[np.arange(16), normal_axis] > 0.0
        local_face = (2 * normal_axis + normal_sign.astype(np.int64)).astype(
            np.int8
        )
        solid_ids.append(solid_id)
        bounds_min.append(lo)
        bounds_max.append(hi)
        triangle_vertices.append(world_triangles)
        triangle_faces.append(local_face)
        m1 = _uniform_identity(solid.material1, f"{kind} inside material")
        m2 = _uniform_identity(solid.material2, f"{kind} outside material")
        surface = _uniform_identity(solid.surface, f"{kind} surface")
        color = solid.color[0]
        if np.any(solid.color != color):
            raise ValueError(f"analytic {kind} color is not uniform")
        material_inside.append(material_lookup[id(m1)])
        material_outside.append(material_lookup[id(m2)])
        surfaces.append(-1 if surface is None else surface_lookup[id(surface)])
        colors.append(color)

    face_mask = np.zeros((len(kinds), 6), dtype=np.bool_)
    # In exact real arithmetic the cavity is behind the opaque active
    # enclosure.  Chroma's flattened float32 geometry nevertheless reaches it
    # when a PMT triangle is coplanar with the active wall: after excluding the
    # PMT's last triangle, the active face is at t <= CHROMA_EPSILON and the
    # next full-BVH winner is a cavity face.  Retaining all six cavity faces is
    # therefore required for history parity and gives numerical escape rays
    # the same terminal reflect00 (100% absorb) boundary as Chroma.
    face_mask[kinds.index("cavity"), :] = True
    # The active box contributes the source-side X wall and all four Y/Z
    # walls.  Only the source-facing cathode face can be first-hit.
    active_x_face = 0 if source_x_sign < 0 else 1
    cathode_x_face = 0 if source_x_sign < 0 else 1
    face_mask[kinds.index("active"), [active_x_face, 2, 3, 4, 5]] = True
    face_mask[kinds.index("cathode"), cathode_x_face] = True
    collision_enabled = np.any(face_mask, axis=1)

    return AnalyticBoxes(
        kinds=kinds,
        solid_id=_array(solid_ids, np.int32),
        bounds_min=_array(bounds_min, np.float32),
        bounds_max=_array(bounds_max, np.float32),
        triangle_vertices=_array(triangle_vertices, np.float32),
        triangle_face=_array(triangle_faces, np.int8),
        material_inside_index=_array(material_inside, np.int32),
        material_outside_index=_array(material_outside, np.int32),
        surface_index=_array(surfaces, np.int32),
        color=_array(colors, np.uint32),
        reachable_face_mask=face_mask,
        collision_enabled=collision_enabled,
    )


def _orthonormal_wire_frame(wire: Mapping[str, Any]):
    # Deliberately mirrors fill_state's FP64 re-orthonormalization, starting
    # from the reference descriptor's float32 fields.
    u = np.asarray(wire["u"], dtype=np.float32).astype(np.float64)
    v_raw = np.asarray(wire["v"], dtype=np.float32).astype(np.float64)
    u /= np.sqrt(np.dot(u, u))
    v = v_raw - np.dot(v_raw, u) * u
    v /= np.sqrt(np.dot(v, v))
    normal = np.cross(u, v)
    return u, v, normal


def _compile_wires(
    geometry: Any,
    material_lookup,
    surface_lookup,
    source_x_sign: int,
    *,
    retain_all: bool = False,
):
    all_wires = tuple(getattr(geometry, "wireplanes", ()))
    keep = (
        np.ones(len(all_wires), dtype=np.bool_)
        if retain_all
        else np.array(
            [
                np.sign(float(wire["origin"][0])) == source_x_sign
                for wire in all_wires
            ],
            dtype=np.bool_,
        )
    )
    kept_indices = np.flatnonzero(keep).astype(np.int32)
    discarded_indices = np.flatnonzero(~keep).astype(np.int32)
    selected = [all_wires[int(index)] for index in kept_indices]

    origins = []
    raw_us, raw_vs = [], []
    us, vs, normals = [], [], []
    pitch, radius = [], []
    umin, umax, vmin, vmax, v0 = [], [], [], [], []
    surfaces, material_outer, material_inner, colors = [], [], [], []
    for wire in selected:
        raw_us.append(np.asarray(wire["u"], dtype=np.float32).astype(np.float64))
        raw_vs.append(np.asarray(wire["v"], dtype=np.float32).astype(np.float64))
        u, v, normal = _orthonormal_wire_frame(wire)
        origins.append(np.asarray(wire["origin"], dtype=np.float32).astype(np.float64))
        us.append(u)
        vs.append(v)
        normals.append(normal)
        pitch.append(float(wire["pitch"]))
        radius.append(float(wire["radius"]))
        umin.append(float(wire["umin"]))
        umax.append(float(wire["umax"]))
        vmin.append(float(wire["vmin"]))
        vmax.append(float(wire["vmax"]))
        v0.append(float(wire["v0"]))
        surfaces.append(surface_lookup[id(wire["surface"])])
        material_outer.append(material_lookup[id(wire["material_outer"])])
        material_inner.append(material_lookup[id(wire["material_inner"])])
        colors.append(wire["color"])

    # GPUGeometry serializes every WirePlane scalar as float32.  photon.h then
    # promotes those stored values to FP64 for intersection arithmetic.  Round
    # through float32 here as well; promoting the Python/config float directly
    # moves cylinder roots at nominal-radius boundary cases.
    def serialized_fp64(values):
        return np.asarray(values, dtype=np.float32).astype(np.float64)

    pitch_array = serialized_fp64(pitch)
    radius_array = serialized_fp64(radius)
    umin_array = serialized_fp64(umin)
    umax_array = serialized_fp64(umax)
    vmin_array = serialized_fp64(vmin)
    vmax_array = serialized_fp64(vmax)
    v0_array = serialized_fp64(v0)
    kmin = np.ceil((vmin_array - v0_array) / pitch_array).astype(np.int32)
    kmax = np.floor((vmax_array - v0_array) / pitch_array).astype(np.int32)
    diameter = 2.0 * radius_array
    pad = radius_array + 1.0e-6

    wires = AnalyticWires(
        source_wireplane_index=_array(kept_indices, np.int32),
        origin=_array(origins, np.float64),
        raw_u=_array(raw_us, np.float64),
        raw_v=_array(raw_vs, np.float64),
        u=_array(us, np.float64),
        v=_array(vs, np.float64),
        n=_array(normals, np.float64),
        pitch=pitch_array,
        inv_pitch=1.0 / pitch_array,
        radius=radius_array,
        radius2=radius_array * radius_array,
        diameter=diameter,
        pad_v=pad.copy(),
        pad_n=pad.copy(),
        umin=umin_array,
        umax=umax_array,
        vmin=vmin_array,
        vmax=vmax_array,
        v0=v0_array,
        kmin=kmin,
        kmax=kmax,
        surface_index=_array(surfaces, np.int32),
        material_outer_index=_array(material_outer, np.int32),
        material_inner_index=_array(material_inner, np.int32),
        color=_array(colors, np.uint32),
    )
    return wires, kept_indices, discarded_indices


def _component_bounds(boxes: AnalyticBoxes, source_x_sign: int):
    active_index = boxes.kinds.index("active")
    cathode_index = boxes.kinds.index("cathode")
    lower = boxes.bounds_min[active_index].copy()
    upper = boxes.bounds_max[active_index].copy()
    if source_x_sign < 0:
        upper[0] = boxes.bounds_min[cathode_index, 0]
    else:
        lower[0] = boxes.bounds_max[cathode_index, 0]
    return lower, upper


def _assert_target_signature(
    tables: OpticalTables,
    canonical: CanonicalPMT,
    instances: PMTInstances,
    wires: AnalyticWires,
    total_channels: int,
    *,
    retain_all_wires: bool = False,
) -> None:
    expected = (2564, 5120, 81, 6 if retain_all_wires else 3, 162)
    actual = (
        len(canonical.vertices),
        canonical.triangle_count,
        instances.count,
        wires.count,
        total_channels,
    )
    if actual != expected:
        raise ValueError(
            "detector_config_reflect_reflect3wires signature changed: "
            f"expected vertices/triangles/kept-PMTs/kept-wires/channels={expected}, "
            f"got {actual}"
        )
    if np.any(tables.material_num_reemission_components != 0):
        raise ValueError("target specialization requires no material reemission components")
    if np.any(tables.surface_models != 0):
        raise ValueError("target specialization requires Chroma's default surface model")


def compile_reflect3wires_scene(
    config_name: str = DEFAULT_CONFIG,
    source_x_sign: int = -1,
    *,
    wavelength_nm: float = 450.0,
    quiet: bool = True,
    retain_all_wires: bool = False,
    calibration=None,
    retain_all_geometry: bool = False,
) -> CompiledReflect3WiresScene:
    """Compile one source-reachable half of the target detector.

    Parameters
    ----------
    config_name:
        Module name under ``chroma_lar.config`` or a configuration file path.
    source_x_sign:
        ``-1`` for production sources in the negative-X component, ``+1`` for
        its exact mirror.  Global channel IDs are never renumbered.
    wavelength_nm:
        Wavelength used only to prove that pruning barriers have no transmit
        probability.  The production photon bomb uses 450 nm.
    quiet:
        Suppress informational prints in the legacy geometry builder.
    retain_all_wires:
        Retain all six analytic wire planes instead of pruning the three in
        the source-inaccessible detector half.  This is intended only for the
        exact global-BVH compatibility backend, whose public source center may
        lie in either half.  The production specialization keeps its original
        three-wire artifact and launch cost.
    calibration:
        Optional OpticalCalibration replacing every material and surface,
        including analytic wire metadata and the PMT coating. When supplied,
        opaque barriers are proved over its full spectral grid. The single
        wavelength tables then describe geometry-query metadata only; use the
        spectral detector backend to propagate calibrated photons.
    retain_all_geometry:
        Retain both PMT arrays, all wire planes and all box faces. This also
        covers rays reaching the cavity through coincident PMT/wall boundaries,
        including calibrations with a reflecting cavity.
    """

    if source_x_sign not in (-1, 1):
        raise ValueError("source_x_sign must be exactly -1 or +1")
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be finite and positive")

    # Lazy imports keep artifact consumers independent of Chroma/PyCUDA.
    from chroma_lar.geometry.config_loader import load_config_from_file
    from chroma_lar.geometry.build_larcube import build_detector

    config = dict(load_config_from_file(config_name))
    detector_type = config.pop("detector_type", "wire")
    if detector_type != "wire":
        raise ValueError("reflect3wires compiler accepts only the wire detector")
    if not config.get("analytic_wires", False):
        raise ValueError("reflect3wires compiler requires analytic_wires=True")

    output = io.StringIO()
    stream = output if quiet else None
    with _numpy2_chroma_linalg_compat():
        if calibration is not None:
            with redirect_stdout(output) if quiet else nullcontext():
                geometry = calibration.build_detector(config_name, analytic_wires=True)
        elif stream is None:
            geometry = build_detector(**config, flatten=False)
        else:
            with redirect_stdout(stream):
                geometry = build_detector(**config, flatten=False)

    box_solid_ids, channel_solid_ids = _locate_solids(config, geometry)
    tables, material_lookup, surface_lookup = _collect_scene_tables(
        geometry, wavelength_nm
    )
    canonical_solid = geometry.solids[int(channel_solid_ids[0])]
    if any(geometry.solids[int(sid)] is not canonical_solid for sid in channel_solid_ids):
        raise ValueError("PMTs no longer share one canonical Solid")
    canonical = _canonical_pmt(canonical_solid, material_lookup, surface_lookup)
    instances, kept_channels, discarded_channels = _compile_instances(
        geometry, canonical, channel_solid_ids, source_x_sign, retain_all=retain_all_geometry
    )
    boxes = _compile_boxes(
        geometry,
        box_solid_ids,
        material_lookup,
        surface_lookup,
        source_x_sign,
    )
    if retain_all_geometry:
        boxes.reachable_face_mask[:] = True
    wires, kept_wires, discarded_wires = _compile_wires(
        geometry,
        material_lookup,
        surface_lookup,
        source_x_sign,
        retain_all=bool(retain_all_wires or retain_all_geometry),
    )

    cathode_surface = config["cathode_surface"]
    active_surface = config["active_surface"]
    if calibration is not None:
        cathode_surface = calibration.surfaces[cathode_surface.name]
        active_surface = calibration.surfaces[active_surface.name]
        # Pruning a detector half is valid only when every possible wavelength
        # sees an opaque, non-WLS barrier. Linear interpolation preserves this
        # condition between the checked calibration grid points.
        for barrier in (cathode_surface, active_surface):
            total = sum(np.interp(calibration.wavelengths, getattr(barrier, field)[:, 0],
                                  getattr(barrier, field)[:, 1]) for field in _SURFACE_DEFAULT_PROBABILITIES)
            if barrier.model != 0 or np.any(total < 1.0 - 1.e-7) or getattr(barrier, "transmissive", 0):
                raise ValueError("spectral reachability requires opaque default-model barriers over the entire wavelength grid")
    cathode_sum = _surface_probability_sum(cathode_surface, wavelength_nm)
    active_sum = _surface_probability_sum(active_surface, wavelength_nm)
    component_min, component_max = _component_bounds(boxes, source_x_sign)
    reachability = Reachability(
        source_x_sign=source_x_sign,
        source_component_bounds_min=_array(component_min, np.float32),
        source_component_bounds_max=_array(component_max, np.float32),
        cathode_probability_sum=cathode_sum,
        active_probability_sum=active_sum,
        cathode_opaque=_surface_is_opaque(cathode_surface, wavelength_nm),
        active_enclosure_opaque=_surface_is_opaque(active_surface, wavelength_nm),
        kept_global_channel_id=_array(kept_channels, np.int32),
        discarded_global_channel_id=_array(discarded_channels, np.int32),
        kept_wireplane_index=_array(kept_wires, np.int32),
        discarded_wireplane_index=_array(discarded_wires, np.int32),
    )

    scene = CompiledReflect3WiresScene(
        config_name=str(config_name),
        wavelength_nm=float(wavelength_nm),
        total_reference_solids=len(geometry.solids),
        total_reference_channels=len(channel_solid_ids),
        tables=tables,
        pmt=canonical,
        instances=instances,
        boxes=boxes,
        wires=wires,
        reachability=reachability,
    )
    if calibration is None and not retain_all_geometry and str(config_name).endswith(DEFAULT_CONFIG):
        _assert_target_signature(
            tables,
            canonical,
            instances,
            wires,
            len(channel_solid_ids),
            retain_all_wires=bool(retain_all_wires),
        )
    scene.validate()
    return scene
