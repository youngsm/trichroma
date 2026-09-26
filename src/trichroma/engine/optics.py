"""Detector-independent host representation of Chroma optical tables.

The legacy :class:`chroma.gpu.geometry.GPUGeometry` resamples every optical
property onto one uniform wavelength grid before copying it to CUDA.  This
module performs the same ``numpy.interp`` resampling and float32 conversion,
but produces a pointer-free, immutable structure-of-arrays representation for
Triton scene compilers.

No GPU package is imported here.  The resulting arrays are C-contiguous and
read-only, so a device backend can copy them directly without reconstructing
Python object graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import Any, Sequence

import numpy as np

from chroma.geometry import standard_wavelengths


SURFACE_DEFAULT = 0
SURFACE_COMPLEX = 1
SURFACE_WLS = 2
SURFACE_DICHROIC = 3
SURFACE_ANGULAR = 4
_VALID_SURFACE_MODELS = frozenset(
    (SURFACE_DEFAULT, SURFACE_COMPLEX, SURFACE_WLS,
     SURFACE_DICHROIC, SURFACE_ANGULAR)
)


class OpticalFeature(IntFlag):
    """Features present in an :class:`OpticalTableIR`.

    These bits describe scene data, not requested simulation options such as
    weighted transport.  They are intended to select minimal kernel variants.
    """

    NONE = 0
    BULK_REEMISSION = 1 << 0
    DEFAULT_SURFACE = 1 << 1
    COMPLEX_SURFACE = 1 << 2
    WLS_SURFACE = 1 << 3
    DICHROIC_SURFACE = 1 << 4
    ANGULAR_SURFACE = 1 << 5
    SURFACE_REEMISSION = 1 << 6
    TRANSMISSIVE_SURFACE = 1 << 7
    WAVELENGTH_DEPENDENT = 1 << 8
    TIME_REEMISSION = 1 << 9


def _immutable_array(value: Any, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _require_array(
    name: str,
    value: np.ndarray,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype != np.dtype(dtype):
        raise ValueError(f"{name} must have dtype {np.dtype(dtype)}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if value.flags.writeable:
        raise ValueError(f"{name} must be read-only")


@dataclass(frozen=True)
class UniformGrid:
    """Uniform device grid and its explicit indexing metadata."""

    values: np.ndarray
    start: np.float32
    step: np.float32

    @property
    def count(self) -> int:
        return int(self.values.size)

    @property
    def stop(self) -> np.float32:
        return np.float32(self.start + np.float32(self.count - 1) * self.step)

    def validate(self, name: str = "grid") -> None:
        if not isinstance(self.values, np.ndarray) or self.values.ndim != 1:
            raise ValueError(f"{name}.values must be one-dimensional")
        _require_array(
            f"{name}.values",
            self.values,
            dtype=np.float32,
            shape=(self.values.size,),
        )
        if self.count < 2:
            raise ValueError(f"{name} must contain at least two points")
        if not np.isfinite(self.start) or not np.isfinite(self.step):
            raise ValueError(f"{name} start and step must be finite")
        if self.step <= 0.0:
            raise ValueError(f"{name} step must be positive")
        expected = (
            np.float32(self.start)
            + np.arange(self.count, dtype=np.float32) * np.float32(self.step)
        )
        tolerance = np.maximum(
            np.spacing(np.abs(expected)).astype(np.float32) * np.float32(2.0),
            np.float32(1.0e-7),
        )
        if np.any(np.abs(self.values - expected) > tolerance):
            raise ValueError(f"{name}.values are inconsistent with start/step")


@dataclass(frozen=True)
class MaterialTableIR:
    """Structure-of-arrays material data with ragged reemission components."""

    names: tuple[str, ...]
    refractive_index: np.ndarray
    absorption_length: np.ndarray
    scattering_length: np.ndarray
    component_offsets: np.ndarray
    component_reemission_prob: np.ndarray
    component_reemission_wavelength_cdf: np.ndarray
    component_reemission_time_cdf: np.ndarray
    component_absorption_length: np.ndarray

    @property
    def count(self) -> int:
        return len(self.names)

    @property
    def component_count(self) -> int:
        return int(self.component_offsets[-1]) if self.component_offsets.size else 0

    def validate(self, wavelength_count: int, time_count: int) -> None:
        nm = self.count
        nc = self.component_count
        for name in (
            "refractive_index", "absorption_length", "scattering_length"
        ):
            _require_array(
                f"materials.{name}",
                getattr(self, name),
                dtype=np.float32,
                shape=(nm, wavelength_count),
            )
        _require_array(
            "materials.component_offsets",
            self.component_offsets,
            dtype=np.int32,
            shape=(nm + 1,),
        )
        if self.component_offsets[0] != 0:
            raise ValueError("materials.component_offsets must start at zero")
        if np.any(np.diff(self.component_offsets) < 0):
            raise ValueError("materials.component_offsets must be nondecreasing")
        for name in (
            "component_reemission_prob",
            "component_reemission_wavelength_cdf",
            "component_absorption_length",
        ):
            _require_array(
                f"materials.{name}",
                getattr(self, name),
                dtype=np.float32,
                shape=(nc, wavelength_count),
            )
        _require_array(
            "materials.component_reemission_time_cdf",
            self.component_reemission_time_cdf,
            dtype=np.float32,
            shape=(nc, time_count),
        )


@dataclass(frozen=True)
class SurfaceTableIR:
    """All Chroma surface models in pointer-free structure-of-arrays form."""

    names: tuple[str, ...]
    present: np.ndarray
    model: np.ndarray
    transmissive: np.ndarray
    thickness: np.ndarray
    detect: np.ndarray
    absorb: np.ndarray
    reemit: np.ndarray
    reflect_diffuse: np.ndarray
    reflect_specular: np.ndarray
    eta: np.ndarray
    k: np.ndarray
    reemission_cdf: np.ndarray
    dichroic_offsets: np.ndarray
    dichroic_angles: np.ndarray
    dichroic_reflect: np.ndarray
    dichroic_transmit: np.ndarray
    angular_offsets: np.ndarray
    angular_angles: np.ndarray
    angular_transmit: np.ndarray
    angular_reflect_specular: np.ndarray
    angular_reflect_diffuse: np.ndarray

    @property
    def count(self) -> int:
        return len(self.names)

    @property
    def dichroic_angle_count(self) -> int:
        return int(self.dichroic_offsets[-1]) if self.dichroic_offsets.size else 0

    @property
    def angular_angle_count(self) -> int:
        return int(self.angular_offsets[-1]) if self.angular_offsets.size else 0

    def validate(self, wavelength_count: int) -> None:
        ns = self.count
        for name, dtype in (
            ("present", np.bool_),
            ("model", np.int32),
            ("transmissive", np.int32),
            ("thickness", np.float32),
        ):
            _require_array(
                f"surfaces.{name}",
                getattr(self, name),
                dtype=dtype,
                shape=(ns,),
            )
        for name in (
            "detect", "absorb", "reemit", "reflect_diffuse",
            "reflect_specular", "eta", "k", "reemission_cdf",
        ):
            _require_array(
                f"surfaces.{name}",
                getattr(self, name),
                dtype=np.float32,
                shape=(ns, wavelength_count),
            )
        self._validate_ragged_angles(
            "dichroic",
            self.dichroic_offsets,
            self.dichroic_angles,
            (
                ("reflect", self.dichroic_reflect, (self.dichroic_angle_count, wavelength_count)),
                (
                    "transmit",
                    self.dichroic_transmit,
                    (self.dichroic_angle_count, wavelength_count),
                ),
            ),
        )
        self._validate_ragged_angles(
            "angular",
            self.angular_offsets,
            self.angular_angles,
            (
                ("transmit", self.angular_transmit, (self.angular_angle_count,)),
                ("reflect_specular", self.angular_reflect_specular, (self.angular_angle_count,)),
                ("reflect_diffuse", self.angular_reflect_diffuse, (self.angular_angle_count,)),
            ),
        )

    def _validate_ragged_angles(
        self,
        prefix: str,
        offsets: np.ndarray,
        angles: np.ndarray,
        arrays: tuple[tuple[str, np.ndarray, tuple[int, ...]], ...],
    ) -> None:
        _require_array(
            f"surfaces.{prefix}_offsets",
            offsets,
            dtype=np.int32,
            shape=(self.count + 1,),
        )
        if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
            raise ValueError(
                f"surfaces.{prefix}_offsets must start at zero and be nondecreasing"
            )
        total = int(offsets[-1])
        _require_array(
            f"surfaces.{prefix}_angles",
            angles,
            dtype=np.float32,
            shape=(total,),
        )
        for name, value, shape in arrays:
            _require_array(
                f"surfaces.{prefix}_{name}",
                value,
                dtype=np.float32,
                shape=shape,
            )


@dataclass(frozen=True)
class OpticalTableIR:
    """Complete immutable host optical scene shared by Triton backends."""

    wavelength_grid: UniformGrid
    time_grid: UniformGrid
    materials: MaterialTableIR
    surfaces: SurfaceTableIR
    feature_mask: OpticalFeature

    def validate(self) -> None:
        self.wavelength_grid.validate("wavelength_grid")
        self.time_grid.validate("time_grid")
        self.materials.validate(
            self.wavelength_grid.count, self.time_grid.count
        )
        self.surfaces.validate(self.wavelength_grid.count)
        expected = _derive_feature_mask(self.materials, self.surfaces)
        if self.feature_mask != expected:
            raise ValueError(
                f"feature mask {self.feature_mask!r} does not match table data {expected!r}"
            )


def _compile_grid(
    values: Any,
    *,
    name: str,
    known_step: float | None = None,
) -> tuple[UniformGrid, np.ndarray]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size < 2:
        raise ValueError(f"{name} must be a one-dimensional grid with at least two points")
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain only finite values")
    if known_step is None:
        try:
            step = np.unique(np.diff(raw)).item()
        except ValueError as exc:
            raise ValueError(f"{name} must be equally spaced apart") from exc
    else:
        step = known_step
    if step <= 0.0:
        raise ValueError(f"{name} must be strictly increasing")
    grid = UniformGrid(
        values=_immutable_array(raw, np.float32),
        start=np.float32(raw[0]),
        step=np.float32(step),
    )
    grid.validate(name)
    return grid, raw


def _object_name(obj: Any) -> str:
    name = getattr(obj, "name", None)
    return str(name) if name is not None else type(obj).__name__


def _property_table(obj: Any, field: str) -> np.ndarray:
    label = f"{_object_name(obj)}.{field}"
    value = getattr(obj, field, None)
    if value is None:
        raise ValueError(f"{label} must not be None")
    table = np.asarray(value)
    if table.ndim != 2 or table.shape[1] != 2 or table.shape[0] < 1:
        raise ValueError(f"{label} must have shape (N, 2)")
    if not np.all(np.isfinite(table[:, 0])):
        raise ValueError(f"{label} coordinates must be finite")
    if table.shape[0] > 1 and np.any(np.diff(table[:, 0]) <= 0.0):
        raise ValueError(f"{label} coordinates must be strictly increasing")
    if np.any(np.isnan(table[:, 1])):
        raise ValueError(f"{label} values must not contain NaN")
    return table


def _resample(obj: Any, field: str, targets: np.ndarray) -> np.ndarray:
    table = _property_table(obj, field)
    # This is deliberately the same operation and cast as GPUGeometry.
    return np.interp(targets, table[:, 0], table[:, 1]).astype(np.float32)


def _stack_rows(rows: list[np.ndarray], width: int) -> np.ndarray:
    if not rows:
        return _immutable_array(np.empty((0, width)), np.float32)
    return _immutable_array(np.stack(rows, axis=0), np.float32)


def _validate_probability(name: str, values: np.ndarray) -> None:
    if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")


def _validate_length(name: str, values: np.ndarray) -> None:
    if np.any(np.isnan(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} must contain non-negative lengths")


def _validate_cdf(name: str, values: np.ndarray) -> None:
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")
    if np.any(np.diff(values, axis=1) < 0.0):
        raise ValueError(f"{name} must be nondecreasing")
    if not np.all(np.isclose(values[:, 0], 0.0, rtol=0.0, atol=1.0e-6)):
        raise ValueError(f"{name} must start at zero on the compiled grid")
    if not np.all(np.isclose(values[:, -1], 1.0, rtol=0.0, atol=1.0e-6)):
        raise ValueError(f"{name} must end at one on the compiled grid")


def _compile_materials(
    materials: Sequence[Any],
    wavelength_targets: np.ndarray,
    time_targets: np.ndarray,
) -> MaterialTableIR:
    if not materials:
        raise ValueError("at least one material is required")
    if any(material is None for material in materials):
        raise ValueError("materials must not contain None")

    names = tuple(_object_name(material) for material in materials)
    refractive = [_resample(m, "refractive_index", wavelength_targets) for m in materials]
    absorption = [_resample(m, "absorption_length", wavelength_targets) for m in materials]
    scattering = [_resample(m, "scattering_length", wavelength_targets) for m in materials]

    component_offsets = [0]
    component_prob: list[np.ndarray] = []
    component_wavelength_cdf: list[np.ndarray] = []
    component_time_cdf: list[np.ndarray] = []
    component_absorption: list[np.ndarray] = []
    for material in materials:
        collections = (
            tuple(getattr(material, "comp_reemission_prob", ())),
            tuple(getattr(material, "comp_reemission_wvl_cdf", ())),
            tuple(getattr(material, "comp_reemission_time_cdf", ())),
            tuple(getattr(material, "comp_absorption_length", ())),
        )
        counts = tuple(len(values) for values in collections)
        if len(set(counts)) != 1:
            raise ValueError(
                f"{_object_name(material)} reemission component arrays must have equal lengths"
            )
        for index in range(counts[0]):
            component_prob.append(
                _resample_component(
                    material, "comp_reemission_prob", collections[0][index],
                    wavelength_targets, index,
                )
            )
            component_wavelength_cdf.append(
                _resample_component(
                    material, "comp_reemission_wvl_cdf", collections[1][index],
                    wavelength_targets, index,
                )
            )
            component_time_cdf.append(
                _resample_component(
                    material, "comp_reemission_time_cdf", collections[2][index],
                    time_targets, index,
                )
            )
            component_absorption.append(
                _resample_component(
                    material, "comp_absorption_length", collections[3][index],
                    wavelength_targets, index,
                )
            )
        component_offsets.append(component_offsets[-1] + counts[0])

    result = MaterialTableIR(
        names=names,
        refractive_index=_stack_rows(refractive, wavelength_targets.size),
        absorption_length=_stack_rows(absorption, wavelength_targets.size),
        scattering_length=_stack_rows(scattering, wavelength_targets.size),
        component_offsets=_immutable_array(component_offsets, np.int32),
        component_reemission_prob=_stack_rows(component_prob, wavelength_targets.size),
        component_reemission_wavelength_cdf=_stack_rows(
            component_wavelength_cdf, wavelength_targets.size
        ),
        component_reemission_time_cdf=_stack_rows(
            component_time_cdf, time_targets.size
        ),
        component_absorption_length=_stack_rows(
            component_absorption, wavelength_targets.size
        ),
    )
    if np.any(~np.isfinite(result.refractive_index)) or np.any(result.refractive_index <= 0.0):
        raise ValueError("material refractive indices must be finite and positive")
    _validate_length("material absorption lengths", result.absorption_length)
    _validate_length("material scattering lengths", result.scattering_length)
    _validate_probability(
        "material component reemission probabilities",
        result.component_reemission_prob,
    )
    _validate_length(
        "material component absorption lengths",
        result.component_absorption_length,
    )
    reemitting_components = np.any(
        result.component_reemission_prob > 0.0, axis=1
    )
    if np.any(reemitting_components):
        _validate_cdf(
            "material component wavelength CDFs",
            result.component_reemission_wavelength_cdf[reemitting_components],
        )
        _validate_cdf(
            "material component time CDFs",
            result.component_reemission_time_cdf[reemitting_components],
        )
    return result


def _resample_component(
    owner: Any,
    field: str,
    value: Any,
    targets: np.ndarray,
    index: int,
) -> np.ndarray:
    proxy = _ComponentProperty(owner, field, value, index)
    return _resample(proxy, "value", targets)


class _ComponentProperty:
    def __init__(self, owner: Any, field: str, value: Any, index: int):
        self.name = f"{_object_name(owner)}.{field}[{index}]"
        self.value = value


def _compile_surfaces(
    surfaces: Sequence[Any],
    wavelength_targets: np.ndarray,
) -> SurfaceTableIR:
    nw = int(wavelength_targets.size)
    names: list[str] = []
    present: list[bool] = []
    models: list[int] = []
    transmissive: list[int] = []
    thickness: list[float] = []
    common: dict[str, list[np.ndarray]] = {
        field: []
        for field in (
            "detect", "absorb", "reemit", "reflect_diffuse",
            "reflect_specular", "eta", "k", "reemission_cdf",
        )
    }
    dichroic_offsets = [0]
    dichroic_angles: list[np.ndarray] = []
    dichroic_reflect: list[np.ndarray] = []
    dichroic_transmit: list[np.ndarray] = []
    angular_offsets = [0]
    angular_angles: list[np.ndarray] = []
    angular_transmit: list[np.ndarray] = []
    angular_reflect_specular: list[np.ndarray] = []
    angular_reflect_diffuse: list[np.ndarray] = []

    for surface in surfaces:
        if surface is None:
            names.append("<none>")
            present.append(False)
            models.append(SURFACE_DEFAULT)
            transmissive.append(0)
            thickness.append(0.0)
            for rows in common.values():
                rows.append(np.zeros(nw, dtype=np.float32))
            dichroic_offsets.append(dichroic_offsets[-1])
            angular_offsets.append(angular_offsets[-1])
            continue

        names.append(_object_name(surface))
        present.append(True)
        model = int(getattr(surface, "model", SURFACE_DEFAULT))
        if model not in _VALID_SURFACE_MODELS:
            raise ValueError(f"{_object_name(surface)} has unknown surface model {model}")
        models.append(model)
        is_transmissive = int(getattr(surface, "transmissive", 0))
        if is_transmissive not in (0, 1):
            raise ValueError(f"{_object_name(surface)}.transmissive must be 0 or 1")
        transmissive.append(is_transmissive)
        surface_thickness = float(getattr(surface, "thickness", 0.0))
        if not np.isfinite(surface_thickness) or surface_thickness < 0.0:
            raise ValueError(f"{_object_name(surface)}.thickness must be finite and non-negative")
        thickness.append(surface_thickness)
        for field, rows in common.items():
            rows.append(_resample(surface, field, wavelength_targets))

        d_count = _append_dichroic(
            surface,
            wavelength_targets,
            dichroic_angles,
            dichroic_reflect,
            dichroic_transmit,
        )
        dichroic_offsets.append(dichroic_offsets[-1] + d_count)
        a_count = _append_angular(
            surface,
            angular_angles,
            angular_transmit,
            angular_reflect_specular,
            angular_reflect_diffuse,
        )
        angular_offsets.append(angular_offsets[-1] + a_count)

    result = SurfaceTableIR(
        names=tuple(names),
        present=_immutable_array(present, np.bool_),
        model=_immutable_array(models, np.int32),
        transmissive=_immutable_array(transmissive, np.int32),
        thickness=_immutable_array(thickness, np.float32),
        detect=_stack_rows(common["detect"], nw),
        absorb=_stack_rows(common["absorb"], nw),
        reemit=_stack_rows(common["reemit"], nw),
        reflect_diffuse=_stack_rows(common["reflect_diffuse"], nw),
        reflect_specular=_stack_rows(common["reflect_specular"], nw),
        eta=_stack_rows(common["eta"], nw),
        k=_stack_rows(common["k"], nw),
        reemission_cdf=_stack_rows(common["reemission_cdf"], nw),
        dichroic_offsets=_immutable_array(dichroic_offsets, np.int32),
        dichroic_angles=_concat_1d(dichroic_angles),
        dichroic_reflect=_stack_rows(dichroic_reflect, nw),
        dichroic_transmit=_stack_rows(dichroic_transmit, nw),
        angular_offsets=_immutable_array(angular_offsets, np.int32),
        angular_angles=_concat_1d(angular_angles),
        angular_transmit=_concat_1d(angular_transmit),
        angular_reflect_specular=_concat_1d(angular_reflect_specular),
        angular_reflect_diffuse=_concat_1d(angular_reflect_diffuse),
    )
    _validate_surface_physics(result)
    return result


def _concat_1d(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        return _immutable_array(np.empty(0), np.float32)
    return _immutable_array(np.concatenate(rows), np.float32)


def _append_dichroic(
    surface: Any,
    wavelength_targets: np.ndarray,
    angles_out: list[np.ndarray],
    reflect_out: list[np.ndarray],
    transmit_out: list[np.ndarray],
) -> int:
    props = getattr(surface, "dichroic_props", None)
    if props is None:
        if int(surface.model) == SURFACE_DICHROIC:
            raise ValueError(f"{_object_name(surface)} requires dichroic_props")
        return 0
    angles = _validated_angles(
        f"{_object_name(surface)}.dichroic_props.angles", props.angles
    )
    reflect = tuple(props.dichroic_reflect)
    transmit = tuple(props.dichroic_transmit)
    if len(reflect) != angles.size or len(transmit) != angles.size:
        raise ValueError(
            f"{_object_name(surface)} dichroic tables must match its angle count"
        )
    angles_out.append(angles)
    for index in range(angles.size):
        reflect_out.append(
            _resample_component(
                surface, "dichroic_reflect", reflect[index],
                wavelength_targets, index,
            )
        )
        transmit_out.append(
            _resample_component(
                surface, "dichroic_transmit", transmit[index],
                wavelength_targets, index,
            )
        )
    return int(angles.size)


def _append_angular(
    surface: Any,
    angles_out: list[np.ndarray],
    transmit_out: list[np.ndarray],
    reflect_specular_out: list[np.ndarray],
    reflect_diffuse_out: list[np.ndarray],
) -> int:
    props = getattr(surface, "angular_props", None)
    if props is None:
        if int(surface.model) == SURFACE_ANGULAR:
            raise ValueError(f"{_object_name(surface)} requires angular_props")
        return 0
    label = f"{_object_name(surface)}.angular_props"
    angles = _validated_angles(f"{label}.angles", props.angles)
    values = (
        np.asarray(props.transmit),
        np.asarray(props.reflect_specular),
        np.asarray(props.reflect_diffuse),
    )
    if any(value.ndim != 1 or value.size != angles.size for value in values):
        raise ValueError(f"{label} probability arrays must match its angle count")
    angles_out.append(angles)
    transmit_out.append(np.asarray(values[0], dtype=np.float32))
    reflect_specular_out.append(np.asarray(values[1], dtype=np.float32))
    reflect_diffuse_out.append(np.asarray(values[2], dtype=np.float32))
    return int(angles.size)


def _validated_angles(name: str, value: Any) -> np.ndarray:
    angles = np.asarray(value, dtype=np.float32)
    if angles.ndim != 1 or angles.size < 2:
        raise ValueError(f"{name} must contain at least two points")
    if np.any(~np.isfinite(angles)) or np.any(np.diff(angles) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return angles


def _validate_surface_physics(surfaces: SurfaceTableIR) -> None:
    probability_fields = (
        "detect", "absorb", "reemit", "reflect_diffuse", "reflect_specular"
    )
    for field in probability_fields:
        _validate_probability(f"surface {field}", getattr(surfaces, field))
    if np.any(~np.isfinite(surfaces.eta)) or np.any(surfaces.eta < 0.0):
        raise ValueError("surface eta must be finite and non-negative")
    if np.any(~np.isfinite(surfaces.k)) or np.any(surfaces.k < 0.0):
        raise ValueError("surface k must be finite and non-negative")

    tolerance = np.float32(2.0e-6)
    for surface_index in range(surfaces.count):
        if not surfaces.present[surface_index]:
            continue
        model = int(surfaces.model[surface_index])
        if model == SURFACE_DEFAULT:
            total = (
                surfaces.detect[surface_index]
                + surfaces.absorb[surface_index]
                + surfaces.reflect_diffuse[surface_index]
                + surfaces.reflect_specular[surface_index]
            )
            if np.any(total > np.float32(1.0) + tolerance):
                raise ValueError(
                    f"{surfaces.names[surface_index]} default surface probabilities exceed one"
                )
        elif model == SURFACE_COMPLEX:
            if np.any(surfaces.eta[surface_index] <= 0.0):
                raise ValueError(
                    f"{surfaces.names[surface_index]} complex eta must be positive"
                )
        elif model == SURFACE_WLS:
            total = (
                surfaces.absorb[surface_index]
                + surfaces.reflect_diffuse[surface_index]
                + surfaces.reflect_specular[surface_index]
            )
            if np.any(total > np.float32(1.0) + tolerance):
                raise ValueError(
                    f"{surfaces.names[surface_index]} WLS probabilities exceed one"
                )
            if np.any(surfaces.reemit[surface_index] > 0.0):
                _validate_cdf(
                    f"{surfaces.names[surface_index]} surface reemission CDF",
                    surfaces.reemission_cdf[surface_index],
                )
        elif model == SURFACE_DICHROIC:
            begin = int(surfaces.dichroic_offsets[surface_index])
            end = int(surfaces.dichroic_offsets[surface_index + 1])
            _validate_probability(
                f"{surfaces.names[surface_index]} dichroic reflect",
                surfaces.dichroic_reflect[begin:end],
            )
            _validate_probability(
                f"{surfaces.names[surface_index]} dichroic transmit",
                surfaces.dichroic_transmit[begin:end],
            )
            if np.any(
                surfaces.dichroic_reflect[begin:end]
                + surfaces.dichroic_transmit[begin:end]
                > np.float32(1.0) + tolerance
            ):
                raise ValueError(
                    f"{surfaces.names[surface_index]} dichroic probabilities exceed one"
                )
        elif model == SURFACE_ANGULAR:
            begin = int(surfaces.angular_offsets[surface_index])
            end = int(surfaces.angular_offsets[surface_index + 1])
            fields = (
                surfaces.angular_transmit[begin:end],
                surfaces.angular_reflect_specular[begin:end],
                surfaces.angular_reflect_diffuse[begin:end],
            )
            for name, values in zip(
                ("transmit", "reflect_specular", "reflect_diffuse"), fields
            ):
                _validate_probability(
                    f"{surfaces.names[surface_index]} angular {name}", values
                )
            if np.any(sum(fields) > np.float32(1.0) + tolerance):
                raise ValueError(
                    f"{surfaces.names[surface_index]} angular probabilities exceed one"
                )


def _rows_vary(values: np.ndarray) -> bool:
    return values.ndim == 2 and values.shape[1] > 1 and bool(
        np.any(values[:, 1:] != values[:, :-1])
    )


def _derive_feature_mask(
    materials: MaterialTableIR,
    surfaces: SurfaceTableIR,
) -> OpticalFeature:
    result = OpticalFeature.NONE
    if materials.component_count:
        result |= OpticalFeature.BULK_REEMISSION | OpticalFeature.TIME_REEMISSION
    present_models = set(int(value) for value in surfaces.model[surfaces.present])
    model_features = {
        SURFACE_DEFAULT: OpticalFeature.DEFAULT_SURFACE,
        SURFACE_COMPLEX: OpticalFeature.COMPLEX_SURFACE,
        SURFACE_WLS: OpticalFeature.WLS_SURFACE,
        SURFACE_DICHROIC: OpticalFeature.DICHROIC_SURFACE,
        SURFACE_ANGULAR: OpticalFeature.ANGULAR_SURFACE,
    }
    for model in present_models:
        result |= model_features[model]
    wls = surfaces.present & (surfaces.model == SURFACE_WLS)
    if np.any(wls) and np.any(surfaces.reemit[wls] > 0.0):
        result |= OpticalFeature.SURFACE_REEMISSION
    can_transmit = bool(np.any(surfaces.present & (surfaces.transmissive != 0)))
    default = surfaces.present & (surfaces.model == SURFACE_DEFAULT)
    if np.any(default):
        default_total = (
            surfaces.detect[default]
            + surfaces.absorb[default]
            + surfaces.reflect_diffuse[default]
            + surfaces.reflect_specular[default]
        )
        can_transmit |= bool(np.any(default_total < np.float32(1.0)))
    wls = surfaces.present & (surfaces.model == SURFACE_WLS)
    if np.any(wls):
        wls_total = (
            surfaces.absorb[wls]
            + surfaces.reflect_diffuse[wls]
            + surfaces.reflect_specular[wls]
        )
        can_transmit |= bool(np.any(wls_total < np.float32(1.0)))
    can_transmit |= bool(np.any(surfaces.dichroic_transmit > 0.0))
    can_transmit |= bool(np.any(surfaces.angular_transmit > 0.0))
    if can_transmit:
        result |= OpticalFeature.TRANSMISSIVE_SURFACE

    spectral_arrays = (
        materials.refractive_index,
        materials.absorption_length,
        materials.scattering_length,
        materials.component_reemission_prob,
        materials.component_reemission_wavelength_cdf,
        materials.component_absorption_length,
        surfaces.detect,
        surfaces.absorb,
        surfaces.reemit,
        surfaces.reflect_diffuse,
        surfaces.reflect_specular,
        surfaces.eta,
        surfaces.k,
        surfaces.reemission_cdf,
        surfaces.dichroic_reflect,
        surfaces.dichroic_transmit,
    )
    if any(_rows_vary(values) for values in spectral_arrays):
        result |= OpticalFeature.WAVELENGTH_DEPENDENT
    return result


def compile_optical_tables(
    materials: Sequence[Any],
    surfaces: Sequence[Any] = (),
    *,
    wavelengths: Any = None,
    times: Any = None,
) -> OpticalTableIR:
    """Compile Chroma materials and surfaces into an immutable host IR.

    ``materials`` and ``surfaces`` retain their input ordering, so geometry
    indices remain valid.  A ``None`` surface also retains its slot and is
    represented by ``surfaces.present == False``.  Materials may not be
    ``None``.

    Property sampling matches ``GPUGeometry``: source tables are linearly
    interpolated with :func:`numpy.interp`, values outside the source domain
    are clamped to the nearest endpoint, and results are converted to float32.
    """

    if wavelengths is None:
        wavelengths = standard_wavelengths
    default_time_grid = times is None
    if default_time_grid:
        times = np.arange(0.0, 1000.0, 0.05)
    wavelength_grid, wavelength_targets = _compile_grid(
        wavelengths, name="wavelengths"
    )
    time_grid, time_targets = _compile_grid(
        times,
        name="times",
        known_step=0.05 if default_time_grid else None,
    )
    material_ir = _compile_materials(
        tuple(materials), wavelength_targets, time_targets
    )
    surface_ir = _compile_surfaces(tuple(surfaces), wavelength_targets)
    result = OpticalTableIR(
        wavelength_grid=wavelength_grid,
        time_grid=time_grid,
        materials=material_ir,
        surfaces=surface_ir,
        feature_mask=_derive_feature_mask(material_ir, surface_ir),
    )
    result.validate()
    return result


__all__ = (
    "SURFACE_DEFAULT",
    "SURFACE_COMPLEX",
    "SURFACE_WLS",
    "SURFACE_DICHROIC",
    "SURFACE_ANGULAR",
    "OpticalFeature",
    "UniformGrid",
    "MaterialTableIR",
    "SurfaceTableIR",
    "OpticalTableIR",
    "compile_optical_tables",
)
