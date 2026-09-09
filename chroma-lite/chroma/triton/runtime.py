"""Backend-neutral runtime contracts for optical photon propagation.

This module is deliberately CPU-only.  It defines the data exchanged between
the public :class:`chroma.sim.Simulation` orchestration layer and a propagation
backend without importing PyCUDA, Torch, Triton, or a CUDA driver.  In
particular it provides:

* immutable, validated structure-of-arrays photon batches;
* event-aware tile plans which may split a large event without losing its
  event index or stable global photon IDs;
* order-independent result collection and event-order reassembly;
* a byte-budget tile planner instead of a device-specific photon constant;
* an explicit three-stream H2D/compute/D2H dependency plan.

The classes here do not construct public ``chroma.event.Event`` instances.
That small policy adapter belongs in ``chroma.sim`` so legacy mutation and
output choices remain separate from backend execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Protocol, runtime_checkable

import numpy as np


PHOTON_RESULT_FIELDS = frozenset(
    {
        "pos",
        "dir",
        "pol",
        "wavelengths",
        "t",
        "last_hit_triangles",
        "flags",
        "weights",
        "evidx",
    }
)

_PHOTON_FIELD_SPECS = {
    "pos": (np.dtype(np.float32), (3,)),
    "dir": (np.dtype(np.float32), (3,)),
    "pol": (np.dtype(np.float32), (3,)),
    "wavelengths": (np.dtype(np.float32), ()),
    "t": (np.dtype(np.float32), ()),
    "last_hit_triangles": (np.dtype(np.int32), ()),
    "flags": (np.dtype(np.uint32), ()),
    "weights": (np.dtype(np.float32), ()),
    "evidx": (np.dtype(np.uint32), ()),
}


class RuntimeContractError(ValueError):
    """A backend/runtime object violates a declared data contract."""


class IncompleteResultError(RuntimeContractError):
    """Not every planned photon has been returned exactly once."""


class MemoryBudgetError(RuntimeContractError):
    """The available device-memory budget cannot hold one photon tile."""


def _readonly_array(
    value: Any,
    *,
    name: str,
    dtype: Optional[Any] = None,
    ndim: Optional[int] = None,
    trailing_shape: tuple[int, ...] = (),
) -> np.ndarray:
    """Return an owned, C-contiguous, read-only NumPy array."""

    try:
        array = np.array(value, dtype=dtype, order="C", copy=True, subok=False)
    except Exception as exc:
        raise RuntimeContractError(f"{name} cannot be converted to a CPU array") from exc
    if ndim is not None and array.ndim != ndim:
        raise RuntimeContractError(f"{name} must have {ndim} dimensions, got {array.ndim}")
    if trailing_shape and array.shape[-len(trailing_shape) :] != trailing_shape:
        raise RuntimeContractError(
            f"{name} must end in shape {trailing_shape}, got {array.shape}"
        )
    array.flags.writeable = False
    return array


def _readonly_field_mapping(
    fields: Mapping[str, Any], count: int, *, name: str
) -> Mapping[str, np.ndarray]:
    frozen: dict[str, np.ndarray] = {}
    for key, value in fields.items():
        if not isinstance(key, str) or not key:
            raise RuntimeContractError(f"{name} keys must be non-empty strings")
        if key in frozen:
            raise RuntimeContractError(f"duplicate {name} field {key!r}")
        array = _readonly_array(value, name=f"{name}[{key!r}]")
        if array.ndim == 0 or array.shape[0] != count:
            raise RuntimeContractError(
                f"{name}[{key!r}] must have leading length {count}, got {array.shape}"
            )
        frozen[key] = array
    return MappingProxyType(frozen)


def _looks_like_photons(value: Any) -> bool:
    return all(hasattr(value, field_name) for field_name in ("pos", "dir", "pol", "wavelengths"))


@dataclass(frozen=True)
class EventFragment:
    """One contiguous slice of an event stored in one execution tile."""

    tile_index: int
    event_index: int
    event_photon_start: int
    event_photon_stop: int
    tile_photon_start: int
    tile_photon_stop: int
    global_photon_start: int
    global_photon_stop: int

    def __post_init__(self) -> None:
        values = (
            self.tile_index,
            self.event_index,
            self.event_photon_start,
            self.event_photon_stop,
            self.tile_photon_start,
            self.tile_photon_stop,
            self.global_photon_start,
            self.global_photon_stop,
        )
        if any(int(value) != value or value < 0 for value in values):
            raise RuntimeContractError("fragment indices must be non-negative integers")
        lengths = (
            self.event_photon_stop - self.event_photon_start,
            self.tile_photon_stop - self.tile_photon_start,
            self.global_photon_stop - self.global_photon_start,
        )
        if lengths[0] <= 0 or len(set(lengths)) != 1:
            raise RuntimeContractError("fragment ranges must have one equal positive length")

    @property
    def photon_count(self) -> int:
        return self.global_photon_stop - self.global_photon_start


@dataclass(frozen=True)
class PlannedEvent:
    """An input event and its stable range in the plan-wide photon namespace."""

    event_index: int
    event_id: Any
    input_object: Any = field(repr=False, compare=False)
    photons: Any = field(repr=False, compare=False)
    was_event: bool
    photon_count: int
    global_photon_start: int
    global_photon_stop: int

    def __post_init__(self) -> None:
        if self.event_index < 0 or self.event_index > np.iinfo(np.uint32).max:
            raise RuntimeContractError("event_index must fit uint32")
        if self.photon_count < 0:
            raise RuntimeContractError("event photon count cannot be negative")
        if self.global_photon_start < 0:
            raise RuntimeContractError("global photon start cannot be negative")
        if self.global_photon_stop - self.global_photon_start != self.photon_count:
            raise RuntimeContractError("planned event global range has the wrong length")


@dataclass(frozen=True)
class BatchTile:
    """A bounded execution tile containing one or more event fragments."""

    tile_index: int
    fragments: tuple[EventFragment, ...]
    photon_count: int

    def __post_init__(self) -> None:
        fragments = tuple(self.fragments)
        object.__setattr__(self, "fragments", fragments)
        if self.tile_index < 0 or self.photon_count <= 0:
            raise RuntimeContractError("a batch tile must have a non-negative ID and photons")
        expected_offset = 0
        for fragment in fragments:
            if fragment.tile_index != self.tile_index:
                raise RuntimeContractError("fragment belongs to a different tile")
            if fragment.tile_photon_start != expected_offset:
                raise RuntimeContractError("tile fragment offsets must be contiguous")
            expected_offset = fragment.tile_photon_stop
        if expected_offset != self.photon_count:
            raise RuntimeContractError("tile fragments do not cover its photon count")


@dataclass(frozen=True)
class PhotonBatch:
    """Immutable CPU structure-of-arrays input for one backend invocation."""

    pos: np.ndarray
    direction: np.ndarray
    polarization: np.ndarray
    wavelengths: np.ndarray
    times: np.ndarray
    last_hit_triangles: np.ndarray
    flags: np.ndarray
    weights: np.ndarray
    event_indices: np.ndarray
    global_photon_ids: np.ndarray
    channels: np.ndarray
    fragments: tuple[EventFragment, ...] = ()

    def __post_init__(self) -> None:
        pos = _readonly_array(
            self.pos, name="pos", dtype=np.float32, ndim=2, trailing_shape=(3,)
        )
        count = int(pos.shape[0])
        vectors = {
            "direction": self.direction,
            "polarization": self.polarization,
        }
        object.__setattr__(self, "pos", pos)
        for name, value in vectors.items():
            array = _readonly_array(
                value, name=name, dtype=np.float32, ndim=2, trailing_shape=(3,)
            )
            if array.shape != pos.shape:
                raise RuntimeContractError(f"{name} must have shape {pos.shape}")
            object.__setattr__(self, name, array)

        scalar_specs = (
            ("wavelengths", self.wavelengths, np.float32),
            ("times", self.times, np.float32),
            ("last_hit_triangles", self.last_hit_triangles, np.int32),
            ("flags", self.flags, np.uint32),
            ("weights", self.weights, np.float32),
            ("event_indices", self.event_indices, np.uint32),
            ("global_photon_ids", self.global_photon_ids, np.int64),
            ("channels", self.channels, np.uint32),
        )
        for name, value, dtype in scalar_specs:
            array = _readonly_array(value, name=name, dtype=dtype, ndim=1)
            if array.shape != (count,):
                raise RuntimeContractError(f"{name} must have shape ({count},)")
            object.__setattr__(self, name, array)
        if count and (
            np.any(self.global_photon_ids < 0)
            or len(np.unique(self.global_photon_ids)) != count
        ):
            raise RuntimeContractError("global photon IDs must be unique and non-negative")
        fragments = tuple(self.fragments)
        object.__setattr__(self, "fragments", fragments)
        if fragments:
            if sum(fragment.photon_count for fragment in fragments) != count:
                raise RuntimeContractError("fragments do not cover the photon batch")
            if len({fragment.tile_index for fragment in fragments}) != 1:
                raise RuntimeContractError("one photon batch cannot span multiple tiles")
            expected_tile_offset = 0
            expected_ids: list[np.ndarray] = []
            expected_events: list[np.ndarray] = []
            for fragment in fragments:
                if fragment.tile_photon_start != expected_tile_offset:
                    raise RuntimeContractError("batch fragment offsets must be contiguous")
                expected_tile_offset = fragment.tile_photon_stop
                expected_ids.append(
                    np.arange(
                        fragment.global_photon_start,
                        fragment.global_photon_stop,
                        dtype=np.int64,
                    )
                )
                expected_events.append(
                    np.full(
                        fragment.photon_count,
                        fragment.event_index,
                        dtype=np.uint32,
                    )
                )
            if not np.array_equal(self.global_photon_ids, np.concatenate(expected_ids)):
                raise RuntimeContractError("global photon IDs disagree with fragments")
            if not np.array_equal(self.event_indices, np.concatenate(expected_events)):
                raise RuntimeContractError("event indices disagree with fragments")

    @property
    def photon_count(self) -> int:
        return int(self.pos.shape[0])

    @property
    def dir(self) -> np.ndarray:
        """Chroma-compatible alias for :attr:`direction`."""

        return self.direction

    @property
    def pol(self) -> np.ndarray:
        """Chroma-compatible alias for :attr:`polarization`."""

        return self.polarization

    @property
    def t(self) -> np.ndarray:
        """Chroma-compatible alias for :attr:`times`."""

        return self.times

    @property
    def evidx(self) -> np.ndarray:
        """Chroma-compatible alias for stable plan event indices."""

        return self.event_indices


def _source_array(
    source: Any,
    attribute: str,
    count: int,
    *,
    default: Optional[Any] = None,
) -> np.ndarray:
    value = getattr(source, attribute, None)
    if value is None:
        if default is None:
            raise RuntimeContractError(f"photon source is missing {attribute!r}")
        value = np.full(count, default)
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise RuntimeContractError(
            f"photon source field {attribute!r} is not CPU-array compatible"
        ) from exc
    if array.ndim == 0 or array.shape[0] != count:
        raise RuntimeContractError(
            f"photon source field {attribute!r} must have leading length {count}"
        )
    return array


@dataclass(frozen=True)
class EventBatchPlan:
    """Immutable event/tile layout for a finite input sequence.

    Event indices are plan-wide and stable.  Photon IDs are the contiguous
    ``int64`` range ``[0, total_photons)`` and therefore do not change when a
    different tile capacity is chosen.
    """

    events: tuple[PlannedEvent, ...]
    tiles: tuple[BatchTile, ...]
    tile_capacity: int
    total_photons: int

    def __post_init__(self) -> None:
        events = tuple(self.events)
        tiles = tuple(self.tiles)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "tiles", tiles)
        if self.tile_capacity <= 0:
            raise RuntimeContractError("tile_capacity must be positive")
        if self.total_photons < 0:
            raise RuntimeContractError("total_photons cannot be negative")

        expected_global = 0
        for event_index, planned in enumerate(events):
            if planned.event_index != event_index:
                raise RuntimeContractError("planned event indices must be contiguous")
            if planned.global_photon_start != expected_global:
                raise RuntimeContractError("planned event photon ranges must be contiguous")
            expected_global = planned.global_photon_stop
        if expected_global != self.total_photons:
            raise RuntimeContractError("planned events do not cover total_photons")

        fragments: list[EventFragment] = []
        for tile_index, tile in enumerate(tiles):
            if tile.tile_index != tile_index:
                raise RuntimeContractError("tile indices must be contiguous")
            if tile.photon_count > self.tile_capacity:
                raise RuntimeContractError("tile exceeds tile_capacity")
            fragments.extend(tile.fragments)

        expected_global = 0
        event_offsets = [0] * len(events)
        for fragment in fragments:
            if fragment.event_index >= len(events):
                raise RuntimeContractError("fragment refers to an unknown event")
            if fragment.global_photon_start != expected_global:
                raise RuntimeContractError("fragments must preserve global photon order")
            expected_global = fragment.global_photon_stop
            if fragment.event_photon_start != event_offsets[fragment.event_index]:
                raise RuntimeContractError("event fragments must be contiguous")
            event_offsets[fragment.event_index] = fragment.event_photon_stop
        if expected_global != self.total_photons:
            raise RuntimeContractError("fragments do not cover total_photons")
        if any(
            offset != planned.photon_count
            for offset, planned in zip(event_offsets, events)
        ):
            raise RuntimeContractError("fragments do not cover every event")

    @classmethod
    def build(cls, inputs: Any, *, tile_capacity: int) -> "EventBatchPlan":
        """Consume ``Photons``/``Event`` inputs and form bounded tiles.

        Objects are recognized structurally to avoid importing ``chroma.event``.
        An object with ``photons_beg`` is treated as an Event; an object with
        the four core photon arrays is treated as a single Photons input.
        """

        tile_capacity = int(tile_capacity)
        if tile_capacity <= 0:
            raise RuntimeContractError("tile_capacity must be positive")
        if hasattr(inputs, "photons_beg") or _looks_like_photons(inputs):
            input_objects = (inputs,)
        else:
            try:
                input_objects = tuple(iter(inputs))
            except TypeError as exc:
                raise RuntimeContractError("inputs must be Photons, Event, or iterable") from exc

        planned_events: list[PlannedEvent] = []
        global_offset = 0
        for event_index, input_object in enumerate(input_objects):
            was_event = hasattr(input_object, "photons_beg")
            photons = input_object.photons_beg if was_event else input_object
            if photons is None or not _looks_like_photons(photons):
                raise RuntimeContractError(
                    f"input {event_index} does not contain a Chroma-like photon source"
                )
            try:
                photon_count = int(len(photons))
            except Exception as exc:
                raise RuntimeContractError(
                    f"input {event_index} photon source has no valid length"
                ) from exc
            if photon_count < 0:
                raise RuntimeContractError("photon source length cannot be negative")
            event_id = getattr(input_object, "id", event_index)
            planned_events.append(
                PlannedEvent(
                    event_index=event_index,
                    event_id=event_id,
                    input_object=input_object,
                    photons=photons,
                    was_event=was_event,
                    photon_count=photon_count,
                    global_photon_start=global_offset,
                    global_photon_stop=global_offset + photon_count,
                )
            )
            global_offset += photon_count

        tile_fragments: list[list[EventFragment]] = []
        tile_counts: list[int] = []
        tile_index = 0
        tile_offset = 0
        for planned in planned_events:
            event_offset = 0
            while event_offset < planned.photon_count:
                if tile_offset == 0:
                    tile_fragments.append([])
                    tile_counts.append(0)
                take = min(
                    planned.photon_count - event_offset,
                    tile_capacity - tile_offset,
                )
                global_start = planned.global_photon_start + event_offset
                fragment = EventFragment(
                    tile_index=tile_index,
                    event_index=planned.event_index,
                    event_photon_start=event_offset,
                    event_photon_stop=event_offset + take,
                    tile_photon_start=tile_offset,
                    tile_photon_stop=tile_offset + take,
                    global_photon_start=global_start,
                    global_photon_stop=global_start + take,
                )
                tile_fragments[tile_index].append(fragment)
                tile_counts[tile_index] += take
                event_offset += take
                tile_offset += take
                if tile_offset == tile_capacity:
                    tile_index += 1
                    tile_offset = 0

        tiles = tuple(
            BatchTile(index, tuple(fragments), tile_counts[index])
            for index, fragments in enumerate(tile_fragments)
        )
        return cls(
            events=tuple(planned_events),
            tiles=tiles,
            tile_capacity=tile_capacity,
            total_photons=global_offset,
        )

    def materialize_tile(self, tile_index: int) -> PhotonBatch:
        """Copy one tile's event slices into a validated CPU photon batch."""

        if tile_index < 0 or tile_index >= len(self.tiles):
            raise IndexError("tile_index is outside the event batch plan")
        tile = self.tiles[tile_index]
        pieces: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "pos",
                "dir",
                "pol",
                "wavelengths",
                "t",
                "last_hit_triangles",
                "flags",
                "weights",
                "channel",
            )
        }
        event_indices: list[np.ndarray] = []
        global_ids: list[np.ndarray] = []
        for fragment in tile.fragments:
            planned = self.events[fragment.event_index]
            source = planned.photons
            count = planned.photon_count
            window = slice(fragment.event_photon_start, fragment.event_photon_stop)
            pieces["pos"].append(_source_array(source, "pos", count)[window])
            pieces["dir"].append(_source_array(source, "dir", count)[window])
            pieces["pol"].append(_source_array(source, "pol", count)[window])
            pieces["wavelengths"].append(
                _source_array(source, "wavelengths", count)[window]
            )
            pieces["t"].append(_source_array(source, "t", count, default=0.0)[window])
            pieces["last_hit_triangles"].append(
                _source_array(source, "last_hit_triangles", count, default=-1)[window]
            )
            pieces["flags"].append(
                _source_array(source, "flags", count, default=np.uint32(0))[window]
            )
            pieces["weights"].append(
                _source_array(source, "weights", count, default=1.0)[window]
            )
            pieces["channel"].append(
                _source_array(source, "channel", count, default=np.uint32(0))[window]
            )
            event_indices.append(
                np.full(fragment.photon_count, fragment.event_index, dtype=np.uint32)
            )
            global_ids.append(
                np.arange(
                    fragment.global_photon_start,
                    fragment.global_photon_stop,
                    dtype=np.int64,
                )
            )

        concatenate = lambda name: np.concatenate(pieces[name], axis=0)
        return PhotonBatch(
            pos=concatenate("pos"),
            direction=concatenate("dir"),
            polarization=concatenate("pol"),
            wavelengths=concatenate("wavelengths"),
            times=concatenate("t"),
            last_hit_triangles=concatenate("last_hit_triangles"),
            flags=concatenate("flags"),
            weights=concatenate("weights"),
            event_indices=np.concatenate(event_indices),
            global_photon_ids=np.concatenate(global_ids),
            channels=concatenate("channel"),
            fragments=tile.fragments,
        )


@dataclass(frozen=True)
class OutputRequest:
    """Public-output projection requested from a propagation backend."""

    keep_photons_beg: bool = False
    keep_photons_end: bool = False
    keep_hits: bool = True
    keep_flat_hits: bool = True
    run_daq: bool = False
    photon_tracking: bool = False

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, (bool, np.bool_)):
                raise RuntimeContractError(f"{name} must be boolean")

    @property
    def needs_terminal_photon_fields(self) -> bool:
        return bool(
            self.keep_photons_end
            or self.keep_hits
            or self.keep_flat_hits
            or self.photon_tracking
        )

    @property
    def needs_detected_channels(self) -> bool:
        return bool(self.keep_hits or self.keep_flat_hits or self.run_daq)

    @property
    def needs_host_result(self) -> bool:
        """Whether propagation must return data to host orchestration.

        ``keep_photons_beg`` is deliberately absent: ordinary CPU inputs are
        already retained by orchestration.  A device-generated source that
        needs a host copy of its initial state can explicitly request D2H when
        constructing its pipeline plan.
        """

        return bool(
            self.keep_photons_end
            or self.keep_hits
            or self.keep_flat_hits
            or self.run_daq
            or self.photon_tracking
        )

    @property
    def required_result_fields(self) -> frozenset[str]:
        fields: set[str] = set()
        if self.needs_terminal_photon_fields:
            fields.update(PHOTON_RESULT_FIELDS)
        elif self.run_daq:
            fields.update({"t", "last_hit_triangles", "flags", "weights", "evidx"})
        return frozenset(fields)


@dataclass(frozen=True)
class TraceRecords:
    """Compact ragged photon tracking records keyed by global photon ID."""

    global_photon_ids: np.ndarray
    step_indices: np.ndarray
    fields: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        photon_ids = _readonly_array(
            self.global_photon_ids,
            name="trace.global_photon_ids",
            dtype=np.int64,
            ndim=1,
        )
        steps = _readonly_array(
            self.step_indices, name="trace.step_indices", dtype=np.int32, ndim=1
        )
        if steps.shape != photon_ids.shape:
            raise RuntimeContractError("trace step indices must match photon IDs")
        if np.any(photon_ids < 0) or np.any(steps < 0):
            raise RuntimeContractError("trace photon IDs and steps must be non-negative")
        object.__setattr__(self, "global_photon_ids", photon_ids)
        object.__setattr__(self, "step_indices", steps)
        object.__setattr__(
            self,
            "fields",
            _readonly_field_mapping(self.fields, len(photon_ids), name="trace.fields"),
        )

    def take_photon_range(self, start: int, stop: int) -> "TraceRecords":
        mask = (self.global_photon_ids >= start) & (self.global_photon_ids < stop)
        return TraceRecords(
            self.global_photon_ids[mask],
            self.step_indices[mask],
            {name: value[mask] for name, value in self.fields.items()},
        )


@dataclass(frozen=True)
class PropagationResult:
    """Backend result rows keyed by stable global photon ID.

    ``fields`` may be an output projection, but every array must have one row
    per ID.  Detected channels use ``-1`` for non-detections.  Results from
    different tiles may arrive in any order.
    """

    global_photon_ids: np.ndarray
    fields: Mapping[str, np.ndarray]
    detected_channels: Optional[np.ndarray] = None
    trace_records: Optional[TraceRecords] = None

    def __post_init__(self) -> None:
        photon_ids = _readonly_array(
            self.global_photon_ids,
            name="result.global_photon_ids",
            dtype=np.int64,
            ndim=1,
        )
        count = len(photon_ids)
        if count and (
            np.any(photon_ids < 0) or len(np.unique(photon_ids)) != count
        ):
            raise RuntimeContractError("result global photon IDs must be unique and non-negative")
        object.__setattr__(self, "global_photon_ids", photon_ids)
        fields = _readonly_field_mapping(self.fields, count, name="result.fields")
        for name, (dtype, trailing_shape) in _PHOTON_FIELD_SPECS.items():
            if name not in fields:
                continue
            value = fields[name]
            expected_shape = (count,) + trailing_shape
            if value.dtype != dtype or value.shape != expected_shape:
                raise RuntimeContractError(
                    f"result field {name!r} must be {dtype} with shape "
                    f"{expected_shape}, got {value.dtype} {value.shape}"
                )
        object.__setattr__(self, "fields", fields)
        if self.detected_channels is not None:
            channels = _readonly_array(
                self.detected_channels,
                name="result.detected_channels",
                dtype=np.int32,
                ndim=1,
            )
            if channels.shape != (count,):
                raise RuntimeContractError(
                    f"detected_channels must have shape ({count},)"
                )
            object.__setattr__(self, "detected_channels", channels)
        if self.trace_records is not None and len(self.trace_records.global_photon_ids):
            if not np.all(np.isin(self.trace_records.global_photon_ids, photon_ids)):
                raise RuntimeContractError("trace records refer to photons outside this result")

    @property
    def photon_count(self) -> int:
        return len(self.global_photon_ids)

    def take_rows(self, rows: Any) -> "PropagationResult":
        selected_ids = self.global_photon_ids[rows]
        if np.ndim(selected_ids) == 0:
            selected_ids = np.asarray([selected_ids], dtype=np.int64)
            rows = np.asarray([rows])
        if self.trace_records is None:
            trace = None
        elif len(selected_ids) == 0:
            trace = TraceRecords(
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int32),
                {
                    name: np.empty((0,) + value.shape[1:], dtype=value.dtype)
                    for name, value in self.trace_records.fields.items()
                },
            )
        else:
            mask = np.isin(self.trace_records.global_photon_ids, selected_ids)
            trace = TraceRecords(
                self.trace_records.global_photon_ids[mask],
                self.trace_records.step_indices[mask],
                {name: value[mask] for name, value in self.trace_records.fields.items()},
            )
        return PropagationResult(
            selected_ids,
            {name: value[rows] for name, value in self.fields.items()},
            None if self.detected_channels is None else self.detected_channels[rows],
            trace,
        )


@dataclass(frozen=True)
class ReassembledEventResult:
    """One planned event and its propagation rows in original photon order."""

    planned_event: PlannedEvent
    propagation: PropagationResult


class EventResultAssembler:
    """Collect out-of-order tile results and restore event/photon order."""

    def __init__(self, plan: EventBatchPlan, request: OutputRequest):
        self.plan = plan
        self.request = request
        self._results: list[PropagationResult] = []
        self._seen = np.zeros(plan.total_photons, dtype=np.bool_)
        self._field_schema: Optional[tuple[str, ...]] = None
        self._has_channels: Optional[bool] = None
        self._trace_schema: Optional[tuple[str, ...]] = None
        self._finished = False
        self._expected_evidx = np.empty(plan.total_photons, dtype=np.uint32)
        for planned in plan.events:
            self._expected_evidx[
                planned.global_photon_start : planned.global_photon_stop
            ] = np.uint32(planned.event_index)

    def add(self, result: PropagationResult) -> None:
        if self._finished:
            raise RuntimeContractError("cannot add results after finish()")
        ids = result.global_photon_ids
        if len(ids) and (np.any(ids >= self.plan.total_photons) or np.any(ids < 0)):
            raise RuntimeContractError("result contains photon IDs outside the plan")
        if len(ids) and np.any(self._seen[ids]):
            raise RuntimeContractError("a global photon ID was returned more than once")

        schema = tuple(sorted(result.fields))
        if self._field_schema is None:
            self._field_schema = schema
        elif schema != self._field_schema:
            raise RuntimeContractError("all tile results must use one field schema")
        has_channels = result.detected_channels is not None
        if self._has_channels is None:
            self._has_channels = has_channels
        elif has_channels != self._has_channels:
            raise RuntimeContractError("all tile results must agree on detected channels")
        trace_schema = (
            None
            if result.trace_records is None
            else tuple(sorted(result.trace_records.fields))
        )
        if self._trace_schema is None and result.trace_records is not None:
            self._trace_schema = trace_schema
        elif result.trace_records is not None and trace_schema != self._trace_schema:
            raise RuntimeContractError("all trace records must use one field schema")

        evidx = result.fields.get("evidx")
        if evidx is not None and not np.array_equal(
            np.asarray(evidx, dtype=np.uint32), self._expected_evidx[ids]
        ):
            raise RuntimeContractError("result evidx does not match planned event boundaries")
        self._seen[ids] = True
        self._results.append(result)

    def finish(self) -> tuple[ReassembledEventResult, ...]:
        if self._finished:
            raise RuntimeContractError("finish() may be called only once")
        missing = np.flatnonzero(~self._seen)
        if len(missing):
            preview = ", ".join(str(int(value)) for value in missing[:8])
            raise IncompleteResultError(
                f"missing {len(missing)} planned photon results (first: {preview})"
            )
        if self.plan.total_photons == 0 and not self._results:
            schema = tuple(sorted(self.request.required_result_fields))
        else:
            schema = self._field_schema or ()
        missing_fields = self.request.required_result_fields.difference(schema)
        if missing_fields:
            raise IncompleteResultError(
                "backend omitted required result fields: "
                + ", ".join(sorted(missing_fields))
            )
        if (
            self.plan.total_photons
            and self.request.needs_detected_channels
            and not self._has_channels
        ):
            raise IncompleteResultError("backend omitted detected channel results")
        if (
            self.plan.total_photons
            and self.request.photon_tracking
            and self._trace_schema is None
        ):
            raise IncompleteResultError("backend omitted requested photon tracking records")

        if self._results:
            ids = np.concatenate([result.global_photon_ids for result in self._results])
            order = np.argsort(ids, kind="stable")
            ordered_ids = ids[order]
            if not np.array_equal(
                ordered_ids, np.arange(self.plan.total_photons, dtype=np.int64)
            ):
                raise IncompleteResultError("result IDs do not form the planned global range")
            fields = {
                name: np.concatenate([result.fields[name] for result in self._results], axis=0)[order]
                for name in schema
            }
            if self._has_channels:
                detected_channels = np.concatenate(
                    [result.detected_channels for result in self._results], axis=0
                )[order]
            else:
                detected_channels = None
            traces = [
                result.trace_records
                for result in self._results
                if result.trace_records is not None
            ]
            if traces:
                trace_ids = np.concatenate([trace.global_photon_ids for trace in traces])
                trace_steps = np.concatenate([trace.step_indices for trace in traces])
                trace_order = np.lexsort((trace_steps, trace_ids))
                trace = TraceRecords(
                    trace_ids[trace_order],
                    trace_steps[trace_order],
                    {
                        name: np.concatenate(
                            [record.fields[name] for record in traces], axis=0
                        )[trace_order]
                        for name in (self._trace_schema or ())
                    },
                )
            else:
                trace = None
            ordered = PropagationResult(
                ordered_ids, fields, detected_channels, trace
            )
        else:
            empty_fields = {
                name: np.empty((0,) + _PHOTON_FIELD_SPECS[name][1], dtype=_PHOTON_FIELD_SPECS[name][0])
                for name in schema
            }
            ordered = PropagationResult(
                np.empty(0, dtype=np.int64),
                empty_fields,
                np.empty(0, dtype=np.int32)
                if self.request.needs_detected_channels
                else None,
                TraceRecords(
                    np.empty(0, dtype=np.int64),
                    np.empty(0, dtype=np.int32),
                    {},
                )
                if self.request.photon_tracking
                else None,
            )

        assembled: list[ReassembledEventResult] = []
        for planned in self.plan.events:
            rows = slice(planned.global_photon_start, planned.global_photon_stop)
            assembled.append(
                ReassembledEventResult(planned, ordered.take_rows(rows))
            )
        self._finished = True
        return tuple(assembled)


@runtime_checkable
class PropagationBackend(Protocol):
    """Minimal synchronous backend consumed by event orchestration."""

    def propagate(
        self, batch: PhotonBatch, request: OutputRequest
    ) -> PropagationResult:
        ...


@dataclass(frozen=True)
class MemoryFootprint:
    """Conservative memory model for one resident execution tile.

    Hit output is budgeted at the worst case of one hit per photon.  Tracking
    cost should include the caller's chosen maximum retained steps.
    """

    fixed_bytes: int = 0
    input_bytes_per_photon: int = 0
    state_bytes_per_photon: int = 0
    scratch_bytes_per_photon: int = 0
    terminal_output_bytes_per_photon: int = 0
    hit_output_bytes_per_photon: int = 0
    tracking_bytes_per_photon: int = 0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if int(value) != value or value < 0:
                raise RuntimeContractError(f"{name} must be a non-negative integer")

    def bytes_per_photon(self, request: OutputRequest) -> int:
        value = (
            self.input_bytes_per_photon
            + self.state_bytes_per_photon
            + self.scratch_bytes_per_photon
        )
        if request.keep_photons_end:
            value += self.terminal_output_bytes_per_photon
        if request.keep_hits or request.keep_flat_hits:
            value += self.hit_output_bytes_per_photon
        if request.photon_tracking:
            value += self.tracking_bytes_per_photon
        return int(value)


@dataclass(frozen=True)
class MemoryTilePlan:
    """Memory-derived tile capacity and predicted peak allocation."""

    total_photons: int
    tile_capacity: int
    tile_count: int
    free_bytes: int
    usable_bytes: int
    bytes_per_photon: int
    estimated_peak_bytes: int


@runtime_checkable
class TilePlanner(Protocol):
    def plan(
        self,
        total_photons: int,
        *,
        free_bytes: int,
        footprint: MemoryFootprint,
        request: OutputRequest,
    ) -> MemoryTilePlan:
        ...


@dataclass(frozen=True)
class MemoryTilePlanner:
    """Choose tile size from live memory rather than a GPU-specific constant."""

    reserve_bytes: int = 0
    usable_fraction: float = 0.85
    alignment: int = 256
    maximum_tile_photons: Optional[int] = None

    def __post_init__(self) -> None:
        if self.reserve_bytes < 0:
            raise RuntimeContractError("reserve_bytes cannot be negative")
        if not (0.0 < self.usable_fraction <= 1.0):
            raise RuntimeContractError("usable_fraction must lie in (0, 1]")
        if self.alignment <= 0:
            raise RuntimeContractError("alignment must be positive")
        if self.maximum_tile_photons is not None and self.maximum_tile_photons <= 0:
            raise RuntimeContractError("maximum_tile_photons must be positive")

    def plan(
        self,
        total_photons: int,
        *,
        free_bytes: int,
        footprint: MemoryFootprint,
        request: OutputRequest,
    ) -> MemoryTilePlan:
        total_photons = int(total_photons)
        free_bytes = int(free_bytes)
        if total_photons < 0 or free_bytes < 0:
            raise RuntimeContractError("photon count and free bytes cannot be negative")
        bytes_per_photon = footprint.bytes_per_photon(request)
        budget = int(math.floor(free_bytes * self.usable_fraction))
        usable = budget - self.reserve_bytes - footprint.fixed_bytes
        if total_photons == 0:
            return MemoryTilePlan(0, 0, 0, free_bytes, max(0, usable), bytes_per_photon, 0)
        if usable < 0:
            raise MemoryBudgetError(
                "fixed allocations and reserve exceed the usable device-memory budget"
            )
        if bytes_per_photon == 0:
            raw_capacity = total_photons
        else:
            raw_capacity = usable // bytes_per_photon
        if self.maximum_tile_photons is not None:
            raw_capacity = min(raw_capacity, self.maximum_tile_photons)
        raw_capacity = min(raw_capacity, total_photons)
        if raw_capacity <= 0:
            required = (
                self.reserve_bytes + footprint.fixed_bytes + max(1, bytes_per_photon)
            )
            raise MemoryBudgetError(
                f"one photon requires at least {required} budgeted bytes"
            )
        if raw_capacity >= self.alignment:
            capacity = (raw_capacity // self.alignment) * self.alignment
        else:
            capacity = raw_capacity
        tile_count = (total_photons + capacity - 1) // capacity
        estimated_peak = footprint.fixed_bytes + capacity * bytes_per_photon
        return MemoryTilePlan(
            total_photons=total_photons,
            tile_capacity=capacity,
            tile_count=tile_count,
            free_bytes=free_bytes,
            usable_bytes=usable,
            bytes_per_photon=bytes_per_photon,
            estimated_peak_bytes=estimated_peak,
        )


class PipelineStageKind(str, Enum):
    HOST_TO_DEVICE = "h2d"
    COMPUTE = "compute"
    DEVICE_TO_HOST = "d2h"


@dataclass(frozen=True)
class PipelineStage:
    """One asynchronously submitted operation and its event dependencies."""

    stage_id: str
    kind: PipelineStageKind
    tile_index: int
    stream_name: str
    buffer_slot: int
    depends_on: tuple[str, ...]
    photon_count: int
    estimated_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        if not self.stage_id or not self.stream_name:
            raise RuntimeContractError("pipeline stage and stream names cannot be empty")
        if self.tile_index < 0 or self.buffer_slot < 0 or self.photon_count <= 0:
            raise RuntimeContractError("pipeline stage indices/counts must be positive")
        if self.estimated_bytes < 0:
            raise RuntimeContractError("estimated stage bytes cannot be negative")


@dataclass(frozen=True)
class AsyncPipelinePlan:
    """Submission DAG for overlapping optional H2D, compute, and D2H streams.

    Tiles reuse ``inflight_tiles`` buffer slots.  Reuse depends on the prior
    tile's actual terminal stage for that slot: D2H when host output is copied,
    otherwise compute.  This keeps the DAG safe when either transfer stage is
    omitted.  Same-stream dependencies serialize each copy/compute engine;
    cross-stream dependencies are suitable for ordinary CUDA events.
    """

    stages: tuple[PipelineStage, ...]
    inflight_tiles: int
    inputs_on_device: bool = False
    copies_outputs_to_host: bool = True

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        object.__setattr__(self, "stages", stages)
        if self.inflight_tiles <= 0:
            raise RuntimeContractError("inflight_tiles must be positive")
        if not isinstance(self.inputs_on_device, (bool, np.bool_)):
            raise RuntimeContractError("inputs_on_device must be boolean")
        if not isinstance(self.copies_outputs_to_host, (bool, np.bool_)):
            raise RuntimeContractError("copies_outputs_to_host must be boolean")
        seen: set[str] = set()
        previous_tile = -1
        for stage in stages:
            if stage.stage_id in seen:
                raise RuntimeContractError("pipeline stage IDs must be unique")
            if stage.tile_index < previous_tile:
                raise RuntimeContractError("pipeline stages must be grouped in tile order")
            missing = set(stage.depends_on).difference(seen)
            if missing:
                raise RuntimeContractError(
                    f"stage {stage.stage_id!r} depends on unknown/later stages {sorted(missing)}"
                )
            seen.add(stage.stage_id)
            previous_tile = stage.tile_index

    @classmethod
    def build(
        cls,
        event_plan: EventBatchPlan,
        *,
        inflight_tiles: int = 3,
        h2d_bytes_per_photon: int = 0,
        d2h_bytes_per_photon: int = 0,
        inputs_on_device: bool = False,
        request: Optional[OutputRequest] = None,
        copy_outputs_to_host: Optional[bool] = None,
    ) -> "AsyncPipelinePlan":
        """Build a transfer/compute DAG for an event plan.

        The no-option form preserves the original H2D/compute/D2H graph.
        ``inputs_on_device=True`` removes H2D.  When ``request`` is supplied,
        D2H is present exactly when it needs a host result; the explicit
        ``copy_outputs_to_host`` setting overrides that policy for device-side
        consumers or custom host projections.
        """

        inflight_tiles = int(inflight_tiles)
        if inflight_tiles <= 0:
            raise RuntimeContractError("inflight_tiles must be positive")
        if h2d_bytes_per_photon < 0 or d2h_bytes_per_photon < 0:
            raise RuntimeContractError("pipeline transfer costs cannot be negative")
        if not isinstance(inputs_on_device, (bool, np.bool_)):
            raise RuntimeContractError("inputs_on_device must be boolean")
        if request is not None and not isinstance(request, OutputRequest):
            raise RuntimeContractError("request must be an OutputRequest")
        if copy_outputs_to_host is not None and not isinstance(
            copy_outputs_to_host, (bool, np.bool_)
        ):
            raise RuntimeContractError("copy_outputs_to_host must be boolean or None")

        # With no request or override, retain the historical three-stage DAG.
        # Supplying a request makes D2H follow its actual host-output needs.
        copies_outputs = (
            bool(copy_outputs_to_host)
            if copy_outputs_to_host is not None
            else True if request is None else request.needs_host_result
        )
        include_h2d = not bool(inputs_on_device)

        stages: list[PipelineStage] = []
        previous_h2d: Optional[str] = None
        previous_compute: Optional[str] = None
        previous_d2h: Optional[str] = None
        terminal_by_slot: dict[int, str] = {}

        def dependencies(*values: Optional[str]) -> tuple[str, ...]:
            # Dependencies have semantic order and should not be duplicated
            # when inflight_tiles == 1 makes same-stream and slot reuse equal.
            return tuple(dict.fromkeys(value for value in values if value is not None))

        for tile in event_plan.tiles:
            index = tile.tile_index
            slot = index % inflight_tiles
            h2d_id = f"h2d:{index}"
            compute_id = f"compute:{index}"
            d2h_id = f"d2h:{index}"
            slot_dependency = terminal_by_slot.get(slot)
            if include_h2d:
                stages.append(
                    PipelineStage(
                        h2d_id,
                        PipelineStageKind.HOST_TO_DEVICE,
                        index,
                        "h2d",
                        slot,
                        dependencies(previous_h2d, slot_dependency),
                        tile.photon_count,
                        tile.photon_count * int(h2d_bytes_per_photon),
                    )
                )
                compute_dependencies = dependencies(h2d_id, previous_compute)
                previous_h2d = h2d_id
            else:
                compute_dependencies = dependencies(
                    previous_compute, slot_dependency
                )
            stages.append(
                PipelineStage(
                    compute_id,
                    PipelineStageKind.COMPUTE,
                    index,
                    "compute",
                    slot,
                    compute_dependencies,
                    tile.photon_count,
                )
            )
            previous_compute = compute_id
            if copies_outputs:
                stages.append(
                    PipelineStage(
                        d2h_id,
                        PipelineStageKind.DEVICE_TO_HOST,
                        index,
                        "d2h",
                        slot,
                        dependencies(compute_id, previous_d2h),
                        tile.photon_count,
                        tile.photon_count * int(d2h_bytes_per_photon),
                    )
                )
                previous_d2h = d2h_id
                terminal_by_slot[slot] = d2h_id
            else:
                terminal_by_slot[slot] = compute_id
        return cls(
            tuple(stages),
            inflight_tiles,
            inputs_on_device=bool(inputs_on_device),
            copies_outputs_to_host=copies_outputs,
        )

    def stages_for_tile(self, tile_index: int) -> tuple[PipelineStage, ...]:
        return tuple(stage for stage in self.stages if stage.tile_index == tile_index)

    @property
    def terminal_stages(self) -> tuple[PipelineStage, ...]:
        """Last stage for each tile, whose completion makes its result ready."""

        terminal: list[PipelineStage] = []
        previous_tile: Optional[int] = None
        for index, stage in enumerate(self.stages):
            if previous_tile is not None and stage.tile_index != previous_tile:
                terminal.append(self.stages[index - 1])
            previous_tile = stage.tile_index
        if self.stages:
            terminal.append(self.stages[-1])
        return tuple(terminal)


__all__ = [
    "AsyncPipelinePlan",
    "BatchTile",
    "EventBatchPlan",
    "EventFragment",
    "EventResultAssembler",
    "IncompleteResultError",
    "MemoryBudgetError",
    "MemoryFootprint",
    "MemoryTilePlan",
    "MemoryTilePlanner",
    "OutputRequest",
    "PHOTON_RESULT_FIELDS",
    "PhotonBatch",
    "PipelineStage",
    "PipelineStageKind",
    "PlannedEvent",
    "PropagationBackend",
    "PropagationResult",
    "ReassembledEventResult",
    "RuntimeContractError",
    "TilePlanner",
    "TraceRecords",
]
