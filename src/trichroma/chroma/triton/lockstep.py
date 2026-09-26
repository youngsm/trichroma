"""First-divergence diagnostics for CUDA/Triton photon lockstep runs.

The ordinary Chroma queue order is intentionally *not* part of this contract.
Records are joined by ``(logical_step_index, global_photon_id)`` and random
draws by an additional draw-slot index.  This makes a diagnostic invariant to
worker assignment and queue compaction while retaining enough information to
identify the first physical disagreement.

The module is CPU-only.  Device adapters may pass NumPy arrays, Torch tensors,
or PyCUDA ``GPUArray`` objects to :func:`capture_trace`; imports of either GPU
runtime remain lazy.  A normal validation loop launches each implementation
with one Chroma transport-loop iteration per call, captures its arrays, and
uses :func:`run_lockstep` to stop at the first differing bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable

import numpy as np

from .rng_alignment import RandomTape, TapeAudit


class LockstepContractError(ValueError):
    """A debug producer emitted an ambiguous or malformed trace."""


class InteractionStage(IntEnum):
    """Coarse stage owning a decision or random draw.

    Codes are deliberately small integers so CUDA and Triton debug buffers can
    write them directly.  ``SURFACE`` means a material-surface model;
    ``DIELECTRIC`` means the Fresnel boundary branch after a surface PASS (or
    when no explicit surface exists).
    """

    UNSET = 0
    GEOMETRY = 1
    BULK = 2
    RAYLEIGH = 3
    SURFACE = 4
    DIELECTRIC = 5
    TERMINAL = 6


class ProcessDecision(IntEnum):
    """Backend-neutral result of one Chroma transport-loop interaction.

    Values 0--8 are shared with the Triton boundary tape debug path.  Later
    values cover legacy branches which that specialized path does not yet
    implement.
    """

    UNSET = 0
    BULK_ABSORB = 1
    BULK_SCATTER = 2
    SURFACE_ABSORB = 3
    SURFACE_DETECT = 4
    SURFACE_DIFFUSE = 5
    SURFACE_SPECULAR = 6
    DIELECTRIC_REFLECT = 7
    DIELECTRIC_TRANSMIT = 8
    # Values 9 and 10 are also emitted directly by the shared CUDA tape
    # contract.  Diagnostic-only outcomes follow them to keep the device and
    # CPU vocabularies unambiguous.
    BULK_REEMIT = 9
    SURFACE_REEMIT = 10
    NO_HIT = 11
    INVALID = 12


_PROCESS_STAGE = {
    ProcessDecision.UNSET: InteractionStage.UNSET,
    ProcessDecision.BULK_ABSORB: InteractionStage.BULK,
    ProcessDecision.BULK_SCATTER: InteractionStage.RAYLEIGH,
    ProcessDecision.SURFACE_ABSORB: InteractionStage.SURFACE,
    ProcessDecision.SURFACE_DETECT: InteractionStage.SURFACE,
    ProcessDecision.SURFACE_DIFFUSE: InteractionStage.SURFACE,
    ProcessDecision.SURFACE_SPECULAR: InteractionStage.SURFACE,
    ProcessDecision.DIELECTRIC_REFLECT: InteractionStage.DIELECTRIC,
    ProcessDecision.DIELECTRIC_TRANSMIT: InteractionStage.DIELECTRIC,
    ProcessDecision.NO_HIT: InteractionStage.GEOMETRY,
    ProcessDecision.BULK_REEMIT: InteractionStage.BULK,
    ProcessDecision.SURFACE_REEMIT: InteractionStage.SURFACE,
    ProcessDecision.INVALID: InteractionStage.TERMINAL,
}


# Chroma history bits.  Kept local to avoid importing the much larger event
# module in a diagnostic which must remain safe in CPU-only processes.
_NO_HIT = np.uint32(1 << 0)
_BULK_ABSORB = np.uint32(1 << 1)
_SURFACE_DETECT = np.uint32(1 << 2)
_SURFACE_ABSORB = np.uint32(1 << 3)
_RAYLEIGH_SCATTER = np.uint32(1 << 4)
_REFLECT_DIFFUSE = np.uint32(1 << 5)
_REFLECT_SPECULAR = np.uint32(1 << 6)
_SURFACE_REEMIT = np.uint32(1 << 7)
_SURFACE_TRANSMIT = np.uint32(1 << 8)
_BULK_REEMIT = np.uint32(1 << 9)
_NAN_ABORT_16 = np.uint32(1 << 15)
_NAN_ABORT_32 = np.uint32(1 << 31)
_CANONICAL_NAN_WORD = np.uint32(0x7FC00000)


def stage_for_process(process: int) -> InteractionStage:
    """Return the canonical coarse stage for a process decision."""

    try:
        decision = ProcessDecision(int(process))
    except ValueError:
        return InteractionStage.UNSET
    return _PROCESS_STAGE[decision]


def chroma_draw_slot_stage(
    process: int,
    draw_count: int,
    draw_slot: int,
) -> InteractionStage:
    """Resolve the shared Chroma tape slot contract to a physical stage.

    Every ordinary interaction begins with bulk absorption/scattering distance
    slots 0 and 1.  A Rayleigh winner uses slots 2 and 3.  Explicit surfaces
    begin at slot 2.  A final dielectric decision with four total draws had no
    explicit surface; five draws means surface selector slot 2 returned PASS
    and Fresnel consumed slots 3/4.  Longer diffuse/reemission branches remain
    owned by their surface/bulk stage respectively.
    """

    slot = int(draw_slot)
    count = int(draw_count)
    if slot < 0 or slot >= count:
        raise ValueError("draw_slot must identify a consumed draw")
    if slot < 2:
        return InteractionStage.BULK
    try:
        decision = ProcessDecision(int(process))
    except ValueError:
        return InteractionStage.UNSET
    if decision == ProcessDecision.BULK_SCATTER:
        return InteractionStage.RAYLEIGH
    if decision in (ProcessDecision.BULK_ABSORB, ProcessDecision.BULK_REEMIT):
        return InteractionStage.BULK
    if decision in (
        ProcessDecision.SURFACE_ABSORB,
        ProcessDecision.SURFACE_DETECT,
        ProcessDecision.SURFACE_DIFFUSE,
        ProcessDecision.SURFACE_SPECULAR,
        ProcessDecision.SURFACE_REEMIT,
    ):
        return InteractionStage.SURFACE
    if decision in (
        ProcessDecision.DIELECTRIC_REFLECT,
        ProcessDecision.DIELECTRIC_TRANSMIT,
    ):
        if count >= 5 and slot == 2:
            return InteractionStage.SURFACE
        return InteractionStage.DIELECTRIC
    return stage_for_process(decision)


def derive_process_decisions(
    history_before: Any,
    history_after: Any,
    *,
    fallback: int | ProcessDecision = ProcessDecision.UNSET,
) -> np.ndarray:
    """Best-effort process codes from newly set Chroma history bits.

    This adapter is useful for the legacy one-step kernel, which naturally
    exposes history but not a separate branch code.  A history word is
    cumulative, so repeated reflections/scatters are fundamentally ambiguous;
    those rows receive ``fallback``.  A producer that knows the actual branch
    should always emit an explicit process buffer instead.
    """

    before = np.asarray(_host_array(history_before), dtype=np.uint32)
    after = np.asarray(_host_array(history_after), dtype=np.uint32)
    if before.ndim != 1 or after.shape != before.shape:
        raise LockstepContractError("history snapshots must be equal-length 1D arrays")
    delta = np.bitwise_and(after, np.bitwise_not(before))
    result = np.full(before.shape, int(fallback), dtype=np.int32)
    # Terminal/unique outcomes take priority if a defensive kernel sets more
    # than one bit (for example NO_HIT|NAN_ABORT).
    rules = (
        (_NAN_ABORT_16 | _NAN_ABORT_32, ProcessDecision.INVALID),
        (_NO_HIT, ProcessDecision.NO_HIT),
        (_BULK_REEMIT, ProcessDecision.BULK_REEMIT),
        (_BULK_ABSORB, ProcessDecision.BULK_ABSORB),
        (_SURFACE_REEMIT, ProcessDecision.SURFACE_REEMIT),
        (_SURFACE_DETECT, ProcessDecision.SURFACE_DETECT),
        (_SURFACE_ABSORB, ProcessDecision.SURFACE_ABSORB),
        (_REFLECT_DIFFUSE, ProcessDecision.SURFACE_DIFFUSE),
        (_REFLECT_SPECULAR, ProcessDecision.SURFACE_SPECULAR),
        (_SURFACE_TRANSMIT, ProcessDecision.DIELECTRIC_TRANSMIT),
        (_RAYLEIGH_SCATTER, ProcessDecision.BULK_SCATTER),
    )
    unresolved = np.ones(before.shape, dtype=np.bool_)
    for bit, decision in rules:
        selected = unresolved & ((delta & bit) != 0)
        result[selected] = int(decision)
        unresolved[selected] = False
    return result


def _host_array(value: Any) -> np.ndarray:
    """Copy an array-like device or host value to a contiguous NumPy array."""

    def normalize(array: Any) -> np.ndarray:
        result = np.ascontiguousarray(array)
        # PyCUDA's ga.vec.float3 is a structured scalar.  Chroma's own
        # GPUPhotons.get() performs this same lossless view conversion.
        fields = result.dtype.fields
        if fields is not None and tuple(fields) == ("x", "y", "z"):
            component_dtypes = tuple(np.dtype(fields[name][0]) for name in fields)
            if component_dtypes == (np.dtype(np.float32),) * 3 and result.dtype.itemsize == 12:
                result = result.view(np.float32).reshape(result.shape + (3,))
        return np.ascontiguousarray(result)

    if isinstance(value, np.ndarray):
        return normalize(value)
    # Torch CUDA tensors cannot be passed to np.asarray directly.
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        numpy = getattr(value, "numpy", None)
        if callable(numpy):
            return normalize(numpy())
    # PyCUDA GPUArray.get() performs the required synchronization/copy.
    get = getattr(value, "get", None)
    if callable(get):
        return normalize(get())
    return normalize(np.asarray(value))


def _typed_array(
    value: Any,
    *,
    name: str,
    dtype: Any,
    count: Optional[int] = None,
    trailing_shape: tuple[int, ...] = (),
) -> np.ndarray:
    array = _host_array(value)
    expected_dtype = np.dtype(dtype)
    if array.dtype != expected_dtype:
        raise LockstepContractError(
            f"{name} must have dtype {expected_dtype}, got {array.dtype}"
        )
    if count is None:
        if array.ndim != 1 + len(trailing_shape):
            raise LockstepContractError(f"{name} has the wrong rank")
    elif array.shape != (count,) + trailing_shape:
        raise LockstepContractError(
            f"{name} must have shape {(count,) + trailing_shape}, got {array.shape}"
        )
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _readonly_extras(values: Mapping[str, Any], count: int) -> Mapping[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    reserved = {
        "global_photon_ids", "step_indices", "tape_rows", "interaction_indices",
        "interaction_cursor", "draw_cursor", "draw_count", "tape_overflow",
        "stage", "process", "position", "direction", "polarization", "time",
        "history",
    }
    for name, value in values.items():
        if name in reserved:
            raise LockstepContractError(f"extra field {name!r} is reserved")
        array = _host_array(value)
        if array.ndim < 1 or array.shape[0] != count:
            raise LockstepContractError(
                f"extra field {name!r} must have leading dimension {count}"
            )
        if array.dtype.kind not in "biuf":
            raise LockstepContractError(
                f"extra field {name!r} must have a bit-comparable numeric dtype"
            )
        array = np.ascontiguousarray(array)
        array.setflags(write=False)
        result[str(name)] = array
    return MappingProxyType(result)


@dataclass(frozen=True)
class RandomDrawTrace:
    """One record per consumed/attempted tape slot."""

    global_photon_ids: np.ndarray
    step_indices: np.ndarray
    interaction_indices: np.ndarray
    draw_slots: np.ndarray
    stages: np.ndarray
    value_words: np.ndarray

    def __post_init__(self) -> None:
        ids = _typed_array(
            self.global_photon_ids, name="draw.global_photon_ids", dtype=np.int64
        )
        count = len(ids)
        fields = (
            ("step_indices", np.int32),
            ("interaction_indices", np.int32),
            ("draw_slots", np.int32),
            ("stages", np.int32),
            ("value_words", np.uint32),
        )
        object.__setattr__(self, "global_photon_ids", ids)
        for name, dtype in fields:
            object.__setattr__(
                self,
                name,
                _typed_array(getattr(self, name), name=f"draw.{name}", dtype=dtype,
                             count=count),
            )
        if np.any(ids < 0) or np.any(self.step_indices < 0):
            raise LockstepContractError("draw photon IDs and logical steps must be non-negative")
        if np.any(self.interaction_indices < 0) or np.any(self.draw_slots < 0):
            raise LockstepContractError("draw interaction and slot indices must be non-negative")
        keys = list(zip(ids.tolist(), self.step_indices.tolist(), self.draw_slots.tolist()))
        if len(keys) != len(set(keys)):
            raise LockstepContractError(
                "draw trace keys (global ID, logical step, slot) must be unique"
            )


@dataclass(frozen=True)
class InteractionTrace:
    """Post-step state records from one or more logical transport steps.

    ``interaction_indices`` identify the tape segment used by the record
    (pre-commit), while ``interaction_cursor`` and ``draw_cursor`` are the
    post-step audit values.  ``draw_count == -1`` explicitly means that a
    backend did not instrument consumption; exact certification should reject
    such a trace through :attr:`certifies_draw_consumption`.
    """

    global_photon_ids: np.ndarray
    step_indices: np.ndarray
    tape_rows: np.ndarray
    interaction_indices: np.ndarray
    interaction_cursor: np.ndarray
    draw_cursor: np.ndarray
    draw_count: np.ndarray
    tape_overflow: np.ndarray
    stage: np.ndarray
    process: np.ndarray
    position: np.ndarray
    direction: np.ndarray
    polarization: np.ndarray
    time: np.ndarray
    history: np.ndarray
    draws: Optional[RandomDrawTrace] = None
    extras: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = _typed_array(
            self.global_photon_ids, name="global_photon_ids", dtype=np.int64
        )
        count = len(ids)
        object.__setattr__(self, "global_photon_ids", ids)
        scalar_fields = (
            ("step_indices", np.int32),
            ("tape_rows", np.int32),
            ("interaction_indices", np.int32),
            ("interaction_cursor", np.int32),
            ("draw_cursor", np.int32),
            ("draw_count", np.int32),
            ("tape_overflow", np.uint32),
            ("stage", np.int32),
            ("process", np.int32),
            ("time", np.float32),
            ("history", np.uint32),
        )
        for name, dtype in scalar_fields:
            object.__setattr__(
                self,
                name,
                _typed_array(getattr(self, name), name=name, dtype=dtype, count=count),
            )
        for name in ("position", "direction", "polarization"):
            object.__setattr__(
                self,
                name,
                _typed_array(getattr(self, name), name=name, dtype=np.float32,
                             count=count, trailing_shape=(3,)),
            )
        if np.any(ids < 0) or np.any(self.step_indices < 0):
            raise LockstepContractError("global photon IDs and logical steps must be non-negative")
        if np.any(self.tape_rows < 0) or np.any(self.interaction_indices < 0):
            raise LockstepContractError("tape rows and interaction indices must be non-negative")
        if np.any(self.draw_count < -1):
            raise LockstepContractError("draw_count may only use -1 as its unavailable sentinel")
        keys = list(zip(self.step_indices.tolist(), ids.tolist()))
        if len(keys) != len(set(keys)):
            raise LockstepContractError(
                "trace keys (logical step, global photon ID) must be unique"
            )
        if self.draws is not None and not isinstance(self.draws, RandomDrawTrace):
            raise LockstepContractError("draws must be a RandomDrawTrace")
        object.__setattr__(self, "extras", _readonly_extras(self.extras, count))

    @property
    def count(self) -> int:
        return len(self.global_photon_ids)

    @property
    def certifies_draw_consumption(self) -> bool:
        return bool(np.all(self.draw_count >= 0) and self.draws is not None)


def draw_trace_from_counts(
    tape: RandomTape,
    *,
    global_photon_ids: Any,
    step_indices: Any,
    tape_rows: Any,
    interaction_indices: Any,
    draw_counts: Any,
    stages: Any,
    slot_stage_resolver: Optional[Callable[[int, int, int], int]] = None,
) -> RandomDrawTrace:
    """Expand per-interaction counts into auditable per-slot tape records.

    ``slot_stage_resolver(record_index, slot, coarse_stage)`` can distinguish,
    for example, bulk distance slots 0/1 from Rayleigh slots 2/3.  Out-of-range
    attempted slots receive the canonical quiet-NaN word instead of wrapping
    into an adjacent interaction or photon.
    """

    ids = np.asarray(_host_array(global_photon_ids), dtype=np.int64)
    steps = np.asarray(_host_array(step_indices), dtype=np.int32)
    rows = np.asarray(_host_array(tape_rows), dtype=np.int32)
    interactions = np.asarray(_host_array(interaction_indices), dtype=np.int32)
    counts = np.asarray(_host_array(draw_counts), dtype=np.int32)
    coarse_stages = np.asarray(_host_array(stages), dtype=np.int32)
    count = len(ids)
    for name, value in (
        ("step_indices", steps), ("tape_rows", rows),
        ("interaction_indices", interactions), ("draw_counts", counts),
        ("stages", coarse_stages),
    ):
        if value.shape != (count,):
            raise LockstepContractError(f"{name} must match global photon IDs")
    if np.any(counts < 0):
        raise LockstepContractError("cannot expand unavailable/negative draw counts")

    total = int(counts.astype(np.int64).sum())
    out_ids = np.empty(total, dtype=np.int64)
    out_steps = np.empty(total, dtype=np.int32)
    out_interactions = np.empty(total, dtype=np.int32)
    out_slots = np.empty(total, dtype=np.int32)
    out_stages = np.empty(total, dtype=np.int32)
    out_words = np.empty(total, dtype=np.uint32)
    cursor = 0
    for record in range(count):
        row = int(rows[record])
        interaction = int(interactions[record])
        mapping_ok = (
            0 <= row < tape.photon_count
            and int(tape.global_photon_ids[row]) == int(ids[record])
        )
        for slot in range(int(counts[record])):
            out_ids[cursor] = ids[record]
            out_steps[cursor] = steps[record]
            out_interactions[cursor] = interaction
            out_slots[cursor] = slot
            stage = int(coarse_stages[record])
            if slot_stage_resolver is not None:
                stage = int(slot_stage_resolver(record, slot, stage))
            out_stages[cursor] = stage
            if (
                mapping_ok
                and 0 <= interaction < tape.spec.max_interactions
                and slot < tape.spec.draws_per_interaction
            ):
                out_words[cursor] = tape.value_bits[row, interaction, slot]
            else:
                out_words[cursor] = _CANONICAL_NAN_WORD
            cursor += 1
    return RandomDrawTrace(
        out_ids, out_steps, out_interactions, out_slots, out_stages, out_words
    )


def capture_trace(
    *,
    global_photon_ids: Any,
    positions: Any,
    directions: Any,
    polarizations: Any,
    times: Any,
    histories: Any,
    process: Any,
    audit: TapeAudit | Any,
    interaction_indices: Any,
    draw_counts: Any,
    step_index: int | Any,
    state_rows: Optional[Any] = None,
    tape_row_indices: Optional[Any] = None,
    stage: Optional[Any] = None,
    tape: Optional[RandomTape] = None,
    slot_stage_resolver: Optional[Callable[[int, int, int], int]] = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> InteractionTrace:
    """Capture state/audit arrays from a CUDA, Triton, or CPU debug step.

    State arrays are indexed by local photon slot.  ``state_rows`` selects the
    lanes actually processed during this logical step.  ``tape_row_indices``
    maps every local state slot to the photon-stable audit row; it defaults to
    the state row.  Device arrays are copied only here, at the validation
    boundary.
    """

    ids_all = _host_array(global_photon_ids)
    if ids_all.dtype != np.int64 or ids_all.ndim != 1:
        raise LockstepContractError("global_photon_ids must be int64 [state]")
    state_count = len(ids_all)
    if state_rows is None:
        rows = np.arange(state_count, dtype=np.int64)
    else:
        rows = np.asarray(_host_array(state_rows), dtype=np.int64)
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= state_count):
            raise LockstepContractError("state_rows contains an invalid state index")
        if len(np.unique(rows)) != len(rows):
            raise LockstepContractError("state_rows must be unique within a step")

    if tape_row_indices is None:
        tape_rows_all = np.arange(state_count, dtype=np.int32)
    else:
        tape_rows_all = _host_array(tape_row_indices)
        if tape_rows_all.dtype != np.int32 or tape_rows_all.shape != (state_count,):
            raise LockstepContractError("tape_row_indices must be int32 [state]")
    selected_tape_rows = tape_rows_all[rows]

    def select_state(value: Any, name: str) -> np.ndarray:
        array = _host_array(value)
        if array.shape[:1] != (state_count,):
            raise LockstepContractError(f"{name} must be state-indexed")
        return np.ascontiguousarray(array[rows])

    def select_record_or_state(value: Any, name: str) -> np.ndarray:
        array = _host_array(value)
        if array.shape == (state_count,):
            return np.ascontiguousarray(array[rows])
        if array.shape == (len(rows),):
            return np.ascontiguousarray(array)
        raise LockstepContractError(
            f"{name} must be state-indexed or have one value per selected row"
        )

    if np.isscalar(step_index):
        steps = np.full(len(rows), int(step_index), dtype=np.int32)
    else:
        steps = select_record_or_state(step_index, "step_index").astype(
            np.int32, copy=False
        )
    selected_process = select_record_or_state(process, "process")
    if selected_process.dtype != np.int32:
        raise LockstepContractError("process must have dtype int32")
    if stage is None:
        selected_stage = np.asarray(
            [int(stage_for_process(value)) for value in selected_process],
            dtype=np.int32,
        )
    else:
        selected_stage = select_record_or_state(stage, "stage")
        if selected_stage.dtype != np.int32:
            raise LockstepContractError("stage must have dtype int32")

    interaction_values = select_record_or_state(
        interaction_indices, "interaction_indices"
    )
    draw_count_values = select_record_or_state(draw_counts, "draw_counts")
    audit_interaction = _host_array(audit.interaction_cursor)
    audit_draw = _host_array(audit.draw_cursor)
    audit_overflow = _host_array(audit.overflow)
    audit_count = len(audit_interaction)
    if (
        audit_interaction.dtype != np.int32
        or audit_draw.dtype != np.int32
        or audit_overflow.dtype not in (np.dtype(np.int32), np.dtype(np.uint32))
        or audit_draw.shape != (audit_count,)
        or audit_overflow.shape != (audit_count,)
        or np.any(selected_tape_rows < 0)
        or np.any(selected_tape_rows >= audit_count)
    ):
        raise LockstepContractError("audit arrays or selected tape rows are invalid")

    selected_extras = {
        name: select_state(value, f"extra field {name!r}")
        for name, value in (extras or {}).items()
    }
    draws = None
    if tape is not None:
        draws = draw_trace_from_counts(
            tape,
            global_photon_ids=ids_all[rows],
            step_indices=steps,
            tape_rows=selected_tape_rows,
            interaction_indices=interaction_values,
            draw_counts=draw_count_values,
            stages=selected_stage,
            slot_stage_resolver=slot_stage_resolver,
        )

    return InteractionTrace(
        global_photon_ids=np.ascontiguousarray(ids_all[rows]),
        step_indices=np.ascontiguousarray(steps, dtype=np.int32),
        tape_rows=np.ascontiguousarray(selected_tape_rows, dtype=np.int32),
        interaction_indices=np.ascontiguousarray(interaction_values, dtype=np.int32),
        interaction_cursor=np.ascontiguousarray(
            audit_interaction[selected_tape_rows], dtype=np.int32
        ),
        draw_cursor=np.ascontiguousarray(
            audit_draw[selected_tape_rows], dtype=np.int32
        ),
        draw_count=np.ascontiguousarray(draw_count_values, dtype=np.int32),
        tape_overflow=np.ascontiguousarray(
            audit_overflow[selected_tape_rows], dtype=np.uint32
        ),
        stage=np.ascontiguousarray(selected_stage, dtype=np.int32),
        process=np.ascontiguousarray(selected_process, dtype=np.int32),
        position=select_state(positions, "positions"),
        direction=select_state(directions, "directions"),
        polarization=select_state(polarizations, "polarizations"),
        time=select_state(times, "times"),
        history=select_state(histories, "histories").astype(np.uint32, copy=False),
        draws=draws,
        extras=selected_extras,
    )


def _state_aligned_trace_fields(
    *,
    state_count: int,
    tape_row_indices: Any,
    interaction_by_row: Any,
    draw_count_by_row: Any,
    decision_by_row: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map row-indexed specialized trace fields back to local state slots."""

    tape_rows = _host_array(tape_row_indices)
    if tape_rows.dtype != np.int32 or tape_rows.shape != (state_count,):
        raise LockstepContractError("tape_row_indices must be int32 [state]")
    interaction = _host_array(interaction_by_row)
    draw_count = _host_array(draw_count_by_row)
    decision = _host_array(decision_by_row)
    row_count = len(interaction)
    if (
        interaction.dtype != np.int32
        or draw_count.dtype != np.int32
        or decision.dtype != np.int32
        or draw_count.shape != (row_count,)
        or decision.shape != (row_count,)
        or np.any(tape_rows < 0)
        or np.any(tape_rows >= row_count)
    ):
        raise LockstepContractError("row-indexed tape trace buffers are invalid")
    return (
        tape_rows,
        np.ascontiguousarray(interaction[tape_rows]),
        np.ascontiguousarray(draw_count[tape_rows]),
        np.ascontiguousarray(decision[tape_rows]),
    )


def capture_boundary_tape_trace(
    *,
    boundary_trace: Any,
    global_photon_ids: Any,
    positions: Any,
    directions: Any,
    polarizations: Any,
    times: Any,
    histories: Any,
    audit: Any,
    tape_row_indices: Any,
    tape: RandomTape,
    step_index: int,
    state_rows: Optional[Any] = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> InteractionTrace:
    """Convert ``chroma_lar``'s row-indexed ``BoundaryTapeTrace`` payload.

    This function intentionally uses a structural interface rather than
    importing ``chroma_lar`` into trichroma.  The payload's public CUDA
    int32 fields are ``interaction``, ``draw_count`` and ``decision``.
    """

    ids = _host_array(global_photon_ids)
    if ids.dtype != np.int64 or ids.ndim != 1:
        raise LockstepContractError("global_photon_ids must be int64 [state]")
    tape_rows, interactions, counts, decisions = _state_aligned_trace_fields(
        state_count=len(ids),
        tape_row_indices=tape_row_indices,
        interaction_by_row=boundary_trace.interaction,
        draw_count_by_row=boundary_trace.draw_count,
        decision_by_row=boundary_trace.decision,
    )

    # capture_trace invokes this resolver after selecting/reordering records;
    # use its record-local process/count arrays rather than state-slot arrays.
    if state_rows is None:
        selected = np.arange(len(ids), dtype=np.int64)
    else:
        selected = np.asarray(_host_array(state_rows), dtype=np.int64)
    selected_decisions = decisions[selected]
    selected_counts = counts[selected]

    def resolve(record: int, slot: int, coarse_stage: int) -> int:
        del coarse_stage
        return int(
            chroma_draw_slot_stage(
                int(selected_decisions[record]), int(selected_counts[record]), slot
            )
        )

    return capture_trace(
        global_photon_ids=ids,
        positions=positions,
        directions=directions,
        polarizations=polarizations,
        times=times,
        histories=histories,
        process=decisions,
        audit=audit,
        interaction_indices=interactions,
        draw_counts=counts,
        step_index=step_index,
        state_rows=state_rows,
        tape_row_indices=tape_rows,
        tape=tape,
        slot_stage_resolver=resolve,
        extras=extras,
    )


def capture_legacy_tape_trace(
    *,
    gpu_photons: Any,
    legacy_trace: Any,
    global_photon_ids: Any,
    audit: Any,
    tape_row_indices: Any,
    tape: RandomTape,
    step_index: int,
    state_rows: Optional[Any] = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> InteractionTrace:
    """Convert ``PyCudaTapeTrace`` plus ``GPUPhotons`` after a one-step launch.

    Legacy trace fields are photon-state indexed, whereas the audit is tape-row
    indexed.  Device process values 0--10 use the same canonical vocabulary as
    this oracle.  Wavelength, weight, evidx, and last-hit triangle are included
    as exact extra fields by default so the adapter is stricter than the
    minimum trajectory contract.
    """

    ids = _host_array(global_photon_ids)
    if ids.dtype != np.int64 or ids.ndim != 1:
        raise LockstepContractError("global_photon_ids must be int64 [state]")
    count = len(ids)
    interactions = _host_array(legacy_trace.interaction)
    draw_counts = _host_array(legacy_trace.draw_count)
    legacy_decisions = _host_array(legacy_trace.process)
    for name, value in (
        ("legacy_trace.interaction", interactions),
        ("legacy_trace.draw_count", draw_counts),
        ("legacy_trace.process", legacy_decisions),
    ):
        if value.dtype != np.int32 or value.shape != (count,):
            raise LockstepContractError(f"{name} must be int32 [state]")
    decisions = legacy_decisions.copy()

    if state_rows is None:
        selected = np.arange(count, dtype=np.int64)
    else:
        selected = np.asarray(_host_array(state_rows), dtype=np.int64)
    selected_decisions = decisions[selected]
    selected_counts = draw_counts[selected]

    def resolve(record: int, slot: int, coarse_stage: int) -> int:
        del coarse_stage
        return int(
            chroma_draw_slot_stage(
                int(selected_decisions[record]), int(selected_counts[record]), slot
            )
        )

    standard_extras: dict[str, Any] = {
        "wavelength": gpu_photons.wavelengths,
        "weight": gpu_photons.weights,
        "evidx": gpu_photons.evidx,
        "last_hit_triangle": gpu_photons.last_hit_triangles,
    }
    if extras:
        overlap = set(standard_extras) & set(extras)
        if overlap:
            raise LockstepContractError(
                f"legacy extra fields duplicate standard fields: {sorted(overlap)}"
            )
        standard_extras.update(extras)
    return capture_trace(
        global_photon_ids=ids,
        positions=gpu_photons.pos,
        directions=gpu_photons.dir,
        polarizations=gpu_photons.pol,
        times=gpu_photons.t,
        histories=gpu_photons.flags,
        process=decisions,
        audit=audit,
        interaction_indices=interactions,
        draw_counts=draw_counts,
        step_index=step_index,
        state_rows=state_rows,
        tape_row_indices=tape_row_indices,
        tape=tape,
        slot_stage_resolver=resolve,
        extras=standard_extras,
    )


def _python_value(value: np.generic | np.ndarray) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value.item() if isinstance(value, np.generic) else value


def _raw_word(value: np.generic | Any, dtype: np.dtype) -> Optional[int]:
    scalar = np.asarray(value, dtype=dtype).reshape(1)
    if dtype == np.dtype(np.float32):
        return int(scalar.view(np.uint32)[0])
    if dtype == np.dtype(np.float64):
        return int(scalar.view(np.uint64)[0])
    return None


def _raw_word_text(word: Optional[int], dtype: Optional[str]) -> str:
    if word is None:
        return "n/a"
    width = 16 if dtype == "float64" else 8
    return f"0x{word:0{width}x}"


def _stage_text(value: Optional[int]) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{InteractionStage(int(value)).name}({int(value)})"
    except ValueError:
        return f"UNKNOWN({int(value)})"


def _row_state(trace: InteractionTrace, row: int) -> Mapping[str, Any]:
    values: dict[str, Any] = {
        "interaction_index": int(trace.interaction_indices[row]),
        "interaction_cursor": int(trace.interaction_cursor[row]),
        "draw_cursor": int(trace.draw_cursor[row]),
        "draw_count": int(trace.draw_count[row]),
        "tape_overflow": f"0x{int(trace.tape_overflow[row]):08x}",
        "stage": int(trace.stage[row]),
        "process": int(trace.process[row]),
        "position": trace.position[row].tolist(),
        "position_words": [f"0x{int(v):08x}" for v in trace.position[row].view(np.uint32)],
        "direction": trace.direction[row].tolist(),
        "direction_words": [f"0x{int(v):08x}" for v in trace.direction[row].view(np.uint32)],
        "polarization": trace.polarization[row].tolist(),
        "polarization_words": [f"0x{int(v):08x}" for v in trace.polarization[row].view(np.uint32)],
        "time": float(trace.time[row]),
        "time_word": f"0x{int(trace.time[row:row + 1].view(np.uint32)[0]):08x}",
        "history": f"0x{int(trace.history[row]):08x}",
    }
    for name, value in trace.extras.items():
        values[name] = _python_value(value[row])
    return MappingProxyType(values)


@dataclass(frozen=True)
class TraceDifference:
    """The earliest bit-level disagreement and its complete state context."""

    global_photon_id: int
    step_index: int
    interaction_index: Optional[int]
    field: str
    left_label: str
    right_label: str
    left_value: Any
    right_value: Any
    dtype: Optional[str] = None
    left_word: Optional[int] = None
    right_word: Optional[int] = None
    draw_slot: Optional[int] = None
    left_draw_stage: Optional[int] = None
    right_draw_stage: Optional[int] = None
    left_draw_word: Optional[int] = None
    right_draw_word: Optional[int] = None
    left_state: Optional[Mapping[str, Any]] = None
    right_state: Optional[Mapping[str, Any]] = None

    def format(self) -> str:
        interaction = "n/a" if self.interaction_index is None else str(self.interaction_index)
        slot = "n/a" if self.draw_slot is None else str(self.draw_slot)
        stage = (
            f"{_stage_text(self.left_draw_stage)}/"
            f"{_stage_text(self.right_draw_stage)}"
        )
        lines = [
            f"first divergence: photon={self.global_photon_id} "
            f"logical_step={self.step_index} interaction={interaction}",
            f"field={self.field}: {self.left_label}={self.left_value!r} "
            f"{self.right_label}={self.right_value!r}",
            f"raw words: {self.left_label}={_raw_word_text(self.left_word, self.dtype)} "
            f"{self.right_label}={_raw_word_text(self.right_word, self.dtype)}",
            f"draw context: slot={slot} stage({self.left_label}/{self.right_label})={stage} "
            f"word({self.left_label}/{self.right_label})="
            f"{_raw_word_text(self.left_draw_word, 'float32')}/"
            f"{_raw_word_text(self.right_draw_word, 'float32')}",
        ]
        if self.left_state is not None:
            lines.append(f"{self.left_label} state={dict(self.left_state)!r}")
        if self.right_state is not None:
            lines.append(f"{self.right_label} state={dict(self.right_state)!r}")
        return "\n".join(lines)


@dataclass(frozen=True)
class LockstepReport:
    matched: bool
    compared_records: int
    left_label: str
    right_label: str
    difference: Optional[TraceDifference] = None
    draw_consumption_certified: bool = False

    def require_match(self, *, require_draw_certification: bool = True) -> None:
        if not self.matched:
            assert self.difference is not None
            raise AssertionError(self.difference.format())
        if require_draw_certification and not self.draw_consumption_certified:
            raise AssertionError(
                "state matched, but one or both traces did not record exact draw consumption"
            )


def _draw_map(trace: InteractionTrace) -> dict[tuple[int, int, int], int]:
    if trace.draws is None:
        return {}
    return {
        (int(gid), int(step), int(slot)): row
        for row, (gid, step, slot) in enumerate(
            zip(
                trace.draws.global_photon_ids,
                trace.draws.step_indices,
                trace.draws.draw_slots,
            )
        )
    }


def _last_draw_context(
    left: InteractionTrace,
    right: InteractionTrace,
    key: tuple[int, int],
    left_draws: Mapping[tuple[int, int, int], int],
    right_draws: Mapping[tuple[int, int, int], int],
) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    step, gid = key
    slots = sorted(
        {slot for g, s, slot in left_draws if g == gid and s == step}
        | {slot for g, s, slot in right_draws if g == gid and s == step}
    )
    if not slots:
        return None, None, None, None, None
    slot = slots[-1]
    li = left_draws.get((gid, step, slot))
    ri = right_draws.get((gid, step, slot))
    return (
        slot,
        None if li is None else int(left.draws.stages[li]),
        None if ri is None else int(right.draws.stages[ri]),
        None if li is None else int(left.draws.value_words[li]),
        None if ri is None else int(right.draws.value_words[ri]),
    )


def compare_traces(
    left: InteractionTrace,
    right: InteractionTrace,
    *,
    left_label: str = "CUDA",
    right_label: str = "Triton",
) -> LockstepReport:
    """Compare exact state and draw bits, returning only the first divergence."""

    left_rows = {
        (int(step), int(gid)): row
        for row, (step, gid) in enumerate(zip(left.step_indices, left.global_photon_ids))
    }
    right_rows = {
        (int(step), int(gid)): row
        for row, (step, gid) in enumerate(zip(right.step_indices, right.global_photon_ids))
    }
    left_draws = _draw_map(left)
    right_draws = _draw_map(right)
    compared = 0

    def difference(
        key: tuple[int, int],
        field_name: str,
        left_value: Any,
        right_value: Any,
        *,
        dtype: Optional[np.dtype] = None,
        draw_slot: Optional[int] = None,
        left_draw_stage: Optional[int] = None,
        right_draw_stage: Optional[int] = None,
        left_draw_word: Optional[int] = None,
        right_draw_word: Optional[int] = None,
    ) -> LockstepReport:
        step, gid = key
        li = left_rows.get(key)
        ri = right_rows.get(key)
        interaction = None
        if li is not None:
            interaction = int(left.interaction_indices[li])
        elif ri is not None:
            interaction = int(right.interaction_indices[ri])
        if draw_slot is None:
            (
                draw_slot, left_draw_stage, right_draw_stage,
                left_draw_word, right_draw_word,
            ) = _last_draw_context(left, right, key, left_draws, right_draws)
        dtype_obj = None if dtype is None else np.dtype(dtype)
        return LockstepReport(
            matched=False,
            compared_records=compared,
            left_label=left_label,
            right_label=right_label,
            draw_consumption_certified=False,
            difference=TraceDifference(
                global_photon_id=gid,
                step_index=step,
                interaction_index=interaction,
                field=field_name,
                left_label=left_label,
                right_label=right_label,
                left_value=_python_value(left_value),
                right_value=_python_value(right_value),
                dtype=None if dtype_obj is None else dtype_obj.name,
                left_word=None if dtype_obj is None or left_value is None else _raw_word(left_value, dtype_obj),
                right_word=None if dtype_obj is None or right_value is None else _raw_word(right_value, dtype_obj),
                draw_slot=draw_slot,
                left_draw_stage=left_draw_stage,
                right_draw_stage=right_draw_stage,
                left_draw_word=left_draw_word,
                right_draw_word=right_draw_word,
                left_state=None if li is None else _row_state(left, li),
                right_state=None if ri is None else _row_state(right, ri),
            ),
        )

    for key in sorted(set(left_rows) | set(right_rows)):
        li = left_rows.get(key)
        ri = right_rows.get(key)
        if li is None or ri is None:
            return difference(
                key,
                "record_presence",
                li is not None,
                ri is not None,
            )

        # Interaction identity precedes slot comparisons.  A different tape
        # segment means even equal random words were consumed in a different
        # physical interaction.
        if left.interaction_indices[li] != right.interaction_indices[ri]:
            return difference(
                key, "interaction_index", left.interaction_indices[li],
                right.interaction_indices[ri], dtype=np.dtype(np.int32)
            )

        gid = key[1]
        step = key[0]
        slots = sorted(
            {slot for g, s, slot in left_draws if g == gid and s == step}
            | {slot for g, s, slot in right_draws if g == gid and s == step}
        )
        for slot in slots:
            ldi = left_draws.get((gid, step, slot))
            rdi = right_draws.get((gid, step, slot))
            if ldi is None or rdi is None:
                return difference(
                    key, "draw_presence", ldi is not None, rdi is not None,
                    draw_slot=slot,
                    left_draw_stage=None if ldi is None else int(left.draws.stages[ldi]),
                    right_draw_stage=None if rdi is None else int(right.draws.stages[rdi]),
                    left_draw_word=None if ldi is None else int(left.draws.value_words[ldi]),
                    right_draw_word=None if rdi is None else int(right.draws.value_words[rdi]),
                )
            if left.draws.interaction_indices[ldi] != right.draws.interaction_indices[rdi]:
                return difference(
                    key, "draw_interaction_index",
                    left.draws.interaction_indices[ldi], right.draws.interaction_indices[rdi],
                    dtype=np.dtype(np.int32), draw_slot=slot,
                    left_draw_stage=int(left.draws.stages[ldi]),
                    right_draw_stage=int(right.draws.stages[rdi]),
                    left_draw_word=int(left.draws.value_words[ldi]),
                    right_draw_word=int(right.draws.value_words[rdi]),
                )
            if left.draws.stages[ldi] != right.draws.stages[rdi]:
                return difference(
                    key, "draw_stage", left.draws.stages[ldi], right.draws.stages[rdi],
                    dtype=np.dtype(np.int32), draw_slot=slot,
                    left_draw_stage=int(left.draws.stages[ldi]),
                    right_draw_stage=int(right.draws.stages[rdi]),
                    left_draw_word=int(left.draws.value_words[ldi]),
                    right_draw_word=int(right.draws.value_words[rdi]),
                )
            if left.draws.value_words[ldi] != right.draws.value_words[rdi]:
                return difference(
                    key, "draw_value", left.draws.value_words[ldi],
                    right.draws.value_words[rdi], dtype=np.dtype(np.uint32),
                    draw_slot=slot,
                    left_draw_stage=int(left.draws.stages[ldi]),
                    right_draw_stage=int(right.draws.stages[rdi]),
                    left_draw_word=int(left.draws.value_words[ldi]),
                    right_draw_word=int(right.draws.value_words[rdi]),
                )

        scalar_fields = (
            "interaction_cursor", "draw_cursor", "draw_count", "tape_overflow",
            "stage", "process",
        )
        for name in scalar_fields:
            left_array = getattr(left, name)
            right_array = getattr(right, name)
            if left_array.dtype != right_array.dtype:
                return difference(key, f"{name}.dtype", left_array.dtype.name,
                                  right_array.dtype.name)
            left_value = left_array[li]
            right_value = right_array[ri]
            if left_array.dtype.kind == "f":
                equal = _raw_word(left_value, left_array.dtype) == _raw_word(
                    right_value, right_array.dtype
                )
            else:
                equal = bool(left_value == right_value)
            if not equal:
                return difference(key, name, left_value, right_value, dtype=left_array.dtype)

        for name in ("position", "direction", "polarization"):
            left_array = getattr(left, name)
            right_array = getattr(right, name)
            left_words = left_array[li].view(np.uint32)
            right_words = right_array[ri].view(np.uint32)
            unequal = np.flatnonzero(left_words != right_words)
            if len(unequal):
                component = int(unequal[0])
                return difference(
                    key, f"{name}[{component}]", left_array[li, component],
                    right_array[ri, component], dtype=np.dtype(np.float32)
                )

        for name in ("time", "history"):
            left_array = getattr(left, name)
            right_array = getattr(right, name)
            left_value = left_array[li]
            right_value = right_array[ri]
            if left_array.dtype.kind == "f":
                equal = _raw_word(left_value, left_array.dtype) == _raw_word(
                    right_value, right_array.dtype
                )
            else:
                equal = bool(left_value == right_value)
            if not equal:
                return difference(key, name, left_value, right_value, dtype=left_array.dtype)

        extra_names = sorted(set(left.extras) | set(right.extras))
        for name in extra_names:
            if name not in left.extras or name not in right.extras:
                return difference(key, f"extra.{name}.presence", name in left.extras,
                                  name in right.extras)
            la = left.extras[name]
            ra = right.extras[name]
            if la.dtype != ra.dtype or la.shape[1:] != ra.shape[1:]:
                return difference(
                    key, f"extra.{name}.schema",
                    (la.dtype.name, la.shape[1:]), (ra.dtype.name, ra.shape[1:])
                )
            lw = np.ascontiguousarray(la[li]).view(np.uint8)
            rw = np.ascontiguousarray(ra[ri]).view(np.uint8)
            mismatch = np.flatnonzero(lw != rw)
            if len(mismatch):
                flat_index = int(mismatch[0])
                return difference(
                    key, f"extra.{name}.byte[{flat_index}]", int(lw[flat_index]),
                    int(rw[flat_index]), dtype=np.dtype(np.uint8)
                )
        compared += 1

    certified = left.certifies_draw_consumption and right.certifies_draw_consumption
    return LockstepReport(
        matched=True,
        compared_records=compared,
        left_label=left_label,
        right_label=right_label,
        draw_consumption_certified=certified,
    )


@runtime_checkable
class LockstepStepAdapter(Protocol):
    """One backend capable of executing and capturing one logical step."""

    @property
    def name(self) -> str:
        ...

    def advance_one(self, logical_step_index: int) -> InteractionTrace:
        ...


def run_lockstep(
    left: LockstepStepAdapter,
    right: LockstepStepAdapter,
    *,
    max_steps: int,
) -> LockstepReport:
    """Advance both backends one interaction at a time and stop on divergence."""

    if int(max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    compared = 0
    certified = True
    for step in range(int(max_steps)):
        left_trace = left.advance_one(step)
        right_trace = right.advance_one(step)
        report = compare_traces(
            left_trace, right_trace, left_label=left.name, right_label=right.name
        )
        if not report.matched:
            return LockstepReport(
                matched=False,
                compared_records=compared + report.compared_records,
                left_label=left.name,
                right_label=right.name,
                difference=report.difference,
                draw_consumption_certified=False,
            )
        compared += report.compared_records
        certified &= report.draw_consumption_certified
        if left_trace.count == 0 and right_trace.count == 0:
            break
    return LockstepReport(
        matched=True,
        compared_records=compared,
        left_label=left.name,
        right_label=right.name,
        draw_consumption_certified=certified,
    )


__all__ = [
    "InteractionStage",
    "InteractionTrace",
    "LockstepContractError",
    "LockstepReport",
    "LockstepStepAdapter",
    "ProcessDecision",
    "RandomDrawTrace",
    "TraceDifference",
    "capture_boundary_tape_trace",
    "capture_legacy_tape_trace",
    "capture_trace",
    "chroma_draw_slot_stage",
    "compare_traces",
    "derive_process_decisions",
    "draw_trace_from_counts",
    "run_lockstep",
    "stage_for_process",
]
