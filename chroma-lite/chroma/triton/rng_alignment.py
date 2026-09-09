"""Photon-stable random tapes for CUDA/Triton lockstep validation.

This module is intentionally separate from the production generators.  A
seed-matched Chroma run is *not* photon-stable: legacy ``propagate.cu`` indexes
XORWOW state by the current worker slot, and compaction may assign a different
slot to the same photon on the next launch.  A random tape instead fixes every
float32 draw by ``(global_photon_id, interaction_cursor, draw_cursor)`` and
keeps the two cursors in photon-indexed audit buffers.

The tape is a bounded debugging artifact, not an unbounded production RNG.
Exhaustion is observable through overflow bits and yields a canonical quiet
NaN; it never wraps into another interaction or photon's values.

An optional dense interaction certificate records one packed process/draw
word per committed tape row.  Unlike a rolling hash it is collision-free and
retains the exact first divergent interaction while adding only one word for
every many-word tape segment.  A second, independently opt-in state
certificate can retain the 15 intrinsic raw ``Photon`` words after every
committed interaction.  Its occupancy is defined by the process certificate,
so signed integer values whose bits equal the empty sentinel remain valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Optional

import numpy as np


DRAW_OVERFLOW = np.uint32(1 << 0)
INTERACTION_OVERFLOW = np.uint32(1 << 1)
GLOBAL_ID_MISMATCH = np.uint32(1 << 2)
ROW_OUT_OF_RANGE = np.uint32(1 << 3)

PROCESS_UNSET = 0
PROCESS_BULK_ABSORB = 1
PROCESS_BULK_SCATTER = 2
PROCESS_SURFACE_ABSORB = 3
PROCESS_SURFACE_DETECT = 4
PROCESS_SURFACE_DIFFUSE = 5
PROCESS_SURFACE_SPECULAR = 6
PROCESS_DIELECTRIC_REFLECT = 7
PROCESS_DIELECTRIC_TRANSMIT = 8
PROCESS_BULK_REEMIT = 9
PROCESS_SURFACE_REEMIT = 10

# A committed interaction fits in one raw word.  The high nibble is the
# process vocabulary above and the remaining 28 bits are the exact number of
# sequential tape slots consumed.  0xffffffff is unambiguous because process
# code 15 is reserved.  The certificate is a debug-only companion to the much
# larger float32 random tape (one word per interaction rather than per draw).
CERTIFICATE_PROCESS_SHIFT = 28
CERTIFICATE_DRAW_MASK = np.uint32((1 << CERTIFICATE_PROCESS_SHIFT) - 1)
CERTIFICATE_EMPTY_WORD = np.uint32(0xFFFFFFFF)

# Raw post-interaction Photon state.  Keep this order synchronized with
# ``record_committed_state`` in cuda/propagate_tape.cu and with Triton debug
# writers.  Detector-derived fields (channel and boundary kind) deliberately
# do not appear here: they are not members of CUDA's Photon structure and are
# checked separately by detector-level validators.
STATE_CERTIFICATE_FIELDS = (
    "position_x",
    "position_y",
    "position_z",
    "direction_x",
    "direction_y",
    "direction_z",
    "polarization_x",
    "polarization_y",
    "polarization_z",
    "wavelength",
    "time",
    "history",
    "last_triangle",
    "weight",
    "evidx",
)
STATE_CERTIFICATE_FIELD_COUNT = len(STATE_CERTIFICATE_FIELDS)
STATE_CERTIFICATE_FIELD_INDEX = {
    name: index for index, name in enumerate(STATE_CERTIFICATE_FIELDS)
}
STATE_CERTIFICATE_EMPTY_WORD = np.uint32(0xFFFFFFFF)

# The proof kernel certifies unweighted, ordinary Chroma propagation.  Pin the
# header switch explicitly so NVCC and NVRTC cannot silently compile different
# physics when the surrounding environment changes.
LEGACY_TAPE_FORCE_SCATTER_AT_PASS = 0

STAGE_MAPPING = 0
STAGE_BULK = 1
STAGE_SURFACE = 2
STAGE_DIELECTRIC = 3

_U64_MAX = (1 << 64) - 1
_I64_MAX = (1 << 63) - 1
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_PHOTON_SALT = np.uint64(0xD2B74407B1CE6E93)
_INTERACTION_SALT = np.uint64(0xCA5A826395121157)
_DRAW_SALT = np.uint64(0x9E6C63D0676A9A99)
_UNIFORM_SCALE = np.float32(2.0 ** -23)
_CANONICAL_NAN = np.asarray([0x7FC00000], dtype=np.uint32).view(np.float32)[0]


def pack_interaction_certificate(process: Any, draw_count: Any) -> np.ndarray:
    """Pack process codes and exact draw counts into canonical uint32 words."""

    raw_process = np.asarray(process)
    raw_draw_count = np.asarray(draw_count)
    if raw_process.dtype.kind not in "iu" or raw_draw_count.dtype.kind not in "iu":
        raise TypeError("certificate process and draw_count must be integers")
    raw_process, raw_draw_count = np.broadcast_arrays(
        raw_process, raw_draw_count
    )
    if np.any(raw_process < 0) or np.any(raw_process > PROCESS_SURFACE_REEMIT):
        raise ValueError("certificate process must be in the shared range 0..10")
    if np.any(raw_draw_count < 0) or np.any(
        raw_draw_count > int(CERTIFICATE_DRAW_MASK)
    ):
        raise ValueError("certificate draw_count must fit 28 bits")
    return np.ascontiguousarray(
        (raw_process.astype(np.uint32) << np.uint32(CERTIFICATE_PROCESS_SHIFT))
        | raw_draw_count.astype(np.uint32)
    )


def unpack_interaction_certificate(
    words: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(committed, process, draw_count)`` arrays for packed words.

    Uncommitted cells decode to ``-1`` for both payload fields.  Signed int32
    arrays are accepted because Torch lacks uniformly supported uint32 tensor
    operations; their bit patterns are reinterpreted rather than converted.
    """

    raw = np.asarray(words)
    if raw.dtype == np.dtype(np.int32):
        raw = raw.view(np.uint32)
    elif raw.dtype != np.dtype(np.uint32):
        raise TypeError("certificate words must have dtype uint32 or int32")
    committed = raw != CERTIFICATE_EMPTY_WORD
    process = (raw >> np.uint32(CERTIFICATE_PROCESS_SHIFT)).astype(np.int32)
    draw_count = (raw & CERTIFICATE_DRAW_MASK).astype(np.int32)
    process = np.where(committed, process, -1).astype(np.int32, copy=False)
    draw_count = np.where(committed, draw_count, -1).astype(
        np.int32, copy=False
    )
    return committed, process, draw_count


def _splitmix64(value: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 finalizer with defined modulo-2**64 arithmetic."""

    with np.errstate(over="ignore"):
        value = value + _GOLDEN
        value = (value ^ (value >> np.uint64(30))) * _MIX1
        value = (value ^ (value >> np.uint64(27))) * _MIX2
        return value ^ (value >> np.uint64(31))


def _global_ids(values: Iterable[int]) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("global_photon_ids must be one-dimensional")
    if raw.dtype.kind not in "iu":
        raise TypeError("global_photon_ids must contain integers")
    if raw.dtype.kind == "i" and np.any(raw < 0):
        raise ValueError("global_photon_ids must be non-negative")
    unsigned = raw.astype(np.uint64, copy=False)
    if np.any(unsigned > np.uint64(_I64_MAX)):
        raise ValueError("global_photon_ids must fit signed int64 for Triton")
    result = np.ascontiguousarray(unsigned.astype(np.int64))
    if np.unique(result).size != result.size:
        raise ValueError("global_photon_ids must be unique within a tape")
    return result


@dataclass(frozen=True)
class RandomTapeSpec:
    """Dimensions and seed for a bounded alignment tape."""

    max_interactions: int
    draws_per_interaction: int
    seed: int = 1

    def __post_init__(self) -> None:
        if int(self.max_interactions) <= 0:
            raise ValueError("max_interactions must be positive")
        if int(self.draws_per_interaction) <= 0:
            raise ValueError("draws_per_interaction must be positive")
        if int(self.seed) < 0 or int(self.seed) > _U64_MAX:
            raise ValueError("seed must fit uint64")

    def estimated_bytes(self, photon_count: int) -> int:
        photon_count = int(photon_count)
        if photon_count < 0:
            raise ValueError("photon_count cannot be negative")
        return (
            4
            * photon_count
            * int(self.max_interactions)
            * int(self.draws_per_interaction)
            + 8 * photon_count
        )


@dataclass(frozen=True)
class RandomTape:
    """Host random tape with row-major ``[photon, interaction, draw]`` values."""

    spec: RandomTapeSpec
    global_photon_ids: np.ndarray
    values: np.ndarray

    @classmethod
    def generate(
        cls,
        global_photon_ids: Iterable[int],
        spec: RandomTapeSpec,
        *,
        max_bytes: Optional[int] = 2 << 30,
    ) -> "RandomTape":
        ids = _global_ids(global_photon_ids)
        required = spec.estimated_bytes(len(ids))
        if max_bytes is not None and required > int(max_bytes):
            raise MemoryError(
                f"random tape needs {required} bytes, exceeding max_bytes={max_bytes}"
            )

        interaction_count = int(spec.max_interactions)
        draw_count = int(spec.draws_per_interaction)
        values = np.empty(
            (len(ids), interaction_count, draw_count), dtype=np.float32
        )
        photon_key = _splitmix64(ids.astype(np.uint64) ^ _PHOTON_SALT)
        draw_key = _splitmix64(
            np.arange(draw_count, dtype=np.uint64) ^ _DRAW_SALT
        )
        seed = np.uint64(int(spec.seed))
        for interaction in range(interaction_count):
            interaction_key = _splitmix64(
                np.asarray(
                    [np.uint64(interaction) ^ _INTERACTION_SALT],
                    dtype=np.uint64,
                )
            )[0]
            mixed = _splitmix64(
                photon_key[:, None] ^ interaction_key ^ draw_key[None, :] ^ seed
            )
            word = (mixed >> np.uint64(32)).astype(np.uint32)
            # Exactly representable k*2^-23 values in (0, 1], matching the
            # endpoint domain of curand_uniform without copying XORWOW's stream.
            numerator = (word >> np.uint32(9)).astype(np.uint32) + np.uint32(1)
            values[:, interaction, :] = (
                numerator.astype(np.float32) * _UNIFORM_SCALE
            )

        ids.setflags(write=False)
        values.setflags(write=False)
        return cls(spec=spec, global_photon_ids=ids, values=values)

    @property
    def photon_count(self) -> int:
        return int(self.global_photon_ids.size)

    @property
    def value_bits(self) -> np.ndarray:
        return self.values.view(np.uint32)

    def rows_for(self, global_photon_ids: Iterable[int]) -> np.ndarray:
        requested = _global_ids(global_photon_ids)
        lookup = {
            int(global_id): row
            for row, global_id in enumerate(self.global_photon_ids)
        }
        try:
            return np.asarray(
                [lookup[int(global_id)] for global_id in requested],
                dtype=np.int32,
            )
        except KeyError as error:
            raise KeyError(f"global photon ID {error.args[0]} is not in the tape")


@dataclass
class TapeAudit:
    """Photon-indexed interaction/draw cursors and sticky overflow flags."""

    interaction_cursor: np.ndarray
    draw_cursor: np.ndarray
    overflow: np.ndarray

    @classmethod
    def zeros(cls, photon_count: int) -> "TapeAudit":
        photon_count = int(photon_count)
        if photon_count < 0:
            raise ValueError("photon_count cannot be negative")
        return cls(
            interaction_cursor=np.zeros(photon_count, dtype=np.int32),
            draw_cursor=np.zeros(photon_count, dtype=np.int32),
            overflow=np.zeros(photon_count, dtype=np.uint32),
        )

    def copy(self) -> "TapeAudit":
        return TapeAudit(
            self.interaction_cursor.copy(),
            self.draw_cursor.copy(),
            self.overflow.copy(),
        )


@dataclass
class InteractionCertificate:
    """Dense exact process/draw ledger indexed by tape row and interaction."""

    words: np.ndarray

    def __post_init__(self) -> None:
        words = np.asarray(self.words)
        if words.dtype != np.dtype(np.uint32) or words.ndim != 2:
            raise TypeError("certificate words must be uint32 [row, interaction]")
        if not words.flags.c_contiguous:
            raise ValueError("certificate words must be C-contiguous")
        self.words = words

    @classmethod
    def empty(
        cls, photon_count: int, max_interactions: int
    ) -> "InteractionCertificate":
        photon_count = int(photon_count)
        max_interactions = int(max_interactions)
        if photon_count < 0:
            raise ValueError("photon_count cannot be negative")
        if max_interactions <= 0:
            raise ValueError("max_interactions must be positive")
        return cls(np.full(
            (photon_count, max_interactions),
            CERTIFICATE_EMPTY_WORD,
            dtype=np.uint32,
        ))

    def unpack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return unpack_interaction_certificate(self.words)


@dataclass
class StateCertificate:
    """Dense raw Photon state indexed by tape row and interaction.

    The final dimension follows :data:`STATE_CERTIFICATE_FIELDS`.  Words are
    stored, never numerically converted: float32 values use their IEEE-754
    bits, ``history`` and ``last_triangle`` use their zero-extended and
    two's-complement uint32 representations respectively, and ``evidx`` is
    copied as its native uint32 word.

    Occupancy must be interpreted using the paired
    :class:`InteractionCertificate`.  In particular, ``0xffffffff`` is both
    the initialization sentinel and a valid raw value for ``last_triangle``
    equal to ``-1``.
    """

    words: np.ndarray

    def __post_init__(self) -> None:
        words = np.asarray(self.words)
        if (
            words.dtype != np.dtype(np.uint32)
            or words.ndim != 3
            or words.shape[2] != STATE_CERTIFICATE_FIELD_COUNT
        ):
            raise TypeError(
                "state certificate words must be uint32 "
                "[row, interaction, field]"
            )
        if not words.flags.c_contiguous:
            raise ValueError("state certificate words must be C-contiguous")
        self.words = words

    @classmethod
    def empty(
        cls, photon_count: int, max_interactions: int
    ) -> "StateCertificate":
        photon_count = int(photon_count)
        max_interactions = int(max_interactions)
        if photon_count < 0:
            raise ValueError("photon_count cannot be negative")
        if max_interactions <= 0:
            raise ValueError("max_interactions must be positive")
        return cls(np.full(
            (
                photon_count,
                max_interactions,
                STATE_CERTIFICATE_FIELD_COUNT,
            ),
            STATE_CERTIFICATE_EMPTY_WORD,
            dtype=np.uint32,
        ))

    def field(self, name: str) -> np.ndarray:
        """Return the raw uint32 plane for one named Photon field."""

        try:
            index = STATE_CERTIFICATE_FIELD_INDEX[name]
        except KeyError as error:
            raise KeyError(f"unknown state certificate field {name!r}") from error
        return self.words[:, :, index]


def validate_interaction_certificate(
    certificate: InteractionCertificate | np.ndarray,
    interaction_cursor: Any,
) -> None:
    """Validate that every cursor describes one hole-free committed prefix."""

    words = (
        certificate.words
        if isinstance(certificate, InteractionCertificate)
        else np.asarray(certificate)
    )
    if words.dtype == np.dtype(np.int32):
        words = words.view(np.uint32)
    if words.dtype != np.dtype(np.uint32) or words.ndim != 2:
        raise TypeError("certificate words must be uint32 or int32 [row, interaction]")
    cursor = np.asarray(interaction_cursor)
    if cursor.dtype.kind not in "iu" or cursor.shape != (words.shape[0],):
        raise TypeError(
            "interaction_cursor must be an integer array with one value per row"
        )
    if np.any(cursor < 0) or np.any(cursor > words.shape[1]):
        raise ValueError("interaction_cursor is outside the certificate width")
    expected = np.arange(words.shape[1])[None, :] < cursor[:, None]
    committed, process, _ = unpack_interaction_certificate(words)
    if not np.array_equal(committed, expected):
        row, interaction = np.argwhere(committed != expected)[0]
        raise ValueError(
            "certificate commit prefix disagrees with interaction_cursor at "
            f"row {int(row)}, interaction {int(interaction)}"
        )
    invalid_process = committed & (
        (process <= PROCESS_UNSET) | (process > PROCESS_SURFACE_REEMIT)
    )
    if np.any(invalid_process):
        row, interaction = np.argwhere(invalid_process)[0]
        raise ValueError(
            "certificate contains an invalid committed process at "
            f"row {int(row)}, interaction {int(interaction)}"
        )


def validate_state_certificate(
    state_certificate: StateCertificate | np.ndarray,
    interaction_certificate: InteractionCertificate | np.ndarray,
    interaction_cursor: Any,
) -> None:
    """Validate state occupancy against a hole-free process ledger.

    A state record is committed when at least one of its 15 raw words differs
    from the initialization sentinel.  Individual sentinel-valued fields are
    allowed.  The paired process certificate is authoritative: all and only
    its committed cells must contain a state record.
    """

    state_words = (
        state_certificate.words
        if isinstance(state_certificate, StateCertificate)
        else np.asarray(state_certificate)
    )
    if state_words.dtype == np.dtype(np.int32):
        state_words = state_words.view(np.uint32)
    if (
        state_words.dtype != np.dtype(np.uint32)
        or state_words.ndim != 3
        or state_words.shape[2] != STATE_CERTIFICATE_FIELD_COUNT
    ):
        raise TypeError(
            "state certificate words must be uint32 or int32 "
            "[row, interaction, field]"
        )

    process_words = (
        interaction_certificate.words
        if isinstance(interaction_certificate, InteractionCertificate)
        else np.asarray(interaction_certificate)
    )
    if process_words.dtype == np.dtype(np.int32):
        process_words = process_words.view(np.uint32)
    if process_words.shape != state_words.shape[:2]:
        raise ValueError(
            "state and interaction certificates must have matching "
            "[row, interaction] dimensions"
        )
    validate_interaction_certificate(process_words, interaction_cursor)
    committed, _, _ = unpack_interaction_certificate(process_words)
    state_committed = np.any(
        state_words != STATE_CERTIFICATE_EMPTY_WORD, axis=2
    )
    if not np.array_equal(state_committed, committed):
        row, interaction = np.argwhere(state_committed != committed)[0]
        expected = "committed" if committed[row, interaction] else "empty"
        observed = "committed" if state_committed[row, interaction] else "empty"
        raise ValueError(
            "state certificate occupancy disagrees with interaction "
            f"certificate at row {int(row)}, interaction "
            f"{int(interaction)}: expected {expected}, observed {observed}"
        )


@dataclass(frozen=True)
class TapeProbeResult:
    values: Any
    audit: Any
    work_overflow: Any


def probe_reference(
    tape: RandomTape,
    row_indices: Iterable[int],
    requested_global_ids: Iterable[int],
    requested_draws: Iterable[int],
    audit: TapeAudit,
    *,
    max_requests: Optional[int] = None,
) -> TapeProbeResult:
    """Consume tape values in arbitrary worker order using photon-row cursors."""

    rows = np.ascontiguousarray(row_indices, dtype=np.int32)
    global_ids = _global_ids(requested_global_ids)
    requests = np.ascontiguousarray(requested_draws, dtype=np.int32)
    if rows.ndim != 1 or global_ids.shape != rows.shape or requests.shape != rows.shape:
        raise ValueError("rows, global IDs, and draw requests must have equal 1D shape")
    if np.any(requests < 0):
        raise ValueError("requested_draws cannot be negative")
    valid_rows = rows[(rows >= 0) & (rows < tape.photon_count)]
    if np.unique(valid_rows).size != valid_rows.size:
        raise ValueError("each photon row may appear at most once per probe launch")
    if audit.interaction_cursor.shape != (tape.photon_count,):
        raise ValueError("audit buffers must match the tape photon count")
    if audit.draw_cursor.shape != (tape.photon_count,):
        raise ValueError("audit buffers must match the tape photon count")
    if audit.overflow.shape != (tape.photon_count,):
        raise ValueError("audit buffers must match the tape photon count")
    width = int(requests.max(initial=0)) if max_requests is None else int(max_requests)
    if width < int(requests.max(initial=0)):
        raise ValueError("max_requests is smaller than a requested draw count")

    output = np.full((len(rows), width), _CANONICAL_NAN, dtype=np.float32)
    result_audit = audit.copy()
    work_overflow = np.zeros(len(rows), dtype=np.uint32)
    for worker, (row, global_id, request_count) in enumerate(
        zip(rows, global_ids, requests)
    ):
        if row < 0 or row >= tape.photon_count:
            work_overflow[worker] |= ROW_OUT_OF_RANGE
            continue
        flags = np.uint32(result_audit.overflow[row])
        interaction = int(result_audit.interaction_cursor[row])
        draw = int(result_audit.draw_cursor[row])
        mapping_ok = int(tape.global_photon_ids[row]) == int(global_id)
        if not mapping_ok:
            flags |= GLOBAL_ID_MISMATCH
        interaction_ok = 0 <= interaction < tape.spec.max_interactions
        if mapping_ok and not interaction_ok:
            flags |= INTERACTION_OVERFLOW
        for request in range(int(request_count)):
            if (
                mapping_ok
                and interaction_ok
                and 0 <= draw < tape.spec.draws_per_interaction
            ):
                output[worker, request] = tape.values[row, interaction, draw]
                draw += 1
            elif mapping_ok and interaction_ok:
                flags |= DRAW_OVERFLOW
                draw = tape.spec.draws_per_interaction
        result_audit.draw_cursor[row] = draw
        result_audit.overflow[row] = flags
        work_overflow[worker] = flags
    return TapeProbeResult(output, result_audit, work_overflow)


def advance_interactions_reference(
    audit: TapeAudit,
    row_indices: Iterable[int],
    spec: RandomTapeSpec,
) -> TapeAudit:
    """Finish one physical interaction, resetting its local draw cursor."""

    rows = np.ascontiguousarray(row_indices, dtype=np.int32)
    if np.any(rows < 0) or np.any(rows >= len(audit.interaction_cursor)):
        raise ValueError("row index is outside the audit buffers")
    if np.unique(rows).size != rows.size:
        raise ValueError("row indices must be unique")
    result = audit.copy()
    result.interaction_cursor[rows] += np.int32(1)
    result.draw_cursor[rows] = np.int32(0)
    exhausted = result.interaction_cursor[rows] >= int(spec.max_interactions)
    result.overflow[rows[exhausted]] |= INTERACTION_OVERFLOW
    return result


@dataclass(frozen=True)
class TorchRandomTape:
    values: Any
    global_photon_ids: Any
    spec: RandomTapeSpec


@dataclass
class TorchTapeAudit:
    interaction_cursor: Any
    draw_cursor: Any
    overflow: Any


@dataclass
class TorchInteractionCertificate:
    """Torch-backed row/interaction certificate stored as raw int32 words."""

    words: Any


@dataclass
class TorchStateCertificate:
    """Torch-backed raw state certificate stored as signed int32 words."""

    words: Any


def to_torch(tape: RandomTape, device: Any = "cuda") -> TorchRandomTape:
    import torch

    return TorchRandomTape(
        values=torch.from_numpy(tape.values.copy()).to(device=device),
        global_photon_ids=torch.from_numpy(tape.global_photon_ids.copy()).to(
            device=device
        ),
        spec=tape.spec,
    )


def allocate_torch_audit(photon_count: int, device: Any = "cuda") -> TorchTapeAudit:
    import torch

    return TorchTapeAudit(
        interaction_cursor=torch.zeros(
            photon_count, dtype=torch.int32, device=device
        ),
        draw_cursor=torch.zeros(photon_count, dtype=torch.int32, device=device),
        overflow=torch.zeros(photon_count, dtype=torch.int32, device=device),
    )


def allocate_torch_certificate(
    photon_count: int,
    max_interactions: int,
    device: Any = "cuda",
) -> TorchInteractionCertificate:
    """Allocate an empty dense certificate without requiring torch.uint32."""

    import torch

    photon_count = int(photon_count)
    max_interactions = int(max_interactions)
    if photon_count < 0:
        raise ValueError("photon_count cannot be negative")
    if max_interactions <= 0:
        raise ValueError("max_interactions must be positive")
    return TorchInteractionCertificate(words=torch.full(
        (photon_count, max_interactions),
        -1,
        dtype=torch.int32,
        device=device,
    ))


def allocate_torch_state_certificate(
    photon_count: int,
    max_interactions: int,
    device: Any = "cuda",
) -> TorchStateCertificate:
    """Allocate a sentinel-filled dense raw state certificate for Triton."""

    import torch

    photon_count = int(photon_count)
    max_interactions = int(max_interactions)
    if photon_count < 0:
        raise ValueError("photon_count cannot be negative")
    if max_interactions <= 0:
        raise ValueError("max_interactions must be positive")
    return TorchStateCertificate(words=torch.full(
        (
            photon_count,
            max_interactions,
            STATE_CERTIFICATE_FIELD_COUNT,
        ),
        -1,
        dtype=torch.int32,
        device=device,
    ))


try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:  # pragma: no cover - CPU-only installations
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def random_tape_uniform_at(
        tape_values,
        row,
        interaction,
        draw,
        valid,
        MAX_INTERACTIONS: tl.constexpr,
        DRAWS_PER_INTERACTION: tl.constexpr,
    ):
        """Load one tape value without mutating cursors.

        Transport kernels can keep cursors in registers, call this primitive,
        and commit them once at the end of the interaction.
        """

        in_range = (
            valid
            & (interaction >= 0)
            & (interaction < MAX_INTERACTIONS)
            & (draw >= 0)
            & (draw < DRAWS_PER_INTERACTION)
        )
        offset = (
            (row * MAX_INTERACTIONS + interaction) * DRAWS_PER_INTERACTION
            + draw
        )
        nan_value = tl.full(row.shape, 0x7FC00000, tl.uint32).to(
            tl.float32, bitcast=True
        )
        return tl.load(tape_values + offset, mask=in_range, other=nan_value)

    @triton.jit
    def _random_tape_probe_kernel(
        tape_values,
        tape_global_ids,
        row_indices,
        requested_global_ids,
        requested_draws,
        interaction_cursor,
        draw_cursor,
        overflow,
        output,
        work_overflow,
        nwork,
        photon_count,
        MAX_INTERACTIONS: tl.constexpr,
        DRAWS_PER_INTERACTION: tl.constexpr,
        MAX_REQUESTS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        worker = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        work_valid = worker < nwork
        row = tl.load(row_indices + worker, mask=work_valid, other=-1)
        row_valid = work_valid & (row >= 0) & (row < photon_count)
        expected_global_id = tl.load(
            requested_global_ids + worker, mask=work_valid, other=-1
        )
        stored_global_id = tl.load(
            tape_global_ids + row, mask=row_valid, other=-2
        )
        interaction = tl.load(
            interaction_cursor + row, mask=row_valid, other=0
        )
        draw = tl.load(draw_cursor + row, mask=row_valid, other=0)
        flags = tl.load(overflow + row, mask=row_valid, other=0).to(tl.int32)
        # Triton JIT functions cannot capture NumPy scalar globals.  Keep these
        # literals synchronized with the public audit-bit constants above.
        flags |= tl.where(work_valid & ~row_valid, 8, 0)
        mapping_ok = row_valid & (stored_global_id == expected_global_id)
        flags |= tl.where(row_valid & ~mapping_ok, 4, 0)
        interaction_ok = (
            (interaction >= 0) & (interaction < MAX_INTERACTIONS)
        )
        flags |= tl.where(mapping_ok & ~interaction_ok, 2, 0)
        request_count = tl.load(
            requested_draws + worker, mask=work_valid, other=0
        )
        nan_value = tl.full(worker.shape, 0x7FC00000, tl.uint32).to(
            tl.float32, bitcast=True
        )
        for request in range(MAX_REQUESTS):
            requested = work_valid & (request < request_count)
            can_draw = (
                requested
                & mapping_ok
                & interaction_ok
                & (draw >= 0)
                & (draw < DRAWS_PER_INTERACTION)
            )
            value = random_tape_uniform_at(
                tape_values,
                row,
                interaction,
                draw,
                can_draw,
                MAX_INTERACTIONS=MAX_INTERACTIONS,
                DRAWS_PER_INTERACTION=DRAWS_PER_INTERACTION,
            )
            tl.store(
                output + worker * MAX_REQUESTS + request,
                tl.where(can_draw, value, nan_value),
                mask=work_valid,
            )
            draw_overflow = requested & mapping_ok & interaction_ok & (
                (draw < 0) | (draw >= DRAWS_PER_INTERACTION)
            )
            flags |= tl.where(draw_overflow, 1, 0)
            draw += can_draw.to(tl.int32)
            draw = tl.where(draw_overflow, DRAWS_PER_INTERACTION, draw)
        tl.store(draw_cursor + row, draw, mask=row_valid)
        tl.store(overflow + row, flags, mask=row_valid)
        tl.store(work_overflow + worker, flags, mask=work_valid)

else:  # pragma: no cover - names remain importable without Triton
    random_tape_uniform_at = None
    _random_tape_probe_kernel = None


def probe_triton(
    tape: TorchRandomTape,
    row_indices: Iterable[int],
    requested_global_ids: Iterable[int],
    requested_draws: Iterable[int],
    audit: TorchTapeAudit,
    *,
    max_requests: Optional[int] = None,
) -> TapeProbeResult:
    """Run the cursor-audited tape probe on Triton, mutating ``audit``."""

    import torch

    if triton is None or _random_tape_probe_kernel is None:
        raise RuntimeError("probe_triton requires Triton")
    device = tape.values.device
    rows_np = np.ascontiguousarray(row_indices, dtype=np.int32)
    ids_np = _global_ids(requested_global_ids)
    requests_np = np.ascontiguousarray(requested_draws, dtype=np.int32)
    if ids_np.shape != rows_np.shape or requests_np.shape != rows_np.shape:
        raise ValueError("rows, global IDs, and requests must have equal shape")
    if np.any(requests_np < 0):
        raise ValueError("requested draws cannot be negative")
    valid_rows = rows_np[(rows_np >= 0) & (rows_np < tape.values.shape[0])]
    if np.unique(valid_rows).size != valid_rows.size:
        raise ValueError("each photon row may appear at most once per probe launch")
    width = (
        int(requests_np.max(initial=0))
        if max_requests is None
        else int(max_requests)
    )
    if width < int(requests_np.max(initial=0)):
        raise ValueError("max_requests is smaller than a requested draw count")
    if width <= 0:
        raise ValueError("Triton probes require max_requests to be positive")
    rows = torch.from_numpy(rows_np).to(device=device)
    ids = torch.from_numpy(ids_np).to(device=device)
    requests = torch.from_numpy(requests_np).to(device=device)
    output = torch.empty(
        (len(rows_np), width), dtype=torch.float32, device=device
    )
    work_overflow = torch.empty(
        len(rows_np), dtype=torch.int32, device=device
    )
    block = 128
    _random_tape_probe_kernel[(triton.cdiv(len(rows_np), block),)](
        tape.values,
        tape.global_photon_ids,
        rows,
        ids,
        requests,
        audit.interaction_cursor,
        audit.draw_cursor,
        audit.overflow,
        output,
        work_overflow,
        len(rows_np),
        tape.values.shape[0],
        MAX_INTERACTIONS=int(tape.spec.max_interactions),
        DRAWS_PER_INTERACTION=int(tape.spec.draws_per_interaction),
        MAX_REQUESTS=width,
        BLOCK_SIZE=block,
        num_warps=4,
    )
    return TapeProbeResult(output, audit, work_overflow)


@dataclass(frozen=True)
class PyCudaRandomTape:
    values: Any
    global_photon_ids: Any
    spec: RandomTapeSpec


@dataclass
class PyCudaTapeAudit:
    interaction_cursor: Any
    draw_cursor: Any
    overflow: Any


@dataclass
class PyCudaInteractionCertificate:
    """PyCUDA-backed dense certificate indexed by stable tape row."""

    words: Any


@dataclass
class PyCudaStateCertificate:
    """PyCUDA-backed raw Photon state indexed by stable tape row."""

    words: Any


@dataclass
class PyCudaTapeTrace:
    """Device trace/audit outputs from one or more legacy tape steps.

    Per-photon fields describe the most recently completed interaction.  The
    first-error scalars are initialized to ``(-1, 0, -1, -1, mapping)`` and
    atomically capture the first mapping/capacity failure in a launch.
    """

    process: Any
    interaction: Any
    draw_count: Any
    stage: Any
    first_error_photon_id: Any
    first_error_flags: Any
    first_error_interaction: Any
    first_error_draw: Any
    first_error_stage: Any


def to_pycuda(tape: RandomTape) -> PyCudaRandomTape:
    from pycuda import gpuarray

    return PyCudaRandomTape(
        values=gpuarray.to_gpu(np.ascontiguousarray(tape.values.reshape(-1))),
        global_photon_ids=gpuarray.to_gpu(tape.global_photon_ids.copy()),
        spec=tape.spec,
    )


def allocate_pycuda_audit(photon_count: int) -> PyCudaTapeAudit:
    from pycuda import gpuarray

    return PyCudaTapeAudit(
        # ``GPUArray.zeros`` launches PyCUDA's generated fill kernel, which
        # adds an unrelated runtime-compiler dependency to this validation
        # path.  Uploading canonical host zeros keeps setup deterministic.
        interaction_cursor=gpuarray.to_gpu(np.zeros(photon_count, dtype=np.int32)),
        draw_cursor=gpuarray.to_gpu(np.zeros(photon_count, dtype=np.int32)),
        overflow=gpuarray.to_gpu(np.zeros(photon_count, dtype=np.uint32)),
    )


def allocate_pycuda_certificate(
    photon_count: int,
    max_interactions: int,
) -> PyCudaInteractionCertificate:
    """Allocate an empty uint32 certificate for the legacy tape kernel."""

    from pycuda import gpuarray

    host = InteractionCertificate.empty(photon_count, max_interactions)
    return PyCudaInteractionCertificate(words=gpuarray.to_gpu(host.words))


def allocate_pycuda_state_certificate(
    photon_count: int,
    max_interactions: int,
) -> PyCudaStateCertificate:
    """Allocate a sentinel-filled uint32 state certificate for CUDA."""

    from pycuda import gpuarray

    host = StateCertificate.empty(photon_count, max_interactions)
    return PyCudaStateCertificate(words=gpuarray.to_gpu(host.words))


def allocate_pycuda_trace(photon_slots: int) -> PyCudaTapeTrace:
    """Allocate deterministic trace buffers for ``propagate_tape``."""

    from pycuda import gpuarray

    photon_slots = int(photon_slots)
    if photon_slots < 0:
        raise ValueError("photon_slots cannot be negative")

    def _i32(values: np.ndarray):
        return gpuarray.to_gpu(np.ascontiguousarray(values, dtype=np.int32))

    return PyCudaTapeTrace(
        process=_i32(np.full(photon_slots, PROCESS_UNSET)),
        interaction=_i32(np.full(photon_slots, -1)),
        draw_count=_i32(np.full(photon_slots, -1)),
        stage=_i32(np.full(photon_slots, STAGE_MAPPING)),
        first_error_photon_id=_i32(np.asarray([-1])),
        first_error_flags=gpuarray.to_gpu(np.zeros(1, dtype=np.uint32)),
        first_error_interaction=_i32(np.asarray([-1])),
        first_error_draw=_i32(np.asarray([-1])),
        first_error_stage=_i32(np.asarray([STAGE_MAPPING])),
    )


_legacy_tape_modules: dict[tuple[int, str], Any] = {}


def _nvrtc_tape_module(source_name: str) -> Any:
    """Compile a tape debug module when standalone ``nvcc`` is unavailable.

    Some Torch installations ship NVRTC and CUDA headers but not the nvcc
    executable.  Lockstep validation should still work there.  NVRTC lacks a
    host C library include path, so three tiny standards-compatible header
    fragments provide only the declarations used by Chroma device headers.
    """

    from cuda.bindings import nvrtc
    import pycuda.driver as cuda
    from chroma.cuda import srcdir

    source_path = Path(srcdir) / source_name
    source = source_path.read_bytes()
    virtual_headers = [
        b'extern "C" __device__ int printf(const char *, ...);\n',
        (
            b"typedef signed char int8_t; typedef unsigned char uint8_t; "
            b"typedef short int16_t; typedef unsigned short uint16_t; "
            b"typedef int int32_t; typedef unsigned int uint32_t; "
            b"typedef long long int64_t; "
            b"typedef unsigned long long uint64_t;\n"
        ),
        b"#define FLT_EPSILON 1.1920928955078125e-7F\n",
    ]
    header_names = [b"stdio.h", b"stdint.h", b"float.h"]
    create_result, program = nvrtc.nvrtcCreateProgram(
        source,
        source_name.encode(),
        len(virtual_headers),
        virtual_headers,
        header_names,
    )
    if int(create_result) != 0:
        raise RuntimeError(f"NVRTC program creation failed: {create_result}")

    major, minor = cuda.Context.get_device().compute_capability()
    options = [
        f"--gpu-architecture=compute_{major}{minor}".encode(),
        b"--use_fast_math",
        b"--std=c++14",
        f"-I{srcdir}".encode(),
        f"-I{Path(sys.prefix) / 'include'}".encode(),
        (
            "-DCHROMA_FORCE_SCATTER_AT_PASS="
            f"{LEGACY_TAPE_FORCE_SCATTER_AT_PASS}"
        ).encode(),
    ]
    compile_result = nvrtc.nvrtcCompileProgram(
        program, len(options), options
    )[0]
    _, log_size = nvrtc.nvrtcGetProgramLogSize(program)
    log = bytearray(log_size)
    nvrtc.nvrtcGetProgramLog(program, log)
    if int(compile_result) != 0:
        message = bytes(log).rstrip(b"\0").decode(errors="replace")
        raise RuntimeError(f"NVRTC {source_name} compile failed:\n{message}")
    _, ptx_size = nvrtc.nvrtcGetPTXSize(program)
    ptx = bytearray(ptx_size)
    nvrtc.nvrtcGetPTX(program, ptx)
    nvrtc.nvrtcDestroyProgram(program)
    return cuda.module_from_buffer(bytes(ptx))


def get_legacy_tape_module() -> Any:
    """Return the context-local PyCUDA module for tape-backed propagation."""

    import pycuda.driver as cuda

    context = cuda.Context.get_current()
    if context is None:
        raise RuntimeError("a current PyCUDA context is required")
    key = (hash(context), "propagate_tape.cu")
    module = _legacy_tape_modules.get(key)
    if module is not None:
        return module
    if shutil.which("nvcc") is not None:
        from chroma.gpu.tools import cuda_options, get_cu_module

        options = cuda_options + (
            "-DCHROMA_FORCE_SCATTER_AT_PASS="
            f"{LEGACY_TAPE_FORCE_SCATTER_AT_PASS}",
        )
        module = get_cu_module("propagate_tape.cu", options=options)
    else:
        module = _nvrtc_tape_module("propagate_tape.cu")
    _legacy_tape_modules[key] = module
    return module


def legacy_tape_compile_policy() -> dict[str, Any]:
    """Return the physics-relevant compiler policy used by the tape module."""

    backend = "nvcc" if shutil.which("nvcc") is not None else "nvrtc"
    if backend == "nvcc":
        from chroma.gpu.tools import cuda_options

        options = list(cuda_options) + [
            "-DCHROMA_FORCE_SCATTER_AT_PASS="
            f"{LEGACY_TAPE_FORCE_SCATTER_AT_PASS}"
        ]
    else:
        options = [
            "--use_fast_math",
            "--std=c++14",
            "-DCHROMA_FORCE_SCATTER_AT_PASS="
            f"{LEGACY_TAPE_FORCE_SCATTER_AT_PASS}",
        ]
    return {
        "backend": backend,
        "force_scatter_at_pass": LEGACY_TAPE_FORCE_SCATTER_AT_PASS,
        "options": options,
    }


def get_rng_alignment_probe_module() -> Any:
    """Return the context-local CUDA tape probe module (nvcc or NVRTC)."""

    import pycuda.driver as cuda

    context = cuda.Context.get_current()
    if context is None:
        raise RuntimeError("a current PyCUDA context is required")
    key = (hash(context), "rng_alignment.cu")
    module = _legacy_tape_modules.get(key)
    if module is not None:
        return module
    if shutil.which("nvcc") is not None:
        from chroma.gpu.tools import cuda_options, get_cu_module

        module = get_cu_module("rng_alignment.cu", options=cuda_options)
    else:
        module = _nvrtc_tape_module("rng_alignment.cu")
    _legacy_tape_modules[key] = module
    return module


def launch_legacy_tape_step(
    *,
    gpu_photons: Any,
    gpu_geometry: Any,
    input_queue: Any,
    output_queue: Any,
    photon_tape_rows: Any,
    photon_global_ids: Any,
    tape: PyCudaRandomTape,
    audit: PyCudaTapeAudit,
    trace: PyCudaTapeTrace,
    certificate: Optional[PyCudaInteractionCertificate] = None,
    state_certificate: Optional[PyCudaStateCertificate] = None,
    nthreads: Optional[int] = None,
    first_photon: int = 0,
    max_steps: int = 1,
    use_weights: bool = False,
    scatter_first: int = 0,
    threads_per_block: int = 256,
) -> PyCudaTapeTrace:
    """Launch the opt-in legacy lockstep kernel.

    ``input_queue`` points directly at photon IDs (unlike Chroma's host queue,
    it has no count element).  ``output_queue[0]`` is the warp-aggregated
    counter and should normally be initialized to one, matching production
    Chroma.  All mapping, tape and audit arrays are PyCUDA device arrays.
    ``state_certificate`` is an optional 15-word raw post-interaction ledger;
    it requires the paired process/draw ``certificate`` so occupancy remains
    independently auditable.
    """

    if nthreads is None:
        nthreads = int(input_queue.size) - int(first_photon)
    nthreads = int(nthreads)
    threads_per_block = int(threads_per_block)
    if nthreads < 0:
        raise ValueError("nthreads cannot be negative")
    if threads_per_block <= 0:
        raise ValueError("threads_per_block must be positive")
    if nthreads == 0:
        return trace
    if certificate is not None:
        if not isinstance(certificate, PyCudaInteractionCertificate):
            raise TypeError(
                "certificate must be a PyCudaInteractionCertificate"
            )
        words = certificate.words
        expected_shape = (
            int(tape.global_photon_ids.size),
            int(tape.spec.max_interactions),
        )
        if words.dtype != np.dtype(np.uint32) or words.shape != expected_shape:
            raise ValueError(
                "certificate words must be uint32 [tape row, interaction]"
            )
        if not words.flags.c_contiguous:
            raise ValueError("certificate words must be C-contiguous")
        if int(tape.spec.draws_per_interaction) > int(CERTIFICATE_DRAW_MASK):
            raise ValueError("certificate draw counts must fit 28 bits")
        certificate_words = words
        certify_interactions = 1
    else:
        # A real device pointer avoids relying on PyCUDA's scalar-to-pointer
        # coercion.  The tape-only CUDA kernel checks this flag before writes.
        certificate_words = audit.overflow
        certify_interactions = 0

    if state_certificate is not None:
        if certificate is None:
            raise ValueError(
                "state_certificate requires an interaction certificate"
            )
        if not isinstance(state_certificate, PyCudaStateCertificate):
            raise TypeError(
                "state_certificate must be a PyCudaStateCertificate"
            )
        state_words = state_certificate.words
        expected_state_shape = (
            int(tape.global_photon_ids.size),
            int(tape.spec.max_interactions),
            STATE_CERTIFICATE_FIELD_COUNT,
        )
        if (
            state_words.dtype != np.dtype(np.uint32)
            or state_words.shape != expected_state_shape
        ):
            raise ValueError(
                "state certificate words must be uint32 "
                "[tape row, interaction, field]"
            )
        if not state_words.flags.c_contiguous:
            raise ValueError("state certificate words must be C-contiguous")
        state_certificate_words = state_words
        certify_states = 1
    else:
        # As above, supply a valid but unwritten device pointer for the
        # certificate-disabled debug-kernel variant.
        state_certificate_words = audit.overflow
        certify_states = 0

    module = get_legacy_tape_module()
    kernel = module.get_function("propagate_tape")
    blocks = (nthreads + threads_per_block - 1) // threads_per_block
    geometry_pointer = getattr(gpu_geometry, "gpudata", gpu_geometry)
    kernel(
        np.int32(first_photon),
        np.int32(nthreads),
        input_queue,
        output_queue,
        gpu_photons.pos,
        gpu_photons.dir,
        gpu_photons.wavelengths,
        gpu_photons.pol,
        gpu_photons.t,
        gpu_photons.flags,
        gpu_photons.last_hit_triangles,
        gpu_photons.weights,
        gpu_photons.evidx,
        np.int32(max_steps),
        np.int32(bool(use_weights)),
        np.int32(scatter_first),
        geometry_pointer,
        photon_tape_rows,
        photon_global_ids,
        tape.values,
        tape.global_photon_ids,
        np.int32(tape.global_photon_ids.size),
        np.int32(tape.spec.max_interactions),
        np.int32(tape.spec.draws_per_interaction),
        audit.interaction_cursor,
        audit.draw_cursor,
        audit.overflow,
        certificate_words,
        np.int32(certify_interactions),
        state_certificate_words,
        np.int32(certify_states),
        trace.process,
        trace.interaction,
        trace.draw_count,
        trace.stage,
        trace.first_error_photon_id,
        trace.first_error_flags,
        trace.first_error_interaction,
        trace.first_error_draw,
        trace.first_error_stage,
        block=(threads_per_block, 1, 1),
        grid=(blocks, 1, 1),
    )
    return trace


__all__ = [
    "CERTIFICATE_DRAW_MASK",
    "CERTIFICATE_EMPTY_WORD",
    "CERTIFICATE_PROCESS_SHIFT",
    "STATE_CERTIFICATE_EMPTY_WORD",
    "STATE_CERTIFICATE_FIELD_COUNT",
    "STATE_CERTIFICATE_FIELD_INDEX",
    "STATE_CERTIFICATE_FIELDS",
    "DRAW_OVERFLOW",
    "GLOBAL_ID_MISMATCH",
    "INTERACTION_OVERFLOW",
    "LEGACY_TAPE_FORCE_SCATTER_AT_PASS",
    "ROW_OUT_OF_RANGE",
    "PROCESS_UNSET",
    "PROCESS_BULK_ABSORB",
    "PROCESS_BULK_SCATTER",
    "PROCESS_SURFACE_ABSORB",
    "PROCESS_SURFACE_DETECT",
    "PROCESS_SURFACE_DIFFUSE",
    "PROCESS_SURFACE_SPECULAR",
    "PROCESS_DIELECTRIC_REFLECT",
    "PROCESS_DIELECTRIC_TRANSMIT",
    "PROCESS_BULK_REEMIT",
    "PROCESS_SURFACE_REEMIT",
    "STAGE_MAPPING",
    "STAGE_BULK",
    "STAGE_SURFACE",
    "STAGE_DIELECTRIC",
    "InteractionCertificate",
    "PyCudaInteractionCertificate",
    "PyCudaStateCertificate",
    "PyCudaRandomTape",
    "PyCudaTapeAudit",
    "PyCudaTapeTrace",
    "RandomTape",
    "RandomTapeSpec",
    "StateCertificate",
    "TapeAudit",
    "TapeProbeResult",
    "TorchInteractionCertificate",
    "TorchStateCertificate",
    "TorchRandomTape",
    "TorchTapeAudit",
    "advance_interactions_reference",
    "allocate_pycuda_audit",
    "allocate_pycuda_certificate",
    "allocate_pycuda_state_certificate",
    "allocate_pycuda_trace",
    "allocate_torch_audit",
    "allocate_torch_certificate",
    "allocate_torch_state_certificate",
    "probe_reference",
    "probe_triton",
    "launch_legacy_tape_step",
    "legacy_tape_compile_policy",
    "get_legacy_tape_module",
    "get_rng_alignment_probe_module",
    "random_tape_uniform_at",
    "pack_interaction_certificate",
    "to_pycuda",
    "to_torch",
    "unpack_interaction_certificate",
    "validate_interaction_certificate",
    "validate_state_certificate",
]
