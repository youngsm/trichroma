"""Device-resident diagnostics for fixed-round transport scheduling.

The asynchronous scheduler knows the logical round on the host but must not
read a device queue count after every launch.  :class:`DeviceRoundTrace`
therefore copies the boundary and next-pending ``DeviceQueue`` counters into
one preallocated CUDA ``int32[round, 2]`` ledger.  Recording is an ordinary
same-stream Triton launch and never materializes either counter on the host.

After all scheduled rounds, :meth:`DeviceRoundTrace.read` performs one batched
device-to-host transfer.  The pure NumPy parser then rejects negative counts
(including signed int32 wraparound) and counts larger than the corresponding
queue buffer observed when that row was recorded.

Torch and Triton are deliberately imported only by GPU entry points.  The
module and its host parser remain usable in CPU-only artifact tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Any, Optional

import numpy as np


class DeviceRoundTraceUnavailable(RuntimeError):
    """Raised when the optional Torch/Triton recording backend is unavailable."""


class RoundTraceValidationError(ValueError):
    """Raised when a materialized device counter violates its queue contract."""


@dataclass(frozen=True)
class RoundTrace:
    """Validated host representation of fixed-round queue counts."""

    records: np.ndarray

    @property
    def rounds(self) -> int:
        return int(len(self.records))

    @property
    def boundary_events(self) -> np.ndarray:
        return self.records[:, 0]

    @property
    def pending_survivors(self) -> np.ndarray:
        return self.records[:, 1]


def _capacity_vector(values: Any, rounds: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = np.full(rounds, array, dtype=np.int64)
    elif array.shape == (rounds,):
        array = np.asarray(array, dtype=np.int64)
    else:
        raise ValueError(f"{name} must be scalar or have shape ({rounds},)")
    if np.any(array < 0):
        raise ValueError(f"{name} cannot contain negative capacities")
    if np.any(array > np.iinfo(np.int32).max):
        raise ValueError(f"{name} exceeds the DeviceQueue int32 count range")
    return np.ascontiguousarray(array, dtype=np.int64)


def parse_round_trace(
    records: Any,
    boundary_capacities: Any,
    pending_capacities: Any,
) -> RoundTrace:
    """Validate and freeze one batched host copy of round counter records.

    Capacities may be scalars when every round reuses the same queue storage,
    or one-dimensional arrays when ping-pong buffers have different sizes.
    """

    raw = np.asarray(records)
    if raw.ndim != 2 or raw.shape[1:] != (2,):
        raise ValueError("records must have shape (rounds, 2)")
    if raw.dtype != np.int32:
        raise TypeError("records must have dtype int32")
    values = np.ascontiguousarray(raw, dtype=np.int32)
    rounds = len(values)
    boundary_limits = _capacity_vector(
        boundary_capacities, rounds, "boundary_capacities"
    )
    pending_limits = _capacity_vector(
        pending_capacities, rounds, "pending_capacities"
    )
    limits = np.column_stack((boundary_limits, pending_limits))

    negative = np.argwhere(values < 0)
    if len(negative):
        row, column = (int(value) for value in negative[0])
        label = "boundary" if column == 0 else "pending"
        raise RoundTraceValidationError(
            f"round {row} has negative {label} count {int(values[row, column])}"
        )
    overflow = np.argwhere(values.astype(np.int64) > limits)
    if len(overflow):
        row, column = (int(value) for value in overflow[0])
        label = "boundary" if column == 0 else "pending"
        raise RoundTraceValidationError(
            f"round {row} {label} count {int(values[row, column])} exceeds "
            f"queue capacity {int(limits[row, column])}"
        )

    frozen = np.array(values, copy=True, order="C")
    frozen.setflags(write=False)
    return RoundTrace(records=frozen)


def _validate_round_index(round_index: Any, capacity: int, next_round: int) -> int:
    if isinstance(round_index, (bool, np.bool_)):
        raise TypeError("round_index must be an integer, not bool")
    try:
        selected = operator.index(round_index)
    except TypeError as error:
        raise TypeError("round_index must be an integer") from error
    if selected < 0 or selected >= capacity:
        raise IndexError(
            f"round_index {selected} is outside trace capacity {capacity}"
        )
    if selected != next_round:
        raise ValueError(
            f"rounds must be recorded once in order: expected {next_round}, "
            f"received {selected}"
        )
    return selected


def _load_snapshot_kernel():
    cached = getattr(_load_snapshot_kernel, "_cached", None)
    if cached is not None:
        return cached
    try:
        import triton
        import triton.language as tl
    except Exception as error:  # pragma: no cover - environment dependent.
        raise DeviceRoundTraceUnavailable(
            "Triton is required to record device round counts"
        ) from error

    # Triton 3.1 resolves JIT names from the defining module rather than the
    # loader's closure.  Publishing only after a successful lazy import keeps
    # CPU-only module import optional.
    globals().update(triton=triton, tl=tl)

    @triton.jit(do_not_specialize=[3])
    def snapshot_round_counts_kernel(
        boundary_count,
        pending_count,
        records,
        round_index,
    ):
        column = tl.arange(0, 2)
        boundary = tl.load(boundary_count)
        pending = tl.load(pending_count)
        value = tl.where(column == 0, boundary, pending)
        tl.store(records + round_index * 2 + column, value)

    cached = (triton, snapshot_round_counts_kernel)
    _load_snapshot_kernel._cached = cached
    return cached


def _validate_device_queue(queue: Any, records: Any, name: str) -> int:
    try:
        import torch
        from chroma.triton.transport import DeviceQueue
    except Exception as error:  # pragma: no cover - environment dependent.
        raise DeviceRoundTraceUnavailable(
            "PyTorch and chroma.triton.transport are required"
        ) from error

    if not isinstance(queue, DeviceQueue):
        raise TypeError(f"{name} must be a DeviceQueue")
    if (
        not isinstance(queue.count, torch.Tensor)
        or not queue.count.is_cuda
        or queue.count.dtype != torch.int32
        or queue.count.shape != (1,)
        or not queue.count.is_contiguous()
        or queue.count.device != records.device
    ):
        raise ValueError(
            f"{name}.count must be contiguous same-device CUDA int32 with shape (1,)"
        )
    if (
        not isinstance(queue.buffer, torch.Tensor)
        or not queue.buffer.is_cuda
        or queue.buffer.ndim != 1
        or queue.buffer.dtype not in (torch.int32, torch.int64)
        or not queue.buffer.is_contiguous()
        or queue.buffer.device != records.device
    ):
        raise ValueError(
            f"{name}.buffer must be contiguous one-dimensional CUDA storage "
            "on the trace device"
        )
    capacity = int(queue.buffer.numel())
    if capacity > np.iinfo(np.int32).max:
        raise ValueError(f"{name} capacity exceeds the int32 counter range")
    return capacity


def _resolve_launch_capacity(
    requested: Optional[int], storage_capacity: int, name: str
) -> int:
    if requested is None:
        return storage_capacity
    if isinstance(requested, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        capacity = operator.index(requested)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if capacity < 0 or capacity > storage_capacity:
        raise ValueError(
            f"{name} must be between 0 and queue storage capacity "
            f"{storage_capacity}, got {capacity}"
        )
    return capacity


@dataclass
class DeviceRoundTrace:
    """Preallocated device ledger for boundary/pending counts by round."""

    records: Any
    _boundary_capacities: np.ndarray
    _pending_capacities: np.ndarray
    _rounds_recorded: int = 0

    @classmethod
    def allocate(cls, round_capacity: int, *, device: Any = "cuda") -> "DeviceRoundTrace":
        """Allocate the fixed ledger without importing Torch at module import."""

        try:
            import torch
        except Exception as error:  # pragma: no cover - environment dependent.
            raise DeviceRoundTraceUnavailable(
                "PyTorch is required to allocate a device round trace"
            ) from error
        if isinstance(round_capacity, (bool, np.bool_)):
            raise TypeError("round_capacity must be an integer, not bool")
        try:
            capacity = operator.index(round_capacity)
        except TypeError as error:
            raise TypeError("round_capacity must be an integer") from error
        if capacity < 0:
            raise ValueError("round_capacity cannot be negative")
        records = torch.empty((capacity, 2), dtype=torch.int32, device=device)
        if not records.is_cuda:
            raise ValueError("device round trace storage must be CUDA-resident")
        return cls(
            records=records,
            _boundary_capacities=np.full(capacity, -1, dtype=np.int64),
            _pending_capacities=np.full(capacity, -1, dtype=np.int64),
        )

    @property
    def capacity(self) -> int:
        return int(self.records.shape[0])

    @property
    def rounds_recorded(self) -> int:
        return int(self._rounds_recorded)

    def snapshot(
        self,
        round_index: int,
        boundary_events: Any,
        next_pending: Any,
        *,
        boundary_capacity: Optional[int] = None,
        next_pending_capacity: Optional[int] = None,
    ) -> None:
        """Asynchronously record two queue counters for one logical round.

        Optional capacities describe a scheduler launch cap smaller than the
        underlying buffer.  They are host metadata only and introduce no
        device read or synchronization.
        """

        selected = _validate_round_index(
            round_index, self.capacity, self._rounds_recorded
        )
        boundary_storage = _validate_device_queue(
            boundary_events, self.records, "boundary_events"
        )
        pending_storage = _validate_device_queue(
            next_pending, self.records, "next_pending"
        )
        selected_boundary_capacity = _resolve_launch_capacity(
            boundary_capacity, boundary_storage, "boundary_capacity"
        )
        selected_pending_capacity = _resolve_launch_capacity(
            next_pending_capacity, pending_storage, "next_pending_capacity"
        )
        triton, kernel = _load_snapshot_kernel()
        kernel[(1,)](
            boundary_events.count,
            next_pending.count,
            self.records,
            selected,
            num_warps=1,
        )
        self._boundary_capacities[selected] = selected_boundary_capacity
        self._pending_capacities[selected] = selected_pending_capacity
        self._rounds_recorded = selected + 1

    def reset(self) -> "DeviceRoundTrace":
        """Reuse the ledger without writing or clearing its device storage.

        Subsequent sequential snapshots overwrite every selected row.  The
        caller must retain ordinary CUDA stream ordering if a prior producer
        is still in flight.
        """

        self._boundary_capacities.fill(-1)
        self._pending_capacities.fill(-1)
        self._rounds_recorded = 0
        return self

    def read(self, rounds: Optional[int] = None) -> RoundTrace:
        """Perform one batched host read and validate every selected row."""

        if rounds is None:
            selected = self._rounds_recorded
        else:
            if isinstance(rounds, (bool, np.bool_)):
                raise TypeError("rounds must be an integer, not bool")
            try:
                selected = operator.index(rounds)
            except TypeError as error:
                raise TypeError("rounds must be an integer") from error
        if selected < 0 or selected > self._rounds_recorded:
            raise ValueError(
                f"rounds must be between 0 and {self._rounds_recorded}, got {selected}"
            )
        # This is intentionally the only device synchronization in the helper.
        host_records = self.records[:selected].detach().cpu().numpy()
        return parse_round_trace(
            host_records,
            self._boundary_capacities[:selected],
            self._pending_capacities[:selected],
        )


__all__ = [
    "DeviceRoundTrace",
    "DeviceRoundTraceUnavailable",
    "RoundTrace",
    "RoundTraceValidationError",
    "parse_round_trace",
]
