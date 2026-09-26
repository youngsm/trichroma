"""CPU-only coordination contracts for distributed Triton propagation.

Photon propagation does not require device-to-device communication.  A
distributed run can therefore keep one immutable scene and one CUDA context
resident in each worker process, send stable global-ID tiles to those
processes, and return only compact outputs.  This module implements that
control plane without importing CUDA, Torch, Triton, or multiprocessing.

The coordinator deliberately talks to a small :class:`WorkerClient` protocol.
An IPC implementation can put a client in the coordinator process and a
server owning exactly one CUDA context in each GPU process.  ``submit`` must
be non-blocking, which lets a worker's local H2D/compute/D2H ring remain full;
``max_inflight`` bounds the amount of queued input and output on either side.

Tiles come from :class:`~chroma.triton.runtime.EventBatchPlan`.  Their identity
and global photon ranges are fixed before workers are considered, so changing
the number or relative speed of GPUs cannot change RNG identity.  Scheduling
is dynamic: the next tile goes to the eligible worker with the lowest
capacity-normalized outstanding photon load.  Fast workers consequently claim
more work instead of receiving a static equal-sized partition.

Validated results enter a bounded single-thread output pipeline.  Per-tile hit
sorting, sparse DAQ reduction, and an optional application consumer therefore
overlap later GPU work rather than forming a serial end-of-run tail.  The
returned result is still restored to plan/global-photon order, and the reducer
uses tile identity to make equal-time DAQ ties independent of completion order.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import math
import threading
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np

from .runtime import EventBatchPlan, RuntimeContractError


class DistributedContractError(RuntimeContractError):
    """A distributed worker or result violated the coordination contract."""


class DistributedExecutionError(RuntimeError):
    """A worker failed while executing a tile.

    ``worker_id`` and ``tile_index`` are retained as structured attributes so
    orchestration can report or retry at the event layer.  This coordinator is
    fail-fast and never retries silently because retry policy may interact with
    output side effects in a concrete IPC implementation.
    """

    def __init__(self, worker_id: str, tile_index: int, cause: BaseException):
        self.worker_id = worker_id
        self.tile_index = int(tile_index)
        self.cause = cause
        super().__init__(
            f"distributed worker {worker_id!r} failed on tile {tile_index}: {cause}"
        )


class DistributedCancelled(RuntimeError):
    """A distributed run was cancelled before all tiles completed."""


class DistributedOutputError(RuntimeError):
    """Host-side result reduction or a streaming consumer failed.

    Output work runs independently of worker submission so that compact-hit
    sorting, sparse DAQ reduction, and application I/O can overlap later GPU
    tiles.  Keeping this error distinct from :class:`DistributedExecutionError`
    avoids incorrectly blaming a device worker for a host consumer failure.
    """

    def __init__(self, tile_index: int, cause: BaseException):
        self.tile_index = int(tile_index)
        self.cause = cause
        super().__init__(f"distributed output processing failed on tile {tile_index}: {cause}")


def _readonly_vector(value: Any, *, name: str, dtype: Any) -> np.ndarray:
    try:
        result = np.array(value, dtype=dtype, order="C", copy=True, subok=False)
    except Exception as exc:
        raise DistributedContractError(f"{name} is not array-compatible") from exc
    if result.ndim != 1:
        raise DistributedContractError(f"{name} must be one-dimensional")
    result.flags.writeable = False
    return result


def _strictly_increasing(values: np.ndarray) -> bool:
    """Return whether an integer vector is unique and already sorted.

    This is linear and allocation-free apart from NumPy's comparison result.
    It matters for large compact-hit buffers: ``np.unique`` sorts, so applying
    it to output that workers already returned in photon order needlessly adds
    an ``O(n log n)`` host-side tail after every GPU has gone idle.
    """

    return len(values) < 2 or bool(np.all(values[1:] > values[:-1]))


@dataclass(frozen=True)
class CompactHits:
    """Detected-photon rows small enough to return instead of full state.

    Rows are keyed by the stable global photon ID.  They intentionally contain
    no position, direction, polarization, wavelength, or traversal scratch.
    The coordinator restores global photon order after arbitrary worker
    completion order.
    """

    global_photon_ids: np.ndarray
    event_indices: np.ndarray
    channels: np.ndarray
    times: np.ndarray
    histories: np.ndarray

    def __post_init__(self) -> None:
        specifications = (
            ("global_photon_ids", self.global_photon_ids, np.int64),
            ("event_indices", self.event_indices, np.uint32),
            ("channels", self.channels, np.int32),
            ("times", self.times, np.float32),
            ("histories", self.histories, np.uint32),
        )
        count: Optional[int] = None
        for name, value, dtype in specifications:
            array = _readonly_vector(value, name=f"hits.{name}", dtype=dtype)
            if count is None:
                count = len(array)
            elif len(array) != count:
                raise DistributedContractError("compact-hit arrays must have equal length")
            object.__setattr__(self, name, array)
        ids_are_unique = not count or _strictly_increasing(self.global_photon_ids)
        if count and not ids_are_unique:
            ids_are_unique = len(np.unique(self.global_photon_ids)) == count
        if count and (np.any(self.global_photon_ids < 0) or not ids_are_unique):
            raise DistributedContractError(
                "compact-hit global photon IDs must be unique and non-negative"
            )
        if count and np.any(self.channels < 0):
            raise DistributedContractError("compact-hit channels must be non-negative")
        if count and not np.all(np.isfinite(self.times)):
            raise DistributedContractError("compact-hit times must be finite")

    @property
    def count(self) -> int:
        return len(self.global_photon_ids)

    @classmethod
    def empty(cls) -> "CompactHits":
        return cls(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.uint32),
        )


def merge_compact_hits(parts: Iterable[CompactHits]) -> CompactHits:
    """Merge compact hit shards into stable global-photon order."""

    parts = tuple(part for part in parts if part.count)
    if not parts:
        return CompactHits.empty()
    ids = np.concatenate([part.global_photon_ids for part in parts])
    if _strictly_increasing(ids):
        order = slice(None)
    else:
        # One stable ordering pass both restores global order and makes the
        # duplicate check linear.  The former implementation sorted once in
        # np.unique and a second time in argsort.
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        if np.any(sorted_ids[1:] == sorted_ids[:-1]):
            raise DistributedContractError(
                "compact-hit shards contain duplicate photon IDs"
            )
    return CompactHits(
        ids[order],
        np.concatenate([part.event_indices for part in parts])[order],
        np.concatenate([part.channels for part in parts])[order],
        np.concatenate([part.times for part in parts])[order],
        np.concatenate([part.histories for part in parts])[order],
    )


@dataclass(frozen=True)
class DaqPartial:
    """Sparse, mergeable DAQ state keyed by ``(event, channel)``.

    Each worker reduces all contributions it owns to at most one row per key.
    Cross-worker reduction is the commutative monoid used by Chroma's DAQ:
    minimum hit time, integer charge sum, and bitwise-OR history.  Missing keys
    are the identity and therefore need not be transmitted.
    """

    event_indices: np.ndarray
    channels: np.ndarray
    times: np.ndarray
    charges: np.ndarray
    histories: np.ndarray

    def __post_init__(self) -> None:
        specifications = (
            ("event_indices", self.event_indices, np.uint32),
            ("channels", self.channels, np.int32),
            ("times", self.times, np.float32),
            ("charges", self.charges, np.int64),
            ("histories", self.histories, np.uint32),
        )
        count: Optional[int] = None
        for name, value, dtype in specifications:
            array = _readonly_vector(value, name=f"daq.{name}", dtype=dtype)
            if count is None:
                count = len(array)
            elif len(array) != count:
                raise DistributedContractError("DAQ partial arrays must have equal length")
            object.__setattr__(self, name, array)
        if count and np.any(self.channels < 0):
            raise DistributedContractError("DAQ channels must be non-negative")
        if count and np.any(self.charges < 0):
            raise DistributedContractError("DAQ integer charges must be non-negative")
        if count and not np.all(np.isfinite(self.times)):
            raise DistributedContractError("DAQ times must be finite")
        if count:
            keys = np.rec.fromarrays(
                [self.event_indices, self.channels], names=("event", "channel")
            )
            if len(np.unique(keys)) != count:
                raise DistributedContractError(
                    "a DAQ partial must contain at most one row per event/channel"
                )

    @property
    def count(self) -> int:
        return len(self.event_indices)

    @classmethod
    def empty(cls) -> "DaqPartial":
        return cls(
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.uint32),
        )


def merge_daq_partials(parts: Iterable[DaqPartial]) -> DaqPartial:
    """Reduce sparse DAQ shards with ``min / integer sum / bitwise OR``.

    Accumulation uses Python integers and explicitly checks the signed-int64
    boundary, avoiding silent NumPy wraparound when many GPU shards meet.
    Output rows are sorted by event and then channel.
    """

    accumulators: dict[tuple[int, int], list[Any]] = {}
    for part in parts:
        for event, channel, time, charge, history in zip(
            part.event_indices,
            part.channels,
            part.times,
            part.charges,
            part.histories,
        ):
            key = (int(event), int(channel))
            row = accumulators.get(key)
            if row is None:
                accumulators[key] = [float(time), int(charge), int(history)]
            else:
                row[0] = min(row[0], float(time))
                row[1] += int(charge)
                row[2] |= int(history)
                if row[1] > np.iinfo(np.int64).max:
                    raise OverflowError(f"DAQ charge overflow for event/channel {key}")
    if not accumulators:
        return DaqPartial.empty()
    keys = sorted(accumulators)
    return DaqPartial(
        np.asarray([key[0] for key in keys], dtype=np.uint32),
        np.asarray([key[1] for key in keys], dtype=np.int32),
        np.asarray([accumulators[key][0] for key in keys], dtype=np.float32),
        np.asarray([accumulators[key][1] for key in keys], dtype=np.int64),
        np.asarray([accumulators[key][2] for key in keys], dtype=np.uint32),
    )


@dataclass(frozen=True)
class TileAssignment:
    """Serializable identity and input reference for one immutable plan tile."""

    run_id: str
    tile_index: int
    photon_count: int
    global_photon_start: int
    global_photon_stop: int
    event_indices: tuple[int, ...]
    payload: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise DistributedContractError("run_id must be a non-empty string")
        integer_values = (
            self.tile_index,
            self.photon_count,
            self.global_photon_start,
            self.global_photon_stop,
        )
        if any(int(value) != value for value in integer_values):
            raise DistributedContractError("tile identity fields must be integers")
        if self.tile_index < 0 or self.photon_count <= 0:
            raise DistributedContractError("a tile must have a non-negative ID and photons")
        if self.global_photon_start < 0:
            raise DistributedContractError("global photon IDs must be non-negative")
        if self.global_photon_stop - self.global_photon_start != self.photon_count:
            raise DistributedContractError("tile global range has the wrong length")
        events = tuple(int(value) for value in self.event_indices)
        if not events or any(value < 0 for value in events):
            raise DistributedContractError("tile event indices must be non-empty/non-negative")
        object.__setattr__(self, "event_indices", events)


def assignments_from_plan(
    plan: EventBatchPlan,
    *,
    run_id: str,
    payload_factory: Optional[Callable[[int], Any]] = None,
) -> tuple[TileAssignment, ...]:
    """Create N-GPU-independent assignments from an event batch plan.

    ``payload_factory`` is called once per tile and may return a materialized
    :class:`PhotonBatch`, a shared-memory descriptor, or an application IPC
    key.  It is deliberately unrelated to workers and GPU count.
    """

    assignments: list[TileAssignment] = []
    for tile in plan.tiles:
        first = tile.fragments[0]
        last = tile.fragments[-1]
        payload = None if payload_factory is None else payload_factory(tile.tile_index)
        assignments.append(
            TileAssignment(
                run_id=run_id,
                tile_index=tile.tile_index,
                photon_count=tile.photon_count,
                global_photon_start=first.global_photon_start,
                global_photon_stop=last.global_photon_stop,
                event_indices=tuple(
                    dict.fromkeys(fragment.event_index for fragment in tile.fragments)
                ),
                payload=payload,
            )
        )
    return tuple(assignments)


@dataclass(frozen=True)
class TileResult:
    """Compact result envelope returned by a worker process."""

    run_id: str
    worker_id: str
    tile_index: int
    global_photon_start: int
    global_photon_stop: int
    compact_hits: Optional[CompactHits] = None
    daq_partial: Optional[DaqPartial] = None
    elapsed_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.worker_id, str) or not self.worker_id:
            raise DistributedContractError("result worker_id must be non-empty")
        if self.elapsed_seconds is not None and (
            not math.isfinite(float(self.elapsed_seconds)) or self.elapsed_seconds < 0
        ):
            raise DistributedContractError("result elapsed_seconds must be finite/non-negative")


@runtime_checkable
class WorkerClient(Protocol):
    """Coordinator-side endpoint for one persistent GPU worker process.

    A concrete implementation normally sends ``TileAssignment`` over IPC and
    returns a :class:`concurrent.futures.Future`.  Its server process selects
    one device once, creates one CUDA context, uploads one scene, and feeds its
    local asynchronous pipeline.  No client method may switch devices.
    """

    @property
    def worker_id(self) -> str:
        """Stable process/GPU endpoint identity."""

    def submit(self, assignment: TileAssignment) -> Future[TileResult]:
        """Enqueue work without waiting for GPU or output completion."""

    def cancel(
        self, run_id: str, tile_indices: tuple[int, ...], reason: str
    ) -> None:
        """Stop/ignore a run and release queued transport/output buffers."""


@dataclass(frozen=True)
class WorkerSpec:
    """Scheduling capacity and backpressure limits for one worker endpoint."""

    client: WorkerClient = field(repr=False, compare=False)
    capacity_weight: float = 1.0
    max_inflight: int = 2
    max_tile_photons: Optional[int] = None
    max_inflight_photons: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.client.worker_id, str) or not self.client.worker_id:
            raise DistributedContractError("worker_id must be a non-empty string")
        if not math.isfinite(float(self.capacity_weight)) or self.capacity_weight <= 0:
            raise DistributedContractError("capacity_weight must be finite and positive")
        if int(self.max_inflight) != self.max_inflight or self.max_inflight <= 0:
            raise DistributedContractError("max_inflight must be a positive integer")
        if self.max_tile_photons is not None and (
            int(self.max_tile_photons) != self.max_tile_photons
            or self.max_tile_photons <= 0
        ):
            raise DistributedContractError("max_tile_photons must be positive or None")
        if self.max_inflight_photons is not None and (
            int(self.max_inflight_photons) != self.max_inflight_photons
            or self.max_inflight_photons <= 0
        ):
            raise DistributedContractError(
                "max_inflight_photons must be positive or None"
            )

    @property
    def worker_id(self) -> str:
        return self.client.worker_id

    def accepts(self, assignment: TileAssignment) -> bool:
        return self.max_tile_photons is None or (
            assignment.photon_count <= self.max_tile_photons
        )

    def has_queue_capacity(
        self, assignment: TileAssignment, *, inflight: int, inflight_photons: int
    ) -> bool:
        """Whether another tile fits both descriptor and photon budgets.

        A tile-count limit alone is unsafe for nonuniform event batches: two
        legal but large tiles can require much more staging/output memory than
        two small ones.  The optional photon budget provides byte-like
        backpressure without coupling this CPU-only module to a buffer layout.
        """

        if inflight >= self.max_inflight or not self.accepts(assignment):
            return False
        return self.max_inflight_photons is None or (
            inflight_photons + assignment.photon_count
            <= self.max_inflight_photons
        )


class CancellationToken:
    """Thread-safe cooperative cancellation usable across coordinator threads."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = "cancelled by caller"

    def cancel(self, reason: str = "cancelled by caller") -> None:
        with self._lock:
            if not self._event.is_set():
                self._reason = str(reason)
                self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason


@dataclass(frozen=True)
class WorkerRunStats:
    worker_id: str
    capacity_weight: float
    submitted_tiles: int
    completed_tiles: int
    submitted_photons: int
    completed_photons: int
    max_observed_inflight: int
    max_observed_inflight_photons: int = 0


@dataclass(frozen=True)
class DistributedScalingEstimate:
    """Steady-state N-GPU throughput bound for an overlapped pipeline.

    This is an accounting model, not a benchmark.  Each worker is limited by
    its slowest local stage because H2D, transport, and D2H are assumed to
    overlap.  Worker rates then add, after an explicit load-balance factor,
    and optional shared host-input/output services cap the aggregate.
    """

    worker_transport_rates: tuple[float, ...]
    worker_pipeline_rates: tuple[float, ...]
    ideal_aggregate_photons_per_second: float
    worker_limited_photons_per_second: float
    estimated_aggregate_photons_per_second: float
    parallel_efficiency: float
    limiting_stage: str

    @property
    def worker_count(self) -> int:
        return len(self.worker_transport_rates)

    def required_parallel_efficiency(self, target_photons_per_second: float) -> float:
        """Fraction of ideal aggregate capacity required for ``target``.

        Values above one mean the target cannot be reached merely by scaling
        this set of worker transport rates; the per-GPU kernel or worker count
        must improve.  This deliberately distinguishes an aggregate N-GPU
        target from a single-GPU throughput claim.
        """

        target = float(target_photons_per_second)
        if not math.isfinite(target) or target < 0:
            raise DistributedContractError(
                "target_photons_per_second must be finite and non-negative"
            )
        return target / self.ideal_aggregate_photons_per_second

    def reaches(self, target_photons_per_second: float) -> bool:
        target = float(target_photons_per_second)
        if not math.isfinite(target) or target < 0:
            raise DistributedContractError(
                "target_photons_per_second must be finite and non-negative"
            )
        return self.estimated_aggregate_photons_per_second >= target


def _rate_vector(
    value: Any,
    *,
    worker_count: Optional[int],
    name: str,
    default: Optional[float] = None,
) -> tuple[float, ...]:
    if value is None:
        if worker_count is None or default is None:
            raise DistributedContractError(f"{name} cannot be None")
        return (float(default),) * worker_count
    if np.isscalar(value):
        if worker_count is None:
            worker_count = 1
        rates = (float(value),) * worker_count
    else:
        try:
            rates = tuple(float(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise DistributedContractError(f"{name} must contain numeric rates") from exc
        if worker_count is not None and len(rates) != worker_count:
            raise DistributedContractError(
                f"{name} must contain exactly {worker_count} rates"
            )
    if not rates:
        raise DistributedContractError(f"{name} cannot be empty")
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise DistributedContractError(f"{name} rates must be finite and positive")
    return rates


def estimate_distributed_scaling(
    worker_transport_rates: Any,
    *,
    worker_count: Optional[int] = None,
    worker_h2d_rates: Any = None,
    worker_d2h_rates: Any = None,
    load_balance_efficiency: float = 1.0,
    shared_input_rate: Optional[float] = None,
    shared_output_rate: Optional[float] = None,
) -> DistributedScalingEstimate:
    """Estimate steady-state *aggregate* throughput for independent GPUs.

    Rates are in photons/s.  A scalar worker rate is broadcast using
    ``worker_count``; sequences model heterogeneous GPUs.  Omitted H2D/D2H
    rates mean those stages are not limiting.  ``shared_input_rate`` covers a
    common source/materialization service and ``shared_output_rate`` covers a
    central reducer or writer.  Startup/drain latency and workloads with fewer
    tiles than workers are intentionally outside this steady-state model.
    """

    if worker_count is not None and (
        int(worker_count) != worker_count or worker_count <= 0
    ):
        raise DistributedContractError("worker_count must be a positive integer or None")
    count = None if worker_count is None else int(worker_count)
    transport = _rate_vector(
        worker_transport_rates,
        worker_count=count,
        name="worker_transport_rates",
    )
    count = len(transport)
    h2d = _rate_vector(
        worker_h2d_rates,
        worker_count=count,
        name="worker_h2d_rates",
        default=math.inf,
    )
    d2h = _rate_vector(
        worker_d2h_rates,
        worker_count=count,
        name="worker_d2h_rates",
        default=math.inf,
    )
    efficiency = float(load_balance_efficiency)
    if not math.isfinite(efficiency) or not (0.0 < efficiency <= 1.0):
        raise DistributedContractError(
            "load_balance_efficiency must be finite and lie in (0, 1]"
        )

    def shared_rate(value: Optional[float], name: str) -> float:
        if value is None:
            return math.inf
        rate = float(value)
        if not math.isfinite(rate) or rate <= 0:
            raise DistributedContractError(f"{name} must be finite and positive")
        return rate

    input_rate = shared_rate(shared_input_rate, "shared_input_rate")
    output_rate = shared_rate(shared_output_rate, "shared_output_rate")
    pipeline = tuple(
        min(transport_rate, h2d_rate, d2h_rate)
        for transport_rate, h2d_rate, d2h_rate in zip(transport, h2d, d2h)
    )
    ideal = float(sum(transport))
    worker_limited = float(sum(pipeline) * efficiency)
    candidates = (
        ("worker_pipeline", worker_limited),
        ("shared_input", input_rate),
        ("shared_output", output_rate),
    )
    limiting_stage, aggregate = min(candidates, key=lambda item: item[1])
    return DistributedScalingEstimate(
        worker_transport_rates=transport,
        worker_pipeline_rates=pipeline,
        ideal_aggregate_photons_per_second=ideal,
        worker_limited_photons_per_second=worker_limited,
        estimated_aggregate_photons_per_second=float(aggregate),
        parallel_efficiency=float(aggregate / ideal),
        limiting_stage=limiting_stage,
    )


@dataclass(frozen=True)
class DistributedRunResult:
    """Ordered compact outputs and scheduling telemetry for a completed run."""

    run_id: str
    tile_results: tuple[TileResult, ...]
    compact_hits: CompactHits
    daq_partial: DaqPartial
    worker_stats: Mapping[str, WorkerRunStats]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tile_results", tuple(self.tile_results))
        object.__setattr__(
            self, "worker_stats", MappingProxyType(dict(self.worker_stats))
        )


@dataclass
class _WorkerState:
    spec: WorkerSpec
    ordinal: int
    inflight: dict[Future[TileResult], TileAssignment] = field(default_factory=dict)
    outstanding_photons: int = 0
    submitted_tiles: int = 0
    completed_tiles: int = 0
    submitted_photons: int = 0
    completed_photons: int = 0
    max_observed_inflight: int = 0
    max_observed_inflight_photons: int = 0

    def projected_score(self, assignment: TileAssignment) -> tuple[float, float, int]:
        return (
            (self.outstanding_photons + assignment.photon_count)
            / float(self.spec.capacity_weight),
            (len(self.inflight) + 1) / float(self.spec.max_inflight),
            self.ordinal,
        )


TileResultConsumer = Callable[[TileResult], None]


def _sort_compact_hits(hits: Optional[CompactHits]) -> Optional[CompactHits]:
    """Sort one shard while it can overlap later device work."""

    if hits is None or hits.count < 2 or _strictly_increasing(hits.global_photon_ids):
        return hits
    order = np.argsort(hits.global_photon_ids, kind="stable")
    return CompactHits(
        hits.global_photon_ids[order],
        hits.event_indices[order],
        hits.channels[order],
        hits.times[order],
        hits.histories[order],
    )


class _OutputAccumulator:
    """Serial, order-independent accumulator used by the output thread.

    Worker completion order is intentionally irrelevant.  Normalized results
    remain keyed by tile and are restored to plan order in :meth:`finish`.
    Compact-hit shards are sorted independently, turning the final global
    merge into a linear concatenation because assignment ID ranges are
    disjoint and ordered.  DAQ rows are reduced as they arrive rather than in
    a serial tail after all devices finish.
    """

    def __init__(self, consumer: Optional[TileResultConsumer]):
        self.consumer = consumer
        self.results: dict[int, TileResult] = {}
        self.hit_parts: dict[int, CompactHits] = {}
        self.daq_accumulators: dict[tuple[int, int], list[Any]] = {}

    def add(self, result: TileResult) -> None:
        tile_index = int(result.tile_index)
        if tile_index in self.results:
            raise DistributedContractError(
                f"output pipeline received tile {tile_index} more than once"
            )
        sorted_hits = _sort_compact_hits(result.compact_hits)
        if sorted_hits is not result.compact_hits:
            result = TileResult(
                run_id=result.run_id,
                worker_id=result.worker_id,
                tile_index=result.tile_index,
                global_photon_start=result.global_photon_start,
                global_photon_stop=result.global_photon_stop,
                compact_hits=sorted_hits,
                daq_partial=result.daq_partial,
                elapsed_seconds=result.elapsed_seconds,
            )
        self.results[tile_index] = result
        if sorted_hits is not None and sorted_hits.count:
            self.hit_parts[tile_index] = sorted_hits

        part = result.daq_partial
        if part is not None:
            for row_index, (event, channel, time, charge, history) in enumerate(
                zip(
                    part.event_indices,
                    part.channels,
                    part.times,
                    part.charges,
                    part.histories,
                )
            ):
                key = (int(event), int(channel))
                row = self.daq_accumulators.get(key)
                if row is None:
                    self.daq_accumulators[key] = [
                        float(time),
                        int(charge),
                        int(history),
                        tile_index,
                        row_index,
                    ]
                else:
                    candidate_time = float(time)
                    candidate_order = (tile_index, row_index)
                    previous_order = (row[3], row[4])
                    if candidate_time < row[0] or (
                        candidate_time == row[0]
                        and candidate_order < previous_order
                    ):
                        # Tie-breaking by plan position exactly reproduces the
                        # old ordered merge, including the bit pattern of
                        # otherwise equal +0.0/-0.0 values, regardless of GPU
                        # completion order.
                        row[0] = candidate_time
                        row[3] = tile_index
                        row[4] = row_index
                    row[1] += int(charge)
                    row[2] |= int(history)
                    if row[1] > np.iinfo(np.int64).max:
                        raise OverflowError(
                            f"DAQ charge overflow for event/channel {key}"
                        )

        # A filesystem/network writer can consume each compact tile here while
        # worker queues remain full.  Calls are serialized on one output thread
        # but intentionally occur in completion order; tile/global IDs carry
        # deterministic identity and the final public tuple remains plan ordered.
        if self.consumer is not None:
            self.consumer(result)

    def finish(
        self, tile_count: int
    ) -> tuple[tuple[TileResult, ...], CompactHits, DaqPartial]:
        expected = set(range(tile_count))
        actual = set(self.results)
        if actual != expected:
            missing = sorted(expected.difference(actual))
            extra = sorted(actual.difference(expected))
            raise DistributedContractError(
                f"output pipeline tile mismatch: missing={missing}, extra={extra}"
            )
        ordered_results = tuple(self.results[index] for index in range(tile_count))
        compact_hits = merge_compact_hits(
            self.hit_parts[index] for index in sorted(self.hit_parts)
        )
        if not self.daq_accumulators:
            daq_partial = DaqPartial.empty()
        else:
            keys = sorted(self.daq_accumulators)
            daq_partial = DaqPartial(
                np.asarray([key[0] for key in keys], dtype=np.uint32),
                np.asarray([key[1] for key in keys], dtype=np.int32),
                np.asarray(
                    [self.daq_accumulators[key][0] for key in keys],
                    dtype=np.float32,
                ),
                np.asarray(
                    [self.daq_accumulators[key][1] for key in keys],
                    dtype=np.int64,
                ),
                np.asarray(
                    [self.daq_accumulators[key][2] for key in keys],
                    dtype=np.uint32,
                ),
            )
        return ordered_results, compact_hits, daq_partial


class _OutputPipeline:
    """Bounded single-thread host-output pipeline.

    One reducer thread is enough because reduction must be serialized, while a
    bounded future queue prevents slow storage or DAQ code from consuming
    unbounded host memory.  Backpressure is applied only after device workers
    have already been refilled, preserving compute/communication overlap.
    """

    def __init__(
        self,
        *,
        overlap: bool,
        max_pending_tiles: int,
        consumer: Optional[TileResultConsumer],
    ) -> None:
        self.accumulator = _OutputAccumulator(consumer)
        self.max_pending_tiles = int(max_pending_tiles)
        self.executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="chroma-output")
            if overlap
            else None
        )
        self.pending: dict[Future[None], int] = {}

    @staticmethod
    def _raise_output_error(tile_index: int, exc: BaseException) -> None:
        if isinstance(exc, DistributedOutputError):
            raise exc
        raise DistributedOutputError(tile_index, exc) from exc

    def _resolve(self, futures: Iterable[Future[None]]) -> None:
        for future in tuple(futures):
            tile_index = self.pending.pop(future)
            try:
                future.result()
            except BaseException as exc:
                self._raise_output_error(tile_index, exc)

    def _drain_ready(self) -> None:
        self._resolve(future for future in tuple(self.pending) if future.done())

    def submit(self, result: TileResult) -> None:
        if self.executor is None:
            try:
                self.accumulator.add(result)
            except BaseException as exc:
                self._raise_output_error(result.tile_index, exc)
            return
        future = self.executor.submit(self.accumulator.add, result)
        self.pending[future] = result.tile_index
        self._drain_ready()

    def apply_backpressure(self) -> None:
        """Bound the reducer queue after newly free GPU slots are refilled."""

        self._drain_ready()
        while len(self.pending) >= self.max_pending_tiles:
            done, _ = wait(tuple(self.pending), return_when=FIRST_COMPLETED)
            self._resolve(done)

    def finish(
        self, tile_count: int
    ) -> tuple[tuple[TileResult, ...], CompactHits, DaqPartial]:
        while self.pending:
            done, _ = wait(tuple(self.pending), return_when=FIRST_COMPLETED)
            self._resolve(done)
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
        return self.accumulator.finish(tile_count)

    def abort(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None


class DistributedCoordinator:
    """Dynamically dispatch stable tiles to bounded asynchronous workers."""

    def __init__(
        self,
        workers: Sequence[WorkerSpec],
        *,
        poll_seconds: float = 0.02,
        overlap_output: bool = True,
        max_pending_output_tiles: Optional[int] = None,
    ):
        self.workers = tuple(workers)
        if not self.workers:
            raise DistributedContractError("at least one distributed worker is required")
        worker_ids = [worker.worker_id for worker in self.workers]
        if len(set(worker_ids)) != len(worker_ids):
            raise DistributedContractError("distributed worker IDs must be unique")
        if not math.isfinite(float(poll_seconds)) or poll_seconds <= 0:
            raise DistributedContractError("poll_seconds must be finite and positive")
        if not isinstance(overlap_output, (bool, np.bool_)):
            raise DistributedContractError("overlap_output must be boolean")
        if max_pending_output_tiles is not None and (
            int(max_pending_output_tiles) != max_pending_output_tiles
            or max_pending_output_tiles <= 0
        ):
            raise DistributedContractError(
                "max_pending_output_tiles must be positive or None"
            )
        self.poll_seconds = float(poll_seconds)
        self.overlap_output = bool(overlap_output)
        self.max_pending_output_tiles = (
            None
            if max_pending_output_tiles is None
            else int(max_pending_output_tiles)
        )
        self._run_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active_token: Optional[CancellationToken] = None

    def cancel(self, reason: str = "cancelled by caller") -> None:
        """Request cancellation of the currently running call, if any."""

        with self._active_lock:
            token = self._active_token
        if token is not None:
            token.cancel(reason)

    @staticmethod
    def _validate_assignments(
        assignments: Sequence[TileAssignment],
    ) -> tuple[TileAssignment, ...]:
        assignments = tuple(assignments)
        if not assignments:
            return assignments
        run_id = assignments[0].run_id
        expected_start = assignments[0].global_photon_start
        for expected_tile, assignment in enumerate(assignments):
            if assignment.run_id != run_id:
                raise DistributedContractError("all assignments must share one run_id")
            if assignment.tile_index != expected_tile:
                raise DistributedContractError(
                    "assignments must contain contiguous tile indices in plan order"
                )
            if assignment.global_photon_start != expected_start:
                raise DistributedContractError(
                    "assignment global photon ranges must be contiguous"
                )
            expected_start = assignment.global_photon_stop
        return assignments

    @staticmethod
    def _validate_result(
        result: TileResult, assignment: TileAssignment, worker_id: str
    ) -> None:
        expected = (
            assignment.run_id,
            worker_id,
            assignment.tile_index,
            assignment.global_photon_start,
            assignment.global_photon_stop,
        )
        actual = (
            result.run_id,
            result.worker_id,
            result.tile_index,
            result.global_photon_start,
            result.global_photon_stop,
        )
        if actual != expected:
            raise DistributedContractError(
                f"worker result identity mismatch: expected {expected}, got {actual}"
            )
        if result.compact_hits is not None and result.compact_hits.count:
            ids = result.compact_hits.global_photon_ids
            if np.any(ids < assignment.global_photon_start) or np.any(
                ids >= assignment.global_photon_stop
            ):
                raise DistributedContractError(
                    "worker returned compact hits outside its assigned global-ID range"
                )
            if not set(map(int, result.compact_hits.event_indices)).issubset(
                assignment.event_indices
            ):
                raise DistributedContractError(
                    "worker returned compact hits for an unassigned event"
                )
        if result.daq_partial is not None and result.daq_partial.count:
            if not set(map(int, result.daq_partial.event_indices)).issubset(
                assignment.event_indices
            ):
                raise DistributedContractError(
                    "worker returned DAQ rows for an unassigned event"
                )

    @staticmethod
    def _cancel_workers(
        states: Sequence[_WorkerState], run_id: str, reason: str
    ) -> None:
        for state in states:
            assignments = tuple(state.inflight.values())
            for future in tuple(state.inflight):
                future.cancel()
            try:
                state.spec.client.cancel(
                    run_id,
                    tuple(assignment.tile_index for assignment in assignments),
                    reason,
                )
            except BaseException:
                # Cancellation is best-effort and must not hide the original
                # worker failure or caller cancellation.
                pass

    def run(
        self,
        assignments: Sequence[TileAssignment],
        *,
        cancellation: Optional[CancellationToken] = None,
        result_consumer: Optional[TileResultConsumer] = None,
    ) -> DistributedRunResult:
        """Execute all assignments, accepting completion in arbitrary order.

        The method may be invoked from a background thread and interrupted by
        :meth:`cancel`.  A separate optional token lets a larger simulation
        cancel several coordinators together.  On any worker error, all queued
        futures and every worker endpoint receive a cancellation notification.

        By default compact-hit normalization and sparse DAQ reduction run on a
        dedicated host thread while later tiles execute.  ``result_consumer``
        is invoked on that same serialized thread as each validated tile is
        processed, enabling overlapped file/network output.  Consumer calls
        occur in worker-completion order; every envelope carries a stable tile
        and global-ID range, while the returned ``tile_results`` tuple and
        merged hits always retain plan/global-photon order.
        """

        assignments = self._validate_assignments(assignments)
        if result_consumer is not None and not callable(result_consumer):
            raise DistributedContractError("result_consumer must be callable or None")
        run_id = assignments[0].run_id if assignments else "empty"
        if cancellation is None:
            cancellation = CancellationToken()
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("this coordinator is already executing a run")
        internal_token = CancellationToken()
        with self._active_lock:
            self._active_token = internal_token

        states = [
            _WorkerState(spec=worker, ordinal=index)
            for index, worker in enumerate(self.workers)
        ]
        pending = deque(assignments)
        active: dict[Future[TileResult], _WorkerState] = {}
        max_pending_outputs = self.max_pending_output_tiles
        if max_pending_outputs is None:
            # Enough slack for every worker pipeline to finish one generation
            # without stalling dispatch, but still bounded for a slow sink.
            max_pending_outputs = max(2, sum(worker.max_inflight for worker in self.workers))
        output_pipeline = _OutputPipeline(
            overlap=self.overlap_output,
            max_pending_tiles=max_pending_outputs,
            consumer=result_consumer,
        )
        try:
            while pending or active:
                if cancellation.cancelled or internal_token.cancelled:
                    reason = (
                        cancellation.reason
                        if cancellation.cancelled
                        else internal_token.reason
                    )
                    raise DistributedCancelled(reason)

                while pending:
                    selected_offset: Optional[int] = None
                    selected_assignment: Optional[TileAssignment] = None
                    selected_state: Optional[_WorkerState] = None
                    # Usually offset zero wins.  Scanning later tiles avoids
                    # head-of-line blocking when heterogeneous workers have
                    # different memory limits and the only GPU able to accept
                    # the first tile is temporarily full.
                    for offset, candidate in enumerate(pending):
                        eligible = [
                            state
                            for state in states
                            if state.spec.has_queue_capacity(
                                candidate,
                                inflight=len(state.inflight),
                                inflight_photons=state.outstanding_photons,
                            )
                        ]
                        if eligible:
                            selected_offset = offset
                            selected_assignment = candidate
                            selected_state = min(
                                eligible,
                                key=lambda item: item.projected_score(candidate),
                            )
                            break
                    if selected_assignment is None or selected_state is None:
                        if active:
                            break
                        assignment = pending[0]
                        raise DistributedContractError(
                            f"no worker can accept tile {assignment.tile_index} "
                            f"with {assignment.photon_count} photons"
                        )
                    assignment = selected_assignment
                    state = selected_state
                    assert selected_offset is not None
                    pending.rotate(-selected_offset)
                    pending.popleft()
                    pending.rotate(selected_offset)
                    try:
                        future = state.spec.client.submit(assignment)
                    except BaseException as exc:
                        raise DistributedExecutionError(
                            state.spec.worker_id, assignment.tile_index, exc
                        ) from exc
                    if not isinstance(future, Future):
                        raise DistributedContractError(
                            f"worker {state.spec.worker_id!r} submit() did not return "
                            "a concurrent.futures.Future"
                        )
                    if future in active:
                        raise DistributedContractError(
                            f"worker {state.spec.worker_id!r} reused an active Future"
                        )
                    state.inflight[future] = assignment
                    active[future] = state
                    state.outstanding_photons += assignment.photon_count
                    state.submitted_tiles += 1
                    state.submitted_photons += assignment.photon_count
                    state.max_observed_inflight = max(
                        state.max_observed_inflight, len(state.inflight)
                    )
                    state.max_observed_inflight_photons = max(
                        state.max_observed_inflight_photons,
                        state.outstanding_photons,
                    )

                # Reducer/I/O backpressure is deliberately checked only after
                # every currently available worker slot has been refilled.
                output_pipeline.apply_backpressure()

                if not active:
                    continue
                done, _ = wait(
                    tuple(active),
                    timeout=self.poll_seconds,
                    return_when=FIRST_COMPLETED,
                )
                for future in sorted(
                    done, key=lambda item: active[item].inflight[item].tile_index
                ):
                    state = active.pop(future)
                    assignment = state.inflight.pop(future)
                    state.outstanding_photons -= assignment.photon_count
                    try:
                        result = future.result()
                    except BaseException as exc:
                        raise DistributedExecutionError(
                            state.spec.worker_id, assignment.tile_index, exc
                        ) from exc
                    try:
                        self._validate_result(result, assignment, state.spec.worker_id)
                    except BaseException as exc:
                        raise DistributedExecutionError(
                            state.spec.worker_id, assignment.tile_index, exc
                        ) from exc
                    output_pipeline.submit(result)
                    state.completed_tiles += 1
                    state.completed_photons += assignment.photon_count

            ordered_results, compact_hits, daq_partial = output_pipeline.finish(
                len(assignments)
            )
            worker_stats = {
                state.spec.worker_id: WorkerRunStats(
                    worker_id=state.spec.worker_id,
                    capacity_weight=float(state.spec.capacity_weight),
                    submitted_tiles=state.submitted_tiles,
                    completed_tiles=state.completed_tiles,
                    submitted_photons=state.submitted_photons,
                    completed_photons=state.completed_photons,
                    max_observed_inflight=state.max_observed_inflight,
                    max_observed_inflight_photons=(
                        state.max_observed_inflight_photons
                    ),
                )
                for state in states
            }
            return DistributedRunResult(
                run_id=run_id,
                tile_results=ordered_results,
                compact_hits=compact_hits,
                daq_partial=daq_partial,
                worker_stats=worker_stats,
            )
        except BaseException as exc:
            output_pipeline.abort()
            self._cancel_workers(states, run_id, str(exc))
            raise
        finally:
            with self._active_lock:
                self._active_token = None
            self._run_lock.release()


__all__ = [
    "CancellationToken",
    "CompactHits",
    "DaqPartial",
    "DistributedCancelled",
    "DistributedContractError",
    "DistributedCoordinator",
    "DistributedExecutionError",
    "DistributedOutputError",
    "DistributedRunResult",
    "DistributedScalingEstimate",
    "TileAssignment",
    "TileResult",
    "TileResultConsumer",
    "WorkerClient",
    "WorkerRunStats",
    "WorkerSpec",
    "assignments_from_plan",
    "estimate_distributed_scaling",
    "merge_compact_hits",
    "merge_daq_partials",
]
