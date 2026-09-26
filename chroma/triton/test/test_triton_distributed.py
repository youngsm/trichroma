from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from chroma.triton.distributed import (
    CompactHits,
    DaqPartial,
    DistributedCancelled,
    DistributedCoordinator,
    DistributedExecutionError,
    DistributedOutputError,
    TileAssignment,
    TileResult,
    WorkerSpec,
    assignments_from_plan,
    estimate_distributed_scaling,
    merge_compact_hits,
    merge_daq_partials,
)
from chroma.triton.runtime import EventBatchPlan


def _photons(count):
    return SimpleNamespace(
        pos=np.zeros((count, 3), dtype=np.float32),
        dir=np.tile(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), (count, 1)),
        pol=np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
        wavelengths=np.full(count, 128.0, dtype=np.float32),
        __len__=lambda self: count,
    )


class _Photons:
    def __init__(self, count):
        source = _photons(count)
        self.pos = source.pos
        self.dir = source.dir
        self.pol = source.pol
        self.wavelengths = source.wavelengths

    def __len__(self):
        return len(self.pos)


def _assignments(count, *, run_id="run"):
    return tuple(
        TileAssignment(
            run_id=run_id,
            tile_index=index,
            photon_count=1,
            global_photon_start=index,
            global_photon_stop=index + 1,
            event_indices=(0,),
            payload=f"input-{index}",
        )
        for index in range(count)
    )


def _sized_assignments(counts, *, run_id="sized"):
    assignments = []
    photon_start = 0
    for index, count in enumerate(counts):
        assignments.append(
            TileAssignment(
                run_id=run_id,
                tile_index=index,
                photon_count=count,
                global_photon_start=photon_start,
                global_photon_stop=photon_start + count,
                event_indices=(0,),
                payload=f"input-{index}",
            )
        )
        photon_start += count
    return tuple(assignments)


class _ThreadWorker:
    """Non-blocking fake for an IPC client backed by asynchronous threads."""

    def __init__(self, worker_id, delay, *, fail_tile=None, daq_time=None):
        self._worker_id = worker_id
        self.delay = delay
        self.fail_tile = fail_tile
        self.daq_time = daq_time or (lambda tile: float(tile))
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.submitted_event = threading.Event()
        self.trace = []
        self.pending = 0
        self.max_pending = 0
        self.cancel_calls = []

    @property
    def worker_id(self):
        return self._worker_id

    def submit(self, assignment):
        with self.lock:
            self.pending += 1
            self.max_pending = max(self.max_pending, self.pending)
            self.trace.append(("submit", assignment.tile_index, time.monotonic()))
            self.submitted_event.set()
        future = self.executor.submit(self._execute, assignment)

        def finish(_future):
            with self.lock:
                self.pending -= 1

        future.add_done_callback(finish)
        return future

    def _execute(self, assignment):
        delay = self.delay(assignment.tile_index)
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            if self.cancel_event.wait(min(0.005, max(0.0, deadline - time.monotonic()))):
                raise RuntimeError("fake worker cancelled")
        if assignment.tile_index == self.fail_tile:
            raise ValueError("injected worker failure")
        event = assignment.event_indices[0]
        hit = CompactHits(
            [assignment.global_photon_start],
            [event],
            [assignment.tile_index % 3],
            [100.0 - assignment.tile_index],
            [1 << (assignment.tile_index % 4)],
        )
        daq = DaqPartial(
            [event],
            [0],
            [self.daq_time(assignment.tile_index)],
            [1],
            [1 << (assignment.tile_index % 4)],
        )
        with self.lock:
            self.trace.append(("complete", assignment.tile_index, time.monotonic()))
        return TileResult(
            run_id=assignment.run_id,
            worker_id=self.worker_id,
            tile_index=assignment.tile_index,
            global_photon_start=assignment.global_photon_start,
            global_photon_stop=assignment.global_photon_stop,
            compact_hits=hit,
            daq_partial=daq,
            elapsed_seconds=delay,
        )

    def cancel(self, run_id, tile_indices, reason):
        with self.lock:
            self.cancel_calls.append((run_id, tile_indices, reason))
        self.cancel_event.set()

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def test_plan_assignments_keep_global_ids_independent_of_worker_count():
    plan = EventBatchPlan.build([_Photons(4), _Photons(5)], tile_capacity=3)
    first = assignments_from_plan(
        plan, run_id="stable", payload_factory=lambda index: f"shared-{index}"
    )
    # There is deliberately no worker-count argument: worker topology only
    # affects dispatch, never task identity, event fragments, or RNG IDs.
    second = assignments_from_plan(
        plan, run_id="stable", payload_factory=lambda index: f"shared-{index}"
    )
    assert first == second
    assert [item.tile_index for item in first] == [0, 1, 2]
    assert [
        (item.global_photon_start, item.global_photon_stop) for item in first
    ] == [(0, 3), (3, 6), (6, 9)]
    assert first[1].event_indices == (0, 1)


def test_compact_hits_restore_global_order_and_daq_partials_form_a_monoid():
    hits = merge_compact_hits(
        [
            CompactHits([9, 7], [1, 1], [3, 2], [9.0, 7.0], [1, 2]),
            CompactHits([2], [0], [1], [2.0], [4]),
        ]
    )
    np.testing.assert_array_equal(hits.global_photon_ids, [2, 7, 9])
    np.testing.assert_array_equal(hits.channels, [1, 2, 3])

    left = DaqPartial(
        [1, 0], [2, 4], [8.0, 5.0], [3, 7], [0b0010, 0b0100]
    )
    right = DaqPartial(
        [0, 1], [4, 2], [3.0, 9.0], [11, 13], [0b1000, 0b0001]
    )
    merged = merge_daq_partials([left, right])
    np.testing.assert_array_equal(merged.event_indices, [0, 1])
    np.testing.assert_array_equal(merged.channels, [4, 2])
    np.testing.assert_array_equal(merged.times, [3.0, 8.0])
    np.testing.assert_array_equal(merged.charges, [18, 16])
    np.testing.assert_array_equal(merged.histories, [0b1100, 0b0011])
    # Grouping is associative and order-independent; this permits tree/RDMA
    # reductions instead of gathering photon states on one rank.
    reverse = merge_daq_partials([right, left])
    np.testing.assert_array_equal(reverse.times, merged.times)
    np.testing.assert_array_equal(reverse.charges, merged.charges)
    np.testing.assert_array_equal(reverse.histories, merged.histories)


def test_threaded_workers_overlap_with_backpressure_and_restore_tile_order():
    fast = _ThreadWorker(
        "gpu-fast", lambda tile: 0.10 if tile == 0 else 0.02
    )
    slow = _ThreadWorker("gpu-slow", lambda tile: 0.30)
    try:
        coordinator = DistributedCoordinator(
            [
                WorkerSpec(fast, capacity_weight=4.0, max_inflight=2),
                WorkerSpec(slow, capacity_weight=1.0, max_inflight=2),
            ],
            poll_seconds=0.002,
        )
        result = coordinator.run(_assignments(12))
    finally:
        fast.close()
        slow.close()

    # Submission filled both double-buffered worker queues before the first
    # result completed: communication can be in flight while earlier work runs.
    trace = fast.trace + slow.trace
    first_completion = min(when for action, _, when in trace if action == "complete")
    assert sum(
        action == "submit" and when < first_completion
        for action, _, when in trace
    ) == 4
    assert fast.max_pending == 2
    assert slow.max_pending == 2
    assert result.worker_stats["gpu-fast"].max_observed_inflight == 2
    assert result.worker_stats["gpu-slow"].max_observed_inflight == 2

    # Tile 1 finishes ahead of deliberately delayed tile 0, but the public
    # result and compact hits recover plan/global-photon order.
    fast_completion_order = [
        tile for action, tile, _ in fast.trace if action == "complete"
    ]
    assert fast_completion_order.index(1) < fast_completion_order.index(0)
    assert [item.tile_index for item in result.tile_results] == list(range(12))
    np.testing.assert_array_equal(result.compact_hits.global_photon_ids, np.arange(12))

    # Capacity weighting plus completion-driven refill lets the faster worker
    # claim more tiles; this is not a static 6/6 split.
    assert result.worker_stats["gpu-fast"].completed_tiles > 6
    assert result.worker_stats["gpu-slow"].completed_tiles < 6
    assert result.daq_partial.count == 1
    assert result.daq_partial.charges[0] == 12
    assert result.daq_partial.times[0] == 0.0
    assert result.daq_partial.histories[0] == 0b1111


def test_worker_failure_cancels_every_endpoint_and_preserves_context():
    bad = _ThreadWorker("gpu-bad", lambda tile: 0.01, fail_tile=0)
    other = _ThreadWorker("gpu-other", lambda tile: 0.20)
    try:
        coordinator = DistributedCoordinator(
            [WorkerSpec(bad, max_inflight=2), WorkerSpec(other, max_inflight=2)],
            poll_seconds=0.002,
        )
        with pytest.raises(DistributedExecutionError) as raised:
            coordinator.run(_assignments(8, run_id="failure"))
        assert raised.value.worker_id == "gpu-bad"
        assert raised.value.tile_index == 0
        assert isinstance(raised.value.cause, ValueError)
        assert bad.cancel_calls
        assert other.cancel_calls
        assert all(call[0] == "failure" for call in bad.cancel_calls + other.cancel_calls)
    finally:
        bad.close()
        other.close()


def test_external_coordinator_cancellation_reaches_worker_process_protocol():
    worker = _ThreadWorker("gpu-0", lambda tile: 0.40)
    coordinator = DistributedCoordinator(
        [WorkerSpec(worker, max_inflight=2)], poll_seconds=0.002
    )
    caller = ThreadPoolExecutor(max_workers=1)
    try:
        running = caller.submit(coordinator.run, _assignments(4, run_id="cancel"))
        assert worker.submitted_event.wait(timeout=1.0)
        coordinator.cancel("simulation generator closed")
        with pytest.raises(DistributedCancelled, match="generator closed"):
            running.result(timeout=1.0)
        assert worker.cancel_calls
        assert worker.cancel_calls[0][0] == "cancel"
        assert "generator closed" in worker.cancel_calls[0][2]

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with worker.lock:
                if worker.pending == 0:
                    break
            time.sleep(0.002)
        assert worker.pending == 0
        worker.cancel_event.clear()
        recovered = coordinator.run(_assignments(1, run_id="after-cancel"))
        assert [item.tile_index for item in recovered.tile_results] == [0]
    finally:
        worker.close()
        caller.shutdown(wait=True, cancel_futures=True)


def test_output_consumer_overlaps_worker_refill_and_final_order_is_stable():
    worker = _ThreadWorker(
        "gpu-0", lambda tile: 0.01 if tile == 0 else 0.15
    )
    consumer_started = threading.Event()
    release_consumer = threading.Event()
    consumed = []

    def consume(result):
        consumed.append(result.tile_index)
        if result.tile_index == 0:
            consumer_started.set()
            assert release_consumer.wait(timeout=2.0)

    coordinator = DistributedCoordinator(
        [WorkerSpec(worker, max_inflight=2)],
        poll_seconds=0.001,
        overlap_output=True,
        max_pending_output_tiles=2,
    )
    caller = ThreadPoolExecutor(max_workers=1)
    try:
        running = caller.submit(
            coordinator.run,
            _assignments(4, run_id="overlap"),
            result_consumer=consume,
        )
        assert consumer_started.wait(timeout=1.0)
        # Tile 0's host consumer is deliberately blocked.  The coordinator
        # must nevertheless refill the newly free worker slot with tile 2,
        # keeping H2D/compute/D2H work ahead of output I/O.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with worker.lock:
                submitted = [
                    tile for action, tile, _ in worker.trace if action == "submit"
                ]
            if 2 in submitted:
                break
            time.sleep(0.002)
        assert 2 in submitted
        release_consumer.set()
        result = running.result(timeout=3.0)
    finally:
        release_consumer.set()
        worker.close()
        caller.shutdown(wait=True, cancel_futures=True)

    assert sorted(consumed) == [0, 1, 2, 3]
    assert [item.tile_index for item in result.tile_results] == [0, 1, 2, 3]
    np.testing.assert_array_equal(result.compact_hits.global_photon_ids, np.arange(4))


def test_photon_backpressure_bounds_nonuniform_tiles():
    worker = _ThreadWorker("gpu-0", lambda tile: 0.01)
    try:
        result = DistributedCoordinator(
            [
                WorkerSpec(
                    worker,
                    max_inflight=8,
                    max_tile_photons=4,
                    max_inflight_photons=4,
                )
            ],
            poll_seconds=0.001,
        ).run(_sized_assignments([3, 3, 1]))
    finally:
        worker.close()

    stats = result.worker_stats["gpu-0"]
    assert stats.max_observed_inflight_photons <= 4
    assert stats.max_observed_inflight == 2


def test_heterogeneous_workers_skip_temporarily_blocked_large_tile():
    large = _ThreadWorker("gpu-large", lambda tile: 0.15)
    small = _ThreadWorker("gpu-small", lambda tile: 0.01)
    try:
        result = DistributedCoordinator(
            [
                WorkerSpec(large, max_inflight=1, max_tile_photons=8),
                WorkerSpec(small, max_inflight=1, max_tile_photons=2),
            ],
            poll_seconds=0.001,
        ).run(_sized_assignments([5, 5, 1], run_id="heterogeneous"))
    finally:
        large.close()
        small.close()

    tile_zero_complete = next(
        when
        for action, tile, when in large.trace
        if action == "complete" and tile == 0
    )
    tile_two_submit = next(
        when
        for action, tile, when in small.trace
        if action == "submit" and tile == 2
    )
    assert tile_two_submit < tile_zero_complete
    assert [item.tile_index for item in result.tile_results] == [0, 1, 2]
    np.testing.assert_array_equal(result.compact_hits.global_photon_ids, [0, 5, 10])


def test_output_failure_is_typed_cancels_workers_and_releases_run_lock():
    worker = _ThreadWorker("gpu-0", lambda tile: 0.005)
    coordinator = DistributedCoordinator(
        [WorkerSpec(worker, max_inflight=1)],
        poll_seconds=0.001,
        max_pending_output_tiles=1,
    )

    def fail_output(result):
        raise OSError(f"sink rejected tile {result.tile_index}")

    try:
        with pytest.raises(DistributedOutputError) as raised:
            coordinator.run(
                _assignments(1, run_id="bad-output"),
                result_consumer=fail_output,
            )
        assert raised.value.tile_index == 0
        assert isinstance(raised.value.cause, OSError)
        assert worker.cancel_calls[-1][0] == "bad-output"

        # Failure cleanup must release both the output executor and the
        # coordinator run lock.  This second run uses the same endpoint and
        # would fail immediately if either resource leaked.
        worker.cancel_event.clear()
        recovered = coordinator.run(_assignments(1, run_id="recovered"))
        assert [item.tile_index for item in recovered.tile_results] == [0]
    finally:
        worker.close()


def test_async_daq_reduction_preserves_plan_order_signed_zero_tie():
    slow_first = _ThreadWorker(
        "gpu-first", lambda tile: 0.08, daq_time=lambda tile: 0.0
    )
    fast_second = _ThreadWorker(
        "gpu-second", lambda tile: 0.005, daq_time=lambda tile: -0.0
    )
    try:
        result = DistributedCoordinator(
            [
                WorkerSpec(slow_first, max_inflight=1),
                WorkerSpec(fast_second, max_inflight=1),
            ],
            poll_seconds=0.001,
        ).run(_assignments(2, run_id="signed-zero"))
    finally:
        slow_first.close()
        fast_second.close()

    # Tile 1 finishes first, but historical ordered reduction sees tile 0's
    # +0.0 first.  The asynchronous reducer uses tile order to preserve that
    # exact float32 sign bit while charge/history remain commutative.
    assert not np.signbit(result.daq_partial.times[0])
    assert result.daq_partial.charges[0] == 2


def test_distributed_scaling_model_separates_aggregate_and_single_gpu_rates():
    estimate = estimate_distributed_scaling(26.555e6, worker_count=2)
    assert estimate.worker_count == 2
    assert estimate.ideal_aggregate_photons_per_second == pytest.approx(53.11e6)
    assert estimate.estimated_aggregate_photons_per_second == pytest.approx(53.11e6)
    assert estimate.required_parallel_efficiency(50e6) == pytest.approx(
        50.0 / 53.11
    )
    assert estimate.reaches(50e6)
    # This is an aggregate two-worker result; no field upgrades the measured
    # 26.555M/s transport rate of either individual GPU to 50M/s.
    assert estimate.worker_transport_rates == (26.555e6, 26.555e6)


def test_distributed_scaling_model_applies_overlapped_and_shared_bottlenecks():
    estimate = estimate_distributed_scaling(
        [30e6, 20e6],
        worker_h2d_rates=[100e6, 18e6],
        worker_d2h_rates=80e6,
        load_balance_efficiency=0.9,
        shared_input_rate=100e6,
        shared_output_rate=40e6,
    )
    assert estimate.worker_pipeline_rates == (30e6, 18e6)
    assert estimate.worker_limited_photons_per_second == pytest.approx(43.2e6)
    assert estimate.estimated_aggregate_photons_per_second == 40e6
    assert estimate.parallel_efficiency == pytest.approx(0.8)
    assert estimate.limiting_stage == "shared_output"

    with pytest.raises(ValueError):
        estimate_distributed_scaling(10e6, worker_count=0)
