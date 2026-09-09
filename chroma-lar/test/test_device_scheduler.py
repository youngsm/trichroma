"""Tests for the fixed-round device queue trace."""

import numpy as np
import pytest

from chroma_lar.triton_scene.device_scheduler import (
    DeviceRoundTrace,
    RoundTraceValidationError,
    _resolve_launch_capacity,
    _validate_round_index,
    parse_round_trace,
)


def test_parse_round_trace_returns_read_only_named_columns():
    records = np.asarray([[3, 17], [8, 9], [0, 0]], dtype=np.int32)
    trace = parse_round_trace(
        records,
        boundary_capacities=np.asarray([3, 10, 10]),
        pending_capacities=20,
    )
    assert trace.rounds == 3
    np.testing.assert_array_equal(trace.boundary_events, [3, 8, 0])
    np.testing.assert_array_equal(trace.pending_survivors, [17, 9, 0])
    assert trace.records.dtype == np.int32
    assert trace.records.flags.c_contiguous
    assert not trace.records.flags.writeable
    records[0, 0] = 1
    assert trace.boundary_events[0] == 3


@pytest.mark.parametrize(
    "records, boundary_capacity, pending_capacity, message",
    [
        (
            np.asarray([[0, -1]], dtype=np.int32),
            4,
            4,
            "round 0 has negative pending count -1",
        ),
        (
            np.asarray([[5, 1]], dtype=np.int32),
            4,
            4,
            "round 0 boundary count 5 exceeds queue capacity 4",
        ),
        (
            np.asarray([[1, 7]], dtype=np.int32),
            4,
            6,
            "round 0 pending count 7 exceeds queue capacity 6",
        ),
    ],
)
def test_parse_round_trace_detects_negative_and_overflow_counts(
    records, boundary_capacity, pending_capacity, message
):
    with pytest.raises(RoundTraceValidationError, match=message):
        parse_round_trace(records, boundary_capacity, pending_capacity)


def test_parse_round_trace_rejects_invalid_storage_and_capacities():
    with pytest.raises(TypeError, match="dtype int32"):
        parse_round_trace(np.zeros((2, 2), dtype=np.int64), 2, 2)
    with pytest.raises(ValueError, match=r"shape \(rounds, 2\)"):
        parse_round_trace(np.zeros(2, dtype=np.int32), 2, 2)
    with pytest.raises(ValueError, match="boundary_capacities"):
        parse_round_trace(
            np.zeros((2, 2), dtype=np.int32), np.asarray([1, 2, 3]), 2
        )
    with pytest.raises(ValueError, match="negative capacities"):
        parse_round_trace(np.zeros((1, 2), dtype=np.int32), -1, 2)


def test_round_indices_are_bounded_integral_and_strictly_sequential():
    assert _validate_round_index(np.int64(0), 3, 0) == 0
    assert _validate_round_index(2, 3, 2) == 2
    with pytest.raises(TypeError, match="integer"):
        _validate_round_index(0.0, 3, 0)
    with pytest.raises(TypeError, match="not bool"):
        _validate_round_index(True, 3, 0)
    with pytest.raises(IndexError, match="outside trace capacity"):
        _validate_round_index(3, 3, 3)
    with pytest.raises(ValueError, match="expected 1, received 0"):
        _validate_round_index(0, 3, 1)


def test_launch_capacity_validation_and_host_only_reset():
    assert _resolve_launch_capacity(None, 17, "cap") == 17
    assert _resolve_launch_capacity(9, 17, "cap") == 9
    with pytest.raises(TypeError, match="integer"):
        _resolve_launch_capacity(3.0, 17, "cap")
    with pytest.raises(ValueError, match="between 0 and queue storage capacity"):
        _resolve_launch_capacity(18, 17, "cap")

    records = np.asarray([[3, 9], [2, 4]], dtype=np.int32)
    trace = DeviceRoundTrace(
        records=records,
        _boundary_capacities=np.asarray([8, 8], dtype=np.int64),
        _pending_capacities=np.asarray([10, 10], dtype=np.int64),
        _rounds_recorded=2,
    )
    before = records.copy()
    assert trace.reset() is trace
    assert trace.rounds_recorded == 0
    np.testing.assert_array_equal(trace.records, before)
    np.testing.assert_array_equal(trace._boundary_capacities, [-1, -1])
    np.testing.assert_array_equal(trace._pending_capacities, [-1, -1])


def test_gpu_snapshots_are_ordered_and_materialized_once_in_a_batch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")
    from chroma.triton.transport import DeviceQueue

    boundary = DeviceQueue.allocate(11, device="cuda")
    pending = DeviceQueue.allocate(17, device="cuda", dtype=torch.int64)
    trace = DeviceRoundTrace.allocate(3, device=boundary.count.device)

    for round_index, (boundary_count, pending_count) in enumerate(
        ((3, 17), (11, 6), (0, 0))
    ):
        boundary.count.fill_(boundary_count)
        pending.count.fill_(pending_count)
        assert trace.snapshot(round_index, boundary, pending) is None
    assert trace.rounds_recorded == 3
    assert trace.records.is_cuda
    actual = trace.read()
    np.testing.assert_array_equal(
        actual.records,
        np.asarray([[3, 17], [11, 6], [0, 0]], dtype=np.int32),
    )

    overflow = DeviceRoundTrace.allocate(1, device=boundary.count.device)
    boundary.count.fill_(5)
    pending.count.zero_()
    overflow.snapshot(0, boundary, pending, boundary_capacity=4)
    with pytest.raises(RoundTraceValidationError, match="exceeds queue capacity"):
        overflow.read()

    overflow.reset()
    assert overflow.rounds_recorded == 0
    boundary.count.fill_(4)
    overflow.snapshot(0, boundary, pending, boundary_capacity=4)
    np.testing.assert_array_equal(overflow.read().boundary_events, [4])

    with pytest.raises(IndexError, match="outside trace capacity"):
        trace.snapshot(3, boundary, pending)
