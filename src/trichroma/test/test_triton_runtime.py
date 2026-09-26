"""CPU-only contracts for backend-neutral Triton event orchestration."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from chroma.triton.executor import (
    AsyncPipelineExecutor,
    PipelineExecutionDriver,
    TorchCudaPipelineDriver,
)
from chroma.triton.runtime import (
    AsyncPipelinePlan,
    EventBatchPlan,
    EventResultAssembler,
    IncompleteResultError,
    MemoryBudgetError,
    MemoryFootprint,
    MemoryTilePlanner,
    OutputRequest,
    PhotonBatch,
    PipelineStageKind,
    PropagationBackend,
    PropagationResult,
    RuntimeContractError,
)


class DummyPhotons:
    def __init__(self, count, offset=0.0):
        values = np.arange(count, dtype=np.float32) + np.float32(offset)
        self.pos = np.stack((values, values + 1.0, values + 2.0), axis=1)
        self.dir = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (count, 1))
        self.pol = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (count, 1))
        self.wavelengths = np.full(count, 450.0, dtype=np.float32)
        self.t = values / np.float32(10.0)
        self.last_hit_triangles = np.full(count, -1, dtype=np.int32)
        self.flags = np.zeros(count, dtype=np.uint32)
        self.weights = np.ones(count, dtype=np.float32)
        # Deliberately wrong source event indices: the plan must replace these.
        self.evidx = np.full(count, 99, dtype=np.uint32)
        self.channel = np.zeros(count, dtype=np.uint32)

    def __len__(self):
        return len(self.pos)


def _events():
    return (
        SimpleNamespace(id=11, photons_beg=DummyPhotons(5, 0.0)),
        SimpleNamespace(id=12, photons_beg=DummyPhotons(2, 100.0)),
        SimpleNamespace(id=13, photons_beg=DummyPhotons(0, 200.0)),
    )


def _result_for_batch(batch, *, reverse=False):
    rows = np.arange(batch.photon_count)
    if reverse:
        rows = rows[::-1]
    fields = {
        "pos": batch.pos[rows],
        "dir": batch.dir[rows],
        "pol": batch.pol[rows],
        "wavelengths": batch.wavelengths[rows],
        "t": batch.t[rows] + np.float32(5.0),
        "last_hit_triangles": batch.last_hit_triangles[rows],
        "flags": batch.flags[rows],
        "weights": batch.weights[rows],
        "evidx": batch.evidx[rows],
    }
    ids = batch.global_photon_ids[rows]
    channels = np.where(ids % 2 == 0, ids % 7, -1).astype(np.int32)
    return PropagationResult(ids, fields, channels)


def test_event_plan_splits_events_without_losing_ids_or_boundaries():
    events = _events()
    plan = EventBatchPlan.build(iter(events), tile_capacity=3)

    assert plan.total_photons == 7
    assert [tile.photon_count for tile in plan.tiles] == [3, 3, 1]
    assert [event.event_id for event in plan.events] == [11, 12, 13]
    assert [(event.global_photon_start, event.global_photon_stop) for event in plan.events] == [
        (0, 5),
        (5, 7),
        (7, 7),
    ]

    batches = [plan.materialize_tile(index) for index in range(len(plan.tiles))]
    np.testing.assert_array_equal(
        np.concatenate([batch.global_photon_ids for batch in batches]),
        np.arange(7, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.concatenate([batch.evidx for batch in batches]),
        [0, 0, 0, 0, 0, 1, 1],
    )
    # The middle tile contains the tail of event 0 and head of event 1.
    np.testing.assert_array_equal(batches[1].evidx, [0, 0, 1])
    np.testing.assert_allclose(batches[1].pos[:, 0], [3.0, 4.0, 100.0])
    assert all(isinstance(batch, PhotonBatch) for batch in batches)
    assert not batches[0].pos.flags.writeable
    assert not batches[0].global_photon_ids.flags.writeable
    with pytest.raises(FrozenInstanceError):
        batches[0].times = np.empty(0)


def test_results_reassemble_in_event_and_photon_order_from_shuffled_tiles():
    plan = EventBatchPlan.build(_events(), tile_capacity=3)
    request = OutputRequest()
    assembler = EventResultAssembler(plan, request)

    # Complete tiles and rows intentionally arrive in reverse order.
    for tile_index in reversed(range(len(plan.tiles))):
        assembler.add(_result_for_batch(plan.materialize_tile(tile_index), reverse=True))
    assembled = assembler.finish()

    assert [result.planned_event.event_id for result in assembled] == [11, 12, 13]
    np.testing.assert_array_equal(assembled[0].propagation.global_photon_ids, np.arange(5))
    np.testing.assert_array_equal(assembled[1].propagation.global_photon_ids, [5, 6])
    assert assembled[2].propagation.photon_count == 0
    np.testing.assert_allclose(assembled[0].propagation.fields["pos"][:, 0], np.arange(5))
    np.testing.assert_allclose(assembled[1].propagation.fields["pos"][:, 0], [100.0, 101.0])
    np.testing.assert_array_equal(assembled[0].propagation.fields["evidx"], 0)
    np.testing.assert_array_equal(assembled[1].propagation.fields["evidx"], 1)


def test_zero_photon_events_reassemble_without_backend_work():
    events = (
        SimpleNamespace(id=20, photons_beg=DummyPhotons(0)),
        SimpleNamespace(id=21, photons_beg=DummyPhotons(0)),
    )
    plan = EventBatchPlan.build(events, tile_capacity=128)
    assert plan.total_photons == 0
    assert plan.tiles == ()
    assembled = EventResultAssembler(plan, OutputRequest()).finish()
    assert [result.planned_event.event_id for result in assembled] == [20, 21]
    assert all(result.propagation.photon_count == 0 for result in assembled)
    assert set(assembled[0].propagation.fields) == {
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


def test_assembler_rejects_duplicate_missing_and_wrong_event_rows():
    plan = EventBatchPlan.build(_events(), tile_capacity=3)
    request = OutputRequest()
    first = _result_for_batch(plan.materialize_tile(0))

    duplicate = EventResultAssembler(plan, request)
    duplicate.add(first)
    with pytest.raises(RuntimeContractError, match="more than once"):
        duplicate.add(first)

    incomplete = EventResultAssembler(plan, request)
    incomplete.add(first)
    with pytest.raises(IncompleteResultError, match="missing"):
        incomplete.finish()
    # A premature completeness probe is non-destructive; asynchronous callers
    # may continue adding later tile completions.
    for tile_index in (1, 2):
        incomplete.add(_result_for_batch(plan.materialize_tile(tile_index)))
    assert len(incomplete.finish()) == 3

    second_batch = plan.materialize_tile(1)
    wrong_fields = dict(_result_for_batch(second_batch).fields)
    wrong_fields["evidx"] = np.zeros(second_batch.photon_count, dtype=np.uint32)
    wrong = PropagationResult(
        second_batch.global_photon_ids,
        wrong_fields,
        np.full(second_batch.photon_count, -1, dtype=np.int32),
    )
    wrong_event = EventResultAssembler(plan, request)
    with pytest.raises(RuntimeContractError, match="evidx"):
        wrong_event.add(wrong)


def test_plan_accepts_single_photons_and_validates_fields_when_materialized():
    photons = DummyPhotons(4)
    plan = EventBatchPlan.build(photons, tile_capacity=2)
    assert len(plan.events) == 1
    assert not plan.events[0].was_event
    assert [tile.photon_count for tile in plan.tiles] == [2, 2]

    malformed = DummyPhotons(2)
    malformed.dir = np.zeros((3, 3), dtype=np.float32)
    bad_plan = EventBatchPlan.build(malformed, tile_capacity=2)
    with pytest.raises(RuntimeContractError, match="leading length"):
        bad_plan.materialize_tile(0)
    with pytest.raises(RuntimeContractError, match="tile_capacity"):
        EventBatchPlan.build(photons, tile_capacity=0)


def test_memory_planner_accounts_for_requested_outputs_and_alignment():
    footprint = MemoryFootprint(
        fixed_bytes=100,
        input_bytes_per_photon=10,
        state_bytes_per_photon=20,
        scratch_bytes_per_photon=30,
        terminal_output_bytes_per_photon=50,
        hit_output_bytes_per_photon=40,
        tracking_bytes_per_photon=500,
    )
    planner = MemoryTilePlanner(
        reserve_bytes=200, usable_fraction=0.5, alignment=16
    )
    hits = planner.plan(
        100,
        free_bytes=10_000,
        footprint=footprint,
        request=OutputRequest(),
    )
    assert hits.bytes_per_photon == 100
    assert hits.usable_bytes == 4700
    assert hits.tile_capacity == 32
    assert hits.tile_count == 4
    assert hits.estimated_peak_bytes == 3300

    end_only = planner.plan(
        100,
        free_bytes=10_000,
        footprint=footprint,
        request=OutputRequest(
            keep_photons_end=True, keep_hits=False, keep_flat_hits=False
        ),
    )
    assert end_only.bytes_per_photon == 110
    with pytest.raises(MemoryBudgetError, match="one photon"):
        MemoryTilePlanner(
            reserve_bytes=200, usable_fraction=1.0, alignment=1
        ).plan(
            1,
            free_bytes=350,
            footprint=footprint,
            request=OutputRequest(),
        )


def test_async_pipeline_describes_overlap_and_safe_buffer_reuse():
    plan = EventBatchPlan.build(_events(), tile_capacity=3)
    pipeline = AsyncPipelinePlan.build(
        plan,
        inflight_tiles=2,
        h2d_bytes_per_photon=44,
        d2h_bytes_per_photon=12,
    )
    assert len(pipeline.stages) == 3 * len(plan.tiles)
    assert [stage.kind for stage in pipeline.stages_for_tile(0)] == [
        PipelineStageKind.HOST_TO_DEVICE,
        PipelineStageKind.COMPUTE,
        PipelineStageKind.DEVICE_TO_HOST,
    ]
    h2d2 = next(stage for stage in pipeline.stages if stage.stage_id == "h2d:2")
    assert h2d2.buffer_slot == 0
    assert "h2d:1" in h2d2.depends_on
    assert "d2h:0" in h2d2.depends_on
    compute1 = next(stage for stage in pipeline.stages if stage.stage_id == "compute:1")
    assert compute1.depends_on == ("h2d:1", "compute:0")
    assert pipeline.stages_for_tile(0)[0].estimated_bytes == 3 * 44
    assert pipeline.stages_for_tile(2)[2].estimated_bytes == 1 * 12


def test_async_pipeline_omits_transfers_and_retargets_slot_dependencies():
    event_plan = EventBatchPlan.build(_events(), tile_capacity=2)
    no_host_result = OutputRequest(keep_hits=False, keep_flat_hits=False)

    device_only = AsyncPipelinePlan.build(
        event_plan,
        inflight_tiles=2,
        inputs_on_device=True,
        request=no_host_result,
    )
    assert device_only.inputs_on_device
    assert not device_only.copies_outputs_to_host
    assert {stage.kind for stage in device_only.stages} == {
        PipelineStageKind.COMPUTE
    }
    # Tile 2 reuses tile 0's slot and also follows tile 1 on the compute
    # stream.  Both dependencies remain explicit.
    compute2 = device_only.stages_for_tile(2)[0]
    assert compute2.depends_on == ("compute:1", "compute:0")

    upload_only = AsyncPipelinePlan.build(
        event_plan,
        inflight_tiles=2,
        request=no_host_result,
    )
    assert [stage.kind for stage in upload_only.stages_for_tile(0)] == [
        PipelineStageKind.HOST_TO_DEVICE,
        PipelineStageKind.COMPUTE,
    ]
    h2d2 = upload_only.stages_for_tile(2)[0]
    assert h2d2.depends_on == ("h2d:1", "compute:0")

    download_only = AsyncPipelinePlan.build(
        event_plan,
        inflight_tiles=2,
        inputs_on_device=True,
        request=OutputRequest(),
    )
    assert [stage.kind for stage in download_only.stages_for_tile(0)] == [
        PipelineStageKind.COMPUTE,
        PipelineStageKind.DEVICE_TO_HOST,
    ]
    compute2 = download_only.stages_for_tile(2)[0]
    assert compute2.depends_on == ("compute:1", "d2h:0")

    # Explicit policy overrides support device-resident downstream consumers.
    forced_device_output = AsyncPipelinePlan.build(
        event_plan,
        request=OutputRequest(),
        copy_outputs_to_host=False,
    )
    assert not forced_device_output.copies_outputs_to_host
    assert all(
        stage.kind is not PipelineStageKind.DEVICE_TO_HOST
        for stage in forced_device_output.stages
    )


class _FakePipelineDriver:
    def __init__(self):
        self.log = []

    def create_stream(self, name):
        self.log.append(("create_stream", name))
        return name

    def create_event(self, stage):
        event = "event:" + stage.stage_id
        self.log.append(("create_event", stage.stage_id))
        return event

    def wait_event(self, stream, event):
        self.log.append(("wait", stream, event))

    def invoke(self, stream, callback, stage):
        self.log.append(("invoke", stream, stage.stage_id))
        return callback(stage, stream)

    def record_event(self, event, stream):
        self.log.append(("record", stream, event))

    def synchronize_event(self, event):
        self.log.append(("synchronize", event))


def test_pipeline_executor_accepts_compute_only_plan_without_copy_callbacks():
    event_plan = EventBatchPlan.build(_events(), tile_capacity=2)
    pipeline = AsyncPipelinePlan.build(
        event_plan,
        inputs_on_device=True,
        request=OutputRequest(keep_hits=False, keep_flat_hits=False),
    )
    driver = _FakePipelineDriver()

    execution = AsyncPipelineExecutor(driver).execute(
        pipeline,
        compute=lambda stage, stream: (stage.tile_index, stream),
    )

    assert set(stage.kind for stage in pipeline.stages) == {
        PipelineStageKind.COMPUTE
    }
    assert execution.synchronized_stage_ids == tuple(
        f"compute:{tile.tile_index}" for tile in event_plan.tiles
    )


def test_pipeline_executor_submits_every_tile_before_terminal_synchronization():
    event_plan = EventBatchPlan.build(_events(), tile_capacity=2)
    pipeline = AsyncPipelinePlan.build(event_plan, inflight_tiles=3)
    driver = _FakePipelineDriver()
    assert isinstance(driver, PipelineExecutionDriver)
    callback_log = []

    def enqueue(stage, stream):
        callback_log.append((stage.stage_id, stream, stage.buffer_slot))
        return (stage.kind.value, stage.tile_index, stage.buffer_slot)

    execution = AsyncPipelineExecutor(driver).execute(
        pipeline, h2d=enqueue, compute=enqueue, d2h=enqueue
    )

    assert [entry[0] for entry in callback_log] == [
        stage.stage_id for stage in pipeline.stages
    ]
    assert execution.synchronized_stage_ids == tuple(
        stage.stage_id for stage in pipeline.terminal_stages
    )
    assert execution.value_for_tile(3, PipelineStageKind.COMPUTE) == (
        "compute",
        3,
        0,
    )
    with pytest.raises(TypeError):
        execution.stage_values["compute:0"] = None

    record_positions = [
        index for index, entry in enumerate(driver.log) if entry[0] == "record"
    ]
    synchronize_positions = [
        index
        for index, entry in enumerate(driver.log)
        if entry[0] == "synchronize"
    ]
    assert min(synchronize_positions) > max(record_positions)

    # The fourth tile reuses slot zero only after tile zero's D2H event.
    wait_for_reuse = ("wait", "h2d", "event:d2h:0")
    assert wait_for_reuse in driver.log
    assert driver.log.index(wait_for_reuse) < driver.log.index(
        ("invoke", "h2d", "h2d:3")
    )
    for stage in pipeline.stages:
        invoke_position = driver.log.index(
            ("invoke", stage.stream_name, stage.stage_id)
        )
        for dependency in stage.depends_on:
            wait = ("wait", stage.stream_name, "event:" + dependency)
            assert wait in driver.log
            assert driver.log.index(wait) < invoke_position


def test_pipeline_executor_validates_callbacks_before_submitting_work():
    event_plan = EventBatchPlan.build(_events(), tile_capacity=3)
    pipeline = AsyncPipelinePlan.build(event_plan)
    driver = _FakePipelineDriver()

    with pytest.raises(RuntimeContractError, match="h2d.*d2h|d2h.*h2d"):
        AsyncPipelineExecutor(driver).execute(
            pipeline, compute=lambda stage, stream: None
        )
    assert driver.log == []


def _torch_cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _torch_cuda_available(), reason="CUDA is not available")
def test_torch_cuda_driver_executes_real_triple_buffer_pipeline():
    import torch

    event_plan = EventBatchPlan.build(DummyPhotons(13), tile_capacity=4)
    pipeline = AsyncPipelinePlan.build(event_plan, inflight_tiles=3)
    driver = TorchCudaPipelineDriver()
    source = torch.empty(event_plan.total_photons, dtype=torch.float32, pin_memory=True)
    source.copy_(torch.arange(event_plan.total_photons, dtype=torch.float32))
    device_slots = [
        torch.empty(event_plan.tile_capacity, dtype=torch.float32, device=driver.device)
        for _ in range(pipeline.inflight_tiles)
    ]
    host_results = [
        torch.empty(tile.photon_count, dtype=torch.float32, pin_memory=True)
        for tile in event_plan.tiles
    ]

    def tile_range(stage):
        fragments = event_plan.tiles[stage.tile_index].fragments
        return (
            fragments[0].global_photon_start,
            fragments[-1].global_photon_stop,
        )

    def enqueue_h2d(stage, stream):
        del stream
        start, stop = tile_range(stage)
        return device_slots[stage.buffer_slot][: stage.photon_count].copy_(
            source[start:stop], non_blocking=True
        )

    def enqueue_compute(stage, stream):
        del stream
        values = device_slots[stage.buffer_slot][: stage.photon_count]
        return values.mul_(2.0).add_(1.0)

    def enqueue_d2h(stage, stream):
        del stream
        return host_results[stage.tile_index].copy_(
            device_slots[stage.buffer_slot][: stage.photon_count],
            non_blocking=True,
        )

    execution = AsyncPipelineExecutor(driver).execute(
        pipeline,
        h2d=enqueue_h2d,
        compute=enqueue_compute,
        d2h=enqueue_d2h,
    )

    observed = torch.cat(host_results).numpy()
    expected = np.arange(event_plan.total_photons, dtype=np.float32) * 2.0 + 1.0
    np.testing.assert_array_equal(observed, expected)
    assert execution.synchronized_stage_ids[-1] == "d2h:3"


def test_backend_protocol_is_structural_and_gpu_independent():
    class FakeBackend:
        def propagate(self, batch, request):
            return _result_for_batch(batch)

    assert isinstance(FakeBackend(), PropagationBackend)
