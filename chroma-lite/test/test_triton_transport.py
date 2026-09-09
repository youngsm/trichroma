import numpy as np


from chroma.triton.transport import (
    ABSORBED,
    BOUNDARY,
    BULK_ABSORB,
    CONTINUE,
    RAYLEIGH_SCATTER,
    certified_aabb_exit_distance,
    collision_first_epoch_reference,
)


def _orthogonal_polarizations(directions):
    helper = np.zeros_like(directions)
    helper[:, 2] = 1.0
    use_y = np.abs(directions[:, 2]) > 0.9
    helper[use_y] = (0.0, 1.0, 0.0)
    result = np.cross(directions, helper)
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def test_certified_exit_is_strictly_conservative():
    position = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [2.0, 0.0, 0.0]])
    direction = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = certified_aabb_exit_distance(position, direction, [-1]*3, [1]*3)
    assert 0.0 < result[0] < 1.0
    assert 0.0 < result[1] < 1.9
    assert result[2] == 0.0


def test_no_collision_advances_before_artificial_exit():
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    directions = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    polarizations = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    times = np.zeros(1, dtype=np.float32)
    histories = np.zeros(1, dtype=np.uint32)
    result = collision_first_epoch_reference(
        positions, directions, polarizations, times, histories,
        np.random.default_rng(4), [-1]*3, [1]*3,
        np.inf, np.inf, 1.23,
    )
    assert result.status[0] == BOUNDARY
    assert 0.98 < positions[0, 0] < 1.0
    assert times[0] > 0.0


def test_detector_scale_exit_guard_survives_float32_rounding():
    position = np.array([[0.0, 0.0, -2100.0]], dtype=np.float32)
    direction = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    distance = certified_aabb_exit_distance(
        position, direction, [-2300.0, -2160.0, -2160.0], [0.0, 2160.0, 2160.0]
    )
    advanced = np.asarray(position + distance[:, None] * direction, dtype=np.float32)
    assert advanced[0, 2] > np.float32(-2160.0)


def test_reference_preserves_rayleigh_frame_and_flags():
    nphotons = 20000
    positions = np.zeros((nphotons, 3), dtype=np.float32)
    directions = np.zeros_like(positions)
    directions[:, 0] = 1.0
    polarizations = _orthogonal_polarizations(directions).astype(np.float32)
    times = np.zeros(nphotons, dtype=np.float32)
    histories = np.zeros(nphotons, dtype=np.uint32)
    result = collision_first_epoch_reference(
        positions, directions, polarizations, times, histories,
        np.random.default_rng(7), [-1.0e6]*3, [1.0e6]*3,
        np.inf, 1.0, 1.23, max_scatter=1,
    )
    assert np.all(result.status == CONTINUE)
    assert np.all(result.scatter_count == 1)
    assert np.all(histories & RAYLEIGH_SCATTER)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(polarizations, axis=1), 1.0, atol=2e-6)
    np.testing.assert_allclose(np.sum(directions*polarizations, axis=1), 0.0, atol=2e-6)
    # E[cos^2] for 3/4(1-cos^2) is 1/5.
    cosine = directions[:, 1]  # initial polarization was -Y
    assert abs(np.mean(cosine*cosine) - 0.2) < 0.01


def test_competing_hazards_match_expected_process_fraction():
    nphotons = 50000
    positions = np.zeros((nphotons, 3), dtype=np.float32)
    directions = np.zeros_like(positions)
    directions[:, 0] = 1.0
    polarizations = _orthogonal_polarizations(directions).astype(np.float32)
    times = np.zeros(nphotons, dtype=np.float32)
    histories = np.zeros(nphotons, dtype=np.uint32)
    result = collision_first_epoch_reference(
        positions, directions, polarizations, times, histories,
        np.random.default_rng(12), [-1.0e6]*3, [1.0e6]*3,
        4.0, 1.0, 1.23, max_scatter=1,
    )
    absorbed = result.status == ABSORBED
    # rate_abs / (rate_abs + rate_scat) = .25 / 1.25 = .2
    assert abs(np.mean(absorbed) - 0.2) < 0.01
    assert np.all((histories[absorbed] & BULK_ABSORB) != 0)


def test_gpu_epoch_and_direct_block_queue_partition():
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")
    from chroma.triton.transport import collision_first_epoch

    nphotons = 131071
    device = "cuda"
    positions = torch.zeros((nphotons, 3), dtype=torch.float32, device=device)
    directions = torch.zeros_like(positions)
    directions[:, 0] = 1.0
    polarizations = torch.zeros_like(positions)
    polarizations[:, 1] = 1.0
    times = torch.zeros(nphotons, dtype=torch.float32, device=device)
    histories = torch.zeros(nphotons, dtype=torch.int32, device=device)
    counters = torch.zeros(nphotons, dtype=torch.int64, device=device)
    last_instances = torch.full(
        (nphotons,), -3, dtype=torch.int32, device=device
    )
    last_triangles = torch.full(
        (nphotons,), 5, dtype=torch.int32, device=device
    )
    queue = torch.arange(nphotons, dtype=torch.int32, device=device)
    result = collision_first_epoch(
        positions, directions, polarizations, times, histories, counters, queue,
        [-1000.0]*3, [1000.0]*3, np.inf, 1.0, 1.23,
        seed=18, max_scatter=3,
        last_instances=last_instances,
        last_triangles=last_triangles,
    )
    counts = sum(q.size() for q in (
        result.continuing, result.boundary, result.absorbed, result.invalid
    ))
    assert counts == nphotons
    assert result.continuing.size() == nphotons
    for target, output in (
        (CONTINUE, result.continuing),
        (BOUNDARY, result.boundary),
        (ABSORBED, result.absorbed),
        (3, result.invalid),
    ):
        expected = queue[result.status == target].sort().values
        actual = output.tensor().sort().values
        assert torch.equal(actual, expected)
    assert torch.all(result.scatter_count == 3)
    assert torch.all(last_instances == -1)
    assert torch.all(last_triangles == -1)
    assert torch.max(torch.abs(torch.linalg.vector_norm(directions, dim=1) - 1.0)) < 3e-5
    assert torch.max(torch.abs(torch.sum(directions*polarizations, dim=1))) < 3e-5


def test_gpu_active_producer_reuses_workspace_without_status_array():
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")
    from chroma.triton.transport import (
        CollisionQueueWorkspace,
        collision_first_epoch,
    )

    nphotons = 4097
    positions = torch.zeros((nphotons, 3), dtype=torch.float32, device="cuda")
    positions[::2, 0] = 2000.0
    directions = torch.zeros_like(positions)
    directions[:, 0] = 1.0
    polarizations = torch.zeros_like(positions)
    polarizations[:, 1] = 1.0
    times = torch.zeros(nphotons, dtype=torch.float32, device="cuda")
    histories = torch.zeros(nphotons, dtype=torch.int32, device="cuda")
    counters = torch.zeros(nphotons, dtype=torch.int64, device="cuda")
    queue = torch.arange(
        nphotons - 1, -1, -1, dtype=torch.int64, device="cuda"
    )
    workspace = CollisionQueueWorkspace.allocate(
        nphotons + 17, device="cuda", dtype=torch.int64
    )
    storage_pointer = workspace.storage.data_ptr()
    count_pointer = workspace.counts.data_ptr()

    result = collision_first_epoch(
        positions, directions, polarizations, times, histories, counters, queue,
        [-1000.0] * 3, [1000.0] * 3, np.inf, 1.0, 1.23,
        seed=31, max_scatter=2, partition="active",
        queue_workspace=workspace,
    )
    assert result.status is None
    assert result.absorbed is None
    assert result.invalid is None
    assert workspace.storage.data_ptr() == storage_pointer
    assert workspace.counts.data_ptr() == count_pointer
    expected_boundary = torch.arange(0, nphotons, 2, device="cuda")
    expected_continuing = torch.arange(1, nphotons, 2, device="cuda")
    first_boundary = result.boundary.tensor().clone().sort().values
    first_continuing = result.continuing.tensor().clone().sort().values
    assert torch.equal(first_boundary, expected_boundary)
    assert torch.equal(first_continuing, expected_continuing)

    # Reusing the workspace resets counts and overwrites its queue views, while
    # retaining the allocations themselves.
    first_result = result
    positions[:, 0] = 2000.0
    result = collision_first_epoch(
        positions, directions, polarizations, times, histories, counters, queue,
        [-1000.0] * 3, [1000.0] * 3, np.inf, 1.0, 1.23,
        seed=31, max_scatter=2, partition="active",
        queue_workspace=workspace,
    )
    assert result.boundary.size() == nphotons
    assert result.continuing.size() == 0
    assert first_result.boundary is result.boundary
    assert first_result.continuing is result.continuing
    assert first_result.continuing.size() == 0
    assert torch.equal(first_boundary, expected_boundary)
    assert torch.equal(first_continuing, expected_continuing)
    assert workspace.storage.data_ptr() == storage_pointer
    assert workspace.counts.data_ptr() == count_pointer


def test_gpu_full_direct_partition_covers_terminal_and_step_budget_statuses():
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")
    from chroma.triton.transport import INVALID, collision_first_epoch

    nphotons = 4099
    positions = torch.zeros((nphotons, 3), dtype=torch.float32, device="cuda")
    positions[:1000, 0] = 2000.0
    positions[1000, 0] = float("nan")
    directions = torch.zeros_like(positions)
    directions[:, 0] = 1.0
    polarizations = torch.zeros_like(positions)
    polarizations[:, 1] = 1.0
    times = torch.zeros(nphotons, dtype=torch.float32, device="cuda")
    histories = torch.zeros(nphotons, dtype=torch.int32, device="cuda")
    counters = torch.zeros(nphotons, dtype=torch.int64, device="cuda")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(6142)
    queue = torch.randperm(
        nphotons, generator=generator, dtype=torch.int64, device="cuda"
    )
    result = collision_first_epoch(
        positions, directions, polarizations, times, histories, counters, queue,
        [-1000.0] * 3, [1000.0] * 3, 1.0e-6, np.inf, 1.23,
        seed=401, max_scatter=1, partition=True,
    )
    for target, output in (
        (CONTINUE, result.continuing),
        (BOUNDARY, result.boundary),
        (ABSORBED, result.absorbed),
        (INVALID, result.invalid),
    ):
        expected = queue[result.status == target].sort().values
        actual = output.tensor().sort().values
        assert torch.equal(actual, expected)
    assert result.boundary.size() == 1000
    assert result.invalid.size() == 1
    assert result.absorbed.size() == nphotons - 1001
    assert result.continuing.size() == 0

    # Status 4 means an alive scatter exhausted the step budget.  It is
    # intentionally not part of any public queue, matching the former
    # status-compaction behavior.
    budget_count = 1024
    positions = torch.zeros(
        (budget_count, 3), dtype=torch.float32, device="cuda"
    )
    directions = torch.zeros_like(positions)
    directions[:, 0] = 1.0
    polarizations = torch.zeros_like(positions)
    polarizations[:, 1] = 1.0
    times = torch.zeros(budget_count, dtype=torch.float32, device="cuda")
    histories = torch.zeros(budget_count, dtype=torch.int32, device="cuda")
    counters = torch.zeros(budget_count, dtype=torch.int64, device="cuda")
    step_counts = torch.zeros(
        budget_count, dtype=torch.int32, device="cuda"
    )
    queue = torch.arange(budget_count, dtype=torch.int32, device="cuda")
    result = collision_first_epoch(
        positions, directions, polarizations, times, histories, counters, queue,
        [-1.0e6] * 3, [1.0e6] * 3, np.inf, 1.0, 1.23,
        seed=402, max_scatter=1, partition=True,
        step_counts=step_counts, max_steps=1,
    )
    assert torch.all(result.status == 4)
    assert sum(queue.size() for queue in (
        result.continuing, result.boundary, result.absorbed, result.invalid
    )) == 0


def test_gpu_device_count_ping_pong_and_persistent_boundary_accumulator():
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")
    from chroma.triton.transport import (
        CollisionQueueWorkspace,
        DeviceQueue,
        collision_first_epoch,
    )

    nphotons = 8192
    positions = torch.zeros((nphotons, 3), dtype=torch.float32, device="cuda")
    directions = torch.zeros_like(positions)
    directions[:, 0] = 1.0
    polarizations = torch.zeros_like(positions)
    polarizations[:, 1] = 1.0
    times = torch.zeros(nphotons, dtype=torch.float32, device="cuda")
    histories = torch.zeros(nphotons, dtype=torch.int32, device="cuda")
    counters = torch.zeros(nphotons, dtype=torch.int64, device="cuda")
    initial = torch.arange(nphotons, dtype=torch.int32, device="cuda")
    workspaces = [
        CollisionQueueWorkspace.allocate(nphotons, device="cuda")
        for _ in range(2)
    ]
    accumulated = DeviceQueue.allocate(nphotons, device="cuda")
    accumulated.reset()

    result = collision_first_epoch(
        positions, directions, polarizations, times, histories, counters, initial,
        [-10.0] * 3, [10.0] * 3, np.inf, 5.0, 1.23,
        seed=177, max_scatter=1, partition="active",
        queue_workspace=workspaces[0],
        boundary_accumulator=accumulated,
        append_boundary=True,
    )
    live = result.continuing
    for epoch in range(1, 12):
        output = workspaces[epoch & 1]
        result = collision_first_epoch(
            positions, directions, polarizations, times, histories, counters, live,
            [-10.0] * 3, [10.0] * 3, np.inf, 5.0, 1.23,
            seed=177, max_scatter=1, partition="active",
            queue_workspace=output,
            input_capacity=nphotons,
            boundary_accumulator=accumulated,
            append_boundary=True,
        )
        live = result.continuing

    # The loop above performs no host count reads.  One final read of both
    # device counters proves every local state index is in exactly one queue.
    boundary_ids = accumulated.tensor()
    continuing_ids = live.tensor()
    assert boundary_ids.numel() > 0
    assert boundary_ids.numel() + continuing_ids.numel() == nphotons
    combined = torch.cat((boundary_ids, continuing_ids)).sort().values
    assert torch.equal(combined, torch.arange(nphotons, device="cuda"))


def test_gpu_device_count_masks_stale_capacity_suffix():
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")
    from chroma.triton.transport import (
        CollisionQueueWorkspace,
        DeviceQueue,
        collision_first_epoch,
    )

    capacity = 1024
    live_count = 37
    positions = torch.zeros((capacity, 3), dtype=torch.float32, device="cuda")
    directions = torch.zeros_like(positions)
    directions[:, 0] = 1.0
    polarizations = torch.zeros_like(positions)
    polarizations[:, 1] = 1.0
    times = torch.zeros(capacity, dtype=torch.float32, device="cuda")
    histories = torch.zeros(capacity, dtype=torch.int32, device="cuda")
    counters = torch.zeros(capacity, dtype=torch.int64, device="cuda")
    # Every suffix lane contains a valid but deliberately repeated state ID.
    # Only the count-limited prefix is allowed to touch state.
    buffer = torch.zeros(capacity, dtype=torch.int32, device="cuda")
    buffer[:live_count] = torch.arange(live_count, device="cuda")
    buffer[live_count:] = live_count
    count = torch.tensor([live_count], dtype=torch.int32, device="cuda")
    input_queue = DeviceQueue(buffer, count)
    workspace = CollisionQueueWorkspace.allocate(capacity, device="cuda")

    result = collision_first_epoch(
        positions, directions, polarizations, times, histories, counters,
        input_queue, [-1.0e6] * 3, [1.0e6] * 3,
        np.inf, 1.0, 1.23, seed=91, max_scatter=1,
        partition="active", input_capacity=capacity,
        queue_workspace=workspace,
    )
    assert result.continuing.size() == live_count
    assert result.boundary.size() == 0
    assert counters[live_count].item() == 0
    assert times[live_count].item() == 0.0


def test_gpu_global_photon_id_map_controls_rng_identity():
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")
    from chroma.triton.transport import collision_first_epoch

    def make_state(count):
        positions = torch.zeros((count, 3), dtype=torch.float32, device="cuda")
        directions = torch.zeros_like(positions)
        directions[:, 0] = 1.0
        polarizations = torch.zeros_like(positions)
        polarizations[:, 1] = 1.0
        return (
            positions,
            directions,
            polarizations,
            torch.zeros(count, dtype=torch.float32, device="cuda"),
            torch.zeros(count, dtype=torch.int32, device="cuda"),
            torch.zeros(count, dtype=torch.int64, device="cuda"),
        )

    global_ids = torch.tensor([101, 1000003], dtype=torch.int64, device="cuda")
    combined = make_state(2)
    collision_first_epoch(
        *combined, torch.arange(2, dtype=torch.int32, device="cuda"),
        [-1.0e6] * 3, [1.0e6] * 3, np.inf, 1.0, 1.23,
        seed=717, photon_id_base=999999, max_scatter=1,
        partition=False, global_photon_ids=global_ids,
    )

    for lane, global_id in enumerate((101, 1000003)):
        single = make_state(1)
        collision_first_epoch(
            *single, torch.zeros(1, dtype=torch.int32, device="cuda"),
            [-1.0e6] * 3, [1.0e6] * 3, np.inf, 1.0, 1.23,
            seed=717, photon_id_base=global_id, max_scatter=1,
            partition=False,
        )
        assert torch.equal(combined[0][lane], single[0][0])
        assert torch.equal(combined[1][lane], single[1][0])
        assert torch.equal(combined[2][lane], single[2][0])
        assert combined[3][lane].item() == single[3][0].item()
        assert combined[5][lane].item() == single[5][0].item()


def test_gpu_random_tape_uses_legacy_bulk_draw_order_and_stable_rows():
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")
    from chroma.triton.rng_alignment import (
        CERTIFICATE_EMPTY_WORD,
        RandomTape,
        RandomTapeSpec,
        allocate_torch_audit,
        allocate_torch_certificate,
        pack_interaction_certificate,
        to_torch,
    )
    from chroma.triton.transport import collision_first_epoch

    # Tape rows deliberately differ from both local state order and queue
    # order.  Slots 0 and 1 are the independent absorption and scattering
    # exponentials from photon.h; a Rayleigh interaction then consumes 2 and 3.
    tape_ids = np.asarray([9003, 7001, 8002, 6004], dtype=np.int64)
    state_ids = np.asarray([7001, 8002, 9003, 6004], dtype=np.int64)
    row_for_state = np.asarray([1, 2, 0, 3], dtype=np.int32)
    spec = RandomTapeSpec(max_interactions=3, draws_per_interaction=4)
    values = np.full((4, 3, 4), np.float32(0.5), dtype=np.float32)
    values[1, 0, :2] = np.exp(np.asarray([-0.25, -5.0])).astype(np.float32)
    values[2, 0, :4] = np.asarray(
        [np.exp(-5.0), np.exp(-0.25), 0.25, 0.5], dtype=np.float32
    )
    values[0, 0, :2] = np.exp(np.asarray([-5.0, -6.0])).astype(np.float32)
    tape = RandomTape(spec=spec, global_photon_ids=tape_ids, values=values)
    device_tape = to_torch(tape)
    audit = allocate_torch_audit(4)
    certificate = allocate_torch_certificate(4, spec.max_interactions)

    positions = torch.zeros((4, 3), dtype=torch.float32, device="cuda")
    positions[3, 0] = 2.0
    initial_positions = positions.clone()
    directions = torch.zeros_like(positions)
    directions[:, 0] = 1.0
    polarizations = torch.zeros_like(positions)
    polarizations[:, 1] = 1.0
    times = torch.zeros(4, dtype=torch.float32, device="cuda")
    histories = torch.zeros(4, dtype=torch.int32, device="cuda")
    counters = torch.zeros(4, dtype=torch.int64, device="cuda")
    queue = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device="cuda")

    result = collision_first_epoch(
        positions,
        directions,
        polarizations,
        times,
        histories,
        counters,
        queue,
        [-1.0] * 3,
        [1.0] * 3,
        1.0,
        1.0,
        1.23,
        max_scatter=1,
        partition=True,
        global_photon_ids=torch.from_numpy(state_ids).to(device="cuda"),
        random_tape=device_tape,
        tape_audit=audit,
        tape_row_indices=torch.from_numpy(row_for_state).to(device="cuda"),
        tape_certificate=certificate,
    )

    assert torch.equal(
        result.absorbed.tensor().sort().values,
        torch.tensor([0], dtype=torch.int32, device="cuda"),
    )
    assert torch.equal(
        result.continuing.tensor().sort().values,
        torch.tensor([1], dtype=torch.int32, device="cuda"),
    )
    assert torch.equal(
        result.boundary.tensor().sort().values,
        torch.tensor([2, 3], dtype=torch.int32, device="cuda"),
    )
    assert result.invalid.size() == 0
    assert histories[0].item() & BULK_ABSORB
    assert histories[1].item() & RAYLEIGH_SCATTER
    assert torch.equal(positions[2], initial_positions[2])
    assert torch.equal(positions[3], initial_positions[3])
    assert times[2].item() == 0.0
    assert times[3].item() == 0.0
    assert torch.equal(counters, torch.zeros_like(counters))
    assert torch.equal(
        audit.interaction_cursor,
        torch.tensor([0, 1, 1, 0], dtype=torch.int32, device="cuda"),
    )
    assert torch.count_nonzero(audit.draw_cursor).item() == 0
    assert torch.count_nonzero(audit.overflow).item() == 0
    certificate_words = certificate.words.cpu().numpy().view(np.uint32)
    expected_certificate = np.full(
        (4, spec.max_interactions),
        CERTIFICATE_EMPTY_WORD,
        dtype=np.uint32,
    )
    expected_certificate[1, 0] = pack_interaction_certificate(1, 2)[0]
    expected_certificate[2, 0] = pack_interaction_certificate(2, 4)[0]
    np.testing.assert_array_equal(certificate_words, expected_certificate)


def test_gpu_random_tape_records_draw_exhaustion_and_id_mismatch():
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")
    from chroma.triton.rng_alignment import (
        DRAW_OVERFLOW,
        GLOBAL_ID_MISMATCH,
        INTERACTION_OVERFLOW,
        RandomTape,
        RandomTapeSpec,
        allocate_torch_audit,
        to_torch,
    )
    from chroma.triton.transport import INVALID, collision_first_epoch

    spec = RandomTapeSpec(max_interactions=1, draws_per_interaction=3)
    tape_ids = np.asarray([11, 22, 33], dtype=np.int64)
    values = np.full((3, 1, 3), np.float32(0.5), dtype=np.float32)
    # Row 0 selects scattering, which needs four draws and must fail closed.
    values[0, 0] = np.asarray(
        [np.exp(-5.0), np.exp(-0.25), 0.25], dtype=np.float32
    )
    # Row 1 selects terminal absorption using exactly the first two draws.
    values[1, 0, :2] = np.exp(np.asarray([-0.25, -5.0])).astype(np.float32)
    device_tape = to_torch(
        RandomTape(spec=spec, global_photon_ids=tape_ids, values=values)
    )
    audit = allocate_torch_audit(3)

    positions = torch.zeros((3, 3), dtype=torch.float32, device="cuda")
    initial_positions = positions.clone()
    directions = torch.zeros_like(positions)
    directions[:, 0] = 1.0
    polarizations = torch.zeros_like(positions)
    polarizations[:, 1] = 1.0
    times = torch.zeros(3, dtype=torch.float32, device="cuda")
    histories = torch.zeros(3, dtype=torch.int32, device="cuda")
    counters = torch.zeros(3, dtype=torch.int64, device="cuda")
    queue = torch.arange(3, dtype=torch.int32, device="cuda")
    requested_ids = torch.tensor([11, 22, 34], dtype=torch.int64, device="cuda")

    result = collision_first_epoch(
        positions,
        directions,
        polarizations,
        times,
        histories,
        counters,
        queue,
        [-1.0] * 3,
        [1.0] * 3,
        1.0,
        1.0,
        1.23,
        max_scatter=1,
        partition=True,
        global_photon_ids=requested_ids,
        random_tape=device_tape,
        tape_audit=audit,
    )

    assert torch.equal(
        result.invalid.tensor().sort().values,
        torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
    )
    assert torch.equal(
        result.absorbed.tensor().sort().values,
        torch.tensor([1], dtype=torch.int32, device="cuda"),
    )
    assert result.continuing.size() == 0
    assert result.boundary.size() == 0
    assert result.status.tolist() == [INVALID, ABSORBED, INVALID]
    assert torch.equal(positions[0], initial_positions[0])
    assert torch.equal(positions[2], initial_positions[2])
    assert audit.draw_cursor.tolist() == [3, 0, 0]
    assert audit.interaction_cursor.tolist() == [0, 1, 0]
    assert audit.overflow.tolist() == [
        int(DRAW_OVERFLOW),
        int(INTERACTION_OVERFLOW),
        int(GLOBAL_ID_MISMATCH),
    ]


def test_gpu_random_tape_certificate_records_every_bulk_commit():
    import pytest
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    pytest.importorskip("triton")
    from chroma.triton.rng_alignment import (
        CERTIFICATE_EMPTY_WORD,
        STATE_CERTIFICATE_EMPTY_WORD,
        STATE_CERTIFICATE_FIELD_INDEX,
        RandomTape,
        RandomTapeSpec,
        allocate_torch_audit,
        allocate_torch_certificate,
        allocate_torch_state_certificate,
        pack_interaction_certificate,
        to_torch,
        validate_state_certificate,
    )
    from chroma.triton.transport import collision_first_epoch

    spec = RandomTapeSpec(max_interactions=4, draws_per_interaction=4)
    values = np.full((1, 4, 4), np.float32(0.5), dtype=np.float32)
    tape = to_torch(RandomTape(
        spec=spec,
        global_photon_ids=np.asarray([0], dtype=np.int64),
        values=values,
    ))
    audit = allocate_torch_audit(1)
    certificate = allocate_torch_certificate(1, spec.max_interactions)
    state_certificate = allocate_torch_state_certificate(
        1, spec.max_interactions
    )
    positions = torch.zeros((1, 3), dtype=torch.float32, device="cuda")
    directions = torch.tensor(
        [[1.0, 0.0, 0.0]], dtype=torch.float32, device="cuda"
    )
    polarizations = torch.tensor(
        [[0.0, 1.0, 0.0]], dtype=torch.float32, device="cuda"
    )
    times = torch.zeros(1, dtype=torch.float32, device="cuda")
    histories = torch.zeros(1, dtype=torch.int32, device="cuda")
    counters = torch.zeros(1, dtype=torch.int64, device="cuda")
    last_instances = torch.full(
        (1,), 12, dtype=torch.int32, device="cuda"
    )
    last_triangles = torch.full(
        (1,), 34, dtype=torch.int32, device="cuda"
    )
    queue = torch.zeros(1, dtype=torch.int32, device="cuda")

    collision_first_epoch(
        positions,
        directions,
        polarizations,
        times,
        histories,
        counters,
        queue,
        [-1.0e6] * 3,
        [1.0e6] * 3,
        np.inf,
        1.0,
        1.23,
        max_scatter=3,
        partition=False,
        random_tape=tape,
        tape_audit=audit,
        tape_certificate=certificate,
        state_certificate=state_certificate,
        # Non-integral and signed-zero constants prove these fields are stored
        # by raw float32 reinterpretation rather than numeric integer casts.
        state_wavelength=450.25,
        state_weight=-0.0,
        state_evidx=np.uint32(0),
        last_instances=last_instances,
        last_triangles=last_triangles,
    )

    assert audit.interaction_cursor.tolist() == [3]
    word = pack_interaction_certificate(2, 4)[0]
    np.testing.assert_array_equal(
        certificate.words.cpu().numpy().view(np.uint32),
        np.asarray(
            [[word, word, word, CERTIFICATE_EMPTY_WORD]], dtype=np.uint32
        ),
    )
    state_words = state_certificate.words.cpu().numpy().view(np.uint32)
    validate_state_certificate(
        state_words,
        certificate.words.cpu().numpy().view(np.uint32),
        audit.interaction_cursor.cpu().numpy(),
    )
    assert np.all(np.any(
        state_words[0, :3]
        != np.asarray(STATE_CERTIFICATE_EMPTY_WORD, dtype=np.uint32),
        axis=1,
    ))
    assert np.all(
        state_words[0, 3]
        == np.asarray(STATE_CERTIFICATE_EMPTY_WORD, dtype=np.uint32)
    )

    field = STATE_CERTIFICATE_FIELD_INDEX
    float_word = lambda value: np.asarray(
        [value], dtype=np.float32
    ).view(np.uint32)[0]
    assert np.all(state_words[0, :3, field["wavelength"]] == float_word(450.25))
    assert np.all(state_words[0, :3, field["weight"]] == float_word(-0.0))
    assert np.all(state_words[0, :3, field["evidx"]] == np.uint32(0))
    assert np.all(state_words[0, :3, field["last_triangle"]] == np.uint32(0xFFFFFFFF))
    assert last_triangles.item() == -1

    # The final transcript cell must be the raw resident state after the third
    # scatter; equality here includes signed zero and every float32 rounding bit.
    final = state_words[0, 2]
    expected_float = np.concatenate((
        positions.cpu().numpy().reshape(-1),
        directions.cpu().numpy().reshape(-1),
        polarizations.cpu().numpy().reshape(-1),
        np.asarray([450.25], dtype=np.float32),
        times.cpu().numpy().reshape(-1),
    )).astype(np.float32, copy=False).view(np.uint32)
    np.testing.assert_array_equal(final[:11], expected_float)
    assert final[field["history"]] == histories.cpu().numpy().view(np.uint32)[0]
    assert final[field["last_triangle"]] == np.uint32(0xFFFFFFFF)
    assert final[field["weight"]] == float_word(-0.0)
    assert final[field["evidx"]] == np.uint32(0)
