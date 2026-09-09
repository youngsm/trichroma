import inspect

import numpy as np
import pytest

import chroma_lar.triton_scene.portals as portal_module
from chroma_lar.triton_scene.portals import (
    DEVICE_PORTAL_PROGRAMS_PER_SM,
    PortalDescriptor,
    PortalWorkspace,
    _persistent_portal_program_count,
    certify_reflect3wires_portals,
    classify_certified_box_portals_numpy,
    partition_certified_box_portals,
)


def _synthetic_descriptor(lower, upper):
    return PortalDescriptor(
        lower=np.asarray(lower, dtype=np.float32),
        upper=np.asarray(upper, dtype=np.float32),
        lar_material=7,
        active_outside_material=8,
        cathode_inside_material=9,
        active_surface=10,
        cathode_surface=11,
        active_box_index=1,
        cathode_box_index=2,
    )


@pytest.mark.parametrize(
    ("input_capacity", "launch_capacity"),
    [(None, None), (4, None), (None, 4)],
)
def test_device_queue_portal_route_requires_both_explicit_capacities(
        input_capacity, launch_capacity):
    torch = pytest.importorskip("torch")
    transport = pytest.importorskip("chroma.triton.transport")
    queue = transport.DeviceQueue(
        torch.arange(4, dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
    )
    vectors = torch.zeros((4, 3), dtype=torch.float32)
    previous = torch.full((4,), -1, dtype=torch.int32)

    with pytest.raises(ValueError, match="requires explicit input_capacity"):
        partition_certified_box_portals(
            vectors,
            vectors,
            queue,
            previous,
            previous,
            _synthetic_descriptor([-1.0] * 3, [1.0] * 3),
            input_capacity=input_capacity,
            launch_capacity=launch_capacity,
        )


def test_tensor_portal_route_rejects_device_grid_tuning():
    torch = pytest.importorskip("torch")
    vectors = torch.zeros((4, 3), dtype=torch.float32)
    previous = torch.full((4,), -1, dtype=torch.int32)

    with pytest.raises(ValueError, match="only valid for DeviceQueue"):
        partition_certified_box_portals(
            vectors,
            vectors,
            torch.arange(4, dtype=torch.int32),
            previous,
            previous,
            _synthetic_descriptor([-1.0] * 3, [1.0] * 3),
            programs_per_sm=8,
        )


def test_device_queue_portal_kernel_has_uniform_guard_and_no_host_count_read():
    signature = inspect.signature(partition_certified_box_portals)
    assert signature.parameters["input_capacity"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["launch_capacity"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["programs_per_sm"].kind is inspect.Parameter.KEYWORD_ONLY

    kernel_source = inspect.getsource(portal_module._load_portal_kernel)
    guard = kernel_source.index("if program_start >= live_items:")
    first_queue_load = kernel_source.index("photon_id = tl.load(input_queue")
    assert kernel_source.index("live_items = tl.load(nitems)") < guard
    assert guard < first_queue_load
    assert "NITEMS_IS_POINTER" in kernel_source
    persistent = kernel_source.index("def portal_partition_persistent_kernel")
    stride = kernel_source.index(
        "program_stride = tl.num_programs(0) * BLOCK", persistent
    )
    loop = kernel_source.index("while program_start < live_items:", persistent)
    direct_atomic = kernel_source.index(
        "direct_base = tl.atomic_add(counts, direct_n)", loop
    )
    fallback_atomic = kernel_source.index(
        "fallback_base = tl.atomic_add(counts + 1, fallback_n)", loop
    )
    assert persistent < stride < loop < direct_atomic < fallback_atomic

    route_source = inspect.getsource(partition_certified_box_portals)
    for forbidden in (".item(", ".size(", ".tensor(", ".cpu(", ".tolist("):
        assert forbidden not in route_source
    assert "_persistent_portal_program_count(" in route_source
    assert "program_count = triton.cdiv(launch_capacity, block_size)" in route_source


@pytest.mark.parametrize(
    ("capacity", "block_size", "sms", "programs_per_sm", "expected"),
    [
        (0, 128, 108, None, 0),
        (1, 128, 108, None, 1),
        (128, 128, 108, None, 1),
        (129, 128, 108, None, 2),
        (110_592, 128, 108, None, 864),
        (110_593, 128, 108, None, 865),
        (5_000_000, 128, 108, None, 39_063),
        (5_000_000, 128, 108, 0, 39_063),
        (5_000_000, 128, 108, 8, 864),
        (5_000_000, 128, 108, 4, 432),
    ],
)
def test_persistent_portal_grid_is_capacity_and_residency_bounded(
        capacity, block_size, sms, programs_per_sm, expected):
    assert _persistent_portal_program_count(
        capacity,
        block_size,
        sms,
        programs_per_sm=programs_per_sm,
    ) == expected
    assert DEVICE_PORTAL_PROGRAMS_PER_SM == 0


def test_persistent_portal_grid_stride_covers_each_slot_exactly_once():
    capacity = 10_003
    block_size = 128
    programs = _persistent_portal_program_count(
        capacity, block_size, 3, programs_per_sm=2
    )
    stride = programs * block_size
    slots = []
    for program in range(programs):
        start = program * block_size
        while start < capacity:
            slots.extend(
                slot
                for slot in range(start, start + block_size)
                if slot < capacity
            )
            start += stride
    assert sorted(slots) == list(range(capacity))
    assert len(slots) == len(set(slots))
    assert programs == 6


@pytest.mark.parametrize(
    ("capacity", "block_size", "sms", "programs_per_sm"),
    [
        (-1, 128, 1, None),
        (1, 0, 1, None),
        (1, 128, 0, None),
        (1, 128, 1, -1),
    ],
)
def test_persistent_portal_grid_rejects_invalid_topology(
        capacity, block_size, sms, programs_per_sm):
    with pytest.raises(ValueError):
        _persistent_portal_program_count(
            capacity,
            block_size,
            sms,
            programs_per_sm=programs_per_sm,
        )


def test_certified_box_portal_reference_classifies_five_real_faces():
    lower = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    upper = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    positions = np.zeros((6, 3), dtype=np.float32)
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    direct, distance, face = classify_certified_box_portals_numpy(
        positions, directions, lower, upper
    )

    np.testing.assert_array_equal(direct, [True, False, True, True, True, True])
    np.testing.assert_array_equal(face, [0, -1, 2, 3, 4, 5])
    np.testing.assert_array_equal(distance[direct], [1.0, 2.0, 2.0, 3.0, 3.0])
    assert np.isinf(distance[1])


def test_certified_box_portal_reference_is_conservative_at_lower_x_tie():
    lower = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    upper = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    positions = np.zeros((3, 3), dtype=np.float32)
    # Row 0 reaches lower X and upper Y simultaneously and must fall back.
    # Row 1 reaches Y first and is safe.  Row 2 starts outside the certificate.
    directions = np.array(
        [[-0.5, 1.0, 0.0], [-0.4, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    positions[2, 0] = 1.5

    direct, distance, face = classify_certified_box_portals_numpy(
        positions, directions, lower, upper
    )

    np.testing.assert_array_equal(direct, [False, True, False])
    np.testing.assert_array_equal(face, [-1, 3, -1])
    assert distance[1] == np.float32(2.0)
    assert np.isinf(distance[[0, 2]]).all()


def test_certified_box_portal_reference_rejects_nonfinite_and_nonforward():
    lower = [-1.0, -1.0, -1.0]
    upper = [1.0, 1.0, 1.0]
    positions = np.array(
        [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    directions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    direct, distance, face = classify_certified_box_portals_numpy(
        positions, directions, lower, upper
    )

    assert not direct.any()
    assert np.isinf(distance).all()
    np.testing.assert_array_equal(face, [-1, -1, -1])


def test_real_detector_portal_certificate_derives_metadata():
    from chroma_lar.triton_backend import _safe_empty_bounds
    from chroma_lar.triton_scene.compiler import compile_reflect3wires_scene
    from chroma_lar.triton_scene.instances import _padded_bounds

    scene = compile_reflect3wires_scene()
    lower, upper = _safe_empty_bounds(scene)
    padded_min, padded_max = _padded_bounds(
        scene.pmt.vertices,
        scene.instances.bounds_min,
        scene.instances.bounds_max,
    )
    descriptor = certify_reflect3wires_portals(
        scene,
        lower,
        upper,
        padded_pmt_bounds_max=padded_max,
    )

    assert descriptor.lar_material == scene.tables.material_names.index(
        "liquid_argon"
    )
    assert descriptor.active_box_index == scene.boxes.kinds.index("active")
    assert descriptor.cathode_box_index == scene.boxes.kinds.index("cathode")
    assert np.max(padded_max[:, 0]) < descriptor.lower[0]
    assert np.max(scene.wires.origin[:, 0] + scene.wires.radius) < descriptor.lower[0]
    assert padded_min.shape == padded_max.shape


def test_certified_box_portal_triton_matches_reference_and_hit_metadata():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    lower = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    upper = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    positions_np = np.zeros((6, 3), dtype=np.float32)
    directions_np = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    positions = torch.as_tensor(positions_np, device="cuda")
    directions = torch.as_tensor(directions_np, device="cuda")
    queue = torch.arange(6, dtype=torch.int32, device="cuda")
    last_instances = torch.full((6,), -1, dtype=torch.int32, device="cuda")
    last_triangles = torch.full((6,), -1, dtype=torch.int32, device="cuda")
    workspace = PortalWorkspace.allocate(6, positions.device)
    descriptor = PortalDescriptor(
        lower=lower,
        upper=upper,
        lar_material=7,
        active_outside_material=8,
        cathode_inside_material=9,
        active_surface=10,
        cathode_surface=11,
        active_box_index=1,
        cathode_box_index=2,
    )

    result = partition_certified_box_portals(
        positions,
        directions,
        queue,
        last_instances,
        last_triangles,
        descriptor,
        workspace=workspace,
        block_size=64,
    )
    direct_count, fallback_count = workspace.counts.cpu().tolist()
    assert (direct_count, fallback_count) == (5, 1)
    np.testing.assert_array_equal(
        result.direct.buffer[:direct_count].cpu().numpy(), [0, 2, 3, 4, 5]
    )
    np.testing.assert_array_equal(
        result.fallback.buffer[:fallback_count].cpu().numpy(), [1]
    )
    distance, normal, material_from, material_to, surface, instance, triangle, channel = (
        value[:direct_count].cpu().numpy() for value in result.hit
    )
    np.testing.assert_array_equal(distance, [1.0, 2.0, 2.0, 3.0, 3.0])
    np.testing.assert_array_equal(
        normal,
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
         [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    )
    np.testing.assert_array_equal(material_from, [7] * 5)
    np.testing.assert_array_equal(material_to, [9, 8, 8, 8, 8])
    np.testing.assert_array_equal(surface, [11, 10, 10, 10, 10])
    np.testing.assert_array_equal(instance, [-4, -3, -3, -3, -3])
    np.testing.assert_array_equal(triangle, [0, 2, 3, 4, 5])
    np.testing.assert_array_equal(channel, [-1] * 5)

    # Reuse clears both counters, and the same box/face identity follows the
    # analytic query's previous-hit suppression into the fallback queue.
    last_instances[0] = -4
    last_triangles[0] = 0
    reused = partition_certified_box_portals(
        positions,
        directions,
        queue,
        last_instances,
        last_triangles,
        descriptor,
        workspace=workspace,
        block_size=64,
    )
    direct_count, fallback_count = workspace.counts.cpu().tolist()
    assert (direct_count, fallback_count) == (4, 2)
    np.testing.assert_array_equal(
        reused.direct.buffer[:direct_count].cpu().numpy(), [2, 3, 4, 5]
    )
    np.testing.assert_array_equal(
        reused.fallback.buffer[:fallback_count].cpu().numpy(), [0, 1]
    )

    empty = partition_certified_box_portals(
        positions,
        directions,
        queue[:0],
        last_instances,
        last_triangles,
        descriptor,
        workspace=workspace,
        block_size=64,
    )
    assert empty.direct.size() == empty.fallback.size() == 0


def test_device_queue_portal_matches_tensor_and_ignores_stale_suffix():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from chroma.triton.transport import DeviceQueue

    lower = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    upper = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    directions_np = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    positions = torch.zeros((6, 3), dtype=torch.float32, device="cuda")
    directions = torch.as_tensor(directions_np, device="cuda")
    previous_instance = torch.full((6,), -1, dtype=torch.int32, device="cuda")
    previous_triangle = torch.full((6,), -1, dtype=torch.int32, device="cuda")
    descriptor = _synthetic_descriptor(lower, upper)

    reference_workspace = PortalWorkspace.allocate(6, positions.device)
    reference = partition_certified_box_portals(
        positions,
        directions,
        torch.arange(6, dtype=torch.int32, device="cuda"),
        previous_instance,
        previous_triangle,
        descriptor,
        workspace=reference_workspace,
        block_size=64,
    )
    reference_counts = reference_workspace.counts.cpu().tolist()

    # The live prefix occupies only six lanes.  Every otherwise-valid stale
    # suffix entry names photon 5, which would produce a direct Z+ hit if the
    # device count were ignored.  The launch also contains a wholly inactive
    # second CTA and is deliberately larger than the input safety cap.
    backing = torch.full((96,), 5, dtype=torch.int32, device="cuda")
    backing[:6] = torch.arange(6, dtype=torch.int32, device="cuda")
    queue = DeviceQueue(
        backing,
        torch.tensor([6], dtype=torch.int32, device="cuda"),
    )
    workspace = PortalWorkspace.allocate(128, positions.device)
    workspace.counts.fill_(91)
    actual = partition_certified_box_portals(
        positions,
        directions,
        queue,
        previous_instance,
        previous_triangle,
        descriptor,
        workspace=workspace,
        block_size=64,
        input_capacity=64,
        launch_capacity=128,
    )
    actual_counts = workspace.counts.cpu().tolist()

    assert actual_counts == reference_counts == [5, 1]
    assert actual.direct.capacity == actual.fallback.capacity == 128
    assert all(value.shape[0] == 128 for value in actual.hit)
    torch.testing.assert_close(
        actual.direct.buffer[:actual_counts[0]],
        reference.direct.buffer[:reference_counts[0]],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual.fallback.buffer[:actual_counts[1]],
        reference.fallback.buffer[:reference_counts[1]],
        rtol=0.0,
        atol=0.0,
    )
    for actual_value, reference_value in zip(actual.hit, reference.hit):
        torch.testing.assert_close(
            actual_value[:actual_counts[0]],
            reference_value[:reference_counts[0]],
            rtol=0.0,
            atol=0.0,
        )


def test_device_queue_portal_zero_count_preserves_stale_storage():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from chroma.triton.transport import DeviceQueue

    positions = torch.zeros((1, 3), dtype=torch.float32, device="cuda")
    directions = torch.tensor(
        [[1.0, 0.0, 0.0]], dtype=torch.float32, device="cuda"
    )
    previous = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    backing = torch.zeros(64, dtype=torch.int32, device="cuda")
    queue = DeviceQueue(
        backing,
        torch.zeros(1, dtype=torch.int32, device="cuda"),
    )
    workspace = PortalWorkspace.allocate(128, positions.device)
    workspace.queues.fill_(77)
    workspace.counts.fill_(83)
    for value in workspace.hit_outputs():
        value.fill_(19)
    queue_snapshot = workspace.queues.clone()
    hit_snapshot = tuple(value.clone() for value in workspace.hit_outputs())

    result = partition_certified_box_portals(
        positions,
        directions,
        queue,
        previous,
        previous,
        _synthetic_descriptor([-1.0] * 3, [1.0] * 3),
        workspace=workspace,
        block_size=64,
        input_capacity=64,
        launch_capacity=128,
    )

    torch.testing.assert_close(workspace.counts, torch.zeros_like(workspace.counts))
    torch.testing.assert_close(workspace.queues, queue_snapshot)
    for actual, expected in zip(workspace.hit_outputs(), hit_snapshot):
        torch.testing.assert_close(actual, expected)
    assert result.direct.capacity == result.fallback.capacity == 128


def test_device_queue_portal_rejects_input_output_aliases():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from chroma.triton.transport import DeviceQueue

    positions = torch.zeros((1, 3), dtype=torch.float32, device="cuda")
    directions = torch.tensor(
        [[1.0, 0.0, 0.0]], dtype=torch.float32, device="cuda"
    )
    previous = torch.full((1,), -1, dtype=torch.int32, device="cuda")
    descriptor = _synthetic_descriptor([-1.0] * 3, [1.0] * 3)
    workspace = PortalWorkspace.allocate(64, positions.device)

    with pytest.raises(ValueError, match="queue buffers must not alias"):
        partition_certified_box_portals(
            positions,
            directions,
            DeviceQueue(
                workspace.queues[0],
                torch.ones(1, dtype=torch.int32, device="cuda"),
            ),
            previous,
            previous,
            descriptor,
            workspace=workspace,
            input_capacity=64,
            launch_capacity=64,
        )
    independent_buffer = torch.zeros(64, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="queue counts must not alias"):
        partition_certified_box_portals(
            positions,
            directions,
            DeviceQueue(independent_buffer, workspace.counts[:1]),
            previous,
            previous,
            descriptor,
            workspace=workspace,
            input_capacity=64,
            launch_capacity=64,
        )
