import inspect

import numpy as np
import pytest

import chroma_lar.triton_scene.fused_portal_boundary as fused_module
from chroma_lar.triton_scene.fused_portal_boundary import (
    FusedPortalBoundaryWorkspace,
    _validate_capacities,
    step_fused_direct_portals,
)
from chroma_lar.triton_scene.portals import PortalDescriptor


def _descriptor():
    return PortalDescriptor(
        lower=np.asarray([-1.0, -2.0, -3.0], dtype=np.float32),
        upper=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        lar_material=0,
        active_outside_material=1,
        cathode_inside_material=2,
        active_surface=0,
        cathode_surface=1,
        active_box_index=1,
        cathode_box_index=2,
    )


@pytest.mark.parametrize(
    ("input_capacity", "launch_capacity", "input_storage", "fallback", "carry"),
    [
        (-1, 4, 4, 4, 4),
        (5, 5, 4, 5, 5),
        (4, 3, 4, 4, 4),
        (4, 5, 4, 4, 5),
        (4, 5, 4, 5, 4),
        (4, np.iinfo(np.int32).max + 1, 4, np.iinfo(np.int32).max + 1,
         np.iinfo(np.int32).max + 1),
    ],
)
def test_fused_portal_capacity_contract_rejects_truncation_or_overflow(
    input_capacity, launch_capacity, input_storage, fallback, carry
):
    with pytest.raises(ValueError):
        _validate_capacities(
            input_capacity, launch_capacity, input_storage, fallback, carry
        )


def test_fused_portal_capacity_contract_accepts_a_stale_suffix_grid():
    assert _validate_capacities(61, 128, 64, 128, 256) == (61, 128)
    with pytest.raises(TypeError, match="not bool"):
        _validate_capacities(True, 128, 128, 128, 128)


def test_fused_kernel_has_no_materialized_direct_queue_or_hit_record():
    kernel_source = inspect.getsource(
        fused_module._load_fused_portal_boundary_kernel
    )
    assert "direct_queue" not in kernel_source
    assert "out_distance" not in kernel_source
    assert "fallback = valid & ~direct" in kernel_source
    assert "fallback_buffer + fallback_base + fallback_local" in kernel_source
    assert "rng_counter += direct.to(tl.int64) * 2" in kernel_source
    assert "mask=direct" in kernel_source
    guard = kernel_source.index("if program_start >= live_items:")
    first_queue_load = kernel_source.index("boundary_buffer + lane")
    assert kernel_source.index("live_items = tl.load(boundary_count)") < guard
    assert guard < first_queue_load

    route_source = inspect.getsource(step_fused_direct_portals)
    for forbidden in (".item(", ".size(", ".tensor(", ".cpu(", ".tolist("):
        assert forbidden not in route_source
    assert "fallback_queue.reset()" in route_source
    assert "carry_queue.reset()" not in route_source


def _cuda_or_skip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    pytest.importorskip("triton")
    return torch


def _synthetic_state(torch, count, device):
    pattern = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    directions = pattern[torch.arange(count, device=device) % len(pattern)].clone()
    positions = torch.zeros((count, 3), dtype=torch.float32, device=device)
    polarizations = torch.zeros_like(positions)
    # One exactly perpendicular basis vector for every axis-aligned ray.
    x_ray = directions[:, 0] != 0.0
    y_ray = directions[:, 1] != 0.0
    polarizations[x_ray, 1] = 1.0
    polarizations[y_ray, 2] = 1.0
    polarizations[~x_ray & ~y_ray, 0] = 1.0
    times = torch.linspace(0.0, 1.0, count, dtype=torch.float32, device=device)
    histories = torch.arange(count, dtype=torch.int32, device=device) & 1
    rng_counters = (
        torch.arange(count, dtype=torch.int64, device=device) % 7
    ) * 2
    last_instances = torch.full(
        (count,), -1, dtype=torch.int32, device=device
    )
    last_triangles = torch.full_like(last_instances, -1)
    detected_channels = torch.full_like(last_instances, -17)
    step_counts = torch.arange(count, dtype=torch.int32, device=device) % 3
    # Exercise previous-hit suppression and an outside-certificate fallback.
    last_instances[0] = -4
    last_triangles[0] = 0
    positions[7, 1] = 3.0
    return (
        positions,
        directions,
        polarizations,
        times,
        histories,
        rng_counters,
        last_instances,
        last_triangles,
        detected_channels,
        step_counts,
    )


def _synthetic_tables(torch, device):
    # Mixed surfaces make the comparison cover absorb, detect, diffuse, and
    # specular selectors; finite bulk lengths also exercise collision-before-
    # portal behavior.  Both surfaces remain opaque (probabilities sum to 1).
    return {
        "tables_material_refractive_index": torch.tensor(
            [1.23, 1.0, 1.48], dtype=torch.float32, device=device
        ),
        "tables_material_absorption_length": torch.tensor(
            [9.0, 20.0, 20.0], dtype=torch.float32, device=device
        ),
        "tables_material_scattering_length": torch.tensor(
            [7.0, 20.0, 20.0], dtype=torch.float32, device=device
        ),
        "tables_surface_detect": torch.tensor(
            [0.20, 0.10], dtype=torch.float32, device=device
        ),
        "tables_surface_absorb": torch.tensor(
            [0.25, 0.35], dtype=torch.float32, device=device
        ),
        "tables_surface_reflect_diffuse": torch.tensor(
            [0.25, 0.15], dtype=torch.float32, device=device
        ),
        "tables_surface_reflect_specular": torch.tensor(
            [0.30, 0.40], dtype=torch.float32, device=device
        ),
    }


def test_fused_direct_portals_match_partition_then_monolithic_bitwise():
    torch = _cuda_or_skip()
    from chroma.triton.transport import DeviceQueue
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation
    from chroma_lar.triton_scene.portals import (
        PortalWorkspace,
        partition_certified_box_portals,
    )

    device = torch.device("cuda")
    photon_count = 512
    live_count = 333
    input_capacity = 384
    launch_capacity = 512
    descriptor = _descriptor()
    tables = _synthetic_tables(torch, device)
    initial = _synthetic_state(torch, photon_count, device)
    reference_state = tuple(value.clone() for value in initial)
    fused_state = tuple(value.clone() for value in initial)
    global_ids = (
        torch.arange(photon_count, dtype=torch.int64, device=device) * 17 + 101
    )

    # The live prefix is smaller than both host bounds; every stale suffix ID
    # would be valid work if either kernel ignored the device counter.
    boundary_buffer = torch.full(
        (input_capacity,), photon_count - 1, dtype=torch.int32, device=device
    )
    boundary_buffer[:live_count] = torch.arange(
        live_count, dtype=torch.int32, device=device
    )
    boundary_count = torch.tensor(
        [live_count], dtype=torch.int32, device=device
    )
    reference_boundary = DeviceQueue(boundary_buffer.clone(), boundary_count.clone())
    fused_boundary = DeviceQueue(boundary_buffer.clone(), boundary_count.clone())

    # Build the established two-kernel reference: portal materialization then
    # the corrected production monolithic boundary consumer.
    portal_workspace = PortalWorkspace.allocate(launch_capacity, device)
    partition = partition_certified_box_portals(
        reference_state[0],
        reference_state[1],
        reference_boundary,
        reference_state[6],
        reference_state[7],
        descriptor,
        workspace=portal_workspace,
        block_size=128,
        input_capacity=input_capacity,
        launch_capacity=launch_capacity,
    )
    reference_carry = DeviceQueue.allocate(launch_capacity, device=device)
    simulator = Reflect3WiresTritonSimulation.__new__(
        Reflect3WiresTritonSimulation
    )
    simulator.torch = torch
    simulator.scene_device = tables
    simulator.branch_specialized_boundary = False
    simulator.legacy_specular_reflection = False
    simulator.chroma_mesh_box_compatibility = False
    simulator.block_size = 128
    simulator._step_boundaries(
        reference_state,
        partition.direct,
        partition.hit,
        seed=919,
        photon_id_base=31,
        max_steps=11,
        global_photon_ids=global_ids,
        input_capacity=launch_capacity,
        survivor_queue=reference_carry,
        reset_survivors=False,
    )

    fused_carry = DeviceQueue.allocate(launch_capacity, device=device)
    fused_workspace = FusedPortalBoundaryWorkspace.allocate(
        launch_capacity, device
    )
    fallback = step_fused_direct_portals(
        fused_state,
        fused_boundary,
        fused_carry,
        descriptor,
        tables,
        workspace=fused_workspace,
        input_capacity=input_capacity,
        launch_capacity=launch_capacity,
        seed=919,
        photon_id_base=31,
        max_steps=11,
        global_photon_ids=global_ids,
        block_size=128,
    )

    for actual, expected in zip(fused_state, reference_state):
        assert torch.equal(actual, expected)
    assert torch.equal(
        torch.sort(fallback.tensor()).values,
        torch.sort(partition.fallback.tensor()).values,
    )
    assert torch.equal(
        torch.sort(fused_carry.tensor()).values,
        torch.sort(reference_carry.tensor()).values,
    )

    fallback_ids = fallback.tensor().to(torch.int64)
    for actual, expected in zip(fused_state, initial):
        assert torch.equal(actual[fallback_ids], expected[fallback_ids])


def test_fused_direct_portals_zero_count_and_alias_checks():
    torch = _cuda_or_skip()
    from chroma.triton.transport import DeviceQueue

    device = torch.device("cuda")
    state = _synthetic_state(torch, 64, device)
    state_snapshot = tuple(value.clone() for value in state)
    tables = _synthetic_tables(torch, device)
    boundary = DeviceQueue.allocate(64, device=device)
    carry = DeviceQueue.allocate(64, device=device)
    carry.count.fill_(1)
    carry.buffer[0] = 47
    workspace = FusedPortalBoundaryWorkspace.allocate(64, device)
    workspace.count.fill_(19)
    workspace.buffer.fill_(23)

    result = step_fused_direct_portals(
        state,
        boundary,
        carry,
        _descriptor(),
        tables,
        workspace=workspace,
        input_capacity=64,
        launch_capacity=64,
        seed=7,
        photon_id_base=0,
        max_steps=10,
    )
    assert result.size() == 0
    assert carry.size() == 1
    assert carry.buffer[0].item() == 47
    for actual, expected in zip(state, state_snapshot):
        assert torch.equal(actual, expected)

    alias_workspace = FusedPortalBoundaryWorkspace(
        boundary.buffer, torch.zeros_like(boundary.count)
    )
    with pytest.raises(ValueError, match="buffers must not alias"):
        step_fused_direct_portals(
            state,
            boundary,
            carry,
            _descriptor(),
            tables,
            workspace=alias_workspace,
            input_capacity=64,
            launch_capacity=64,
            seed=7,
            photon_id_base=0,
            max_steps=10,
        )


def test_fused_workspace_rejects_negative_capacity_without_cuda():
    with pytest.raises(ValueError, match="cannot be negative"):
        FusedPortalBoundaryWorkspace.allocate(-1, device="cpu")
