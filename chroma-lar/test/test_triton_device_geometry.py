"""Focused parity tests for GPU-resident boundary queue geometry."""

from types import SimpleNamespace

import numpy as np
import pytest


def _synthetic_pmt_scene():
    vertices = np.asarray(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    translations = np.asarray(
        [[0.0, 0.0, 3.0], [0.0, 0.0, 6.0]], dtype=np.float32
    )
    inverse_rotations = np.transpose(rotations, (0, 2, 1)).copy()
    inverse_translations = -np.einsum(
        "nij,nj->ni", inverse_rotations, translations
    ).astype(np.float32)
    bounds_min = np.empty((2, 3), dtype=np.float32)
    bounds_max = np.empty((2, 3), dtype=np.float32)
    for index, translation in enumerate(translations):
        world = vertices + translation
        bounds_min[index] = world.min(axis=0)
        bounds_max[index] = world.max(axis=0)
    return SimpleNamespace(
        pmt=SimpleNamespace(vertices=vertices, triangles=triangles),
        instances=SimpleNamespace(
            channel_id=np.asarray([7, 42], dtype=np.int32),
            world_to_object_rotation=inverse_rotations,
            world_to_object_translation=inverse_translations,
            object_to_world_rotation=rotations,
            object_to_world_translation=translations,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        ),
    )


def _fill_boundary_sentinel(result):
    import torch

    for field in result.__dataclass_fields__:
        value = getattr(result, field)
        if value.dtype == torch.bool:
            value.fill_(True)
        elif value.dtype.is_floating_point:
            value.fill_(12345.0)
        else:
            value.fill_(77)


@pytest.mark.skipif(
    pytest.importorskip("torch").cuda.is_available() is False,
    reason="CUDA is required",
)
def test_device_count_geometry_matches_synchronized_prefix_and_guards_suffix():
    import torch

    pytest.importorskip("triton")
    from chroma.triton.transport import DeviceQueue
    from chroma_lar.triton_scene.compiler import compile_reflect3wires_scene
    from chroma_lar.triton_scene.device_geometry import (
        DeviceBoundaryMergeWorkspace,
        DeviceBoundaryRayWorkspace,
        gather_boundary_rays_device_count,
        merge_boundaries_device_count,
        nextafter_positive_inf_device_count,
    )
    from chroma_lar.triton_scene.instances import (
        build_pmt_instance_accelerator,
        nearest_pmt_hit,
        nearest_pmt_hit_tlas_device_count,
    )
    from chroma_lar.triton_scene.intersect import (
        allocate_split_intersection_workspace,
        intersect_scene_triton_device_count,
        intersect_scene_triton_split,
        prepare_scene_triton,
    )

    device = torch.device("cuda")
    state_capacity = 128
    input_capacity = 64
    launch_capacity = 96
    live = 37
    rng = np.random.default_rng(19331)
    positions_cpu = np.column_stack(
        (
            rng.uniform(-2000.0, -20.0, state_capacity),
            rng.uniform(-2000.0, 2000.0, state_capacity),
            rng.uniform(-2000.0, 2000.0, state_capacity),
        )
    ).astype(np.float32)
    directions_cpu = rng.normal(size=(state_capacity, 3)).astype(np.float32)
    directions_cpu /= np.linalg.norm(directions_cpu, axis=1, keepdims=True)
    positions = torch.from_numpy(positions_cpu).to(device)
    directions = torch.from_numpy(directions_cpu).to(device)
    last_instances = torch.full(
        (state_capacity,), -1, dtype=torch.int32, device=device
    )
    last_triangles = torch.full_like(last_instances, -1)
    queue = DeviceQueue.allocate(128, device=device)
    selected = torch.randperm(state_capacity, device=device)[:input_capacity].to(
        torch.int32
    )
    queue.buffer[:input_capacity].copy_(selected)
    queue.count.fill_(live)

    ray_storage = DeviceBoundaryRayWorkspace.allocate(128, device)
    ray_storage.origins.fill_(12345.0)
    ray_storage.directions.fill_(12345.0)
    ray_storage.last_instances.fill_(77)
    ray_storage.last_triangles.fill_(77)
    ray_storage.pmt_tmax.fill_(12345.0)
    rays = gather_boundary_rays_device_count(
        positions,
        directions,
        last_instances,
        last_triangles,
        queue,
        input_capacity=input_capacity,
        launch_capacity=launch_capacity,
        out=ray_storage,
    )
    expected_ids = selected[:live].to(torch.int64)
    torch.testing.assert_close(
        rays.origins[:live], positions[expected_ids], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        rays.directions[:live], directions[expected_ids], rtol=0.0, atol=0.0
    )
    assert torch.all(rays.origins[live:] == 12345.0)
    assert torch.all(rays.last_instances[live:] == 77)
    assert torch.all(rays.pmt_tmax == 12345.0)

    scene = compile_reflect3wires_scene()
    prepared = prepare_scene_triton(scene, device)
    expected_analytic = intersect_scene_triton_split(
        prepared,
        rays.origins[:live],
        rays.directions[:live],
        last_instance=rays.last_instances[:live],
        last_triangle=rays.last_triangles[:live],
    )
    analytic_workspace = allocate_split_intersection_workspace(
        launch_capacity, device
    )
    analytic = analytic_workspace.outputs(launch_capacity)
    _fill_boundary_sentinel(analytic)
    actual_analytic = intersect_scene_triton_device_count(
        prepared,
        rays.origins,
        rays.directions,
        queue.count,
        last_instance=rays.last_instances,
        last_triangle=rays.last_triangles,
        workspace=analytic_workspace,
        out=analytic,
    )
    for field in actual_analytic.__dataclass_fields__:
        actual = getattr(actual_analytic, field)
        expected = getattr(expected_analytic, field)
        assert torch.equal(actual[:live], expected), field
        suffix = actual[live:]
        if actual.dtype == torch.bool:
            assert torch.all(suffix)
        elif actual.dtype.is_floating_point:
            assert torch.all(suffix == 12345.0)
        else:
            assert torch.all(suffix == 77)

    rays.pmt_tmax.fill_(-9876.0)
    nextafter_positive_inf_device_count(
        actual_analytic.distance,
        queue.count,
        launch_capacity=launch_capacity,
        out=rays.pmt_tmax,
    )
    assert torch.equal(
        rays.pmt_tmax[:live],
        torch.nextafter(
            actual_analytic.distance[:live],
            torch.tensor(float("inf"), dtype=torch.float32, device=device),
        ),
    )
    assert torch.all(rays.pmt_tmax[live:] == -9876.0)

    pmt_scene = _synthetic_pmt_scene()
    accelerator = build_pmt_instance_accelerator(pmt_scene, device=device)
    pmt_origins = torch.zeros(
        (launch_capacity, 3), dtype=torch.float32, device=device
    )
    pmt_directions = torch.zeros_like(pmt_origins)
    pmt_directions[:, 2] = 1.0
    # Interleave rays parallel to and outside the conservative PMT union.  The
    # device path must initialize these rows to misses without a TLAS walk.
    pmt_origins[1:live:3] = torch.tensor(
        [20.0, 20.0, 20.0], dtype=torch.float32, device=device
    )
    pmt_directions[1:live:3] = torch.tensor(
        [0.0, 1.0, 0.0], dtype=torch.float32, device=device
    )
    pmt_tmax = torch.full(
        (launch_capacity,), float("inf"), dtype=torch.float32, device=device
    )
    pmt_last_instance = torch.full(
        (launch_capacity,), -1, dtype=torch.int32, device=device
    )
    pmt_last_triangle = torch.full_like(pmt_last_instance, -1)
    expected_pmt = nearest_pmt_hit(
        accelerator,
        pmt_origins[:live],
        pmt_directions[:live],
        tmax=pmt_tmax[:live],
        last_instance=pmt_last_instance[:live],
        last_triangle=pmt_last_triangle[:live],
        use_tlas=True,
    )
    pmt_workspace = accelerator.allocate_workspace(
        launch_capacity, result_capacity=launch_capacity
    )
    pmt_out = pmt_workspace.outputs(launch_capacity)
    for field in pmt_out.__dataclass_fields__:
        value = getattr(pmt_out, field)
        value.fill_(123.0 if value.dtype.is_floating_point else 77)
    actual_pmt = nearest_pmt_hit_tlas_device_count(
        accelerator,
        pmt_origins,
        pmt_directions,
        queue.count,
        launch_capacity=launch_capacity,
        tmax=pmt_tmax,
        last_instance=pmt_last_instance,
        last_triangle=pmt_last_triangle,
        workspace=pmt_workspace,
        out=pmt_out,
    )
    for field in actual_pmt.__dataclass_fields__:
        actual = getattr(actual_pmt, field)
        expected = getattr(expected_pmt, field)
        assert torch.equal(actual[:live], expected), field
        suffix = actual[live:]
        if actual.dtype.is_floating_point:
            assert torch.all(suffix == 123.0)
        else:
            assert torch.all(suffix == 77)
    expected_union_rows = torch.tensor(
        [row for row in range(live) if row % 3 != 1],
        dtype=torch.int32,
        device=device,
    )
    assert pmt_workspace.device_candidate_count.item() == len(
        expected_union_rows
    )
    # This population fits in one compaction block, making the documented
    # stable-within-block ordering directly observable.
    assert torch.equal(
        pmt_workspace.active_ray_ids[: len(expected_union_rows)],
        expected_union_rows,
    )

    # Feed real analytic records and synthetic PMT records to both the device
    # merge and its direct eager expression.  Geometry provenance is irrelevant
    # to the merge contract; field words and tie/material decisions are not.
    scene_device = {
        "pmt_scene_material1_index": torch.tensor(
            [1, 2], dtype=torch.int32, device=device
        ),
        "pmt_scene_material2_index": torch.tensor(
            [3, 4], dtype=torch.int32, device=device
        ),
        "pmt_scene_surface_index": torch.tensor(
            [5, 6], dtype=torch.int32, device=device
        ),
    }
    merge_workspace = DeviceBoundaryMergeWorkspace.allocate(128, device)
    for output in merge_workspace.outputs():
        output.fill_(321.0 if output.dtype.is_floating_point else 77)
    merged = merge_boundaries_device_count(
        scene_device,
        actual_analytic,
        actual_pmt,
        pmt_directions,
        queue.count,
        launch_capacity=launch_capacity,
        out=merge_workspace,
    )
    analytic_hit = actual_analytic.kind[:live] != 0
    pmt_hit = actual_pmt.instance_ids[:live] >= 0
    choose_analytic = analytic_hit & (
        ~pmt_hit
        | (
            actual_analytic.distance[:live].to(torch.float64) + 1.0e-12
            < actual_pmt.distances[:live].to(torch.float64)
        )
    )
    choose_pmt = pmt_hit & ~choose_analytic
    expected_distance = torch.where(
        choose_analytic,
        actual_analytic.distance[:live],
        actual_pmt.distances[:live],
    )
    assert torch.equal(merged[0][:live], expected_distance)
    expected_instance = torch.where(
        choose_pmt,
        actual_pmt.instance_ids[:live],
        torch.where(
            choose_analytic & (actual_analytic.kind[:live] == 1),
            -actual_analytic.index[:live] - 2,
            -1,
        ),
    )
    assert torch.equal(merged[5][:live], expected_instance)
    for output in merged:
        suffix = output[live:]
        if output.dtype.is_floating_point:
            assert torch.all(suffix == 321.0)
        else:
            assert torch.all(suffix == 77)

    # A zero live count is also a no-write operation even though every grid is
    # still launched at the fixed batch capacity.
    queue.count.zero_()
    rays.origins.fill_(11.0)
    gather_boundary_rays_device_count(
        positions,
        directions,
        last_instances,
        last_triangles,
        queue,
        input_capacity=input_capacity,
        launch_capacity=launch_capacity,
        out=ray_storage,
    )
    torch.cuda.synchronize()
    assert torch.all(rays.origins == 11.0)

    for field in pmt_out.__dataclass_fields__:
        value = getattr(pmt_out, field)
        value.fill_(456.0 if value.dtype.is_floating_point else 88)
    nearest_pmt_hit_tlas_device_count(
        accelerator,
        pmt_origins,
        pmt_directions,
        queue.count,
        launch_capacity=launch_capacity,
        tmax=pmt_tmax,
        last_instance=pmt_last_instance,
        last_triangle=pmt_last_triangle,
        workspace=pmt_workspace,
        out=pmt_out,
    )
    torch.cuda.synchronize()
    assert pmt_workspace.device_candidate_count.item() == 0
    for field in pmt_out.__dataclass_fields__:
        value = getattr(pmt_out, field)
        if value.dtype.is_floating_point:
            assert torch.all(value == 456.0)
        else:
            assert torch.all(value == 88)
