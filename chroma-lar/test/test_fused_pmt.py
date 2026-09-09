"""Focused contracts for the experimental fused PMT lattice traversal."""

import numpy as np
import pytest

from chroma_lar.triton_scene.fused_pmt import (
    FUSED_GRID_MAX_CANDIDATES,
    FusedPMTRoutingCounters,
    _ray_box_interval_cpu,
    nearest_pmt_hit_fused_grid,
    nearest_pmt_hit_fused_grid_device_count,
    plan_fused_pmt_candidates,
)
from chroma_lar.triton_scene.instances import (
    _infer_pmt_grid_locator,
    build_pmt_instance_accelerator,
    nearest_pmt_hit,
    triton_instance_available,
)


def _staggered_grid_bounds(side=9):
    centers = []
    for row in range(side):
        for column in range(side):
            centers.append(
                [
                    -2170.0,
                    -1900.0 + row * 471.0 + (column % 2) * 180.0,
                    -1900.0 + column * 471.0,
                ]
            )
    centers = np.asarray(centers, dtype=np.float32)
    half = np.asarray([63.0, 56.0, 56.0], dtype=np.float32)
    return centers - half, centers + half


def _fixture_lattice():
    lower, upper = _staggered_grid_bounds()
    locator = _infer_pmt_grid_locator(lower, upper)
    assert locator is not None
    union_lower = np.nextafter(
        lower.min(axis=0), np.float32(-np.inf)
    ).astype(np.float32)
    union_upper = np.nextafter(
        upper.max(axis=0), np.float32(np.inf)
    ).astype(np.float32)
    return locator, lower, upper, union_lower, union_upper


def _exact_box_hits(origin, direction, tmax, lower, upper):
    return np.asarray(
        [
            instance
            for instance in range(len(lower))
            if _ray_box_interval_cpu(
                origin, direction, lower[instance], upper[instance], tmax
            )[0]
        ],
        dtype=np.int32,
    )


def test_cpu_plan_is_conservative_and_keeps_ascending_tie_order():
    locator, lower, upper, union_lower, union_upper = _fixture_lattice()
    rng = np.random.default_rng(20260902)
    count = 512
    origins = np.empty((count, 3), dtype=np.float32)
    origins[:, 0] = rng.uniform(-500.0, 500.0, count)
    origins[:, 1:] = rng.uniform(-2500.0, 2500.0, (count, 2))
    targets = rng.uniform(
        union_lower.astype(np.float64),
        union_upper.astype(np.float64),
        (count, 3),
    ).astype(np.float32)
    directions = targets - origins
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    tmax = rng.uniform(1000.0, 10000.0, count).astype(np.float32)

    plan = plan_fused_pmt_candidates(
        locator,
        lower,
        upper,
        union_lower,
        union_upper,
        origins,
        directions,
        tmax=tmax,
    )
    assert plan.fallback.dtype == np.bool_
    assert plan.union_hit.dtype == np.bool_
    assert len(plan.instance_ids) == count
    for ray, selected in enumerate(plan.instance_ids):
        exact = _exact_box_hits(
            origins[ray], directions[ray], tmax[ray], lower, upper
        )
        assert set(exact.tolist()).issubset(set(selected.tolist()))
        assert np.all(selected[1:] > selected[:-1])
        if plan.fallback[ray]:
            np.testing.assert_array_equal(selected, np.arange(81))
        else:
            assert len(selected) <= FUSED_GRID_MAX_CANDIDATES


def test_cpu_plan_falls_back_only_when_the_lattice_interval_is_uncertifiable():
    locator, lower, upper, union_lower, union_upper = _fixture_lattice()
    center = (union_lower.astype(np.float64) + union_upper) * 0.5
    origins = np.asarray(
        [
            # Excessive origin magnitude violates the finite-coordinate proof.
            [1.0e6, center[1], center[2]],
            # A diagonal ray spans 9x9, but the default 81-box cutoff now
            # certifies it because this cannot cost more than fallback.
            [center[0], union_lower[1], union_lower[2]],
            # A plain union miss must not pay for the fallback.
            [0.0, union_upper[1] + 1000.0, union_upper[2] + 1000.0],
        ],
        dtype=np.float32,
    )
    directions = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [0.0, np.sqrt(0.5), np.sqrt(0.5)],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    tmax = np.asarray([2.0e6, 1.0e5, 1.0e5], dtype=np.float32)
    plan = plan_fused_pmt_candidates(
        locator,
        lower,
        upper,
        union_lower,
        union_upper,
        origins,
        directions,
        tmax=tmax,
    )
    np.testing.assert_array_equal(plan.union_hit, [True, True, False])
    np.testing.assert_array_equal(plan.fallback, [True, False, False])
    np.testing.assert_array_equal(plan.instance_ids[0], np.arange(81))
    np.testing.assert_array_equal(plan.instance_ids[1], np.arange(81))
    assert plan.instance_ids[2].size == 0


@pytest.mark.parametrize("cutoff", [1, 16, 81, 512])
def test_selected_visit_count_never_exceeds_exact_fallback(cutoff):
    """Certified rectangles are weakly dominated by the all-instance scan."""

    locator, lower, upper, union_lower, union_upper = _fixture_lattice()
    rng = np.random.default_rng(1701 + cutoff)
    count = 257
    origins = rng.uniform(-3000.0, 3000.0, (count, 3)).astype(np.float32)
    targets = rng.uniform(union_lower, union_upper, (count, 3)).astype(
        np.float32
    )
    directions = targets - origins
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    plan = plan_fused_pmt_candidates(
        locator,
        lower,
        upper,
        union_lower,
        union_upper,
        origins,
        directions,
        maximum_grid_candidates=cutoff,
    )
    fallback_visits = locator.rows * locator.columns
    assert len(plan.instance_ids) == count
    for selected, fallback in zip(plan.instance_ids, plan.fallback):
        assert len(selected) <= fallback_visits
        if fallback:
            assert len(selected) == fallback_visits
        else:
            assert len(selected) <= min(cutoff, fallback_visits)


def test_cpu_plan_validates_shape_and_limit_contracts():
    locator, lower, upper, union_lower, union_upper = _fixture_lattice()
    origins = np.zeros((2, 3), dtype=np.float32)
    directions = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="directions"):
        plan_fused_pmt_candidates(
            locator,
            lower,
            upper,
            union_lower,
            union_upper,
            origins,
            directions[:1],
        )
    with pytest.raises(ValueError, match="positive"):
        plan_fused_pmt_candidates(
            locator,
            lower,
            upper,
            union_lower,
            union_upper,
            origins,
            directions,
            maximum_grid_candidates=0,
        )


@pytest.mark.skipif(
    not triton_instance_available(require_cuda=True),
    reason="PyTorch, Triton, and a CUDA device are required",
)
def test_fused_grid_matches_exact_tlas_and_device_count_suffix_contract():
    import torch

    from chroma_lar.triton_scene.compiler import compile_reflect3wires_scene

    scene = compile_reflect3wires_scene()
    accelerator = build_pmt_instance_accelerator(scene)
    assert accelerator.grid_locator is not None
    rng = np.random.default_rng(54173)
    random_count = 2048
    instance = rng.integers(0, accelerator.instance_count, random_count)
    lower = accelerator.host_bounds_min[instance]
    upper = accelerator.host_bounds_max[instance]
    targets = rng.uniform(lower, upper).astype(np.float32)
    origins = np.zeros((random_count, 3), dtype=np.float32)
    origins[:, 0] = rng.uniform(-500.0, 500.0, random_count)
    origins[:, 1:] = rng.uniform(-1000.0, 1000.0, (random_count, 2))
    directions = targets - origins
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Add both fallback classes: a wide in-plane rectangle and a direction
    # outside the unit-vector certification envelope.  The latter is still a
    # valid parameterized ray and must match the general traversal exactly.
    union_center = (
        accelerator.host_union_bounds_min.astype(np.float64)
        + accelerator.host_union_bounds_max.astype(np.float64)
    ) * 0.5
    extra_origins = np.asarray(
        [
            [
                union_center[0],
                accelerator.host_union_bounds_min[1],
                accelerator.host_union_bounds_min[2],
            ],
            [0.0, 0.0, 0.0],
            [0.0, 10_000.0, 10_000.0],
        ],
        dtype=np.float32,
    )
    extra_directions = np.asarray(
        [
            [0.0, np.sqrt(0.5), np.sqrt(0.5)],
            [-0.25, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    origins = np.concatenate([origins, extra_origins])
    directions = np.concatenate([directions, extra_directions])
    expected = nearest_pmt_hit(
        accelerator, origins, directions, use_tlas=True
    )
    routing = FusedPMTRoutingCounters.allocate(accelerator.device)
    plan = plan_fused_pmt_candidates(
        accelerator.grid_locator,
        accelerator.host_bounds_min,
        accelerator.host_bounds_max,
        accelerator.host_union_bounds_min,
        accelerator.host_union_bounds_max,
        origins,
        directions,
    )
    expected_routing = {
        "certified_rays": int(np.count_nonzero(plan.union_hit & ~plan.fallback)),
        "fallback_rays": int(np.count_nonzero(plan.union_hit & plan.fallback)),
        "instance_box_visits": sum(map(len, plan.instance_ids)),
    }
    for compact_union in (True, False):
        routing.reset()
        actual = nearest_pmt_hit_fused_grid(
            accelerator,
            origins,
            directions,
            routing_counters=routing,
            compact_union=compact_union,
        )
        for field in actual.__dataclass_fields__:
            assert torch.equal(
                getattr(actual, field), getattr(expected, field)
            ), (compact_union, field)
        assert routing.snapshot() == expected_routing

    live = len(origins)
    capacity = live + 113
    device = accelerator.device
    origin_storage = torch.zeros(
        (capacity, 3), dtype=torch.float32, device=device
    )
    direction_storage = torch.zeros_like(origin_storage)
    origin_storage[:live] = torch.as_tensor(origins, device=device)
    direction_storage[:live] = torch.as_tensor(directions, device=device)
    count = torch.tensor([live], dtype=torch.int32, device=device)
    for compact_union in (True, False):
        workspace = accelerator.allocate_workspace(
            capacity, result_capacity=capacity
        )
        output = workspace.outputs(capacity)
        for field in output.__dataclass_fields__:
            value = getattr(output, field)
            value.fill_(123.0 if value.dtype.is_floating_point else 77)
        device_actual = nearest_pmt_hit_fused_grid_device_count(
            accelerator,
            origin_storage,
            direction_storage,
            count,
            launch_capacity=capacity,
            workspace=workspace,
            out=output,
            compact_union=compact_union,
        )
        for field in device_actual.__dataclass_fields__:
            value = getattr(device_actual, field)
            assert torch.equal(
                value[:live], getattr(expected, field)
            ), (compact_union, field)
            suffix = value[live:]
            sentinel = 123.0 if value.dtype.is_floating_point else 77
            assert torch.all(suffix == sentinel), (compact_union, field)


@pytest.mark.skipif(
    not triton_instance_available(require_cuda=True),
    reason="PyTorch, Triton, and a CUDA device are required",
)
def test_fused_grid_retains_the_nonrepresentable_micro_hit_fix():
    import torch

    from chroma_lar.triton_scene.compiler import compile_reflect3wires_scene

    scene = compile_reflect3wires_scene()
    accelerator = build_pmt_instance_accelerator(scene)
    origins = np.asarray(
        [
            [-2201.6728515625, 814.914794921875, -26.203067779541016],
            [-2201.6728515625, -1819.4339599609375, 1372.3004150390625],
        ],
        dtype=np.float32,
    )
    directions = np.asarray(
        [
            [-0.29941707849502563, 0.8183780312538147, 0.4905168116092682],
            [-0.28882408142089844, 0.6429386734962463, 0.7093732953071594],
        ],
        dtype=np.float32,
    )
    last_instance = np.asarray([58, 7], dtype=np.int32)
    last_triangle = np.asarray([453, 1736], dtype=np.int32)
    expected = nearest_pmt_hit(
        accelerator,
        origins,
        directions,
        last_instance=last_instance,
        last_triangle=last_triangle,
        use_tlas=True,
    )
    actual = nearest_pmt_hit_fused_grid(
        accelerator,
        origins,
        directions,
        last_instance=last_instance,
        last_triangle=last_triangle,
    )
    for field in actual.__dataclass_fields__:
        assert torch.equal(getattr(actual, field), getattr(expected, field)), field
    np.testing.assert_array_equal(
        actual.triangle_ids.cpu().numpy(), [4293, 3016]
    )
