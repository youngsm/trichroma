"""Parity tests for exact repeated-PMT instance traversal."""

from types import SimpleNamespace

import numpy as np
import pytest

import chroma_lar.triton_scene.instances as instance_module
from chroma_lar.triton_scene.instances import (
    _build_instance_tlas,
    _build_chroma_world_vertices,
    _infer_pmt_grid_locator,
    build_pmt_instance_accelerator,
    nearest_pmt_hit,
    nearest_pmt_hit_cpu,
    triton_instance_available,
)


def _staggered_grid_bounds(side=3):
    centers = []
    for row in range(side):
        for column in range(side):
            centers.append(
                [
                    -2170.0,
                    -900.0 + row * 471.0 + (column % 2) * 180.0,
                    -700.0 + column * 471.0,
                ]
            )
    centers = np.asarray(centers, dtype=np.float32)
    half = np.asarray([63.0, 56.0, 56.0], dtype=np.float32)
    return centers - half, centers + half


def test_grid_locator_requires_and_describes_ascending_row_major_lattice():
    lower, upper = _staggered_grid_bounds()
    locator = _infer_pmt_grid_locator(lower, upper)
    assert locator is not None
    assert (locator.rows, locator.columns) == (3, 3)
    assert locator.row_pitch == pytest.approx(471.0)
    assert locator.column_pitch == pytest.approx(471.0)
    assert locator.row_zero_min_y == pytest.approx(-900.0)
    assert locator.row_zero_max_y == pytest.approx(-720.0)
    assert locator.column_zero_z == pytest.approx(-700.0)
    assert locator.half_y >= 56.0
    assert locator.half_z >= 56.0

    # Reordering two boxes invalidates the instance=row*columns+column proof;
    # the general exact scanner must remain selected for that layout.
    reordered_lower = lower.copy()
    reordered_upper = upper.copy()
    reordered_lower[[0, 1]] = reordered_lower[[1, 0]]
    reordered_upper[[0, 1]] = reordered_upper[[1, 0]]
    assert _infer_pmt_grid_locator(reordered_lower, reordered_upper) is None


def test_grid_locator_rejects_non_square_or_overlapping_layouts():
    lower, upper = _staggered_grid_bounds()
    assert _infer_pmt_grid_locator(lower[:-1], upper[:-1]) is None
    expanded_lower = lower.copy()
    expanded_upper = upper.copy()
    expanded_lower[:, 1:] -= np.float32(300.0)
    expanded_upper[:, 1:] += np.float32(300.0)
    assert _infer_pmt_grid_locator(expanded_lower, expanded_upper) is None


def test_balanced_tlas_is_conservative_and_retains_ascending_leaf_order():
    lower = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, -1.0, 0.0], [4.0, 2.0, -3.0]],
        dtype=np.float32,
    )
    upper = lower + np.float32(0.5)
    node_min, node_max, left, right, instance, stack_capacity = (
        _build_instance_tlas(lower, upper)
    )
    leaves = []
    maximum_depth = 0

    def visit(node, depth):
        nonlocal maximum_depth
        maximum_depth = max(maximum_depth, depth)
        if instance[node] >= 0:
            leaf = int(instance[node])
            leaves.append(leaf)
            assert np.all(node_min[node] <= lower[leaf])
            assert np.all(node_max[node] >= upper[leaf])
            return
        assert np.all(node_min[node] <= node_min[left[node]])
        assert np.all(node_min[node] <= node_min[right[node]])
        assert np.all(node_max[node] >= node_max[left[node]])
        assert np.all(node_max[node] >= node_max[right[node]])
        visit(int(left[node]), depth + 1)
        visit(int(right[node]), depth + 1)

    visit(0, 0)
    assert leaves == [0, 1, 2]
    assert stack_capacity == maximum_depth


def test_cpu_rejects_nonrepresentable_adjacent_triangle_micro_hits():
    """Regression for the two exact PMT limit cycles from seed 2654443884."""

    from chroma_lar.triton_scene.compiler import compile_reflect3wires_scene

    scene = compile_reflect3wires_scene()
    origins = np.asarray(
        [
            [-2201.6728515625, 814.914794921875, -26.203067779541016],
            [-2201.6728515625, 814.914794921875, -26.203067779541016],
            [-2201.6728515625, -1819.4339599609375, 1372.3004150390625],
            [-2201.6728515625, -1819.4339599609375, 1372.3004150390625],
        ],
        dtype=np.float32,
    )
    directions = np.asarray(
        [
            [-0.29941707849502563, 0.8183780312538147, 0.4905168116092682],
            [-0.28882431983947754, 0.8211740255355835, 0.49219265580177307],
            [-0.29941675066947937, 0.6407495737075806, 0.706957995891571],
            [-0.28882408142089844, 0.6429386734962463, 0.7093732953071594],
        ],
        dtype=np.float32,
    )
    last_instance = np.asarray([58, 58, 7, 7], dtype=np.int32)
    last_triangle = np.asarray([453, 1733, 456, 1736], dtype=np.int32)

    result = nearest_pmt_hit_cpu(
        scene,
        origins,
        directions,
        last_instance=last_instance,
        last_triangle=last_triangle,
    )
    # The rejected reciprocal candidates were (1733,453,1736,456), all with
    # distance below 8e-6 mm and a bitwise-zero state update.  Traversal must
    # continue to the real opposite piece of the PMT rather than discard the
    # nearest candidate after the fact.
    np.testing.assert_array_equal(result.triangle_ids, [4293, 3013, 4296, 3016])
    np.testing.assert_array_equal(result.instance_ids, last_instance)
    assert np.all(result.distances > np.float32(1.6))
    advanced = origins + result.distances[:, None] * directions
    assert np.all(np.any(advanced != origins, axis=1))


@pytest.mark.skipif(
    not triton_instance_available(require_cuda=True),
    reason="PyTorch, Triton, and a CUDA device are required",
)
def test_tlas_and_legacy_reject_exact_nonrepresentable_micro_hits():
    from chroma_lar.triton_scene.compiler import compile_reflect3wires_scene

    scene = compile_reflect3wires_scene()
    accelerator = build_pmt_instance_accelerator(scene)
    assert accelerator.grid_locator is not None
    assert (
        accelerator.grid_locator.rows,
        accelerator.grid_locator.columns,
    ) == (9, 9)
    origins = np.asarray(
        [
            [-2201.6728515625, 814.914794921875, -26.203067779541016],
            [-2201.6728515625, 814.914794921875, -26.203067779541016],
            [-2201.6728515625, -1819.4339599609375, 1372.3004150390625],
            [-2201.6728515625, -1819.4339599609375, 1372.3004150390625],
        ],
        dtype=np.float32,
    )
    directions = np.asarray(
        [
            [-0.29941707849502563, 0.8183780312538147, 0.4905168116092682],
            [-0.28882431983947754, 0.8211740255355835, 0.49219265580177307],
            [-0.29941675066947937, 0.6407495737075806, 0.706957995891571],
            [-0.28882408142089844, 0.6429386734962463, 0.7093732953071594],
        ],
        dtype=np.float32,
    )
    last_instance = np.asarray([58, 58, 7, 7], dtype=np.int32)
    last_triangle = np.asarray([453, 1733, 456, 1736], dtype=np.int32)
    expected = nearest_pmt_hit_cpu(
        scene,
        origins,
        directions,
        last_instance=last_instance,
        last_triangle=last_triangle,
    )
    tlas = nearest_pmt_hit(
        accelerator,
        origins,
        directions,
        last_instance=last_instance,
        last_triangle=last_triangle,
        use_tlas=True,
    )
    legacy = nearest_pmt_hit(
        accelerator,
        origins,
        directions,
        last_instance=last_instance,
        last_triangle=last_triangle,
        use_tlas=False,
    )
    for actual in (tlas, legacy):
        np.testing.assert_array_equal(
            actual.triangle_ids.cpu().numpy(), expected.triangle_ids
        )
        np.testing.assert_array_equal(
            actual.instance_ids.cpu().numpy(), expected.instance_ids
        )
        np.testing.assert_allclose(
            actual.distances.cpu().numpy(), expected.distances, rtol=5e-6, atol=3e-4
        )
        assert not bool(actual.overflow.any().item())

    # The certificate mode intentionally removes the guard above: historical
    # Chroma accepts the adjacent triangle even though applying its distance
    # cannot change a detector-scale float32 position.  Keeping this behavior
    # behind the explicit compatibility flag lets us prove the legacy result
    # first, then quantify the production loop-prevention fix separately.
    compatibility_accelerator = build_pmt_instance_accelerator(
        scene, chroma_world_compatibility=True
    )
    for use_tlas in (False, True):
        historical = nearest_pmt_hit(
            compatibility_accelerator,
            origins,
            directions,
            last_instance=last_instance,
            last_triangle=last_triangle,
            use_tlas=use_tlas,
            chroma_world_compatibility=True,
        )
        np.testing.assert_array_equal(
            historical.triangle_ids.cpu().numpy(),
            [1733, 453, 4296, 3016],
        )
        historical_distance = historical.distances.cpu().numpy()
        # Exact distance words from the independent flattened Chroma CUDA
        # oracle.  The first pair are accepted adjacent-triangle micro-hits;
        # world-space rounding makes the second pair ordinary opposite-piece
        # hits, unlike the transformed-local CPU diagnostic above.
        np.testing.assert_array_equal(
            historical_distance.view(np.uint32),
            np.asarray(
                [926978051, 926067459, 1070817660, 1070819808],
                dtype=np.uint32,
            ),
        )
        assert np.all(historical_distance[:2] < np.float32(2.0e-5))
        assert np.all(
            historical.triangle_ids.cpu().numpy()[:2]
            != expected.triangle_ids[:2]
        )


def _rotation_y_90():
    return np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=np.float32,
    )


def _synthetic_scene(*, duplicate_first=False):
    # A canonical two-triangle square.  The diagonal intentionally exercises
    # original-triangle tie ordering and last-triangle exclusion.
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
    rotations = np.stack(
        (np.eye(3, dtype=np.float32), _rotation_y_90(), np.eye(3, dtype=np.float32))
    )
    translations = np.asarray(
        [[0.0, 0.0, 3.0], [3.0, 0.0, 0.0], [0.0, 0.0, 6.0]],
        dtype=np.float32,
    )
    if duplicate_first:
        translations[1] = translations[0]
        rotations[1] = rotations[0]
    inverse_rotations = np.transpose(rotations, (0, 2, 1)).copy()
    inverse_translations = -np.einsum(
        "nij,nj->ni", inverse_rotations, translations
    ).astype(np.float32)
    bounds_min = np.empty((3, 3), dtype=np.float32)
    bounds_max = np.empty((3, 3), dtype=np.float32)
    for index, (rotation, translation) in enumerate(zip(rotations, translations)):
        world = vertices @ rotation.T + translation
        bounds_min[index] = world.min(axis=0)
        bounds_max[index] = world.max(axis=0)
    return SimpleNamespace(
        pmt=SimpleNamespace(vertices=vertices, triangles=triangles),
        instances=SimpleNamespace(
            channel_id=np.asarray([7, 42, 105], dtype=np.int32),
            world_to_object_rotation=inverse_rotations,
            world_to_object_translation=inverse_translations,
            object_to_world_rotation=rotations,
            object_to_world_translation=translations,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        ),
    )


def test_chroma_world_vertices_replay_flatten_expression_and_rounding():
    scene = _synthetic_scene()
    actual = _build_chroma_world_vertices(scene)
    expected = np.empty_like(actual)
    for index, (rotation, translation) in enumerate(
        zip(
            scene.instances.object_to_world_rotation,
            scene.instances.object_to_world_translation,
        )
    ):
        expected[index] = (
            np.inner(scene.pmt.vertices, rotation) + translation
        )
    expected = expected.round(decimals=12)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.skipif(
    not triton_instance_available(require_cuda=True),
    reason="PyTorch, Triton, and a CUDA device are required",
)
def test_chroma_world_leaf_mode_is_opt_in_and_shared_by_both_traversals():
    scene = _synthetic_scene()
    default = build_pmt_instance_accelerator(scene)
    assert default.chroma_world_vertices is None
    assert default.chroma_triangle_indices is None
    with pytest.raises(ValueError, match="chroma_world_compatibility=True"):
        nearest_pmt_hit(
            default,
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            chroma_world_compatibility=True,
        )

    accelerator = build_pmt_instance_accelerator(
        scene, chroma_world_compatibility=True
    )
    assert tuple(accelerator.chroma_world_vertices.shape) == (3, 4, 3)
    assert tuple(accelerator.chroma_triangle_indices.shape) == (2, 3)
    origins = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [20.0, 20.0, 20.0]],
        dtype=np.float32,
    )
    directions = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    for use_tlas in (False, True):
        result = nearest_pmt_hit(
            accelerator,
            origins,
            directions,
            chroma_world_compatibility=True,
            use_tlas=use_tlas,
        )
        np.testing.assert_array_equal(
            result.triangle_ids.cpu().numpy(), [0, 0, -1]
        )
        np.testing.assert_array_equal(
            result.instance_ids.cpu().numpy(), [0, 2, -1]
        )
        np.testing.assert_array_equal(
            result.channel_ids.cpu().numpy(), [7, 105, -1]
        )
        np.testing.assert_array_equal(
            result.distances.cpu().numpy().view(np.uint32),
            np.asarray([3.0, 2.0, np.inf], dtype=np.float32).view(np.uint32),
        )
        np.testing.assert_array_equal(
            result.world_normals.cpu().numpy().view(np.uint32),
            np.asarray(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                dtype=np.float32,
            ).view(np.uint32),
        )


@pytest.mark.skipif(
    not triton_instance_available(require_cuda=True),
    reason="PyTorch, Triton, and a CUDA device are required",
)
def test_chroma_world_mode_tests_all_leaves_before_selecting_winner():
    """A world-space shared-vertex hit must not be lost before refinement.

    Transforming this ray into PMT-local space moves its barycentric result
    outside Chroma's tolerance, so the normal path reports a miss.  Testing
    only that path's eventual winner in world space cannot recover it.  The
    strict path evaluates every visited leaf using the flattened vertex words
    and finds original local triangle 669 in both traversal organizations.
    """

    from chroma_lar.triton_scene.compiler import compile_reflect3wires_scene

    scene = compile_reflect3wires_scene()
    accelerator = build_pmt_instance_accelerator(
        scene, chroma_world_compatibility=True
    )
    instance = 31
    target = scene.pmt.vertices[23:24]
    local_origin = target.copy()
    local_origin[:, 1] = np.float32(500.0)
    rotation = scene.instances.object_to_world_rotation[instance]
    translation = scene.instances.object_to_world_translation[instance]
    origin = (
        np.inner(local_origin, rotation) + translation
    ).astype(np.float32)
    world_target = (
        np.inner(target, rotation) + translation
    ).astype(np.float32)
    direction = (world_target - origin).astype(np.float32)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    local_space = nearest_pmt_hit(accelerator, origin, direction)
    assert int(local_space.triangle_ids.item()) == -1

    expected_distance_bits = np.asarray([0x4407CDD7], dtype=np.uint32)
    expected_normal_bits = np.asarray(
        [[0xBD923109, 0x3F77B1C0, 0xBE782E10]], dtype=np.uint32
    )
    for use_tlas in (False, True):
        result = nearest_pmt_hit(
            accelerator,
            origin,
            direction,
            use_tlas=use_tlas,
            chroma_world_compatibility=True,
        )
        np.testing.assert_array_equal(result.triangle_ids.cpu().numpy(), [669])
        np.testing.assert_array_equal(result.instance_ids.cpu().numpy(), [31])
        np.testing.assert_array_equal(result.channel_ids.cpu().numpy(), [31])
        np.testing.assert_array_equal(
            result.distances.cpu().numpy().view(np.uint32),
            expected_distance_bits,
        )
        np.testing.assert_array_equal(
            result.world_normals.cpu().numpy().view(np.uint32),
            expected_normal_bits,
        )


def test_cpu_reference_resolves_instances_channels_normals_and_misses():
    scene = _synthetic_scene()
    origins = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    directions = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    result = nearest_pmt_hit_cpu(scene, origins, directions)
    np.testing.assert_array_equal(result.instance_ids, [0, 1, 2, -1])
    np.testing.assert_array_equal(result.channel_ids, [7, 42, 105, -1])
    np.testing.assert_allclose(result.distances[:3], [3.0, 3.0, 2.0])
    assert np.isinf(result.distances[3])
    np.testing.assert_allclose(
        result.world_normals,
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        atol=2e-7,
    )
    np.testing.assert_array_equal(result.candidate_counts, [2, 1, 1, 0])


def test_cpu_tmax_last_triangle_and_tie_order_are_exact():
    scene = _synthetic_scene()
    origin = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    direction = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
    clipped = nearest_pmt_hit_cpu(scene, origin, direction, tmax=3.0)
    assert clipped.triangle_ids[0] == -1  # tmax is exclusive, as in Chroma.

    excluded = nearest_pmt_hit_cpu(
        scene,
        origin,
        direction,
        last_instance=0,
        last_triangle=0,
    )
    assert excluded.instance_ids[0] == 0
    assert excluded.triangle_ids[0] == 1

    duplicate = _synthetic_scene(duplicate_first=True)
    tied = nearest_pmt_hit_cpu(duplicate, origin, direction)
    assert tied.instance_ids[0] == 0
    assert tied.channel_ids[0] == 7
    with pytest.raises(ValueError, match="supplied together"):
        nearest_pmt_hit_cpu(scene, origin, direction, last_triangle=0)


@pytest.mark.skipif(
    not triton_instance_available(require_cuda=True),
    reason="PyTorch, Triton, and a CUDA device are required",
)
def test_triton_random_rays_match_cpu_and_workspace_reuse():
    scene = _synthetic_scene()
    accelerator = build_pmt_instance_accelerator(scene)
    assert np.all(
        accelerator.union_bounds_min.cpu().numpy()
        <= accelerator.bounds_min.cpu().numpy().min(axis=0)
    )
    assert np.all(
        accelerator.union_bounds_max.cpu().numpy()
        >= accelerator.bounds_max.cpu().numpy().max(axis=0)
    )
    rng = np.random.default_rng(20260901)
    count = 2048
    chosen = rng.integers(0, 3, size=count)
    local_targets = np.column_stack(
        (
            rng.uniform(-0.95, 0.95, count),
            rng.uniform(-0.95, 0.95, count),
            np.zeros(count),
        )
    ).astype(np.float32)
    rotations = scene.instances.object_to_world_rotation[chosen]
    translations = np.asarray(
        [[0.0, 0.0, 3.0], [3.0, 0.0, 0.0], [0.0, 0.0, 6.0]],
        dtype=np.float32,
    )[chosen]
    targets = np.einsum("nij,nj->ni", rotations, local_targets) + translations
    normals = rotations[:, :, 2]
    launch_distance = rng.uniform(0.2, 4.0, count).astype(np.float32)
    origins = (targets - normals * launch_distance[:, None]).astype(np.float32)
    directions = normals.astype(np.float32)
    # Mix in broadphase misses and exact tmax clipping.
    origins[::11] = np.asarray([20.0, 20.0, 20.0], dtype=np.float32)
    directions[::11] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    tmax = np.full(count, np.inf, dtype=np.float32)
    tmax[1::13] = launch_distance[1::13] * np.float32(0.9)

    expected = nearest_pmt_hit_cpu(
        scene, origins, directions, tmax=tmax
    )
    workspace = accelerator.allocate_workspace(ray_capacity=127)
    reusable = workspace.outputs(count)
    output_pointers = {
        name: getattr(reusable, name).data_ptr()
        for name in reusable.__dataclass_fields__
    }
    actual = nearest_pmt_hit(
        accelerator,
        origins,
        directions,
        tmax=tmax,
        ray_tile=127,
        workspace=workspace,
        out=reusable,
        check_overflow=True,
    )
    assert actual is reusable
    np.testing.assert_array_equal(
        actual.triangle_ids.cpu().numpy(), expected.triangle_ids
    )
    np.testing.assert_array_equal(
        actual.instance_ids.cpu().numpy(), expected.instance_ids
    )
    np.testing.assert_array_equal(
        actual.channel_ids.cpu().numpy(), expected.channel_ids
    )
    np.testing.assert_array_equal(
        actual.candidate_counts.cpu().numpy(), expected.candidate_counts
    )
    hit = expected.triangle_ids >= 0
    np.testing.assert_allclose(
        actual.distances.cpu().numpy()[hit],
        expected.distances[hit],
        rtol=3e-6,
        atol=2e-5,
    )
    assert np.isinf(actual.distances.cpu().numpy()[~hit]).all()
    np.testing.assert_allclose(
        actual.world_normals.cpu().numpy()[hit],
        expected.world_normals[hit],
        rtol=2e-6,
        atol=2e-6,
    )
    assert int(actual.overflow.sum().item()) == 0

    candidate_pointers = tuple(
        getattr(workspace, name).data_ptr()
        for name in (
            "local_origins",
            "local_directions",
            "candidate_tmax",
            "candidate_last_triangle",
            "candidate_instances",
            "candidate_triangles",
            "candidate_distances",
            "candidate_overflow",
        )
    )

    # The same dynamically grown buffers are safe on a second epoch, including
    # paired self-hit exclusion.
    repeated_out = workspace.outputs(127)
    for name in repeated_out.__dataclass_fields__:
        assert getattr(repeated_out, name).data_ptr() == output_pointers[name]
    repeated = nearest_pmt_hit(
        accelerator,
        origins[:127],
        directions[:127],
        last_instance=expected.instance_ids[:127],
        last_triangle=expected.triangle_ids[:127],
        ray_tile=127,
        workspace=workspace,
        out=repeated_out,
    )
    repeated_expected = nearest_pmt_hit_cpu(
        accelerator,
        origins[:127],
        directions[:127],
        last_instance=expected.instance_ids[:127],
        last_triangle=expected.triangle_ids[:127],
    )
    np.testing.assert_array_equal(
        repeated.triangle_ids.cpu().numpy(), repeated_expected.triangle_ids
    )
    np.testing.assert_array_equal(
        repeated.instance_ids.cpu().numpy(), repeated_expected.instance_ids
    )
    assert candidate_pointers == tuple(
        getattr(workspace, name).data_ptr()
        for name in (
            "local_origins",
            "local_directions",
            "candidate_tmax",
            "candidate_last_triangle",
            "candidate_instances",
            "candidate_triangles",
            "candidate_distances",
            "candidate_overflow",
        )
    )


@pytest.mark.skipif(
    not triton_instance_available(require_cuda=True),
    reason="PyTorch, Triton, and a CUDA device are required",
)
def test_union_prefilter_zero_candidate_tiles_do_not_grow_pair_storage():
    scene = _synthetic_scene()
    accelerator = build_pmt_instance_accelerator(scene)
    count = 513
    origins = np.tile([20.0, 20.0, 20.0], (count, 1)).astype(np.float32)
    directions = np.tile([0.0, 1.0, 0.0], (count, 1)).astype(np.float32)
    workspace = accelerator.allocate_workspace(ray_capacity=127)
    reusable = workspace.outputs(count)
    # Prove the fused initializer clears stale data on the union-empty path.
    reusable.triangle_ids.fill_(99)
    reusable.distances.fill_(0.0)
    reusable.instance_ids.fill_(99)
    reusable.channel_ids.fill_(99)
    reusable.world_normals.fill_(99.0)
    reusable.overflow.fill_(1)
    reusable.candidate_counts.fill_(99)
    result = nearest_pmt_hit(
        accelerator,
        origins,
        directions,
        ray_tile=127,
        workspace=workspace,
        out=reusable,
    )
    assert workspace.candidate_capacity == 0
    assert np.all(result.triangle_ids.cpu().numpy() == -1)
    assert np.all(result.instance_ids.cpu().numpy() == -1)
    assert np.all(result.channel_ids.cpu().numpy() == -1)
    assert np.all(result.candidate_counts.cpu().numpy() == 0)
    assert np.all(np.isinf(result.distances.cpu().numpy()))
    assert np.all(result.world_normals.cpu().numpy() == 0.0)


@pytest.mark.skipif(
    not triton_instance_available(require_cuda=True),
    reason="PyTorch, Triton, and a CUDA device are required",
)
def test_exact_reflect3wires_scene_randomized_targeted_rays(monkeypatch):
    # This is the correctness-bar geometry, not merely the synthetic unit mesh.
    from chroma_lar.triton_scene.compiler import compile_reflect3wires_scene

    scene = compile_reflect3wires_scene()
    accelerator = build_pmt_instance_accelerator(scene)
    monkeypatch.setattr(instance_module, "GRID_MIN_ACTIVE_RAYS", 0)
    rng = np.random.default_rng(81005120)
    count = 64
    instance = rng.integers(0, scene.instances.count, size=count)
    # Cast from the active volume toward non-degenerate points on the curved
    # PMT face.  A ray through local (x,z)=(0,0) lands on the shared polar
    # vertex of many triangles, where different but optically equivalent
    # adjacent-triangle choices are expected from floating operation order.
    radius = np.sqrt(rng.uniform(0.01, 0.8, count)) * np.float32(50.0)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
    local_origins = np.column_stack(
        (
            radius * np.cos(azimuth),
            np.full(count, 500.0),
            radius * np.sin(azimuth),
        )
    ).astype(np.float32)
    local_targets = local_origins.copy()
    local_targets[:, 1] = -150.0
    rotation = scene.instances.object_to_world_rotation[instance]
    translation = scene.instances.object_to_world_translation[instance]
    origins = (
        np.einsum("nij,nj->ni", rotation, local_origins) + translation
    ).astype(np.float32)
    targets = (
        np.einsum("nij,nj->ni", rotation, local_targets) + translation
    ).astype(np.float32)
    directions = (targets - origins).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    expected = nearest_pmt_hit_cpu(scene, origins, directions)
    results = (
        nearest_pmt_hit(accelerator, origins, directions, ray_tile=31),
        nearest_pmt_hit(
            accelerator, origins, directions, ray_tile=31, use_grid=False
        ),
        nearest_pmt_hit(
            accelerator, origins, directions, ray_tile=31, use_tlas=True
        ),
    )
    hit = expected.triangle_ids >= 0
    for actual in results:
        np.testing.assert_array_equal(
            actual.triangle_ids.cpu().numpy(), expected.triangle_ids
        )
        np.testing.assert_array_equal(
            actual.channel_ids.cpu().numpy(), expected.channel_ids
        )
        np.testing.assert_array_equal(
            actual.candidate_counts.cpu().numpy(), expected.candidate_counts
        )
        np.testing.assert_allclose(
            actual.distances.cpu().numpy()[hit],
            expected.distances[hit],
            rtol=5e-6,
            atol=3e-4,
        )
        assert int(actual.overflow.sum().item()) == 0


@pytest.mark.skipif(
    not triton_instance_available(require_cuda=True),
    reason="PyTorch, Triton, and a CUDA device are required",
)
def test_grid_locator_routes_wide_grazing_rectangles_to_exact_tlas(monkeypatch):
    """A ray spanning more than the fast 16-box budget remains uncapped."""

    from chroma_lar.triton_scene.compiler import compile_reflect3wires_scene

    scene = compile_reflect3wires_scene()
    accelerator = build_pmt_instance_accelerator(scene)
    monkeypatch.setattr(instance_module, "GRID_MIN_ACTIVE_RAYS", 0)
    locator = accelerator.grid_locator
    assert locator is not None
    lower = accelerator.host_union_bounds_min
    upper = accelerator.host_union_bounds_max
    x = np.float32((float(lower[0]) + float(upper[0])) * 0.5)
    origins = np.asarray(
        [
            [x, lower[1] - 200.0, lower[2] - 200.0],
            [x, upper[1] + 200.0, lower[2] - 200.0],
        ],
        dtype=np.float32,
    )
    directions = np.asarray(
        [[0.0, 1.0, 1.0], [0.0, -1.0, 1.0]], dtype=np.float32
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    grid = nearest_pmt_hit(accelerator, origins, directions)
    legacy = nearest_pmt_hit(
        accelerator, origins, directions, use_grid=False
    )
    np.testing.assert_array_equal(
        grid.triangle_ids.cpu().numpy(), legacy.triangle_ids.cpu().numpy()
    )
    np.testing.assert_array_equal(
        grid.instance_ids.cpu().numpy(), legacy.instance_ids.cpu().numpy()
    )
    np.testing.assert_array_equal(
        grid.candidate_counts.cpu().numpy(),
        legacy.candidate_counts.cpu().numpy(),
    )
    np.testing.assert_array_equal(
        grid.distances.cpu().numpy().view(np.uint32),
        legacy.distances.cpu().numpy().view(np.uint32),
    )
    assert not bool(grid.overflow.any().item())


@pytest.mark.skipif(
    not triton_instance_available(require_cuda=True),
    reason="PyTorch, Triton, and a CUDA device are required",
)
def test_grid_locator_matches_general_scan_for_arbitrary_detector_rays(monkeypatch):
    from chroma_lar.triton_scene.compiler import compile_reflect3wires_scene

    scene = compile_reflect3wires_scene()
    accelerator = build_pmt_instance_accelerator(scene)
    monkeypatch.setattr(instance_module, "GRID_MIN_ACTIVE_RAYS", 0)
    rng = np.random.default_rng(20260903)
    count = 4096
    origins = np.column_stack(
        (
            rng.uniform(-2050.0, 1800.0, count),
            rng.uniform(-2300.0, 2300.0, count),
            rng.uniform(-2300.0, 2300.0, count),
        )
    ).astype(np.float32)
    directions = rng.normal(size=(count, 3)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    tmax = rng.uniform(10.0, 8000.0, count).astype(np.float32)
    tmax[::7] = np.float32(np.inf)

    grid = nearest_pmt_hit(
        accelerator, origins, directions, tmax=tmax, ray_tile=1021
    )
    legacy = nearest_pmt_hit(
        accelerator,
        origins,
        directions,
        tmax=tmax,
        ray_tile=1021,
        use_grid=False,
    )
    for grid_values, legacy_values in (
        (grid.triangle_ids, legacy.triangle_ids),
        (grid.instance_ids, legacy.instance_ids),
        (grid.channel_ids, legacy.channel_ids),
        (grid.candidate_counts, legacy.candidate_counts),
        (grid.overflow, legacy.overflow),
    ):
        np.testing.assert_array_equal(
            grid_values.cpu().numpy(), legacy_values.cpu().numpy()
        )
    np.testing.assert_array_equal(
        grid.distances.cpu().numpy().view(np.uint32),
        legacy.distances.cpu().numpy().view(np.uint32),
    )
    np.testing.assert_array_equal(
        grid.world_normals.cpu().numpy().view(np.uint32),
        legacy.world_normals.cpu().numpy().view(np.uint32),
    )


@pytest.mark.skipif(
    not triton_instance_available(require_cuda=True),
    reason="PyTorch, Triton, and a CUDA device are required",
)
def test_exact_reflect3wires_pmt_two_boundary_self_hit_and_optical_sides():
    """Lock the outer-glass -> photocathode path used by Chroma fill_state.

    Every retained PMT is exercised.  The second query begins at the exact
    first hit and excludes only its ``(instance, local triangle)`` pair, which
    is the instanced equivalent of Chroma's flattened global-triangle
    ``last_hit_triangle`` state.
    """

    from chroma_lar.triton_scene.compiler import compile_reflect3wires_scene

    scene = compile_reflect3wires_scene()
    accelerator = build_pmt_instance_accelerator(scene)
    rng = np.random.default_rng(971305)
    rays_per_instance = 8
    instance = np.repeat(
        np.arange(scene.instances.count, dtype=np.int32), rays_per_instance
    )
    count = len(instance)
    radius = np.sqrt(rng.uniform(0.02, 0.70, count)) * np.float32(50.0)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
    local_origins = np.column_stack(
        (
            radius * np.cos(azimuth),
            np.full(count, 500.0),
            radius * np.sin(azimuth),
        )
    ).astype(np.float32)
    local_targets = local_origins.copy()
    local_targets[:, 1] = np.float32(-150.0)
    rotation = scene.instances.object_to_world_rotation[instance]
    translation = scene.instances.object_to_world_translation[instance]
    origins = (
        np.einsum("nij,nj->ni", rotation, local_origins) + translation
    ).astype(np.float32)
    targets = (
        np.einsum("nij,nj->ni", rotation, local_targets) + translation
    ).astype(np.float32)
    directions = (targets - origins).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    first_cpu = nearest_pmt_hit_cpu(scene, origins, directions)
    first = nearest_pmt_hit(accelerator, origins, directions, ray_tile=257)
    first_triangle = first.triangle_ids.cpu().numpy()
    first_instance = first.instance_ids.cpu().numpy()
    first_distance = first.distances.cpu().numpy()
    first_normal = first.world_normals.cpu().numpy()
    np.testing.assert_array_equal(first_triangle, first_cpu.triangle_ids)
    np.testing.assert_array_equal(first_instance, instance)
    np.testing.assert_array_equal(first.channel_ids.cpu().numpy(), instance)
    np.testing.assert_allclose(
        first_distance, first_cpu.distances, rtol=5e-6, atol=6e-4
    )

    # The first boundary is the unsurfaced LAr/glass envelope.  Its raw normal
    # faces these incoming rays, so Chroma selects material2 -> material1.
    names = scene.tables.material_names
    lar = names.index("liquid_argon")
    glass = names.index("glass")
    vacuum = names.index("vacuum")
    np.testing.assert_array_equal(
        scene.pmt.scene_surface_index[first_triangle],
        np.full(count, -1, dtype=np.int32),
    )
    assert np.all(np.sum(first_normal * -directions, axis=1) > 0.0)
    np.testing.assert_array_equal(
        scene.pmt.scene_material2_index[first_triangle],
        np.full(count, lar, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        scene.pmt.scene_material1_index[first_triangle],
        np.full(count, glass, dtype=np.int32),
    )

    boundary_origins = (
        origins + first_distance[:, None] * directions
    ).astype(np.float32)
    second_cpu = nearest_pmt_hit_cpu(
        scene,
        boundary_origins,
        directions,
        last_instance=first_instance,
        last_triangle=first_triangle,
    )
    second = nearest_pmt_hit(
        accelerator,
        boundary_origins,
        directions,
        last_instance=first_instance,
        last_triangle=first_triangle,
        ray_tile=257,
    )
    second_triangle = second.triangle_ids.cpu().numpy()
    second_normal = second.world_normals.cpu().numpy()
    np.testing.assert_array_equal(second_triangle, second_cpu.triangle_ids)
    np.testing.assert_array_equal(second.instance_ids.cpu().numpy(), instance)
    np.testing.assert_array_equal(second.channel_ids.cpu().numpy(), instance)
    np.testing.assert_allclose(
        second.distances.cpu().numpy(),
        second_cpu.distances,
        rtol=5e-6,
        atol=2e-4,
    )

    perfect = scene.tables.surface_names.index("perfect_pmt_photocathode")
    np.testing.assert_array_equal(
        scene.pmt.scene_surface_index[second_triangle],
        np.full(count, perfect, dtype=np.int32),
    )
    assert np.all(np.sum(second_normal * -directions, axis=1) > 0.0)
    np.testing.assert_array_equal(
        scene.pmt.scene_material2_index[second_triangle],
        np.full(count, glass, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        scene.pmt.scene_material1_index[second_triangle],
        np.full(count, vacuum, dtype=np.int32),
    )
