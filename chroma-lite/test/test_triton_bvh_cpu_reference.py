"""Exact validation for scalable host construction and CPU broadphase."""

import numpy as np
import pytest

from chroma.triton.bvh import _parent_layer, build_packed_bvh, nearest_hit_cpu, nearest_hit_bvh_cpu


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 17, 1025, 4 * (1 << 18) + 1])
def test_vectorized_parent_words_match_scalar_construction(count):
    rng = np.random.default_rng(1901)
    lower = rng.integers(0, 32768, (count, 3), dtype=np.uint32)
    upper = lower + rng.integers(0, 32768, (count, 3), dtype=np.uint32)
    children = np.column_stack(
        [lower | (upper << np.uint32(16)), np.arange(count, dtype=np.uint32)]
    )
    actual = _parent_layer(children, 4)
    # Audit every small-tree parent and a sample spanning the large work-buffer
    # boundary, including the partial last parent and all padding behavior.
    ids = (
        np.arange(len(actual))
        if count < 2000
        else np.unique(
            np.r_[0, (1 << 18) - 1, 1 << 18, len(actual) - 1, rng.integers(len(actual), size=1000)]
        )
    )
    for index in ids:
        group = children[index * 4 : (index + 1) * 4]
        expected = np.r_[
            (group[:, :3] & np.uint32(0xFFFF)).min(0)
            | ((group[:, :3] >> np.uint32(16)).max(0) << np.uint32(16)),
            np.uint32((len(group) << 28) | int(index * 4)),
        ]
        np.testing.assert_array_equal(actual[index], expected)


def test_cpu_bvh_matches_brute_force_for_random_rays_and_limits():
    rng = np.random.default_rng(481)
    centers = rng.uniform(-1000.0, 1000.0, (500, 1, 3))
    vertices = (centers + rng.normal(0, 40.0, (500, 3, 3))).reshape(-1, 3).astype(np.float32)
    triangles = np.arange(len(vertices)).reshape(-1, 3)
    bvh = build_packed_bvh(vertices, triangles)
    origins = rng.uniform(-1500, 1500, (1000, 3)).astype(np.float32)
    directions = rng.normal(size=(1000, 3))
    directions[:500] = centers[:, 0] - origins[:500]
    directions = (directions / np.linalg.norm(directions, axis=1)[:, None]).astype(np.float32)
    limits = np.where(np.arange(1000) % 3, np.inf, 100.0)
    previous = rng.integers(-1, 500, size=1000)
    expected = nearest_hit_cpu(bvh, origins, directions, tmax=limits, last_hit=previous)
    actual = nearest_hit_bvh_cpu(bvh, origins, directions, tmax=limits, last_hit=previous)
    np.testing.assert_array_equal(actual.triangle_ids, expected.triangle_ids)
    np.testing.assert_array_equal(actual.distances, expected.distances)


def test_cpu_bvh_parallel_grazing_ties_and_excluded_previous_triangle():
    vertices = np.array(
        [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 3], [10, 0, 3], [0, 10, 3]], np.float32
    )
    bvh = build_packed_bvh(vertices, np.array([[0, 1, 2], [0, 1, 2], [3, 4, 5]]))
    origins = np.array(
        [[1, 1, -1], [-2e-6, 1, -1], [5, 5, -1], [20, 20, 0], [1, 1, 0], [1, 1, 5]], np.float32
    )
    directions = np.array([[0, 0, 1]] * 3 + [[1, 0, 0], [0, 0, 1], [0, 0, -1]], np.float32)
    previous = np.array([-1, -1, -1, -1, 2, 2])
    expected = nearest_hit_cpu(bvh, origins, directions, last_hit=previous)
    actual = nearest_hit_bvh_cpu(bvh, origins, directions, last_hit=previous)
    np.testing.assert_array_equal(actual.triangle_ids, expected.triangle_ids)
    np.testing.assert_array_equal(actual.distances, expected.distances)
    assert actual.triangle_ids[0] == 0 and actual.triangle_ids[-1] == 0
