"""Correctness tests for the optional instance-local Triton BVH backend."""

from .unittest_find import unittest

import numpy as np

from chroma.triton.bvh import build_packed_bvh, nearest_hit_cpu
from chroma.triton import bvh_kernels


def _cube_mesh():
    vertices = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    triangles = np.array(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 7, 6], [3, 6, 2],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ],
        dtype=np.int32,
    )
    return vertices, triangles


class _MeshLike(object):
    def __init__(self, vertices, triangles):
        self.vertices = vertices
        self.triangles = triangles


class TestPackedBVHBuilder(unittest.TestCase):
    def test_layout_retains_ids_and_conservative_leaf_bounds(self):
        vertices, triangles = _cube_mesh()
        bvh = build_packed_bvh(vertices, triangles)

        self.assertEqual(bvh.degree, 4)
        self.assertEqual(bvh.layer_counts, (1, 3, 12))
        self.assertEqual(bvh.layer_offsets, (0, 1, 4))
        self.assertEqual(bvh.depth, 2)
        self.assertEqual(bvh.stack_capacity, 4)
        self.assertEqual(bvh.nodes.dtype, np.uint32)
        self.assertFalse(bvh.nodes.flags.writeable)

        leaves = bvh.nodes[bvh.layer_offsets[-1] :]
        np.testing.assert_array_equal(
            np.sort(leaves[:, 3]), np.arange(len(triangles), dtype=np.uint32)
        )
        tolerance = float(bvh.world_scale) * 1.0e-3 + 1.0e-6
        for leaf in leaves:
            triangle_id = int(leaf[3])
            lower_fixed = leaf[:3] & np.uint32(0xFFFF)
            upper_fixed = leaf[:3] >> np.uint32(16)
            lower = lower_fixed.astype(np.float32) * bvh.world_scale + bvh.world_origin
            upper = upper_fixed.astype(np.float32) * bvh.world_scale + bvh.world_origin
            points = bvh.triangle_vertices[triangle_id]
            self.assertTrue(np.all(points.min(axis=0) >= lower - tolerance))
            self.assertTrue(np.all(points.max(axis=0) <= upper + tolerance))

        # Every inner child range stays within the following layer.
        for layer_number in range(len(bvh.layer_counts) - 1):
            start = bvh.layer_offsets[layer_number]
            stop = start + bvh.layer_counts[layer_number]
            child_start = bvh.layer_offsets[layer_number + 1]
            child_stop = child_start + bvh.layer_counts[layer_number + 1]
            for node in bvh.nodes[start:stop]:
                count = int(node[3] >> np.uint32(28))
                first = int(node[3] & np.uint32(0x0FFFFFFF))
                self.assertGreaterEqual(count, 1)
                self.assertLessEqual(count, 4)
                self.assertGreaterEqual(first, child_start)
                self.assertLessEqual(first + count, child_stop)

    def test_mesh_like_input_and_degree_validation(self):
        vertices, triangles = _cube_mesh()
        bvh = build_packed_bvh(_MeshLike(vertices, triangles))
        self.assertEqual(bvh.triangle_count, len(triangles))
        with self.assertRaises(ValueError):
            build_packed_bvh(vertices, triangles, degree=2)

    def test_cpu_reference_tmax_and_last_hit(self):
        vertices = np.array(
            [
                [-2.0, -2.0, 1.0], [2.0, -2.0, 1.0], [0.0, 2.0, 1.0],
                [-2.0, -2.0, 2.0], [2.0, -2.0, 2.0], [0.0, 2.0, 2.0],
            ],
            dtype=np.float32,
        )
        triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        bvh = build_packed_bvh(vertices, triangles)
        origins = np.zeros((4, 3), dtype=np.float32)
        directions = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (4, 1))
        result = nearest_hit_cpu(
            bvh,
            origins,
            directions,
            tmax=np.array([3.0, 0.5, 1.0, 3.0], np.float32),
            last_hit=np.array([-1, -1, -1, 0], np.int32),
        )
        np.testing.assert_array_equal(result.triangle_ids, [0, -1, -1, 1])
        np.testing.assert_allclose(result.distances[[0, 3]], [1.0, 2.0])
        self.assertTrue(np.isinf(result.distances[[1, 2]]).all())

    def test_optional_backend_module_always_imports(self):
        self.assertIsInstance(bvh_kernels.triton_available(), bool)


@unittest.skipUnless(
    bvh_kernels.triton_available(require_cuda=True),
    "PyTorch, Triton, and a CUDA device are required",
)
class TestTritonPackedBVH(unittest.TestCase):
    def test_random_local_rays_match_cpu_with_tmax(self):
        vertices, triangles = _cube_mesh()
        host_bvh = build_packed_bvh(vertices, triangles)
        device_bvh = host_bvh.to_triton()

        generator = np.random.default_rng(20260901)
        count = 4096
        origins = generator.normal(size=(count, 3)).astype(np.float32)
        origins /= np.linalg.norm(origins, axis=1, keepdims=True)
        origins *= generator.uniform(2.0, 5.0, size=(count, 1)).astype(np.float32)
        targets = generator.uniform(-0.75, 0.75, size=(count, 3)).astype(np.float32)
        directions = targets - origins
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        unrestricted = nearest_hit_cpu(host_bvh, origins, directions)
        # Exercise scalar/per-ray cutoff semantics: every third otherwise-hit
        # ray receives a cutoff just before its intersection.
        tmax = np.full(count, np.inf, dtype=np.float32)
        finite = np.isfinite(unrestricted.distances)
        clipped = finite & (np.arange(count) % 3 == 0)
        tmax[clipped] = unrestricted.distances[clipped] * np.float32(0.9)
        expected = nearest_hit_cpu(
            host_bvh, origins, directions, tmax=tmax
        )

        workspace = device_bvh.allocate_workspace(count)
        actual = bvh_kernels.nearest_hit_local(
            device_bvh,
            origins,
            directions,
            tmax=tmax,
            workspace=workspace,
            collect_stats=True,
            check_overflow=True,
        )
        actual_ids = actual.triangle_ids.cpu().numpy()
        actual_distances = actual.distances.cpu().numpy()
        np.testing.assert_array_equal(actual_ids, expected.triangle_ids)
        hit = expected.triangle_ids >= 0
        np.testing.assert_allclose(
            actual_distances[hit],
            expected.distances[hit],
            rtol=3.0e-6,
            atol=2.0e-5,
        )
        self.assertTrue(np.isinf(actual_distances[~hit]).all())
        self.assertEqual(int(actual.overflow.sum().item()), 0)
        self.assertLessEqual(
            int(actual.max_stack.max().item()), host_bvh.stack_capacity
        )

    def test_last_hit_and_empty_batch(self):
        vertices = np.array(
            [
                [-2.0, -2.0, 1.0], [2.0, -2.0, 1.0], [0.0, 2.0, 1.0],
                [-2.0, -2.0, 2.0], [2.0, -2.0, 2.0], [0.0, 2.0, 2.0],
            ],
            dtype=np.float32,
        )
        triangles = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        bvh = build_packed_bvh(vertices, triangles).to_triton()
        result = bvh_kernels.nearest_hit_local(
            bvh,
            np.array([[0.0, 0.0, 0.0]], np.float32),
            np.array([[0.0, 0.0, 1.0]], np.float32),
            last_hit=0,
        )
        self.assertEqual(int(result.triangle_ids.item()), 1)
        self.assertAlmostEqual(float(result.distances.item()), 2.0, places=5)

        empty = np.empty((0, 3), dtype=np.float32)
        empty_result = bvh_kernels.nearest_hit_local(bvh, empty, empty)
        self.assertEqual(empty_result.triangle_ids.numel(), 0)
        self.assertEqual(empty_result.distances.numel(), 0)


if __name__ == "__main__":
    unittest.main()
