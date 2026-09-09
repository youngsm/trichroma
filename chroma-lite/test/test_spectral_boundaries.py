"""Regression tests for finite-precision boundary re-intersections."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from chroma.triton.boundary import offset_boundary_points
from chroma.triton.boundary_kernels import offset_boundary_point
from chroma.triton.bvh import build_packed_bvh, nearest_hit_cpu


def regression():
    data = json.loads((Path(__file__).parent / "data/theia_coplanar_boundary.json").read_text())
    return (
        np.asarray(data["triangles"], np.float32),
        np.asarray([data["point"]], np.float32),
        np.asarray([data["outgoing"]], np.float32),
    )


def test_original_pmt_adjacent_face_reintersection_is_removed():
    triangles, point, direction = regression()
    bvh = build_packed_bvh(triangles.reshape(-1, 3), np.arange(6).reshape(-1, 3))
    old = nearest_hit_cpu(bvh, point, direction, last_hit=0)
    assert old.triangle_ids[0] == 1 and 0 < old.distances[0] < 0.003
    fixed = offset_boundary_points(point, triangles[:1], direction)
    new = nearest_hit_cpu(bvh, fixed, direction, last_hit=0)
    assert new.triangle_ids[0] == -1
    assert np.linalg.norm(fixed - point) < 0.02


def test_offset_preserves_close_physical_interface():
    first = np.array([[1999, 1999, 2000], [2001, 1999, 2000], [2000, 2001, 2000]], np.float32)
    second = first + [0, 0, 0.15]
    triangles = np.asarray([first, second], np.float32)
    bvh = build_packed_bvh(triangles.reshape(-1, 3), np.arange(6).reshape(-1, 3))
    point, direction = np.array([[2000, 2000, 2000]], np.float32), np.array([[0, 0, 1]], np.float32)
    shifted = offset_boundary_points(point, triangles[:1], direction)
    result = nearest_hit_cpu(bvh, shifted, direction, last_hit=0)
    assert result.triangle_ids[0] == 1
    assert 0.14 < result.distances[0] < 0.15
    assert np.linalg.norm(shifted - point) < 0.001


def grazing_regression():
    data = json.loads((Path(__file__).parent / "data/theia_grazing_boundary.json").read_text())
    triangles = np.asarray(data["triangles"], np.float32)
    bvh = build_packed_bvh(triangles.reshape(-1, 3), np.arange(6).reshape(-1, 3))
    return (
        bvh,
        np.asarray([data["origin"]], np.float32),
        np.asarray([data["direction"]], np.float32),
    )


def test_world_scale_grazing_ray_rejects_both_false_float32_facets():
    from chroma.triton.bvh import nearest_hit_bvh_cpu

    bvh, origin, direction = grazing_regression()
    assert nearest_hit_cpu(bvh, origin, direction).triangle_ids[0] == 0
    for query in (nearest_hit_cpu, nearest_hit_bvh_cpu):
        assert query(bvh, origin, direction, high_precision=True).triangle_ids[0] == -1


def test_gpu_high_precision_grazing_and_nearby_rays_match_cpu():
    from chroma.triton.bvh import nearest_hit_bvh_cpu
    from chroma.triton.bvh_kernels import DevicePackedBVH, nearest_hit_local

    if not torch.cuda.is_available():
        pytest.skip("local CUDA GPU required")
    bvh, origin, direction = grazing_regression()
    rng = np.random.default_rng(8115)
    origins = np.broadcast_to(origin, (4096, 3)).copy()
    directions = np.broadcast_to(direction, (4096, 3)).copy()
    directions[1:] += rng.normal(0, 2e-5, (4095, 3)).astype(np.float32)
    expected = nearest_hit_cpu(bvh, origins, directions, high_precision=True)
    accelerated = nearest_hit_bvh_cpu(bvh, origins, directions, high_precision=True)
    np.testing.assert_array_equal(accelerated.triangle_ids, expected.triangle_ids)
    np.testing.assert_array_equal(accelerated.distances, expected.distances)
    assert 0 < np.count_nonzero(expected.triangle_ids >= 0) < len(origins)
    actual = nearest_hit_local(
        DevicePackedBVH.from_host(bvh),
        origins,
        directions,
        high_precision=True,
        check_overflow=True,
    )
    np.testing.assert_array_equal(actual.triangle_ids.cpu(), expected.triangle_ids)
    np.testing.assert_array_equal(actual.distances.cpu(), expected.distances)


def test_gpu_boundary_origins_match_cpu_and_lie_on_outgoing_side():
    if not torch.cuda.is_available():
        pytest.skip("local CUDA GPU required")

    @triton.jit
    def kernel(points, triangles, directions, result, N: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        mask = i < N
        x = tl.load(points + 3 * i, mask, 0.0)
        y = tl.load(points + 3 * i + 1, mask, 0.0)
        z = tl.load(points + 3 * i + 2, mask, 0.0)
        dx = tl.load(directions + 3 * i, mask, 0.0)
        dy = tl.load(directions + 3 * i + 1, mask, 0.0)
        dz = tl.load(directions + 3 * i + 2, mask, 0.0)
        x, y, z = offset_boundary_point(triangles, i, x, y, z, dx, dy, dz, mask)
        tl.store(result + 3 * i, x, mask)
        tl.store(result + 3 * i + 1, y, mask)
        tl.store(result + 3 * i + 2, z, mask)

    rng = np.random.default_rng(812)
    centers = rng.uniform(-30000, 30000, (2048, 1, 3))
    triangles = (centers + rng.uniform(-100, 100, (2048, 3, 3))).astype(np.float32)
    points = (triangles.astype(float).mean(axis=1) + rng.uniform(-0.005, 0.005, (2048, 3))).astype(
        np.float32
    )
    directions = rng.normal(size=points.shape).astype(np.float32)
    expected = offset_boundary_points(points, triangles, directions)
    inputs = [torch.from_numpy(v).cuda() for v in (points, triangles, directions)]
    actual = torch.empty_like(inputs[0])
    kernel[(triton.cdiv(len(points), 128),)](
        *inputs, actual, N=len(points), B=128, enable_fp_fusion=False
    )
    np.testing.assert_array_equal(actual.cpu(), expected)
    tri64 = triangles.astype(float)
    normals = np.cross(tri64[:, 1] - tri64[:, 0], tri64[:, 2] - tri64[:, 0])
    distance = np.sum((expected.astype(float) - tri64[:, 0]) * normals, axis=1)
    outgoing = np.sum(directions.astype(float) * normals, axis=1)
    assert np.all(distance * outgoing > 0)
