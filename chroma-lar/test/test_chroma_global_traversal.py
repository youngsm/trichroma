"""Focused tests for the strict flattened Chroma traversal oracle."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from chroma_lar.triton_scene.chroma_global_bvh import (
    build_chroma_global_bvh_artifact,
)
from chroma_lar.triton_scene.chroma_global_traversal import (
    ChromaGlobalBVHDevice,
    nearest_chroma_global_hit,
    triton_chroma_global_available,
)


if triton_chroma_global_available():
    import torch
    import triton
    import triton.language as tl

    from chroma_lar.triton_scene.chroma_global_traversal import (
        _chroma_oriented_normal_words,
    )

    @triton.jit
    def _normal_orientation_probe(
        normal, incident_dot, output, nitems, BLOCK_SIZE: tl.constexpr
    ):
        lane = tl.arange(0, BLOCK_SIZE)
        valid = lane < nitems
        nx = tl.load(normal + lane * 3, mask=valid, other=0.0)
        ny = tl.load(normal + lane * 3 + 1, mask=valid, other=0.0)
        nz = tl.load(normal + lane * 3 + 2, mask=valid, other=0.0)
        dot = tl.load(incident_dot + lane, mask=valid, other=0.0)
        ox, oy, oz = _chroma_oriented_normal_words(nx, ny, nz, dot)
        tl.store(output + lane * 3, ox, mask=valid)
        tl.store(output + lane * 3 + 1, oy, mask=valid)
        tl.store(output + lane * 3 + 2, oz, mask=valid)

else:
    _normal_orientation_probe = None


def _packed_axis(lower, upper):
    return np.uint32(lower) | (np.uint32(upper) << np.uint32(16))


def _table(value):
    return np.asarray([[400.0, value - 0.1], [500.0, value + 0.1]])


def _material(name, refractive_index, absorption_length, scattering_length):
    return SimpleNamespace(
        name=name,
        refractive_index=_table(refractive_index),
        absorption_length=_table(absorption_length),
        scattering_length=_table(scattering_length),
        comp_reemission_prob=(),
    )


def _surface(name):
    return SimpleNamespace(
        name=name,
        model=0,
        transmissive=0,
        thickness=0.25,
        detect=_table(0.20),
        absorb=_table(0.30),
        reemit=_table(0.0),
        reflect_diffuse=_table(0.10),
        reflect_specular=_table(0.15),
        eta=_table(1.0),
        k=_table(0.0),
        reemission_cdf=_table(0.0),
    )


def _optical_objects():
    return (
        (
            _material("argon", 1.23, 100.0, 200.0),
            _material("glass", 1.50, 50.0, 75.0),
        ),
        (_surface("photocathode"), None),
    )


def _wireplane(materials, surfaces):
    return {
        "origin": np.asarray([2.0, 3.0, 4.0], dtype=np.float64),
        "u": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        "v": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        "pitch": 0.5,
        "radius": 0.025,
        "umin": -5.0,
        "umax": 5.0,
        "vmin": -6.0,
        "vmax": 6.0,
        "v0": 0.125,
        "surface": surfaces[0],
        "material_outer": materials[0],
        "material_inner": materials[1],
        "color": 0x00ABCDEF,
    }


class _Mesh:
    def __init__(self):
        self.vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.triangles = np.asarray(
            [[0, 1, 2], [0, 3, 1]], dtype=np.int32
        )

    def md5(self):
        import hashlib

        digest = hashlib.md5()
        digest.update(memoryview(self.vertices).cast("B"))
        digest.update(memoryview(self.triangles).cast("B"))
        return digest.hexdigest()


def _artifact():
    bound = _packed_axis(0, 11)
    materials, surfaces = _optical_objects()
    nodes = np.asarray(
        [
            [bound, bound, bound, np.uint32((2 << 28) | 1)],
            [bound, bound, bound, np.uint32(0)],
            [bound, bound, bound, np.uint32(1)],
        ],
        dtype=np.uint32,
    )
    detector = SimpleNamespace(
        mesh=_Mesh(),
        bvh=SimpleNamespace(
            nodes=nodes,
            world_coords=SimpleNamespace(
                world_origin=np.zeros(3, dtype=np.float32),
                world_scale=np.float32(0.125),
            ),
            layer_offsets=np.asarray([0, 1], dtype=np.uint32),
        ),
        solid_id=np.asarray([0, 1], dtype=np.uint32),
        material1_index=np.asarray([0, 1], dtype=np.int32),
        material2_index=np.asarray([1, 0], dtype=np.int32),
        surface_index=np.asarray([-1, 0], dtype=np.int32),
        colors=np.asarray([0x10203040, 0x50607080], dtype=np.uint32),
        solid_id_to_channel_index=np.asarray([-1, 0], dtype=np.int32),
        channel_index_to_solid_id=np.asarray([1], dtype=np.int32),
        unique_materials=materials,
        unique_surfaces=surfaces,
        wireplanes=(_wireplane(materials, surfaces),),
    )
    return build_chroma_global_bvh_artifact(detector)


def _eager_sibling_tie_artifact():
    """Two equal hits below sibling internals expose mesh.h's LIFO order."""

    bound = _packed_axis(0, 11)
    materials, surfaces = _optical_objects()
    nodes = np.asarray(
        [
            # mesh.h scans A then B, eagerly pushing both child ranges.  It
            # consequently pops B first and sees triangle 1 before triangle 0.
            [bound, bound, bound, np.uint32((2 << 28) | 1)],
            [bound, bound, bound, np.uint32((1 << 28) | 3)],  # A -> triangle 0
            [bound, bound, bound, np.uint32((1 << 28) | 4)],  # B -> triangle 1
            [bound, bound, bound, np.uint32(0)],
            [bound, bound, bound, np.uint32(1)],
        ],
        dtype=np.uint32,
    )
    detector = SimpleNamespace(
        mesh=_Mesh(),
        bvh=SimpleNamespace(
            nodes=nodes,
            world_coords=SimpleNamespace(
                world_origin=np.zeros(3, dtype=np.float32),
                world_scale=np.float32(0.125),
            ),
            layer_offsets=np.asarray([0, 1, 3], dtype=np.uint32),
        ),
        solid_id=np.asarray([0, 1], dtype=np.uint32),
        material1_index=np.asarray([0, 1], dtype=np.int32),
        material2_index=np.asarray([1, 0], dtype=np.int32),
        surface_index=np.asarray([-1, 0], dtype=np.int32),
        colors=np.asarray([0x10203040, 0x50607080], dtype=np.uint32),
        solid_id_to_channel_index=np.asarray([-1, 0], dtype=np.int32),
        channel_index_to_solid_id=np.asarray([1], dtype=np.int32),
        unique_materials=materials,
        unique_surfaces=surfaces,
        wireplanes=(_wireplane(materials, surfaces),),
    )
    return build_chroma_global_bvh_artifact(detector)


def _shared_edge_rays(count):
    inverse_root_two = np.float32(1.0 / np.sqrt(2.0))
    origins = np.repeat(
        np.asarray([[0.25, -1.0, -1.0]], dtype=np.float32), count, axis=0
    )
    directions = np.repeat(
        np.asarray(
            [[0.0, inverse_root_two, inverse_root_two]], dtype=np.float32
        ),
        count,
        axis=0,
    )
    return origins, directions


def _device_artifact(artifact):
    return ChromaGlobalBVHDevice.from_host(
        artifact,
        expected_traversal_sha256=artifact.traversal_sha256,
        expected_sha256=artifact.sha256,
    )


def _host_words(result):
    return {
        name: np.ascontiguousarray(value.detach().cpu().numpy()).view(np.uint8)
        for name, value in vars(result).items()
    }


def test_global_traversal_import_and_artifact_are_cpu_safe():
    artifact = _artifact()
    assert artifact.stack_capacity == 1
    assert artifact.triangle_count == 2

    eager = _eager_sibling_tie_artifact()
    assert eager.stack_capacity == 2
    assert eager.triangle_count == 2


@pytest.mark.skipif(
    not triton_chroma_global_available(require_cuda=True),
    reason="PyTorch, Triton, and a CUDA device are required",
)
def test_global_traversal_retains_first_leaf_tie_and_global_last_triangle():
    artifact = _artifact()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ChromaGlobalBVHDevice.from_host(
            artifact, expected_traversal_sha256="0" * 64
        )
    accelerator = ChromaGlobalBVHDevice.from_host(
        artifact,
        expected_traversal_sha256=artifact.traversal_sha256,
        expected_sha256=artifact.sha256,
    )
    inverse_root_two = np.float32(1.0 / np.sqrt(2.0))
    origins = np.asarray(
        [[0.25, -1.0, -1.0], [0.25, -1.0, -1.0], [2.0, 2.0, 2.0]],
        dtype=np.float32,
    )
    directions = np.asarray(
        [
            [0.0, inverse_root_two, inverse_root_two],
            [0.0, inverse_root_two, inverse_root_two],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    result = nearest_chroma_global_hit(
        accelerator,
        origins,
        directions,
        last_triangle=np.asarray([-1, 0, -1], dtype=np.int32),
        ray_tile=2,
    )

    np.testing.assert_array_equal(
        result.triangle_ids.cpu().numpy(), [0, 1, -1]
    )
    np.testing.assert_array_equal(result.solid_ids.cpu().numpy(), [0, 1, -1])
    np.testing.assert_array_equal(result.channel_ids.cpu().numpy(), [-1, 0, -1])
    np.testing.assert_array_equal(
        result.surface_indices.cpu().numpy(), [-1, 0, -1]
    )
    np.testing.assert_array_equal(
        result.material_from_indices.cpu().numpy(), [0, 1, -1]
    )
    np.testing.assert_array_equal(
        result.material_to_indices.cpu().numpy(), [1, 0, -1]
    )
    np.testing.assert_array_equal(
        result.inside_to_outside.cpu().numpy(), [1, 1, 0]
    )
    np.testing.assert_array_equal(
        result.raw_normals.cpu().numpy().view(np.uint32),
        np.asarray(
            [[0.0, -0.0, 1.0], [-0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        ).view(np.uint32),
    )
    # fill_state negates the raw normal for inside-to-outside motion, including
    # the otherwise easy-to-lose signed-zero components.
    np.testing.assert_array_equal(
        result.surface_normals.cpu().numpy().view(np.uint32),
        np.asarray(
            [[-0.0, 0.0, -1.0], [0.0, -1.0, -0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        ).view(np.uint32),
    )
    assert np.isfinite(result.distances.cpu().numpy()[:2]).all()
    assert np.isinf(result.distances.cpu().numpy()[2])
    assert not bool(result.overflow.any().item())


@pytest.mark.skipif(
    not triton_chroma_global_available(require_cuda=True),
    reason="PyTorch, Triton, and a CUDA device are required",
)
def test_global_normal_orientation_uses_ordered_cuda_comparison():
    normal = torch.tensor(
        [[1.0, -0.0, 2.0]] * 5, dtype=torch.float32, device="cuda"
    )
    incident_dot = torch.tensor(
        [float("nan"), 0.0, -0.0, 1.0, -1.0],
        dtype=torch.float32,
        device="cuda",
    )
    output = torch.empty_like(normal)
    _normal_orientation_probe[(1,)](
        normal, incident_dot, output, nitems=5, BLOCK_SIZE=8
    )

    # C/CUDA's ordered ``dot > 0.0f`` is false for NaN and both zero signs.
    # Those lanes therefore take fill_state's normal-negation arm.
    expected = np.asarray(
        [
            [-1.0, 0.0, -2.0],
            [-1.0, 0.0, -2.0],
            [-1.0, 0.0, -2.0],
            [1.0, -0.0, 2.0],
            [-1.0, 0.0, -2.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        output.cpu().numpy().view(np.uint32), expected.view(np.uint32)
    )


@pytest.mark.skipif(
    not triton_chroma_global_available(require_cuda=True),
    reason="PyTorch, Triton, and a CUDA device are required",
)
def test_global_traversal_eager_sibling_lifo_controls_equal_hit_tie():
    artifact = _eager_sibling_tie_artifact()
    accelerator = _device_artifact(artifact)
    origins, directions = _shared_edge_rays(2)

    result = nearest_chroma_global_hit(
        accelerator,
        origins,
        directions,
        last_triangle=np.asarray([-1, 1], dtype=np.int32),
        ray_tile=2,
    )

    # Both triangles intersect at the exact same point.  Eagerly pushing A then
    # B makes B's leaf the first visited hit; strict distance comparison keeps
    # it.  Excluding that global triangle falls back to A.
    np.testing.assert_array_equal(result.triangle_ids.cpu().numpy(), [1, 0])
    assert not bool(result.overflow.any().item())


@pytest.mark.skipif(
    not triton_chroma_global_available(require_cuda=True),
    reason="PyTorch, Triton, and a CUDA device are required",
)
def test_global_traversal_is_bitwise_invariant_across_adversarial_tile_sizes():
    artifact = _eager_sibling_tie_artifact()
    accelerator = _device_artifact(artifact)
    count = 67
    origins, directions = _shared_edge_rays(count)
    last_triangle = np.resize(
        np.asarray([-1, 1, 0], dtype=np.int32), count
    )
    expected_triangle = np.where(last_triangle == 1, 0, 1)
    workspace = accelerator.allocate_workspace(33)

    baseline = None
    for ray_tile in (1, 31, 32, 33):
        result = nearest_chroma_global_hit(
            accelerator,
            origins,
            directions,
            last_triangle=last_triangle,
            ray_tile=ray_tile,
            workspace=workspace,
        )
        np.testing.assert_array_equal(
            result.triangle_ids.cpu().numpy(), expected_triangle
        )
        words = _host_words(result)
        if baseline is None:
            baseline = words
        else:
            assert words.keys() == baseline.keys()
            for name in words:
                np.testing.assert_array_equal(words[name], baseline[name])
        assert not workspace.sticky_overflowed()


@pytest.mark.skipif(
    not triton_chroma_global_available(require_cuda=True),
    reason="PyTorch, Triton, and a CUDA device are required",
)
def test_global_traversal_stack_overflow_is_sticky_and_fails_closed():
    artifact = _eager_sibling_tie_artifact()
    accelerator = _device_artifact(artifact)
    # Deliberately bypass the validated topology capacity to exercise the
    # device audit path.  The root's two intersecting internal children need
    # two pending ranges, while this test accelerator permits only one.
    undersized = replace(accelerator, stack_capacity=1)
    workspace = undersized.allocate_workspace(1)
    origins, directions = _shared_edge_rays(1)

    result = nearest_chroma_global_hit(
        undersized,
        origins,
        directions,
        ray_tile=1,
        workspace=workspace,
        check_overflow=False,
    )
    np.testing.assert_array_equal(result.overflow.cpu().numpy(), [1])
    assert workspace.sticky_overflowed()

    workspace.clear_sticky_overflow()
    assert not workspace.sticky_overflowed()
    with pytest.raises(RuntimeError, match="stack overflowed"):
        nearest_chroma_global_hit(
            undersized,
            origins,
            directions,
            ray_tile=1,
            workspace=workspace,
            check_overflow=True,
        )
    assert workspace.sticky_overflowed()
