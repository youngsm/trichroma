"""Correctness tests for detector-specialized analytic intersections."""

import numpy as np
import pytest

from chroma_lar.triton_scene import compile_reflect3wires_scene
from chroma_lar.triton_scene.intersect import (
    BoundaryIntersections,
    GeometryKind,
    WIRE_T_MIN,
    allocate_split_intersection_workspace,
    intersect_boxes_numpy,
    intersect_scene_numpy,
    intersect_scene_triton,
    intersect_scene_triton_split,
    intersect_wires_bruteforce_numpy,
    intersect_wires_numpy,
    prepare_scene_triton,
)


@pytest.fixture(scope="module")
def scene():
    return compile_reflect3wires_scene()


def test_macro_faces_and_fill_state_classification(scene):
    origins = np.tile(np.asarray([-1000.0, 0.0, 0.0], np.float32), (4, 1))
    directions = np.asarray(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float32,
    )
    result = intersect_boxes_numpy(scene.boxes, origins, directions)

    np.testing.assert_array_equal(result.kind, int(GeometryKind.BOX))
    # cathode -x, active -x, active +y, active -z
    np.testing.assert_array_equal(result.index, [2, 1, 1, 1])
    np.testing.assert_array_equal(result.primitive_index, [0, 0, 3, 4])
    np.testing.assert_allclose(
        result.distance,
        [997.0, 1295.4148, 2160.0, 2160.0],
        rtol=0.0,
        atol=2.0e-4,
    )
    # fill_state always gives optical physics a normal facing the incoming ray.
    assert np.all(np.sum(result.surface_normal * directions, axis=1) <= 0.0)
    np.testing.assert_array_equal(
        result.inside_to_outside, [False, True, True, True]
    )
    np.testing.assert_array_equal(result.material_from_index, 0)  # liquid argon


def test_box_tmax_is_exclusive_and_starting_face_is_not_a_self_hit(scene):
    origin = np.asarray([[-1000.0, 0.0, 0.0]], np.float32)
    direction = np.asarray([[1.0, 0.0, 0.0]], np.float32)
    exact = intersect_boxes_numpy(scene.boxes, origin, direction)
    capped = intersect_boxes_numpy(scene.boxes, origin, direction, exact.distance)
    assert not capped.hit[0]

    active = scene.boxes.kinds.index("active")
    on_wall = np.asarray(
        [[scene.boxes.bounds_min[active, 0], 0.0, 0.0]], dtype=np.float32
    )
    outward = np.asarray([[-1.0, 0.0, 0.0]], np.float32)
    self_query = intersect_boxes_numpy(scene.boxes, on_wall, outward, tmax=1.0)
    assert not self_query.hit[0]


def test_periodic_wire_entry_exit_and_world_normals(scene):
    origins = np.asarray(
        [
            [-2150.0, 0.0, 0.0],  # outside, enter the closest -x wire plane
            [-2154.0, 0.0, 0.0],  # center of its k=0 cylinder, exit
        ],
        dtype=np.float32,
    )
    directions = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32)
    result = intersect_wires_numpy(scene.wires, origins, directions)

    np.testing.assert_array_equal(result.kind, int(GeometryKind.WIRE))
    np.testing.assert_array_equal(result.index, [2, 2])
    np.testing.assert_array_equal(result.primitive_index, [0, 0])
    np.testing.assert_allclose(result.distance, [3.925, 0.075], atol=2.0e-5)
    np.testing.assert_allclose(result.outward_normal, [[1, 0, 0], [1, 0, 0]], atol=1e-7)
    np.testing.assert_array_equal(result.inside_to_outside, [False, True])
    np.testing.assert_allclose(result.surface_normal, [[1, 0, 0], [-1, 0, 0]])


def test_wire_boundary_uses_chroma_point_one_micron_step(scene):
    # Plane zero has v=+Y.  Putting the offset in Y avoids the coarse float32
    # ulp of the -2160 mm X coordinate and constructs CUDA's on-surface case.
    radius32 = np.float32(scene.wires.radius[0])
    origin = np.asarray([[-2160.0, radius32, 0.0]], dtype=np.float32)
    direction = np.asarray([[0.0, -1.0, 0.0]], dtype=np.float32)
    result = intersect_wires_numpy(scene.wires, origin, direction)
    assert result.hit[0]
    assert result.index[0] == 0
    assert result.distance[0] == pytest.approx(WIRE_T_MIN, abs=1e-9)


def test_axial_on_cylinder_matches_chroma_observable_no_wire(scene):
    """A later NaN lattice candidate makes CUDA reject the analytic hit."""

    radius32 = np.float32(scene.wires.radius[0])
    origins = np.asarray(
        [[scene.wires.origin[0, 0], radius32, 0.0]], dtype=np.float32
    )
    directions = np.asarray([scene.wires.raw_u[0]], dtype=np.float32)
    assert not intersect_wires_numpy(scene.wires, origins, directions).hit[0]

    try:
        import torch
        if not torch.cuda.is_available():
            return
        pytest.importorskip("triton")
    except ImportError:
        return
    prepared = prepare_scene_triton(scene, device="cuda")
    result = intersect_scene_triton(
        prepared,
        torch.from_numpy(origins).cuda(),
        torch.from_numpy(directions).cuda(),
        chroma_wire_frame=True,
        chroma_wire_full_scan=True,
        _box_count_override=0,
    )
    torch.cuda.synchronize()
    assert not result.hit.item()




def _stress_rays(scene, seed=901, n_volume=180, n_per_plane=40):
    rng = np.random.default_rng(seed)
    bounds_lo = scene.reachability.source_component_bounds_min
    bounds_hi = scene.reachability.source_component_bounds_max
    origins = rng.uniform(bounds_lo + 2.0, bounds_hi - 2.0, (n_volume, 3)).astype(
        np.float32
    )
    directions = rng.normal(size=(n_volume, 3)).astype(np.float32)

    # Add normal, grazing, and nearly coplanar rays immediately around every
    # wire slab; this is where a non-conservative lattice cull would fail.
    near_origins = []
    near_directions = []
    for x in scene.wires.origin[:, 0]:
        yz = rng.uniform(-2100.0, 2100.0, (n_per_plane, 2))
        near_origins.append(
            np.column_stack(
                (
                    np.full(n_per_plane, x) + rng.uniform(-0.2, 0.2, n_per_plane),
                    yz,
                )
            )
        )
        d = rng.normal(size=(n_per_plane, 3))
        d[: n_per_plane // 3, 0] *= 1.0e-10
        d[n_per_plane // 3 : 2 * n_per_plane // 3, 0] *= 1.0e-5
        near_directions.append(d)
    origins = np.vstack((origins, *near_origins)).astype(np.float32)
    directions = np.vstack((directions, *near_directions)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return origins, directions


def test_conservative_wire_cull_matches_all_cylinders_bruteforce(scene):
    origins, directions = _stress_rays(scene)
    rng = np.random.default_rng(1776)
    cap = rng.uniform(0.01, 7000.0, len(origins)).astype(np.float32)
    optimized = intersect_wires_numpy(scene.wires, origins, directions, cap)
    brute = intersect_wires_bruteforce_numpy(scene.wires, origins, directions, cap)

    for field in (
        "kind",
        "index",
        "primitive_index",
        "surface_index",
        "material_inner_index",
        "material_outer_index",
        "material_from_index",
        "material_to_index",
        "inside_to_outside",
    ):
        np.testing.assert_array_equal(getattr(optimized, field), getattr(brute, field))
    np.testing.assert_array_equal(optimized.distance, brute.distance)
    np.testing.assert_array_equal(optimized.outward_normal, brute.outward_normal)
    np.testing.assert_array_equal(optimized.surface_normal, brute.surface_normal)


def test_scene_query_selects_nearest_wire_before_macro_wall(scene):
    origin = np.asarray([[-1000.0, 0.0, 0.0]], np.float32)
    direction = np.asarray([[-1.0, 0.0, 0.0]], np.float32)
    box = intersect_boxes_numpy(scene.boxes, origin, direction)
    combined = intersect_scene_numpy(scene, origin, direction)
    assert box.kind[0] == int(GeometryKind.BOX)
    assert combined.kind[0] == int(GeometryKind.WIRE)
    assert combined.distance[0] < box.distance[0]
    assert not intersect_scene_numpy(scene, origin, direction, tmax=1000.0).hit[0]


def test_cavity_catches_float32_escape_from_coplanar_active_wall(scene):
    """Match the full BVH when the active wall is at triangle epsilon."""

    active = scene.boxes.kinds.index("active")
    cavity = scene.boxes.kinds.index("cavity")
    origin = np.asarray(
        [[scene.boxes.bounds_min[active, 0], 500.0, -900.0]], np.float32
    )
    direction = np.asarray([[-1.0, 0.0, 0.0]], np.float32)
    result = intersect_scene_numpy(scene, origin, direction)
    assert result.kind[0] == int(GeometryKind.BOX)
    assert result.index[0] == cavity
    assert result.primitive_index[0] == 0
    expected = (
        scene.boxes.bounds_min[active, 0]
        - scene.boxes.bounds_min[cavity, 0]
    )
    assert result.distance[0] == pytest.approx(expected, abs=5.0e-4)
    reflect00 = scene.tables.surface_names.index("reflect00")
    assert result.surface_index[0] == reflect00
    assert scene.tables.surface_absorb[reflect00] == np.float32(1.0)


def test_triton_matches_numpy_on_a100_when_available(scene):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not visible in this test process")
    pytest.importorskip("triton")

    origins, directions = _stress_rays(
        scene, seed=31415, n_volume=10_000, n_per_plane=160
    )
    rng = np.random.default_rng(2718)
    cap = rng.uniform(0.01, 7500.0, len(origins)).astype(np.float32)
    reference = intersect_scene_numpy(scene, origins, directions, cap)

    prepared = prepare_scene_triton(scene, device="cuda")
    gpu = intersect_scene_triton(
        prepared,
        torch.from_numpy(origins).cuda(),
        torch.from_numpy(directions).cuda(),
        torch.from_numpy(cap).cuda(),
    )
    torch.cuda.synchronize()

    for field in (
        "kind",
        "index",
        "primitive_index",
        "surface_index",
        "material_inner_index",
        "material_outer_index",
        "material_from_index",
        "material_to_index",
        "inside_to_outside",
    ):
        np.testing.assert_array_equal(
            getattr(gpu, field).cpu().numpy(), getattr(reference, field)
        )
    np.testing.assert_allclose(
        # Box division may fuse differently on host and device; observed
        # disagreement is at most two float32 ulps at detector-scale lengths.
        gpu.distance.cpu().numpy(), reference.distance, rtol=0.0, atol=5e-4
    )
    np.testing.assert_allclose(
        gpu.outward_normal.cpu().numpy(), reference.outward_normal, rtol=0.0, atol=3e-5
    )
    np.testing.assert_allclose(
        gpu.surface_normal.cpu().numpy(), reference.surface_normal, rtol=0.0, atol=3e-5
    )


def test_split_triton_is_bit_exact_to_fused_and_compacts_work_on_a100(scene):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not visible in this test process")
    pytest.importorskip("triton")

    origins, directions = _stress_rays(
        scene, seed=8675309, n_volume=50_000, n_per_plane=160
    )
    rng = np.random.default_rng(123456)
    cap = rng.uniform(0.01, 8000.0, len(origins)).astype(np.float32)
    prepared = prepare_scene_triton(scene, device="cuda")
    origins_gpu = torch.from_numpy(origins).cuda()
    directions_gpu = torch.from_numpy(directions).cuda()
    cap_gpu = torch.from_numpy(cap).cuda()
    workspace = allocate_split_intersection_workspace(len(origins), "cuda")
    reusable = workspace.outputs(len(origins))
    reusable_pointers = {
        field: getattr(reusable, field).data_ptr()
        for field in BoundaryIntersections.__dataclass_fields__
    }

    fused = intersect_scene_triton(
        prepared, origins_gpu, directions_gpu, cap_gpu
    )
    split = intersect_scene_triton_split(
        prepared,
        origins_gpu,
        directions_gpu,
        cap_gpu,
        workspace=workspace,
        out=reusable,
    )
    torch.cuda.synchronize()
    assert split is reusable

    for field in BoundaryIntersections.__dataclass_fields__:
        fused_value = getattr(fused, field)
        split_value = getattr(split, field)
        assert torch.equal(split_value, fused_value), field
        assert split_value.data_ptr() == reusable_pointers[field]
    # Locks in the intended collision-first workload reduction without making
    # wall-clock timing part of the correctness suite.
    assert 0 < workspace.candidate_count.item() < 0.45 * len(origins)


def test_triton_suppresses_only_the_previous_macro_face(scene):
    """Reflected box hits reproduce Chroma's last-triangle suppression."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not visible in this test process")
    pytest.importorskip("triton")

    active = scene.boxes.kinds.index("active")
    cathode = scene.boxes.kinds.index("cathode")
    # A 0.3 micron overshoot is representative of the float32 boundary-point
    # error observed in full transport and exceeds Chroma's triangle epsilon.
    origins = np.asarray(
        [[-2.9997, 0.0, 0.0], [-1000.0, 0.0, 2160.0003]],
        dtype=np.float32,
    )
    directions = np.asarray(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32
    )
    origins_gpu = torch.from_numpy(origins).cuda()
    directions_gpu = torch.from_numpy(directions).cuda()
    prepared = prepare_scene_triton(scene, "cuda")

    unsuppressed = intersect_scene_triton(
        prepared, origins_gpu, directions_gpu
    )
    np.testing.assert_array_equal(
        unsuppressed.index.cpu().numpy(), [cathode, active]
    )
    np.testing.assert_array_equal(
        unsuppressed.primitive_index.cpu().numpy(), [0, 5]
    )
    assert torch.all(unsuppressed.distance < 1.0e-3)

    previous_instance = torch.tensor(
        [-(cathode + 2), -(active + 2)], dtype=torch.int32, device="cuda"
    )
    previous_face = torch.tensor([0, 5], dtype=torch.int32, device="cuda")
    suppressed = intersect_scene_triton_split(
        prepared,
        origins_gpu,
        directions_gpu,
        last_instance=previous_instance,
        last_triangle=previous_face,
    )
    assert torch.all(suppressed.distance > 1000.0)
    same_face = (
        (suppressed.kind == int(GeometryKind.BOX))
        & (suppressed.index == torch.tensor([cathode, active], device="cuda"))
        & (suppressed.primitive_index == previous_face)
    )
    assert not torch.any(same_face)

    # Non-negative PMT instances retain their old namespace and do not
    # suppress an unrelated macro face.
    pmt_instance = torch.zeros(2, dtype=torch.int32, device="cuda")
    pmt_triangle = torch.zeros(2, dtype=torch.int32, device="cuda")
    with_pmt_previous = intersect_scene_triton(
        prepared,
        origins_gpu,
        directions_gpu,
        last_instance=pmt_instance,
        last_triangle=pmt_triangle,
    )
    torch.testing.assert_close(
        with_pmt_previous.distance, unsuppressed.distance, rtol=0.0, atol=0.0
    )
    assert torch.equal(with_pmt_previous.index, unsuppressed.index)
