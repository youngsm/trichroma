"""Integration invariants for the detector-specialized scheduler."""

from types import SimpleNamespace

import numpy as np
import pytest


def test_production_boundary_specialization_routes_strict_modes_unchanged():
    from chroma_lar.triton_backend import (
        _production_branch_specialization_enabled,
        _production_device_scheduler_enabled,
        _production_fused_pmt_enabled,
        _production_fused_portal_boundary_enabled,
        _production_portal_boundary_enabled,
    )

    enabled = _production_branch_specialization_enabled
    assert enabled(
        True,
        use_random_tape=False,
        legacy_specular_reflection=False,
        chroma_mesh_box_compatibility=False,
    )
    for tape, legacy_specular, global_mesh in (
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, True),
    ):
        assert not enabled(
            True,
            use_random_tape=tape,
            legacy_specular_reflection=legacy_specular,
            chroma_mesh_box_compatibility=global_mesh,
        )
    assert not enabled(
        False,
        use_random_tape=False,
        legacy_specular_reflection=False,
        chroma_mesh_box_compatibility=False,
    )
    portal = _production_portal_boundary_enabled
    assert portal(
        True,
        branch_specialized_boundary=True,
        use_random_tape=False,
        legacy_specular_reflection=False,
        chroma_mesh_box_compatibility=False,
    )
    assert not portal(
        True,
        branch_specialized_boundary=False,
        use_random_tape=False,
        legacy_specular_reflection=False,
        chroma_mesh_box_compatibility=False,
    )

    fused = _production_fused_pmt_enabled
    assert fused(
        True,
        use_random_tape=False,
        legacy_specular_reflection=False,
        chroma_mesh_box_compatibility=False,
        pmt_grid=False,
    )
    for changed in (
        "use_random_tape",
        "legacy_specular_reflection",
        "chroma_mesh_box_compatibility",
        "pmt_grid",
    ):
        arguments = dict(
            use_random_tape=False,
            legacy_specular_reflection=False,
            chroma_mesh_box_compatibility=False,
            pmt_grid=False,
        )
        arguments[changed] = True
        assert not fused(True, **arguments)
    assert not fused(
        False,
        use_random_tape=False,
        legacy_specular_reflection=False,
        chroma_mesh_box_compatibility=False,
        pmt_grid=False,
    )
    assert not portal(
        True,
        branch_specialized_boundary=True,
        use_random_tape=True,
        legacy_specular_reflection=False,
        chroma_mesh_box_compatibility=False,
    )

    device_scheduler = _production_device_scheduler_enabled
    assert device_scheduler(
        True,
        use_random_tape=False,
        legacy_specular_reflection=False,
        chroma_mesh_box_compatibility=False,
        branch_specialized_boundary=False,
        portal_boundary=False,
        pmt_grid=False,
    )
    assert device_scheduler(
        True,
        use_random_tape=False,
        legacy_specular_reflection=False,
        chroma_mesh_box_compatibility=False,
        branch_specialized_boundary=False,
        portal_boundary=True,
        pmt_grid=False,
    )
    assert device_scheduler(
        True,
        use_random_tape=False,
        legacy_specular_reflection=False,
        chroma_mesh_box_compatibility=False,
        branch_specialized_boundary=False,
        portal_boundary=False,
        pmt_grid=False,
        fused_pmt=True,
    )
    for changed in (
        "use_random_tape",
        "legacy_specular_reflection",
        "chroma_mesh_box_compatibility",
        "branch_specialized_boundary",
        "pmt_grid",
    ):
        for portal_requested in (False, True):
            arguments = dict(
                use_random_tape=False,
                legacy_specular_reflection=False,
                chroma_mesh_box_compatibility=False,
                branch_specialized_boundary=False,
                portal_boundary=portal_requested,
                pmt_grid=False,
            )
            arguments[changed] = True
            assert not device_scheduler(True, **arguments)
    assert not device_scheduler(
        False,
        use_random_tape=False,
        legacy_specular_reflection=False,
        chroma_mesh_box_compatibility=False,
        branch_specialized_boundary=False,
        portal_boundary=False,
        pmt_grid=False,
        fused_pmt=True,
    )

    fused_portal = _production_fused_portal_boundary_enabled
    fused_portal_arguments = dict(
        device_scheduler=True,
        portal_boundary=True,
        use_random_tape=False,
        legacy_specular_reflection=False,
        chroma_mesh_box_compatibility=False,
        branch_specialized_boundary=False,
        pmt_grid=False,
        fused_pmt=False,
    )
    assert fused_portal(True, **fused_portal_arguments)
    assert not fused_portal(False, **fused_portal_arguments)
    for changed in (
        "device_scheduler",
        "portal_boundary",
        "use_random_tape",
        "legacy_specular_reflection",
        "chroma_mesh_box_compatibility",
        "branch_specialized_boundary",
        "pmt_grid",
    ):
        arguments = dict(fused_portal_arguments)
        arguments[changed] = not arguments[changed]
        assert not fused_portal(True, **arguments)


def test_device_boundary_resolution_routes_only_opt_in_to_fused_pmt(
    monkeypatch,
):
    """The experimental query replaces only the device PMT call site."""

    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation
    from chroma_lar.triton_scene import device_geometry, fused_pmt, instances
    from chroma_lar.triton_scene import intersect as intersect_module

    events = []
    rays = SimpleNamespace(
        origins=object(),
        directions=object(),
        pmt_tmax=object(),
        last_instances=object(),
        last_triangles=object(),
    )
    analytic = SimpleNamespace(distance=object())
    fused_hit = object()
    tlas_hit = object()
    merged = object()

    def gather(*args, **kwargs):
        events.append(("gather", kwargs["launch_capacity"]))
        return rays

    def intersect(*args, **kwargs):
        events.append(("analytic", kwargs["split_wires"]))
        return analytic

    def nextafter(*args, **kwargs):
        events.append(("nextafter", kwargs["launch_capacity"]))

    def fused(*args, **kwargs):
        events.append(
            (
                "fused",
                kwargs["launch_capacity"],
                kwargs["maximum_grid_candidates"],
                kwargs["routing_counters"],
                kwargs["compact_union"],
            )
        )
        return fused_hit

    def tlas(*args, **kwargs):
        events.append(("tlas", kwargs["launch_capacity"]))
        return tlas_hit

    def merge(*args, **kwargs):
        events.append(("merge", args[2], kwargs["launch_capacity"]))
        return merged

    monkeypatch.setattr(
        device_geometry, "gather_boundary_rays_device_count", gather
    )
    monkeypatch.setattr(
        device_geometry, "nextafter_positive_inf_device_count", nextafter
    )
    monkeypatch.setattr(
        device_geometry, "merge_boundaries_device_count", merge
    )
    monkeypatch.setattr(
        intersect_module, "intersect_scene_triton_device_count", intersect
    )
    monkeypatch.setattr(
        fused_pmt, "nearest_pmt_hit_fused_grid_device_count", fused
    )
    monkeypatch.setattr(
        instances, "nearest_pmt_hit_tlas_device_count", tlas
    )

    simulation = SimpleNamespace(
        chroma_mesh_box_compatibility=False,
        fused_pmt=True,
        fused_pmt_max_candidates=81,
        fused_pmt_compact_union=False,
        fused_pmt_routing_counters=None,
        device_boundary_ray_workspace=object(),
        device_boundary_merge_workspace=object(),
        analytic_scene=object(),
        analytic_workspace=SimpleNamespace(outputs=lambda capacity: object()),
        pmt_accelerator=object(),
        pmt_workspace=SimpleNamespace(outputs=lambda capacity: object()),
        scene_device=object(),
        _ensure_device_boundary_workspace=lambda capacity: events.append(
            ("ensure", capacity)
        ),
    )
    state = tuple(object() for _ in range(8))
    queue = SimpleNamespace(count=object())
    result = Reflect3WiresTritonSimulation._resolve_boundaries_device(
        simulation, state, queue, 64
    )
    assert result is merged
    assert ("fused", 64, 81, None, False) in events
    assert not any(event[0] == "tlas" for event in events)
    assert ("merge", fused_hit, 64) in events

    events.clear()
    simulation.fused_pmt = False
    Reflect3WiresTritonSimulation._resolve_boundaries_device(
        simulation, state, queue, 64
    )
    assert ("tlas", 64) in events
    assert not any(event[0] == "fused" for event in events)
    assert ("merge", tlas_hit, 64) in events


def test_device_portal_round_routes_only_fallback_through_geometry(monkeypatch):
    """The host-free portal path must append twice without reading counts."""

    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation
    from chroma_lar.triton_scene import portals

    events = []
    direct_queue = object()
    fallback_queue = object()
    direct_hit = object()
    fallback_hit = object()
    survivor_queue = object()
    final_queue = object()
    workspace = object()
    descriptor = object()

    def ensure_portal_workspace(capacity):
        events.append(("ensure", capacity))

    def partition(*args, **kwargs):
        events.append(
            (
                "partition",
                args[2],
                kwargs["input_capacity"],
                kwargs["launch_capacity"],
                kwargs["workspace"],
            )
        )
        return SimpleNamespace(
            direct=direct_queue,
            fallback=fallback_queue,
            hit=direct_hit,
        )

    def step_boundaries(state, queue, hit, seed, photon_base, max_steps, **kwargs):
        events.append(
            (
                "step",
                queue,
                hit,
                kwargs["input_capacity"],
                kwargs["survivor_queue"],
                kwargs["reset_survivors"],
            )
        )
        return final_queue

    def resolve_boundaries(state, queue, launch_capacity):
        events.append(("geometry", queue, launch_capacity))
        return fallback_hit

    monkeypatch.setattr(portals, "partition_certified_box_portals", partition)
    simulation = SimpleNamespace(
        fused_portal_boundary=False,
        block_size=128,
        portal_descriptor=descriptor,
        portal_workspace=workspace,
        _ensure_portal_workspace=ensure_portal_workspace,
        _step_boundaries=step_boundaries,
        _resolve_boundaries_device=resolve_boundaries,
    )
    state = tuple(object() for _ in range(10))
    input_queue = object()
    result = Reflect3WiresTritonSimulation._step_portal_boundaries_device(
        simulation,
        state,
        input_queue,
        17,
        23,
        1000,
        input_capacity=41,
        launch_capacity=64,
        survivor_queue=survivor_queue,
    )

    assert result is final_queue
    assert events == [
        ("ensure", 64),
        ("partition", input_queue, 41, 64, workspace),
        ("step", direct_queue, direct_hit, 41, survivor_queue, False),
        ("geometry", fallback_queue, 41),
        ("step", fallback_queue, fallback_hit, 41, survivor_queue, False),
    ]


def test_fused_device_portal_round_materializes_only_fallback(monkeypatch):
    """The fused opt-in must replace direct queue/hit traffic with one call."""

    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation
    from chroma_lar.triton_scene import fused_portal_boundary

    events = []
    input_queue = object()
    fallback_queue = object()
    fallback_hit = object()
    survivor_queue = object()
    workspace = object()
    descriptor = object()
    scene_device = object()

    def ensure_workspace(capacity):
        events.append(("ensure_fused", capacity))

    def fused_step(*args, **kwargs):
        events.append(
            (
                "fused",
                args[1],
                args[2],
                args[3],
                args[4],
                kwargs["workspace"],
                kwargs["input_capacity"],
                kwargs["launch_capacity"],
            )
        )
        return fallback_queue

    def resolve_boundaries(state, queue, launch_capacity):
        events.append(("geometry", queue, launch_capacity))
        return fallback_hit

    def step_boundaries(state, queue, hit, seed, photon_base, max_steps, **kwargs):
        events.append(
            (
                "step",
                queue,
                hit,
                kwargs["input_capacity"],
                kwargs["survivor_queue"],
                kwargs["reset_survivors"],
            )
        )
        return survivor_queue

    monkeypatch.setattr(
        fused_portal_boundary, "step_fused_direct_portals", fused_step
    )
    simulation = SimpleNamespace(
        fused_portal_boundary=True,
        block_size=128,
        portal_descriptor=descriptor,
        scene_device=scene_device,
        fused_portal_boundary_workspace=workspace,
        _ensure_fused_portal_boundary_workspace=ensure_workspace,
        _resolve_boundaries_device=resolve_boundaries,
        _step_boundaries=step_boundaries,
    )
    state = tuple(object() for _ in range(10))
    result = Reflect3WiresTritonSimulation._step_portal_boundaries_device(
        simulation,
        state,
        input_queue,
        17,
        23,
        1000,
        input_capacity=41,
        launch_capacity=64,
        survivor_queue=survivor_queue,
    )

    assert result is survivor_queue
    assert events == [
        ("ensure_fused", 64),
        (
            "fused",
            input_queue,
            survivor_queue,
            descriptor,
            scene_device,
            workspace,
            41,
            64,
        ),
        ("geometry", fallback_queue, 41),
        ("step", fallback_queue, fallback_hit, 41, survivor_queue, False),
    ]


def test_empty_region_is_inside_lar_and_clear_of_reachable_wires():
    from chroma_lar.triton_backend import _safe_empty_bounds
    from chroma_lar.triton_scene import compile_reflect3wires_scene

    scene = compile_reflect3wires_scene()
    lower, upper = _safe_empty_bounds(scene)
    assert np.all(lower < upper)
    assert lower[0] > np.max(scene.wires.origin[:, 0] + scene.wires.radius)
    cathode = scene.boxes.kinds.index("cathode")
    assert upper[0] == scene.boxes.bounds_min[cathode, 0]
    active = scene.boxes.kinds.index("active")
    assert lower[1] == scene.boxes.bounds_min[active, 1]
    assert upper[1] == scene.boxes.bounds_max[active, 1]

    exact_scene = compile_reflect3wires_scene(retain_all_wires=True)
    exact_lower, exact_upper = _safe_empty_bounds(exact_scene)
    np.testing.assert_array_equal(exact_lower, lower)
    np.testing.assert_array_equal(exact_upper, upper)
    assert np.any(exact_scene.wires.origin[:, 0] > 0.0)


def test_backend_rejects_physics_outside_specialization():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    simulation = Reflect3WiresTritonSimulation(tile_size=1024)
    with pytest.raises(ValueError, match="wavelength"):
        simulation.simulate(10, (0, 0, 0), wavelength=128.0)


def test_fused_boundary_merge_matches_eager_reference_and_reuses_storage():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from chroma_lar.triton_backend import (
        Reflect3WiresTritonSimulation,
        _merge_boundaries_torch_reference,
    )

    simulation = Reflect3WiresTritonSimulation(tile_size=128)
    device = simulation.device
    count = 4097
    generator = torch.Generator(device=device)
    generator.manual_seed(73019)

    directions = torch.randn(
        (count, 3), generator=generator, device=device, dtype=torch.float32
    )
    directions /= torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    analytic_normal = torch.randn(
        (count, 3), generator=generator, device=device, dtype=torch.float32
    )
    analytic_normal /= torch.linalg.vector_norm(
        analytic_normal, dim=1, keepdim=True
    )
    pmt_normal = torch.randn(
        (count, 3), generator=generator, device=device, dtype=torch.float32
    )
    pmt_normal /= torch.linalg.vector_norm(pmt_normal, dim=1, keepdim=True)

    kind = torch.randint(
        0, 3, (count,), generator=generator, device=device, dtype=torch.int8
    )
    analytic_distance = torch.rand(
        count, generator=generator, device=device, dtype=torch.float32
    ) * 5000.0
    material_count = len(simulation.scene.tables.material_names)
    analytic = SimpleNamespace(
        kind=kind,
        index=torch.randint(
            0, simulation.scene.boxes.count, (count,), generator=generator,
            device=device, dtype=torch.int32,
        ),
        primitive_index=torch.randint(
            0, 6, (count,), generator=generator,
            device=device, dtype=torch.int32,
        ),
        distance=analytic_distance,
        surface_normal=analytic_normal,
        material_from_index=torch.randint(
            0, material_count, (count,), generator=generator,
            device=device, dtype=torch.int32,
        ),
        material_to_index=torch.randint(
            0, material_count, (count,), generator=generator,
            device=device, dtype=torch.int32,
        ),
        surface_index=torch.randint(
            -1, len(simulation.scene.tables.surface_names), (count,),
            generator=generator, device=device, dtype=torch.int32,
        ),
    )

    instance = torch.randint(
        -1, simulation.scene.instances.count, (count,), generator=generator,
        device=device, dtype=torch.int32,
    )
    triangle = torch.randint(
        0, simulation.scene.pmt.triangle_count, (count,), generator=generator,
        device=device, dtype=torch.int32,
    )
    pmt_distance = torch.rand(
        count, generator=generator, device=device, dtype=torch.float32
    ) * 5000.0
    miss = instance < 0
    triangle[miss] = -1
    pmt_distance[miss] = float("inf")
    pmt_normal[miss] = 0.0
    safe_instance = torch.clamp(instance.to(torch.int64), min=0)
    channels = simulation.scene_device["instances_channel_id"][safe_instance]
    channels = torch.where(instance >= 0, channels, -1).to(torch.int32)
    pmt = SimpleNamespace(
        instance_ids=instance,
        triangle_ids=triangle,
        channel_ids=channels,
        distances=pmt_distance,
        world_normals=pmt_normal,
    )

    # Explicitly cover miss, analytic-only, PMT-only, exact mesh-on-tie,
    # strictly closer analytic/PMT winners, raw_dot==0 orientation, and both
    # sides of the historical FP64 1e-12 tie window.
    kind[:9] = torch.tensor(
        [0, 1, 0, 1, 1, 1, 1, 1, 1], dtype=torch.int8, device=device
    )
    instance[:9] = torch.tensor(
        [-1, -1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device=device
    )
    triangle[:9] = torch.tensor(
        [-1, -1, 0, 1, 2, 3, 4, 5, 6], dtype=torch.int32, device=device
    )
    channels[:9] = torch.tensor(
        [-1, -1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device=device
    )
    analytic_distance[:9] = torch.tensor(
        [
            float("inf"), 40.0, float("inf"), 50.0, 49.0, 51.0, 20.0,
            0.0, 0.0,
        ],
        dtype=torch.float32,
        device=device,
    )
    pmt_distance[:9] = torch.tensor(
        [
            float("inf"), float("inf"), 30.0, 50.0, 50.0, 50.0, 10.0,
            5.0e-13, 2.0e-12,
        ],
        dtype=torch.float32,
        device=device,
    )
    pmt_normal[0:2] = 0.0
    pmt_normal[6] = torch.tensor([1.0, 0.0, 0.0], device=device)
    directions[6] = torch.tensor([0.0, 1.0, 0.0], device=device)

    expected = _merge_boundaries_torch_reference(
        simulation.scene_device, analytic, pmt, directions
    )
    actual = simulation._merge_boundaries(analytic, pmt, directions)
    for fused, reference in zip(actual, expected):
        torch.testing.assert_close(fused, reference, rtol=0.0, atol=0.0)
    # Analytic boxes reuse the existing last-hit pair without colliding with
    # non-negative PMT instances: -(box+2), face.  Wires retain (-1,-1).
    assert actual[5][1].item() == -(analytic.index[1].item() + 2)
    assert actual[6][1].item() == analytic.primitive_index[1].item()

    first_pointers = tuple(value.data_ptr() for value in actual)
    smaller_analytic = SimpleNamespace(
        **{name: value[:1000] for name, value in vars(analytic).items()}
    )
    smaller_pmt = SimpleNamespace(
        **{name: value[:1000] for name, value in vars(pmt).items()}
    )
    reused = simulation._merge_boundaries(
        smaller_analytic, smaller_pmt, directions[:1000]
    )
    assert tuple(value.data_ptr() for value in reused) == first_pointers
    assert simulation.merge_workspace.capacity >= count


def test_chroma_global_merge_preserves_mesh_ties_and_wire_epsilon():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from chroma_lar.triton_backend import (
        Reflect3WiresTritonSimulation,
        _BoundaryMergeWorkspace,
    )

    # Construct only the merge owner: this focused test does not need to build
    # or upload either detector acceleration structure.
    simulation = object.__new__(Reflect3WiresTritonSimulation)
    simulation.torch = torch
    simulation.device = torch.device("cuda")
    simulation.merge_workspace = _BoundaryMergeWorkspace.allocate(
        torch, simulation.device, 0
    )
    device = simulation.device
    mesh_distance = torch.tensor(
        [
            10.0,
            1.0000010e-6,
            1.0000020e-6,
            float("inf"),
            4.0,
            float("inf"),
        ],
        dtype=torch.float32,
        device=device,
    )
    wire_distance = torch.tensor(
        [
            10.0,
            1.0000006e-6,
            1.0000000e-6,
            3.0,
            5.0,
            float("inf"),
        ],
        dtype=torch.float32,
        device=device,
    )
    analytic = SimpleNamespace(
        distance=wire_distance,
        kind=torch.tensor([2, 2, 2, 2, 2, 0], dtype=torch.int8, device=device),
        surface_normal=torch.tensor(
            [
                [-0.0, 1.0, 0.0],
                [-0.0, 1.0, 0.0],
                [-0.0, 1.0, 0.0],
                [-0.0, 1.0, 0.0],
                [-0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        ),
        material_from_index=torch.full(
            (6,), 7, dtype=torch.int32, device=device
        ),
        material_to_index=torch.full(
            (6,), 8, dtype=torch.int32, device=device
        ),
        surface_index=torch.full(
            (6,), 9, dtype=torch.int32, device=device
        ),
    )
    mesh = SimpleNamespace(
        distances=mesh_distance,
        surface_normals=torch.tensor(
            [
                [1.0, -0.0, 0.0],
                [1.0, -0.0, 0.0],
                [1.0, -0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, -0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        ),
        material_from_indices=torch.full(
            (6,), 1, dtype=torch.int32, device=device
        ),
        material_to_indices=torch.full(
            (6,), 2, dtype=torch.int32, device=device
        ),
        surface_indices=torch.full(
            (6,), 3, dtype=torch.int32, device=device
        ),
        triangle_ids=torch.tensor(
            [100, 101, 102, -1, 104, -1], dtype=torch.int32, device=device
        ),
        channel_ids=torch.tensor(
            [10, 11, 12, -1, 14, -1], dtype=torch.int32, device=device
        ),
    )

    merged = simulation._merge_chroma_global_boundaries(analytic, mesh)
    distance, normal, material_from, material_to, surface, instance, triangle, channel = (
        value.cpu().numpy() for value in merged
    )
    # Exact tie, within-1e-12 wire advantage, strict wire advantage, mesh
    # miss, wire behind mesh, and double miss respectively.
    np.testing.assert_array_equal(triangle, [100, 101, -2, -2, 104, -1])
    np.testing.assert_array_equal(instance, [-1, -1, -1, -1, -1, -1])
    np.testing.assert_array_equal(channel, [10, 11, -1, -1, 14, -1])
    np.testing.assert_array_equal(material_from, [1, 1, 7, 7, 1, -1])
    np.testing.assert_array_equal(material_to, [2, 2, 8, 8, 2, -1])
    np.testing.assert_array_equal(surface, [3, 3, 9, 9, 3, -1])
    selected_distance = np.asarray(
        [
            10.0,
            1.0000010e-6,
            1.0000000e-6,
            3.0,
            4.0,
            float("inf"),
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        distance.view(np.uint32), selected_distance.view(np.uint32)
    )
    expected_normal = np.asarray(
        [
            [1.0, -0.0, 0.0],
            [1.0, -0.0, 0.0],
            [-0.0, 1.0, 0.0],
            [-0.0, 1.0, 0.0],
            [1.0, -0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        normal.view(np.uint32), expected_normal.view(np.uint32)
    )


def test_small_end_to_end_run_has_valid_flat_hits():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    simulation = Reflect3WiresTritonSimulation(
        tile_size=16384, history_length=4
    )
    result = simulation.simulate(
        16384, (-1000.0, 0.0, 0.0), voxel_size=30.0, seed=8123
    )
    times, channels = result.flat_hits.to_numpy()
    assert times.shape == channels.shape
    assert np.all(np.isfinite(times))
    assert np.all(times >= 0.0)
    assert np.all((channels >= 0) & (channels < 81))
    assert result.stats.photons == 16384
    assert result.stats.elapsed_seconds > 0.0


def test_branch_specialized_boundary_matches_monolithic_first_decisions():
    """The queue split preserves decisions and branch-vector accuracy.

    Moving diffuse arithmetic into a smaller launch changes a handful of
    Triton FMA choices by a few ULPs.  Stop after the first two interactions so
    those local roundoff differences are tested directly instead of after
    chaotic amplification by later geometry queries.
    """

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    common = dict(
        tile_size=4096,
        history_length=4,
        reservoir_rounds=None,
    )
    monolithic = Reflect3WiresTritonSimulation(
        **common, branch_specialized_boundary=False
    )
    specialized = Reflect3WiresTritonSimulation(
        **common, branch_specialized_boundary=True
    )
    arguments = dict(
        nphotons=4096,
        center=(-1000.0, 0.0, 0.0),
        voxel_size=30.0,
        seed=8123,
        max_steps=2,
        keep_final_states=True,
    )
    reference = monolithic.simulate(**arguments)
    actual = specialized.simulate(**arguments)
    assert len(reference.final_states) == len(actual.final_states) == 1
    split_state = actual.final_states[0]
    reference_state = reference.final_states[0]
    # Positions/times and all discrete process state remain bitwise exact.
    for index in (0, 3, 4, 5, 6, 7, 8, 9):
        torch.testing.assert_close(
            split_state[index], reference_state[index],
            rtol=0.0, atol=0.0, equal_nan=True,
        )
    # Direction and polarization differ only for one diffuse interaction in
    # this population, by at most a few float32 ULPs.
    for index in (1, 2):
        torch.testing.assert_close(
            split_state[index], reference_state[index],
            rtol=8.0e-7, atol=2.0e-7, equal_nan=True,
        )
    assert torch.any((split_state[4] & (1 << 5)) != 0)
    reference_times, reference_channels = reference.flat_hits.to_numpy()
    actual_times, actual_channels = actual.flat_hits.to_numpy()
    np.testing.assert_array_equal(
        actual_times.view(np.uint32), reference_times.view(np.uint32)
    )
    np.testing.assert_array_equal(actual_channels, reference_channels)
    assert actual.stats.boundary_events == reference.stats.boundary_events


def test_monolithic_boundary_consumes_device_count_and_appends_survivors():
    """Capacity-sized scheduling neither reads nor executes the stale suffix."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from chroma.triton.transport import DeviceQueue
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    capacity = 1024
    queue_storage = 2 * capacity
    live_count = 37
    sentinel_state = live_count
    simulation = Reflect3WiresTritonSimulation(
        tile_size=capacity,
        branch_specialized_boundary=False,
    )
    device = simulation.device

    def make_state():
        positions = torch.zeros(
            (capacity, 3), dtype=torch.float32, device=device
        )
        directions = torch.zeros_like(positions)
        directions[:, 0] = 1.0
        polarizations = torch.zeros_like(positions)
        polarizations[:, 1] = 1.0
        return (
            positions,
            directions,
            polarizations,
            torch.zeros(capacity, dtype=torch.float32, device=device),
            torch.zeros(capacity, dtype=torch.int32, device=device),
            torch.zeros(capacity, dtype=torch.int64, device=device),
            torch.full(
                (capacity,), -1, dtype=torch.int32, device=device
            ),
            torch.full(
                (capacity,), -1, dtype=torch.int32, device=device
            ),
            torch.full(
                (capacity,), -1, dtype=torch.int32, device=device
            ),
            torch.zeros(capacity, dtype=torch.int32, device=device),
        )

    distances = torch.zeros(capacity, dtype=torch.float32, device=device)
    normals = torch.zeros((capacity, 3), dtype=torch.float32, device=device)
    normals[:, 0] = -1.0
    # Vacuum on both sides suppresses bulk work at a zero-distance glossy
    # surface.  Every valid lane survives through either diffuse or specular
    # reflection, while the suffix contains otherwise valid, stale metadata.
    hit = (
        distances,
        normals,
        torch.ones(capacity, dtype=torch.int32, device=device),
        torch.ones(capacity, dtype=torch.int32, device=device),
        torch.full(
            (capacity,), 2, dtype=torch.int32, device=device
        ),
        torch.full(
            (capacity,), -1, dtype=torch.int32, device=device
        ),
        torch.full(
            (capacity,), -1, dtype=torch.int32, device=device
        ),
        torch.full(
            (capacity,), -1, dtype=torch.int32, device=device
        ),
    )

    reference_state = make_state()
    reference = simulation._step_boundaries(
        reference_state,
        torch.arange(live_count, dtype=torch.int32, device=device),
        tuple(value[:live_count] for value in hit),
        seed=991,
        photon_id_base=0,
        max_steps=100,
    ).tensor()

    actual_state = make_state()
    input_buffer = torch.full(
        (queue_storage,), sentinel_state, dtype=torch.int32, device=device
    )
    input_buffer[:live_count] = torch.arange(
        live_count, dtype=torch.int32, device=device
    )
    input_queue = DeviceQueue(
        input_buffer,
        torch.tensor([live_count], dtype=torch.int32, device=device),
    )
    output = DeviceQueue.allocate(capacity + 1, device=device)
    output.buffer[0] = 777
    output.count.fill_(1)
    too_short_hit = (distances[:-1],) + hit[1:]
    with pytest.raises(ValueError, match="hit distance"):
        simulation._step_boundaries(
            actual_state,
            input_queue,
            too_short_hit,
            seed=991,
            photon_id_base=0,
            max_steps=100,
            input_capacity=capacity,
            survivor_queue=output,
            reset_survivors=False,
        )
    with pytest.raises(ValueError, match="must not alias"):
        simulation._step_boundaries(
            actual_state,
            input_queue,
            hit,
            seed=991,
            photon_id_base=0,
            max_steps=100,
            input_capacity=capacity,
            survivor_queue=DeviceQueue(input_buffer, output.count),
            reset_survivors=False,
        )
    returned = simulation._step_boundaries(
        actual_state,
        input_queue,
        hit,
        seed=991,
        photon_id_base=0,
        max_steps=100,
        input_capacity=capacity,
        survivor_queue=output,
        reset_survivors=False,
    )
    assert returned is output
    appended = output.tensor()
    assert appended[0].item() == 777
    torch.testing.assert_close(
        appended[1:], reference, rtol=0.0, atol=0.0
    )
    for actual, expected in zip(actual_state, reference_state):
        torch.testing.assert_close(
            actual, expected, rtol=0.0, atol=0.0, equal_nan=True
        )
    # The repeated valid ID in every unused input slot would have changed if
    # either a partial-CTA mask or the uniform suffix-CTA return were missing.
    assert actual_state[5][sentinel_state].item() == 0
    assert actual_state[9][sentinel_state].item() == 0

    # A zero live count still launches the capacity grid, preserves a prior
    # appended prefix, and performs no state writes.
    snapshot = tuple(value.clone() for value in actual_state)
    input_queue.count.zero_()
    simulation._step_boundaries(
        actual_state,
        input_queue,
        hit,
        seed=991,
        photon_id_base=0,
        max_steps=100,
        input_capacity=capacity,
        survivor_queue=output,
        reset_survivors=False,
    )
    assert output.size() == appended.numel()
    for actual, expected in zip(actual_state, snapshot):
        torch.testing.assert_close(
            actual, expected, rtol=0.0, atol=0.0, equal_nan=True
        )


def test_portal_boundaries_match_general_scene_first_decision_bitwise():
    """Known-wall routing changes neither hit metadata nor production state."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    common = dict(
        tile_size=4096,
        history_length=4,
        reservoir_rounds=None,
        branch_specialized_boundary=True,
    )
    general = Reflect3WiresTritonSimulation(**common, portal_boundary=False)
    portal = Reflect3WiresTritonSimulation(**common, portal_boundary=True)
    arguments = dict(
        nphotons=4096,
        center=(-1000.0, 0.0, 0.0),
        voxel_size=30.0,
        seed=8123,
        max_steps=1,
        keep_final_states=True,
    )
    reference = general.simulate(**arguments)
    actual = portal.simulate(**arguments)
    assert portal.last_portal_counts[0] > 0
    assert sum(portal.last_portal_counts) == actual.stats.boundary_events
    for actual_value, reference_value in zip(
        actual.final_states[0], reference.final_states[0]
    ):
        torch.testing.assert_close(
            actual_value, reference_value, rtol=0.0, atol=0.0,
            equal_nan=True,
        )
    reference_times, reference_channels = reference.flat_hits.to_numpy()
    actual_times, actual_channels = actual.flat_hits.to_numpy()
    np.testing.assert_array_equal(
        actual_times.view(np.uint32), reference_times.view(np.uint32)
    )
    np.testing.assert_array_equal(actual_channels, reference_channels)
    assert actual.stats.boundary_events == reference.stats.boundary_events


def test_device_scheduler_matches_synchronized_full_state_bitwise():
    """Changing inter-photon scheduling must not change any photon word."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    count = 8192
    common = dict(
        tile_size=count,
        history_length=8,
        block_size=128,
        reservoir_rounds=None,
        history_epochs_per_poll=2,
        branch_specialized_boundary=False,
        portal_boundary=False,
        pmt_grid=False,
    )
    synchronized = Reflect3WiresTritonSimulation(
        **common, device_scheduler=False
    )
    asynchronous = Reflect3WiresTritonSimulation(
        **common, device_scheduler=True, device_round_batch=8
    )
    arguments = dict(
        nphotons=count,
        center=(-1000.0, 0.0, 0.0),
        voxel_size=30.0,
        seed=8123,
        max_steps=1000,
        keep_final_states=True,
    )
    expected = synchronized.simulate(**arguments)
    actual = asynchronous.simulate(**arguments)

    assert expected.stats.detections == actual.stats.detections
    assert expected.stats.boundary_events == actual.stats.boundary_events
    assert asynchronous.last_device_scheduler_syncs > 0
    assert (
        asynchronous.last_device_scheduler_syncs
        < actual.stats.boundary_rounds
    )
    assert len(expected.final_states) == len(actual.final_states) == 1
    for expected_field, actual_field in zip(
        expected.final_states[0], actual.final_states[0]
    ):
        assert torch.equal(expected_field, actual_field)
    assert torch.equal(expected.flat_hits.time, actual.flat_hits.time)
    assert torch.equal(expected.flat_hits.channel, actual.flat_hits.channel)


def test_fused_pmt_schedulers_match_baseline_full_state_bitwise():
    """Both fused integrations must preserve every word through termination."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    count = 8192
    common = dict(
        tile_size=count,
        history_length=8,
        block_size=128,
        reservoir_rounds=None,
        history_epochs_per_poll=2,
        branch_specialized_boundary=False,
        portal_boundary=False,
        pmt_grid=False,
    )
    baseline = Reflect3WiresTritonSimulation(
        **common, fused_pmt=False, device_scheduler=False
    )
    synchronized_fused = Reflect3WiresTritonSimulation(
        **common,
        fused_pmt=True,
        fused_pmt_compact_union=False,
        device_scheduler=False,
    )
    device_fused = Reflect3WiresTritonSimulation(
        **common,
        fused_pmt=True,
        fused_pmt_compact_union=False,
        device_scheduler=True,
        device_round_batch=8,
    )
    arguments = dict(
        nphotons=count,
        center=(-1000.0, 0.0, 0.0),
        voxel_size=30.0,
        seed=8123,
        max_steps=1000,
        keep_final_states=True,
    )
    expected = baseline.simulate(**arguments)
    for simulation in (synchronized_fused, device_fused):
        actual = simulation.simulate(**arguments)
        assert expected.stats.detections == actual.stats.detections
        assert expected.stats.boundary_rounds == actual.stats.boundary_rounds
        assert expected.stats.boundary_events == actual.stats.boundary_events
        assert len(expected.final_states) == len(actual.final_states) == 1
        for expected_field, actual_field in zip(
            expected.final_states[0], actual.final_states[0]
        ):
            assert torch.equal(expected_field, actual_field)
        assert torch.equal(expected.flat_hits.time, actual.flat_hits.time)
        assert torch.equal(expected.flat_hits.channel, actual.flat_hits.channel)


def test_device_scheduler_portal_matches_synchronized_full_state_bitwise():
    """Certified direct faces must be invisible to every final-state word."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    count = 8192
    common = dict(
        tile_size=count,
        history_length=8,
        block_size=128,
        reservoir_rounds=None,
        history_epochs_per_poll=2,
        branch_specialized_boundary=False,
        pmt_grid=False,
    )
    synchronized = Reflect3WiresTritonSimulation(
        **common,
        portal_boundary=False,
        device_scheduler=False,
    )
    asynchronous_portal = Reflect3WiresTritonSimulation(
        **common,
        portal_boundary=True,
        device_scheduler=True,
        device_round_batch=8,
    )
    arguments = dict(
        nphotons=count,
        center=(-1000.0, 0.0, 0.0),
        voxel_size=30.0,
        seed=8123,
        max_steps=1000,
        keep_final_states=True,
    )
    expected = synchronized.simulate(**arguments)
    actual = asynchronous_portal.simulate(**arguments)

    assert asynchronous_portal.portal_workspace.capacity >= count
    assert expected.stats.detections == actual.stats.detections
    assert expected.stats.boundary_rounds == actual.stats.boundary_rounds
    assert expected.stats.boundary_events == actual.stats.boundary_events
    assert asynchronous_portal.last_device_scheduler_syncs > 0
    assert (
        asynchronous_portal.last_device_scheduler_syncs
        < actual.stats.boundary_rounds
    )
    assert len(expected.final_states) == len(actual.final_states) == 1
    for expected_field, actual_field in zip(
        expected.final_states[0], actual.final_states[0]
    ):
        assert torch.equal(expected_field, actual_field)
    assert torch.equal(expected.flat_hits.time, actual.flat_hits.time)
    assert torch.equal(expected.flat_hits.channel, actual.flat_hits.channel)


def test_device_scheduler_fused_portal_matches_existing_portal_bitwise():
    """Fusing direct physics must change neither state nor hit words."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    count = 8192
    common = dict(
        tile_size=count,
        history_length=8,
        block_size=128,
        reservoir_rounds=None,
        history_epochs_per_poll=2,
        branch_specialized_boundary=False,
        portal_boundary=True,
        pmt_grid=False,
        device_scheduler=True,
        device_round_batch=8,
    )
    materialized = Reflect3WiresTritonSimulation(
        **common, fused_portal_boundary=False
    )
    fused = Reflect3WiresTritonSimulation(
        **common, fused_portal_boundary=True
    )
    arguments = dict(
        nphotons=count,
        center=(-1000.0, 0.0, 0.0),
        voxel_size=30.0,
        seed=8123,
        max_steps=1000,
        keep_final_states=True,
    )
    expected = materialized.simulate(**arguments)
    actual = fused.simulate(**arguments)

    assert fused.fused_portal_boundary_workspace.capacity >= count
    assert expected.stats.detections == actual.stats.detections
    assert expected.stats.boundary_rounds == actual.stats.boundary_rounds
    assert expected.stats.boundary_events == actual.stats.boundary_events
    assert len(expected.final_states) == len(actual.final_states) == 1
    for expected_field, actual_field in zip(
        expected.final_states[0], actual.final_states[0]
    ):
        assert torch.equal(expected_field, actual_field)
    assert torch.equal(expected.flat_hits.time, actual.flat_hits.time)
    assert torch.equal(expected.flat_hits.channel, actual.flat_hits.channel)
