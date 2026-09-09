"""Scheduling and lazy-workspace invariants for the Triton backend."""

import inspect

import pytest


def test_default_tile_policy_scales_with_available_memory():
    from chroma_lar.triton_backend import (
        AUTO_TILE_BYTES_PER_PHOTON,
        AUTO_TILE_RESERVE_BYTES,
        AUTO_TILE_USABLE_FRACTION,
        Reflect3WiresTritonSimulation,
        _simulation_tile_capacity,
        _simulation_tile_plan,
    )

    assert AUTO_TILE_BYTES_PER_PHOTON == 512
    assert AUTO_TILE_USABLE_FRACTION == 0.70
    assert AUTO_TILE_RESERVE_BYTES == 512 * 1024 * 1024
    memory = 40 * 1024**3
    plan = _simulation_tile_plan(
        300_000_000, available_memory_bytes=memory
    )
    assert plan.tile_capacity == 57_671_680
    assert plan.tile_count == 6
    assert plan.available_bytes == memory
    assert plan.usable_bytes == 29_527_900_160
    assert plan.estimated_peak_bytes == plan.tile_capacity * 512
    assert not plan.explicit
    assert _simulation_tile_capacity(
        10_000, available_memory_bytes=memory
    ) == 10_000
    assert _simulation_tile_capacity(
        0, available_memory_bytes=memory
    ) == 1
    assert (
        inspect.signature(Reflect3WiresTritonSimulation).parameters[
            "tile_size"
        ].default
        is None
    )
    assert (
        inspect.signature(Reflect3WiresTritonSimulation).parameters[
            "reservoir_rounds"
        ].default
        == 128
    )
    # A device-count epoch is available for experiments, but batching several
    # capacity-sized launches before inspecting a usually-empty continuation
    # queue loses on the production population.  Keep the measured fast path
    # as the default.
    assert (
        inspect.signature(Reflect3WiresTritonSimulation).parameters[
            "history_epochs_per_poll"
        ].default
        == 1
    )


def test_explicit_tile_policy_remains_a_hard_cap():
    from chroma_lar.triton_backend import (
        _simulation_tile_capacity,
        _simulation_tile_plan,
    )

    assert _simulation_tile_capacity(15_000_000, 262_144) == 262_144
    assert _simulation_tile_capacity(10_000, 262_144) == 10_000
    plan = _simulation_tile_plan(1_000_000, 65_537)
    assert plan.tile_capacity == 65_537
    assert plan.tile_count == 16
    assert plan.available_bytes is None
    assert plan.explicit
    with pytest.raises(ValueError, match="tile_size"):
        _simulation_tile_capacity(10, 0)
    with pytest.raises(ValueError, match="nphotons"):
        _simulation_tile_capacity(-1, 1)


def test_auto_tile_policy_is_cpu_testable_and_fails_closed():
    from chroma_lar.triton_backend import _simulation_tile_plan

    # Use simple injected values so the arithmetic and alignment are explicit:
    # floor(10,000 * .8) - 1,000 = 7,000 bytes, or 70 photons.
    plan = _simulation_tile_plan(
        1_000,
        available_memory_bytes=10_000,
        bytes_per_photon=100,
        usable_fraction=0.8,
        reserve_bytes=1_000,
        alignment=32,
    )
    assert plan.tile_capacity == 64
    assert plan.tile_count == 16
    assert plan.usable_bytes == 7_000
    assert plan.estimated_peak_bytes == 6_400

    with pytest.raises(ValueError, match="available_memory_bytes"):
        _simulation_tile_plan(10)
    with pytest.raises(MemoryError, match="cannot fit one photon"):
        _simulation_tile_plan(
            10,
            available_memory_bytes=1_000,
            bytes_per_photon=100,
            usable_fraction=0.5,
            reserve_bytes=500,
        )


def test_cuda_available_memory_includes_reusable_torch_cache():
    from chroma_lar.triton_backend import _cuda_available_memory_bytes

    class FakeCuda:
        @staticmethod
        def mem_get_info(device):
            assert device == "cuda:7"
            return 1_000, 4_000

        @staticmethod
        def memory_reserved(device):
            assert device == "cuda:7"
            return 700

        @staticmethod
        def memory_allocated(device):
            assert device == "cuda:7"
            return 250

    class FakeTorch:
        cuda = FakeCuda()

    assert _cuda_available_memory_bytes(FakeTorch, "cuda:7") == 1_450


def test_boundary_workspaces_are_empty_then_grow_to_observed_queue():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    simulation = Reflect3WiresTritonSimulation()
    assert simulation.merge_workspace.capacity == 0
    assert simulation.analytic_workspace.capacity == 0
    assert simulation.pmt_workspace.ray_capacity == 0
    assert simulation.pmt_workspace.candidate_capacity == 0

    simulation._ensure_boundary_workspace(10_000)
    assert simulation.analytic_workspace.capacity == 10_000
    assert simulation.pmt_workspace.ray_capacity == 10_000
    assert simulation.pmt_workspace.candidate_capacity == 0
    simulation._ensure_survivor_queue(10_000)
    assert simulation.survivor_queue_buffer.numel() == 10_000

    analytic_pointer = simulation.analytic_workspace.candidate_ids.data_ptr()
    ray_pointer = simulation.pmt_workspace.counts.data_ptr()
    simulation._ensure_boundary_workspace(100)
    assert simulation.analytic_workspace.candidate_ids.data_ptr() == analytic_pointer
    assert simulation.pmt_workspace.counts.data_ptr() == ray_pointer
    survivor_pointer = simulation.survivor_queue_buffer.data_ptr()
    simulation._ensure_survivor_queue(100)
    assert simulation.survivor_queue_buffer.data_ptr() == survivor_pointer


def test_cross_tile_reservoir_preserves_exact_hit_ensemble():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    import numpy as np
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    common = dict(tile_size=4096, history_length=4)
    legacy = Reflect3WiresTritonSimulation(
        **common, reservoir_rounds=None
    ).simulate(8192, (-1000.0, 0.0, 0.0), seed=17021)
    pooled = Reflect3WiresTritonSimulation(
        **common, reservoir_rounds=2
    ).simulate(8192, (-1000.0, 0.0, 0.0), seed=17021)

    legacy_time, legacy_channel = legacy.flat_hits.to_numpy()
    pooled_time, pooled_channel = pooled.flat_hits.to_numpy()
    legacy_order = np.lexsort((legacy_time, legacy_channel))
    pooled_order = np.lexsort((pooled_time, pooled_channel))
    np.testing.assert_array_equal(
        legacy_channel[legacy_order], pooled_channel[pooled_order]
    )
    np.testing.assert_array_equal(
        legacy_time[legacy_order], pooled_time[pooled_order]
    )
    assert legacy.stats.boundary_events == pooled.stats.boundary_events
    assert pooled.stats.reservoir_photons > 0
    assert pooled.stats.reservoir_rounds > 0
