"""Exact draw-order checks for the detector boundary random-tape path."""

import numpy as np
import pytest


def _state(torch, count, device):
    positions = torch.zeros((count, 3), dtype=torch.float32, device=device)
    directions = torch.zeros_like(positions)
    directions[:, 0] = 1.0
    polarizations = torch.zeros_like(positions)
    polarizations[:, 1] = 1.0
    times = torch.zeros(count, dtype=torch.float32, device=device)
    histories = torch.zeros(count, dtype=torch.int32, device=device)
    rng_counters = torch.zeros(count, dtype=torch.int64, device=device)
    last_instances = torch.full(
        (count,), -1, dtype=torch.int32, device=device
    )
    last_triangles = torch.full_like(last_instances, -1)
    detected_channels = torch.full_like(last_instances, -1)
    step_counts = torch.zeros_like(last_instances)
    return (
        positions,
        directions,
        polarizations,
        times,
        histories,
        rng_counters,
        last_instances,
        last_triangles,
        detected_channels,
        step_counts,
    )


def _hit(torch, count, device, surfaces):
    distances = torch.full(
        (count,), 10.0, dtype=torch.float32, device=device
    )
    normals = torch.zeros((count, 3), dtype=torch.float32, device=device)
    normals[:, 0] = -1.0
    material_from = torch.zeros(count, dtype=torch.int32, device=device)
    material_to = torch.ones(count, dtype=torch.int32, device=device)
    surface = torch.tensor(surfaces, dtype=torch.int32, device=device)
    instance = torch.full_like(surface, -1)
    triangle = torch.full_like(surface, -1)
    channel = torch.arange(count, dtype=torch.int32, device=device)
    return (
        distances,
        normals,
        material_from,
        material_to,
        surface,
        instance,
        triangle,
        channel,
    )


def _device_tape(torch, values, max_interactions=3):
    from chroma.triton.rng_alignment import RandomTapeSpec, TorchRandomTape

    tensor = torch.tensor(values, dtype=torch.float32, device="cuda")
    count, draws = tensor.shape
    full = torch.ones(
        (count, max_interactions, draws),
        dtype=torch.float32,
        device="cuda",
    )
    full[:, 0, :] = tensor
    return TorchRandomTape(
        values=full.contiguous(),
        global_photon_ids=torch.arange(
            count, dtype=torch.int64, device="cuda"
        ),
        spec=RandomTapeSpec(
            max_interactions=max_interactions,
            draws_per_interaction=draws,
            seed=1,
        ),
    )


def test_boundary_tape_exact_conditional_decisions_and_cursors():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from chroma.triton.rng_alignment import (
        CERTIFICATE_EMPTY_WORD,
        STATE_CERTIFICATE_EMPTY_WORD,
        allocate_torch_audit,
        allocate_torch_certificate,
        allocate_torch_state_certificate,
        pack_interaction_certificate,
        validate_state_certificate,
    )
    from chroma_lar.triton_backend import (
        BoundaryTapeTrace,
        Reflect3WiresTritonSimulation,
        TAPE_PROCESS_BULK_ABSORB,
        TAPE_PROCESS_BULK_SCATTER,
        TAPE_PROCESS_DIELECTRIC_TRANSMIT,
        TAPE_PROCESS_SURFACE_ABSORB,
        TAPE_PROCESS_SURFACE_DETECT,
        TAPE_PROCESS_SURFACE_DIFFUSE,
        TAPE_PROCESS_SURFACE_SPECULAR,
    )

    simulation = Reflect3WiresTritonSimulation(tile_size=8)
    device = simulation.device
    count = 8
    state = _state(torch, count, device)
    queue = torch.arange(count, dtype=torch.int32, device=device)
    # reflect00 absorb, perfect PMT detect, glossy specular/diffuse, a
    # deliberately sub-unit default surface for PASS, then no surface.
    hit = _hit(torch, count, device, [-1, -1, 0, 1, 2, 2, 3, -1])
    simulation.scene_device["tables_surface_absorb"][3] = 0.1
    simulation.scene_device["tables_surface_reflect_specular"][3] = 0.4

    values = [[0.5] * 8 for _ in range(count)]
    values[0][0] = 1.0  # zero absorption distance
    values[1][0] = 0.5
    values[1][1] = 1.0  # zero scattering distance
    values[2][2] = 0.1  # reflect00 -> absorb
    values[3][2] = 0.1  # perfect PMT -> detect
    values[4][2] = 0.75  # glossy -> specular
    values[5][2] = 0.25  # glossy -> diffuse
    values[5][3] = 0.5  # direction sphere points along -x
    values[5][4] = 0.5
    values[5][5] = 0.5  # accepted on first rejection-loop attempt
    values[5][6] = 0.25
    values[5][7] = 0.75  # polarization sphere
    values[6][2] = 0.9  # sub-unit surface -> PASS
    values[6][3] = 0.5
    values[6][4] = 1.0  # Fresnel transmit
    values[7][2] = 0.5
    values[7][3] = 1.0  # no-surface Fresnel transmit
    tape = _device_tape(torch, values)
    audit = allocate_torch_audit(count, device=device)
    certificate = allocate_torch_certificate(count, 3, device=device)
    state_certificate = allocate_torch_state_certificate(
        count, 3, device=device
    )
    trace = BoundaryTapeTrace.allocate(count, device=device)

    survivors = simulation._step_boundaries(
        state,
        queue,
        hit,
        seed=7,
        photon_id_base=0,
        max_steps=100,
        random_tape=tape,
        tape_audit=audit,
        tape_trace=trace,
        tape_certificate=certificate,
        state_certificate=state_certificate,
    ).tensor()

    torch.testing.assert_close(
        audit.interaction_cursor,
        torch.ones(count, dtype=torch.int32, device=device),
        rtol=0,
        atol=0,
    )
    assert audit.draw_cursor.tolist() == [0] * count
    assert audit.overflow.tolist() == [0] * count
    assert trace.interaction.tolist() == [0] * count
    assert trace.draw_count.tolist() == [2, 4, 3, 3, 3, 8, 5, 4]
    assert trace.decision.tolist() == [
        TAPE_PROCESS_BULK_ABSORB,
        TAPE_PROCESS_BULK_SCATTER,
        TAPE_PROCESS_SURFACE_ABSORB,
        TAPE_PROCESS_SURFACE_DETECT,
        TAPE_PROCESS_SURFACE_SPECULAR,
        TAPE_PROCESS_SURFACE_DIFFUSE,
        TAPE_PROCESS_DIELECTRIC_TRANSMIT,
        TAPE_PROCESS_DIELECTRIC_TRANSMIT,
    ]
    expected_certificate = np.full(
        (count, 3), CERTIFICATE_EMPTY_WORD, dtype=np.uint32
    )
    expected_certificate[:, 0] = pack_interaction_certificate(
        np.asarray(trace.decision.cpu(), dtype=np.int32),
        np.asarray(trace.draw_count.cpu(), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        certificate.words.cpu().numpy().view(np.uint32),
        expected_certificate,
    )
    state_words = np.ascontiguousarray(
        state_certificate.words.cpu().numpy().view(np.uint32)
    )
    validate_state_certificate(
        state_words,
        expected_certificate,
        audit.interaction_cursor.cpu().numpy(),
    )
    expected_state = np.empty((count, 15), dtype=np.uint32)
    expected_state[:, 0:3] = np.ascontiguousarray(
        state[0].cpu().numpy()
    ).view(np.uint32)
    expected_state[:, 3:6] = np.ascontiguousarray(
        state[1].cpu().numpy()
    ).view(np.uint32)
    expected_state[:, 6:9] = np.ascontiguousarray(
        state[2].cpu().numpy()
    ).view(np.uint32)
    expected_state[:, 9] = np.asarray(
        [np.float32(450.0)], dtype=np.float32
    ).view(np.uint32)[0]
    expected_state[:, 10] = np.ascontiguousarray(
        state[3].cpu().numpy()
    ).view(np.uint32)
    expected_state[:, 11] = np.ascontiguousarray(
        state[4].cpu().numpy()
    ).view(np.uint32)
    expected_state[:, 12] = np.ascontiguousarray(
        state[7].cpu().numpy()
    ).view(np.uint32)
    expected_state[:, 13] = np.asarray(
        [np.float32(1.0)], dtype=np.float32
    ).view(np.uint32)[0]
    expected_state[:, 14] = np.uint32(0)
    np.testing.assert_array_equal(state_words[:, 0], expected_state)
    assert np.all(state_words[:, 1:] == STATE_CERTIFICATE_EMPTY_WORD)
    assert sorted(survivors.tolist()) == [1, 4, 5, 6, 7]
    assert state[8][3].item() == 3
    assert state[5].tolist() == [0] * count  # tape mode never touches Philox


def test_legacy_specular_switch_executes_chroma_rodrigues_path():
    """Compatibility mode retains Chroma's historical surface numerics."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from chroma.triton.rng_alignment import allocate_torch_audit
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    simulation = Reflect3WiresTritonSimulation(
        tile_size=1, legacy_specular_reflection=True
    )
    device = simulation.device
    state = _state(torch, 1, device)
    state[1][0] = torch.tensor([0.6, 0.8, 0.0], device=device)
    state[2][0] = torch.tensor([0.0, 0.0, 1.0], device=device)
    queue = torch.zeros(1, dtype=torch.int32, device=device)
    hit = _hit(torch, 1, device, [2])  # glossy: selector 0.75 -> specular
    tape = _device_tape(torch, [[0.5, 0.5, 0.75, 0.5, 0.5]])
    audit = allocate_torch_audit(1, device=device)

    simulation._step_boundaries(
        state,
        queue,
        hit,
        seed=17,
        photon_id_base=0,
        max_steps=100,
        random_tape=tape,
        tape_audit=audit,
    )

    torch.testing.assert_close(
        state[1][0],
        torch.tensor([-0.6, 0.8, 0.0], device=device),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    assert simulation.legacy_specular_reflection


def test_boundary_tape_overflow_and_global_id_mismatch_fail_closed():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from chroma.triton.rng_alignment import allocate_torch_audit
    from chroma_lar.triton_backend import (
        BoundaryTapeTrace,
        Reflect3WiresTritonSimulation,
        TAPE_PROCESS_UNSET,
    )

    simulation = Reflect3WiresTritonSimulation(tile_size=3)
    device = simulation.device
    state = _state(torch, 3, device)
    queue = torch.arange(3, dtype=torch.int32, device=device)
    hit = _hit(torch, 3, device, [-1, -1, -1])
    tape = _device_tape(torch, [[0.5] * 3 for _ in range(3)])
    tape.global_photon_ids[1] = 999
    audit = allocate_torch_audit(3, device=device)
    trace = BoundaryTapeTrace.allocate(3, device=device)
    tape_rows = torch.tensor([0, 1, -1], dtype=torch.int32, device=device)

    survivors = simulation._step_boundaries(
        state,
        queue,
        hit,
        seed=11,
        photon_id_base=0,
        max_steps=100,
        random_tape=tape,
        tape_audit=audit,
        tape_row_indices=tape_rows,
        tape_trace=trace,
    ).tensor()

    assert survivors.numel() == 0
    assert audit.interaction_cursor.tolist() == [0, 0, 0]
    assert audit.draw_cursor.tolist() == [3, 0, 0]
    assert audit.overflow.tolist() == [1, 4, 0]
    assert trace.interaction.tolist() == [0, 0, -1]
    assert trace.draw_count.tolist() == [4, 0, 0]
    assert trace.decision.tolist() == [TAPE_PROCESS_UNSET] * 3
    assert trace.work_overflow.tolist() == [1, 4, 8]
    # NO_HIT | NAN_ABORT records that alignment failed rather than silently
    # switching either lane back to an unrelated Philox stream.
    assert state[4].tolist() == [(1 << 0) | (1 << 15)] * 3


def test_boundary_tape_marks_exhausted_interaction_after_exact_commit():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    from chroma.triton.rng_alignment import allocate_torch_audit
    from chroma_lar.triton_backend import (
        BoundaryTapeTrace,
        Reflect3WiresTritonSimulation,
        TAPE_PROCESS_SURFACE_ABSORB,
    )

    simulation = Reflect3WiresTritonSimulation(tile_size=1)
    device = simulation.device
    state = _state(torch, 1, device)
    queue = torch.zeros(1, dtype=torch.int32, device=device)
    hit = _hit(torch, 1, device, [0])
    tape = _device_tape(torch, [[0.5, 0.5, 0.1]], max_interactions=1)
    audit = allocate_torch_audit(1, device=device)
    trace = BoundaryTapeTrace.allocate(1, device=device)

    simulation._step_boundaries(
        state,
        queue,
        hit,
        seed=13,
        photon_id_base=0,
        max_steps=100,
        random_tape=tape,
        tape_audit=audit,
        tape_trace=trace,
    )
    assert audit.interaction_cursor.item() == 1
    assert audit.draw_cursor.item() == 0
    assert audit.overflow.item() == 2
    assert trace.draw_count.item() == 3
    assert trace.decision.item() == TAPE_PROCESS_SURFACE_ABSORB


def test_full_simulation_plumbs_one_tape_through_tiles_and_result():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("triton")
    import numpy as np
    from chroma.triton.rng_alignment import (
        RandomTape,
        RandomTapeSpec,
        allocate_torch_state_certificate,
        to_torch,
        validate_interaction_certificate,
        validate_state_certificate,
    )
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    count = 16
    host_tape = RandomTape.generate(
        np.arange(count, dtype=np.int64),
        RandomTapeSpec(max_interactions=8, draws_per_interaction=32, seed=17),
        max_bytes=None,
    )
    tape = to_torch(host_tape)
    state_certificate = allocate_torch_state_certificate(
        count, 8, device=tape.values.device
    )
    result = Reflect3WiresTritonSimulation(
        tile_size=8, history_length=2, reservoir_rounds=2
    ).simulate(
        count,
        (-1000.0, 0.0, 0.0),
        seed=19,
        max_steps=4,
        random_tape=tape,
        state_certificate=state_certificate,
    )

    assert result.tape_audit is not None
    assert result.tape_certificate is not None
    assert result.state_certificate is state_certificate
    assert result.boundary_tape_trace is not None
    assert result.tape_audit.interaction_cursor.shape == (count,)
    assert result.tape_certificate.words.shape == (count, 8)
    assert result.boundary_tape_trace.decision.shape == (count,)
    assert torch.all(result.tape_audit.interaction_cursor >= 0)
    assert torch.all(result.tape_audit.interaction_cursor <= 4)
    validate_interaction_certificate(
        result.tape_certificate.words.cpu().numpy(),
        result.tape_audit.interaction_cursor.cpu().numpy(),
    )
    validate_state_certificate(
        result.state_certificate.words.cpu().numpy(),
        result.tape_certificate.words.cpu().numpy(),
        result.tape_audit.interaction_cursor.cpu().numpy(),
    )
