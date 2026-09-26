"""Bit-exact CUDA/Triton validation for photon-stable random tapes."""

from pathlib import Path

import numpy as np
import pytest

from chroma.triton.rng_alignment import (
    CERTIFICATE_EMPTY_WORD,
    STATE_CERTIFICATE_EMPTY_WORD,
    STATE_CERTIFICATE_FIELD_COUNT,
    STATE_CERTIFICATE_FIELD_INDEX,
    STATE_CERTIFICATE_FIELDS,
    DRAW_OVERFLOW,
    GLOBAL_ID_MISMATCH,
    INTERACTION_OVERFLOW,
    LEGACY_TAPE_FORCE_SCATTER_AT_PASS,
    RandomTape,
    RandomTapeSpec,
    TapeAudit,
    InteractionCertificate,
    StateCertificate,
    allocate_pycuda_certificate,
    allocate_pycuda_state_certificate,
    allocate_pycuda_audit,
    allocate_pycuda_trace,
    allocate_torch_audit,
    allocate_torch_certificate,
    allocate_torch_state_certificate,
    pack_interaction_certificate,
    probe_reference,
    probe_triton,
    get_legacy_tape_module,
    get_rng_alignment_probe_module,
    launch_legacy_tape_step,
    legacy_tape_compile_policy,
    to_pycuda,
    to_torch,
    unpack_interaction_certificate,
    validate_interaction_certificate,
    validate_state_certificate,
)


def test_legacy_tape_compile_policy_pins_reference_physics_macro():
    policy = legacy_tape_compile_policy()
    assert policy["backend"] in ("nvcc", "nvrtc")
    assert policy["force_scatter_at_pass"] == 0
    assert LEGACY_TAPE_FORCE_SCATTER_AT_PASS == 0
    assert "-DCHROMA_FORCE_SCATTER_AT_PASS=0" in policy["options"]


def _case():
    global_ids = np.asarray(
        [9001, 17, 888888, 42, 73, 101, 555, 1234567], dtype=np.int64
    )
    spec = RandomTapeSpec(
        max_interactions=3, draws_per_interaction=6, seed=0x123456789ABCDEF0
    )
    tape = RandomTape.generate(global_ids, spec)
    audit = TapeAudit.zeros(len(global_ids))
    audit.interaction_cursor[:] = [0, 1, 2, 0, 1, 2, 0, 1]
    audit.draw_cursor[:] = [0, 4, 5, 6, 1, -1, 3, 5]
    rows = np.asarray([6, 0, 7, 2, 5, 1, 4, 3], dtype=np.int32)
    requested_ids = global_ids[rows].copy()
    requested_ids[3] += 1  # one explicit row/global-ID audit failure
    requests = np.asarray([3, 6, 2, 1, 5, 4, 0, 2], dtype=np.int32)
    return tape, audit, rows, requested_ids, requests


def _assert_probe_equal(actual, expected):
    np.testing.assert_array_equal(
        np.asarray(actual.values).view(np.uint32), expected.values.view(np.uint32)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.audit.interaction_cursor),
        expected.audit.interaction_cursor,
    )
    np.testing.assert_array_equal(
        np.asarray(actual.audit.draw_cursor), expected.audit.draw_cursor
    )
    np.testing.assert_array_equal(
        np.asarray(actual.audit.overflow).astype(np.uint32),
        expected.audit.overflow,
    )
    np.testing.assert_array_equal(
        np.asarray(actual.work_overflow).astype(np.uint32),
        expected.work_overflow,
    )


def test_interaction_certificate_pack_decode_and_prefix_validation():
    process = np.asarray([[1, 2, 8], [6, 0, 0]], dtype=np.int32)
    draws = np.asarray([[2, 4, 5], [3, 0, 0]], dtype=np.int32)
    certificate = InteractionCertificate.empty(2, 3)
    certificate.words[0] = pack_interaction_certificate(process[0], draws[0])
    certificate.words[1, 0] = pack_interaction_certificate(6, 3)[0]

    committed, decoded_process, decoded_draws = certificate.unpack()
    np.testing.assert_array_equal(
        committed,
        np.asarray([[True, True, True], [True, False, False]]),
    )
    np.testing.assert_array_equal(
        decoded_process,
        np.asarray([[1, 2, 8], [6, -1, -1]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        decoded_draws,
        np.asarray([[2, 4, 5], [3, -1, -1]], dtype=np.int32),
    )
    validate_interaction_certificate(
        certificate, np.asarray([3, 1], dtype=np.int32)
    )

    malformed = certificate.words.copy()
    malformed[1, 2] = pack_interaction_certificate(7, 4)[0]
    with pytest.raises(ValueError, match="commit prefix"):
        validate_interaction_certificate(
            malformed, np.asarray([3, 1], dtype=np.int32)
        )
    assert certificate.words[1, 2] == CERTIFICATE_EMPTY_WORD

    unset = InteractionCertificate.empty(1, 1)
    unset.words[0, 0] = pack_interaction_certificate(0, 2)[0]
    with pytest.raises(ValueError, match="invalid committed process"):
        validate_interaction_certificate(
            unset, np.asarray([1], dtype=np.int32)
        )


def test_interaction_certificate_rejects_invalid_payloads():
    with pytest.raises(ValueError, match="process"):
        pack_interaction_certificate(11, 2)
    with pytest.raises(ValueError, match="28 bits"):
        pack_interaction_certificate(1, 1 << 28)
    with pytest.raises(TypeError, match="uint32 or int32"):
        unpack_interaction_certificate(np.zeros(1, dtype=np.uint64))


def test_torch_interaction_certificate_uses_raw_signed_words():
    torch = pytest.importorskip("torch")
    certificate = allocate_torch_certificate(2, 3, device="cpu")
    assert certificate.words.dtype == torch.int32
    np.testing.assert_array_equal(
        certificate.words.numpy().view(np.uint32),
        np.full((2, 3), CERTIFICATE_EMPTY_WORD, dtype=np.uint32),
    )


def test_state_certificate_schema_and_process_defined_occupancy():
    assert STATE_CERTIFICATE_FIELD_COUNT == 15
    assert STATE_CERTIFICATE_FIELDS == (
        "position_x", "position_y", "position_z",
        "direction_x", "direction_y", "direction_z",
        "polarization_x", "polarization_y", "polarization_z",
        "wavelength", "time", "history", "last_triangle", "weight", "evidx",
    )
    assert STATE_CERTIFICATE_FIELD_INDEX["history"] == 11
    assert STATE_CERTIFICATE_FIELD_INDEX["last_triangle"] == 12
    assert STATE_CERTIFICATE_FIELD_INDEX["evidx"] == 14

    process = InteractionCertificate.empty(2, 3)
    process.words[0, :2] = pack_interaction_certificate([2, 7], [4, 5])
    process.words[1, 0] = pack_interaction_certificate(3, 3)[0]
    state = StateCertificate.empty(2, 3)
    state.words[0, 0] = np.arange(15, dtype=np.uint32)
    state.words[0, 1] = np.arange(100, 115, dtype=np.uint32)
    state.words[1, 0] = np.arange(200, 215, dtype=np.uint32)
    # A legitimate -1 last-triangle word is indistinguishable from the
    # per-word sentinel; occupancy comes from the paired process ledger.
    state.words[0, 0, STATE_CERTIFICATE_FIELD_INDEX["last_triangle"]] = (
        np.uint32(0xFFFFFFFF)
    )
    validate_state_certificate(state, process, np.asarray([2, 1]))
    np.testing.assert_array_equal(
        state.field("history"),
        state.words[:, :, STATE_CERTIFICATE_FIELD_INDEX["history"]],
    )
    with pytest.raises(KeyError, match="unknown state certificate field"):
        state.field("detected_channel")

    missing = StateCertificate(state.words.copy())
    missing.words[0, 1] = STATE_CERTIFICATE_EMPTY_WORD
    with pytest.raises(ValueError, match="occupancy disagrees"):
        validate_state_certificate(missing, process, np.asarray([2, 1]))

    stray = StateCertificate(state.words.copy())
    stray.words[1, 2, 0] = 0
    with pytest.raises(ValueError, match="occupancy disagrees"):
        validate_state_certificate(stray, process, np.asarray([2, 1]))


def test_state_certificate_rejects_bad_layout_and_mismatched_process_shape():
    with pytest.raises(TypeError, match="state certificate words"):
        StateCertificate(np.zeros((2, 3, 14), dtype=np.uint32))
    with pytest.raises(TypeError, match="state certificate words"):
        StateCertificate(np.zeros((2, 3, 15), dtype=np.int32))
    state = StateCertificate.empty(2, 3)
    process = InteractionCertificate.empty(2, 2)
    with pytest.raises(ValueError, match="matching"):
        validate_state_certificate(state, process, np.zeros(2, dtype=np.int32))


def test_torch_state_certificate_uses_raw_signed_words():
    torch = pytest.importorskip("torch")
    certificate = allocate_torch_state_certificate(2, 3, device="cpu")
    assert certificate.words.dtype == torch.int32
    assert tuple(certificate.words.shape) == (2, 3, 15)
    np.testing.assert_array_equal(
        certificate.words.numpy().view(np.uint32),
        np.full((2, 3, 15), STATE_CERTIFICATE_EMPTY_WORD, dtype=np.uint32),
    )


def test_random_tape_has_stable_golden_float32_words():
    tape = RandomTape.generate([11, 22], RandomTapeSpec(3, 4, seed=17))
    np.testing.assert_array_equal(
        tape.value_bits.reshape(-1)[:8],
        np.asarray(
            [
                1060282524,
                1061885792,
                1060011072,
                1062412346,
                1062751820,
                1044081656,
                1055659592,
                1063275906,
            ],
            dtype=np.uint32,
        ),
    )
    assert np.all(tape.values > np.float32(0.0))
    assert np.all(tape.values <= np.float32(1.0))


def test_tape_rows_are_keyed_by_global_id_not_worker_order():
    spec = RandomTapeSpec(4, 7, seed=20260901)
    ids = np.asarray([81, 5, 900, 12, 63], dtype=np.int64)
    first = RandomTape.generate(ids, spec)
    permutation = np.asarray([3, 0, 4, 1, 2])
    second = RandomTape.generate(ids[permutation], spec)
    rows = second.rows_for(ids)
    np.testing.assert_array_equal(second.value_bits[rows], first.value_bits)

    audit = TapeAudit.zeros(len(ids))
    forward = probe_reference(first, np.arange(len(ids)), ids, [2] * len(ids), audit)
    reverse_rows = np.arange(len(ids) - 1, -1, -1, dtype=np.int32)
    reverse = probe_reference(
        first,
        reverse_rows,
        ids[reverse_rows],
        [2] * len(ids),
        audit,
    )
    np.testing.assert_array_equal(
        reverse.values[::-1].view(np.uint32), forward.values.view(np.uint32)
    )


def test_reference_audits_draw_and_global_id_failures_without_wrapping():
    tape, audit, rows, requested_ids, requests = _case()
    result = probe_reference(
        tape, rows, requested_ids, requests, audit, max_requests=6
    )
    mismatch_worker = 3
    assert result.work_overflow[mismatch_worker] & GLOBAL_ID_MISMATCH
    # Includes both upper-bound exhaustion and a corrupted negative cursor;
    # all consumers must saturate either case without indexing adjacent data.
    draw_overflow_workers = [2, 4, 7]
    assert all(
        result.work_overflow[worker] & DRAW_OVERFLOW
        for worker in draw_overflow_workers
    )
    assert result.audit.draw_cursor[rows[2]] == tape.spec.draws_per_interaction
    # Canonical NaN bits make overflow output itself bit-auditable.
    assert result.values.view(np.uint32)[mismatch_worker, 0] == 0x7FC00000

    exhausted = audit.copy()
    exhausted.interaction_cursor[0] = tape.spec.max_interactions
    interaction_result = probe_reference(
        tape, [0], [tape.global_photon_ids[0]], [1], exhausted
    )
    assert interaction_result.work_overflow[0] & INTERACTION_OVERFLOW


def test_tape_allocation_guard_is_explicit():
    spec = RandomTapeSpec(100, 32, seed=1)
    with pytest.raises(MemoryError, match="max_bytes"):
        RandomTape.generate(np.arange(1000), spec, max_bytes=1024)


@pytest.mark.skipif(
    not Path("/dev/nvidiactl").exists(), reason="CUDA device is not visible"
)
def test_legacy_cuda_probe_matches_reference_bit_for_bit():
    pytest.importorskip("pycuda")
    # Use CUDA's primary context so this test can coexist with Torch in the
    # same process.  Chroma's ordinary Simulation context remains untouched.
    pytest.importorskip("pycuda.autoprimaryctx")
    from pycuda import driver as cuda
    from pycuda import gpuarray as ga

    tape, initial_audit, rows, requested_ids, requests = _case()
    expected = probe_reference(
        tape, rows, requested_ids, requests, initial_audit, max_requests=6
    )
    device_tape = to_pycuda(tape)
    device_audit = allocate_pycuda_audit(tape.photon_count)
    device_audit.interaction_cursor.set(initial_audit.interaction_cursor)
    device_audit.draw_cursor.set(initial_audit.draw_cursor)
    device_audit.overflow.set(initial_audit.overflow)
    rows_gpu = ga.to_gpu(rows)
    ids_gpu = ga.to_gpu(requested_ids)
    requests_gpu = ga.to_gpu(requests)
    output_gpu = ga.empty((len(rows), 6), dtype=np.float32)
    work_overflow_gpu = ga.empty(len(rows), dtype=np.uint32)
    module = get_rng_alignment_probe_module()
    kernel = module.get_function("rng_alignment_probe")
    kernel(
        np.int32(len(rows)),
        np.int32(6),
        device_tape.values,
        device_tape.global_photon_ids,
        np.int32(tape.photon_count),
        np.int32(tape.spec.max_interactions),
        np.int32(tape.spec.draws_per_interaction),
        rows_gpu,
        ids_gpu,
        requests_gpu,
        device_audit.interaction_cursor,
        device_audit.draw_cursor,
        device_audit.overflow,
        output_gpu,
        work_overflow_gpu,
        block=(128, 1, 1),
        grid=((len(rows) + 127) // 128, 1, 1),
    )
    cuda.Context.synchronize()
    actual = type(expected)(
        output_gpu.get(),
        TapeAudit(
            device_audit.interaction_cursor.get(),
            device_audit.draw_cursor.get(),
            device_audit.overflow.get(),
        ),
        work_overflow_gpu.get(),
    )
    _assert_probe_equal(actual, expected)


@pytest.mark.skipif(
    not Path("/dev/nvidiactl").exists(), reason="CUDA device is not visible"
)
def test_legacy_tape_module_audits_queue_resolved_photon_mapping():
    """The full tape module compiles and maps rows after queue resolution."""

    pytest.importorskip("pycuda")
    pytest.importorskip("pycuda.autoprimaryctx")
    from pycuda import driver as cuda
    from pycuda import gpuarray as ga

    tape = RandomTape.generate(
        [11, 22, 33], RandomTapeSpec(3, 8, seed=99)
    )
    device_tape = to_pycuda(tape)
    audit = allocate_pycuda_audit(tape.photon_count)
    trace = allocate_pycuda_trace(4)
    # Photon array order is unrelated to tape row order.  Photon 1 has a
    # deliberately wrong global ID; photon 3 deliberately names row 3, which
    # is just outside this three-row tape.
    photon_rows = ga.to_gpu(np.asarray([1, 2, 0, 3], dtype=np.int32))
    photon_ids = ga.to_gpu(np.asarray([22, 999, 11, 33], dtype=np.int64))
    queue = ga.to_gpu(np.asarray([3, 2, 0, 1], dtype=np.uint32))

    module = get_legacy_tape_module()
    # Resolving this symbol also verifies that the complete geometry/physics
    # tape kernel compiled, even though this focused launch needs no geometry.
    module.get_function("propagate_tape")
    probe = module.get_function("propagate_tape_mapping_probe")
    probe(
        np.int32(0),
        np.int32(4),
        queue,
        photon_rows,
        photon_ids,
        device_tape.global_photon_ids,
        np.int32(tape.photon_count),
        np.int32(tape.spec.max_interactions),
        np.int32(tape.spec.draws_per_interaction),
        audit.interaction_cursor,
        audit.draw_cursor,
        audit.overflow,
        trace.stage,
        trace.first_error_photon_id,
        trace.first_error_flags,
        trace.first_error_interaction,
        trace.first_error_draw,
        trace.first_error_stage,
        block=(128, 1, 1),
        grid=(1, 1, 1),
    )
    cuda.Context.synchronize()

    np.testing.assert_array_equal(
        audit.overflow.get(),
        np.asarray([0, 0, int(GLOBAL_ID_MISMATCH)], dtype=np.uint32),
    )
    np.testing.assert_array_equal(trace.stage.get(), np.zeros(4, dtype=np.int32))
    first_photon = int(trace.first_error_photon_id.get()[0])
    first_flags = int(trace.first_error_flags.get()[0])
    if first_photon == 1:
        assert first_flags == int(GLOBAL_ID_MISMATCH)
    else:
        assert first_photon == 3
        assert first_flags == 8  # ROW_OUT_OF_RANGE
    assert int(trace.first_error_stage.get()[0]) == 0


@pytest.mark.skipif(
    not Path("/dev/nvidiactl").exists(), reason="CUDA device is not visible"
)
def test_cuda_state_certificate_preserves_raw_float_and_signed_integer_words():
    """Exercise the exact CUDA writer, including -0 and fractional floats."""

    pytest.importorskip("pycuda")
    pytest.importorskip("pycuda.autoprimaryctx")
    from pycuda import driver as cuda
    from pycuda import gpuarray as ga

    words = np.full(STATE_CERTIFICATE_FIELD_COUNT, 0, dtype=np.uint32)
    float_fields = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13,
    ]
    float_values = np.asarray(
        [
            -0.0, 1.25, -2.75,
            0.125, -0.5, 3.125,
            -4.5, 7.75, -8.25,
            450.125, -0.0, 0.333251953125,
        ],
        dtype=np.float32,
    )
    words[float_fields] = float_values.view(np.uint32)
    words[STATE_CERTIFICATE_FIELD_INDEX["history"]] = np.uint32(0x8042)
    words[STATE_CERTIFICATE_FIELD_INDEX["last_triangle"]] = np.uint32(
        0xFFFFFFFF
    )
    words[STATE_CERTIFICATE_FIELD_INDEX["evidx"]] = np.uint32(0xFEDCBA98)
    output = ga.to_gpu(np.full_like(words, STATE_CERTIFICATE_EMPTY_WORD))
    module = get_legacy_tape_module()
    module.get_function("propagate_tape_debug_write_state_certificate")(
        ga.to_gpu(words), output, block=(1, 1, 1), grid=(1, 1, 1)
    )
    cuda.Context.synchronize()
    np.testing.assert_array_equal(output.get(), words)


@pytest.mark.skipif(
    not Path("/dev/nvidiactl").exists(), reason="CUDA device is not visible"
)
def test_legacy_tape_full_kernel_no_hit_consumes_no_draws():
    """Smoke the real queue/geometry kernel, not only its mapping probe."""

    pytest.importorskip("pycuda")
    pytest.importorskip("pycuda.autoprimaryctx")
    from types import SimpleNamespace

    from pycuda import driver as cuda
    from pycuda import gpuarray as ga

    from chroma.gpu.tools import to_float3

    module = get_legacy_tape_module()
    size_gpu = ga.empty(1, dtype=np.uint64)
    module.get_function("propagate_tape_debug_geometry_size")(
        size_gpu, block=(1, 1, 1), grid=(1, 1, 1)
    )
    geometry_storage = cuda.mem_alloc(int(size_gpu.get()[0]))
    root = ga.empty(1, dtype=ga.vec.uint4)
    module.get_function("propagate_tape_init_no_hit_geometry")(
        geometry_storage, root, block=(1, 1, 1), grid=(1, 1, 1)
    )

    def vec3(value):
        return ga.to_gpu(to_float3(np.asarray([value], dtype=np.float32)))

    photons = SimpleNamespace(
        pos=vec3([10.0, 10.0, 10.0]),
        dir=vec3([0.0, 0.0, 1.0]),
        pol=vec3([1.0, 0.0, 0.0]),
        wavelengths=ga.to_gpu(np.asarray([128.0], dtype=np.float32)),
        t=ga.to_gpu(np.asarray([0.0], dtype=np.float32)),
        flags=ga.to_gpu(np.asarray([0], dtype=np.uint32)),
        last_hit_triangles=ga.to_gpu(np.asarray([-1], dtype=np.int32)),
        weights=ga.to_gpu(np.asarray([1.0], dtype=np.float32)),
        evidx=ga.to_gpu(np.asarray([0], dtype=np.uint32)),
    )
    tape = RandomTape.generate([7001], RandomTapeSpec(2, 8, seed=123))
    device_tape = to_pycuda(tape)
    audit = allocate_pycuda_audit(1)
    trace = allocate_pycuda_trace(1)
    certificate = allocate_pycuda_certificate(1, tape.spec.max_interactions)
    state_certificate = allocate_pycuda_state_certificate(
        1, tape.spec.max_interactions
    )
    queue = ga.to_gpu(np.asarray([0], dtype=np.uint32))
    output_queue = ga.to_gpu(np.asarray([1, 0], dtype=np.uint32))
    launch_legacy_tape_step(
        gpu_photons=photons,
        gpu_geometry=SimpleNamespace(gpudata=geometry_storage),
        input_queue=queue,
        output_queue=output_queue,
        photon_tape_rows=ga.to_gpu(np.asarray([0], dtype=np.int32)),
        photon_global_ids=ga.to_gpu(np.asarray([7001], dtype=np.int64)),
        tape=device_tape,
        audit=audit,
        trace=trace,
        certificate=certificate,
        state_certificate=state_certificate,
        max_steps=1,
    )
    cuda.Context.synchronize()

    assert int(photons.flags.get()[0]) == 1  # NO_HIT
    assert int(audit.interaction_cursor.get()[0]) == 0
    assert int(audit.draw_cursor.get()[0]) == 0
    assert int(trace.process.get()[0]) == 0
    assert int(trace.draw_count.get()[0]) == 0
    np.testing.assert_array_equal(
        certificate.words.get(),
        np.full((1, 2), CERTIFICATE_EMPTY_WORD, dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        state_certificate.words.get(),
        np.full(
            (1, 2, STATE_CERTIFICATE_FIELD_COUNT),
            STATE_CERTIFICATE_EMPTY_WORD,
            dtype=np.uint32,
        ),
    )
    assert int(trace.first_error_photon_id.get()[0]) == -1
    assert int(output_queue.get()[0]) == 1


def test_triton_probe_matches_reference_bit_for_bit():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not visible to Torch")
    pytest.importorskip("triton")

    tape, initial_audit, rows, requested_ids, requests = _case()
    expected = probe_reference(
        tape, rows, requested_ids, requests, initial_audit, max_requests=6
    )
    device_tape = to_torch(tape)
    device_audit = allocate_torch_audit(tape.photon_count)
    device_audit.interaction_cursor.copy_(
        torch.from_numpy(initial_audit.interaction_cursor).cuda()
    )
    device_audit.draw_cursor.copy_(torch.from_numpy(initial_audit.draw_cursor).cuda())
    device_audit.overflow.copy_(
        torch.from_numpy(initial_audit.overflow.astype(np.int32)).cuda()
    )
    actual_gpu = probe_triton(
        device_tape,
        rows,
        requested_ids,
        requests,
        device_audit,
        max_requests=6,
    )
    torch.cuda.synchronize()
    actual = type(expected)(
        actual_gpu.values.cpu().numpy(),
        TapeAudit(
            actual_gpu.audit.interaction_cursor.cpu().numpy(),
            actual_gpu.audit.draw_cursor.cpu().numpy(),
            actual_gpu.audit.overflow.cpu().numpy().astype(np.uint32),
        ),
        actual_gpu.work_overflow.cpu().numpy().astype(np.uint32),
    )
    _assert_probe_equal(actual, expected)
