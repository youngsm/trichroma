"""CPU contracts for CUDA/Triton first-divergence diagnostics."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from chroma.triton.lockstep import (
    InteractionStage,
    LockstepContractError,
    ProcessDecision,
    capture_boundary_tape_trace,
    capture_legacy_tape_trace,
    capture_trace,
    compare_traces,
    derive_process_decisions,
    run_lockstep,
)
from chroma.triton.rng_alignment import RandomTape, RandomTapeSpec, TapeAudit


def _trace(*, order=(0, 1), perturb=None, with_tape=True):
    ids = np.asarray([91, 7], dtype=np.int64)
    tape = RandomTape.generate(ids, RandomTapeSpec(4, 8, seed=17))
    audit = TapeAudit.zeros(2)
    audit.interaction_cursor[:] = 1
    position = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    direction = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    polarization = np.asarray([[0, 1, 0], [0, 0, 1]], dtype=np.float32)
    times = np.asarray([0.5, 1.5], dtype=np.float32)
    histories = np.asarray([2, 16], dtype=np.uint32)
    processes = np.asarray(
        [ProcessDecision.BULK_ABSORB, ProcessDecision.BULK_SCATTER],
        dtype=np.int32,
    )
    draw_counts = np.asarray([2, 4], dtype=np.int32)
    if perturb is not None:
        perturb(
            position, direction, polarization, times, histories, processes,
            draw_counts, audit,
        )

    def slot_stage(record, slot, coarse):
        del record
        if coarse == InteractionStage.RAYLEIGH and slot < 2:
            return InteractionStage.BULK
        return coarse

    return capture_trace(
        global_photon_ids=ids,
        positions=position,
        directions=direction,
        polarizations=polarization,
        times=times,
        histories=histories,
        process=processes,
        audit=audit,
        interaction_indices=np.asarray([0, 0], dtype=np.int32),
        draw_counts=draw_counts,
        step_index=0,
        state_rows=np.asarray(order, dtype=np.int32),
        tape=tape if with_tape else None,
        slot_stage_resolver=slot_stage,
        extras={"last_hit_triangle": np.asarray([-1, 42], dtype=np.int32)},
    )


def test_lockstep_is_invariant_to_worker_and_compaction_order():
    cuda = _trace(order=(0, 1))
    triton = _trace(order=(1, 0))
    report = compare_traces(cuda, triton)
    assert report.matched
    assert report.compared_records == 2
    assert report.draw_consumption_certified
    report.require_match()


def test_first_float_difference_reports_raw_word_draw_and_full_state():
    cuda = _trace()

    def perturb(position, *_):
        # One ULP above 5.0f.  A tolerance comparison would incorrectly hide
        # this, while the lockstep oracle must expose it.
        position[1, 1] = np.nextafter(
            np.float32(5.0), np.float32(np.inf), dtype=np.float32
        )

    triton = _trace(perturb=perturb)
    report = compare_traces(cuda, triton)
    assert not report.matched
    difference = report.difference
    assert difference.global_photon_id == 7
    assert difference.step_index == 0
    assert difference.interaction_index == 0
    assert difference.field == "position[1]"
    assert difference.left_word == 0x40A00000
    assert difference.right_word == 0x40A00001
    assert difference.draw_slot == 3
    assert difference.left_draw_word == difference.right_draw_word
    assert difference.left_state["position_words"][1] == "0x40a00000"
    message = difference.format()
    assert "photon=7" in message
    assert "slot=3" in message
    assert "0x40a00000" in message
    with pytest.raises(AssertionError, match="position\\[1\\]"):
        report.require_match()


def test_draw_consumption_divergence_precedes_equal_final_state():
    cuda = _trace()

    def perturb(*values):
        values[6][1] = 3  # draw_counts

    triton = _trace(perturb=perturb)
    report = compare_traces(cuda, triton)
    assert not report.matched
    assert report.difference.field == "draw_presence"
    assert report.difference.global_photon_id == 7
    assert report.difference.draw_slot == 3
    assert report.difference.left_draw_word is not None
    assert report.difference.right_draw_word is None


def test_draw_stage_divergence_reports_same_shared_tape_word():
    left = _trace()
    draws = left.draws
    stages = draws.stages.copy()
    target = np.flatnonzero(
        (draws.global_photon_ids == 7) & (draws.draw_slots == 2)
    )[0]
    stages[target] = int(InteractionStage.DIELECTRIC)
    right = replace(left, draws=replace(draws, stages=stages))
    report = compare_traces(left, right)
    assert report.difference.field == "draw_stage"
    assert report.difference.draw_slot == 2
    assert report.difference.left_draw_word == report.difference.right_draw_word


def test_matching_state_without_draw_records_is_not_exactly_certified():
    left = _trace(with_tape=False)
    report = compare_traces(left, left)
    assert report.matched
    assert not report.draw_consumption_certified
    with pytest.raises(AssertionError, match="did not record exact draw"):
        report.require_match()
    report.require_match(require_draw_certification=False)


def test_process_derivation_is_explicit_about_cumulative_history_ambiguity():
    before = np.asarray([0, 16, 0, 0], dtype=np.uint32)
    after = np.asarray([16, 16, 2, (1 << 0) | (1 << 15)], dtype=np.uint32)
    decisions = derive_process_decisions(before, after)
    np.testing.assert_array_equal(
        decisions,
        np.asarray(
            [
                ProcessDecision.BULK_SCATTER,
                ProcessDecision.UNSET,
                ProcessDecision.BULK_ABSORB,
                ProcessDecision.INVALID,
            ],
            dtype=np.int32,
        ),
    )


def test_capture_rejects_duplicate_photon_rows_in_one_logical_step():
    ids = np.asarray([4, 5], dtype=np.int64)
    zeros3 = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(LockstepContractError, match="state_rows must be unique"):
        capture_trace(
            global_photon_ids=ids,
            positions=zeros3,
            directions=zeros3,
            polarizations=zeros3,
            times=np.zeros(2, dtype=np.float32),
            histories=np.zeros(2, dtype=np.uint32),
            process=np.zeros(2, dtype=np.int32),
            audit=TapeAudit.zeros(2),
            interaction_indices=np.zeros(2, dtype=np.int32),
            draw_counts=np.zeros(2, dtype=np.int32),
            step_index=0,
            state_rows=np.asarray([0, 0], dtype=np.int32),
        )


def test_run_lockstep_stops_at_first_failing_interaction():
    class Adapter:
        def __init__(self, name, fail_at=None):
            self.name = name
            self.fail_at = fail_at
            self.calls = []

        def advance_one(self, logical_step_index):
            self.calls.append(logical_step_index)
            trace = _trace()
            steps = np.full(trace.count, logical_step_index, dtype=np.int32)
            if logical_step_index == self.fail_at:
                position = trace.position.copy()
                position[0, 0] += np.float32(1.0)
                trace = replace(trace, position=position)
            return replace(
                trace,
                step_indices=steps,
                draws=replace(trace.draws, step_indices=np.repeat(
                    steps, trace.draw_count
                )),
            )

    cuda = Adapter("legacy CUDA")
    triton = Adapter("Triton", fail_at=2)
    report = run_lockstep(cuda, triton, max_steps=10)
    assert not report.matched
    assert report.difference.step_index == 2
    assert cuda.calls == [0, 1, 2]
    assert triton.calls == [0, 1, 2]


def test_concrete_legacy_and_boundary_payload_adapters_match_end_to_end():
    ids = np.asarray([91, 7], dtype=np.int64)
    # Deliberately reverse tape row order relative to local state order.
    tape = RandomTape.generate([7, 91], RandomTapeSpec(3, 8, seed=44))
    tape_rows = np.asarray([1, 0], dtype=np.int32)
    audit = TapeAudit.zeros(2)
    audit.interaction_cursor[:] = 1
    pos = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    direction = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    polarization = np.asarray([[0, 1, 0], [0, 0, 1]], dtype=np.float32)
    times = np.asarray([0.5, 1.5], dtype=np.float32)
    histories = np.asarray([2, 16], dtype=np.uint32)
    decisions_state = np.asarray([1, 2], dtype=np.int32)
    counts_state = np.asarray([2, 4], dtype=np.int32)
    # Specialized boundary payload is indexed by tape row.
    boundary_payload = SimpleNamespace(
        interaction=np.asarray([0, 0], dtype=np.int32),
        draw_count=counts_state[[1, 0]],
        decision=decisions_state[[1, 0]],
    )
    shared_extras = {
        "wavelength": np.asarray([128, 129], dtype=np.float32),
        "weight": np.ones(2, dtype=np.float32),
        "evidx": np.asarray([3, 4], dtype=np.uint32),
        "last_hit_triangle": np.asarray([-1, 8], dtype=np.int32),
    }
    boundary = capture_boundary_tape_trace(
        boundary_trace=boundary_payload,
        global_photon_ids=ids,
        positions=pos,
        directions=direction,
        polarizations=polarization,
        times=times,
        histories=histories,
        audit=audit,
        tape_row_indices=tape_rows,
        tape=tape,
        step_index=0,
        state_rows=np.asarray([1, 0], dtype=np.int32),
        extras=shared_extras,
    )
    # Legacy payload and GPUPhotons arrays are indexed by local photon slot.
    legacy_payload = SimpleNamespace(
        interaction=np.asarray([0, 0], dtype=np.int32),
        draw_count=counts_state,
        process=decisions_state,
    )
    gpu_photons = SimpleNamespace(
        pos=pos,
        dir=direction,
        pol=polarization,
        t=times,
        flags=histories,
        wavelengths=shared_extras["wavelength"],
        weights=shared_extras["weight"],
        evidx=shared_extras["evidx"],
        last_hit_triangles=shared_extras["last_hit_triangle"],
    )
    legacy = capture_legacy_tape_trace(
        gpu_photons=gpu_photons,
        legacy_trace=legacy_payload,
        global_photon_ids=ids,
        audit=audit,
        tape_row_indices=tape_rows,
        tape=tape,
        step_index=0,
        state_rows=np.asarray([0, 1], dtype=np.int32),
    )
    report = compare_traces(legacy, boundary, left_label="legacy", right_label="triton")
    assert report.matched
    assert report.draw_consumption_certified


def test_legacy_adapter_understands_pycuda_float3_storage_words():
    vec3 = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32)])
    vectors = np.zeros(1, dtype=vec3)
    vectors["x"], vectors["y"], vectors["z"] = 1, 2, 3
    tape = RandomTape.generate([5], RandomTapeSpec(2, 4, seed=2))
    audit = TapeAudit.zeros(1)
    audit.interaction_cursor[0] = 1
    gpu_photons = SimpleNamespace(
        pos=vectors,
        dir=vectors,
        pol=vectors,
        t=np.zeros(1, dtype=np.float32),
        flags=np.asarray([2], dtype=np.uint32),
        wavelengths=np.asarray([128], dtype=np.float32),
        weights=np.ones(1, dtype=np.float32),
        evidx=np.zeros(1, dtype=np.uint32),
        last_hit_triangles=np.asarray([-1], dtype=np.int32),
    )
    trace = capture_legacy_tape_trace(
        gpu_photons=gpu_photons,
        legacy_trace=SimpleNamespace(
            interaction=np.asarray([0], dtype=np.int32),
            draw_count=np.asarray([2], dtype=np.int32),
            process=np.asarray([1], dtype=np.int32),
        ),
        global_photon_ids=np.asarray([5], dtype=np.int64),
        audit=audit,
        tape_row_indices=np.asarray([0], dtype=np.int32),
        tape=tape,
        step_index=0,
    )
    np.testing.assert_array_equal(trace.position, [[1, 2, 3]])
