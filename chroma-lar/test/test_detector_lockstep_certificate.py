"""CPU-only reporting tests for the dense lockstep interaction certificate."""

from argparse import Namespace

import numpy as np

from benchmarks.validate_detector_lockstep import (
    STATE_CERTIFICATE_EMPTY_WORD,
    STATE_CERTIFICATE_FIELDS,
    _active_set_comparison,
    _certificate_comparison,
    _comparison_multistep,
    _photon_ids_from_file,
    _photon_ids_from_npz,
    _single_event_evidx_invariant,
    _state_certificate_comparison,
)
from chroma.triton.rng_alignment import (
    CERTIFICATE_EMPTY_WORD,
    pack_interaction_certificate,
)


def _ledger_values(words, cursor, *, global_ids=(536, 673)):
    return {
        "interaction_certificate": np.ascontiguousarray(words, dtype=np.uint32),
        "interaction_cursor": np.ascontiguousarray(cursor, dtype=np.int32),
        "draw_cursor": np.zeros(len(cursor), dtype=np.int32),
        "overflow": np.zeros(len(cursor), dtype=np.uint32),
        "global_ids": np.ascontiguousarray(global_ids, dtype=np.int64),
        "state_certificate": _valid_state_ledger(cursor),
    }


def _valid_state_ledger(cursor):
    cursor = np.ascontiguousarray(cursor, dtype=np.int32)
    words = np.full(
        (len(cursor), 4, len(STATE_CERTIFICATE_FIELDS)),
        STATE_CERTIFICATE_EMPTY_WORD,
        dtype=np.uint32,
    )
    for row, count in enumerate(cursor):
        for interaction in range(int(count)):
            # Finite float state words plus the legitimate int32 -1 triangle
            # sentinel make each committed record distinguishable from an
            # untouched all-0xffffffff suffix.
            values = np.zeros(len(STATE_CERTIFICATE_FIELDS), dtype=np.uint32)
            values[0] = np.asarray(
                [np.float32(row + interaction + 0.5)], dtype=np.float32
            ).view(np.uint32)[0]
            values[9] = np.asarray([np.float32(450.0)]).view(np.uint32)[0]
            values[12] = np.asarray([-1], dtype=np.int32).view(np.uint32)[0]
            values[13] = np.asarray([np.float32(1.0)]).view(np.uint32)[0]
            words[row, interaction] = values
    return words


def _valid_ledger():
    words = np.full((2, 4), CERTIFICATE_EMPTY_WORD, dtype=np.uint32)
    words[0, :2] = pack_interaction_certificate([2, 5], [4, 8])
    words[1, 0] = pack_interaction_certificate(7, 5)[0]
    return words, np.asarray([2, 1], dtype=np.int32)


def test_certificate_exact_words_and_hole_free_prefix_certify_draws():
    words, cursor = _valid_ledger()
    report = _certificate_comparison(
        _ledger_values(words, cursor), _ledger_values(words.copy(), cursor)
    )

    assert report["ledger_equal"]
    assert report["prefix_valid"] == {"chroma": True, "triton": True}
    assert report["cursor_equal"]
    assert report["overflow_free"]
    assert report["draw_consumption_certified"]
    assert report["first_difference"] is None
    assert report["exact_words"] == words.size


def test_certificate_first_difference_reports_identity_process_and_draw():
    cuda_words, cursor = _valid_ledger()
    triton_words = cuda_words.copy()
    triton_words[1, 0] = pack_interaction_certificate(8, 4)[0]
    report = _certificate_comparison(
        _ledger_values(cuda_words, cursor),
        _ledger_values(triton_words, cursor),
    )

    assert not report["draw_consumption_certified"]
    assert report["exact_words"] == cuda_words.size - 1
    assert report["first_difference"] == {
        "row": 1,
        "global_photon_id": 673,
        "interaction": 0,
        "chroma_word": int(pack_interaction_certificate(7, 5)[0]),
        "triton_word": int(pack_interaction_certificate(8, 4)[0]),
        "chroma_committed": True,
        "triton_committed": True,
        "chroma_process": 7,
        "triton_process": 8,
        "chroma_process_name": "dielectric_reflect",
        "triton_process_name": "dielectric_transmit",
        "chroma_draw_count": 5,
        "triton_draw_count": 4,
    }


def test_certificate_equal_but_holed_prefix_is_not_certified():
    words, cursor = _valid_ledger()
    words[0, 1] = CERTIFICATE_EMPTY_WORD
    values = _ledger_values(words, cursor)
    report = _certificate_comparison(values, values)

    assert report["ledger_equal"]
    assert not report["prefix_valid"]["chroma"]
    assert not report["prefix_valid"]["triton"]
    assert not report["draw_consumption_certified"]


def test_certificate_requires_both_audit_cursors_to_match():
    words, cursor = _valid_ledger()
    cuda_values = _ledger_values(words, cursor)
    triton_values = _ledger_values(words.copy(), cursor.copy())
    triton_values["draw_cursor"][1] = 1

    report = _certificate_comparison(cuda_values, triton_values)

    assert report["ledger_equal"]
    assert report["interaction_cursor_equal"]
    assert not report["draw_cursor_equal"]
    assert not report["cursor_equal"]
    assert not report["draw_consumption_certified"]


def test_state_certificate_compares_every_committed_word_and_suffix():
    _, cursor = _valid_ledger()
    values = _ledger_values(*_valid_ledger())
    report = _state_certificate_comparison(values, values)

    assert report["post_interaction_state_certified"]
    assert report["committed_records"] == {"chroma": 3, "triton": 3}
    assert report["compared_committed_words"] == 3 * len(
        STATE_CERTIFICATE_FIELDS
    )
    assert report["exact_committed_words"] == report[
        "compared_committed_words"
    ]
    assert report["suffix_valid"] == {"chroma": True, "triton": True}
    assert report["first_difference"] is None


def test_state_certificate_reports_raw_field_mismatch():
    words, cursor = _valid_ledger()
    cuda_values = _ledger_values(words, cursor)
    triton_values = _ledger_values(words.copy(), cursor.copy())
    triton_values["state_certificate"] = triton_values[
        "state_certificate"
    ].copy()
    triton_values["state_certificate"][1, 0, 4] = np.asarray(
        [np.float32(-0.0)], dtype=np.float32
    ).view(np.uint32)[0]

    report = _state_certificate_comparison(cuda_values, triton_values)

    assert not report["post_interaction_state_certified"]
    assert report["exact_committed_words"] == (
        report["compared_committed_words"] - 1
    )
    assert report["first_difference"]["row"] == 1
    assert report["first_difference"]["global_photon_id"] == 673
    assert report["first_difference"]["interaction"] == 0
    assert report["first_difference"]["field"] == "direction_y"
    assert report["first_difference"]["chroma_word_hex"] == "0x00000000"
    assert report["first_difference"]["triton_word_hex"] == "0x80000000"


def test_state_certificate_rejects_written_suffix_and_empty_prefix():
    words, cursor = _valid_ledger()
    cuda_values = _ledger_values(words, cursor)

    written_suffix = _ledger_values(words.copy(), cursor.copy())
    written_suffix["state_certificate"] = written_suffix[
        "state_certificate"
    ].copy()
    written_suffix["state_certificate"][0, 3, 0] = np.uint32(0)
    suffix_report = _state_certificate_comparison(
        cuda_values, written_suffix
    )
    assert not suffix_report["post_interaction_state_certified"]
    assert not suffix_report["suffix_valid"]["triton"]
    assert suffix_report["first_difference"]["kind"] == "sentinel_suffix"

    empty_prefix = _ledger_values(words.copy(), cursor.copy())
    empty_prefix["state_certificate"] = empty_prefix[
        "state_certificate"
    ].copy()
    empty_prefix["state_certificate"][0, 0] = STATE_CERTIFICATE_EMPTY_WORD
    prefix_report = _state_certificate_comparison(cuda_values, empty_prefix)
    assert not prefix_report["post_interaction_state_certified"]
    assert not prefix_report["prefix_records_present"]["triton"]


def _multistep_values():
    words, cursor = _valid_ledger()
    count = 2
    return {
        "position": np.zeros((count, 3), dtype=np.float32),
        "direction": np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
        "polarization": np.asarray(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        ),
        "time": np.zeros(count, dtype=np.float32),
        "history": np.zeros(count, dtype=np.uint32),
        "detected_channel": np.full(count, -1, dtype=np.int32),
        "boundary_kind": np.asarray([1, 0], dtype=np.int32),
        "last_triangle": np.asarray([17, -1], dtype=np.int32),
        "evidx": np.zeros(count, dtype=np.uint32),
        "interaction_cursor": cursor,
        "draw_cursor": np.zeros(count, dtype=np.int32),
        "overflow": np.zeros(count, dtype=np.uint32),
        "global_ids": np.asarray([536, 673], dtype=np.int64),
        "active_ids": np.asarray([673], dtype=np.int64),
        "interaction_certificate": words,
        "state_certificate": _valid_state_ledger(cursor),
    }


def _args():
    return Namespace(
        count=2,
        source_population=700,
        photon_ids=[536, 673],
        max_steps=2,
        center=(-1000.0, 0.0, 0.0),
        voxel_size=30.0,
        source_seed=8123,
        tape_seed=99173,
        tape_interactions=4,
        draws_per_interaction=64,
    )


def test_multistep_draw_certificate_is_independent_of_state_equality():
    cuda_values = _multistep_values()
    triton_values = {name: value.copy() for name, value in cuda_values.items()}
    triton_values["last_instance"] = np.full(2, -1, dtype=np.int32)
    triton_values["position"][0, 0] = np.float32(1.0)

    report = _comparison_multistep(_args(), cuda_values, triton_values)

    assert not report["matched_bitwise"]
    assert report["draw_consumption_certified"]
    assert report["active_set"]["exact"]
    assert report["last_instance_invariant"]["satisfied"]


def test_multistep_match_gates_last_triangle_instance_and_active_global_ids():
    cuda_values = _multistep_values()

    def triton_copy():
        values = {name: value.copy() for name, value in cuda_values.items()}
        values["last_instance"] = np.full(2, -1, dtype=np.int32)
        return values

    exact = _comparison_multistep(_args(), cuda_values, triton_copy())
    assert exact["matched_bitwise"]

    last_triangle = triton_copy()
    last_triangle["last_triangle"][0] += 1
    assert not _comparison_multistep(
        _args(), cuda_values, last_triangle
    )["matched_bitwise"]

    last_instance = triton_copy()
    last_instance["last_instance"][0] = 0
    assert not _comparison_multistep(
        _args(), cuda_values, last_instance
    )["matched_bitwise"]

    active_local_id = triton_copy()
    active_local_id["active_ids"] = np.asarray([1], dtype=np.int64)
    assert not _comparison_multistep(
        _args(), cuda_values, active_local_id
    )["matched_bitwise"]

    state_word = triton_copy()
    state_word["state_certificate"][1, 0, 10] ^= np.uint32(1)
    state_report = _comparison_multistep(_args(), cuda_values, state_word)
    assert not state_report["matched_bitwise"]
    assert not state_report["post_interaction_state_certified"]
    assert state_report["first_state_difference"]["field"] == "time"


def test_active_comparison_uses_global_ids_and_rejects_duplicates():
    exact = _active_set_comparison(
        {"active_ids": np.asarray([673, 536], dtype=np.int64)},
        {"active_ids": np.asarray([536, 673], dtype=np.int64)},
    )
    duplicate = _active_set_comparison(
        {"active_ids": np.asarray([536, 536], dtype=np.int64)},
        {"active_ids": np.asarray([536, 536], dtype=np.int64)},
    )

    assert exact["exact"]
    assert not duplicate["exact"]
    assert not duplicate["chroma_unique"]


def test_single_event_evidx_invariant_requires_zero_and_endpoint_equality():
    zero = {"evidx": np.zeros(3, dtype=np.uint32)}
    exact = _single_event_evidx_invariant(zero, zero)
    assert exact["satisfied"]
    assert exact["endpoint_equal"]

    nonzero_cuda = {"evidx": np.asarray([0, 1, 0], dtype=np.uint32)}
    mismatch = _single_event_evidx_invariant(nonzero_cuda, zero)
    assert not mismatch["satisfied"]
    assert not mismatch["chroma_zero"]
    assert not mismatch["endpoint_equal"]


def test_photon_id_file_accepts_json_and_line_formats(tmp_path):
    json_path = tmp_path / "ids.json"
    json_path.write_text("[6620, 721, 163]\n")
    line_path = tmp_path / "ids.txt"
    line_path.write_text("6620\n\n721\n163\n")

    assert _photon_ids_from_file(json_path) == [6620, 721, 163]
    assert _photon_ids_from_file(line_path) == [6620, 721, 163]


def test_photon_id_from_npz_reads_only_integer_active_ids(tmp_path):
    valid_path = tmp_path / "tail.npz"
    np.savez(valid_path, active_ids=np.asarray([6620, 721], dtype=np.int64))
    assert _photon_ids_from_npz(valid_path) == [6620, 721]

    invalid_path = tmp_path / "invalid.npz"
    np.savez(invalid_path, active_ids=np.asarray([1.0], dtype=np.float32))
    try:
        _photon_ids_from_npz(invalid_path)
    except ValueError as error:
        assert "one-dimensional integer array" in str(error)
    else:
        raise AssertionError("float active_ids must fail closed")
