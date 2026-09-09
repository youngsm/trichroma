"""CPU checks for the fail-closed full-trajectory certificate composer."""

import numpy as np
import pytest

from benchmarks.compose_detector_full_history_certificate import (
    CERTIFICATE_SENTINEL,
    STATE_CERTIFICATE_FIELDS,
    _state_ledger_summary,
    _validate_prefix_tail_continuity,
)


def _state_chunk():
    process = np.full((2, 4), CERTIFICATE_SENTINEL, dtype=np.uint32)
    process[0, :2] = np.asarray([0x20000004, 0x70000002], dtype=np.uint32)
    process[1, :1] = np.asarray([0x30000003], dtype=np.uint32)
    cursor = np.asarray([2, 1], dtype=np.int32)
    state = np.full(
        (2, 4, len(STATE_CERTIFICATE_FIELDS)),
        CERTIFICATE_SENTINEL,
        dtype=np.uint32,
    )
    committed = np.arange(process.shape[1])[None, :] < cursor[:, None]
    # A committed state may itself contain sentinel-valued fields (the usual
    # last-triangle=-1 case), so occupancy comes solely from the process mask.
    state[committed] = np.arange(
        np.count_nonzero(committed) * len(STATE_CERTIFICATE_FIELDS),
        dtype=np.uint32,
    ).reshape(-1, len(STATE_CERTIFICATE_FIELDS))
    state[0, 0, 12] = CERTIFICATE_SENTINEL
    return state, process, cursor


def test_state_ledger_summary_accepts_exact_prefix_and_raw_sentinel_field():
    state, process, cursor = _state_chunk()
    summary = _state_ledger_summary((state, process, cursor))

    assert summary["committed_state_records"] == 3
    assert summary["raw_words_per_record"] == len(STATE_CERTIFICATE_FIELDS)
    assert summary["fields"] == list(STATE_CERTIFICATE_FIELDS)
    assert len(summary["ordered_raw_words_sha256"]) == 64


def test_state_ledger_summary_rejects_any_suffix_write():
    state, process, cursor = _state_chunk()
    state[1, 2, 3] = np.uint32(0)

    with pytest.raises(RuntimeError, match="non-sentinel suffix"):
        _state_ledger_summary((state, process, cursor))


def test_state_ledger_summary_rejects_process_cursor_hole():
    state, process, cursor = _state_chunk()
    process[0, 1] = CERTIFICATE_SENTINEL

    with pytest.raises(RuntimeError, match="process mask is not a cursor prefix"):
        _state_ledger_summary((state, process, cursor))


def _continuity_pair():
    state, process, cursor = _state_chunk()
    initial_position = np.asarray(
        [[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]], dtype=np.float32
    )
    direction = np.asarray(
        [[1.0, -0.0, 0.0], [0.0, 1.0, -0.0]], dtype=np.float32
    )
    polarization = np.asarray(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    prefix = {
        "global_ids": np.asarray([10, 20], dtype=np.int64),
        "initial_position": initial_position,
        "normalized_direction": direction,
        "normalized_polarization": polarization,
        "interaction_cursor": cursor,
        "interaction_certificate": process,
        "state_certificate": state,
    }
    tail_process = np.full((1, 6), CERTIFICATE_SENTINEL, dtype=np.uint32)
    tail_process[0, :3] = np.asarray(
        [process[0, 0], process[0, 1], 0x30000003], dtype=np.uint32
    )
    tail_state = np.full(
        (1, 6, len(STATE_CERTIFICATE_FIELDS)),
        CERTIFICATE_SENTINEL,
        dtype=np.uint32,
    )
    tail_state[0, :2] = state[0, :2]
    tail_state[0, 2] = np.arange(
        len(STATE_CERTIFICATE_FIELDS), dtype=np.uint32
    ) + 100
    tail = {
        "global_ids": np.asarray([10], dtype=np.int64),
        "initial_position": initial_position[[0]].copy(),
        "normalized_direction": direction[[0]].copy(),
        "normalized_polarization": polarization[[0]].copy(),
        "interaction_cursor": np.asarray([3], dtype=np.int32),
        "interaction_certificate": tail_process,
        "state_certificate": tail_state,
    }
    return prefix, tail


def test_prefix_tail_continuity_checks_source_process_and_every_state_word():
    prefix, tail = _continuity_pair()
    assert _validate_prefix_tail_continuity(prefix, tail) == 2

    tail["state_certificate"][0, 1, 4] ^= np.uint32(1)
    with pytest.raises(RuntimeError, match="state history diverges"):
        _validate_prefix_tail_continuity(prefix, tail)


def test_prefix_tail_continuity_rejects_different_normalized_source():
    prefix, tail = _continuity_pair()
    tail["normalized_direction"][0, 1] = np.float32(0.0)

    with pytest.raises(RuntimeError, match="source discontinuity"):
        _validate_prefix_tail_continuity(prefix, tail)
