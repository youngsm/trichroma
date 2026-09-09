"""CPU-only contracts for the device-count PMT persistent launch."""

import pytest

from chroma_lar.triton_scene.instances import (
    DEVICE_TLAS_PROGRAMS_PER_SM,
    _persistent_tlas_program_count,
)


@pytest.mark.parametrize(
    ("capacity", "block_size", "sms", "expected"),
    [
        (0, 32, 108, 0),
        (1, 32, 108, 1),
        (32, 32, 108, 1),
        (33, 32, 108, 2),
        (27_648, 32, 108, 864),
        (27_649, 32, 108, 865),
        (300_000_000, 32, 108, 3_456),
    ],
)
def test_persistent_tlas_grid_is_capacity_and_residency_bounded(
    capacity, block_size, sms, expected
):
    assert _persistent_tlas_program_count(
        capacity, block_size, sms, programs_per_sm=32
    ) == expected


def test_persistent_tlas_grid_stride_covers_each_slot_exactly_once():
    capacity = 10_003
    block_size = 32
    sms = 3
    programs = _persistent_tlas_program_count(
        capacity, block_size, sms, programs_per_sm=32
    )
    stride = programs * block_size
    slots = []
    for program in range(programs):
        start = program * block_size
        while start < capacity:
            slots.extend(
                slot
                for slot in range(start, start + block_size)
                if slot < capacity
            )
            start += stride
    assert sorted(slots) == list(range(capacity))
    assert len(slots) == len(set(slots))
    assert programs == sms * 32


def test_persistent_tlas_grid_can_be_tuned_or_uncapped():
    capacity = 1_000_000
    assert _persistent_tlas_program_count(
        capacity, 32, 108, programs_per_sm=16
    ) == 1_728
    assert _persistent_tlas_program_count(
        capacity, 32, 108, programs_per_sm=128
    ) == 13_824
    assert _persistent_tlas_program_count(
        capacity, 32, 108, programs_per_sm=0
    ) == 31_250
    assert DEVICE_TLAS_PROGRAMS_PER_SM == 0
    assert _persistent_tlas_program_count(capacity, 32, 108) == 31_250


@pytest.mark.parametrize(
    ("capacity", "block_size", "sms", "programs_per_sm"),
    [
        (-1, 32, 1, None),
        (1, 0, 1, None),
        (1, 32, 0, None),
        (1, 32, 1, -1),
    ],
)
def test_persistent_tlas_grid_rejects_invalid_topology(
    capacity, block_size, sms, programs_per_sm
):
    with pytest.raises(ValueError):
        _persistent_tlas_program_count(
            capacity, block_size, sms, programs_per_sm=programs_per_sm
        )
