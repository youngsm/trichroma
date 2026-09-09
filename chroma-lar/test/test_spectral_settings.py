"""Reject invalid scheduling inputs before detector compilation or allocation."""

import pytest

from chroma_lar.spectral_state import TransportSettings


@pytest.mark.parametrize(
    "options",
    [
        {"history_length": 1.5},
        {"history_length": True},
        {"epochs_per_poll": 0},
        {"epochs_per_poll": "2"},
        {"block_size": 96},
        {"block_size": 128.0},
        {"fused_pmt": "false"},
    ],
)
def test_invalid_tuning_is_rejected(options):
    with pytest.raises(ValueError):
        TransportSettings(**options)
