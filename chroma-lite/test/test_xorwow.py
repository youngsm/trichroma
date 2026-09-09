"""Native CUDA goldens for independent XORWOW initialization and advancement."""

import json
from pathlib import Path

import numpy as np
import pytest

from chroma.triton.xorwow import initialize_xorwow, xorwow_uint32

GOLDENS = json.loads((Path(__file__).parent / "data/xorwow_native_words.json").read_text())


@pytest.mark.parametrize("case", GOLDENS["cases"])
def test_native_xorwow_words(case):
    states = initialize_xorwow(
        case["seed"], np.asarray([case["subsequence"]], np.uint64), case["offset"]
    )
    np.testing.assert_array_equal(states[0], np.asarray(case["initial"], np.uint32))
    integers = np.asarray([xorwow_uint32(states)[0] for _ in range(32)], np.uint32)
    np.testing.assert_array_equal(integers, np.asarray(case["integers"], np.uint32))
    np.testing.assert_array_equal(states[0], np.asarray(case["final"], np.uint32))
    uniforms = integers.astype(np.float32) * np.float32(2.0**-32) + np.float32(2.0**-33)
    np.testing.assert_array_equal(uniforms.view(np.uint32), np.asarray(case["uniforms"], np.uint32))


def test_offset_matches_advancement_without_changing_subsequence():
    subsequences = np.asarray([0, 19, 2**32 + 7, 2**63], np.uint64)
    states = initialize_xorwow(1093, subsequences)
    for _ in range(257):
        xorwow_uint32(states)
    np.testing.assert_array_equal(states, initialize_xorwow(1093, subsequences, offset=257))


@pytest.mark.parametrize(
    "seed,offset,subsequences",
    [
        (-1, 0, [0]),
        (2**64, 0, [0]),
        (0, -1, [0]),
        (0, 2**64, [0]),
        (0, 0, [-1]),
        (0, 0, [0.5]),
    ],
)
def test_invalid_stream_identifiers_fail(seed, offset, subsequences):
    with pytest.raises(ValueError):
        initialize_xorwow(seed, subsequences, offset)
