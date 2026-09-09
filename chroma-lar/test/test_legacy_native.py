"""Small immutable regressions from the audited original CUDA installation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from chroma.triton.xorwow import initialize_xorwow
from chroma_lar.triton_scene.legacy_spectral import propagate_legacy


@pytest.mark.parametrize("record_history", [True, False])
def test_original_wire_words_and_rng(record_history):
    if not torch.cuda.is_available():
        pytest.skip("native arithmetic regression requires CUDA")
    path = Path(__file__).with_name("data") / "legacy_wire_native.npz"
    provenance = json.loads(path.with_suffix(".json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    with np.load(path) as archive:
        data = dict(archive)
    rng = initialize_xorwow(int(data["seed"]), data["photon_ids"].astype(np.uint64))
    actual = propagate_legacy(
        data,
        data["source_words"],
        max_steps=provenance["max_steps"],
        rng_words=rng,
        record_history=record_history,
    )
    np.testing.assert_array_equal(actual["final_words"], data["original_words"])
    np.testing.assert_array_equal(actual["native_rng_words"], data["original_rng_words"])
    np.testing.assert_array_equal(actual["interaction_counts"], data["interaction_counts"])
    assert not np.any(actual["overflow"])
    if record_history:
        np.testing.assert_array_equal(actual["state_words"], data["state_words"])
        np.testing.assert_array_equal(actual["draw_counts"], data["draw_counts"])
    else:
        assert "state_words" not in actual
        np.testing.assert_array_equal(
            actual["draw_counts"], np.maximum(data["draw_counts"], 0).sum(axis=1)
        )
